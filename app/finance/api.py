from __future__ import annotations

import asyncio
import json
import os
import queue
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.finance.models import (
    DocumentListResponse, DocumentStatus, ErrorResponse, FinancialDocument,
    HealthResponse, ImportSubmissionResponse, ImportTask, MessageListResponse,
    QueryRequest, QueryResult, RetryImportResponse, Session, SessionListResponse,
)
from app.finance.parser import MinerUParser
from app.finance.repository import FinanceRepository
from app.finance.service import FinanceService


router = APIRouter(prefix="/api/v1")
_service: FinanceService | None = None


def get_service() -> FinanceService:
    global _service
    if _service is None:
        _service = FinanceService(
            FinanceRepository(os.getenv("MONGO_URL", "mongodb://127.0.0.1:27017"), os.getenv("FINANCE_MONGO_DB_NAME", "finance_knowledge_base")),
            MinerUParser(os.getenv("MINERU_BASE_URL", ""), os.getenv("MINERU_API_TOKEN", "")),
            Path(os.getenv("FINANCE_WORK_DIR", "output/finance")),
        )
    return _service


class DocumentPatch(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=512)
    document_type: str | None = None
    metadata: dict | None = None


def _not_found(resource: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"未找到{resource}")


@router.get("/health", tags=["Health"], summary="读取服务和依赖状态", response_model=HealthResponse, operation_id="getHealth")
def health(service: Annotated[FinanceService, Depends(get_service)]):
    dependencies = {"mongo": False, "milvus": False, "minio": False}
    try:
        dependencies["mongo"] = service.repo.readiness()
    except Exception:
        pass
    try:
        dependencies["milvus"] = service.milvus_ready()
    except Exception:
        pass
    try:
        dependencies["minio"] = service.minio_ready()
    except Exception:
        pass
    return {"ok": True, "dependencies": dependencies, "ready": all(dependencies.values())}


@router.post(
    "/documents", tags=["Documents"], summary="异步导入金融资料", response_model=ImportSubmissionResponse,
    status_code=202, operation_id="createDocumentImport", responses={400: {"model": ErrorResponse}},
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(...)],
    service: Annotated[FinanceService, Depends(get_service)],
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="缺少文件名")
    if Path(file.filename).suffix.lower() not in MinerUParser.SUPPORTED:
        raise HTTPException(status_code=400, detail="仅支持 PDF、DOC、DOCX 和 Markdown 文件")
    document, task = service.create_import(file.filename, await file.read())
    if task.status != DocumentStatus.ACTIVE:
        background_tasks.add_task(service.import_document, task.task_id)
    return {"document": document, "task": task}


@router.get("/documents", tags=["Documents"], summary="分页前的资料列表", response_model=DocumentListResponse, operation_id="listDocuments")
def list_documents(
    status: str | None = None,
    document_type: str | None = None,
    service: FinanceService = Depends(get_service),
):
    return {"items": service.repo.list_documents(status, document_type)}


@router.get("/documents/{document_id}", tags=["Documents"], summary="读取资料详情", response_model=FinancialDocument, operation_id="getDocument", responses={404: {"model": ErrorResponse}})
def get_document(document_id: str, service: FinanceService = Depends(get_service)):
    document = service.repo.get_document(document_id)
    if not document:
        raise _not_found("资料")
    return document


@router.post(
    "/documents/{document_id}/versions", tags=["Documents"], summary="异步上传资料新版本", response_model=ImportSubmissionResponse,
    status_code=202, operation_id="createDocumentVersion", responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def upload_document_version(
    document_id: str,
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(...)],
    service: Annotated[FinanceService, Depends(get_service)],
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="缺少文件名")
    if Path(file.filename).suffix.lower() not in MinerUParser.SUPPORTED:
        raise HTTPException(status_code=400, detail="仅支持 PDF、DOC、DOCX 和 Markdown 文件")
    try:
        document, task = service.create_version_import(document_id, file.filename, await file.read())
    except KeyError:
        raise _not_found("资料")
    if task.status != DocumentStatus.ACTIVE:
        background_tasks.add_task(service.import_document, task.task_id)
    return {"document": document, "task": task}


@router.patch("/documents/{document_id}", tags=["Documents"], summary="修正人工元数据", response_model=FinancialDocument, operation_id="updateDocument", responses={404: {"model": ErrorResponse}})
def patch_document(document_id: str, patch: DocumentPatch, service: FinanceService = Depends(get_service)):
    try:
        return service.update_document(document_id, patch.title, patch.document_type, patch.metadata)
    except KeyError:
        raise _not_found("资料")


@router.post("/documents/{document_id}/disable", tags=["Documents"], summary="停用资料", response_model=FinancialDocument, operation_id="disableDocument", responses={404: {"model": ErrorResponse}})
def disable_document(document_id: str, service: FinanceService = Depends(get_service)):
    try:
        return service.disable_document(document_id)
    except KeyError:
        raise _not_found("资料")


@router.get("/documents/{document_id}/file", tags=["Documents"], summary="下载原始资料", operation_id="downloadDocument", responses={404: {"model": ErrorResponse}})
def original_file(document_id: str, service: FinanceService = Depends(get_service)):
    document = service.repo.get_document(document_id)
    if not document or not document.active_version_id:
        raise _not_found("资料文件")
    version = next((item for item in document.versions if item.version_id == document.active_version_id), None)
    if version and Path(version.stored_path).exists():
        return FileResponse(version.stored_path, filename=version.original_name)
    try:
        url = service.original_file_url(document_id)
    except KeyError:
        url = None
    if not url:
        raise _not_found("资料文件")
    return RedirectResponse(url)


@router.get("/import-tasks/{task_id}", tags=["Import tasks"], summary="查询导入任务状态", response_model=ImportTask, operation_id="getImportTask", responses={404: {"model": ErrorResponse}})
def get_import_task(task_id: str, service: FinanceService = Depends(get_service)):
    task = service.repo.get_task(task_id)
    if not task:
        raise _not_found("导入任务")
    return task


@router.post("/import-tasks/{task_id}/retry", tags=["Import tasks"], summary="重试失败导入任务", response_model=RetryImportResponse, status_code=202, operation_id="retryImportTask", responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}})
def retry_import(task_id: str, background_tasks: BackgroundTasks, service: FinanceService = Depends(get_service)):
    task = service.repo.get_task(task_id)
    if not task:
        raise _not_found("导入任务")
    if task.status == DocumentStatus.PROCESSING:
        raise HTTPException(status_code=409, detail="任务正在处理")
    background_tasks.add_task(service.import_document, task_id)
    return {"task_id": task_id, "status": "pending"}


@router.post("/sessions", tags=["Sessions"], summary="创建问答会话", response_model=Session, status_code=201, operation_id="createSession")
def create_session(service: FinanceService = Depends(get_service)):
    return service.create_session()


@router.get("/sessions", tags=["Sessions"], summary="列出问答会话", response_model=SessionListResponse, operation_id="listSessions")
def list_sessions(service: FinanceService = Depends(get_service)):
    return {"items": service.repo.list_sessions()}


@router.get("/sessions/{session_id}/messages", tags=["Sessions"], summary="读取会话消息", response_model=MessageListResponse, operation_id="listSessionMessages", responses={404: {"model": ErrorResponse}})
def list_messages(session_id: str, service: FinanceService = Depends(get_service)):
    if not service.repo.get_session(session_id):
        raise _not_found("会话")
    return {"items": service.repo.list_messages(session_id)}


@router.delete("/sessions/{session_id}", tags=["Sessions"], summary="删除会话和消息", status_code=204, operation_id="deleteSession", responses={404: {"model": ErrorResponse}})
def delete_session(session_id: str, service: FinanceService = Depends(get_service)):
    if not service.repo.delete_session(session_id):
        raise _not_found("会话")


@router.post("/queries", tags=["Queries"], summary="提交金融资料问答", response_model=QueryResult, status_code=202, operation_id="createQuery")
def create_query(request: QueryRequest, background_tasks: BackgroundTasks, service: FinanceService = Depends(get_service)):
    try:
        result = service.submit_query(request.query, request.session_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    if request.stream:
        from app.utils.sse_utils import create_sse_queue
        create_sse_queue(result.query_id)
    background_tasks.add_task(service.answer_query, result.query_id, request.query)
    return result


@router.get("/queries/{query_id}", tags=["Queries"], summary="读取问答结果", response_model=QueryResult, operation_id="getQuery", responses={404: {"model": ErrorResponse}})
def get_query(query_id: str, service: FinanceService = Depends(get_service)):
    result = service.repo.get_query(query_id)
    if not result:
        raise _not_found("查询")
    return result


@router.get("/queries/{query_id}/events", tags=["Queries"], summary="订阅问答进度与最终结果", response_class=StreamingResponse, operation_id="streamQueryEvents", responses={200: {"description": "SSE，事件类型为 progress、delta、final 或 error", "content": {"text/event-stream": {}}}})
async def query_events(query_id: str, request: Request, service: FinanceService = Depends(get_service)):
    """订阅指定查询的事件流。

    - 后台任务把 progress/delta 事件写入 query_id 对应的队列，这里实时转发；
    - 断线重连时若 final 已持久化，直接补发 final 后结束，保证不丢结果。
    """
    from app.utils.sse_utils import get_sse_queue, remove_sse_queue

    async def event_stream():
        existing = service.repo.get_query(query_id)
        if not existing:
            yield "event: error\ndata: {\"error\": \"查询不存在\"}\n\n"
            return
        if existing.status in {"completed", "failed", "clarification_needed"}:
            # 断线重连或刷新后补发最终结果。
            yield f"event: final\ndata: {json.dumps(existing.model_dump(mode='json'), ensure_ascii=False)}\n\n"
            return
        stream_queue = get_sse_queue(query_id)
        if stream_queue is None:
            yield "event: error\ndata: {\"error\": \"该查询未开启流式推送，请直接读取结果接口\"}\n\n"
            return
        loop = asyncio.get_running_loop()
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    message = await loop.run_in_executor(None, stream_queue.get, True, 1.0)
                except queue.Empty:
                    continue
                event, data = message["event"], message["data"]
                yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                if event in {"final", "error"}:
                    break
        except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
            return
        finally:
            remove_sse_queue(query_id)

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
