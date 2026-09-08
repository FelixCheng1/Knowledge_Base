from pathlib import Path
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from app.core.logger import logger

from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from app.clients.mongo_history_utils import *
from app.query_process.agent.main_graph import query_app


# 定义fastapi对象
app = FastAPI(title="query service", description="掌柜智库查询服务！")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 返回chat.html页面
@app.get("/chat.html")
async def chat():
    current_dir_parent_path = Path(__file__).absolute().parent.parent
    chat_html_path = current_dir_parent_path / "page" / "chat.html"
    if not chat_html_path.exists():
        logger.error(f"页面不存在：{chat_html_path}")
        raise HTTPException(status_code=404, detail=f"没有查询到页面，地址为：{chat_html_path}！")
    logger.info("成功加载 chat.html 页面")
    return FileResponse(chat_html_path)

# 定义接口接收的数据结构
class QueryRequest(BaseModel):
    """查询请求数据结构"""
    query:str = Field(..., description="查询内容")
    session_id:str = Field(None, description="会话ID")
    is_stream:bool = Field(False, description="是否流式返回")

# 核心查询处理函数
def run_query_graph(session_id:str, user_query:str, is_stream:bool = True):
    logger.info(f"[{session_id}] 开始执行查询流程，流式模式{is_stream}")
    default_state = {"original_query": user_query, "session_id": session_id, "is_stream": is_stream}
    try:
        query_app.invoke(default_state)
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
        logger.info(f"[{session_id}] 查询流程执行完成")
    except Exception as e:
        logger.exception(f"[{session_id}] 查询流程异常：{str(e)}")
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        if is_stream:
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})

# 查询接口
@app.post("/query")
async def query(background_tasks: BackgroundTasks, request: QueryRequest):
    """
    1.解析参数
    2.更新任务状态
    3.调用处理流程图
    4.返回结果
    """
    user_query = request.query
    session_id = request.session_id if request.session_id else str(uuid.uuid4())
    is_stream = request.is_stream

    if is_stream:
        create_sse_queue(session_id)
        logger.info(f"[{session_id}] 已创建 SSE 消息队列")

    update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)
    logger.info(f"[{session_id}] 任务开始处理，查询内容：{user_query}")

    if is_stream:
        background_tasks.add_task(run_query_graph, session_id, user_query, is_stream)
        logger.info(f"[{session_id}] 流式任务已提交到后台处理")
        return {
            "message": "结果正在处理中...",
            "session_id": session_id
        }
    else:
        run_query_graph(session_id, user_query, is_stream)
        answer = get_task_result(session_id, "answer", "")
        logger.info(f"[{session_id}] 非流式查询处理完成")
        return {
            "message": "处理完成！",
            "session_id": session_id,
            "answer": answer,
            "done_list": []
        }

# SSE 流式推送接口
@app.get("/stream/{session_id}")
async def stream(session_id:str, request:Request):
    logger.info(f"[{session_id}] 客户端已建立 SSE 流式连接")
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# 健康检查
@app.get("/health")
async def health():
    """服务健康检查"""
    logger.info("健康检查接口调用成功")
    return {"ok": True}

@app("/history/{session_id}")
async def history(session_id:str, limit:int = 50):
    """
    查询当前会话的历史记录
    """
    try:
        records = get_recent_messages(session_id, limit)
        items = []
        for r in records:
            items.append({
                "message_id": str(r.get("_id")) if r.get("_id") is not None else "",
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "ts": r.get("ts")
            })
        return {"session_id": session_id, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")

@app.delete("/history/{session_id}")
async def clear_chat_history(session_id:str):
    count = clear_history(session_id)
    return {"message": "History cleared", "deleted_count": count}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)