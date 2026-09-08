from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List
import shutil
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from app.core.logger import PROJECT_ROOT, logger

from app.import_process.agent.state import get_default_state
from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from app.import_process.agent.main_graph import kb_import_app


app = FastAPI(title="query service",description="掌柜智库查询服务！")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/import.html",response_class=FileResponse)
async def get_import_page():
    # 拼接html路径
    html_path = PROJECT_ROOT / "app" / "import_process" / "page" / "import.html"
    # 日志记录页面访问的文件路径，方便排查文件不存在的问题
    logger.info(f"前端页面访问，文件绝对路径：{html_path}")

    # 校验文件存在
    if not html_path.exists():
        logger.error(f"前端页面不存在，路径：{html_path}")
        raise HTTPException(status_code=404,detail="import.html page not found")

    # 返回文件
    return FileResponse(
        path=html_path,
        media_type="text/html"
    )

# --------------------------
# 后台任务：LangGraph全流程执行
# 独立于主请求线程，由BackgroundTasks触发，避免阻塞接口响应
# --------------------------
def run_graph_task(task_id:str, local_dir:str, local_file_path:str):
    """
    LangGraph全流程执行后台任务
    核心流程：初始化状态 → 流式执行图节点 → 实时更新任务状态 → 异常捕获
    任务状态更新：pending → processing → completed/failed
    节点进度更新：每完成一个节点，将节点名加入done_list，供前端轮询查看
    Args:
        task_id: 全局唯一任务ID，关联单个文件的全流程处理
        local_dir: 该任务的本地文件存储目录（含临时文件/解析结果）
        local_file_path: 上传文件的本地绝对路径
    """
    try:
        # 1.更新任务的全局状态
        update_task_status(task_id, "processing")
        logger.info(f"[{task_id}] 开始执行LangGraph全流程，本地文件路径：{local_file_path}")

        # 2. 初始化LangGraph状态：加载默认状态 + 注入当前任务的核心参数
        init_state = get_default_state()
        init_state["task_id"] = task_id
        init_state["local_dir"] = local_dir
        init_state["local_file_path"] = local_file_path

        # 3.流式执行LangGraph全流程，（stream模式：实时获取每个节点的执行结果）
        for event in kb_import_app.stream(init_state):
            for node_name, node_result in event.items():
                # 记录每个节点完成的日志，包含任务ID和节点名，方便追踪执行顺序
                logger.info(f"[{task_id}] LangGraph节点执行完成：{node_name}")
                # 将完成的节点名加入【已完成列表】，前端轮询/status/{task_id}可实时获取
                add_done_task(task_id, node_name)

        # 4. 全流程执行完成，更新任务全局状态：已完成
        update_task_status(task_id, "completed")
        logger.info(f"[{task_id}] LangGraph全流程执行完毕，任务完成")

    except Exception as e:
        # 5. 捕获全流程异常，更新任务全局状态为：失败
        update_task_status(task_id, "failed")
        logger.error(f"[{task_id}] LangGraph全流程执行失败，异常信息：{e}", exc_info=True)

# --------------------------
# 核心接口：多文件上传接口（不上传 MinIO）
# 支持多文件批量上传，核心流程：接收文件 → 本地保存 → 启动后台任务
# 访问地址：http://localhost:8000/upload （POST请求，form-data格式传参）
# --------------------------
from pathlib import Path

@app.post("/upload", summary="文件上传接口", description="支持多文件批量上传，自动触发知识库导入全流程")
async def upload_files(
    background_task: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """
    文件上传核心接口（不上传 MinIO）
    1. 接收前端上传的多文件（PDF/MD为主）
    2. 按「日期/任务ID」分层保存到本地输出目录，避免文件冲突
    3. 为每个文件生成唯一TaskID，启动独立的LangGraph后台处理任务
    4. 实时更新任务状态，供前端轮询监控进度

    Args:
        background_tasks: FastAPI后台任务对象，用于异步执行LangGraph流程
        files: 前端上传的文件列表（form-data格式）
    Returns:
      包含上传结果和所有任务ID的JSON响应
    """
    #1. 构建本地存储根目录：项目根目录/output/YYYYMMDD （按日期分层，方便管理）
    today_str = datetime.now().strftime("%Y%m%d")
    date_based_root_dir:Path = PROJECT_ROOT / "output" / today_str

    # 初始化任务ID列表，用于返回给前端（一个文件对应一个task—id）
    task_ids = []

    # 2.遍历处理每一个上传的文件（多文件批量处理，各自独立生成task-id）
    for file in files:
        # 生成全局唯一的TaskID
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        logger.info(f"[{task_id}] 开始处理上传文件，文件名：{file.filename},文件类型：{file.content_type}")

        # 3.标记 【文件上传】 阶段为 【运行中】，前端轮询查询
        add_running_task(task_id, "upload_file")

        # 4.构建该任务的本地独立目录，output/YYYYMMDD/TaskID，避免多文件重名冲突
        task_local_dir:Path = date_based_root_dir / task_id
        task_local_dir.mkdir(parents=True, exist_ok=True)

        # 5. 构建上传文件的本地保存绝对路径
        local_file_abs_path:Path = task_local_dir / file.filename

        # 6. 将构建上传文件保存到本地临时目录
        with local_file_abs_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"[{task_id}] 文件已保存至本地，路径：{local_file_abs_path}")

        # 7. 标记【文件上传】 阶段为 【已完成】
        add_done_task(task_id, "upload_file")

        # 8. 将LangGraph全流程处理加入FastAPI后台任务
        background_task.add_task(
            run_graph_task,
            task_id,
            str(task_local_dir),
            str(local_file_abs_path)
        )
        logger.info(f"[{task_id}] 已将LangGraph全流程加入后台任务，任务已启动")

    # 9. 所有文件处理完毕，返回上传成功信息和所有的TaskID
    logger.info(f"多文件上传处理完毕，共处理{len(files)}个文件，生成的TaskID列表：{task_ids}")

    return {
        "code":200,
        "message": f"Files uploaded successfully, total: {len(files)}",
        "task_ids": task_ids
    }

# --------------------------
# 核心接口：任务状态查询接口
# 前端轮询此接口获取单个任务的处理进度和状态
# 访问地址：http://localhost:8000/status/{task_id} （GET请求）
# --------------------------
@app.get("/status/{task_id}", summary="任务状态查询", description="根据TaskID查询单个文件的处理进度和全局状态")
async def get_task_progress(task_id:str):
    """
    任务状态查询接口
    前端轮询此接口（如每秒1次），获取任务的实时处理进度
    返回数据均来自内存中的任务管理字典（task_utils.py），高性能无IO

    :param task_id: 全局唯一任务ID（由/upload接口返回）
    :return: 包含任务全局状态、已完成节点、运行中节点的JSON响应
    """
    # 构造任务状态返回体
    task_status_info:Dict[str,Any] = {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),  # 任务全局状态：pending/processing/completed/failed
        "done_list": get_done_task_list(task_id),  # 已完成的节点/阶段列表
        "running_list": get_running_task_list(task_id)  # 正在运行的节点/阶段列表
    }
    # 记录状态查询日志，方便追踪前端轮询情况
    logger.info(
        f"[{task_id}] 任务状态查询，当前状态：{task_status_info['status']}, 已完成节点：{task_status_info['done_list']}")

    return task_status_info

# --------------------------
# 服务启动入口
# 直接运行此脚本即可启动FastAPI服务，无需额外执行uvicorn命令
# --------------------------
if __name__ == "__main__":
    """服务启动入口：本地开发环境直接运行"""
    logger.info("File Import Service 服务启动中...")
    # 启动uvicorn服务，绑定本地IP和8000端口，关闭自动重载（生产环境建议用workers多进程）
    uvicorn.run(
        app=app,
        host="127.0.0.1",  # 仅本地访问，生产环境改为0.0.0.0（允许所有IP访问）
        port=8000  # 服务端口
    )
