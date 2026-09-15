from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.finance.api import router as finance_router


app = FastAPI(
    title="掌柜智库·金融知识库 API",
    summary="面向金融资料导入、检索、可溯源问答的本地 API。",
    description="""所有业务接口使用 `/api/v1` 前缀。\n\n
- 文档导入是异步任务，创建后轮询 `import-tasks` 查询状态。
- 问答创建后返回 `query_id`，前端通过 SSE 订阅最终答案与引用。
- 金融回答仅依据已导入资料，不提供个性化投资建议。""",
    version="1.0.0",
    openapi_tags=[
        {"name": "Health", "description": "服务与依赖就绪状态。"},
        {"name": "Documents", "description": "金融资料的导入、查看和停用。"},
        {"name": "Import tasks", "description": "异步导入任务状态与失败重试。"},
        {"name": "Sessions", "description": "金融问答会话和历史消息。"},
        {"name": "Queries", "description": "可溯源金融资料问答与 SSE 事件流。"},
    ],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(finance_router)


@app.on_event("startup")
def recover_interrupted_finance_jobs() -> None:
    """单进程重启后，明确标记未完成的查询和导入任务，避免永久停留在 processing。"""
    try:
        from app.finance.api import get_service
        service = get_service()
        service.recover_interrupted_queries()
        service.recover_interrupted_imports()
    except Exception:
        # 服务可以在中间件启动前先提供 /health；错误由健康检查显示。
        pass
