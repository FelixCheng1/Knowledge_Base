import json
import uuid
import sys
from pathlib import Path
from IPython.display import display

# 以脚本方式运行时，把项目根目录加入 sys.path，才能导入 app 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.query_process.agent.main_graph import query_app
from app.query_process.agent.state import create_query_default_state
from app.core.logger import logger

logger.info("===== 开始测试 =====")

# ⚠️ 每次运行用新的 session_id，避免读到 Mongo 里的旧历史污染本轮测试
session_id = f"test_query_{uuid.uuid4().hex[:8]}"

initial_state = create_query_default_state(
    session_id=session_id,
    # 使用知识库中已导入的商品（如 万用表RS-12 / hak180）才能测到完整检索链路；
    # 若用库里没有的商品（如"华为P60怎么样?"）会走"拒绝回答"分支，仅能测反问/拒绝路径
    original_query="'RS PRO RS-12数字万用表怎么使用?"
    # original_query="华为P60怎么样?"  # 拒绝分支测试用
)
final_state = None

# 只输出更最终的状态值（字典形式），不包含节点名称、执行日志、元数据等额外信息
for event in query_app.stream(initial_state):
    for key, value in event.items():
        logger.info(f"节点: {key}")
        final_state = value

# 格式化输出最终状态
# default=str：兜底 Milvus 原生对象等不可 JSON 序列化的内容
logger.info(f"最终状态: {json.dumps(final_state, indent=4, ensure_ascii=False, default=str)}")

logger.info("图结构:")
# uv add grandalf
# query_app.get_graph().print_ascii()

# display() 仅在 VS Code 交互式窗口 / Jupyter 中渲染图形，终端运行无效果
display(query_app)

logger.info(f"本次测试 session_id: {session_id}（可用 /history/{session_id} 或 Mongo 查看写入的历史）")
logger.info("===== 测试结束 =====")