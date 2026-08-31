import sys

from app.core.logger import logger, node_log
from app.import_process.agent.state import ImportGraphState

@node_log("node_entry")
def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点：入口节点(node_entry)
    未来要实现
    1. 接收文件路径
    2. 判断文件类型
    3. 设置 state 中的路由标记 (is_md_read_enabled / is_pdf_read_enabled)
    """

    # 模拟简单的路由逻辑，防止报错 (仅 node_entry 需要)
    if "local_file_path" in state:
        path = state["local_file_path"]
        if path.endswith(".pdf"):
            state["is_pdf_read_enabled"] = True
        elif path.endswith(".md"):
            state["is_md_read_enabled"] = True

    return state
