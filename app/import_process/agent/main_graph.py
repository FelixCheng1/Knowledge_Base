from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from app.core.logger import logger
from  typing import Literal

from app.import_process.agent.state import ImportGraphState, create_default_state
from app.import_process.agent.nodes.node_entry import node_entry  # 入口节点：初始化参数、校验输入
from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md  # PDF转MD：解析PDF文件为markdown格式
from app.import_process.agent.nodes.node_md_img import node_md_img  # MD图片处理：提取/下载markdown中的图片、修复图片路径
from app.import_process.agent.nodes.node_document_split import node_document_split  # 文档分块：将长文档切分为符合模型要求的小片段
from app.import_process.agent.nodes.node_item_name_recognition import node_item_name_recognition  # 项目名识别：从分块中提取核心项目名称（业务定制化）
from app.import_process.agent.nodes.node_bge_embedding import node_bge_embedding  # BGE向量化：将文本分块转换为向量表示（适配Milvus向量库）
from app.import_process.agent.nodes.node_import_milvus import node_import_milvus  # 导入Milvus：将向量数据写入Milvus向量数据库

# 初始化环境变量
load_dotenv(override=True)

# 1.初始化状态图
workflow = StateGraph(ImportGraphState)

# 2.注册所有业务节点
workflow.add_node("node_entry",node_entry)
workflow.add_node("node_pdf_to_md",node_pdf_to_md)
workflow.add_node("node_md_img",node_md_img)
workflow.add_node("node_document_split",node_document_split)
workflow.add_node("node_item_name_recognition",node_item_name_recognition)
workflow.add_node("node_bge_embedding",node_bge_embedding)
workflow.add_node("node_import_milvus",node_import_milvus)

# 设置工作流入口节点
workflow.set_entry_point("node_entry")

# 3.定义路由函数
def router(state:ImportGraphState) -> Literal["node_pdf_to_md","node_md_img","__end__"]:
    """
    入口节点后的路由逻辑
    :param state:工作流全量状态对象，包含所有配置项和中间结果
    :return:目标节点标识/END，Langgraph会自动跳转到对应节点
    """
    if state.get("is_md_read_enabled"):
        return "node_md_img"
    elif state.get("is_pdf_read_enabled"):
        return "node_pdf_to_md"
    else:
        return END

# 注册条件边
workflow.add_conditional_edges(
    "node_entry",
    router,
    path_map={
        "node_md_img":"node_md_img",
        "node_pdf_to_md":"node_pdf_to_md",
        END:END
    }
)

# 5.注册静态边
workflow.add_edge("node_pdf_to_md", "node_md_img")  # PDF转MD完成 → MD图片处理
workflow.add_edge("node_md_img", "node_document_split")  # MD处理完成 → 文档分块
workflow.add_edge("node_document_split", "node_item_name_recognition")  # 分块完成 → 项目名识别
workflow.add_edge("node_item_name_recognition", "node_bge_embedding")  # 项目名识别完成 → BGE向量化
workflow.add_edge("node_bge_embedding", "node_import_milvus")  # 向量化完成 → 导入Milvus向量库
workflow.add_edge("node_import_milvus", END)  # Milvus入库完成 → 工作流执行结束（END是内置结束节点）

# 6.编译工作流
kb_import_app = workflow.compile()