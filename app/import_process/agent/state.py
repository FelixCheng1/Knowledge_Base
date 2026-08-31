from typing import TypedDict
import copy
from app.core.logger import logger

class ImportGraphState(TypedDict):
    """
    图的状态定义，包含所有节点和消费的数据字段
    """
    task_id = str       #任务唯一ID，用于追踪任务

    # —————— 流程控制标记 ——————
    is_md_read_enabled:bool     # 是否启用Markdown读取路径
    is_pdf_read_enabled:bool    # 是否启用PDF读取路径

    # —————— 路径相关 ——————
    local_dir:str       #当前工作目录或输出目录
    local_file_path:str #原始输入文件按路径
    file_title:str      #文件标题（去后缀）
    pdf_path:str        #PDF文件路径
    md_path:str         #Markdown文件路径（转换后或直接输入的）

    # —————— 内容数据 ——————
    md_content:str      #Markdown的全文内容
    chunks:list         #切片后的文本列表，包括 metadata
    item_name:str       #识别出的主体名称，用于增强检索

    # —————— 数据库相关 ——————
    embedings_content:list  #包含向量数据的列表，准备写入Milvus


# 定义图状态的默认初始对象，方便以后使用
graph_default_state: ImportGraphState = {
    "task_id":"",
    "is_md_read_enabled":False,
    "is_pdf_read_enabled":False,
    "local_dir":"",
    "local_file_path":"",
    "pdf_path":"",
    "md_path":"",
    "file_title":"",
    "md_content":"",
    "chunks":[],
    "item_name":"",
    "embedings_content":[]
}

def create_default_state(**overrides) ->ImportGraphState:
    """
    创建默认状态，支持覆盖
    Args：
        **overrides：要覆盖的字段（关键字参数解包）
    Returns:
        新的状态实例
    Examples:
        state = create_default_state(task_id="task_001",local_file_path="doc.pdf")
    """
    # 默认状态
    state = copy.deepcopy(graph_default_state)
    # 用 overrides 覆盖默认默认值
    state.update(overrides)
    # 返回创建好的状态字典实例
    return state

def get_default_state() -> ImportGraphState:
    """
    返回一个新的状态实列，避免全局变量污染
    """
    return copy.deepcopy(graph_default_state)

if __name__ == "__main__":
    """
    测试
    """
    state = create_default_state(local_file_path="万用表RS-2的使用.pdf")
    logger.info(state)