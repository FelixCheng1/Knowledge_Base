import json
import os
import re
from pathlib import Path
from typing import Tuple, List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task

# ====================== 全局配置（可根据模型调整）======================
# 单个文本块最大长度（控制不超过模型上下文）
CHUNK_SIZE = 200 # 小值方便测试切割
# 块之间重叠长度（保证语义不丢失）
CHUNK_OVERLAP = 20

@step_log("step_1_get_content")
def step_1_get_content(state:ImportGraphState) -> Tuple[str, str]:
    """
    数据清洗，处理md_conten中不同系统的换行分割，统一处理
    并获取文件file_title用于整个内容的title兜底
    Args:
        state:流状态

    Returns:
        (MD文件, title)
    """
    md_content = state.get('md_content', '').strip()
    file_title = state.get("file_title") or "default_file"
    if not md_content:
        logger.error("没有输出内容，请检查输入内容是否正确！")
        raise  RuntimeError("没有输出内容，请检查输入内容是否正确！")
    md_content = md_content.replace("\r\n","\n").replace("\r","\n")
    return md_content, file_title


@step_log("step_2_split_by_title")
def step_2_split_by_title(md_content:str, file_title:str) -> List[dict]:
    """
    语义切割，根据标题，进行内容切割
    Args:
        md_content:需要切割的md内容
        file_titel:md文档名称
    Returns:
        [{content,titel,file_titel}]
    """
    # 1.定义切割正则,按行切割
    title_pattern = re.compile(r"^\s*#{1,6}\s+.+")
    lines = md_content.split("\n")
    # 准备存储容器
    chunks = []
    current_title = ""  # 当前标题
    current_lines = []  # 当前标题下内容
    in_code_block = False   # 记录是否在代码块中

    # 2. 循环处理每行数据
    for line in lines:
        strip_line = line.strip()
        # 判断是否在代码块中
        if line.startswith("~~~") or line.startswith("```"):
            in_code_block = not in_code_block
            continue
        # 不在代码块且是标题
        if not in_code_block and title_pattern.match(strip_line):
            # 到了新标题，将之前内容追加到chunks里
            if current_title:
                chunks.append({
                    "content":"\n".join(current_lines),
                    "title":current_title.strip(),
                    "file_title":file_title
                })
            elif current_lines:
            # 第一个标题之前已经存在内容
                chunks.append({
                    "title": "前言",
                    "content": "\n".join(current_lines),
                    "file_title": file_title
                })
            # 追加后重置变量
            current_title = strip_line
            current_lines = [strip_line]
        else:
            # 当前行不是标题
            current_lines.append(strip_line)
    # 3. 最后一个chunk进行存储
    if current_title:
        chunks.append({
            "content":"\n".join(current_lines),
            "title":current_title.strip(),
            "file_title":file_title
        })
    # 4. 进行无标题兜底处理
    if not chunks:
        chunks = [{
            "content":md_content,
            "title":"无主题",
            "file_title":file_title
        }]
    return chunks

def step_3_refine_chunks(sections:List[dict]) -> List[dict]:
    """
    同一标题下，同一语义，进行长度二次切割
    Args:
        setions:按照标题切割的的数据
    Returns:
        二次切割的数据
    """
    spliter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        # 切割递归顺序
        separators = ["\n\n","\n","。","！","；"," ",""]
    )
    # 进行切割
    final_chunks = []
    for section in sections:
        # 进行二次切割
        sub_chunks = spliter.split_text(section["content"])
        has_multiple_chunks = len(sub_chunks) > 1
        # 生成带编号的子块
        for idx,chunk in enumerate(sub_chunks,start=1):
            current_title = f"{section['title']}_{idx}" if has_multiple_chunks else section["title"]
            final_chunks.append({
                "title": current_title,
                "content":chunk.strip(),
                "file_title":section["file_title"],
                "parent_title": section["title"],
                "part": idx
            })
    return final_chunks

def step_4_backup_chunks(final_chunks:List[dict],state:ImportGraphState):
    """
    进行最终数据备份
    Args:
        final_chunks:  要备份的数据
        state: 节点中的转递状态
    Returns:

    """
    backup_file_path = Path(state["md_path"]).parent / "backup_chunks.json"

    with open(backup_file_path, "w", encoding="utf-8") as f:
        json.dump(
            final_chunks,
            f,
            ensure_ascii=False, #中文直接替换
            indent=4 # json带有缩进4
        )
    logger.debug(f"数据备份成功，备份地址：{backup_file_path}")


@node_log("node_document_split")
def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
    """
    # 1. 进行任务和日志处理
    add_running_task(state['task_id'],'node_document_split')
    # 2. 进行state中数据清晰(md_content / file_title (做标题兜底))
    md_content, file_title = step_1_get_content(state)
    # 3. 按标题语义初切
    # [{content:标题的内容,title：标题,file_title：文件名},{},{}]
    sections = step_2_split_by_title(md_content,file_title)
    # 4. 进行语义内递归切割
    # [{content:标题的内容,title：标题,file_title：文件名,parent_title,part},{},{}]
    final_chunks = step_3_refine_chunks(sections)
    # 5. 数据备份和修改state chunks
    state['chunks'] = final_chunks
    step_4_backup_chunks(final_chunks,state)
    add_done_task(state['task_id'], 'node_document_split')
    return state

if __name__ == '__main__':
    """
    单元测试：联合node_md_img（图片处理节点）进行集成测试
    测试条件：1.已配置.env（MinIO/大模型环境） 2.存在测试MD文件 3.能导入node_md_img
    测试流程：先运行图片处理→再运行文档切分，验证端到端流程
    """

    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.nodes.node_md_img import node_md_img

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": "",
            "file_title": "hak180产品安全手册",
            "local_dir":os.path.join(PROJECT_ROOT, "output"),
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
        logger.info("\n=== 开始执行文档切分节点集成测试 ===")

        logger.info(">> 开始运行当前节点：node_document_split（文档切分）")
        final_state = node_document_split(result_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk{final_chunks}")