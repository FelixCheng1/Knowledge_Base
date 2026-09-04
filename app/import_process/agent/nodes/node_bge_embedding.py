import os
from typing import Any, List, Dict
import copy

from dotenv import load_dotenv
from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
from app.utils.task_utils import add_running_task,add_done_task
from app.core.logger import logger,node_log,step_log



"""
 1. 输入校验：验证chunks有效性，核心数据缺失则终止当前节点
 2. 模型初始化：获取BGE-M3单例模型实例，避免重复加载
 3. 批量向量化：分批拼接文本、生成双向量，为切片绑定向量字段
 4. 状态更新：将带向量的chunks更新回全局状态，供下游Milvus入库节点使用
"""

@step_log("step_1_validate_input")
def step_1_validate_input(state: ImportGraphState) -> List[Dict[str,Any]]:
    """
    向量化前置步骤1：输入数据的有效性校验
    核心作用：
        1：从全局状态提取待向量化的chunks切片列表
        2：严格检验chunks类型和非空性，无效数据则终止向量化
    """
    texts2emdd = state.get("chunks")
    #校验：必须为非空列表，不然无法向量化
    if not isinstance(texts2emdd ,list) or not texts2emdd:
        logger.error(f"向量化chunks失败，chunks字段为空或非有效列表")
        raise ValueError("错误：无有效文本切片数据，无法执行向量化处理")

    logger.info(f"向量化输入校验通过，待处理文本切片数量：{len(texts2emdd)}")
    return texts2emdd

@step_log("step_2_init_model")
def step_2_init_model():
    """
    向量化步骤2.获取嵌入模型实例（单例模式）
    """
    try:
        ef = get_bge_m3_ef()
        #校验模型实例是否有效
        if ef is None:
            raise ValueError("BGE-M3模型实例为None：pymilvus.model模块未找到或模型加载失败")
        logger.info("BGE-M3模型实例初始化成功（单例模式）")
        return ef
    except Exception as e:
        # 包装异常信息，明确错误原因和排查方向
        error_msg = f"BGE-M3模型初始化失败：{e}，请检查模型路径/环境变量配置是否正确"
        logger.error(error_msg)
        raise ValueError(error_msg)

@step_log("step_3_generate_embeddings")
def step_3_generate_embeddings(texts2emdd:List[dict],bge_m3_ef) -> List[Dict[str,Any]]:
    """
    批量进行向量生成，返回稠密向量和稀疏向量
    Args:
        texts2emdd:准备向量化的chunks：List[dict]
        bge_m3_ef:BGE-M3模型实例
    Returns:
        final_chunks: 添加向量化的chunks
    """
    #初始化返回结果列表
    final_chunks = []
    #定义批次大小，根据显存和嵌入模型动态调整
    CHUNKS_SIZE = 5
    chunks_len = len(texts2emdd)
    #循环进行批处理
    for i in range(0, chunks_len, CHUNKS_SIZE):
        # 获取当前批次切片chunks
        current_chunks = texts2emdd[i: i + CHUNKS_SIZE]
        # 使用异常捕获进行批处理
        try:
            # 该批准备向量化的文本，使用ietm—name与content拼接
            input_texts = []
            for chunk in current_chunks:
                item_name = chunk["item_name"]
                content = chunk["content"]
                current_text = f"商品：{item_name}，介绍：{content}" if item_name else content
                input_texts.append(current_text)
            # 调用封装函数向量化向量
            current_embddings = generate_embeddings(input_texts)
            if not current_embddings:
                logger.error("该批次向量化为空，请检查输入文本是否为空")
                # 保证数据完整性
                final_chunks.extend(current_chunks)
            # 给chunks新填向量字段
            for j, chunk in enumerate(current_chunks):
                item = copy.copy(chunk)
                item["dense_vector"] = current_embddings["dense"][j]
                item["sparse_vector"] = current_embddings["sparse"][j]
                final_chunks.append(item)
        except Exception as e:
            logger.error(f"该批向量化失败：{e}")
            # 保证数据完整性
            final_chunks.extend(current_chunks)
            continue
    return final_chunks

@node_log("node_bge_embedding")
def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 向量化 (node_bge_embedding)
    为什么叫这个名字: 使用 BGE-M3 模型将文本转换为向量 (Embedding)。
    """
    # 日志和任务队列处理
    add_running_task(state['task_id'], "node_bge_embedding")
    # 步骤1：输入数据校验，核心chunks无效则抛出异常
    texts_to_embed = step_1_validate_input(state)
    # 步骤2：初始化BGE-M3模型（单例模式，仅加载一次）
    bge_m3_ef = step_2_init_model()
    # 步骤3：批量生成双向量，为切片绑定向量字段
    output_data = step_3_generate_embeddings(texts_to_embed, bge_m3_ef)
    # 步骤4: 输出数据处理
    state['chunks'] = output_data
    add_done_task(state['task_id'], "node_bge_embedding")
    return state

# ==========================================
# 本地单元测试入口
# 功能：独立验证向量化节点全链路逻辑，无需启动整个LangGraph流程
# 适用场景：本地开发、调试、模型有效性验证
# ==========================================
if __name__ == '__main__':
    # 加载环境变量：定位项目根目录下的.env，读取模型路径/设备等配置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    # 构造模拟测试状态：模拟上游节点输出的chunks数据，贴合真实业务场景
    test_state = ImportGraphState({
        "task_id": "test_task_embedding_001",  # 测试任务ID
        "chunks": [  # 模拟带item_name的文本切片（上游商品名称识别节点产出）
            {
                "content": "这是一个测试文档的内容，用于验证向量化是否成功。",
                "title": "测试文档标题",
                "item_name": "测试项目",
                "file_title": "测试文件.pdf"
            },
            {
                "content": "这是第二个测试文档的内容，用于验证批量处理逻辑。",
                "title": "测试文档标题2",
                "item_name": "测试项目",
                "file_title": "测试文件.pdf"
            }
        ]
    })
    
    # 执行本地测试
    logger.info("=== BGE-M3向量化节点本地单元测试启动 ===")
    try:
        # 调用核心节点函数
        result_state = node_bge_embedding(test_state)
        # 提取测试结果
        result_chunks = result_state.get("chunks", [])

        # 打印测试结果统计
        logger.info(f"=== 向量化节点本地测试完成 ===")
        logger.info(f"测试任务ID：{test_state.get('task_id')}")
        logger.info(f"待处理切片数：2 | 实际处理切片数：{len(result_chunks)}")

        # 验证向量生成结果（打印向量字段是否存在）
        for idx, chunk in enumerate(result_chunks):
            has_dense = "dense_vector" in chunk
            has_sparse = "sparse_vector" in chunk
            logger.info(
                f"第{idx + 1}条切片：稠密向量生成{'' if has_dense else '未'}成功 | 稀疏向量生成{'' if has_sparse else '未'}成功")

    except Exception as e:
        logger.error(f"=== 向量化节点本地测试失败 ===" f"错误原因：{str(e)}", exc_info=True)
        # 新手友好提示：给出核心排查方向
        logger.warning("排查提示：请检查BGE-M3模型路径、显存是否充足、环境变量配置是否正确")