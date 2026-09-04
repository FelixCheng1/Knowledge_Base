import os
from typing import Tuple

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pymilvus import DataType
from app.conf.milvus_config import milvus_config
# 导入自定义模块：
# 1. 流程状态载体：ImportGraphState为LangGraph流程的统一状态管理对象
from app.import_process.agent.state import ImportGraphState
# 2. Milvus工具：获取单例Milvus客户端，实现连接复用
from app.clients.milvus_utils import get_milvus_client
# 3. 大模型工具：获取大模型客户端，统一模型调用入口
from app.lm.lm_utils import get_llm_client
# 4. 向量工具：BGE-M3模型实例、向量生成方法（稠密+稀疏向量）
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
# 5. 稀疏向量工具：归一化处理，保证向量长度为1，提升检索准确性
from app.utils.normalize_sparse_vector import normalize_sparse_vector
# 6. 任务工具：更新任务运行状态，用于任务监控和管理
from app.utils.task_utils import add_running_task, add_done_task
# 7. 日志工具：项目统一日志入口，分级输出（info/warning/error）
from app.core.logger import logger,node_log,step_log
# 8. 提示词工具：加载本地prompt模板，实现提示词与代码解耦
from app.core.load_prompt import load_prompt

from app.utils.escape_milvus_string_utils import escape_milvus_string

# --- 配置参数 (Configuration) ---
# 大模型识别商品名称的上下文切片数：取前5个切片，避免上下文过长导致大模型输入超限
DEFAULT_ITEM_NAME_CHUNK_K = 5
# 单个切片内容截断长度：防止单切片内容过长，占满大模型上下文
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
# 大模型上下文总字符数上限：适配主流大模型输入限制，默认2500
CONTEXT_TOTAL_MAX_CHARS = 2500



"""
  主要目标：
     1. 录用文本大模型识别当前chunks对应的item_name！用于区分不同的文档
     2. 使用嵌入式模型，将item_name生成向量存储到向量数据库 
     3. 修改state[chunks] -> chunk {title parent_title part file_title content item_name => 每个赋值 }
  实现步骤：
     1. 校验和取值 （file_title,chunks）
     2. 构建上下文环境  chunks -> top 5 -> 拼接成context文本 
     3. 调用模型，拼接提示词，识别chunks对应item_name
     4. 修改state chunks -》 item_name 
     5. item_name生成向量（稠密/稀疏）
     6. 存储向量到向量数据库 kb_item_name (id / file_title / item_name / 稠密 和 稀疏)
 """
@step_log("step_1_get_chunks_and_file_title")
def step_1_get_chunks_and_file_title(state:ImportGraphState) -> Tuple[list,str]:
    """
    进行参数校验和处理
    Args:
        state:节点间传递状态
    Returns:
        state中的chunks和file_title
    """
    chunks = state.get("chunks")
    file_tetle = state.get("file_title")

    if not chunks:
        raise ValueError("chunks没有值，无法继续进行，抛出异常处理")
    if not file_tetle:
        file_tetle = os.path.basename(state.get("md_path"))
        state["file_title"] = file_tetle

    return chunks, file_tetle

@step_log("step_2_build_context")
def step_2_build_context(chunks) -> str:
    """
    构建提示词上下文环境
    截取内容限制： 1. 最多截取前top个 （5） 2. 最多字符不能超过 CONTEXT_TOTAL_MAX_CHARS
    截取内容处理：
          切片：{1}，标题:{title},内容：{content} \n\n
          切片：{2}，标题:{title},内容：{content} \n\n
          切片：{3}，标题:{title},内容：{content} \n\n
          切片：{4}，标题:{title},内容：{content} \n\n
          切片：{5}，标题:{title},内容：{content} \n\n
    限制：
    1. 最多取 DEFAULT_ITEM_NAME_CHUNK_K 个 chunk
    2. 单个 chunk content 最大 SINGLE_CHUNK_CONTENT_MAX_LEN
    3. 总 context 最大 CONTEXT_TOTAL_MAX_CHARS
    Args:
        chunks: chunks列表
    Returns:
        构建的context 方便后面LLM总结
    """
    # 1. 前置准备工作
    parts = []  # 存储处理后的切片
    total_chars = 0
    separator = "\n\n" 
    # 2. 循环处理content + 判断
    for index, chunk in enumerate(chunks[:DEFAULT_ITEM_NAME_CHUNK_K],start=1):
        title = chunk["title"]
        content = chunk["content"][:SINGLE_CHUNK_CONTENT_MAX_LEN]

        prefix = f"切片：{index}，标题：{title}，内容："
        separator_len = len(separator) if parts else 0

        available_content_chars = (
            CONTEXT_TOTAL_MAX_CHARS
            - total_chars
            - separator_len
            - len(prefix)
        )

        #剩余空间连当前切片的固定格式都放不下
        if available_content_chars <= 0:
            logger.info(f"上下文剩余窗口不足，当前字符数：{total_chars}，停止拼接！")
            break

        content = content[:available_content_chars]
        data = prefix + content

        parts.append(data)
        total_chars += separator_len + len(data)

        if total_chars >= CONTEXT_TOTAL_MAX_CHARS:
            logger.info(f"已经达到最大字符数：{CONTEXT_TOTAL_MAX_CHARS}，停止拼接！")
            break
    return "\n\n".join(parts)

@step_log("step_3_call_llm")
def step_3_call_llm(context:str, file_title:str) -> str:
    """
    LLM调用，获取item_name
    使用file_title兜底
    Args:
        context:上一步构筑的上下文
        file_title:md文件名，兜底用
    Returns:
        LLM总结的item_name
    """
    # 1. 构筑提示词
    human_prompt = load_prompt("item_name_recognition",file_title=file_title,context=context)
    system_prompt = load_prompt("product_recognition_system")
    messages = [
        SystemMessage(system_prompt),
        HumanMessage(human_prompt)
    ]
    # 2.获取模型对象
    llm = get_llm_client()
    # 3.构建调用链
    chain = llm | StrOutputParser()
    item_name = chain.invoke(messages)
    # 4.进行判断
    if not item_name:
        item_name = file_title
    # 5.返回结果
    return item_name

@step_log("step_4_update_chunks_and_state")
def step_4_update_chunks_and_state(state:ImportGraphState,item_name:str,chunks:list):
    """
    state[item_name] = item_name
    chunks -> {item_name:item_name}
    """
    state["item_name"] = item_name

    for chunk in chunks:
        chunk["item_name"] = item_name
    state["chunks"] = chunks
    logger.info("完成了chunks和state['item_name']的赋值和修改")

@step_log("step_5_generate_embeddings")
def step_5_generate_embeddings(item_name:str):
    """
    根据item_name生成两种向量 稠密 + 稀疏
    Args:
        item_name:需要的转换成向量的字符串
    Returns:
        两种向量
    enerate_embeddings 自己封装的嵌入式模式生成向量的函数！！ 
          embeddings list对应的向量 = model.encode_documents(texts) 传入的字符串list 
          参数：生成向量的字符串 ["1","2","3"] 
          返回结果： 
             result = {
                        "dense": [1的稠密,2的稠密,3的稠密],  #稠密向量
                        "sparse": [1的稀疏,2的稀疏,3的稀疏], #稀疏向量
                      }
    """
    result = generate_embeddings([item_name])
    dense_vector,sparse_vector = result["dense"][0],result["sparse"][0]
    return dense_vector,sparse_vector

@step_log("step_6_save_to_vector_db")
def step_6_save_to_vector_db(file_title, item_name, dense_vector, sparse_vector):
    """
    将向量和对应的字段保存到向量数据库中
    Args:
        file_title:原始文档名称，用于数据溯源与文档级管理
        item_name:文档核心主体（如 “iPhone 17 Pro Max”），解决跨文档主体模糊问题
        dense_vector:主体名称的语义向量，支持模糊语义匹配
        sparse_vector:主体名称的关键词向量，支持专业术语、实体的精确匹配
    """
    # 1.获取milvus客户端
    milvus_client = get_milvus_client()
    # 2.判断是否存在集合，不存在创建集合
    if not milvus_client.has_collection(collection_name=milvus_config.item_name_collection):
        #创建集合
        # 添加约束
        schema = milvus_client.create_schema(
            auto_id  = True,
            enable_dynamic_field = True
        )
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True,auto_id=True)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length =65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length =65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        # 添加索引
        index_params =milvus_client.prepare_index_params()

        index_params.add_index(
            field_name="dense_vector",  # 给哪个向量列创建索引
            index_name="dense_vector_index", # 索引名称
            index_type="HNSW",  # 配置查找所用的算法
            metric_type="COSINE"
        )

        index_params.add_index(
                    field_name="sparse_vector",  # 给哪个向量列创建索引
                    index_name="sparse_vector_index", # 索引名称
                    index_type="SPARSE_INVERTED_INDEX",  # 配置查找所用的算法
                    metric_type="IP"
        )

        milvus_client.create_collection(
            collection_name=milvus_config.item_name_collection,
            schema=schema,
            index_params=index_params
        )

    # 删除之前存在的item_name
    milvus_client.delete(
        collection_name=milvus_config.item_name_collection,
        filter = f'item_name=="{escape_milvus_string(item_name)}"'
    )
    # 4. 向集合插入新的数据
    item = {
        "file_title":file_title,
        "item_name":item_name,
        "dense_vector":dense_vector,
        "sparse_vector":sparse_vector
    }
    milvus_client.insert(collection_name=milvus_config.item_name_collection,data=[item])
    milvus_client.load_collection(collection_name=milvus_config.item_name_collection)
    logger.info(f"保存了item_name：{item_name}的数据到向量数据库中！")

@node_log("node_item_name_recognition")
def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 主体识别 (node_item_name_recognition)
    为什么叫这个名字: 识别文档核心描述的物品/商品名称 (Item Name)。
    """
    # 日志和任务处理
    add_running_task(state['task_id'],'node_item_name_recognition')
    # 1. 校验和取值 （file_title,chunks）
    # 获取前置的材料！ file_title = 为了兜底，没有识别到item_name
    chunks, file_title = step_1_get_chunks_and_file_title(state)
    # 2. 构建上下文环境  chunks -> top 5 -> 拼接成context文本
    context = step_2_build_context(chunks)
    # 3. 调用模型，拼接提示词，识别chunks对应item_name
    item_name = step_3_call_llm(context, file_title)
    # 4. 修改state chunks -》 item_name -> chunks [{title parent_title context part item_name [没有值]}]
    step_4_update_chunks_and_state(state, item_name, chunks)
    # 5. item_name生成向量（稠密/稀疏）
    dense_vector, sparse_vector = step_5_generate_embeddings(item_name)
    # 6. 将向量存储到向量数据库 kb_item_name (id / file_title / item_name / 稠密 和 稀疏)
    step_6_save_to_vector_db(file_title, item_name, dense_vector, sparse_vector)
    
    add_done_task(state['task_id'], 'node_item_name_recognition')
    return state

# ===================== 本地测试方法（直接运行调试，无需启动LangGraph） =====================
def test_node_item_name_recognition():
    """
    商品名称识别节点本地测试方法
    功能：模拟LangGraph流程输入，独立测试node_item_name_recognition节点全链路逻辑
    适用场景：本地开发、调试、单节点功能验证，无需启动整个LangGraph流程
    测试前准备：
        1. 确保项目环境变量配置完成（MILVUS_URL/ITEM_NAME_COLLECTION等）
        2. 确保大模型、Milvus、BGE-M3服务均可正常访问
        3. 确保prompt模板（item_name_recognition/product_recognition_system）已存在
    使用方法：
        直接运行该函数：if __name__ == "__main__": test_node_item_name_recognition()
    """
    logger.info("=== 开始执行商品名称识别节点本地测试 ===")
    try:
        # 1. 构造模拟的ImportGraphState状态（模拟上游节点产出数据）
        mock_state = ImportGraphState({
            "task_id": "test_task_123456",  # 测试任务ID
            "file_title": "华为Mate60 Pro手机使用说明书",  # 模拟文件标题
            "file_name": "华为Mate60Pro说明书.pdf",  # 模拟原始文件名（兜底用）
            # 模拟文本切片列表（上游切片节点产出，含title/content字段）
            "chunks": [
                {
                    "title": "产品简介",
                    "content": "华为Mate60 Pro是华为公司2023年发布的旗舰智能手机，搭载麒麟9000S芯片，支持卫星通话功能，屏幕尺寸6.82英寸，分辨率2700×1224。"
                },
                {
                    "title": "拍照功能",
                    "content": "华为Mate60 Pro后置5000万像素超光变摄像头+1200万像素超广角摄像头+4800万像素长焦摄像头，支持5倍光学变焦，100倍数字变焦。"
                },
                {
                    "title": "电池参数",
                    "content": "电池容量5000mAh，支持88W有线超级快充，50W无线超级快充，反向无线充电功能。"
                }
            ]
        })

        # 2. 调用商品名称识别核心节点
        result_state = node_item_name_recognition(mock_state)

        # 3. 打印测试结果（调试用）
        logger.info("=== 商品名称识别节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"最终识别商品名称：{result_state.get('item_name')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")
        logger.info(f"第一个切片商品名称：{result_state.get('chunks', [{}])[0].get('item_name')}")

        # 4. 验证Milvus存储（可选）
        milvus_client = get_milvus_client()
        collection_name = os.environ.get("ITEM_NAME_COLLECTION")
        if milvus_client and collection_name:
            milvus_client.load_collection(collection_name)
            # 检索测试结果
            item_name = result_state.get('item_name')
            safe_name = escape_milvus_string(item_name)
            res = milvus_client.query(
                collection_name=collection_name,
                filter=f'item_name=="{safe_name}"',
                output_fields=["file_title", "item_name"]
            )
            logger.info(f"Milvus中检索到的数据：{res}")

    except Exception as e:
        logger.error(f"商品名称识别节点本地测试失败，原因：{str(e)}", exc_info=True)


# 测试方法运行入口：直接执行该文件即可触发测试
if __name__ == "__main__":
    # 执行本地测试
    test_node_item_name_recognition()