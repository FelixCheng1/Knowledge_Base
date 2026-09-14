import sys
from typing import List, Dict, Any
from app.utils.task_utils import add_running_task, add_done_task
from app.query_process.agent.state import QueryGraphState
from app.core.logger import logger, node_log, step_log

# 将向量化检索结果进行处理，转化为只包含实体信息的列表
def _as_entity_list(chunks:list) -> List[Dict[str,Any]]:

    out:List[Dict[str,Any]] = []
    for chunk in chunks:
        if not chunk:
            continue

        final_entity = {}

        # ==============================================
        # 情况A：处理 Milvus 返回的 Hit 对象（含 entity、id、distance）
        # ==============================================
        if hasattr(chunk, "entity") and hasattr(chunk, "distance"):
            # 提取entity内容
            entity = chunk.entity
            if hasattr(entity, "to_dict"):
                final_entity = entity.to_dict()
            elif isinstance(entity, dict):
                final_entity = entity.copy()
            else:
                # 尝试强行转字典，兼容不同SDK版本
                try:
                    final_entity = dict(entity)
                except:
                    pass

            # 补充唯一ID（Milvus Hit 对象的属性是 id，不是 chunkid）
            if "chunk_id" not in final_entity:
                final_entity["id"] = getattr(chunk, "id", None)
            # 补充相似性
            if hasattr(chunk, "distance"):
                final_entity["score"] = chunk.distance

        # ==============================================
        # 情况B：chunk 已经是字典（模拟数据 / 已格式化数据）
        # ==============================================
        elif isinstance(chunk, dict):
            # 子情况：字典嵌套 entity 结构 {entity:{...}, id:...}
            if "entity" in chunk:
                ent = chunk["entity"]
                if isinstance(ent, dict):
                    final_entity = ent.copy()
                # 补充 ID 和分数
                if "id" in chunk and "id" not in final_entity:
                    final_entity["id"] = chunk["id"]
                if "distance" in chunk:
                    final_entity["score"] = chunk["distance"]
            else:
                # 扁平字典，直接使用
                final_entity = chunk
        # ==============================================
        # 情况C：支持 .get() 方法的其他对象
        # ==============================================
        elif hasattr(chunk, "get"):
            ent = chunk.get("entity") or chunk
            if isinstance(ent, dict):
                final_entity = ent

        # 只保留合法非空字典
        if final_entity and isinstance(final_entity, dict):
            out.append(final_entity)

    return out

@step_log("step_2_rrf")
def step_2_rrf( 
        source_weights: list,
        k: int = 60,
        max_results: int = None,
) -> List[tuple]:
    """
    通用带权重的RRF算法实现

    :param source_weights:  列表，每个元素是(来源文档列表, 权重)的元组
                            例如: [([doc1, doc2], 1.0), ([doc2, doc3], 0.8)]
    :param k:     RRF 常数，默认 60。用于平滑排名影响，避免高排名文档占据过大优势。
    :param max_results: 只返回前 N 个，None 表示全部
    :return:      [(元素, RRF 得分), ...] 按得分降序排列
    """
    # 存储每个文档的总得分
    score_map = {}
    # 存储每个文档的完整内容
    chunk_map = {}

    # 遍历每一路召回结果，计算 RRF 结果
    for chunks, weight in source_weights:
        # rank 从 1 开始
        for rank, chunk in enumerate(chunks, start=1):
            # 获取chunk_id（兼容 Milvus 实体只返回主键 id 的情况）
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id is None:
                # 无唯一标识无法参与融合，跳过
                continue
            # 计算chunk_id所对应的分数
            score_map[chunk_id] = score_map.get(chunk_id, 0.0) + weight * (1 / (k + rank))
            # 存储chunk_id 所对应的数据
            chunk_map.setdefault(chunk_id, chunk)

    # 按 RRF 总分排序
    merged = []
    # 对score—map 进行遍历
    for chunk_id, score in score_map.items():
        chunk = chunk_map[chunk_id]
        merged.append((chunk, score))

    # 得分从高到低排序
    merged.sort(key=lambda x: x[1], reverse=True)

    # 截断最多返回N体
    if max_results is not None:
        merged = merged[:max_results]

    return merged


@node_log("node_rrf")
def node_rrf(state:QueryGraphState):
    """
    节点功能：Reciprocal Rank Fusion
    将多路召回的结果（向量、HyDE、Web、KG）进行加权融合排序。
    """
    logger.info("---RRF (倒数排名融合) 开始处理---")
    add_running_task(state["session_id"], "node_rrf", state.get("is_stream"))

    # 步骤1：从 state 取出两路召回结果

    embedding_chunks = _as_entity_list(state.get("embedding_chunks"))
    hyde_embedding_chunks = _as_entity_list(state.get("hyde_embedding_chunks"))

    # 配置多路权重（可根据业务调整）
    source_weights = [
        (embedding_chunks, 1.0),
        (hyde_embedding_chunks, 1.0)
    ]

    # 步骤3：执行 RRF 融合排序
    rrf_res = step_2_rrf(source_weights, k=60, max_results=10)

    # 步骤4：提取最终文档列表
    rrf_chunks = [chunk for chunk, score in rrf_res]

    # 任务完成标记
    add_done_task(state['session_id'], "node_rrf", state.get("is_stream"))

    # 把融合结果存入 state
    return {"rrf_chunks": rrf_chunks}

# ================================
# 本地测试入口
# ================================
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_rrf 本地测试")
    print("=" * 50)

    mock_state = {
        "session_id": "test_rrf_session",
        "is_stream": False,
        "original_query": "RS PRO RS-12数字万用表怎么操作？",
        "rewritten_query": "RS PRO RS-12数字万用表的具体操作步骤是什么？",
        "item_names": ["RS PRO RS-12数字万用表"]
    }

    try:
        from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
        from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde

        emb_res = node_search_embedding(mock_state)
        hyde_res = node_search_embedding_hyde(mock_state)
        mock_state['embedding_chunks'] = emb_res.get("embedding_chunks") or []
        mock_state['hyde_embedding_chunks'] = hyde_res.get("hyde_embedding_chunks") or []

        result = node_rrf(mock_state)
        rrf_chunks = result.get("rrf_chunks", [])

        emb_cnt = len(mock_state.get("embedding_chunks") or [])
        hyde_cnt = len(mock_state.get("hyde_embedding_chunks") or [])

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"输入数量: Embedding={emb_cnt}, HyDE={hyde_cnt}")
        print(f"输出数量: {len(rrf_chunks)}")
        print("-" * 30)

        print("最终排名:")
        for i, doc in enumerate(rrf_chunks, 1):
            doc_id = doc.get("chunk_id") or doc.get("id")
            content = (doc.get("content") or "")[:20]
            print(f"Rank {i}: ID={doc_id}, Content={content}...")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")