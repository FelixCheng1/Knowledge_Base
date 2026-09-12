import asyncio
import os
import json
import sys
import re
from app.query_process.agent.state import QueryGraphState
from app.core.logger import logger, node_log, step_log
from agents.mcp import MCPServerSse # pip install openai-agents
from agents.mcp import MCPServerStreamableHttp
from dotenv import load_dotenv # pip install openai-agents
from rich import print as rprint
from app.conf.bailian_mcp_config import mcp_config
from app.utils.task_utils import add_running_task,add_done_task

load_dotenv(override=True)

MCP_WEB_SEARCH_URL = os.environ.get("MCP_WEB_SEARCH_URL",)

async def mcp_call_streamable(query):
    search_mcp = MCPServerStreamableHttp(
        name="search_mcp",
        params={
            "url": MCP_WEB_SEARCH_URL,
            # "headers": {"Authorization": DASHSCOPE_API_KEY},
            "timeout": 300,
            "sse_read_timeout": 300,
            "terminate_on_close": True,
        },
        max_retry_attempts=2,
    )
    try:
        await search_mcp.connect()
        result = await search_mcp.call_tool(
            tool_name="tavily-search",
            arguments={"query": query, "max_results": 5},
        )
        return result
    finally:
        await search_mcp.cleanup()

def parse_search_docs(text: str) -> list[dict[str, str]]:
    text = text.replace("\r\n", "\n")

    pattern = (
        r"^Title:[ \t]*(?P<title>[^\n]*)\n"
        r"URL:[ \t]*(?P<url>[^\n]*)\n"
        r"Content:[ \t]*(?P<content>.*?)"
        r"(?=\n+Title:[^\n]*\nURL:[^\n]*\nContent:|\Z)"
    )

    return [
        {
            "Title": match.group("title").strip(),
            "URL": match.group("url").strip(),
            "Content": match.group("content").strip(),
        }
        for match in re.finditer(
            pattern, text, flags=re.MULTILINE | re.DOTALL
        )
    ]

@node_log("node_web_search_mcp")
def node_web_search_mcp(state:QueryGraphState):
    """
    节点功能，调用外部搜索引擎补充信息

    :param state: 图的流转状态
    :return: 需要更新的图状态
    """
    add_running_task(state["session_id"], "node_web_search_mcp", state["is_stream"])

    query = state.get("rewritten_query","")
    docs = []
    # 如果没有查询内容，直接返回
    if query:
        result = asyncio.run(mcp_call_streamable(query))
        if result:
            docs.extend(parse_search_docs(result.content[0].text))

    add_done_task(state["session_id"], "node_web_search_mcp", state["is_stream"])

    if docs:
        return {"web_search_docs": docs}
    return {}

if __name__ == "__main__":
    query = "agent/全栈工程师岗位所需技术栈"
    test_state = {
        "session_id": "node_web_search_mcp_001",
        "rewritten_query": f"{query}",
        "is_stream": False
    }
    # 调用 websearch_node 函数
    result_state = node_web_search_mcp(test_state)

    # 验证结果
    rprint("测试结果:")
    rprint(f"查询内容: {test_state.get('rewritten_query')}")

    # 输出搜索结果
    search_results = result_state.get('web_search_docs', [])
    rprint(f"搜索结果数量: {len(search_results)}")
    rprint("search_results", search_results)

    