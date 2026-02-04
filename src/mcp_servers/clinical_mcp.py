"""
Clinical MCP Server
가이드라인/논문 검색 (Milvus + Hybrid Search)
"""

from typing import Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from loguru import logger

from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import Reranker


# MCP 서버 인스턴스
server = Server("clinical-mcp")

# 검색 엔진 (lazy initialization)
_hybrid_search = None
_reranker = None


def get_hybrid_search() -> HybridSearch:
    global _hybrid_search
    if _hybrid_search is None:
        _hybrid_search = HybridSearch()
    return _hybrid_search


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


@server.list_tools()
async def list_tools() -> list[Tool]:
    """사용 가능한 도구 목록"""
    return [
        Tool(
            name="search_clinical_guidelines",
            description="임상 가이드라인 및 의료 문헌을 검색합니다. 증상, 진단, 치료 방법 등에 대한 정보를 찾을 때 사용합니다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색할 의료 관련 쿼리 (증상, 질병명, 치료법 등)",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "반환할 결과 수 (기본값: 5)",
                        "default": 5,
                    },
                    "use_rerank": {
                        "type": "boolean",
                        "description": "Reranking 사용 여부 (기본값: true)",
                        "default": True,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="search_by_disease",
            description="특정 질병에 대한 가이드라인을 검색합니다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "disease_name": {
                        "type": "string",
                        "description": "질병명 (예: 당뇨병, 고혈압, 폐렴)",
                    },
                    "aspect": {
                        "type": "string",
                        "description": "검색할 측면 (diagnosis, treatment, prevention, prognosis)",
                        "enum": ["diagnosis", "treatment", "prevention", "prognosis"],
                    },
                },
                "required": ["disease_name"],
            },
        ),
        Tool(
            name="search_treatment_protocol",
            description="특정 증상이나 상태에 대한 치료 프로토콜을 검색합니다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "condition": {
                        "type": "string",
                        "description": "증상 또는 상태",
                    },
                    "patient_context": {
                        "type": "string",
                        "description": "환자 맥락 정보 (나이, 기저질환 등)",
                    },
                },
                "required": ["condition"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """도구 실행"""
    logger.info(f"Clinical MCP tool called: {name}")

    if name == "search_clinical_guidelines":
        return await search_clinical_guidelines(
            query=arguments["query"],
            top_k=arguments.get("top_k", 5),
            use_rerank=arguments.get("use_rerank", True),
        )

    elif name == "search_by_disease":
        disease = arguments["disease_name"]
        aspect = arguments.get("aspect", "")
        query = f"{disease} {aspect} 가이드라인 치료지침".strip()
        return await search_clinical_guidelines(query=query, top_k=5)

    elif name == "search_treatment_protocol":
        condition = arguments["condition"]
        context = arguments.get("patient_context", "")
        query = f"{condition} 치료 프로토콜 {context}".strip()
        return await search_clinical_guidelines(query=query, top_k=5)

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def search_clinical_guidelines(
    query: str,
    top_k: int = 5,
    use_rerank: bool = True,
) -> list[TextContent]:
    """임상 가이드라인 검색 실행"""
    try:
        hybrid_search = get_hybrid_search()

        # Hybrid Search 수행
        results = hybrid_search.search(query=query, top_k=20)

        # Reranking
        if use_rerank and results:
            reranker = get_reranker()
            results = reranker.rerank(query=query, documents=results, top_k=top_k)
        else:
            results = results[:top_k]

        # 결과 포맷팅
        if not results:
            return [TextContent(
                type="text",
                text=f"'{query}'에 대한 검색 결과가 없습니다."
            )]

        formatted_results = []
        for i, doc in enumerate(results, 1):
            score = doc.get("rerank_score", doc.get("hybrid_score", doc.get("score", 0)))
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "Unknown")

            formatted_results.append(
                f"[{i}] (출처: {source}, 관련도: {score:.3f})\n{content}"
            )

        response_text = f"## '{query}' 검색 결과 ({len(results)}건)\n\n"
        response_text += "\n\n---\n\n".join(formatted_results)

        return [TextContent(type="text", text=response_text)]

    except Exception as e:
        logger.error(f"Search error: {e}")
        return [TextContent(type="text", text=f"검색 중 오류 발생: {str(e)}")]


async def main():
    """MCP 서버 실행"""
    logger.info("Starting Clinical MCP Server...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
