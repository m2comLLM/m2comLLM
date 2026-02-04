#!/usr/bin/env python3
"""
MCP 서버 테스트 스크립트
"""

import sys
import asyncio
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


async def test_clinical_mcp():
    """Clinical MCP 테스트"""
    print("\n" + "=" * 60)
    print("Clinical MCP 테스트")
    print("=" * 60)

    from src.mcp_servers.clinical_mcp import list_tools, call_tool

    # 도구 목록 조회
    tools = await list_tools()
    print(f"\n사용 가능한 도구 ({len(tools)}개):")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:50]}...")

    # 테스트 쿼리 (DB 연결 없이 구조만 테스트)
    print("\n[테스트] search_clinical_guidelines")
    print("  → DB 미연결 상태에서는 실제 검색이 수행되지 않습니다.")


async def test_patient_mcp():
    """Patient MCP 테스트"""
    print("\n" + "=" * 60)
    print("Patient MCP 테스트")
    print("=" * 60)

    from src.mcp_servers.patient_mcp import list_tools, call_tool

    # 도구 목록 조회
    tools = await list_tools()
    print(f"\n사용 가능한 도구 ({len(tools)}개):")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:50]}...")

    # Mock 데이터 테스트
    print("\n[테스트] get_patient_info")
    result = await call_tool("get_patient_info", {"patient_id": "P12345"})
    print(result[0].text[:300] + "...")

    print("\n[테스트] get_patient_medications")
    result = await call_tool("get_patient_medications", {"patient_id": "P12345"})
    print(result[0].text)


async def test_drug_mcp():
    """Drug MCP 테스트"""
    print("\n" + "=" * 60)
    print("Drug MCP 테스트")
    print("=" * 60)

    from src.mcp_servers.drug_mcp import list_tools, call_tool

    # 도구 목록 조회
    tools = await list_tools()
    print(f"\n사용 가능한 도구 ({len(tools)}개):")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:50]}...")

    # 약물 정보 테스트
    print("\n[테스트] get_drug_info - 메트포르민")
    result = await call_tool("get_drug_info", {"drug_name": "metformin"})
    print(result[0].text[:500] + "...")

    # 상호작용 테스트
    print("\n[테스트] check_drug_interaction - 아스피린 + 와파린")
    result = await call_tool("check_drug_interaction", {
        "drug1": "aspirin",
        "drug2": "warfarin"
    })
    print(result[0].text)

    # 다중 상호작용 테스트
    print("\n[테스트] check_multiple_interactions")
    result = await call_tool("check_multiple_interactions", {
        "drugs": ["metformin", "amlodipine", "aspirin", "warfarin"]
    })
    print(result[0].text)


async def main():
    """메인 테스트 함수"""
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    print("=" * 60)
    print("MCP 서버 테스트")
    print("=" * 60)

    try:
        await test_patient_mcp()
        await test_drug_mcp()
        await test_clinical_mcp()

        print("\n" + "=" * 60)
        print("✅ 모든 MCP 서버 테스트 완료")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
