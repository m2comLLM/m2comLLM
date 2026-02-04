#!/usr/bin/env python3
"""
MCP 서버 테스트 스크립트
"""

import sys
import asyncio
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_clinical_mcp():
    """Clinical MCP 테스트"""
    print("=" * 60)
    print("Clinical MCP 테스트")
    print("=" * 60)

    from src.mcp_servers.clinical_mcp import search_clinical_guidelines

    # 테스트 쿼리
    queries = [
        "당뇨병 치료 가이드라인",
        "고혈압 약물 치료",
    ]

    for query in queries:
        print(f"\n쿼리: {query}")
        try:
            results = await search_clinical_guidelines(query, top_k=3, use_rerank=False)
            for r in results:
                print(f"  - {r.text[:100]}...")
        except Exception as e:
            print(f"  오류: {e}")


async def test_patient_mcp():
    """Patient MCP 테스트"""
    print("\n" + "=" * 60)
    print("Patient MCP 테스트")
    print("=" * 60)

    from src.mcp_servers.patient_mcp import (
        get_patient_info,
        get_patient_vitals,
        get_patient_diagnoses,
        get_patient_medications,
        get_patient_allergies,
    )

    patient_id = "P12345"

    tests = [
        ("환자 정보", get_patient_info(patient_id)),
        ("바이탈", get_patient_vitals(patient_id, 5)),
        ("진단 이력", get_patient_diagnoses(patient_id, False)),
        ("복용 약물", get_patient_medications(patient_id, False)),
        ("알레르기", get_patient_allergies(patient_id)),
    ]

    for name, coro in tests:
        print(f"\n{name}:")
        try:
            results = await coro
            for r in results:
                print(f"{r.text}")
        except Exception as e:
            print(f"  오류: {e}")


async def test_drug_mcp():
    """Drug MCP 테스트"""
    print("\n" + "=" * 60)
    print("Drug MCP 테스트")
    print("=" * 60)

    from src.mcp_servers.drug_mcp import (
        get_drug_info,
        check_drug_interaction,
        check_multiple_interactions,
        check_contraindications,
    )

    # 약물 정보 조회
    print("\n약물 정보 (메트포르민):")
    results = await get_drug_info("metformin")
    for r in results:
        print(r.text)

    # 약물 상호작용 확인
    print("\n약물 상호작용 (아스피린 + 와파린):")
    results = await check_drug_interaction("aspirin", "warfarin")
    for r in results:
        print(r.text)

    # 다중 상호작용 확인
    print("\n다중 약물 상호작용:")
    results = await check_multiple_interactions(["metformin", "amlodipine", "aspirin"])
    for r in results:
        print(r.text)

    # 금기 확인
    print("\n금기 확인 (메트포르민 + 신부전):")
    results = await check_contraindications("metformin", ["신부전", "탈수"])
    for r in results:
        print(r.text)


async def test_agent():
    """에이전트 테스트 (Mock)"""
    print("\n" + "=" * 60)
    print("LangGraph 에이전트 테스트")
    print("=" * 60)

    from src.agent.langgraph_agent import MedicalAgent

    agent = MedicalAgent()

    # Mock 도구 테스트
    print("\n환자 정보 조회:")
    result = await agent._get_patient_info("P12345")
    print(result)

    print("\n복용 약물 조회:")
    result = await agent._get_patient_medications("P12345")
    print(result)

    print("\n알레르기 정보:")
    result = await agent._get_patient_allergies("P12345")
    print(result)

    print("\n✅ 에이전트 구조 테스트 완료")
    print("(실제 LLM 호출은 vLLM 서버 필요)")


async def main():
    """메인 테스트"""
    print("Medical RAG Agent - MCP 테스트")
    print("=" * 60)

    # Drug MCP 테스트 (DB 불필요)
    await test_drug_mcp()

    # Patient MCP 테스트 (Mock 데이터)
    await test_patient_mcp()

    # 에이전트 테스트
    await test_agent()

    # Clinical MCP는 Milvus/ES 필요
    # await test_clinical_mcp()

    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
