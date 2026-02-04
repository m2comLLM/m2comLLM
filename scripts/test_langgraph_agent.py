"""
LangGraph 에이전트 + vLLM 연동 테스트
"""

import sys
sys.path.insert(0, "/workspace/dev")

import asyncio
from src.agent.langgraph_agent import MedicalAgent
from src.config import settings


async def test_agent():
    """에이전트 테스트"""
    print("=" * 60)
    print("LangGraph 의료 에이전트 테스트")
    print("=" * 60)
    print(f"모델: {settings.vllm_model}")
    print(f"서버: {settings.vllm_base_url}")
    print()

    # 에이전트 생성
    agent = MedicalAgent(max_iterations=3)
    print("✅ 에이전트 생성 완료\n")

    # 테스트 쿼리
    queries = [
        "천식 환자의 급성 발작 시 응급 처치 방법을 알려주세요.",
        "환자 P12345의 현재 복용 약물을 알려주세요.",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"테스트 {i}: {query}")
        print("-" * 60)

        try:
            response = await agent.run(query, patient_id="P12345")
            print(f"\n💡 응답:")
            print(response)
        except Exception as e:
            print(f"❌ 오류: {e}")

    print("\n" + "=" * 60)
    print("✅ 에이전트 테스트 완료")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_agent())
