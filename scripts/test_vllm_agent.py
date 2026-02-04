"""
vLLM 연동 테스트 스크립트
실제 GPU에서 DeepSeek-R1-Distill 모델 사용
"""

import sys
sys.path.insert(0, "/workspace/dev")

import asyncio
import httpx
from src.config import settings


async def test_vllm_direct():
    """vLLM 직접 호출 테스트"""
    print("=" * 60)
    print("vLLM 직접 호출 테스트")
    print("=" * 60)
    print(f"모델: {settings.vllm_model}")
    print(f"서버: {settings.vllm_base_url}")
    print()

    async with httpx.AsyncClient(timeout=120.0) as client:
        # 의료 질문 테스트
        questions = [
            "천식 환자의 급성 발작 시 응급 처치 방법을 알려주세요.",
            "COPD 환자에게 권장되는 흡입기 종류는?",
            "폐렴 환자의 입원 기준은 무엇인가요?",
        ]

        for i, question in enumerate(questions, 1):
            print(f"\n{'='*60}")
            print(f"질문 {i}: {question}")
            print("-" * 60)

            response = await client.post(
                f"{settings.vllm_base_url}/v1/chat/completions",
                json={
                    "model": settings.vllm_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "당신은 호흡기내과 전문의입니다. 정확하고 전문적인 의료 정보를 제공하세요. 한국어로 답변하세요."
                        },
                        {"role": "user", "content": question}
                    ],
                    "max_tokens": 1024,
                    "temperature": 0.7,
                },
            )

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]

                # <think> 태그 처리 (DeepSeek-R1 특성)
                if "</think>" in content:
                    # 추론 과정과 최종 답변 분리
                    parts = content.split("</think>")
                    thinking = parts[0].replace("<think>", "").strip() if "<think>" in parts[0] else ""
                    answer = parts[1].strip() if len(parts) > 1 else content

                    print(f"\n📝 추론 과정 (요약):")
                    # 추론 과정 첫 200자만 표시
                    if thinking:
                        print(f"   {thinking[:200]}..." if len(thinking) > 200 else f"   {thinking}")

                    print(f"\n💡 답변:")
                    print(f"   {answer}")
                else:
                    print(f"\n💡 답변:")
                    print(f"   {content}")

                # 토큰 사용량
                usage = result.get("usage", {})
                print(f"\n📊 토큰: 입력 {usage.get('prompt_tokens', 0)}, 출력 {usage.get('completion_tokens', 0)}")
            else:
                print(f"❌ 오류: {response.status_code}")
                print(response.text)


async def test_medical_context():
    """의료 컨텍스트 포함 테스트"""
    print("\n" + "=" * 60)
    print("의료 컨텍스트 포함 테스트")
    print("=" * 60)

    # 환자 정보와 함께 질문
    patient_context = """
환자 정보:
- 나이: 65세 남성
- 진단: COPD Stage III, 고혈압
- 현재 약물: Tiotropium 18mcg 1일 1회, Amlodipine 5mg 1일 1회
- 최근 증상: 호흡곤란 악화, 객담 증가 (황색)
"""

    question = "이 환자에게 추가로 필요한 치료는 무엇인가요?"

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{settings.vllm_base_url}/v1/chat/completions",
            json={
                "model": settings.vllm_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "당신은 호흡기내과 전문의입니다. 환자 정보를 바탕으로 치료 계획을 제안하세요. 한국어로 답변하세요."
                    },
                    {"role": "user", "content": f"{patient_context}\n\n질문: {question}"}
                ],
                "max_tokens": 1024,
                "temperature": 0.7,
            },
        )

        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # <think> 태그 처리
            if "</think>" in content:
                parts = content.split("</think>")
                answer = parts[1].strip() if len(parts) > 1 else content
            else:
                answer = content

            print(f"\n환자 정보:")
            print(patient_context)
            print(f"\n질문: {question}")
            print(f"\n💡 의사 답변:")
            print(answer)
        else:
            print(f"❌ 오류: {response.status_code}")


async def main():
    """메인 테스트"""
    # vLLM 서버 상태 확인
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{settings.vllm_base_url}/v1/models")
            if response.status_code != 200:
                print("❌ vLLM 서버가 준비되지 않았습니다.")
                return
        except Exception as e:
            print(f"❌ vLLM 서버 연결 실패: {e}")
            return

    print("✅ vLLM 서버 연결 성공!\n")

    await test_vllm_direct()
    await test_medical_context()

    print("\n" + "=" * 60)
    print("✅ 모든 테스트 완료")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
