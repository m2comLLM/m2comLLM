"""
Evaluator Agent
모델 답변 품질 검증 및 벤치마킹을 담당하는 에이전트
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class EvalConfig:
    """평가 설정"""
    model_url: str = "http://localhost:8000"
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    judge_model: str = "vllm"  # vllm, openai
    eval_samples: int = 100
    concurrency: int = 5


@dataclass
class EvalSummary:
    """평가 요약"""
    total_samples: int
    avg_accuracy: float
    avg_completeness: float
    avg_relevance: float
    avg_safety: float
    overall_score: float

    # 지연시간
    avg_ttft_ms: float
    avg_total_time_ms: float
    avg_tps: float

    # 메타정보
    model_name: str
    evaluated_at: str


class EvaluatorAgent:
    """
    Evaluator Agent

    역할:
    - LLM-as-Judge를 통한 답변 품질 평가
    - 지연시간/처리량 벤치마킹
    - 모델 비교 (A/B 테스트)
    - 평가 보고서 생성

    사용 스킬:
    - llm-judge: LLM 기반 품질 평가
    - latency-monitor: 성능 벤치마킹
    """

    SYSTEM_PROMPT = """당신은 LLM 평가 전문가입니다.

## 역할
- 모델 답변의 품질 평가
- 성능 벤치마킹 및 분석
- 개선점 도출

## 평가 기준
1. 정확성 (Accuracy): 사실적으로 정확한가?
2. 완전성 (Completeness): 완전한 답변인가?
3. 관련성 (Relevance): 질문과 관련 있는가?
4. 안전성 (Safety): 유해 콘텐츠가 없는가?

## 벤치마크 메트릭
- TTFT: Time To First Token
- TPS: Tokens Per Second
- Throughput: Requests Per Second
"""

    def __init__(self, config: Optional[EvalConfig] = None):
        self.config = config or EvalConfig()
        self.eval_history: List[Dict] = []

    async def evaluate_model(
        self,
        test_samples: List[Dict],
    ) -> EvalSummary:
        """모델 평가 실행"""
        from skills.llm_judge.eval_judge import LLMJudge

        judge = LLMJudge(
            judge_model=self.config.judge_model,
            base_url=self.config.model_url,
            model_name=self.config.model_name,
        )

        results = await judge.evaluate_batch(test_samples)

        # 통계 계산
        if not results:
            raise ValueError("평가 결과가 없습니다.")

        avg_accuracy = sum(r.criteria.accuracy for r in results) / len(results)
        avg_completeness = sum(r.criteria.completeness for r in results) / len(results)
        avg_relevance = sum(r.criteria.relevance for r in results) / len(results)
        avg_safety = sum(r.criteria.safety for r in results) / len(results)
        overall = sum(r.overall_score for r in results) / len(results)

        return EvalSummary(
            total_samples=len(results),
            avg_accuracy=avg_accuracy,
            avg_completeness=avg_completeness,
            avg_relevance=avg_relevance,
            avg_safety=avg_safety,
            overall_score=overall,
            avg_ttft_ms=0,  # 별도 벤치마크에서 측정
            avg_total_time_ms=0,
            avg_tps=0,
            model_name=self.config.model_name,
            evaluated_at=datetime.now().isoformat(),
        )

    async def run_latency_benchmark(
        self,
        num_requests: int = 50,
        concurrency: int = 5,
    ) -> Dict[str, Any]:
        """지연시간 벤치마크"""
        from skills.latency_monitor.latency_monitor import LatencyMonitor

        monitor = LatencyMonitor(
            base_url=self.config.model_url,
            model_name=self.config.model_name,
        )

        prompts = [f"테스트 #{i+1}: 간단히 답변해주세요." for i in range(num_requests)]

        result = await monitor.run_load_test(prompts, concurrency, max_tokens=128)

        return {
            "total_requests": result.total_requests,
            "successful_requests": result.successful_requests,
            "duration_seconds": result.duration_seconds,
            "ttft_p50_ms": result.ttft_p50_ms,
            "ttft_p95_ms": result.ttft_p95_ms,
            "total_time_p50_ms": result.total_time_p50_ms,
            "avg_tps": result.avg_tps,
            "throughput_rps": result.throughput_rps,
        }

    async def compare_models(
        self,
        model_a_url: str,
        model_b_url: str,
        test_prompts: List[str],
    ) -> Dict[str, Any]:
        """모델 A/B 비교"""
        from skills.llm_judge.eval_judge import LLMJudge
        import httpx

        results = {
            "model_a_wins": 0,
            "model_b_wins": 0,
            "ties": 0,
            "comparisons": [],
        }

        judge = LLMJudge(
            judge_model=self.config.judge_model,
            base_url=self.config.model_url,
            model_name=self.config.model_name,
        )

        async with httpx.AsyncClient(timeout=60.0) as client:
            for prompt in test_prompts:
                # 모델 A 응답
                resp_a = await client.post(
                    f"{model_a_url}/v1/chat/completions",
                    json={
                        "model": "model",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 256,
                    },
                )
                answer_a = resp_a.json()["choices"][0]["message"]["content"]

                # 모델 B 응답
                resp_b = await client.post(
                    f"{model_b_url}/v1/chat/completions",
                    json={
                        "model": "model",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 256,
                    },
                )
                answer_b = resp_b.json()["choices"][0]["message"]["content"]

                # 비교 평가
                comparison = await judge.compare(prompt, answer_a, answer_b)

                if comparison["winner"] == "A":
                    results["model_a_wins"] += 1
                elif comparison["winner"] == "B":
                    results["model_b_wins"] += 1
                else:
                    results["ties"] += 1

                results["comparisons"].append(comparison)

        return results

    async def generate_report(
        self,
        quality_results: Optional[EvalSummary] = None,
        latency_results: Optional[Dict] = None,
    ) -> str:
        """종합 평가 보고서 생성"""
        report = f"""# LLM 평가 보고서

생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
모델: {self.config.model_name}

---

"""
        if quality_results:
            report += f"""## 품질 평가

| 기준 | 점수 (1-10) |
|------|------------|
| 정확성 (Accuracy) | {quality_results.avg_accuracy:.1f} |
| 완전성 (Completeness) | {quality_results.avg_completeness:.1f} |
| 관련성 (Relevance) | {quality_results.avg_relevance:.1f} |
| 안전성 (Safety) | {quality_results.avg_safety:.1f} |
| **종합** | **{quality_results.overall_score:.1f}** |

평가 샘플: {quality_results.total_samples}개

"""

        if latency_results:
            report += f"""## 성능 벤치마크

| 메트릭 | 값 |
|--------|-----|
| TTFT (P50) | {latency_results.get('ttft_p50_ms', 0):.0f}ms |
| TTFT (P95) | {latency_results.get('ttft_p95_ms', 0):.0f}ms |
| 응답시간 (P50) | {latency_results.get('total_time_p50_ms', 0):.0f}ms |
| TPS | {latency_results.get('avg_tps', 0):.1f} tokens/sec |
| 처리량 | {latency_results.get('throughput_rps', 0):.2f} req/sec |

테스트 요청: {latency_results.get('total_requests', 0)}개
성공: {latency_results.get('successful_requests', 0)}개

"""

        report += """---

## 권장사항

- 품질 점수 7.0 이상: 프로덕션 배포 가능
- TTFT P95 500ms 이하: 사용자 경험 양호
- TPS 30 이상: 대화형 서비스에 적합
"""

        return report

    async def handle_task(self, task: str) -> str:
        """에이전트 태스크 처리"""
        task_lower = task.lower()

        if "품질" in task_lower or "평가" in task_lower:
            # 테스트 샘플
            test_samples = [
                {"question": "당뇨병의 주요 증상은?", "answer": ""},
                {"question": "고혈압 관리 방법은?", "answer": ""},
            ]
            # 실제 평가는 test_samples에 모델 응답 필요
            return "품질 평가를 위해 테스트 샘플이 필요합니다. evaluate_model() 메서드를 직접 호출해주세요."

        elif "벤치마크" in task_lower or "성능" in task_lower or "지연" in task_lower:
            try:
                results = await self.run_latency_benchmark(num_requests=20, concurrency=3)
                return f"""성능 벤치마크 결과:
- TTFT (P50): {results['ttft_p50_ms']:.0f}ms
- TTFT (P95): {results['ttft_p95_ms']:.0f}ms
- 평균 TPS: {results['avg_tps']:.1f} tokens/sec
- 처리량: {results['throughput_rps']:.2f} req/sec
- 성공률: {results['successful_requests']}/{results['total_requests']}"""
            except Exception as e:
                return f"벤치마크 실패: {str(e)}\n(vLLM 서버가 실행 중인지 확인하세요)"

        elif "보고서" in task_lower or "report" in task_lower:
            report = await self.generate_report()
            return report

        else:
            return "지원하지 않는 태스크입니다. 사용 가능: 품질 평가, 성능 벤치마크, 보고서 생성"


async def main():
    """테스트 실행"""
    agent = EvaluatorAgent(
        config=EvalConfig(
            model_url="http://localhost:8000",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        )
    )

    # 보고서 생성
    report = await agent.generate_report()
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
