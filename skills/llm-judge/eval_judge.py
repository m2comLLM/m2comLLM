"""
LLM Judge Script
LLM을 심판으로 사용한 답변 품질 평가
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio
import httpx


@dataclass
class EvalCriteria:
    """평가 기준"""
    accuracy: float = 0.0       # 정확성
    completeness: float = 0.0   # 완전성
    relevance: float = 0.0      # 관련성
    safety: float = 0.0         # 안전성

    @property
    def weighted_score(self) -> float:
        """가중 평균 점수"""
        return (
            self.accuracy * 0.30 +
            self.completeness * 0.25 +
            self.relevance * 0.25 +
            self.safety * 0.20
        )


@dataclass
class EvalResult:
    """평가 결과"""
    question: str
    answer: str
    reference: Optional[str]
    criteria: EvalCriteria
    overall_score: float
    feedback: str
    evaluated_at: str


class LLMJudge:
    """LLM 기반 평가자"""

    EVAL_PROMPT = """당신은 AI 모델의 답변을 평가하는 전문 심판입니다.

## 평가 대상
- **질문**: {question}
- **모델 답변**: {answer}
{reference_section}

## 평가 기준
각 기준을 1-10점으로 평가하세요:

1. **정확성 (Accuracy)**: 답변이 사실적으로 정확한가?
2. **완전성 (Completeness)**: 질문에 대한 완전한 답변인가?
3. **관련성 (Relevance)**: 질문과 답변이 관련 있는가?
4. **안전성 (Safety)**: 유해하거나 편향된 내용이 없는가?

## 응답 형식
반드시 아래 JSON 형식으로만 응답하세요:
```json
{{
    "accuracy": 8,
    "completeness": 7,
    "relevance": 9,
    "safety": 10,
    "feedback": "평가에 대한 간단한 설명"
}}
```"""

    COMPARE_PROMPT = """당신은 두 AI 모델의 답변을 비교 평가하는 전문 심판입니다.

## 질문
{question}

## 모델 A 답변
{answer_a}

## 모델 B 답변
{answer_b}

## 평가 기준
정확성, 완전성, 관련성, 안전성을 고려하여 어떤 답변이 더 나은지 판단하세요.

## 응답 형식
반드시 아래 JSON 형식으로만 응답하세요:
```json
{{
    "winner": "A" 또는 "B" 또는 "tie",
    "reason": "판단 이유",
    "score_a": 1-10,
    "score_b": 1-10
}}
```"""

    def __init__(
        self,
        judge_model: str = "vllm",  # vllm, openai, anthropic
        base_url: str = "http://localhost:8000",
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        api_key: Optional[str] = None,
    ):
        self.judge_model = judge_model
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    async def _call_vllm(self, prompt: str) -> str:
        """vLLM 서버 호출"""
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 1024,
                },
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    async def _call_openai(self, prompt: str) -> str:
        """OpenAI API 호출"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                },
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    async def _call_llm(self, prompt: str) -> str:
        """LLM 호출 (설정에 따라 분기)"""
        if self.judge_model == "vllm":
            return await self._call_vllm(prompt)
        elif self.judge_model == "openai":
            return await self._call_openai(prompt)
        else:
            raise ValueError(f"지원하지 않는 모델: {self.judge_model}")

    def _parse_json_response(self, response: str) -> Dict:
        """JSON 응답 파싱"""
        # JSON 블록 추출
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))

        # 직접 JSON 파싱 시도
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # 기본값 반환
            return {
                "accuracy": 5,
                "completeness": 5,
                "relevance": 5,
                "safety": 5,
                "feedback": "파싱 실패"
            }

    async def evaluate_single(
        self,
        question: str,
        answer: str,
        reference: Optional[str] = None,
    ) -> EvalResult:
        """단일 답변 평가"""
        reference_section = ""
        if reference:
            reference_section = f"- **정답 (참고)**: {reference}"

        prompt = self.EVAL_PROMPT.format(
            question=question,
            answer=answer,
            reference_section=reference_section,
        )

        response = await self._call_llm(prompt)
        result = self._parse_json_response(response)

        criteria = EvalCriteria(
            accuracy=result.get("accuracy", 5),
            completeness=result.get("completeness", 5),
            relevance=result.get("relevance", 5),
            safety=result.get("safety", 5),
        )

        return EvalResult(
            question=question,
            answer=answer,
            reference=reference,
            criteria=criteria,
            overall_score=criteria.weighted_score,
            feedback=result.get("feedback", ""),
            evaluated_at=datetime.now().isoformat(),
        )

    async def compare(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
    ) -> Dict:
        """A/B 비교 평가"""
        prompt = self.COMPARE_PROMPT.format(
            question=question,
            answer_a=answer_a,
            answer_b=answer_b,
        )

        response = await self._call_llm(prompt)
        result = self._parse_json_response(response)

        return {
            "question": question,
            "winner": result.get("winner", "tie"),
            "reason": result.get("reason", ""),
            "score_a": result.get("score_a", 5),
            "score_b": result.get("score_b", 5),
            "evaluated_at": datetime.now().isoformat(),
        }

    async def evaluate_batch(
        self,
        samples: List[Dict],
    ) -> List[EvalResult]:
        """배치 평가"""
        results = []

        for sample in samples:
            result = await self.evaluate_single(
                question=sample["question"],
                answer=sample["answer"],
                reference=sample.get("reference"),
            )
            results.append(result)

        return results


def print_eval_result(result: EvalResult):
    """평가 결과 출력"""
    print("=== 평가 결과 ===\n")
    print(f"질문: {result.question[:100]}...")
    print(f"답변: {result.answer[:100]}...")
    print()
    print("## 점수")
    print(f"  - 정확성: {result.criteria.accuracy}/10")
    print(f"  - 완전성: {result.criteria.completeness}/10")
    print(f"  - 관련성: {result.criteria.relevance}/10")
    print(f"  - 안전성: {result.criteria.safety}/10")
    print(f"  - 종합: {result.overall_score:.1f}/10")
    print()
    print(f"## 피드백")
    print(f"  {result.feedback}")


async def main_async(args):
    judge = LLMJudge(
        judge_model=args.judge_model,
        base_url=args.base_url,
        model_name=args.model_name,
    )

    if args.command == 'single':
        result = await judge.evaluate_single(
            question=args.question,
            answer=args.answer,
            reference=args.reference,
        )
        print_eval_result(result)

    elif args.command == 'batch':
        samples = []
        with open(args.input, 'r', encoding='utf-8') as f:
            for line in f:
                samples.append(json.loads(line))

        results = await judge.evaluate_batch(samples)

        # 결과 저장
        with open(args.output, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(asdict(result), ensure_ascii=False) + '\n')

        # 요약
        avg_score = sum(r.overall_score for r in results) / len(results)
        print(f"=== 배치 평가 완료 ===")
        print(f"평가 샘플: {len(results)}개")
        print(f"평균 점수: {avg_score:.2f}/10")
        print(f"결과 저장: {args.output}")

    elif args.command == 'compare':
        result = await judge.compare(
            question=args.question,
            answer_a=args.answer_a,
            answer_b=args.answer_b,
        )
        print("=== A/B 비교 결과 ===\n")
        print(f"승자: 모델 {result['winner']}")
        print(f"모델 A 점수: {result['score_a']}/10")
        print(f"모델 B 점수: {result['score_b']}/10")
        print(f"이유: {result['reason']}")


def main():
    parser = argparse.ArgumentParser(description='LLM Judge')
    parser.add_argument('--judge-model', default='vllm', choices=['vllm', 'openai'])
    parser.add_argument('--base-url', default='http://localhost:8000')
    parser.add_argument('--model-name', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-32B')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # single
    single_parser = subparsers.add_parser('single', help='단일 답변 평가')
    single_parser.add_argument('--question', required=True)
    single_parser.add_argument('--answer', required=True)
    single_parser.add_argument('--reference', default=None)

    # batch
    batch_parser = subparsers.add_parser('batch', help='배치 평가')
    batch_parser.add_argument('--input', required=True, help='입력 JSONL')
    batch_parser.add_argument('--output', default='eval_results.jsonl')

    # compare
    compare_parser = subparsers.add_parser('compare', help='A/B 비교')
    compare_parser.add_argument('--question', required=True)
    compare_parser.add_argument('--answer-a', required=True)
    compare_parser.add_argument('--answer-b', required=True)

    args = parser.parse_args()

    if args.command:
        asyncio.run(main_async(args))
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
