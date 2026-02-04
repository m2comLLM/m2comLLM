---
name: llm-judge
description: LLM을 심판으로 사용하여 모델 답변 품질을 평가합니다.
user-invocable: true
---

# LLM Judge 스킬

외부 LLM(Claude, GPT-4 등)을 심판으로 활용하여 내부 모델의 답변 품질을 자동 평가합니다.

## 기능

1. **답변 품질 채점**: 1-10점 척도로 답변 평가
2. **다중 기준 평가**: 정확성, 완전성, 관련성, 안전성
3. **비교 평가**: A/B 테스트 방식 모델 비교
4. **벤치마크 실행**: 표준 벤치마크 데이터셋 평가

## 사용법

```bash
# 단일 답변 평가
python scripts/eval_judge.py single \
    --question "질문" \
    --answer "모델 답변" \
    --reference "정답"

# 배치 평가
python scripts/eval_judge.py batch \
    --input eval_samples.jsonl \
    --output eval_results.jsonl

# A/B 비교
python scripts/eval_judge.py compare \
    --question "질문" \
    --answer-a "모델A 답변" \
    --answer-b "모델B 답변"
```

## 평가 기준

| 기준 | 설명 | 가중치 |
|------|------|-------|
| 정확성 (Accuracy) | 사실적 정확도 | 30% |
| 완전성 (Completeness) | 답변의 완성도 | 25% |
| 관련성 (Relevance) | 질문과의 관련성 | 25% |
| 안전성 (Safety) | 유해성/편향 여부 | 20% |

## 의존성

- `openai`: OpenAI API (GPT-4 사용 시)
- `anthropic`: Anthropic API (Claude 사용 시)
- `httpx`: HTTP 클라이언트
