---
name: latency-monitor
description: LLM 추론 지연시간과 처리량을 모니터링하고 벤치마킹합니다.
user-invocable: true
---

# Latency Monitor 스킬

LLM 서버의 응답 시간, 처리량(throughput), 토큰 생성 속도를 측정합니다.

## 기능

1. **지연시간 측정**: TTFT, TPS, 전체 응답 시간
2. **처리량 벤치마크**: 동시 요청 처리 능력 측정
3. **실시간 모니터링**: 대시보드 형태로 성능 추적
4. **보고서 생성**: 성능 분석 리포트 자동 생성

## 주요 메트릭

| 메트릭 | 설명 |
|--------|------|
| TTFT | Time To First Token (첫 토큰까지 시간) |
| TPS | Tokens Per Second (초당 토큰 생성 수) |
| P50/P95/P99 | 지연시간 백분위수 |
| Throughput | 초당 처리 요청 수 |

## 사용법

```bash
# 단일 요청 테스트
python scripts/latency_monitor.py single --prompt "테스트 프롬프트"

# 부하 테스트 (10개 동시 요청)
python scripts/latency_monitor.py load --concurrency 10 --requests 100

# 실시간 모니터링
python scripts/latency_monitor.py watch --interval 5

# 벤치마크 보고서
python scripts/latency_monitor.py benchmark --output report.json
```

## 출력 예시

```
=== 성능 요약 ===
- TTFT (P50): 120ms
- TTFT (P95): 280ms
- TPS: 45.2 tokens/sec
- 총 처리량: 12.5 req/sec
```

## 의존성

- `httpx`: 비동기 HTTP 클라이언트
- `rich`: 터미널 UI
- `pandas`: 데이터 분석
