"""
Latency Monitor Script
LLM 추론 성능 모니터링 및 벤치마킹
"""

import os
import json
import time
import argparse
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime
import statistics
import httpx


@dataclass
class LatencyMetrics:
    """지연시간 메트릭"""
    ttft_ms: float = 0.0          # Time To First Token
    total_time_ms: float = 0.0     # 전체 응답 시간
    tokens_generated: int = 0       # 생성된 토큰 수
    tps: float = 0.0               # Tokens Per Second
    prompt_tokens: int = 0          # 입력 토큰 수
    timestamp: str = ""


@dataclass
class BenchmarkResult:
    """벤치마크 결과"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    duration_seconds: float

    # 지연시간 통계
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    total_time_p50_ms: float
    total_time_p95_ms: float

    # 처리량
    avg_tps: float
    throughput_rps: float  # Requests Per Second

    # 메타정보
    concurrency: int
    model_name: str
    started_at: str
    ended_at: str


class LatencyMonitor:
    """LLM 지연시간 모니터"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.metrics_history: List[LatencyMetrics] = []

    async def measure_single(
        self,
        prompt: str,
        max_tokens: int = 256,
        stream: bool = True,
    ) -> LatencyMetrics:
        """단일 요청 지연시간 측정"""
        metrics = LatencyMetrics(timestamp=datetime.now().isoformat())

        start_time = time.perf_counter()
        first_token_time = None
        tokens_generated = 0

        async with httpx.AsyncClient(timeout=120.0) as client:
            if stream:
                # 스트리밍 모드
                async with client.stream(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "stream": True,
                    },
                ) as response:
                    async for chunk in response.aiter_lines():
                        if chunk.startswith("data: "):
                            if first_token_time is None:
                                first_token_time = time.perf_counter()

                            data = chunk[6:]
                            if data.strip() == "[DONE]":
                                break

                            try:
                                parsed = json.loads(data)
                                delta = parsed.get("choices", [{}])[0].get("delta", {})
                                if delta.get("content"):
                                    tokens_generated += 1
                            except json.JSONDecodeError:
                                pass
            else:
                # 비스트리밍 모드
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "stream": False,
                    },
                )
                first_token_time = time.perf_counter()
                result = response.json()
                usage = result.get("usage", {})
                tokens_generated = usage.get("completion_tokens", 0)
                metrics.prompt_tokens = usage.get("prompt_tokens", 0)

        end_time = time.perf_counter()

        # 메트릭 계산
        metrics.ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
        metrics.total_time_ms = (end_time - start_time) * 1000
        metrics.tokens_generated = tokens_generated

        if tokens_generated > 0 and metrics.total_time_ms > 0:
            metrics.tps = tokens_generated / (metrics.total_time_ms / 1000)

        self.metrics_history.append(metrics)
        return metrics

    async def run_load_test(
        self,
        prompts: List[str],
        concurrency: int = 10,
        max_tokens: int = 256,
    ) -> BenchmarkResult:
        """부하 테스트 실행"""
        start_time = datetime.now()
        start_perf = time.perf_counter()

        # 세마포어로 동시성 제어
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_request(prompt: str) -> Optional[LatencyMetrics]:
            async with semaphore:
                try:
                    return await self.measure_single(prompt, max_tokens, stream=False)
                except Exception as e:
                    print(f"요청 실패: {e}")
                    return None

        # 모든 요청 실행
        tasks = [bounded_request(p) for p in prompts]
        results = await asyncio.gather(*tasks)

        end_time = datetime.now()
        duration = time.perf_counter() - start_perf

        # 성공한 결과만 필터링
        successful = [r for r in results if r is not None]
        failed = len(results) - len(successful)

        if not successful:
            raise RuntimeError("모든 요청이 실패했습니다.")

        # 통계 계산
        ttft_values = [m.ttft_ms for m in successful]
        total_time_values = [m.total_time_ms for m in successful]
        tps_values = [m.tps for m in successful if m.tps > 0]

        def percentile(data: List[float], p: float) -> float:
            if not data:
                return 0.0
            sorted_data = sorted(data)
            k = (len(sorted_data) - 1) * p / 100
            f = int(k)
            c = f + 1 if f + 1 < len(sorted_data) else f
            return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)

        return BenchmarkResult(
            total_requests=len(results),
            successful_requests=len(successful),
            failed_requests=failed,
            duration_seconds=duration,
            ttft_p50_ms=percentile(ttft_values, 50),
            ttft_p95_ms=percentile(ttft_values, 95),
            ttft_p99_ms=percentile(ttft_values, 99),
            total_time_p50_ms=percentile(total_time_values, 50),
            total_time_p95_ms=percentile(total_time_values, 95),
            avg_tps=statistics.mean(tps_values) if tps_values else 0,
            throughput_rps=len(successful) / duration if duration > 0 else 0,
            concurrency=concurrency,
            model_name=self.model_name,
            started_at=start_time.isoformat(),
            ended_at=end_time.isoformat(),
        )

    async def watch(self, interval: int = 5):
        """실시간 모니터링"""
        test_prompt = "안녕하세요. 간단한 인사를 해주세요."

        print("=== 실시간 모니터링 시작 ===")
        print("(Ctrl+C로 종료)\n")

        try:
            while True:
                metrics = await self.measure_single(test_prompt, max_tokens=50)

                print(f"[{metrics.timestamp}]")
                print(f"  TTFT: {metrics.ttft_ms:.0f}ms")
                print(f"  총 시간: {metrics.total_time_ms:.0f}ms")
                print(f"  TPS: {metrics.tps:.1f} tokens/sec")
                print()

                await asyncio.sleep(interval)

        except KeyboardInterrupt:
            print("\n모니터링 종료")


def print_benchmark_result(result: BenchmarkResult):
    """벤치마크 결과 출력"""
    print("=" * 50)
    print("           벤치마크 결과")
    print("=" * 50)
    print()
    print(f"모델: {result.model_name}")
    print(f"동시성: {result.concurrency}")
    print(f"소요 시간: {result.duration_seconds:.1f}초")
    print()
    print("## 요청 통계")
    print(f"  - 총 요청: {result.total_requests}")
    print(f"  - 성공: {result.successful_requests}")
    print(f"  - 실패: {result.failed_requests}")
    print()
    print("## 지연시간 (TTFT)")
    print(f"  - P50: {result.ttft_p50_ms:.0f}ms")
    print(f"  - P95: {result.ttft_p95_ms:.0f}ms")
    print(f"  - P99: {result.ttft_p99_ms:.0f}ms")
    print()
    print("## 총 응답 시간")
    print(f"  - P50: {result.total_time_p50_ms:.0f}ms")
    print(f"  - P95: {result.total_time_p95_ms:.0f}ms")
    print()
    print("## 처리량")
    print(f"  - 평균 TPS: {result.avg_tps:.1f} tokens/sec")
    print(f"  - 처리량: {result.throughput_rps:.2f} req/sec")
    print()


async def main_async(args):
    monitor = LatencyMonitor(
        base_url=args.base_url,
        model_name=args.model_name,
    )

    if args.command == 'single':
        metrics = await monitor.measure_single(args.prompt, args.max_tokens)
        print("=== 단일 요청 측정 결과 ===\n")
        print(f"TTFT: {metrics.ttft_ms:.0f}ms")
        print(f"총 시간: {metrics.total_time_ms:.0f}ms")
        print(f"생성 토큰: {metrics.tokens_generated}")
        print(f"TPS: {metrics.tps:.1f} tokens/sec")

    elif args.command == 'load':
        # 테스트 프롬프트 생성
        prompts = [f"테스트 프롬프트 #{i+1}: 간단한 답변을 해주세요." for i in range(args.requests)]

        print(f"부하 테스트 시작: {args.requests}개 요청, 동시성 {args.concurrency}")
        result = await monitor.run_load_test(prompts, args.concurrency, args.max_tokens)
        print_benchmark_result(result)

    elif args.command == 'watch':
        await monitor.watch(args.interval)

    elif args.command == 'benchmark':
        # 단계별 부하 테스트
        prompts = [f"벤치마크 테스트 #{i+1}" for i in range(100)]
        results = []

        for concurrency in [1, 5, 10, 20]:
            print(f"\n동시성 {concurrency} 테스트 중...")
            result = await monitor.run_load_test(prompts[:50], concurrency, 128)
            results.append(asdict(result))
            print(f"  처리량: {result.throughput_rps:.2f} req/sec")

        # 결과 저장
        output_path = Path(args.output)
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"\n벤치마크 결과 저장: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Latency Monitor')
    parser.add_argument('--base-url', default='http://localhost:8000')
    parser.add_argument('--model-name', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-32B')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # single
    single_parser = subparsers.add_parser('single', help='단일 요청 측정')
    single_parser.add_argument('--prompt', default='안녕하세요')
    single_parser.add_argument('--max-tokens', type=int, default=256)

    # load
    load_parser = subparsers.add_parser('load', help='부하 테스트')
    load_parser.add_argument('--concurrency', type=int, default=10)
    load_parser.add_argument('--requests', type=int, default=100)
    load_parser.add_argument('--max-tokens', type=int, default=256)

    # watch
    watch_parser = subparsers.add_parser('watch', help='실시간 모니터링')
    watch_parser.add_argument('--interval', type=int, default=5)

    # benchmark
    benchmark_parser = subparsers.add_parser('benchmark', help='벤치마크')
    benchmark_parser.add_argument('--output', default='benchmark_report.json')

    args = parser.parse_args()

    if args.command:
        asyncio.run(main_async(args))
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
