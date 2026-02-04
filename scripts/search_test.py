#!/usr/bin/env python3
"""
검색 테스트 스크립트
Hybrid Search + Reranking 테스트
"""

import sys
import argparse
from pathlib import Path
from loguru import logger

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="검색 테스트")
    parser.add_argument("query", help="검색 쿼리")
    parser.add_argument("--top-k", type=int, default=5, help="반환할 결과 수")
    parser.add_argument("--search-top-k", type=int, default=20, help="초기 검색 결과 수")
    parser.add_argument("--no-rerank", action="store_true", help="Reranking 비활성화")
    parser.add_argument("--dense-only", action="store_true", help="Dense 검색만")
    parser.add_argument("--sparse-only", action="store_true", help="BM25 검색만")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dense 가중치")
    parser.add_argument("-v", "--verbose", action="store_true", help="상세 출력")

    args = parser.parse_args()

    # 로거 설정
    logger.remove()
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.add(sys.stderr, level=log_level)

    from src.retrieval.hybrid_search import HybridSearch
    from src.retrieval.reranker import Reranker

    logger.info(f"검색 쿼리: {args.query}")
    logger.info("=" * 60)

    hybrid_search = HybridSearch()

    # 검색 모드 선택
    if args.dense_only:
        logger.info("Dense 검색 모드")
        results = hybrid_search.search_dense_only(args.query, args.search_top_k)
    elif args.sparse_only:
        logger.info("BM25 검색 모드")
        results = hybrid_search.search_sparse_only(args.query, args.search_top_k)
    else:
        logger.info(f"Hybrid 검색 모드 (alpha={args.alpha})")
        results = hybrid_search.search(
            query=args.query,
            top_k=args.search_top_k,
            alpha=args.alpha,
        )

    # Reranking
    if not args.no_rerank and results:
        logger.info("Reranking 수행 중...")
        reranker = Reranker()
        results = reranker.rerank(
            query=args.query,
            documents=results,
            top_k=args.top_k,
        )
    else:
        results = results[:args.top_k]

    # 결과 출력
    logger.info("=" * 60)
    logger.info(f"검색 결과 ({len(results)}건)")
    logger.info("=" * 60)

    for i, doc in enumerate(results, 1):
        print(f"\n[{i}] ID: {doc.get('id', 'N/A')}")

        # 점수 출력
        scores = []
        if "rerank_score" in doc:
            scores.append(f"rerank={doc['rerank_score']:.4f}")
        if "hybrid_score" in doc:
            scores.append(f"hybrid={doc['hybrid_score']:.4f}")
        if "score" in doc:
            scores.append(f"score={doc['score']:.4f}")
        if scores:
            print(f"    Score: {', '.join(scores)}")

        # 메타데이터 출력
        if doc.get("metadata"):
            meta_str = ", ".join(
                f"{k}={v}" for k, v in doc["metadata"].items()
                if k not in ["source", "row_index"]
            )
            if meta_str:
                print(f"    Meta: {meta_str}")

        # 내용 출력 (200자 제한)
        content = doc.get("content", "")
        if len(content) > 200:
            content = content[:200] + "..."
        print(f"    Content: {content}")

    print()


if __name__ == "__main__":
    main()
