"""
Hybrid Search: BM25 + Dense Vector 결합 검색
RRF (Reciprocal Rank Fusion) 알고리즘 사용
"""

from typing import List, Dict, Any, Optional
import numpy as np
from loguru import logger

from src.config import settings
from src.embeddings.embedding_model import get_embedding_model
from src.vectorstore.milvus_client import MilvusClient
from src.vectorstore.elasticsearch_client import ElasticsearchClient


class HybridSearch:
    """Hybrid Search (BM25 + Dense Vector)"""

    def __init__(
        self,
        milvus_client: Optional[MilvusClient] = None,
        es_client: Optional[ElasticsearchClient] = None,
        alpha: float = None,
    ):
        """
        Args:
            milvus_client: Milvus 클라이언트
            es_client: Elasticsearch 클라이언트
            alpha: Dense 검색 가중치 (0=BM25 only, 1=Dense only)
        """
        self.milvus_client = milvus_client or MilvusClient()
        self.es_client = es_client or ElasticsearchClient()
        self.alpha = alpha if alpha is not None else settings.hybrid_alpha
        self.embedding_model = get_embedding_model()

    def search(
        self,
        query: str,
        top_k: int = None,
        alpha: float = None,
        use_rrf: bool = True,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        하이브리드 검색 수행

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            alpha: Dense 검색 가중치 (None이면 기본값 사용)
            use_rrf: RRF 알고리즘 사용 여부
            rrf_k: RRF 파라미터 (기본값 60)

        Returns:
            검색 결과 리스트
        """
        top_k = top_k or settings.rerank_top_k
        alpha = alpha if alpha is not None else self.alpha
        search_top_k = settings.search_top_k  # 초기 검색은 더 많이

        logger.debug(f"Hybrid search: query='{query[:50]}...', alpha={alpha}")

        # 1. Dense Vector 검색 (Milvus)
        query_embedding = self.embedding_model.embed_query(query)
        dense_results = self.milvus_client.search(
            query_embedding=query_embedding,
            top_k=search_top_k,
        )

        # 2. BM25 키워드 검색 (Elasticsearch)
        sparse_results = self.es_client.search(
            query=query,
            top_k=search_top_k,
        )

        # 3. 결과 융합
        if use_rrf:
            fused_results = self._rrf_fusion(
                dense_results, sparse_results, k=rrf_k
            )
        else:
            fused_results = self._weighted_fusion(
                dense_results, sparse_results, alpha=alpha
            )

        # 4. 상위 결과 반환
        return fused_results[:top_k]

    def _rrf_fusion(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        RRF (Reciprocal Rank Fusion) 알고리즘

        RRF(d) = Σ 1 / (k + rank(d))

        Args:
            dense_results: Dense 검색 결과
            sparse_results: Sparse (BM25) 검색 결과
            k: RRF 파라미터

        Returns:
            융합된 검색 결과
        """
        doc_scores = {}
        doc_data = {}

        # Dense 결과 처리
        for rank, doc in enumerate(dense_results, 1):
            doc_id = doc["id"]
            rrf_score = 1.0 / (k + rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            doc_data[doc_id] = doc

        # Sparse 결과 처리
        for rank, doc in enumerate(sparse_results, 1):
            doc_id = doc["id"]
            rrf_score = 1.0 / (k + rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            if doc_id not in doc_data:
                doc_data[doc_id] = doc

        # 점수 기준 정렬
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in sorted_docs:
            result = doc_data[doc_id].copy()
            result["hybrid_score"] = score
            results.append(result)

        return results

    def _weighted_fusion(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
        alpha: float,
    ) -> List[Dict[str, Any]]:
        """
        가중치 기반 점수 융합

        score = alpha * dense_score + (1 - alpha) * sparse_score

        Args:
            dense_results: Dense 검색 결과
            sparse_results: Sparse 검색 결과
            alpha: Dense 가중치

        Returns:
            융합된 검색 결과
        """
        # 점수 정규화 (min-max)
        def normalize_scores(results: List[Dict]) -> Dict[str, float]:
            if not results:
                return {}
            scores = [r["score"] for r in results]
            min_s, max_s = min(scores), max(scores)
            range_s = max_s - min_s if max_s != min_s else 1.0
            return {
                r["id"]: (r["score"] - min_s) / range_s
                for r in results
            }

        dense_scores = normalize_scores(dense_results)
        sparse_scores = normalize_scores(sparse_results)

        # 모든 문서 수집
        all_doc_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
        doc_data = {}

        for doc in dense_results + sparse_results:
            if doc["id"] not in doc_data:
                doc_data[doc["id"]] = doc

        # 가중 점수 계산
        results = []
        for doc_id in all_doc_ids:
            d_score = dense_scores.get(doc_id, 0)
            s_score = sparse_scores.get(doc_id, 0)
            hybrid_score = alpha * d_score + (1 - alpha) * s_score

            result = doc_data[doc_id].copy()
            result["hybrid_score"] = hybrid_score
            result["dense_score"] = d_score
            result["sparse_score"] = s_score
            results.append(result)

        # 점수 기준 정렬
        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results

    def search_dense_only(
        self, query: str, top_k: int = None
    ) -> List[Dict[str, Any]]:
        """Dense 검색만 수행"""
        top_k = top_k or settings.search_top_k
        query_embedding = self.embedding_model.embed_query(query)
        return self.milvus_client.search(query_embedding, top_k)

    def search_sparse_only(
        self, query: str, top_k: int = None
    ) -> List[Dict[str, Any]]:
        """BM25 검색만 수행"""
        top_k = top_k or settings.search_top_k
        return self.es_client.search(query, top_k)
