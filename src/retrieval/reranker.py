"""
Cross-Encoder Reranker
검색 결과를 재순위화하여 정확도 향상
"""

from typing import List, Dict, Any, Union, Tuple
from loguru import logger

from src.config import settings


class Reranker:
    """Cross-Encoder 기반 Reranker"""

    def __init__(
        self,
        model_name: str = None,
        device: str = "cuda",
    ):
        """
        Args:
            model_name: Reranker 모델 이름
            device: 실행 디바이스
        """
        self.model_name = model_name or settings.reranker_model
        self.device = device
        self._model = None

    @property
    def model(self):
        """Lazy loading으로 모델 로드"""
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        """모델 로드"""
        logger.info(f"Reranker 모델 로드 중: {self.model_name}")
        try:
            from FlagEmbedding import FlagReranker
            self._model = FlagReranker(
                self.model_name,
                device=self.device,
                use_fp16=True,
            )
            logger.info("Reranker 모델 로드 완료")
        except Exception as e:
            logger.error(f"Reranker 모델 로드 실패: {e}")
            raise

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = None,
        content_key: str = "content",
    ) -> List[Dict[str, Any]]:
        """
        문서 재순위화

        Args:
            query: 검색 쿼리
            documents: 재순위화할 문서 리스트
            top_k: 반환할 상위 문서 수
            content_key: 문서 내용 키

        Returns:
            재순위화된 문서 리스트
        """
        if not documents:
            return []

        top_k = top_k or settings.rerank_top_k

        # 쿼리-문서 쌍 생성
        pairs = [
            [query, doc[content_key]]
            for doc in documents
            if doc.get(content_key)
        ]

        if not pairs:
            logger.warning("재순위화할 유효한 문서가 없습니다")
            return documents[:top_k]

        # Cross-Encoder 점수 계산
        scores = self.model.compute_score(pairs, normalize=True)

        # 리스트가 아닌 경우 리스트로 변환
        if not isinstance(scores, list):
            scores = [scores]

        # 점수와 문서 매핑
        scored_docs = []
        for doc, score in zip(documents, scores):
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = float(score)
            scored_docs.append(doc_copy)

        # 점수 기준 정렬
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        logger.debug(
            f"Reranked {len(documents)} docs, "
            f"top score: {scored_docs[0]['rerank_score']:.4f}"
        )

        return scored_docs[:top_k]

    def compute_scores(
        self,
        query: str,
        texts: List[str],
    ) -> List[float]:
        """
        쿼리-텍스트 쌍의 관련성 점수 계산

        Args:
            query: 검색 쿼리
            texts: 텍스트 리스트

        Returns:
            점수 리스트
        """
        pairs = [[query, text] for text in texts]
        scores = self.model.compute_score(pairs, normalize=True)

        if not isinstance(scores, list):
            scores = [scores]

        return [float(s) for s in scores]


class HybridSearchWithRerank:
    """Hybrid Search + Reranking 통합 클래스"""

    def __init__(
        self,
        hybrid_search=None,
        reranker=None,
    ):
        """
        Args:
            hybrid_search: HybridSearch 인스턴스
            reranker: Reranker 인스턴스
        """
        from src.retrieval.hybrid_search import HybridSearch

        self.hybrid_search = hybrid_search or HybridSearch()
        self.reranker = reranker or Reranker()

    def search(
        self,
        query: str,
        top_k: int = None,
        search_top_k: int = None,
        use_rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid Search + Reranking

        Args:
            query: 검색 쿼리
            top_k: 최종 반환할 결과 수
            search_top_k: 초기 검색 결과 수
            use_rerank: Reranking 사용 여부

        Returns:
            검색 결과 리스트
        """
        top_k = top_k or settings.rerank_top_k
        search_top_k = search_top_k or settings.search_top_k

        # 1. Hybrid Search
        results = self.hybrid_search.search(
            query=query,
            top_k=search_top_k,
        )

        # 2. Reranking (선택적)
        if use_rerank and results:
            results = self.reranker.rerank(
                query=query,
                documents=results,
                top_k=top_k,
            )
        else:
            results = results[:top_k]

        return results


# 싱글톤 인스턴스
_reranker = None


def get_reranker() -> Reranker:
    """Reranker 싱글톤 인스턴스 반환"""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
