"""
BGE-M3 임베딩 모델 래퍼
"""

from typing import List, Union
import numpy as np
from loguru import logger

from src.config import settings


class EmbeddingModel:
    """BGE-M3 임베딩 모델"""

    def __init__(
        self,
        model_name: str = None,
        device: str = "cuda",
        normalize: bool = True,
    ):
        """
        Args:
            model_name: 모델 이름 또는 경로
            device: 실행 디바이스 ('cuda' 또는 'cpu')
            normalize: 임베딩 정규화 여부
        """
        self.model_name = model_name or settings.embedding_model
        self.device = device
        self.normalize = normalize
        self._model = None

    @property
    def model(self):
        """Lazy loading으로 모델 로드"""
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        """모델 로드"""
        logger.info(f"임베딩 모델 로드 중: {self.model_name}")
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True,
            )
            logger.info(f"임베딩 모델 로드 완료 (차원: {self.dimension})")
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            raise

    @property
    def dimension(self) -> int:
        """임베딩 차원"""
        return settings.embedding_dimension

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        텍스트를 임베딩 벡터로 변환

        Args:
            texts: 단일 텍스트 또는 텍스트 리스트

        Returns:
            임베딩 벡터 (shape: [n, dimension])
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 100,
        )

        return np.array(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """
        쿼리 텍스트 임베딩 (검색용)

        Args:
            query: 검색 쿼리

        Returns:
            임베딩 벡터 (shape: [dimension])
        """
        return self.embed(query)[0]

    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """
        문서 리스트 임베딩 (인덱싱용)

        Args:
            documents: 문서 텍스트 리스트

        Returns:
            임베딩 벡터 (shape: [n, dimension])
        """
        return self.embed(documents)


# 싱글톤 인스턴스
_embedding_model = None


def get_embedding_model() -> EmbeddingModel:
    """임베딩 모델 싱글톤 인스턴스 반환"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model
