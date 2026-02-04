"""
Milvus 벡터 데이터베이스 클라이언트
"""

from typing import List, Dict, Any, Optional
import numpy as np
from loguru import logger

from src.config import settings
from src.loaders.excel_loader import Document


class MilvusClient:
    """Milvus 벡터 DB 클라이언트"""

    def __init__(
        self,
        collection_name: str = None,
        dimension: int = None,
        uri: str = None,
    ):
        """
        Args:
            collection_name: 컬렉션 이름
            dimension: 벡터 차원
            uri: Milvus 서버 URI
        """
        self.collection_name = collection_name or settings.milvus_collection
        self.dimension = dimension or settings.embedding_dimension
        self.uri = uri or settings.milvus_uri
        self._client = None
        self._collection = None

    @property
    def client(self):
        """Lazy loading으로 클라이언트 연결"""
        if self._client is None:
            self._connect()
        return self._client

    def _connect(self):
        """Milvus 연결"""
        from pymilvus import connections, Collection, utility

        logger.info(f"Milvus 연결 중: {self.uri}")
        connections.connect(
            alias="default",
            uri=self.uri,
        )
        self._client = connections
        logger.info("Milvus 연결 완료")

    def create_collection(self, drop_existing: bool = False):
        """
        컬렉션 생성

        Args:
            drop_existing: 기존 컬렉션 삭제 여부
        """
        from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility

        # 연결 확인
        _ = self.client

        if utility.has_collection(self.collection_name):
            if drop_existing:
                logger.warning(f"기존 컬렉션 삭제: {self.collection_name}")
                utility.drop_collection(self.collection_name)
            else:
                logger.info(f"기존 컬렉션 사용: {self.collection_name}")
                self._collection = Collection(self.collection_name)
                self._collection.load()
                return

        # 스키마 정의
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]
        schema = CollectionSchema(fields=fields, description="Clinical documents")

        # 컬렉션 생성
        logger.info(f"컬렉션 생성: {self.collection_name}")
        self._collection = Collection(
            name=self.collection_name,
            schema=schema,
        )

        # 인덱스 생성 (IVF_FLAT)
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        }
        self._collection.create_index(
            field_name="embedding",
            index_params=index_params,
        )
        self._collection.load()
        logger.info("컬렉션 생성 및 인덱스 완료")

    @property
    def collection(self):
        """컬렉션 반환"""
        if self._collection is None:
            from pymilvus import Collection
            _ = self.client
            self._collection = Collection(self.collection_name)
            self._collection.load()
        return self._collection

    def insert(
        self,
        documents: List[Document],
        embeddings: np.ndarray,
        batch_size: int = 100,
    ) -> int:
        """
        문서와 임베딩 삽입

        Args:
            documents: Document 리스트
            embeddings: 임베딩 벡터 배열
            batch_size: 배치 크기

        Returns:
            삽입된 문서 수
        """
        if len(documents) != len(embeddings):
            raise ValueError("문서 수와 임베딩 수가 일치하지 않습니다")

        total_inserted = 0

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]

            data = [
                [doc.doc_id for doc in batch_docs],  # id
                [doc.content[:65000] for doc in batch_docs],  # content (max length)
                batch_embeddings.tolist(),  # embedding
                [doc.metadata for doc in batch_docs],  # metadata
            ]

            self.collection.insert(data)
            total_inserted += len(batch_docs)
            logger.debug(f"삽입 진행: {total_inserted}/{len(documents)}")

        self.collection.flush()
        logger.info(f"총 {total_inserted}개 문서 삽입 완료")
        return total_inserted

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = None,
        filter_expr: str = None,
    ) -> List[Dict[str, Any]]:
        """
        벡터 유사도 검색

        Args:
            query_embedding: 쿼리 임베딩 벡터
            top_k: 반환할 결과 수
            filter_expr: 필터 표현식

        Returns:
            검색 결과 리스트
        """
        top_k = top_k or settings.search_top_k

        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 16},
        }

        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=["content", "metadata"],
        )

        documents = []
        for hits in results:
            for hit in hits:
                documents.append({
                    "id": hit.id,
                    "content": hit.entity.get("content"),
                    "metadata": hit.entity.get("metadata"),
                    "score": hit.score,
                })

        return documents

    def count(self) -> int:
        """컬렉션 문서 수 반환"""
        return self.collection.num_entities

    def delete_collection(self):
        """컬렉션 삭제"""
        from pymilvus import utility
        _ = self.client
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            logger.info(f"컬렉션 삭제됨: {self.collection_name}")
            self._collection = None
