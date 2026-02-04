"""
Elasticsearch BM25 검색 클라이언트
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from src.config import settings
from src.loaders.excel_loader import Document


class ElasticsearchClient:
    """Elasticsearch BM25 검색 클라이언트"""

    def __init__(
        self,
        index_name: str = None,
        url: str = None,
    ):
        """
        Args:
            index_name: 인덱스 이름
            url: Elasticsearch URL
        """
        self.index_name = index_name or settings.es_index
        self.url = url or settings.es_url
        self._client = None

    @property
    def client(self):
        """Lazy loading으로 클라이언트 연결"""
        if self._client is None:
            self._connect()
        return self._client

    def _connect(self):
        """Elasticsearch 연결"""
        from elasticsearch import Elasticsearch

        logger.info(f"Elasticsearch 연결 중: {self.url}")

        auth = None
        if settings.es_user and settings.es_password:
            auth = (settings.es_user, settings.es_password)

        self._client = Elasticsearch(
            self.url,
            basic_auth=auth,
            verify_certs=False,
        )

        if not self._client.ping():
            raise ConnectionError("Elasticsearch 연결 실패")

        logger.info("Elasticsearch 연결 완료")

    def create_index(self, drop_existing: bool = False):
        """
        인덱스 생성

        Args:
            drop_existing: 기존 인덱스 삭제 여부
        """
        if self.client.indices.exists(index=self.index_name):
            if drop_existing:
                logger.warning(f"기존 인덱스 삭제: {self.index_name}")
                self.client.indices.delete(index=self.index_name)
            else:
                logger.info(f"기존 인덱스 사용: {self.index_name}")
                return

        # 인덱스 설정 (한국어 + 영어 분석기)
        index_settings = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "korean_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "content": {
                        "type": "text",
                        "analyzer": "korean_analyzer",
                    },
                    "metadata": {"type": "object", "enabled": True},
                }
            }
        }

        logger.info(f"인덱스 생성: {self.index_name}")
        self.client.indices.create(index=self.index_name, body=index_settings)
        logger.info("인덱스 생성 완료")

    def insert(
        self,
        documents: List[Document],
        batch_size: int = 100,
    ) -> int:
        """
        문서 삽입

        Args:
            documents: Document 리스트
            batch_size: 배치 크기

        Returns:
            삽입된 문서 수
        """
        from elasticsearch.helpers import bulk

        def generate_actions():
            for doc in documents:
                yield {
                    "_index": self.index_name,
                    "_id": doc.doc_id,
                    "_source": {
                        "id": doc.doc_id,
                        "content": doc.content,
                        "metadata": doc.metadata,
                    }
                }

        success, errors = bulk(
            self.client,
            generate_actions(),
            chunk_size=batch_size,
            raise_on_error=False,
        )

        if errors:
            logger.warning(f"삽입 오류 {len(errors)}건")

        logger.info(f"총 {success}개 문서 삽입 완료")
        return success

    def search(
        self,
        query: str,
        top_k: int = None,
        filter_query: Dict = None,
    ) -> List[Dict[str, Any]]:
        """
        BM25 키워드 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            filter_query: 필터 쿼리

        Returns:
            검색 결과 리스트
        """
        top_k = top_k or settings.search_top_k

        search_query = {
            "query": {
                "bool": {
                    "must": {
                        "match": {
                            "content": {
                                "query": query,
                                "operator": "or",
                            }
                        }
                    }
                }
            },
            "size": top_k,
        }

        if filter_query:
            search_query["query"]["bool"]["filter"] = filter_query

        response = self.client.search(index=self.index_name, body=search_query)

        documents = []
        for hit in response["hits"]["hits"]:
            documents.append({
                "id": hit["_id"],
                "content": hit["_source"]["content"],
                "metadata": hit["_source"].get("metadata", {}),
                "score": hit["_score"],
            })

        return documents

    def count(self) -> int:
        """인덱스 문서 수 반환"""
        response = self.client.count(index=self.index_name)
        return response["count"]

    def delete_index(self):
        """인덱스 삭제"""
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
            logger.info(f"인덱스 삭제됨: {self.index_name}")
