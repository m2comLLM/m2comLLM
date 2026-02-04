"""
환경변수 기반 설정 관리 모듈
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """애플리케이션 설정"""

    # Milvus 설정
    milvus_host: str = Field(default="localhost", description="Milvus 서버 호스트")
    milvus_port: int = Field(default=19530, description="Milvus 서버 포트")
    milvus_collection: str = Field(default="clinical_documents", description="Milvus 컬렉션 이름")

    # Elasticsearch 설정
    es_host: str = Field(default="localhost", description="Elasticsearch 호스트")
    es_port: int = Field(default=9200, description="Elasticsearch 포트")
    es_index: str = Field(default="clinical_documents", description="Elasticsearch 인덱스 이름")
    es_user: Optional[str] = Field(default=None, description="Elasticsearch 사용자명")
    es_password: Optional[str] = Field(default=None, description="Elasticsearch 비밀번호")

    # vLLM 설정
    vllm_base_url: str = Field(default="http://localhost:8000", description="vLLM 서버 URL")
    vllm_model: str = Field(default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", description="사용할 모델")

    # 임베딩 모델 설정
    embedding_model: str = Field(default="BAAI/bge-m3", description="임베딩 모델 경로")
    embedding_dimension: int = Field(default=1024, description="임베딩 차원")

    # Reranker 설정
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3", description="Reranker 모델 경로")
    rerank_top_k: int = Field(default=5, description="Reranking 후 반환할 문서 수")

    # 검색 설정
    search_top_k: int = Field(default=20, description="초기 검색 결과 수")
    hybrid_alpha: float = Field(default=0.5, description="Hybrid Search 가중치 (0=BM25, 1=Dense)")

    # 데이터베이스 설정 (Patient MCP용)
    db_host: str = Field(default="localhost", description="EMR DB 호스트")
    db_port: int = Field(default=5432, description="EMR DB 포트")
    db_name: str = Field(default="emr", description="EMR DB 이름")
    db_user: Optional[str] = Field(default=None, description="EMR DB 사용자명")
    db_password: Optional[str] = Field(default=None, description="EMR DB 비밀번호")

    # 로깅 설정
    log_level: str = Field(default="INFO", description="로그 레벨")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @property
    def milvus_uri(self) -> str:
        """Milvus 연결 URI"""
        return f"http://{self.milvus_host}:{self.milvus_port}"

    @property
    def es_url(self) -> str:
        """Elasticsearch 연결 URL"""
        return f"http://{self.es_host}:{self.es_port}"


# 싱글톤 설정 인스턴스
settings = Settings()
