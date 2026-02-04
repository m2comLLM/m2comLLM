#!/usr/bin/env python3
"""
데이터 인덱싱 스크립트
Excel 파일을 Milvus와 Elasticsearch에 인덱싱
"""

import sys
import argparse
from pathlib import Path
from loguru import logger

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders.excel_loader import ExcelLoader
from src.embeddings.embedding_model import get_embedding_model
from src.vectorstore.milvus_client import MilvusClient
from src.vectorstore.elasticsearch_client import ElasticsearchClient


def main():
    parser = argparse.ArgumentParser(description="데이터 인덱싱")
    parser.add_argument("path", help="Excel 파일 또는 디렉토리 경로")
    parser.add_argument("--content-cols", nargs="+", help="본문 컬럼들")
    parser.add_argument("--metadata-cols", nargs="+", help="메타데이터 컬럼들")
    parser.add_argument("--chunk-size", type=int, default=512, help="청크 크기")
    parser.add_argument("--drop-existing", action="store_true", help="기존 데이터 삭제")
    parser.add_argument("--skip-milvus", action="store_true", help="Milvus 인덱싱 스킵")
    parser.add_argument("--skip-es", action="store_true", help="Elasticsearch 인덱싱 스킵")

    args = parser.parse_args()

    # 로거 설정
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    path = Path(args.path)

    # 1. 데이터 로드
    logger.info("=" * 50)
    logger.info("1단계: 데이터 로드")
    logger.info("=" * 50)

    loader = ExcelLoader(
        content_columns=args.content_cols,
        metadata_columns=args.metadata_cols,
        chunk_size=args.chunk_size,
    )

    if path.is_file():
        documents = loader.load(path)
    elif path.is_dir():
        documents = loader.load_directory(path)
    else:
        logger.error(f"경로를 찾을 수 없습니다: {path}")
        sys.exit(1)

    if not documents:
        logger.warning("로드된 문서가 없습니다")
        sys.exit(0)

    logger.info(f"총 {len(documents)}개 문서 로드됨")

    # 2. 임베딩 생성
    logger.info("=" * 50)
    logger.info("2단계: 임베딩 생성")
    logger.info("=" * 50)

    if not args.skip_milvus:
        embedding_model = get_embedding_model()
        contents = [doc.content for doc in documents]
        embeddings = embedding_model.embed_documents(contents)
        logger.info(f"임베딩 완료: shape={embeddings.shape}")
    else:
        embeddings = None

    # 3. Milvus 인덱싱
    if not args.skip_milvus:
        logger.info("=" * 50)
        logger.info("3단계: Milvus 인덱싱")
        logger.info("=" * 50)

        milvus_client = MilvusClient()
        milvus_client.create_collection(drop_existing=args.drop_existing)
        milvus_client.insert(documents, embeddings)
        logger.info(f"Milvus 문서 수: {milvus_client.count()}")

    # 4. Elasticsearch 인덱싱
    if not args.skip_es:
        logger.info("=" * 50)
        logger.info("4단계: Elasticsearch 인덱싱")
        logger.info("=" * 50)

        es_client = ElasticsearchClient()
        es_client.create_index(drop_existing=args.drop_existing)
        es_client.insert(documents)
        logger.info(f"Elasticsearch 문서 수: {es_client.count()}")

    logger.info("=" * 50)
    logger.info("인덱싱 완료!")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
