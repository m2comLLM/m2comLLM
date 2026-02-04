"""
CSV 파일을 Document 객체로 변환하는 로더
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional
from loguru import logger

from src.loaders.excel_loader import Document


class CSVLoader:
    """CSV 파일 로더"""

    def __init__(
        self,
        content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
        encoding: str = "utf-8",
    ):
        """
        Args:
            content_columns: 본문으로 사용할 컬럼들 (None이면 모든 컬럼 결합)
            metadata_columns: 메타데이터로 저장할 컬럼들
            encoding: 파일 인코딩
        """
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns or []
        self.encoding = encoding

    def load(self, file_path: str | Path) -> List[Document]:
        """
        CSV 파일을 로드하여 Document 리스트로 변환

        Args:
            file_path: CSV 파일 경로

        Returns:
            Document 리스트
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        logger.info(f"CSV 파일 로드 중: {file_path}")

        # CSV 읽기
        df = pd.read_csv(file_path, encoding=self.encoding)
        logger.info(f"총 {len(df)}개 행 로드됨, 컬럼: {list(df.columns)}")

        documents = []
        for idx, row in df.iterrows():
            doc = self._row_to_document(row, file_path.name, idx)
            if doc:
                documents.append(doc)

        logger.info(f"총 {len(documents)}개 문서 생성됨")
        return documents

    def _row_to_document(
        self, row: pd.Series, source: str, row_idx: int
    ) -> Optional[Document]:
        """행을 Document로 변환"""
        # 본문 생성
        if self.content_columns:
            content_parts = []
            for col in self.content_columns:
                if col in row and pd.notna(row[col]):
                    content_parts.append(f"{col}: {row[col]}")
            content = "\n".join(content_parts)
        else:
            # 모든 컬럼 결합
            content_parts = []
            for col, val in row.items():
                if pd.notna(val):
                    content_parts.append(f"{col}: {val}")
            content = "\n".join(content_parts)

        if not content.strip():
            return None

        # 메타데이터 생성
        metadata = {
            "source": source,
            "row_index": row_idx,
        }

        # 모든 컬럼을 메타데이터에 추가
        for col, val in row.items():
            if pd.notna(val):
                metadata[col] = val

        return Document(content=content, metadata=metadata)
