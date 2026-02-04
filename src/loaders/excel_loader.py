"""
Excel 파일을 Document 객체로 변환하는 로더
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class Document:
    """문서 객체"""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None

    def __post_init__(self):
        if self.doc_id is None:
            import hashlib
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()[:12]


class ExcelLoader:
    """Excel 파일 로더"""

    def __init__(
        self,
        content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
        sheet_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: int = 100,
    ):
        """
        Args:
            content_columns: 본문으로 사용할 컬럼들 (None이면 모든 컬럼 결합)
            metadata_columns: 메타데이터로 저장할 컬럼들
            sheet_name: 읽을 시트 이름 (None이면 첫 번째 시트)
            chunk_size: 청킹할 크기 (None이면 청킹 안 함)
            chunk_overlap: 청크 간 중복 문자 수
        """
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns or []
        self.sheet_name = sheet_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load(self, file_path: str | Path) -> List[Document]:
        """
        Excel 파일을 로드하여 Document 리스트로 변환

        Args:
            file_path: Excel 파일 경로

        Returns:
            Document 리스트
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        logger.info(f"Excel 파일 로드 중: {file_path}")

        # Excel 읽기
        df = pd.read_excel(
            file_path,
            sheet_name=self.sheet_name or 0,
            engine="openpyxl"
        )

        logger.info(f"총 {len(df)}개 행 로드됨")

        documents = []
        for idx, row in df.iterrows():
            doc = self._row_to_document(row, file_path.name, idx)
            if doc:
                if self.chunk_size:
                    documents.extend(self._chunk_document(doc))
                else:
                    documents.append(doc)

        logger.info(f"총 {len(documents)}개 문서 생성됨")
        return documents

    def load_directory(self, dir_path: str | Path) -> List[Document]:
        """
        디렉토리 내 모든 Excel 파일 로드

        Args:
            dir_path: 디렉토리 경로

        Returns:
            Document 리스트
        """
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"디렉토리가 아닙니다: {dir_path}")

        documents = []
        excel_files = list(dir_path.glob("*.xlsx")) + list(dir_path.glob("*.xls"))

        logger.info(f"{len(excel_files)}개 Excel 파일 발견")

        for file_path in excel_files:
            try:
                docs = self.load(file_path)
                documents.extend(docs)
            except Exception as e:
                logger.error(f"파일 로드 실패 {file_path}: {e}")

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
                    content_parts.append(str(row[col]))
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
        for col in self.metadata_columns:
            if col in row and pd.notna(row[col]):
                metadata[col] = row[col]

        return Document(content=content, metadata=metadata)

    def _chunk_document(self, doc: Document) -> List[Document]:
        """문서를 청크로 분할"""
        content = doc.content
        if len(content) <= self.chunk_size:
            return [doc]

        chunks = []
        start = 0
        chunk_idx = 0

        while start < len(content):
            end = start + self.chunk_size

            # 문장 경계에서 자르기 시도
            if end < len(content):
                for sep in ["\n", ". ", ", ", " "]:
                    last_sep = content[start:end].rfind(sep)
                    if last_sep > self.chunk_size // 2:
                        end = start + last_sep + len(sep)
                        break

            chunk_content = content[start:end].strip()
            if chunk_content:
                chunk_metadata = doc.metadata.copy()
                chunk_metadata["chunk_index"] = chunk_idx
                chunks.append(Document(
                    content=chunk_content,
                    metadata=chunk_metadata,
                    doc_id=f"{doc.doc_id}_{chunk_idx}"
                ))
                chunk_idx += 1

            start = end - self.chunk_overlap

        return chunks
