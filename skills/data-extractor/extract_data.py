"""
Data Extractor Script
사내 문서를 LLM 학습용 JSONL로 변환
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse


@dataclass
class TrainingSample:
    """학습 데이터 샘플"""
    instruction: str
    input: str
    output: str
    source: str
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class PIIMasker:
    """개인정보 마스킹"""

    PATTERNS = {
        'email': (r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL]'),
        'phone': (r'\d{2,3}-\d{3,4}-\d{4}', '[PHONE]'),
        'rrn': (r'\d{6}-[1-4]\d{6}', '[RRN]'),  # 주민등록번호
        'card': (r'\d{4}-\d{4}-\d{4}-\d{4}', '[CARD]'),
    }

    @classmethod
    def mask(cls, text: str) -> str:
        """모든 PII 패턴 마스킹"""
        masked = text
        for name, (pattern, replacement) in cls.PATTERNS.items():
            masked = re.sub(pattern, replacement, masked)
        return masked


class DataExtractor:
    """문서 데이터 추출기"""

    def __init__(self, mask_pii: bool = True):
        self.mask_pii = mask_pii
        self.pii_masker = PIIMasker()

    def extract_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """CSV 파일에서 데이터 추출"""
        import pandas as pd

        df = pd.read_csv(file_path, encoding='utf-8')
        records = []

        for _, row in df.iterrows():
            # 모든 컬럼을 텍스트로 결합
            content = " | ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))

            if self.mask_pii:
                content = self.pii_masker.mask(content)

            records.append({
                'content': content,
                'metadata': dict(row),
                'source': file_path.name
            })

        return records

    def extract_excel(self, file_path: Path) -> List[Dict[str, Any]]:
        """Excel 파일에서 데이터 추출"""
        import pandas as pd

        df = pd.read_excel(file_path)
        return self._dataframe_to_records(df, file_path.name)

    def extract_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        """PDF 파일에서 데이터 추출"""
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("pdfplumber 설치 필요: pip install pdfplumber")

        records = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""

                if self.mask_pii:
                    text = self.pii_masker.mask(text)

                if text.strip():
                    records.append({
                        'content': text,
                        'metadata': {'page': i + 1},
                        'source': file_path.name
                    })

        return records

    def extract_docx(self, file_path: Path) -> List[Dict[str, Any]]:
        """Word 문서에서 데이터 추출"""
        try:
            from docx import Document
        except ImportError:
            raise ImportError("python-docx 설치 필요: pip install python-docx")

        doc = Document(file_path)
        paragraphs = []

        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                if self.mask_pii:
                    text = self.pii_masker.mask(text)
                paragraphs.append(text)

        return [{
            'content': "\n".join(paragraphs),
            'metadata': {},
            'source': file_path.name
        }]

    def extract_txt(self, file_path: Path) -> List[Dict[str, Any]]:
        """텍스트 파일에서 데이터 추출"""
        content = file_path.read_text(encoding='utf-8')

        if self.mask_pii:
            content = self.pii_masker.mask(content)

        return [{
            'content': content,
            'metadata': {},
            'source': file_path.name
        }]

    def extract(self, file_path: Path) -> List[Dict[str, Any]]:
        """파일 형식에 따라 적절한 추출기 호출"""
        suffix = file_path.suffix.lower()

        extractors = {
            '.csv': self.extract_csv,
            '.xlsx': self.extract_excel,
            '.xls': self.extract_excel,
            '.pdf': self.extract_pdf,
            '.docx': self.extract_docx,
            '.txt': self.extract_txt,
        }

        extractor = extractors.get(suffix)
        if not extractor:
            raise ValueError(f"지원하지 않는 파일 형식: {suffix}")

        return extractor(file_path)

    def _dataframe_to_records(self, df, source: str) -> List[Dict[str, Any]]:
        """DataFrame을 레코드 리스트로 변환"""
        import pandas as pd

        records = []
        for _, row in df.iterrows():
            content = " | ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))

            if self.mask_pii:
                content = self.pii_masker.mask(content)

            records.append({
                'content': content,
                'metadata': dict(row),
                'source': source
            })

        return records


class TrainingDataConverter:
    """추출된 데이터를 학습 형식으로 변환"""

    def __init__(self, extractor: DataExtractor):
        self.extractor = extractor

    def convert_to_instruction_format(
        self,
        records: List[Dict],
        instruction_template: str = "다음 정보를 요약해주세요:"
    ) -> List[TrainingSample]:
        """Instruction 형식으로 변환"""
        samples = []

        for record in records:
            sample = TrainingSample(
                instruction=instruction_template,
                input=record['content'][:2000],  # 입력 길이 제한
                output="",  # 레이블링 필요
                source=record['source']
            )
            samples.append(sample)

        return samples

    def convert_to_qa_format(
        self,
        records: List[Dict]
    ) -> List[TrainingSample]:
        """Q&A 형식으로 변환 (학술행사 데이터용)"""
        samples = []

        for record in records:
            metadata = record.get('metadata', {})

            # 행사 정보인 경우 Q&A 형식으로 변환
            if '행사명' in metadata:
                event_name = metadata.get('행사명', '')
                start_date = metadata.get('행사 시작일', '')
                location = metadata.get('행사장소', '')
                credit = metadata.get('평점', '')
                url = metadata.get('url', '')

                # 여러 Q&A 쌍 생성
                qa_pairs = [
                    (f"{event_name}의 일정이 언제인가요?", f"{event_name}은 {start_date}에 개최됩니다."),
                    (f"{event_name}의 장소는 어디인가요?", f"{location}에서 진행됩니다."),
                    (f"{event_name}의 평점은 어떻게 되나요?", f"{credit}"),
                ]

                for question, answer in qa_pairs:
                    if answer.strip():
                        samples.append(TrainingSample(
                            instruction=question,
                            input="",
                            output=answer,
                            source=record['source']
                        ))
            else:
                # 일반 문서
                samples.append(TrainingSample(
                    instruction="다음 내용에 대해 설명해주세요:",
                    input=record['content'][:2000],
                    output="",
                    source=record['source']
                ))

        return samples

    def save_jsonl(self, samples: List[TrainingSample], output_path: Path):
        """JSONL 형식으로 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(asdict(sample), ensure_ascii=False) + '\n')

        print(f"저장 완료: {output_path} ({len(samples)}개 샘플)")


def main():
    parser = argparse.ArgumentParser(description='데이터 추출 및 변환')
    parser.add_argument('input', type=str, help='입력 파일 또는 디렉토리')
    parser.add_argument('--output', '-o', type=str, default='train_data.jsonl', help='출력 파일')
    parser.add_argument('--format', type=str, choices=['instruction', 'qa'], default='qa', help='출력 형식')
    parser.add_argument('--no-mask-pii', action='store_true', help='PII 마스킹 비활성화')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    extractor = DataExtractor(mask_pii=not args.no_mask_pii)
    converter = TrainingDataConverter(extractor)

    all_records = []

    if input_path.is_file():
        all_records = extractor.extract(input_path)
    elif input_path.is_dir():
        for file in input_path.glob('**/*'):
            if file.is_file() and file.suffix.lower() in ['.csv', '.xlsx', '.pdf', '.docx', '.txt']:
                try:
                    records = extractor.extract(file)
                    all_records.extend(records)
                    print(f"추출: {file.name} ({len(records)}개 레코드)")
                except Exception as e:
                    print(f"오류: {file.name} - {e}")

    # 형식 변환
    if args.format == 'qa':
        samples = converter.convert_to_qa_format(all_records)
    else:
        samples = converter.convert_to_instruction_format(all_records)

    # 저장
    converter.save_jsonl(samples, output_path)


if __name__ == '__main__':
    main()
