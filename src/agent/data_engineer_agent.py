"""
Data Engineer Agent
사내 데이터 수집, 정제, 변환을 담당하는 에이전트
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

# 스킬 임포트
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from skills.data_extractor.extract_data import DataExtractor, TrainingDataConverter
from skills.pii_scrubber.scrub_pii import PIIScrubber


@dataclass
class DataPipelineConfig:
    """데이터 파이프라인 설정"""
    input_path: str
    output_path: str = "train_data.jsonl"
    mask_pii: bool = True
    output_format: str = "qa"  # qa, instruction
    chunk_size: int = 512
    chunk_overlap: int = 128


@dataclass
class PipelineResult:
    """파이프라인 실행 결과"""
    total_files: int
    total_records: int
    total_samples: int
    pii_masked: int
    output_path: str
    duration_seconds: float
    errors: List[str]


class DataEngineerAgent:
    """
    Data Engineer Agent

    역할:
    - 사내 데이터(PDF, CSV, XLSX, DOCX) 수집
    - 개인정보(PII) 탐지 및 마스킹
    - LLM 학습용 JSONL 형식 변환

    사용 스킬:
    - data-extractor: 다양한 포맷 데이터 추출
    - pii-scrubber: 개인정보 마스킹
    """

    SYSTEM_PROMPT = """당신은 LLM 학습 데이터 엔지니어입니다.

## 역할
- 사내 문서에서 학습 데이터 추출
- 개인정보 탐지 및 제거
- 고품질 학습 데이터셋 생성

## 원칙
1. 데이터 품질 우선: 노이즈 데이터 필터링
2. 개인정보 보호: 모든 PII 완벽 마스킹
3. 형식 표준화: Alpaca/ShareGPT 형식 준수
4. 메타데이터 보존: 출처, 날짜 정보 유지

## 출력 형식
모든 데이터는 JSONL 형식으로 출력합니다:
{"instruction": "질문", "input": "맥락", "output": "답변", "source": "출처"}
"""

    def __init__(self, config: Optional[DataPipelineConfig] = None):
        self.config = config or DataPipelineConfig(input_path="data/")
        self.extractor = DataExtractor(mask_pii=self.config.mask_pii)
        self.pii_scrubber = PIIScrubber()
        self.converter = TrainingDataConverter(self.extractor)

    async def run_pipeline(
        self,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> PipelineResult:
        """전체 데이터 파이프라인 실행"""
        start_time = datetime.now()
        errors = []

        input_dir = Path(input_path or self.config.input_path)
        output_file = Path(output_path or self.config.output_path)

        logger.info(f"데이터 파이프라인 시작: {input_dir}")

        # 1. 파일 수집
        files = self._collect_files(input_dir)
        logger.info(f"수집된 파일: {len(files)}개")

        # 2. 데이터 추출
        all_records = []
        for file in files:
            try:
                records = self.extractor.extract(file)
                all_records.extend(records)
                logger.debug(f"추출: {file.name} ({len(records)}개 레코드)")
            except Exception as e:
                error_msg = f"{file.name}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"추출 오류: {error_msg}")

        logger.info(f"총 레코드: {len(all_records)}개")

        # 3. PII 추가 검증 (이미 extractor에서 처리됨)
        pii_count = 0
        if self.config.mask_pii:
            for record in all_records:
                _, matches = self.pii_scrubber.scrub(record.get('content', ''))
                pii_count += len(matches)

        # 4. 학습 형식 변환
        if self.config.output_format == 'qa':
            samples = self.converter.convert_to_qa_format(all_records)
        else:
            samples = self.converter.convert_to_instruction_format(all_records)

        logger.info(f"변환된 샘플: {len(samples)}개")

        # 5. 저장
        self.converter.save_jsonl(samples, output_file)

        duration = (datetime.now() - start_time).total_seconds()

        return PipelineResult(
            total_files=len(files),
            total_records=len(all_records),
            total_samples=len(samples),
            pii_masked=pii_count,
            output_path=str(output_file),
            duration_seconds=duration,
            errors=errors,
        )

    def _collect_files(self, input_path: Path) -> List[Path]:
        """지원되는 파일 수집"""
        supported_extensions = {'.csv', '.xlsx', '.xls', '.pdf', '.docx', '.txt'}

        if input_path.is_file():
            return [input_path] if input_path.suffix.lower() in supported_extensions else []

        files = []
        for ext in supported_extensions:
            files.extend(input_path.glob(f'**/*{ext}'))

        return sorted(files)

    async def analyze_data_quality(self, data_path: str) -> Dict[str, Any]:
        """데이터 품질 분석"""
        import json

        path = Path(data_path)
        if not path.exists():
            return {"error": "파일이 존재하지 않습니다."}

        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                samples.append(json.loads(line))

        # 품질 메트릭 계산
        total = len(samples)
        empty_output = sum(1 for s in samples if not s.get('output', '').strip())
        short_instruction = sum(1 for s in samples if len(s.get('instruction', '')) < 10)
        long_input = sum(1 for s in samples if len(s.get('input', '')) > 2000)

        avg_instruction_len = sum(len(s.get('instruction', '')) for s in samples) / total if total else 0
        avg_output_len = sum(len(s.get('output', '')) for s in samples) / total if total else 0

        return {
            "total_samples": total,
            "empty_output": empty_output,
            "short_instruction": short_instruction,
            "long_input": long_input,
            "avg_instruction_length": avg_instruction_len,
            "avg_output_length": avg_output_len,
            "quality_score": (total - empty_output - short_instruction) / total * 100 if total else 0,
        }

    async def handle_task(self, task: str) -> str:
        """에이전트 태스크 처리"""
        task_lower = task.lower()

        if "추출" in task_lower or "변환" in task_lower:
            result = await self.run_pipeline()
            return f"""데이터 추출 완료:
- 처리 파일: {result.total_files}개
- 추출 레코드: {result.total_records}개
- 학습 샘플: {result.total_samples}개
- PII 마스킹: {result.pii_masked}건
- 출력: {result.output_path}
- 소요 시간: {result.duration_seconds:.1f}초
{"- 오류: " + ", ".join(result.errors) if result.errors else ""}"""

        elif "품질" in task_lower or "분석" in task_lower:
            quality = await self.analyze_data_quality(self.config.output_path)
            return f"""데이터 품질 분석:
- 총 샘플: {quality.get('total_samples', 0)}개
- 빈 출력: {quality.get('empty_output', 0)}개
- 짧은 instruction: {quality.get('short_instruction', 0)}개
- 품질 점수: {quality.get('quality_score', 0):.1f}%"""

        else:
            return f"지원하지 않는 태스크입니다. 사용 가능: 데이터 추출, 품질 분석"


async def main():
    """테스트 실행"""
    agent = DataEngineerAgent(
        config=DataPipelineConfig(
            input_path="data/",
            output_path="train_data.jsonl",
            mask_pii=True,
            output_format="qa",
        )
    )

    # 파이프라인 실행
    result = await agent.run_pipeline()

    print("=== Data Engineer Agent 실행 완료 ===")
    print(f"파일: {result.total_files}개")
    print(f"레코드: {result.total_records}개")
    print(f"샘플: {result.total_samples}개")
    print(f"출력: {result.output_path}")


if __name__ == "__main__":
    asyncio.run(main())
