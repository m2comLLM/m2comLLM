---
name: data-extractor
description: 사내 원시 데이터(PDF, CSV, Docx)를 LLM 학습용 JSONL 형식으로 변환합니다.
user-invocable: true
---

# Data Extractor 스킬

사내 문서를 LLM 학습에 적합한 정제된 데이터셋으로 변환합니다.

## 기능

1. **다양한 포맷 지원**: PDF, CSV, XLSX, DOCX, TXT 파일 처리
2. **PII 마스킹**: 개인정보(이메일, 전화번호, 이름) 자동 제거
3. **JSONL 변환**: Hugging Face 학습 형식으로 출력
4. **메타데이터 보존**: 출처, 날짜, 카테고리 정보 유지

## 사용법

```bash
# 단일 파일 처리
python scripts/extract_data.py data/document.pdf --output train_data.jsonl

# 디렉토리 전체 처리
python scripts/extract_data.py data/ --output train_data.jsonl --format jsonl
```

## 출력 형식

```json
{"instruction": "질문", "input": "컨텍스트", "output": "답변", "source": "파일명"}
```

## 의존성

- `pdfplumber`: PDF 텍스트 추출
- `python-docx`: Word 문서 처리
- `pandas`: CSV/Excel 처리
