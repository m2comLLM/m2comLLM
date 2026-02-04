---
name: pii-scrubber
description: 텍스트에서 개인식별정보(PII)를 탐지하고 마스킹합니다.
user-invocable: true
---

# PII Scrubber 스킬

데이터에서 개인정보를 자동으로 탐지하고 제거합니다.

## 지원하는 PII 유형

| 유형 | 패턴 | 마스킹 결과 |
|------|------|------------|
| 이메일 | user@domain.com | [EMAIL] |
| 전화번호 | 010-1234-5678 | [PHONE] |
| 주민등록번호 | 901231-1234567 | [RRN] |
| 신용카드 | 1234-5678-9012-3456 | [CARD] |
| 이름 | 홍길동 | [NAME] |
| 주소 | 서울시 강남구... | [ADDRESS] |

## 사용법

```bash
# 파일 처리
python scripts/scrub_pii.py input.txt --output cleaned.txt

# 디렉토리 전체 처리
python scripts/scrub_pii.py data/ --output cleaned_data/

# 보고서 생성
python scripts/scrub_pii.py input.txt --report pii_report.json
```

## 의존성

- `regex`: 고급 정규표현식
- `presidio-analyzer`: Microsoft PII 탐지 (선택)
