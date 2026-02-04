"""
PII Scrubber Script
개인식별정보 탐지 및 마스킹
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import argparse


@dataclass
class PIIMatch:
    """PII 탐지 결과"""
    pii_type: str
    original: str
    start: int
    end: int
    masked: str


class PIIScrubber:
    """개인정보 탐지 및 마스킹"""

    # 정규표현식 패턴들
    PATTERNS: Dict[str, Tuple[str, str]] = {
        'email': (
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            '[EMAIL]'
        ),
        'phone': (
            r'(\d{2,3}[-.\s]?\d{3,4}[-.\s]?\d{4})',
            '[PHONE]'
        ),
        'rrn': (
            r'\d{6}[-\s]?[1-4]\d{6}',
            '[RRN]'
        ),
        'card': (
            r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}',
            '[CARD]'
        ),
        'ip_address': (
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            '[IP]'
        ),
        'date_of_birth': (
            r'\b(19|20)\d{2}[-./](0[1-9]|1[0-2])[-./](0[1-9]|[12]\d|3[01])\b',
            '[DOB]'
        ),
    }

    # 한국 이름 패턴 (성 + 이름)
    KOREAN_SURNAMES = ['김', '이', '박', '최', '정', '강', '조', '윤', '장', '임',
                       '한', '오', '서', '신', '권', '황', '안', '송', '류', '전']

    def __init__(self, custom_patterns: Optional[Dict] = None):
        self.patterns = {**self.PATTERNS}
        if custom_patterns:
            self.patterns.update(custom_patterns)

        self.stats = defaultdict(int)

    def detect(self, text: str) -> List[PIIMatch]:
        """텍스트에서 PII 탐지"""
        matches = []

        for pii_type, (pattern, replacement) in self.patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matches.append(PIIMatch(
                    pii_type=pii_type,
                    original=match.group(),
                    start=match.start(),
                    end=match.end(),
                    masked=replacement
                ))

        # 한국 이름 탐지
        name_pattern = f"({'|'.join(self.KOREAN_SURNAMES)})[가-힣]{{1,3}}"
        for match in re.finditer(name_pattern, text):
            name = match.group()
            if len(name) >= 2:  # 최소 2글자
                matches.append(PIIMatch(
                    pii_type='name',
                    original=name,
                    start=match.start(),
                    end=match.end(),
                    masked='[NAME]'
                ))

        # 위치순으로 정렬
        matches.sort(key=lambda x: x.start)

        return matches

    def scrub(self, text: str) -> Tuple[str, List[PIIMatch]]:
        """텍스트에서 PII 제거"""
        matches = self.detect(text)

        # 뒤에서부터 교체 (인덱스 유지)
        result = text
        for match in reversed(matches):
            result = result[:match.start] + match.masked + result[match.end:]
            self.stats[match.pii_type] += 1

        return result, matches

    def scrub_file(self, input_path: Path, output_path: Path) -> List[PIIMatch]:
        """파일에서 PII 제거"""
        text = input_path.read_text(encoding='utf-8')
        scrubbed, matches = self.scrub(text)
        output_path.write_text(scrubbed, encoding='utf-8')
        return matches

    def generate_report(self, matches: List[PIIMatch]) -> Dict:
        """PII 탐지 보고서 생성"""
        report = {
            'total_pii_found': len(matches),
            'by_type': defaultdict(list),
            'summary': {}
        }

        for match in matches:
            report['by_type'][match.pii_type].append({
                'original': match.original[:3] + '***',  # 일부만 표시
                'position': f"{match.start}-{match.end}"
            })

        for pii_type in report['by_type']:
            report['summary'][pii_type] = len(report['by_type'][pii_type])

        return dict(report)


def main():
    parser = argparse.ArgumentParser(description='PII 탐지 및 마스킹')
    parser.add_argument('input', type=str, help='입력 파일 또는 디렉토리')
    parser.add_argument('--output', '-o', type=str, help='출력 경로')
    parser.add_argument('--report', '-r', type=str, help='보고서 출력 경로')
    parser.add_argument('--dry-run', action='store_true', help='탐지만 수행 (파일 수정 안함)')

    args = parser.parse_args()

    input_path = Path(args.input)
    scrubber = PIIScrubber()

    all_matches = []

    if input_path.is_file():
        if args.dry_run:
            text = input_path.read_text(encoding='utf-8')
            _, matches = scrubber.scrub(text)
            all_matches.extend(matches)
            print(f"탐지된 PII: {len(matches)}개")
        else:
            output_path = Path(args.output) if args.output else input_path.with_suffix('.cleaned.txt')
            matches = scrubber.scrub_file(input_path, output_path)
            all_matches.extend(matches)
            print(f"처리 완료: {input_path.name} -> {output_path.name} ({len(matches)}개 PII 마스킹)")

    elif input_path.is_dir():
        output_dir = Path(args.output) if args.output else input_path / 'cleaned'
        output_dir.mkdir(exist_ok=True)

        for file in input_path.glob('**/*.txt'):
            output_file = output_dir / file.name
            matches = scrubber.scrub_file(file, output_file)
            all_matches.extend(matches)
            print(f"처리: {file.name} ({len(matches)}개 PII)")

    # 보고서 생성
    if args.report:
        report = scrubber.generate_report(all_matches)
        report_path = Path(args.report)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"보고서 저장: {report_path}")

    # 요약 출력
    print(f"\n=== PII 처리 요약 ===")
    print(f"총 탐지: {len(all_matches)}개")
    for pii_type, count in scrubber.stats.items():
        print(f"  - {pii_type}: {count}개")


if __name__ == '__main__':
    main()
