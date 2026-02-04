"""
학술행사 안내 챗봇
대한결핵 및 호흡기학회 학술행사 데이터 기반
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from loguru import logger


class EventChatbot:
    """학술행사 안내 챗봇"""

    def __init__(self, data_path: str = None):
        """
        Args:
            data_path: CSV 데이터 파일 경로
        """
        self.data_path = data_path or "data/대한결핵 및 호흡기학회 학술행사(csv).csv"
        self.df = None
        self._load_data()

    def _load_data(self):
        """데이터 로드"""
        try:
            self.df = pd.read_csv(self.data_path, encoding="utf-8")
            # 날짜 컬럼 변환
            date_columns = ["행사 시작일", "행사 종료일", "등록 시작일", "등록 마감일"]
            for col in date_columns:
                if col in self.df.columns:
                    self.df[col] = pd.to_datetime(self.df[col], errors="coerce")
            logger.info(f"데이터 로드 완료: {len(self.df)}개 행사")
        except Exception as e:
            logger.error(f"데이터 로드 실패: {e}")
            self.df = pd.DataFrame()

    def search_events(
        self,
        keyword: str = None,
        start_date: str = None,
        end_date: str = None,
        location: str = None,
        limit: int = 10,
    ) -> List[dict]:
        """
        행사 검색

        Args:
            keyword: 검색 키워드 (행사명에서 검색)
            start_date: 시작일 이후 (YYYY-MM-DD)
            end_date: 종료일 이전 (YYYY-MM-DD)
            location: 장소 키워드
            limit: 최대 결과 수

        Returns:
            검색된 행사 목록
        """
        if self.df is None or self.df.empty:
            return []

        result = self.df.copy()

        # 키워드 검색
        if keyword:
            mask = result["행사명"].str.contains(keyword, case=False, na=False)
            result = result[mask]

        # 날짜 필터
        if start_date:
            start = pd.to_datetime(start_date)
            result = result[result["행사 시작일"] >= start]

        if end_date:
            end = pd.to_datetime(end_date)
            result = result[result["행사 시작일"] <= end]

        # 장소 필터
        if location:
            mask = result["행사장소"].str.contains(location, case=False, na=False)
            result = result[mask]

        # 최신순 정렬
        result = result.sort_values("행사 시작일", ascending=False)

        return result.head(limit).to_dict("records")

    def get_upcoming_events(self, days: int = 30, limit: int = 5) -> List[dict]:
        """
        다가오는 행사 조회

        Args:
            days: 앞으로 며칠 내 행사
            limit: 최대 결과 수

        Returns:
            다가오는 행사 목록
        """
        if self.df is None or self.df.empty:
            return []

        today = pd.Timestamp.now()
        future = today + pd.Timedelta(days=days)

        mask = (self.df["행사 시작일"] >= today) & (self.df["행사 시작일"] <= future)
        result = self.df[mask].sort_values("행사 시작일")

        return result.head(limit).to_dict("records")

    def get_event_by_name(self, name: str) -> Optional[dict]:
        """행사명으로 상세 정보 조회"""
        if self.df is None or self.df.empty:
            return None

        mask = self.df["행사명"].str.contains(name, case=False, na=False)
        result = self.df[mask]

        if result.empty:
            return None

        return result.iloc[0].to_dict()

    def get_registration_open_events(self, limit: int = 10) -> List[dict]:
        """
        등록 가능한 행사 조회

        Returns:
            현재 등록 가능한 행사 목록
        """
        if self.df is None or self.df.empty:
            return []

        today = pd.Timestamp.now()

        mask = (
            (self.df["등록 시작일"] <= today) &
            (self.df["등록 마감일"] >= today)
        )
        result = self.df[mask].sort_values("등록 마감일")

        return result.head(limit).to_dict("records")

    def format_event(self, event: dict) -> str:
        """행사 정보를 읽기 좋은 형태로 포맷팅"""
        lines = []
        lines.append(f"📌 **{event.get('행사명', 'N/A')}**")
        lines.append("")

        # 날짜 정보
        start = event.get("행사 시작일")
        end = event.get("행사 종료일")
        if pd.notna(start):
            start_str = pd.Timestamp(start).strftime("%Y-%m-%d")
            if pd.notna(end) and start != end:
                end_str = pd.Timestamp(end).strftime("%Y-%m-%d")
                lines.append(f"📅 일시: {start_str} ~ {end_str}")
            else:
                lines.append(f"📅 일시: {start_str}")

        # 장소
        location = event.get("행사장소")
        if pd.notna(location):
            lines.append(f"📍 장소: {location}")

        # 평점
        credit = event.get("평점")
        if pd.notna(credit):
            lines.append(f"🏆 평점: {credit}")

        # 등록 기간
        reg_start = event.get("등록 시작일")
        reg_end = event.get("등록 마감일")
        if pd.notna(reg_start) and pd.notna(reg_end):
            reg_start_str = pd.Timestamp(reg_start).strftime("%Y-%m-%d")
            reg_end_str = pd.Timestamp(reg_end).strftime("%Y-%m-%d")
            lines.append(f"📝 등록기간: {reg_start_str} ~ {reg_end_str}")

        # URL
        url = event.get("url")
        if pd.notna(url):
            lines.append(f"🔗 링크: {url}")

        return "\n".join(lines)

    def format_event_list(self, events: List[dict]) -> str:
        """행사 목록 포맷팅"""
        if not events:
            return "검색 결과가 없습니다."

        result = []
        for i, event in enumerate(events, 1):
            name = event.get("행사명", "N/A")
            start = event.get("행사 시작일")
            location = event.get("행사장소", "")

            if pd.notna(start):
                start_str = pd.Timestamp(start).strftime("%Y-%m-%d")
            else:
                start_str = "미정"

            # 장소 축약
            if location and len(location) > 20:
                location = location[:20] + "..."

            result.append(f"{i}. **{name}**")
            result.append(f"   📅 {start_str} | 📍 {location}")
            result.append("")

        return "\n".join(result)

    def chat(self, query: str) -> str:
        """
        자연어 쿼리 처리

        Args:
            query: 사용자 질문

        Returns:
            응답 텍스트
        """
        query_lower = query.lower()

        # 다가오는 행사
        if any(kw in query_lower for kw in ["다가오는", "예정", "upcoming", "앞으로", "다음"]):
            events = self.get_upcoming_events(days=60, limit=5)
            if events:
                return f"## 다가오는 행사\n\n{self.format_event_list(events)}"
            return "다가오는 행사가 없습니다."

        # 등록 가능한 행사
        if any(kw in query_lower for kw in ["등록", "신청", "registration"]):
            events = self.get_registration_open_events(limit=5)
            if events:
                return f"## 현재 등록 가능한 행사\n\n{self.format_event_list(events)}"
            return "현재 등록 가능한 행사가 없습니다."

        # 특정 키워드 검색
        keywords = ["천식", "copd", "결핵", "폐암", "폐기능", "호흡", "감염", "금연",
                    "ild", "수면", "기침", "폐혈관", "재활", "환경"]
        for kw in keywords:
            if kw in query_lower:
                events = self.search_events(keyword=kw, limit=5)
                if events:
                    return f"## '{kw}' 관련 행사\n\n{self.format_event_list(events)}"
                return f"'{kw}' 관련 행사를 찾을 수 없습니다."

        # 연도 검색
        import re
        year_match = re.search(r'20\d{2}', query)
        if year_match:
            year = year_match.group()
            events = self.search_events(
                start_date=f"{year}-01-01",
                end_date=f"{year}-12-31",
                limit=10
            )
            if events:
                return f"## {year}년 행사\n\n{self.format_event_list(events)}"
            return f"{year}년 행사를 찾을 수 없습니다."

        # 기본 검색
        events = self.search_events(keyword=query, limit=5)
        if events:
            return f"## 검색 결과\n\n{self.format_event_list(events)}"

        # 전체 행사 안내
        return """## 대한결핵 및 호흡기학회 학술행사 안내

다음과 같이 질문해 보세요:
- "다가오는 행사 알려줘"
- "등록 가능한 행사"
- "천식 관련 행사"
- "2025년 행사"
- "COPD 심포지엄"

또는 특정 키워드로 검색할 수 있습니다."""


# 싱글톤 인스턴스
_chatbot = None


def get_chatbot() -> EventChatbot:
    """챗봇 싱글톤 인스턴스"""
    global _chatbot
    if _chatbot is None:
        _chatbot = EventChatbot()
    return _chatbot
