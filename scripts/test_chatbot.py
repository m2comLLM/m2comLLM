"""
EventChatbot 테스트 스크립트
"""

import sys
sys.path.insert(0, "/workspace/dev")

from src.chatbot import EventChatbot, get_chatbot

def test_chatbot():
    """챗봇 테스트"""
    print("=" * 60)
    print("EventChatbot 테스트")
    print("=" * 60)

    # 챗봇 초기화
    chatbot = get_chatbot()
    print(f"\n✅ 챗봇 초기화 완료")
    print(f"   - 데이터 경로: {chatbot.data_path}")
    print(f"   - 로드된 행사 수: {len(chatbot.df) if chatbot.df is not None else 0}")

    # 컬럼 확인
    if chatbot.df is not None and not chatbot.df.empty:
        print(f"   - 컬럼: {list(chatbot.df.columns)}")

    print("\n" + "-" * 60)
    print("테스트 1: 다가오는 행사 조회")
    print("-" * 60)
    response = chatbot.chat("다가오는 행사 알려줘")
    print(response)

    print("\n" + "-" * 60)
    print("테스트 2: 등록 가능한 행사")
    print("-" * 60)
    response = chatbot.chat("등록 가능한 행사")
    print(response)

    print("\n" + "-" * 60)
    print("테스트 3: 키워드 검색 - 천식")
    print("-" * 60)
    response = chatbot.chat("천식 관련 행사")
    print(response)

    print("\n" + "-" * 60)
    print("테스트 4: 연도 검색 - 2024")
    print("-" * 60)
    response = chatbot.chat("2024년 행사")
    print(response)

    print("\n" + "-" * 60)
    print("테스트 5: 일반 검색")
    print("-" * 60)
    response = chatbot.chat("심포지엄")
    print(response)

    print("\n" + "-" * 60)
    print("테스트 6: 상세 정보 조회")
    print("-" * 60)
    events = chatbot.search_events(limit=1)
    if events:
        print(chatbot.format_event(events[0]))

    print("\n" + "=" * 60)
    print("✅ 모든 테스트 완료")
    print("=" * 60)

if __name__ == "__main__":
    test_chatbot()
