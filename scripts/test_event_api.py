"""
Event API 테스트 스크립트
"""

import sys
sys.path.insert(0, "/workspace/dev")

from fastapi.testclient import TestClient
from src.api.event_api import app

def test_event_api():
    """Event API 테스트"""
    client = TestClient(app)

    print("=" * 60)
    print("Event API 테스트")
    print("=" * 60)

    # 1. 헬스 체크
    print("\n1. 헬스 체크...")
    response = client.get("/health")
    assert response.status_code == 200
    print(f"   ✅ {response.json()}")

    # 2. 모델 목록
    print("\n2. 모델 목록...")
    response = client.get("/v1/models")
    assert response.status_code == 200
    models = [m["id"] for m in response.json()["data"]]
    print(f"   ✅ 사용 가능한 모델: {models}")
    assert "event-chatbot" in models

    # 3. 챗봇 API
    print("\n3. 챗봇 API 테스트...")
    response = client.post("/v1/chat", json={
        "query": "다가오는 행사 알려줘"
    })
    assert response.status_code == 200
    result = response.json()
    print(f"   ✅ 응답 길이: {len(result['response'])} characters")
    print(f"   응답 미리보기:\n{result['response'][:500]}...")

    # 4. 검색 API
    print("\n4. 검색 API 테스트...")
    response = client.post("/v1/search", json={
        "keyword": "천식",
        "limit": 3
    })
    assert response.status_code == 200
    result = response.json()
    print(f"   ✅ 검색 결과: {len(result.get('events', []))}개 행사")
    print(f"   응답:\n{result['response']}")

    # 5. 다가오는 행사
    print("\n5. 다가오는 행사 API...")
    response = client.get("/v1/upcoming?days=60&limit=3")
    assert response.status_code == 200
    result = response.json()
    print(f"   ✅ 결과: {len(result.get('events', []))}개 행사")
    print(f"   응답:\n{result['response']}")

    # 6. 등록 가능한 행사
    print("\n6. 등록 가능한 행사 API...")
    response = client.get("/v1/registration?limit=3")
    assert response.status_code == 200
    result = response.json()
    print(f"   ✅ 결과: {len(result.get('events', []))}개 행사")
    print(f"   응답:\n{result['response']}")

    # 7. 행사 상세 정보
    print("\n7. 행사 상세 정보 API...")
    response = client.get("/v1/event/천식연구회")
    assert response.status_code == 200
    result = response.json()
    if result.get('events'):
        print(f"   ✅ 행사 찾음")
        print(f"   응답:\n{result['response']}")
    else:
        print(f"   결과: {result['response']}")

    print("\n" + "=" * 60)
    print("✅ 모든 Event API 테스트 통과")
    print("=" * 60)

if __name__ == "__main__":
    test_event_api()
