"""
학술행사 챗봇 API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import time
from loguru import logger

from src.chatbot import get_chatbot

app = FastAPI(
    title="학술행사 안내 챗봇 API",
    description="대한결핵 및 호흡기학회 학술행사 안내",
    version="1.0.0",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# 요청/응답 모델
# ============================================================================

class ChatRequest(BaseModel):
    query: str


class SearchRequest(BaseModel):
    keyword: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    location: Optional[str] = None
    limit: int = 10


class EventResponse(BaseModel):
    response: str
    events: Optional[List[dict]] = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# ============================================================================
# API 엔드포인트
# ============================================================================

@app.get("/")
async def root():
    """헬스 체크"""
    return {
        "status": "ok",
        "service": "학술행사 안내 챗봇",
        "description": "대한결핵 및 호흡기학회 학술행사 안내 챗봇 API"
    }


@app.get("/health")
async def health():
    """헬스 체크"""
    return {"status": "healthy"}


@app.get("/v1/models")
async def list_models() -> ModelList:
    """사용 가능한 모델 목록"""
    return ModelList(
        data=[
            ModelInfo(
                id="event-chatbot",
                created=int(time.time()),
                owned_by="local",
            ),
        ]
    )


@app.post("/v1/chat")
async def chat(request: ChatRequest) -> EventResponse:
    """
    자연어 챗봇 대화 API
    """
    try:
        chatbot = get_chatbot()
        response = chatbot.chat(request.query)
        return EventResponse(response=response)
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/search")
async def search(request: SearchRequest) -> EventResponse:
    """
    학술행사 검색 API
    """
    try:
        chatbot = get_chatbot()
        events = chatbot.search_events(
            keyword=request.keyword,
            start_date=request.start_date,
            end_date=request.end_date,
            location=request.location,
            limit=request.limit,
        )
        formatted = chatbot.format_event_list(events)
        return EventResponse(response=formatted, events=events)
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/upcoming")
async def upcoming_events(days: int = 30, limit: int = 5) -> EventResponse:
    """다가오는 행사 조회"""
    try:
        chatbot = get_chatbot()
        events = chatbot.get_upcoming_events(days=days, limit=limit)
        formatted = chatbot.format_event_list(events)
        return EventResponse(response=formatted, events=events)
    except Exception as e:
        logger.error(f"Upcoming events error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/registration")
async def registration_open_events(limit: int = 10) -> EventResponse:
    """등록 가능한 행사 조회"""
    try:
        chatbot = get_chatbot()
        events = chatbot.get_registration_open_events(limit=limit)
        formatted = chatbot.format_event_list(events)
        return EventResponse(response=formatted, events=events)
    except Exception as e:
        logger.error(f"Registration events error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/event/{event_name}")
async def get_event_detail(event_name: str) -> EventResponse:
    """행사 상세 정보 조회"""
    try:
        chatbot = get_chatbot()
        event = chatbot.get_event_by_name(event_name)
        if event:
            formatted = chatbot.format_event(event)
            return EventResponse(response=formatted, events=[event])
        else:
            return EventResponse(response="해당 행사를 찾을 수 없습니다.", events=[])
    except Exception as e:
        logger.error(f"Event detail error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# 서버 실행
# ============================================================================

def run_server(host: str = "0.0.0.0", port: int = 8081):
    """서버 실행"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
