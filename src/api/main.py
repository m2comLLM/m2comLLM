"""
OpenAI 호환 API 서버
Open WebUI 연동용
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, AsyncGenerator
import asyncio
import json
import time
import uuid
from loguru import logger

from src.agent.langgraph_agent import MedicalAgent
from src.chatbot import EventChatbot, get_chatbot
from src.config import settings

app = FastAPI(
    title="Medical RAG Agent API",
    description="OpenAI 호환 API - Open WebUI 연동",
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

# 에이전트 인스턴스 (lazy loading)
_agent: Optional[MedicalAgent] = None


def get_agent() -> MedicalAgent:
    """에이전트 싱글톤"""
    global _agent
    if _agent is None:
        _agent = MedicalAgent()
        logger.info("Medical Agent initialized")
    return _agent


# ============================================================================
# 요청/응답 모델 (OpenAI 호환)
# ============================================================================

class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "medical-rag-agent"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 4096
    stream: Optional[bool] = False
    user: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[dict]


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
    return {"status": "ok", "service": "Medical RAG Agent API"}


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
                id="medical-rag-agent",
                created=int(time.time()),
                owned_by="local",
            ),
            ModelInfo(
                id="deepseek-medical",
                created=int(time.time()),
                owned_by="local",
            ),
            ModelInfo(
                id="event-chatbot",
                created=int(time.time()),
                owned_by="local",
            ),
        ]
    )


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str) -> ModelInfo:
    """모델 정보 조회"""
    return ModelInfo(
        id=model_id,
        created=int(time.time()),
        owned_by="local",
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    채팅 완성 API (OpenAI 호환)
    Open WebUI에서 호출하는 메인 엔드포인트
    """
    try:
        # 스트리밍 여부에 따라 분기
        if request.stream:
            return StreamingResponse(
                stream_response(request),
                media_type="text/event-stream",
            )
        else:
            return await generate_response(request)

    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def generate_response(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """비스트리밍 응답 생성"""
    agent = get_agent()

    # 마지막 사용자 메시지 추출
    user_message = ""
    patient_id = None

    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message = msg.content
            # 환자 ID 추출 시도 (예: "환자 P12345" 패턴)
            import re
            match = re.search(r'환자\s*[ID]?\s*[:\s]?\s*([A-Za-z0-9]+)', msg.content)
            if match:
                patient_id = match.group(1)
            break

    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")

    # 에이전트 실행
    response_text = await agent.run(user_message, patient_id=patient_id)

    # 토큰 수 추정 (대략적)
    prompt_tokens = sum(len(m.content) // 4 for m in request.messages)
    completion_tokens = len(response_text) // 4

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=Message(role="assistant", content=response_text),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


async def stream_response(request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    """스트리밍 응답 생성"""
    agent = get_agent()

    # 마지막 사용자 메시지 추출
    user_message = ""
    patient_id = None

    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message = msg.content
            import re
            match = re.search(r'환자\s*[ID]?\s*[:\s]?\s*([A-Za-z0-9]+)', msg.content)
            if match:
                patient_id = match.group(1)
            break

    if not user_message:
        yield f"data: {json.dumps({'error': 'No user message found'})}\n\n"
        return

    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    try:
        # 에이전트 실행 (비동기)
        response_text = await agent.run(user_message, patient_id=patient_id)

        # 응답을 청크로 나누어 스트리밍
        chunk_size = 20  # 한 번에 보낼 문자 수

        for i in range(0, len(response_text), chunk_size):
            chunk = response_text[i:i + chunk_size]

            data = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": chunk},
                    "finish_reason": None,
                }],
            }

            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.01)  # 약간의 지연

        # 완료 신호
        data = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        }
        yield f"data: {json.dumps(data)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


# ============================================================================
# 추가 엔드포인트
# ============================================================================

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    use_rerank: bool = True


class SearchResult(BaseModel):
    content: str
    score: float
    metadata: dict


@app.post("/v1/search")
async def search_guidelines(request: SearchRequest) -> List[SearchResult]:
    """임상 가이드라인 검색 (직접 접근용)"""
    try:
        from src.retrieval.reranker import HybridSearchWithRerank

        searcher = HybridSearchWithRerank()
        results = searcher.search(
            query=request.query,
            top_k=request.top_k,
            use_rerank=request.use_rerank,
        )

        return [
            SearchResult(
                content=doc.get("content", ""),
                score=doc.get("rerank_score", doc.get("hybrid_score", 0)),
                metadata=doc.get("metadata", {}),
            )
            for doc in results
        ]

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class DrugInteractionRequest(BaseModel):
    drugs: List[str]


@app.post("/v1/drug-interactions")
async def check_drug_interactions(request: DrugInteractionRequest):
    """약물 상호작용 확인 (직접 접근용)"""
    try:
        from src.mcp_servers.drug_mcp import check_multiple_interactions

        results = await check_multiple_interactions(request.drugs)
        return {"result": results[0].text if results else "결과 없음"}

    except Exception as e:
        logger.error(f"Drug interaction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class PatientRequest(BaseModel):
    patient_id: str


@app.post("/v1/patient/summary")
async def get_patient_summary(request: PatientRequest):
    """환자 요약 정보 (직접 접근용)"""
    agent = get_agent()

    info = await agent._get_patient_info(request.patient_id)
    meds = await agent._get_patient_medications(request.patient_id)
    diagnoses = await agent._get_patient_diagnoses(request.patient_id)
    allergies = await agent._get_patient_allergies(request.patient_id)

    return {
        "patient_id": request.patient_id,
        "info": info,
        "medications": meds,
        "diagnoses": diagnoses,
        "allergies": allergies,
    }


# ============================================================================
# 학술행사 챗봇 엔드포인트
# ============================================================================

class EventChatRequest(BaseModel):
    query: str
    model: str = "event-chatbot"


class EventSearchRequest(BaseModel):
    keyword: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    location: Optional[str] = None
    limit: int = 10


class EventResponse(BaseModel):
    response: str
    events: Optional[List[dict]] = None


@app.post("/v1/events/chat")
async def event_chat(request: EventChatRequest) -> EventResponse:
    """
    학술행사 챗봇 API
    자연어로 학술행사 정보를 질의
    """
    try:
        chatbot = get_chatbot()
        response = chatbot.chat(request.query)
        return EventResponse(response=response)
    except Exception as e:
        logger.error(f"Event chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/events/search")
async def event_search(request: EventSearchRequest) -> EventResponse:
    """
    학술행사 검색 API
    키워드, 날짜, 장소로 필터링
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
        logger.error(f"Event search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/events/upcoming")
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


@app.get("/v1/events/registration-open")
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


@app.get("/v1/events/{event_name}")
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

def run_server(host: str = "0.0.0.0", port: int = 8080):
    """서버 실행"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
