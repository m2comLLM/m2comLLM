"""
LangGraph 기반 의료 RAG 에이전트
DeepSeek-R1의 Chain of Thought 추론 활용
"""

from typing import TypedDict, Annotated, List, Optional, Any
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from loguru import logger
import httpx
import json
import re

from src.config import settings


# 상태 정의
class AgentState(TypedDict):
    """에이전트 상태"""
    messages: Annotated[List, add_messages]
    current_patient_id: Optional[str]
    thinking: Optional[str]  # <think> 과정 저장
    tool_calls: List[dict]
    iteration: int
    max_iterations: int


# 시스템 프롬프트
SYSTEM_PROMPT = """당신은 의료 전문 AI 어시스턴트입니다. 의료진이 환자를 진료하는 데 도움을 주는 역할을 합니다.

## 핵심 원칙
1. **Reasoning-First**: 모든 질문에 즉답하지 말고, 반드시 <think> 태그 안에서 추론 과정을 거친 후 답변하세요.
2. **근거 기반**: 임상 가이드라인과 최신 의학 근거를 기반으로 답변하세요.
3. **안전 우선**: 약물 상호작용, 금기사항 등 안전 관련 정보를 항상 확인하세요.

## 사용 가능한 도구
1. **search_clinical_guidelines**: 임상 가이드라인 및 의료 문헌 검색
2. **get_patient_info**: 환자 기본 정보 조회
3. **get_patient_vitals**: 환자 바이탈 사인 조회
4. **get_patient_diagnoses**: 환자 진단 이력 조회
5. **get_patient_medications**: 환자 복용 약물 조회
6. **get_patient_lab_results**: 환자 검사 결과 조회
7. **get_patient_allergies**: 환자 알레르기 정보 조회
8. **get_drug_info**: 약물 상세 정보 조회
9. **check_drug_interaction**: 약물 상호작용 확인
10. **check_multiple_interactions**: 다중 약물 상호작용 확인

## 추론 과정 예시
<think>
1. 환자 정보 확인 필요 → get_patient_info 호출
2. 현재 복용 약물 확인 → get_patient_medications 호출
3. 새 처방약과 상호작용 확인 → check_drug_interaction 호출
4. 관련 가이드라인 검색 → search_clinical_guidelines 호출
5. 종합 분석 및 권고안 작성
</think>

반드시 <think> 태그로 추론 과정을 시작하고, 필요한 도구를 순차적으로 호출한 후 최종 답변을 제공하세요.
"""


class MedicalAgent:
    """의료 RAG 에이전트"""

    def __init__(
        self,
        model_url: str = None,
        model_name: str = None,
        max_iterations: int = 10,
    ):
        self.model_url = model_url or settings.vllm_base_url
        self.model_name = model_name or settings.vllm_model
        self.max_iterations = max_iterations
        self.graph = self._build_graph()

        # MCP 도구 등록
        self.tools = self._register_tools()

    def _register_tools(self) -> dict:
        """MCP 도구 등록"""
        return {
            # Clinical MCP
            "search_clinical_guidelines": self._search_clinical_guidelines,
            "search_by_disease": self._search_by_disease,
            "search_treatment_protocol": self._search_treatment_protocol,
            # Patient MCP
            "get_patient_info": self._get_patient_info,
            "get_patient_vitals": self._get_patient_vitals,
            "get_patient_diagnoses": self._get_patient_diagnoses,
            "get_patient_medications": self._get_patient_medications,
            "get_patient_lab_results": self._get_patient_lab_results,
            "get_patient_allergies": self._get_patient_allergies,
            # Drug MCP
            "get_drug_info": self._get_drug_info,
            "check_drug_interaction": self._check_drug_interaction,
            "check_multiple_interactions": self._check_multiple_interactions,
            "check_contraindications": self._check_contraindications,
        }

    def _build_graph(self) -> StateGraph:
        """LangGraph 그래프 구성"""
        workflow = StateGraph(AgentState)

        # 노드 추가
        workflow.add_node("reason", self._reason_node)
        workflow.add_node("execute_tools", self._execute_tools_node)
        workflow.add_node("generate_response", self._generate_response_node)

        # 엣지 설정
        workflow.set_entry_point("reason")

        workflow.add_conditional_edges(
            "reason",
            self._should_execute_tools,
            {
                "execute_tools": "execute_tools",
                "generate_response": "generate_response",
            }
        )

        workflow.add_conditional_edges(
            "execute_tools",
            self._should_continue,
            {
                "reason": "reason",
                "generate_response": "generate_response",
            }
        )

        workflow.add_edge("generate_response", END)

        return workflow.compile()

    async def _call_llm(self, messages: List[dict]) -> str:
        """vLLM 서버 호출"""
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.model_url}/v1/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 4096,
                },
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

    def _parse_tool_calls(self, content: str) -> List[dict]:
        """응답에서 도구 호출 파싱"""
        tool_calls = []

        # 도구 호출 패턴: [tool_name](args) 또는 TOOL: tool_name(args)
        patterns = [
            r'\[(\w+)\]\((\{.*?\})\)',  # [tool_name]({"arg": "value"})
            r'TOOL:\s*(\w+)\((\{.*?\})\)',  # TOOL: tool_name({"arg": "value"})
            r'<tool>(\w+)</tool>\s*<args>(\{.*?\})</args>',  # XML 스타일
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                tool_name, args_str = match
                try:
                    args = json.loads(args_str)
                    tool_calls.append({"name": tool_name, "arguments": args})
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse tool args: {args_str}")

        return tool_calls

    def _extract_thinking(self, content: str) -> tuple[str, str]:
        """<think> 태그에서 추론 과정 추출"""
        think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            # <think> 태그 제거한 나머지
            rest = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            return thinking, rest
        return "", content

    async def _reason_node(self, state: AgentState) -> AgentState:
        """추론 노드: LLM 호출하여 추론"""
        logger.info(f"Reason node - iteration {state['iteration']}")

        # 메시지 구성
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, ToolMessage):
                messages.append({"role": "tool", "content": msg.content, "name": msg.name})

        # LLM 호출
        response = await self._call_llm(messages)

        # 추론 과정 추출
        thinking, rest = self._extract_thinking(response)

        # 도구 호출 파싱
        tool_calls = self._parse_tool_calls(response)

        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=response)],
            "thinking": thinking,
            "tool_calls": tool_calls,
            "iteration": state["iteration"] + 1,
        }

    def _should_execute_tools(self, state: AgentState) -> str:
        """도구 실행 여부 결정"""
        if state["tool_calls"]:
            return "execute_tools"
        return "generate_response"

    async def _execute_tools_node(self, state: AgentState) -> AgentState:
        """도구 실행 노드"""
        logger.info(f"Executing {len(state['tool_calls'])} tools")

        tool_results = []
        for tool_call in state["tool_calls"]:
            tool_name = tool_call["name"]
            args = tool_call["arguments"]

            if tool_name in self.tools:
                try:
                    result = await self.tools[tool_name](**args)
                    tool_results.append(ToolMessage(
                        content=result,
                        name=tool_name,
                    ))
                    logger.info(f"Tool {tool_name} executed successfully")
                except Exception as e:
                    tool_results.append(ToolMessage(
                        content=f"Error: {str(e)}",
                        name=tool_name,
                    ))
                    logger.error(f"Tool {tool_name} failed: {e}")
            else:
                tool_results.append(ToolMessage(
                    content=f"Unknown tool: {tool_name}",
                    name=tool_name,
                ))

        return {
            **state,
            "messages": state["messages"] + tool_results,
            "tool_calls": [],
        }

    def _should_continue(self, state: AgentState) -> str:
        """계속 추론할지 결정"""
        if state["iteration"] >= state["max_iterations"]:
            logger.warning("Max iterations reached")
            return "generate_response"
        return "reason"

    async def _generate_response_node(self, state: AgentState) -> AgentState:
        """최종 응답 생성 노드"""
        logger.info("Generating final response")

        # 이미 응답이 있으면 그대로 반환
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage):
            # <think> 태그 제거
            _, clean_response = self._extract_thinking(last_message.content)
            return {
                **state,
                "messages": state["messages"][:-1] + [AIMessage(content=clean_response)],
            }

        return state

    async def run(self, query: str, patient_id: str = None) -> str:
        """에이전트 실행"""
        initial_state: AgentState = {
            "messages": [HumanMessage(content=query)],
            "current_patient_id": patient_id,
            "thinking": None,
            "tool_calls": [],
            "iteration": 0,
            "max_iterations": self.max_iterations,
        }

        final_state = await self.graph.ainvoke(initial_state)

        # 마지막 AI 메시지 반환
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage):
                return msg.content

        return "응답을 생성하지 못했습니다."

    # === MCP 도구 구현 (Mock) ===
    # 실제 환경에서는 MCP 클라이언트를 통해 호출

    async def _search_clinical_guidelines(self, query: str, top_k: int = 5, **kwargs) -> str:
        """임상 가이드라인 검색"""
        from src.retrieval.reranker import HybridSearchWithRerank
        try:
            searcher = HybridSearchWithRerank()
            results = searcher.search(query=query, top_k=top_k)
            return self._format_search_results(results)
        except Exception as e:
            return f"검색 오류: {str(e)}"

    async def _search_by_disease(self, disease_name: str, aspect: str = None, **kwargs) -> str:
        query = f"{disease_name} {aspect or ''} 가이드라인".strip()
        return await self._search_clinical_guidelines(query)

    async def _search_treatment_protocol(self, condition: str, patient_context: str = None, **kwargs) -> str:
        query = f"{condition} 치료 프로토콜 {patient_context or ''}".strip()
        return await self._search_clinical_guidelines(query)

    async def _get_patient_info(self, patient_id: str, **kwargs) -> str:
        # Mock 데이터
        return f"""환자 정보 (ID: {patient_id})
- 이름: 홍길동
- 나이: 45세
- 성별: 남성
- 혈액형: A+
- 진료과: 내과"""

    async def _get_patient_vitals(self, patient_id: str, limit: int = 10, **kwargs) -> str:
        return f"""바이탈 (ID: {patient_id})
- 혈압: 125/82 mmHg
- 맥박: 75 bpm
- 체온: 36.7°C
- SpO2: 97%"""

    async def _get_patient_diagnoses(self, patient_id: str, **kwargs) -> str:
        return f"""진단 이력 (ID: {patient_id})
- [E11.9] 제2형 당뇨병 (Active)
- [I10] 본태성 고혈압 (Active)"""

    async def _get_patient_medications(self, patient_id: str, **kwargs) -> str:
        return f"""복용 약물 (ID: {patient_id})
- 메트포르민 500mg 1일 2회
- 암로디핀 5mg 1일 1회
- 아스피린 100mg 1일 1회"""

    async def _get_patient_lab_results(self, patient_id: str, test_type: str = None, **kwargs) -> str:
        return f"""검사 결과 (ID: {patient_id})
- Glucose: 145 mg/dL (H)
- HbA1c: 7.2%
- Creatinine: 1.0 mg/dL"""

    async def _get_patient_allergies(self, patient_id: str, **kwargs) -> str:
        return f"""알레르기 (ID: {patient_id})
- 페니실린: 발진, 두드러기 (중등도)"""

    async def _get_drug_info(self, drug_name: str, **kwargs) -> str:
        return f"""약물 정보: {drug_name}
적응증, 금기, 부작용 정보는 Drug MCP 참조"""

    async def _check_drug_interaction(self, drug1: str, drug2: str, **kwargs) -> str:
        return f"""상호작용 확인: {drug1} + {drug2}
상세 정보는 Drug MCP 참조"""

    async def _check_multiple_interactions(self, drugs: List[str], **kwargs) -> str:
        return f"""다중 상호작용 확인: {', '.join(drugs)}
상세 정보는 Drug MCP 참조"""

    async def _check_contraindications(self, drug_name: str, conditions: List[str], **kwargs) -> str:
        return f"""금기 확인: {drug_name}
환자 상태: {', '.join(conditions)}
상세 정보는 Drug MCP 참조"""

    def _format_search_results(self, results: List[dict]) -> str:
        """검색 결과 포맷팅"""
        if not results:
            return "검색 결과가 없습니다."

        formatted = []
        for i, doc in enumerate(results, 1):
            content = doc.get("content", "")[:500]
            score = doc.get("rerank_score", doc.get("hybrid_score", 0))
            formatted.append(f"[{i}] (관련도: {score:.3f})\n{content}")

        return "\n\n---\n\n".join(formatted)


# 팩토리 함수
def create_agent(**kwargs) -> MedicalAgent:
    """에이전트 생성"""
    return MedicalAgent(**kwargs)
