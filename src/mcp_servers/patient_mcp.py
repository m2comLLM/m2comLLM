"""
Patient MCP Server
EMR 데이터베이스 조회 (SQL Read-only)
"""

from typing import Any, Optional
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from loguru import logger

from src.config import settings


# MCP 서버 인스턴스
server = Server("patient-mcp")

# DB 연결 (lazy initialization)
_db_connection = None


def get_db_connection():
    """데이터베이스 연결 획득"""
    global _db_connection
    if _db_connection is None:
        try:
            import psycopg2
            _db_connection = psycopg2.connect(
                host=settings.db_host,
                port=settings.db_port,
                database=settings.db_name,
                user=settings.db_user,
                password=settings.db_password,
            )
            _db_connection.set_session(readonly=True)  # Read-only 모드
            logger.info("EMR 데이터베이스 연결 완료")
        except Exception as e:
            logger.error(f"데이터베이스 연결 실패: {e}")
            raise
    return _db_connection


def execute_query(query: str, params: tuple = None) -> list[dict]:
    """쿼리 실행 (Read-only)"""
    # 안전성 검사: SELECT만 허용
    query_upper = query.strip().upper()
    if not query_upper.startswith("SELECT"):
        raise ValueError("READ-ONLY: SELECT 쿼리만 허용됩니다.")

    # 위험한 키워드 차단
    dangerous_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE"]
    for keyword in dangerous_keywords:
        if keyword in query_upper:
            raise ValueError(f"READ-ONLY: {keyword} 명령은 허용되지 않습니다.")

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query, params)
    columns = [desc[0] for desc in cursor.description]
    results = [dict(zip(columns, row)) for row in cursor.fetchall()]
    cursor.close()
    return results


@server.list_tools()
async def list_tools() -> list[Tool]:
    """사용 가능한 도구 목록"""
    return [
        Tool(
            name="get_patient_info",
            description="환자 기본 정보를 조회합니다 (이름, 나이, 성별, 혈액형 등)",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "환자 ID",
                    },
                },
                "required": ["patient_id"],
            },
        ),
        Tool(
            name="get_patient_vitals",
            description="환자의 최근 바이탈 사인을 조회합니다 (혈압, 맥박, 체온, 호흡수 등)",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "환자 ID",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "조회할 기록 수 (기본값: 10)",
                        "default": 10,
                    },
                },
                "required": ["patient_id"],
            },
        ),
        Tool(
            name="get_patient_diagnoses",
            description="환자의 진단 이력을 조회합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "환자 ID",
                    },
                    "active_only": {
                        "type": "boolean",
                        "description": "활성 진단만 조회 (기본값: false)",
                        "default": False,
                    },
                },
                "required": ["patient_id"],
            },
        ),
        Tool(
            name="get_patient_medications",
            description="환자의 현재 복용 중인 약물 목록을 조회합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "환자 ID",
                    },
                    "include_history": {
                        "type": "boolean",
                        "description": "과거 처방 포함 여부 (기본값: false)",
                        "default": False,
                    },
                },
                "required": ["patient_id"],
            },
        ),
        Tool(
            name="get_patient_lab_results",
            description="환자의 검사 결과를 조회합니다 (혈액검사, 소변검사 등)",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "환자 ID",
                    },
                    "test_type": {
                        "type": "string",
                        "description": "검사 종류 (예: CBC, BMP, LFT, UA)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "조회할 기록 수 (기본값: 5)",
                        "default": 5,
                    },
                },
                "required": ["patient_id"],
            },
        ),
        Tool(
            name="get_patient_allergies",
            description="환자의 알레르기 정보를 조회합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "환자 ID",
                    },
                },
                "required": ["patient_id"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """도구 실행"""
    logger.info(f"Patient MCP tool called: {name}")
    patient_id = arguments.get("patient_id")

    try:
        if name == "get_patient_info":
            return await get_patient_info(patient_id)

        elif name == "get_patient_vitals":
            limit = arguments.get("limit", 10)
            return await get_patient_vitals(patient_id, limit)

        elif name == "get_patient_diagnoses":
            active_only = arguments.get("active_only", False)
            return await get_patient_diagnoses(patient_id, active_only)

        elif name == "get_patient_medications":
            include_history = arguments.get("include_history", False)
            return await get_patient_medications(patient_id, include_history)

        elif name == "get_patient_lab_results":
            test_type = arguments.get("test_type")
            limit = arguments.get("limit", 5)
            return await get_patient_lab_results(patient_id, test_type, limit)

        elif name == "get_patient_allergies":
            return await get_patient_allergies(patient_id)

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.error(f"Patient MCP error: {e}")
        return [TextContent(type="text", text=f"조회 중 오류 발생: {str(e)}")]


async def get_patient_info(patient_id: str) -> list[TextContent]:
    """환자 기본 정보 조회"""
    # 실제 구현에서는 DB 쿼리 사용
    # 예시 응답 (DB 미연결시)
    mock_data = {
        "patient_id": patient_id,
        "name": "홍길동",
        "age": 45,
        "gender": "남성",
        "blood_type": "A+",
        "height": 175,
        "weight": 70,
        "admission_date": "2024-01-15",
        "department": "내과",
        "attending_physician": "김의사",
    }

    result = f"""## 환자 정보 (ID: {patient_id})

- 이름: {mock_data['name']}
- 나이: {mock_data['age']}세
- 성별: {mock_data['gender']}
- 혈액형: {mock_data['blood_type']}
- 신장/체중: {mock_data['height']}cm / {mock_data['weight']}kg
- 입원일: {mock_data['admission_date']}
- 진료과: {mock_data['department']}
- 담당의: {mock_data['attending_physician']}
"""
    return [TextContent(type="text", text=result)]


async def get_patient_vitals(patient_id: str, limit: int) -> list[TextContent]:
    """환자 바이탈 조회"""
    mock_vitals = [
        {"date": "2024-01-20 09:00", "bp": "120/80", "hr": 72, "temp": 36.5, "rr": 16, "spo2": 98},
        {"date": "2024-01-20 15:00", "bp": "125/82", "hr": 75, "temp": 36.7, "rr": 18, "spo2": 97},
    ]

    result = f"## 환자 바이탈 (ID: {patient_id})\n\n"
    result += "| 측정시간 | 혈압 | 맥박 | 체온 | 호흡수 | SpO2 |\n"
    result += "|----------|------|------|------|--------|------|\n"

    for v in mock_vitals[:limit]:
        result += f"| {v['date']} | {v['bp']} | {v['hr']} | {v['temp']}°C | {v['rr']} | {v['spo2']}% |\n"

    return [TextContent(type="text", text=result)]


async def get_patient_diagnoses(patient_id: str, active_only: bool) -> list[TextContent]:
    """환자 진단 이력 조회"""
    mock_diagnoses = [
        {"code": "E11.9", "name": "제2형 당뇨병", "date": "2020-03-15", "status": "active"},
        {"code": "I10", "name": "본태성 고혈압", "date": "2019-08-20", "status": "active"},
        {"code": "J06.9", "name": "급성 상기도 감염", "date": "2024-01-10", "status": "resolved"},
    ]

    if active_only:
        mock_diagnoses = [d for d in mock_diagnoses if d["status"] == "active"]

    result = f"## 진단 이력 (ID: {patient_id})\n\n"
    for d in mock_diagnoses:
        status_emoji = "🔴" if d["status"] == "active" else "✅"
        result += f"- {status_emoji} [{d['code']}] {d['name']} (진단일: {d['date']})\n"

    return [TextContent(type="text", text=result)]


async def get_patient_medications(patient_id: str, include_history: bool) -> list[TextContent]:
    """환자 복용 약물 조회"""
    mock_meds = [
        {"name": "메트포르민 500mg", "dosage": "1일 2회", "route": "경구", "status": "active"},
        {"name": "암로디핀 5mg", "dosage": "1일 1회", "route": "경구", "status": "active"},
        {"name": "아스피린 100mg", "dosage": "1일 1회", "route": "경구", "status": "active"},
    ]

    result = f"## 복용 약물 (ID: {patient_id})\n\n"
    for m in mock_meds:
        result += f"- **{m['name']}**: {m['dosage']} ({m['route']})\n"

    return [TextContent(type="text", text=result)]


async def get_patient_lab_results(
    patient_id: str, test_type: Optional[str], limit: int
) -> list[TextContent]:
    """환자 검사 결과 조회"""
    mock_labs = {
        "CBC": [
            {"item": "WBC", "value": "7.2", "unit": "10³/μL", "ref": "4.0-11.0", "flag": ""},
            {"item": "RBC", "value": "4.8", "unit": "10⁶/μL", "ref": "4.5-5.5", "flag": ""},
            {"item": "Hgb", "value": "14.2", "unit": "g/dL", "ref": "13.5-17.5", "flag": ""},
            {"item": "Plt", "value": "250", "unit": "10³/μL", "ref": "150-400", "flag": ""},
        ],
        "BMP": [
            {"item": "Glucose", "value": "145", "unit": "mg/dL", "ref": "70-100", "flag": "H"},
            {"item": "BUN", "value": "18", "unit": "mg/dL", "ref": "7-20", "flag": ""},
            {"item": "Creatinine", "value": "1.0", "unit": "mg/dL", "ref": "0.7-1.3", "flag": ""},
        ],
    }

    result = f"## 검사 결과 (ID: {patient_id})\n\n"

    if test_type and test_type in mock_labs:
        labs = {test_type: mock_labs[test_type]}
    else:
        labs = mock_labs

    for test_name, items in labs.items():
        result += f"### {test_name}\n"
        result += "| 항목 | 결과 | 단위 | 참고치 | 비고 |\n"
        result += "|------|------|------|--------|------|\n"
        for item in items:
            flag = "⚠️" if item["flag"] else ""
            result += f"| {item['item']} | {item['value']} | {item['unit']} | {item['ref']} | {flag} |\n"
        result += "\n"

    return [TextContent(type="text", text=result)]


async def get_patient_allergies(patient_id: str) -> list[TextContent]:
    """환자 알레르기 정보 조회"""
    mock_allergies = [
        {"allergen": "페니실린", "reaction": "발진, 두드러기", "severity": "중등도"},
        {"allergen": "아스피린", "reaction": "호흡곤란", "severity": "심각"},
    ]

    result = f"## 알레르기 정보 (ID: {patient_id})\n\n"

    if not mock_allergies:
        result += "등록된 알레르기 정보가 없습니다."
    else:
        result += "⚠️ **알레르기 주의**\n\n"
        for a in mock_allergies:
            severity_emoji = "🔴" if a["severity"] == "심각" else "🟡"
            result += f"- {severity_emoji} **{a['allergen']}**: {a['reaction']} (심각도: {a['severity']})\n"

    return [TextContent(type="text", text=result)]


async def main():
    """MCP 서버 실행"""
    logger.info("Starting Patient MCP Server...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
