"""
Drug MCP Server
약물 정보 및 상호작용 데이터 조회
"""

from typing import Any, List
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from loguru import logger


# MCP 서버 인스턴스
server = Server("drug-mcp")

# 약물 데이터베이스 (실제 환경에서는 외부 DB 사용)
DRUG_DATABASE = {
    "metformin": {
        "name_ko": "메트포르민",
        "name_en": "Metformin",
        "class": "비구아나이드계 당뇨병 치료제",
        "indications": ["제2형 당뇨병"],
        "contraindications": ["신부전", "급성 대사성 산증", "탈수"],
        "side_effects": ["오심", "설사", "복통", "젖산산증(드물게)"],
        "dosage": "500-2000mg/일, 식사와 함께 복용",
        "interactions": {
            "alcohol": {"severity": "중등도", "effect": "젖산산증 위험 증가"},
            "contrast_media": {"severity": "심각", "effect": "급성 신손상 위험"},
            "cimetidine": {"severity": "경미", "effect": "메트포르민 혈중농도 증가"},
        },
    },
    "amlodipine": {
        "name_ko": "암로디핀",
        "name_en": "Amlodipine",
        "class": "칼슘채널차단제",
        "indications": ["고혈압", "협심증"],
        "contraindications": ["심인성 쇼크", "중증 대동맥 협착"],
        "side_effects": ["부종", "두통", "홍조", "피로"],
        "dosage": "5-10mg/일, 1일 1회",
        "interactions": {
            "simvastatin": {"severity": "중등도", "effect": "근육병증 위험 증가"},
            "cyclosporine": {"severity": "중등도", "effect": "암로디핀 혈중농도 증가"},
        },
    },
    "aspirin": {
        "name_ko": "아스피린",
        "name_en": "Aspirin",
        "class": "비스테로이드 항염증제 / 항혈소판제",
        "indications": ["심혈관 질환 예방", "해열", "진통"],
        "contraindications": ["활동성 출혈", "출혈성 질환", "아스피린 과민반응"],
        "side_effects": ["위장장애", "출혈 경향", "이명"],
        "dosage": "심혈관 예방: 75-100mg/일",
        "interactions": {
            "warfarin": {"severity": "심각", "effect": "출혈 위험 증가"},
            "ibuprofen": {"severity": "중등도", "effect": "아스피린 항혈소판 효과 감소"},
        },
    },
    "warfarin": {
        "name_ko": "와파린",
        "name_en": "Warfarin",
        "class": "항응고제",
        "indications": ["심방세동", "심부정맥혈전증", "폐색전증"],
        "contraindications": ["활동성 출혈", "임신", "중증 간질환"],
        "side_effects": ["출혈", "피부 괴사(드물게)"],
        "dosage": "INR 목표에 따라 개별 조절",
        "interactions": {
            "aspirin": {"severity": "심각", "effect": "출혈 위험 증가"},
            "vitamin_k": {"severity": "심각", "effect": "항응고 효과 감소"},
        },
    },
}

# 약물 상호작용 매트릭스
INTERACTION_MATRIX = {
    ("aspirin", "warfarin"): {
        "severity": "심각",
        "mechanism": "항혈소판 + 항응고 효과 중복",
        "effect": "출혈 위험 현저히 증가",
        "recommendation": "병용 시 출혈 증상 모니터링 필수",
    },
    ("metformin", "contrast_media"): {
        "severity": "심각",
        "mechanism": "조영제로 인한 급성 신손상 시 메트포르민 축적",
        "effect": "젖산산증 위험",
        "recommendation": "조영제 투여 48시간 전후 메트포르민 중단",
    },
    ("amlodipine", "simvastatin"): {
        "severity": "중등도",
        "mechanism": "CYP3A4 억제로 심바스타틴 혈중농도 증가",
        "effect": "근육병증/횡문근융해증 위험",
        "recommendation": "심바스타틴 20mg/일 초과 금지",
    },
}


@server.list_tools()
async def list_tools() -> list[Tool]:
    """사용 가능한 도구 목록"""
    return [
        Tool(
            name="get_drug_info",
            description="약물의 상세 정보를 조회합니다 (적응증, 금기, 부작용, 용량 등)",
            inputSchema={
                "type": "object",
                "properties": {
                    "drug_name": {
                        "type": "string",
                        "description": "약물명 (영문 또는 한글)",
                    },
                },
                "required": ["drug_name"],
            },
        ),
        Tool(
            name="check_drug_interaction",
            description="두 약물 간의 상호작용을 확인합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "drug1": {"type": "string", "description": "첫 번째 약물명"},
                    "drug2": {"type": "string", "description": "두 번째 약물명"},
                },
                "required": ["drug1", "drug2"],
            },
        ),
        Tool(
            name="check_multiple_interactions",
            description="여러 약물 간의 상호작용을 한번에 확인합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "drugs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "약물명 목록",
                    },
                },
                "required": ["drugs"],
            },
        ),
        Tool(
            name="check_contraindications",
            description="환자 상태에 따른 약물 금기를 확인합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "drug_name": {"type": "string", "description": "약물명"},
                    "conditions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "환자의 현재 상태/질환 목록",
                    },
                },
                "required": ["drug_name", "conditions"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """도구 실행"""
    logger.info(f"Drug MCP tool called: {name}")

    try:
        if name == "get_drug_info":
            return await get_drug_info(arguments["drug_name"])
        elif name == "check_drug_interaction":
            return await check_drug_interaction(arguments["drug1"], arguments["drug2"])
        elif name == "check_multiple_interactions":
            return await check_multiple_interactions(arguments["drugs"])
        elif name == "check_contraindications":
            return await check_contraindications(
                arguments["drug_name"], arguments["conditions"]
            )
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        logger.error(f"Drug MCP error: {e}")
        return [TextContent(type="text", text=f"조회 중 오류 발생: {str(e)}")]


def normalize_drug_name(name: str) -> str:
    """약물명 정규화"""
    name = name.lower().strip()
    ko_to_en = {
        "메트포르민": "metformin",
        "암로디핀": "amlodipine",
        "아스피린": "aspirin",
        "와파린": "warfarin",
    }
    return ko_to_en.get(name, name)


async def get_drug_info(drug_name: str) -> list[TextContent]:
    """약물 정보 조회"""
    normalized = normalize_drug_name(drug_name)
    drug = DRUG_DATABASE.get(normalized)

    if not drug:
        return [TextContent(type="text", text=f"'{drug_name}' 약물 정보를 찾을 수 없습니다.")]

    result = f"""## {drug['name_ko']} ({drug['name_en']})

**약물 분류**: {drug['class']}

### 적응증
{chr(10).join(f'- {i}' for i in drug['indications'])}

### 금기
{chr(10).join(f'- {c}' for c in drug['contraindications'])}

### 부작용
{chr(10).join(f'- {s}' for s in drug['side_effects'])}

### 용량
{drug['dosage']}
"""
    return [TextContent(type="text", text=result)]


async def check_drug_interaction(drug1: str, drug2: str) -> list[TextContent]:
    """약물 상호작용 확인"""
    d1, d2 = normalize_drug_name(drug1), normalize_drug_name(drug2)
    interaction = INTERACTION_MATRIX.get((d1, d2)) or INTERACTION_MATRIX.get((d2, d1))

    if not interaction:
        drug_data = DRUG_DATABASE.get(d1, {})
        if d2 in drug_data.get("interactions", {}):
            interaction = drug_data["interactions"][d2]

    if not interaction:
        return [TextContent(
            type="text",
            text=f"**{drug1}** + **{drug2}**: 알려진 주요 상호작용이 없습니다."
        )]

    severity_map = {"심각": "[심각]", "중등도": "[중등도]", "경미": "[경미]"}
    result = f"""## 약물 상호작용 주의

**{drug1}** + **{drug2}**

- 심각도: {severity_map.get(interaction.get('severity', ''), '')} {interaction.get('severity', 'N/A')}
- 영향: {interaction.get('effect', 'N/A')}
- 권고: {interaction.get('recommendation', '담당의와 상담 필요')}
"""
    return [TextContent(type="text", text=result)]


async def check_multiple_interactions(drugs: List[str]) -> list[TextContent]:
    """다중 약물 상호작용 확인"""
    if len(drugs) < 2:
        return [TextContent(type="text", text="2개 이상의 약물을 입력해주세요.")]

    interactions_found = []
    normalized = [normalize_drug_name(d) for d in drugs]

    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            d1, d2 = normalized[i], normalized[j]
            interaction = INTERACTION_MATRIX.get((d1, d2)) or INTERACTION_MATRIX.get((d2, d1))
            if interaction:
                interactions_found.append((drugs[i], drugs[j], interaction))

    result = f"## 다중 약물 상호작용 분석\n\n**분석 약물**: {', '.join(drugs)}\n\n"

    if not interactions_found:
        result += "주요 상호작용이 발견되지 않았습니다."
    else:
        result += f"**{len(interactions_found)}건의 상호작용 발견**\n\n"
        for d1, d2, info in interactions_found:
            result += f"- **{d1} + {d2}**: {info.get('effect', 'N/A')}\n"

    return [TextContent(type="text", text=result)]


async def check_contraindications(drug_name: str, conditions: List[str]) -> list[TextContent]:
    """금기 확인"""
    normalized = normalize_drug_name(drug_name)
    drug = DRUG_DATABASE.get(normalized)

    if not drug:
        return [TextContent(type="text", text=f"'{drug_name}' 약물 정보를 찾을 수 없습니다.")]

    found = []
    for condition in conditions:
        for contra in drug.get("contraindications", []):
            if condition.lower() in contra.lower() or contra.lower() in condition.lower():
                found.append((condition, contra))

    result = f"## 금기 확인: {drug['name_ko']}\n\n**환자 상태**: {', '.join(conditions)}\n\n"

    if found:
        result += "### 금기 해당\n"
        for condition, contra in found:
            result += f"- **{condition}** -> 금기: {contra}\n"
    else:
        result += "확인된 금기 사항 없음"

    return [TextContent(type="text", text=result)]


async def main():
    """MCP 서버 실행"""
    logger.info("Starting Drug MCP Server...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
