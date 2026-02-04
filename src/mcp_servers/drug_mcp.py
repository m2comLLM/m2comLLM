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
            "simvastatin": {"severity": "중등도", "effect": "근육병증 위험 증가, 심바스타틴 20mg 이하 권장"},
            "cyclosporine": {"severity": "중등도", "effect": "암로디핀 혈중농도 증가"},
            "grapefruit": {"severity": "경미", "effect": "암로디핀 혈중농도 증가"},
        },
    },
    "aspirin": {
        "name_ko": "아스피린",
        "name_en": "Aspirin",
        "class": "비스테로이드 항염증제 / 항혈소판제",
        "indications": ["심혈관 질환 예방", "해열", "진통", "항염"],
        "contraindications": ["활동성 출혈", "출혈성 질환", "아스피린 과민반응"],
        "side_effects": ["위장장애", "출혈 경향", "이명"],
        "dosage": "심혈관 예방: 75-100mg/일",
        "interactions": {
            "warfarin": {"severity": "심각", "effect": "출혈 위험 증가"},
            "ibuprofen": {"severity": "중등도", "effect": "아스피린 항혈소판 효과 감소"},
            "methotrexate": {"severity": "심각", "effect": "메토트렉세이트 독성 증가"},
        },
    },
    "warfarin": {
        "name_ko": "와파린",
        "name_en": "Warfarin",
        "class": "항응고제",
        "indications": ["심방세동", "심부정맥혈전증", "폐색전증", "판막 치환술 후"],
        "contraindications": ["활동성 출혈", "임신", "중증 간질환"],
        "side_effects": ["출혈", "피부 괴사(드물게)"],
        "dosage": "INR 목표에 따라 개별 조절",
        "interactions": {
            "aspirin": {"severity": "심각", "effect": "출혈 위험 증가"},
            "vitamin_k": {"severity": "심각", "effect": "항응고 효과 감소"},
            "antibiotics": {"severity": "중등도", "effect": "INR 변동 가능"},
        },
    },
}

# 약물 상호작용 매트릭스
INTERACTION_MATRIX = {
    ("aspirin", "warfarin"): {
        "severity": "심각",
        "mechanism": "항혈소판 + 항응고 효과 중복",
        "effect": "출혈 위험 현저히 증가",
        "recommendation": "병용 시 출혈 증상 모니터링 필수, 가능하면 대체 약물 고려",
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
                    "drug1": {
                        "type": "string",
                        "description": "첫 번째 약물명",
                    },
                    "drug2": {
                        "type": "string",
                        "description": "두 번째 약물명",
                    },
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
                    "drug_name": {
                        "type": "string",
                        "description": "약물명",
                    },
                    "conditions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "환자의 현재 상태/질환 목록",
                    },
                },
                "required": ["drug_name", "conditions"],
            },
        ),
        Tool(
            name="get_dosage_info",
            description="약물의 용량 정보를 조회합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "drug_name": {
                        "type": "string",
                        "description": "약물명",
                    },
                    "indication": {
                        "type": "string",
                        "description": "적응증 (용량이 적응증에 따라 다른 경우)",
                    },
                },
                "required": ["drug_name"],
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
            return await check_drug_interaction(
                arguments["drug1"], arguments["drug2"]
            )

        elif name == "check_multiple_interactions":
            return await check_multiple_interactions(arguments["drugs"])

        elif name == "check_contraindications":
            return await check_contraindications(
                arguments["drug_name"], arguments["conditions"]
            )

        elif name == "get_dosage_info":
            return await get_dosage_info(
                arguments["drug_name"], arguments.get("indication")
            )

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.error(f"Drug MCP error: {e}")
        return [TextContent(type="text", text=f"조회 중 오류 발생: {str(e)}")]


def normalize_drug_name(name: str) -> str:
    """약물명 정규화"""
    name = name.lower().strip()
    # 한글-영문 매핑
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
        return [TextContent(
            type="text",
            text=f"'{drug_name}' 약물 정보를 찾을 수 없습니다."
        )]

    result = f"""## 💊 {drug['name_ko']} ({drug['name_en']})

**약물 분류**: {drug['class']}

### 적응증
{chr(10).join(f'- {i}' for i in drug['indications'])}

### 금기
{chr(10).join(f'- ⚠️ {c}' for c in drug['contraindications'])}

### 부작용
{chr(10).join(f'- {s}' for s in drug['side_effects'])}

### 용량
{drug['dosage']}

### 주요 상호작용
"""
    for interact_drug, info in drug.get("interactions", {}).items():
        severity_emoji = {"심각": "🔴", "중등도": "🟡", "경미": "🟢"}.get(info["severity"], "⚪")
        result += f"- {severity_emoji} **{interact_drug}**: {info['effect']}\n"

    return [TextContent(type="text", text=result)]


async def check_drug_interaction(drug1: str, drug2: str) -> list[TextContent]:
    """약물 상호작용 확인"""
    d1, d2 = normalize_drug_name(drug1), normalize_drug_name(drug2)

    # 정방향/역방향 모두 확인
    interaction = INTERACTION_MATRIX.get((d1, d2)) or INTERACTION_MATRIX.get((d2, d1))

    # 개별 약물의 상호작용 정보도 확인
    if not interaction:
        drug_data = DRUG_DATABASE.get(d1, {})
        interactions = drug_data.get("interactions", {})
        if d2 in interactions:
            interaction = {
                "severity": interactions[d2]["severity"],
                "effect": interactions[d2]["effect"],
                "mechanism": "상세 정보 없음",
                "recommendation": "주의하여 사용",
            }

    if not interaction:
        return [TextContent(
            type="text",
            text=f"## 약물 상호작용 확인\n\n"
                 f"**{drug1}** + **{drug2}**\n\n"
                 f"✅ 알려진 주요 상호작용이 없습니다.\n\n"
                 f"*단, 데이터베이스에 없는 상호작용이 있을 수 있으니 주의가 필요합니다.*"
        )]

    severity_emoji = {"심각": "🔴", "중등도": "🟡", "경미": "🟢"}.get(
        interaction["severity"], "⚪"
    )

    result = f"""## ⚠️ 약물 상호작용 주의

**{drug1}** + **{drug2}**

### 심각도
{severity_emoji} **{interaction['severity']}**

### 기전
{interaction.get('mechanism', 'N/A')}

### 영향
{interaction['effect']}

### 권고사항
{interaction.get('recommendation', '담당의와 상담 필요')}
"""
    return [TextContent(type="text", text=result)]


async def check_multiple_interactions(drugs: List[str]) -> list[TextContent]:
    """다중 약물 상호작용 확인"""
    if len(drugs) < 2:
        return [TextContent(type="text", text="2개 이상의 약물을 입력해주세요.")]

    interactions_found = []
    normalized_drugs = [normalize_drug_name(d) for d in drugs]

    # 모든 조합 확인
    for i in range(len(normalized_drugs)):
        for j in range(i + 1, len(normalized_drugs)):
            d1, d2 = normalized_drugs[i], normalized_drugs[j]
            interaction = INTERACTION_MATRIX.get((d1, d2)) or INTERACTION_MATRIX.get((d2, d1))

            if not interaction:
                drug_data = DRUG_DATABASE.get(d1, {})
                if d2 in drug_data.get("interactions", {}):
                    interaction = drug_data["interactions"][d2]

            if interaction:
                interactions_found.append({
                    "drug1": drugs[i],
                    "drug2": drugs[j],
                    "info": interaction,
                })

    result = f"## 다중 약물 상호작용 분석\n\n"
    result += f"**분석 약물**: {', '.join(drugs)}\n\n"

    if not interactions_found:
        result += "✅ 주요 상호작용이 발견되지 않았습니다."
    else:
        result += f"⚠️ **{len(interactions_found)}건의 상호작용 발견**\n\n"
        for item in interactions_found:
            severity = item["info"].get("severity", "알 수 없음")
            severity_emoji = {"심각": "🔴", "중등도": "🟡", "경미": "🟢"}.get(severity, "⚪")
            result += f"### {item['drug1']} + {item['drug2']}\n"
            result += f"- 심각도: {severity_emoji} {severity}\n"
            result += f"- 영향: {item['info'].get('effect', 'N/A')}\n\n"

    return [TextContent(type="text", text=result)]


async def check_contraindications(
    drug_name: str, conditions: List[str]
) -> list[TextContent]:
    """금기 확인"""
    normalized = normalize_drug_name(drug_name)
    drug = DRUG_DATABASE.get(normalized)

    if not drug:
        return [TextContent(
            type="text",
            text=f"'{drug_name}' 약물 정보를 찾을 수 없습니다."
        )]

    contraindications = drug.get("contraindications", [])
    found_contraindications = []

    for condition in conditions:
        condition_lower = condition.lower()
        for contra in contraindications:
            if condition_lower in contra.lower() or contra.lower() in condition_lower:
                found_contraindications.append((condition, contra))

    result = f"## 금기 확인: {drug['name_ko']}\n\n"
    result += f"**환자 상태**: {', '.join(conditions)}\n\n"

    if found_contraindications:
        result += "### 🔴 금기 해당\n\n"
        for condition, contra in found_contraindications:
            result += f"- **{condition}** → 금기 사유: {contra}\n"
        result += "\n⚠️ **해당 약물 사용에 주의가 필요합니다.**"
    else:
        result += "### ✅ 확인된 금기 사항 없음\n\n"
        result += "*단, 모든 금기 사항이 확인된 것은 아니므로 주의가 필요합니다.*"

    return [TextContent(type="text", text=result)]


async def get_dosage_info(drug_name: str, indication: str = None) -> list[TextContent]:
    """용량 정보 조회"""
    normalized = normalize_drug_name(drug_name)
    drug = DRUG_DATABASE.get(normalized)

    if not drug:
        return [TextContent(
            type="text",
            text=f"'{drug_name}' 약물 정보를 찾을 수 없습니다."
        )]

    result = f"""## 💊 {drug['name_ko']} 용량 정보

### 표준 용량
{drug['dosage']}

### 적응증
{chr(10).join(f'- {i}' for i in drug['indications'])}

### 주의사항
- 신기능/간기능에 따라 용량 조절 필요할 수 있음
- 고령 환자는 저용량으로 시작 권장
- 개별 환자 상태에 따라 담당의가 용량 조절
"""
    return [TextContent(type="text", text=result)]


async def main():
    """MCP 서버 실행"""
    logger.info("Starting Drug MCP Server...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
