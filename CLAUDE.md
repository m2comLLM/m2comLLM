# Medical RAG Agent

온프레미스 환경에서 구동되는 의료 특화 추론형 RAG 에이전트

## 개요

외부 클라우드 없이 100% 로컬에서 동작하며, DeepSeek-R1-Distill 모델의 Chain of Thought 추론 능력을 활용합니다.

## 기술 스택

| 구성요소 | 기술 |
|---------|-----|
| LLM Engine | vLLM (DeepSeek-R1-Distill-Qwen-32B) |
| Orchestration | LangGraph |
| Protocol | MCP (Model Context Protocol) |
| Vector DB | Milvus |
| Search | Elasticsearch (BM25) |
| Interface | Open WebUI |

## 설치

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경변수 설정

`.env` 파일을 생성하고 필요한 설정을 추가합니다:

```bash
# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Elasticsearch
ES_HOST=localhost
ES_PORT=9200

# vLLM
VLLM_BASE_URL=http://localhost:8000
VLLM_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B

# Embedding
EMBEDDING_MODEL=BAAI/bge-m3
```

### 3. 인프라 실행

```bash
cd docker && docker-compose up -d
```

## 사용법

### 데이터 인덱싱

```bash
# 단일 파일
python scripts/index_data.py data/your_file.xlsx

# 디렉토리 전체
python scripts/index_data.py data/
```

### 검색 테스트

```bash
python scripts/search_test.py "검색 쿼리"
python scripts/search_test.py "검색 쿼리" --top-k 10 --no-rerank
```

### MCP 서버 테스트

```bash
python scripts/test_mcp.py
```

### Open WebUI 연동

```bash
# 1. vLLM 서버 실행 (호스트에서)
python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --port 8000

# 2. 전체 스택 실행 (Docker)
cd docker && docker-compose up -d

# 3. Open WebUI 접속
# http://localhost:3000
```

### API 서버만 실행

```bash
# 개발 모드
uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload

# 또는
python -m src.api.main
```

## 아키텍처

```
User → Open WebUI → LangGraph Agent → vLLM (DeepSeek-R1)
                          ↓
                    MCP Protocol
            ↙         ↓         ↘
   Clinical MCP   Patient MCP   Drug MCP
   (Milvus)       (SQL/EMR)     (Drug DB)
```

## 핵심 원칙

- **Zero Data Leakage**: 모든 데이터 처리는 사내망 내에서만
- **Reasoning-First**: 즉답 금지, 반드시 `<think>` 과정 후 응답
- **Modular Connectivity**: 새 데이터 소스는 MCP 서버 추가로 연결

## API 엔드포인트

| 엔드포인트 | 설명 |
|-----------|------|
| `POST /v1/chat/completions` | OpenAI 호환 채팅 API |
| `GET /v1/models` | 사용 가능한 모델 목록 |
| `POST /v1/search` | 임상 가이드라인 검색 |
| `POST /v1/drug-interactions` | 약물 상호작용 확인 |
| `POST /v1/patient/summary` | 환자 요약 정보 |

## 포트 구성

| 서비스 | 포트 | 설명 |
|--------|------|------|
| Open WebUI | 3000 | 웹 인터페이스 |
| Medical API | 8080 | OpenAI 호환 API |
| vLLM | 8000 | LLM 추론 서버 |
| Milvus | 19530 | 벡터 DB |
| Elasticsearch | 9200 | BM25 검색 |
| MinIO Console | 9001 | 오브젝트 스토리지 |

---

## LLM 개발 에이전트

내부 LLM 개발을 위한 3개의 핵심 에이전트가 포함되어 있습니다.

### 에이전트 역할

| 에이전트 | 역할 | 사용 스킬 |
|---------|------|----------|
| Data Engineer Agent | 데이터 수집/정제/변환 | data-extractor, pii-scrubber |
| Model Trainer Agent | 학습 스크립트/모니터링 | hf-trainer, gpu-task-manager |
| Evaluator Agent | 품질 검증/벤치마킹 | llm-judge, latency-monitor |

### 워크플로우

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Data Engineer   │ ──▶ │  Model Trainer   │ ──▶ │    Evaluator     │
│     Agent        │     │     Agent        │     │     Agent        │
└──────────────────┘     └──────────────────┘     └──────────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
   train_data.jsonl        adapter_model/          eval_report.md
```

### 스킬 (Skills)

#### 1. data-extractor
사내 문서를 LLM 학습용 JSONL로 변환
```bash
python skills/data-extractor/extract_data.py data/ --output train_data.jsonl
```

#### 2. pii-scrubber
개인정보(PII) 탐지 및 마스킹
```bash
python skills/pii-scrubber/scrub_pii.py input.txt --output cleaned.txt
```

#### 3. hf-trainer
Hugging Face 기반 LoRA/QLoRA 파인튜닝
```bash
python skills/hf-trainer/train_lora.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --data train_data.jsonl \
    --quantize 4bit
```

#### 4. gpu-task-manager
GPU 리소스 모니터링 및 작업 관리
```bash
python skills/gpu-task-manager/gpu_manager.py status
python skills/gpu-task-manager/gpu_manager.py optimal-batch --model 7B
```

#### 5. llm-judge
LLM을 심판으로 활용한 답변 품질 평가
```bash
python skills/llm-judge/eval_judge.py single \
    --question "질문" --answer "답변"
```

#### 6. latency-monitor
추론 지연시간/처리량 벤치마킹
```bash
python skills/latency-monitor/latency_monitor.py benchmark --output report.json
```

### 전체 파이프라인 실행

```python
from src.agent.agent_workflow import run_llm_development_pipeline

result = await run_llm_development_pipeline(
    data_path="data/",
    model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    output_dir="./output"
)
```

### MCP 서버 연동

| MCP 서버 | 용도 |
|---------|------|
| Clinical MCP | 학술행사/가이드라인 검색 (Milvus) |
| Postgres MCP | 학습 데이터 DB 직접 쿼리 |
| Git MCP | 학습 스크립트 버전 관리 |
| Fetch MCP | 최신 LLM 논문/가이드 검색 |
