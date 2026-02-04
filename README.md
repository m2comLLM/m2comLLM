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
