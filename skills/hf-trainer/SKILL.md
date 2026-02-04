---
name: hf-trainer
description: Hugging Face 라이브러리를 통해 LoRA/QLoRA 파인튜닝을 수행합니다. (6GB VRAM 최적화)
user-invocable: true
disable-model-invocation: true
---

# HF Trainer 스킬

Hugging Face Transformers와 PEFT를 사용한 효율적인 LLM 파인튜닝을 수행합니다.
**6GB VRAM (GTX 1660 등)에서도 학습 가능하도록 최적화되어 있습니다.**

## 지원 모델 (VRAM별)

| VRAM | 권장 모델 | 비고 |
|------|----------|------|
| 4GB | Qwen/Qwen2.5-0.5B-Instruct | 한국어 지원 |
| 6GB | Qwen/Qwen2.5-1.5B-Instruct | 한국어 지원, **기본값** |
| 8GB | Qwen/Qwen2.5-3B-Instruct | 한국어 지원 |
| 12GB+ | LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct | 한국어 특화 |
| 24GB+ | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 추론 능력 우수 |

## 사용법

```bash
# 6GB VRAM (GTX 1660 SUPER) - 기본 설정
python skills/hf-trainer/train_lora.py \
    --data train_data.jsonl

# 더 작은 모델 사용 (4GB VRAM)
python skills/hf-trainer/train_lora.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --data train_data.jsonl \
    --lora-rank 8

# 더 큰 모델 (24GB VRAM)
python skills/hf-trainer/train_lora.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --data train_data.jsonl \
    --lora-rank 32 \
    --batch-size 4 \
    --max-seq-length 2048

# 학습 재개
python skills/hf-trainer/train_lora.py \
    --data train_data.jsonl \
    --resume ./output/checkpoint-500
```

## 주요 파라미터 (6GB 기본값)

| 파라미터 | 6GB 기본값 | 24GB 권장 | 설명 |
|---------|-----------|----------|------|
| --model | Qwen2.5-1.5B | DeepSeek-7B | 모델명 |
| --lora-rank | 16 | 32-64 | LoRA rank |
| --batch-size | 1 | 4-8 | 배치 크기 |
| --max-seq-length | 512 | 2048 | 최대 시퀀스 |
| --quantize | 4bit | 4bit/none | 양자화 |

## 의존성

```bash
pip install transformers peft bitsandbytes accelerate trl datasets
```
