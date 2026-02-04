"""
LoRA/QLoRA Fine-tuning Script
Hugging Face PEFT를 사용한 효율적인 파인튜닝
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import torch
from datetime import datetime


@dataclass
class TrainingConfig:
    """학습 설정"""
    # 6GB VRAM 호환 모델 (기본값: Qwen2.5-1.5B)
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    data_path: str = "train_data.jsonl"
    output_dir: str = "./output"

    # LoRA 설정 (6GB VRAM 최적화)
    lora_rank: int = 16  # 작은 rank로 메모리 절약
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])

    # 학습 설정 (6GB VRAM 최적화)
    num_epochs: int = 3
    batch_size: int = 1  # 작은 배치
    gradient_accumulation_steps: int = 8  # 실효 배치 = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    max_seq_length: int = 512  # 짧은 시퀀스

    # 양자화 (6GB VRAM에서는 4bit 권장)
    quantize: Optional[str] = "4bit"

    # 기타
    seed: int = 42
    logging_steps: int = 10
    save_steps: int = 100
    resume_from: Optional[str] = None


def load_dataset(data_path: str):
    """JSONL 데이터셋 로드"""
    from datasets import Dataset

    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))

    return Dataset.from_list(samples)


def format_instruction(sample: Dict) -> str:
    """Alpaca 형식으로 포맷팅"""
    if sample.get('input'):
        return f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}"""
    else:
        return f"""### Instruction:
{sample['instruction']}

### Response:
{sample['output']}"""


def get_model_and_tokenizer(config: TrainingConfig):
    """모델과 토크나이저 로드"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # 양자화 설정
    bnb_config = None
    if config.quantize == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif config.quantize == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 모델
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not bnb_config else None,
    )

    # 그래디언트 체크포인팅
    model.gradient_checkpointing_enable()

    return model, tokenizer


def setup_lora(model, config: TrainingConfig):
    """LoRA 설정 적용"""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if config.quantize:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def train(config: TrainingConfig):
    """학습 실행"""
    from transformers import TrainingArguments
    from trl import SFTTrainer

    print(f"=== 학습 시작 ===")
    print(f"모델: {config.model_name}")
    print(f"데이터: {config.data_path}")
    print(f"출력: {config.output_dir}")
    print(f"LoRA Rank: {config.lora_rank}")
    print(f"양자화: {config.quantize or 'None'}")

    # 데이터셋 로드
    dataset = load_dataset(config.data_path)
    print(f"데이터셋 크기: {len(dataset)}")

    # 모델 로드
    model, tokenizer = get_model_and_tokenizer(config)

    # LoRA 적용
    model = setup_lora(model, config)

    # 학습 인자
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=3,
        fp16=False,
        bf16=True,
        optim="paged_adamw_32bit",
        report_to="none",
        seed=config.seed,
    )

    # SFT Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        formatting_func=format_instruction,
        max_seq_length=config.max_seq_length,
        packing=False,
    )

    # 학습 재개
    if config.resume_from:
        print(f"체크포인트에서 재개: {config.resume_from}")
        trainer.train(resume_from_checkpoint=config.resume_from)
    else:
        trainer.train()

    # 모델 저장
    final_path = Path(config.output_dir) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    print(f"\n=== 학습 완료 ===")
    print(f"모델 저장: {final_path}")

    # 학습 로그 저장
    log_path = Path(config.output_dir) / "training_log.json"
    log_data = {
        "config": {
            "model_name": config.model_name,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "quantize": config.quantize,
        },
        "dataset_size": len(dataset),
        "completed_at": datetime.now().isoformat(),
    }
    log_path.write_text(json.dumps(log_data, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description='LoRA Fine-tuning (6GB VRAM 최적화)')
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help='모델명 (6GB: Qwen2.5-0.5B/1.5B, 8GB+: Qwen2.5-3B)')
    parser.add_argument('--data', type=str, required=True, help='학습 데이터 경로 (JSONL)')
    parser.add_argument('--output', type=str, default='./output', help='출력 디렉토리')
    parser.add_argument('--lora-rank', type=int, default=16, help='LoRA rank (6GB: 8-16 권장)')
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=1, help='배치 크기 (6GB: 1-2)')
    parser.add_argument('--quantize', choices=['4bit', '8bit'], default='4bit',
                        help='양자화 (6GB VRAM에서 4bit 필수)')
    parser.add_argument('--resume', type=str, default=None, help='체크포인트 경로')
    parser.add_argument('--max-seq-length', type=int, default=512, help='최대 시퀀스 길이 (6GB: 256-512)')

    args = parser.parse_args()

    config = TrainingConfig(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        quantize=args.quantize,
        resume_from=args.resume,
        max_seq_length=args.max_seq_length,
    )

    train(config)


if __name__ == '__main__':
    main()
