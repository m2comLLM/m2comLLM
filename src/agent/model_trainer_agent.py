"""
Model Trainer Agent
LLM 학습 스크립트 작성 및 학습 모니터링을 담당하는 에이전트
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class TrainingConfig:
    """학습 설정"""
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    data_path: str = "train_data.jsonl"
    output_dir: str = "./output"

    # LoRA
    lora_rank: int = 32
    lora_alpha: int = 64
    target_modules: List[str] = None

    # 학습 파라미터
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 2048

    # 양자화
    quantize: Optional[str] = None  # 4bit, 8bit

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


@dataclass
class TrainingStatus:
    """학습 상태"""
    status: str  # preparing, training, completed, failed
    current_epoch: int = 0
    total_epochs: int = 0
    current_step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    gpu_memory_used: float = 0.0
    elapsed_time: float = 0.0
    eta_seconds: float = 0.0


class ModelTrainerAgent:
    """
    Model Trainer Agent

    역할:
    - 학습 스크립트 생성 및 설정
    - GPU 리소스 모니터링
    - 학습 진행 상황 보고
    - 체크포인트 관리

    사용 스킬:
    - hf-trainer: Hugging Face 기반 학습
    - gpu-task-manager: GPU 리소스 관리
    """

    SYSTEM_PROMPT = """당신은 LLM 학습 전문가입니다.

## 역할
- LoRA/QLoRA 파인튜닝 설정 및 실행
- GPU 리소스 최적화
- 학습 진행 모니터링

## 원칙
1. 메모리 효율: QLoRA, gradient checkpointing 활용
2. 안정적 학습: 적절한 learning rate, warmup 설정
3. 체크포인트: 정기적 저장으로 손실 방지

## 권장 설정 (7B 모델 기준)
- LoRA rank: 32-64
- Learning rate: 1e-4 ~ 3e-4
- Batch size: 4 (gradient accumulation 4-8)
- 양자화: 24GB VRAM 이하시 QLoRA 권장
"""

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.training_status: Optional[TrainingStatus] = None

    def generate_training_script(self) -> str:
        """학습 스크립트 생성"""
        script = f'''#!/bin/bash
# Auto-generated training script
# Generated at: {datetime.now().isoformat()}

# 환경 변수
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# 학습 실행
python -m skills.hf_trainer.train_lora \\
    --model "{self.config.model_name}" \\
    --data "{self.config.data_path}" \\
    --output "{self.config.output_dir}" \\
    --lora-rank {self.config.lora_rank} \\
    --lora-alpha {self.config.lora_alpha} \\
    --lr {self.config.learning_rate} \\
    --epochs {self.config.num_epochs} \\
    --batch-size {self.config.batch_size} \\
    --max-seq-length {self.config.max_seq_length} \\
    {f"--quantize {self.config.quantize}" if self.config.quantize else ""}

echo "Training completed!"
'''
        return script

    def generate_config_yaml(self) -> str:
        """학습 설정 YAML 생성"""
        import yaml

        config_dict = {
            "model": {
                "name": self.config.model_name,
                "quantization": self.config.quantize,
            },
            "lora": {
                "rank": self.config.lora_rank,
                "alpha": self.config.lora_alpha,
                "target_modules": self.config.target_modules,
                "dropout": 0.1,
            },
            "training": {
                "epochs": self.config.num_epochs,
                "batch_size": self.config.batch_size,
                "gradient_accumulation_steps": 4,
                "learning_rate": self.config.learning_rate,
                "warmup_ratio": 0.1,
                "max_seq_length": self.config.max_seq_length,
                "fp16": False,
                "bf16": True,
            },
            "data": {
                "path": self.config.data_path,
                "format": "alpaca",
            },
            "output": {
                "dir": self.config.output_dir,
                "save_steps": 100,
                "save_total_limit": 3,
            },
        }

        return yaml.dump(config_dict, allow_unicode=True, default_flow_style=False)

    async def estimate_resources(self) -> Dict[str, Any]:
        """필요 리소스 추정"""
        # 모델 크기 추정
        model_size = self.config.model_name.lower()
        if "7b" in model_size or "8b" in model_size:
            param_count = 7e9
        elif "13b" in model_size:
            param_count = 13e9
        elif "32b" in model_size or "34b" in model_size:
            param_count = 32e9
        elif "70b" in model_size:
            param_count = 70e9
        else:
            param_count = 7e9  # 기본값

        # 메모리 추정 (GB)
        bytes_per_param = 2 if self.config.quantize == "4bit" else 2  # bf16

        # 모델 메모리
        model_memory = param_count * bytes_per_param / (1024**3)

        # LoRA 메모리
        lora_memory = param_count * 0.01 * self.config.lora_rank / 32

        # 활성화 메모리 (배치 크기에 비례)
        activation_memory = self.config.batch_size * self.config.max_seq_length * 4096 * 4 / (1024**3)

        # 양자화 적용
        if self.config.quantize == "4bit":
            model_memory *= 0.25
        elif self.config.quantize == "8bit":
            model_memory *= 0.5

        total_memory = model_memory + lora_memory + activation_memory

        return {
            "model_params": f"{param_count/1e9:.1f}B",
            "estimated_vram_gb": round(total_memory, 1),
            "recommended_gpu": self._recommend_gpu(total_memory),
            "batch_size": self.config.batch_size,
            "quantization": self.config.quantize or "none",
            "lora_rank": self.config.lora_rank,
        }

    def _recommend_gpu(self, required_vram: float) -> str:
        """GPU 권장사항"""
        if required_vram <= 8:
            return "RTX 3070/4070 (8GB)"
        elif required_vram <= 12:
            return "RTX 3080/4070Ti (12GB)"
        elif required_vram <= 16:
            return "RTX 4080 (16GB)"
        elif required_vram <= 24:
            return "RTX 3090/4090 (24GB)"
        elif required_vram <= 48:
            return "A6000 (48GB)"
        elif required_vram <= 80:
            return "A100 (80GB)"
        else:
            return "Multi-GPU 또는 DeepSpeed 필요"

    async def check_gpu_status(self) -> Dict[str, Any]:
        """GPU 상태 확인"""
        try:
            from skills.gpu_task_manager.gpu_manager import GPUMonitor
            monitor = GPUMonitor()
            gpus = monitor.get_gpu_status()

            return {
                "available": True,
                "gpu_count": len(gpus),
                "gpus": [
                    {
                        "index": g.index,
                        "name": g.name,
                        "memory_used_gb": round(g.memory_used_gb, 1),
                        "memory_total_gb": round(g.memory_total_gb, 1),
                        "utilization": g.utilization,
                    }
                    for g in gpus
                ],
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e),
            }

    async def start_training(self) -> str:
        """학습 시작 (실제로는 스크립트 생성)"""
        # 설정 파일 저장
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 스크립트 저장
        script_path = output_dir / "train.sh"
        script_path.write_text(self.generate_training_script())

        # 설정 저장
        config_path = output_dir / "config.yaml"
        config_path.write_text(self.generate_config_yaml())

        return f"""학습 준비 완료:

## 생성된 파일
- 스크립트: {script_path}
- 설정: {config_path}

## 학습 실행 방법
```bash
chmod +x {script_path}
{script_path}
```

또는 직접 실행:
```bash
python -m skills.hf_trainer.train_lora \\
    --model "{self.config.model_name}" \\
    --data "{self.config.data_path}" \\
    --output "{self.config.output_dir}"
```
"""

    async def handle_task(self, task: str) -> str:
        """에이전트 태스크 처리"""
        task_lower = task.lower()

        if "학습" in task_lower and ("시작" in task_lower or "실행" in task_lower):
            return await self.start_training()

        elif "리소스" in task_lower or "추정" in task_lower:
            resources = await self.estimate_resources()
            return f"""리소스 추정:
- 모델: {resources['model_params']}
- 예상 VRAM: {resources['estimated_vram_gb']}GB
- 권장 GPU: {resources['recommended_gpu']}
- 배치 크기: {resources['batch_size']}
- 양자화: {resources['quantization']}"""

        elif "gpu" in task_lower or "상태" in task_lower:
            status = await self.check_gpu_status()
            if status["available"]:
                gpu_info = "\n".join([
                    f"  GPU {g['index']}: {g['name']} ({g['memory_used_gb']}/{g['memory_total_gb']}GB)"
                    for g in status["gpus"]
                ])
                return f"GPU 상태:\n{gpu_info}"
            else:
                return f"GPU 확인 실패: {status.get('error', 'Unknown error')}"

        elif "설정" in task_lower or "config" in task_lower:
            return f"현재 설정:\n```yaml\n{self.generate_config_yaml()}```"

        else:
            return "지원하지 않는 태스크입니다. 사용 가능: 학습 시작, 리소스 추정, GPU 상태, 설정 확인"


async def main():
    """테스트 실행"""
    agent = ModelTrainerAgent(
        config=TrainingConfig(
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            data_path="train_data.jsonl",
            output_dir="./output",
            lora_rank=32,
            quantize="4bit",
        )
    )

    # 리소스 추정
    resources = await agent.estimate_resources()
    print("=== 리소스 추정 ===")
    print(json.dumps(resources, indent=2, ensure_ascii=False))

    # 학습 준비
    result = await agent.start_training()
    print("\n" + result)


if __name__ == "__main__":
    asyncio.run(main())
