"""
GPU Task Manager Script
GPU 리소스 모니터링 및 작업 관리
"""

import os
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import time


@dataclass
class GPUStatus:
    """GPU 상태 정보"""
    index: int
    name: str
    memory_used_gb: float
    memory_total_gb: float
    memory_percent: float
    temperature: int
    utilization: int
    power_draw: int
    power_limit: int


@dataclass
class TrainingTask:
    """학습 작업 정보"""
    task_id: str
    script: str
    gpu_id: int
    status: str  # queued, running, completed, failed
    submitted_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    pid: Optional[int] = None


class GPUMonitor:
    """GPU 모니터링"""

    def __init__(self):
        self._check_nvidia_smi()

    def _check_nvidia_smi(self):
        """nvidia-smi 사용 가능 확인"""
        try:
            subprocess.run(['nvidia-smi'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("nvidia-smi를 찾을 수 없습니다. NVIDIA 드라이버가 설치되어 있는지 확인하세요.")

    def get_gpu_status(self) -> List[GPUStatus]:
        """모든 GPU 상태 조회"""
        try:
            import pynvml
            pynvml.nvmlInit()

            gpus = []
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')

                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                power = pynvml.nvmlDeviceGetPowerUsage(handle) // 1000
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) // 1000

                gpus.append(GPUStatus(
                    index=i,
                    name=name,
                    memory_used_gb=memory.used / (1024**3),
                    memory_total_gb=memory.total / (1024**3),
                    memory_percent=(memory.used / memory.total) * 100,
                    temperature=temp,
                    utilization=util.gpu,
                    power_draw=power,
                    power_limit=power_limit,
                ))

            pynvml.nvmlShutdown()
            return gpus

        except ImportError:
            # pynvml이 없으면 nvidia-smi 파싱
            return self._parse_nvidia_smi()

    def _parse_nvidia_smi(self) -> List[GPUStatus]:
        """nvidia-smi 출력 파싱"""
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,temperature.gpu,utilization.gpu,power.draw,power.limit',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )

        gpus = []
        for line in result.stdout.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 8:
                mem_used = float(parts[2]) / 1024  # MB to GB
                mem_total = float(parts[3]) / 1024

                gpus.append(GPUStatus(
                    index=int(parts[0]),
                    name=parts[1],
                    memory_used_gb=mem_used,
                    memory_total_gb=mem_total,
                    memory_percent=(mem_used / mem_total) * 100,
                    temperature=int(parts[4]),
                    utilization=int(parts[5]),
                    power_draw=int(float(parts[6])),
                    power_limit=int(float(parts[7])),
                ))

        return gpus

    def print_status(self):
        """GPU 상태 출력"""
        gpus = self.get_gpu_status()

        print("=== GPU 상태 ===\n")
        for gpu in gpus:
            print(f"GPU {gpu.index}: {gpu.name}")
            print(f"  - 메모리: {gpu.memory_used_gb:.1f} / {gpu.memory_total_gb:.1f} GB ({gpu.memory_percent:.1f}%)")
            print(f"  - 온도: {gpu.temperature}°C")
            print(f"  - 사용률: {gpu.utilization}%")
            print(f"  - 전력: {gpu.power_draw}W / {gpu.power_limit}W")
            print()

    def find_available_gpu(self, min_memory_gb: float = 10.0) -> Optional[int]:
        """사용 가능한 GPU 찾기"""
        gpus = self.get_gpu_status()

        for gpu in gpus:
            available_gb = gpu.memory_total_gb - gpu.memory_used_gb
            if available_gb >= min_memory_gb and gpu.utilization < 50:
                return gpu.index

        return None


class BatchSizeOptimizer:
    """최적 배치 크기 계산"""

    # 모델별 파라미터 수 (approximate)
    MODEL_PARAMS = {
        '1B': 1e9,
        '3B': 3e9,
        '7B': 7e9,
        '13B': 13e9,
        '32B': 32e9,
        '70B': 70e9,
    }

    def calculate_memory_per_sample(
        self,
        model_size: str,
        seq_length: int,
        dtype: str = 'bf16',
        lora_rank: int = 32,
    ) -> float:
        """샘플당 메모리 사용량 추정 (GB)"""
        params = self.MODEL_PARAMS.get(model_size, 7e9)

        # 바이트 크기
        dtype_bytes = 2 if dtype in ['bf16', 'fp16'] else 4

        # 모델 메모리 (양자화 고려 안함)
        model_memory = params * dtype_bytes / (1024**3)

        # LoRA 추가 메모리
        lora_memory = params * 0.01 * lora_rank / 32 * dtype_bytes / (1024**3)

        # 활성화 메모리 (대략적 추정)
        hidden_size = int((params / 1e9) ** 0.5 * 4096)  # 근사치
        activation_memory = seq_length * hidden_size * dtype_bytes * 4 / (1024**3)

        # 그래디언트 메모리
        gradient_memory = activation_memory * 0.5

        return model_memory + lora_memory + activation_memory + gradient_memory

    def suggest_batch_size(
        self,
        available_memory_gb: float,
        model_size: str,
        seq_length: int = 2048,
        dtype: str = 'bf16',
        lora_rank: int = 32,
    ) -> int:
        """최적 배치 크기 제안"""
        memory_per_sample = self.calculate_memory_per_sample(
            model_size, seq_length, dtype, lora_rank
        )

        # 안전 마진 20%
        usable_memory = available_memory_gb * 0.8

        batch_size = int(usable_memory / memory_per_sample)
        return max(1, batch_size)


class TaskQueue:
    """작업 큐 관리"""

    def __init__(self, queue_file: str = ".gpu_task_queue.json"):
        self.queue_file = Path(queue_file)
        self.tasks: List[TrainingTask] = []
        self._load_queue()

    def _load_queue(self):
        """큐 파일 로드"""
        if self.queue_file.exists():
            data = json.loads(self.queue_file.read_text())
            self.tasks = [TrainingTask(**t) for t in data]

    def _save_queue(self):
        """큐 파일 저장"""
        data = [asdict(t) for t in self.tasks]
        self.queue_file.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def submit(self, script: str, gpu_id: int = 0) -> TrainingTask:
        """작업 제출"""
        task = TrainingTask(
            task_id=f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            script=script,
            gpu_id=gpu_id,
            status="queued",
            submitted_at=datetime.now().isoformat(),
        )
        self.tasks.append(task)
        self._save_queue()
        return task

    def list_tasks(self):
        """작업 목록 출력"""
        print("=== 작업 큐 ===\n")
        for task in self.tasks:
            status_icon = {
                'queued': '⏳',
                'running': '🔄',
                'completed': '✅',
                'failed': '❌',
            }.get(task.status, '?')

            print(f"{status_icon} [{task.task_id}] {task.script}")
            print(f"   GPU: {task.gpu_id}, 상태: {task.status}")
            print(f"   제출: {task.submitted_at}")
            print()


def main():
    parser = argparse.ArgumentParser(description='GPU Task Manager')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # status
    subparsers.add_parser('status', help='GPU 상태 확인')

    # submit
    submit_parser = subparsers.add_parser('submit', help='작업 제출')
    submit_parser.add_argument('--script', required=True, help='학습 스크립트')
    submit_parser.add_argument('--gpu', type=int, default=0, help='GPU ID')

    # queue
    subparsers.add_parser('queue', help='작업 큐 확인')

    # optimal-batch
    batch_parser = subparsers.add_parser('optimal-batch', help='최적 배치 크기 계산')
    batch_parser.add_argument('--model', required=True, help='모델 크기 (1B, 7B, 13B, 32B, 70B)')
    batch_parser.add_argument('--seq-length', type=int, default=2048)
    batch_parser.add_argument('--gpu', type=int, default=0, help='GPU ID')

    args = parser.parse_args()

    if args.command == 'status':
        monitor = GPUMonitor()
        monitor.print_status()

    elif args.command == 'submit':
        queue = TaskQueue()
        task = queue.submit(args.script, args.gpu)
        print(f"작업 제출됨: {task.task_id}")

    elif args.command == 'queue':
        queue = TaskQueue()
        queue.list_tasks()

    elif args.command == 'optimal-batch':
        monitor = GPUMonitor()
        optimizer = BatchSizeOptimizer()

        gpus = monitor.get_gpu_status()
        if args.gpu < len(gpus):
            gpu = gpus[args.gpu]
            available = gpu.memory_total_gb - gpu.memory_used_gb

            batch_size = optimizer.suggest_batch_size(
                available_memory_gb=available,
                model_size=args.model,
                seq_length=args.seq_length,
            )

            print(f"=== 최적 배치 크기 ===")
            print(f"GPU {args.gpu}: {gpu.name}")
            print(f"가용 메모리: {available:.1f} GB")
            print(f"모델: {args.model}")
            print(f"시퀀스 길이: {args.seq_length}")
            print(f"추천 배치 크기: {batch_size}")
        else:
            print(f"GPU {args.gpu}을 찾을 수 없습니다.")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
