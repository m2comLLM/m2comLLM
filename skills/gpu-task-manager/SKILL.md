---
name: gpu-task-manager
description: GPU 리소스 모니터링 및 학습 작업 관리를 수행합니다.
user-invocable: true
---

# GPU Task Manager 스킬

GPU 상태 모니터링, 작업 큐 관리, 리소스 최적화를 담당합니다.

## 기능

1. **GPU 상태 모니터링**: 메모리 사용량, 온도, 사용률 확인
2. **작업 큐 관리**: 학습 작업 스케줄링 및 우선순위 관리
3. **자동 배치 크기 조정**: 가용 메모리에 따른 최적 배치 크기 계산
4. **OOM 방지**: 메모리 부족 예측 및 경고

## 사용법

```bash
# GPU 상태 확인
python scripts/gpu_manager.py status

# 학습 작업 제출
python scripts/gpu_manager.py submit --script train.py --gpu 0

# 작업 큐 확인
python scripts/gpu_manager.py queue

# 최적 배치 크기 계산
python scripts/gpu_manager.py optimal-batch --model 7B --seq-length 2048
```

## 출력 예시

```
=== GPU 상태 ===
GPU 0: NVIDIA A100 80GB
  - 메모리: 12.5 / 80.0 GB (15.6%)
  - 온도: 45°C
  - 사용률: 85%
  - 전력: 250W / 400W
```

## 의존성

- `pynvml`: NVIDIA GPU 모니터링
- `psutil`: 시스템 리소스 모니터링
