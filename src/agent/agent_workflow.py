"""
Agent Workflow
에이전트 협업 워크플로우 정의
"""

import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from loguru import logger

from .data_engineer_agent import DataEngineerAgent, DataPipelineConfig
from .model_trainer_agent import ModelTrainerAgent, TrainingConfig
from .evaluator_agent import EvaluatorAgent, EvalConfig


class WorkflowStage(Enum):
    """워크플로우 단계"""
    DATA_PREPARATION = "data_preparation"
    TRAINING = "training"
    EVALUATION = "evaluation"
    DEPLOYMENT = "deployment"


@dataclass
class WorkflowConfig:
    """워크플로우 설정"""
    # 데이터 설정
    input_data_path: str = "data/"
    train_data_path: str = "train_data.jsonl"

    # 학습 설정
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    output_dir: str = "./output"
    lora_rank: int = 32
    quantize: Optional[str] = "4bit"

    # 평가 설정
    eval_samples: int = 50
    min_quality_score: float = 7.0

    # vLLM 서버
    vllm_url: str = "http://localhost:8000"


@dataclass
class WorkflowResult:
    """워크플로우 결과"""
    success: bool
    stages_completed: List[str]
    data_samples: int
    quality_score: float
    model_path: str
    duration_seconds: float
    errors: List[str]


class AgentOrchestrator:
    """
    에이전트 오케스트레이터

    3개의 핵심 에이전트를 조율하여 LLM 개발 파이프라인을 자동화합니다.

    워크플로우:
    1. Data Engineer Agent: 데이터 수집/정제
    2. Model Trainer Agent: 학습 실행
    3. Evaluator Agent: 품질 검증

    ```
    ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
    │  Data Engineer   │ ──▶ │  Model Trainer   │ ──▶ │    Evaluator     │
    │     Agent        │     │     Agent        │     │     Agent        │
    └──────────────────┘     └──────────────────┘     └──────────────────┘
           │                        │                        │
           ▼                        ▼                        ▼
      train_data.jsonl        adapter_model/          eval_report.md
    ```
    """

    def __init__(self, config: Optional[WorkflowConfig] = None):
        self.config = config or WorkflowConfig()

        # 에이전트 초기화
        self.data_agent = DataEngineerAgent(
            config=DataPipelineConfig(
                input_path=self.config.input_data_path,
                output_path=self.config.train_data_path,
                mask_pii=True,
            )
        )

        self.trainer_agent = ModelTrainerAgent(
            config=TrainingConfig(
                model_name=self.config.model_name,
                data_path=self.config.train_data_path,
                output_dir=self.config.output_dir,
                lora_rank=self.config.lora_rank,
                quantize=self.config.quantize,
            )
        )

        self.evaluator_agent = EvaluatorAgent(
            config=EvalConfig(
                model_url=self.config.vllm_url,
                model_name=self.config.model_name,
            )
        )

        self.current_stage: Optional[WorkflowStage] = None

    async def run_full_pipeline(self) -> WorkflowResult:
        """전체 파이프라인 실행"""
        start_time = datetime.now()
        stages_completed = []
        errors = []

        logger.info("=== LLM 개발 파이프라인 시작 ===")

        # 1단계: 데이터 준비
        self.current_stage = WorkflowStage.DATA_PREPARATION
        logger.info("1단계: 데이터 준비")

        try:
            data_result = await self.data_agent.run_pipeline()
            stages_completed.append("data_preparation")
            logger.info(f"  - 추출 샘플: {data_result.total_samples}개")
            logger.info(f"  - PII 마스킹: {data_result.pii_masked}건")
        except Exception as e:
            errors.append(f"데이터 준비 실패: {str(e)}")
            logger.error(errors[-1])

        # 2단계: 학습 준비
        self.current_stage = WorkflowStage.TRAINING
        logger.info("2단계: 학습 준비")

        try:
            # 리소스 추정
            resources = await self.trainer_agent.estimate_resources()
            logger.info(f"  - 예상 VRAM: {resources['estimated_vram_gb']}GB")
            logger.info(f"  - 권장 GPU: {resources['recommended_gpu']}")

            # 학습 스크립트 생성
            train_result = await self.trainer_agent.start_training()
            stages_completed.append("training_prepared")
            logger.info("  - 학습 스크립트 생성 완료")
        except Exception as e:
            errors.append(f"학습 준비 실패: {str(e)}")
            logger.error(errors[-1])

        # 3단계: 평가 (서버가 실행 중인 경우만)
        self.current_stage = WorkflowStage.EVALUATION
        logger.info("3단계: 평가 준비")

        quality_score = 0.0
        try:
            # 보고서 템플릿 생성
            report = await self.evaluator_agent.generate_report()
            stages_completed.append("evaluation_prepared")
            logger.info("  - 평가 보고서 템플릿 생성 완료")
        except Exception as e:
            errors.append(f"평가 준비 실패: {str(e)}")
            logger.error(errors[-1])

        duration = (datetime.now() - start_time).total_seconds()

        return WorkflowResult(
            success=len(errors) == 0,
            stages_completed=stages_completed,
            data_samples=data_result.total_samples if 'data_result' in dir() else 0,
            quality_score=quality_score,
            model_path=f"{self.config.output_dir}/final",
            duration_seconds=duration,
            errors=errors,
        )

    async def run_stage(self, stage: WorkflowStage, **kwargs) -> Dict[str, Any]:
        """특정 단계만 실행"""
        self.current_stage = stage

        if stage == WorkflowStage.DATA_PREPARATION:
            result = await self.data_agent.run_pipeline(**kwargs)
            return {
                "stage": stage.value,
                "samples": result.total_samples,
                "output": result.output_path,
            }

        elif stage == WorkflowStage.TRAINING:
            result = await self.trainer_agent.start_training()
            return {
                "stage": stage.value,
                "result": result,
            }

        elif stage == WorkflowStage.EVALUATION:
            result = await self.evaluator_agent.run_latency_benchmark(**kwargs)
            return {
                "stage": stage.value,
                "benchmark": result,
            }

        else:
            return {"error": f"Unknown stage: {stage}"}

    def get_status(self) -> Dict[str, Any]:
        """현재 상태 조회"""
        return {
            "current_stage": self.current_stage.value if self.current_stage else None,
            "config": {
                "model": self.config.model_name,
                "data_path": self.config.input_data_path,
                "output_dir": self.config.output_dir,
            },
        }


# 간편 사용을 위한 함수
async def run_llm_development_pipeline(
    data_path: str = "data/",
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    output_dir: str = "./output",
) -> WorkflowResult:
    """LLM 개발 파이프라인 실행"""
    config = WorkflowConfig(
        input_data_path=data_path,
        model_name=model_name,
        output_dir=output_dir,
    )

    orchestrator = AgentOrchestrator(config)
    return await orchestrator.run_full_pipeline()


async def main():
    """테스트 실행"""
    result = await run_llm_development_pipeline()

    print("\n=== 파이프라인 결과 ===")
    print(f"성공: {result.success}")
    print(f"완료 단계: {', '.join(result.stages_completed)}")
    print(f"데이터 샘플: {result.data_samples}개")
    print(f"소요 시간: {result.duration_seconds:.1f}초")

    if result.errors:
        print(f"오류: {', '.join(result.errors)}")


if __name__ == "__main__":
    asyncio.run(main())
