"""질문, 생성 답변, 기준 답변을 네 가지 점수로 평가한다."""

import logging
import os
from collections.abc import Callable, Sequence
from typing import Literal

from dotenv import load_dotenv

from .mlflow_evaluation import evaluate_with_mlflow
from .prompts import ALLGANIZE_CORRECTNESS_PROMPT, TONIC_SIMILARITY_PROMPT
from .results import EvaluationResult
from .validation import validate_error_policy, validate_inputs

logger = logging.getLogger(__name__)


class EvaluationError(RuntimeError):
    """평가 모델 호출 또는 점수 해석에 실패했다."""


class RAGEvaluator:
    """선택한 모델 제공자로 네 가지 답변 평가 점수를 계산한다.

    RAGEvaluator()는 Claude Haiku를, model_provider="openai"는 GPT nano를 쓴다.
    .env에서 API 키를 읽으며 기존 환경 변수는 덮어쓰지 않는다.
    테스트에서는 similarity_judge와 correctness_judge를 직접 주입할 수 있다.

    on_error='raise'는 첫 평가 실패에서 중단한다. 'record'는 개별 점수
    실패를 -1로 기록한다. MLflow 실행 전체의 실패는 항상 예외로 전달한다.
    """

    def __init__(
        self,
        model_provider: Literal["claude", "openai", "local"] = "claude",
        *,
        env_file: str | os.PathLike[str] = ".env",
        on_error: Literal["raise", "record"] = "raise",
        similarity_judge: Callable[[str], str] | None = None,
        correctness_judge: Callable[[str], str] | None = None,
        mlflow_model: str | None = None,
        mlflow_evaluator: Callable[..., tuple[list[float], list[float]]] | None = None,
    ) -> None:
        validate_error_policy(on_error)
        if model_provider == "local":
            raise NotImplementedError("model_provider='local' is not supported yet.")
        if model_provider not in ("claude", "openai"):
            raise ValueError("model_provider must be 'claude', 'openai', or 'local'.")

        load_dotenv(env_file, override=False)

        model_name = (
            "claude-haiku-4-5" if model_provider == "claude" else "gpt-4.1-nano"
        )
        mlflow_provider = "anthropic" if model_provider == "claude" else "openai"
        self.model_provider = model_provider
        self.model = None
        if similarity_judge is None or correctness_judge is None:
            from langchain_core.output_parsers import StrOutputParser

            if model_provider == "claude":
                from langchain_anthropic import ChatAnthropic

                self.model = ChatAnthropic(model=model_name, temperature=0)
            else:
                from langchain_openai import ChatOpenAI

                self.model = ChatOpenAI(model=model_name, temperature=0)
            judge = (self.model | StrOutputParser()).invoke
            if similarity_judge is None:
                similarity_judge = judge
            if correctness_judge is None:
                correctness_judge = judge

        self.similarity_judge = similarity_judge
        self.correctness_judge = correctness_judge
        self.mlflow_model = mlflow_model or f"{mlflow_provider}:/{model_name}"
        self.on_error = on_error
        self.mlflow_evaluator = mlflow_evaluator

    def evaluate(
        self,
        questions: Sequence[str],
        generated_answers: Sequence[str],
        reference_answers: Sequence[str],
    ) -> EvaluationResult:
        """동일한 길이의 문자열 목록을 평가하고 점수와 최종 판정을 반환한다."""
        validate_inputs(questions, generated_answers, reference_answers)
        if not questions:
            return EvaluationResult([], [], [], [])

        rows = list(zip(questions, generated_answers, reference_answers, strict=True))
        tonic_scores = self._score_answers(
            rows,
            self.similarity_judge,
            TONIC_SIMILARITY_PROMPT,
            metric="tonic_similarity",
            maximum=5,
        )
        mlflow_evaluator = self.mlflow_evaluator
        if mlflow_evaluator is None:
            mlflow_evaluator = evaluate_with_mlflow
        mlflow_similarity, mlflow_correctness = mlflow_evaluator(
            questions,
            generated_answers,
            reference_answers,
            model=self.mlflow_model,
            on_error=self.on_error,
        )
        allganize_scores = self._score_answers(
            rows,
            self.correctness_judge,
            ALLGANIZE_CORRECTNESS_PROMPT,
            metric="allganize_correctness",
            maximum=1,
        )
        return EvaluationResult(
            tonic_similarity=tonic_scores,
            mlflow_similarity=mlflow_similarity,
            mlflow_correctness=mlflow_correctness,
            allganize_correctness=allganize_scores,
        )

    def _score_answers(
        self,
        rows: Sequence[tuple[str, str, str]],
        judge: Callable[[str], str],
        template: str,
        *,
        metric: str,
        maximum: int,
    ) -> list[int]:
        scores = []
        for index, (question, generated_answer, reference_answer) in enumerate(rows):
            prompt = template.format(
                question=question,
                reference_answer=reference_answer,
                llm_answer=generated_answer,
            )
            try:
                response = judge(prompt)
                if not isinstance(response, str):
                    raise TypeError("Judge must return a string.")
                score = int(response.strip())
                if not 0 <= score <= maximum:
                    raise ValueError(f"Score must be between 0 and {maximum}.")
            except Exception as error:
                message = f"{metric} failed at row {index}"
                if self.on_error == "raise":
                    raise EvaluationError(message) from error
                logger.warning("%s (%s)", message, type(error).__name__)
                score = -1
            scores.append(score)
        return scores
