"""질문, 생성 답변, 기준 답변을 네 가지 점수로 평가한다."""

import logging
import os
from collections.abc import Callable, Sequence
from typing import Literal

from .mlflow_evaluation import evaluate_with_mlflow
from .prompts import ALLGANIZE_CORRECTNESS_PROMPT, TONIC_SIMILARITY_PROMPT
from .results import EvaluationResult
from .validation import validate_error_policy, validate_inputs

logger = logging.getLogger(__name__)


class EvaluationError(RuntimeError):
    """평가 모델 호출 또는 점수 해석에 실패했다."""


class RAGEvaluator:
    """프롬프트를 받아 문자열을 반환하는 평가 함수를 주입받는다.

    on_error='raise'는 첫 평가 실패에서 중단한다. 'record'는 개별 점수
    실패를 -1로 기록한다. MLflow 실행 전체의 실패는 항상 예외로 전달한다.
    """

    def __init__(
        self,
        *,
        similarity_judge: Callable[[str], str],
        correctness_judge: Callable[[str], str],
        mlflow_model: str,
        on_error: Literal["raise", "record"] = "raise",
        mlflow_evaluator: Callable[..., tuple[list[float], list[float]]] | None = None,
    ) -> None:
        validate_error_policy(on_error)
        self.similarity_judge = similarity_judge
        self.correctness_judge = correctness_judge
        self.mlflow_model = mlflow_model
        self.on_error = on_error
        self.mlflow_evaluator = mlflow_evaluator

    @classmethod
    def from_env(
        cls,
        env_file: str | os.PathLike[str] = ".env",
        *,
        on_error: Literal["raise", "record"] = "raise",
    ) -> "RAGEvaluator":
        """.env를 읽어 구성한다. 기존 환경 변수가 파일 값보다 우선한다."""
        validate_error_policy(on_error)
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)
        required_settings = (
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "EVAL_OPENAI_MODEL",
            "EVAL_ANTHROPIC_MODEL",
        )
        missing = [
            name for name in required_settings if not os.environ.get(name, "").strip()
        ]
        if missing:
            raise ValueError(f"Missing required settings: {', '.join(missing)}")

        openai_model = os.environ["EVAL_OPENAI_MODEL"].strip()
        mlflow_model = os.environ.get("EVAL_MLFLOW_MODEL", "").strip()
        return cls.from_models(
            openai_model=openai_model,
            anthropic_model=os.environ["EVAL_ANTHROPIC_MODEL"].strip(),
            mlflow_model=mlflow_model or f"openai:/{openai_model}",
            on_error=on_error,
        )

    @classmethod
    def from_models(
        cls,
        *,
        openai_model: str,
        anthropic_model: str,
        mlflow_model: str,
        on_error: Literal["raise", "record"] = "raise",
    ) -> "RAGEvaluator":
        """환경 변수의 API 키로 노트북과 같은 모델 구성을 만든다."""
        validate_error_policy(on_error)
        from langchain_anthropic import ChatAnthropic
        from langchain_core.output_parsers import StrOutputParser
        from langchain_openai import ChatOpenAI

        similarity_chain = ChatOpenAI(model=openai_model) | StrOutputParser()
        correctness_chain = ChatAnthropic(model=anthropic_model) | StrOutputParser()
        return cls(
            similarity_judge=similarity_chain.invoke,
            correctness_judge=correctness_chain.invoke,
            mlflow_model=mlflow_model,
            on_error=on_error,
        )

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


def main() -> None:
    """현재 디렉터리의 .env로 한 건의 예시 평가를 실행한다."""
    evaluator = RAGEvaluator.from_env()
    result = evaluator.evaluate(
        questions=["대한민국의 수도는?"],
        generated_answers=["서울"],
        reference_answers=["서울"],
    )
    print(result.verdicts)


if __name__ == "__main__":
    main()
