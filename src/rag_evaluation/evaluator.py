"""질문, 생성 답변, 기준 답변을 네 가지 점수로 평가한다."""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from .mlflow_evaluation import evaluate_with_mlflow
from .prompts import ALLGANIZE_CORRECTNESS_PROMPT, TONIC_SIMILARITY_PROMPT

logger = logging.getLogger(__name__)


class EvaluationError(RuntimeError):
    """평가 모델 호출 또는 점수 해석에 실패했다."""


@dataclass
class EvaluationResult:
    """입력 순서대로 저장한 점수. -1은 평가 실패를 나타낸다."""

    tonic_similarity: list[int]
    mlflow_similarity: list[float]
    mlflow_correctness: list[float]
    allganize_correctness: list[int]

    def __post_init__(self) -> None:
        lengths = {
            len(self.tonic_similarity),
            len(self.mlflow_similarity),
            len(self.mlflow_correctness),
            len(self.allganize_correctness),
        }
        if len(lengths) != 1:
            raise ValueError("All score lists must have the same length.")

    @property
    def verdicts(self) -> list[str]:
        """4표 중 3표 이상이 정답이면 O, 동점을 포함한 나머지는 X."""
        verdicts = []
        for tonic, similarity, correctness, allganize in zip(
            self.tonic_similarity,
            self.mlflow_similarity,
            self.mlflow_correctness,
            self.allganize_correctness,
            strict=True,
        ):
            positive_votes = sum(
                (tonic >= 4, similarity >= 4, correctness >= 4, allganize == 1)
            )
            verdicts.append("O" if positive_votes >= 3 else "X")
        return verdicts


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
    ) -> None:
        if on_error not in ("raise", "record"):
            raise ValueError("on_error must be 'raise' or 'record'.")
        self.similarity_judge = similarity_judge
        self.correctness_judge = correctness_judge
        self.mlflow_model = mlflow_model
        self.on_error = on_error

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
        for name, values in (
            ("questions", questions),
            ("generated_answers", generated_answers),
            ("reference_answers", reference_answers),
        ):
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise TypeError(f"{name} must be a sequence of strings.")
            if any(not isinstance(value, str) for value in values):
                raise TypeError(f"{name} must contain only strings.")
        if not len(questions) == len(generated_answers) == len(reference_answers):
            raise ValueError("All input lists must have the same length.")
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
        mlflow_similarity, mlflow_correctness = evaluate_with_mlflow(
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
