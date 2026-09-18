"""노트북의 MLflow 유사성·정확성 메트릭 연동."""

import logging
import math
from collections.abc import Sequence
from typing import Literal

logger = logging.getLogger(__name__)


def evaluate_with_mlflow(
    questions: Sequence[str],
    generated_answers: Sequence[str],
    reference_answers: Sequence[str],
    *,
    model: str,
    on_error: Literal["raise", "record"] = "raise",
) -> tuple[list[float], list[float]]:
    """MLflow의 1~5점 척도를 반환한다. 호출자가 연 run은 유지한다.

    MLflow 실행 및 저장 실패는 그대로 전파한다. 개별 누락/비정상 점수는
    on_error='record'일 때만 -1로 변환한다.
    """
    import mlflow
    import pandas as pd
    from mlflow.metrics.genai import answer_correctness, answer_similarity

    data = pd.DataFrame(
        {
            "inputs": questions,
            "predictions": generated_answers,
            "ground_truth": reference_answers,
        }
    )
    with mlflow.start_run(nested=mlflow.active_run() is not None):
        result = mlflow.models.evaluate(
            data=data,
            targets="ground_truth",
            predictions="predictions",
            extra_metrics=[
                answer_similarity(model=model),
                answer_correctness(model=model),
            ],
            evaluators="default",
        )
        table = result.tables["eval_results_table"]

    if len(table) != len(questions):
        raise ValueError("MLflow returned a different number of rows.")

    score_columns = []
    for metric in ("answer_similarity", "answer_correctness"):
        scores = []
        for index, value in enumerate(table[f"{metric}/v1/score"].tolist()):
            try:
                score = float(value)
                if not math.isfinite(score) or not 1 <= score <= 5:
                    raise ValueError("MLflow score must be between 1 and 5.")
            except (TypeError, ValueError) as error:
                message = f"MLflow {metric} failed at row {index}"
                if on_error == "raise":
                    raise ValueError(message) from error
                logger.warning(message)
                score = -1.0
            scores.append(score)
        score_columns.append(scores)
    return score_columns[0], score_columns[1]
