"""평가 점수와 최종 판정. 외부 서비스에 의존하지 않는다."""

from dataclasses import dataclass
from numbers import Real


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
        for metric, scores, minimum, maximum in (
            ("tonic_similarity", self.tonic_similarity, 0, 5),
            ("mlflow_similarity", self.mlflow_similarity, 1, 5),
            ("mlflow_correctness", self.mlflow_correctness, 1, 5),
            ("allganize_correctness", self.allganize_correctness, 0, 1),
        ):
            for index, score in enumerate(scores):
                if (
                    isinstance(score, bool)
                    or not isinstance(score, Real)
                    or not (score == -1 or minimum <= score <= maximum)
                    or (
                        metric in ("tonic_similarity", "allganize_correctness")
                        and score != int(score)
                    )
                ):
                    raise ValueError(f"Invalid {metric} score at row {index}.")

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
