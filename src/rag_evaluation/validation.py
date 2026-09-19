"""전체 평가와 개별 평가 진입점에서 사용하는 입력 검증."""

from collections.abc import Sequence


def validate_error_policy(on_error: str) -> None:
    if on_error not in ("raise", "record"):
        raise ValueError("on_error must be 'raise' or 'record'.")


def validate_inputs(
    questions: Sequence[str],
    generated_answers: Sequence[str],
    reference_answers: Sequence[str],
) -> None:
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
