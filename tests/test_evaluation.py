import itertools
from unittest.mock import patch

import pytest

from rag_evaluation import EvaluationError, EvaluationResult, RAGEvaluator


def test_majority_requires_three_positive_votes():
    combinations = list(itertools.product((False, True), repeat=4))
    result = EvaluationResult(
        tonic_similarity=[4 if row[0] else 3 for row in combinations],
        mlflow_similarity=[4 if row[1] else 3 for row in combinations],
        mlflow_correctness=[4 if row[2] else 3 for row in combinations],
        allganize_correctness=[int(row[3]) for row in combinations],
    )
    assert result.verdicts == [
        "X",
        "X",
        "X",
        "X",
        "X",
        "X",
        "X",
        "O",
        "X",
        "X",
        "X",
        "O",
        "X",
        "O",
        "O",
        "O",
    ]


def make_evaluator(**kwargs):
    return RAGEvaluator(
        similarity_judge=lambda prompt: " 5\n",
        correctness_judge=lambda prompt: "1",
        mlflow_model="openai:/test-model",
        **kwargs,
    )


def test_evaluation_keeps_scores_and_formats_input():
    prompts = []
    evaluator = make_evaluator()
    evaluator.similarity_judge = lambda prompt: prompts.append(prompt) or "4"
    with patch(
        "rag_evaluation.evaluator.evaluate_with_mlflow",
        return_value=([5, 2], [3, 4]),
    ):
        result = evaluator.evaluate(["q{1}", "q2"], ["a1", "a2"], ["r1", "r2"])
    assert result.tonic_similarity == [4, 4]
    assert result.mlflow_similarity == [5, 2]
    assert result.mlflow_correctness == [3, 4]
    assert result.allganize_correctness == [1, 1]
    assert result.verdicts == ["O", "O"]
    assert "QUESTION: q{1}" in prompts[0]
    assert "REFERENCE ANSWER: r1" in prompts[0]
    assert "NEW ANSWER: a1" in prompts[0]


@pytest.mark.parametrize(
    "inputs",
    [
        (["q"], [], ["r"]),
        ("question", ["a"], ["r"]),
        (["q"], [None], ["r"]),
    ],
)
def test_invalid_inputs_fail_before_judging(inputs):
    evaluator = make_evaluator()
    evaluator.similarity_judge = lambda prompt: pytest.fail("Judge called")
    with pytest.raises((TypeError, ValueError)):
        evaluator.evaluate(*inputs)


def test_empty_inputs_do_not_call_external_services():
    evaluator = make_evaluator()
    evaluator.similarity_judge = lambda prompt: pytest.fail("Judge called")
    with patch(
        "rag_evaluation.evaluator.evaluate_with_mlflow",
        side_effect=AssertionError("MLflow called"),
    ):
        assert evaluator.evaluate([], [], []).verdicts == []


@pytest.mark.parametrize("response", ["6", "-1", "4.5", "score: 4", ""])
def test_invalid_judge_scores_raise_with_row_context(response):
    evaluator = make_evaluator()
    evaluator.similarity_judge = lambda prompt: response
    with pytest.raises(EvaluationError, match="tonic_similarity.*row 0"):
        evaluator.evaluate(["q"], ["a"], ["r"])


def test_continue_on_error_preserves_row_alignment():
    evaluator = make_evaluator(on_error="record")
    responses = iter(["bad", "5"])
    evaluator.similarity_judge = lambda prompt: next(responses)
    with patch(
        "rag_evaluation.evaluator.evaluate_with_mlflow",
        return_value=([3, 4], [3, 4]),
    ):
        result = evaluator.evaluate(["q1", "q2"], ["a1", "a2"], ["r1", "r2"])
    assert result.tonic_similarity == [-1, 5]
    assert result.verdicts == ["X", "O"]


def test_api_error_is_chained():
    evaluator = make_evaluator()
    error = TimeoutError("timeout")

    def unavailable(prompt):
        raise error

    evaluator.similarity_judge = unavailable
    with pytest.raises(EvaluationError) as caught:
        evaluator.evaluate(["q"], ["a"], ["r"])
    assert caught.value.__cause__ is error


def test_invalid_error_policy_is_rejected():
    with pytest.raises(ValueError, match="on_error"):
        make_evaluator(on_error="ignore")


def test_result_rejects_misaligned_scores():
    with pytest.raises(ValueError, match="length"):
        EvaluationResult([5], [], [5], [1])


def test_mlflow_evaluator_can_be_injected():
    def score_with_mlflow(questions, generated_answers, reference_answers, **options):
        assert questions == ["q"]
        assert generated_answers == ["a"]
        assert reference_answers == ["r"]
        assert options == {"model": "openai:/test-model", "on_error": "raise"}
        return [2], [5]

    evaluator = make_evaluator(mlflow_evaluator=score_with_mlflow)
    result = evaluator.evaluate(["q"], ["a"], ["r"])
    assert result.mlflow_similarity == [2]
    assert result.mlflow_correctness == [5]
    assert result.verdicts == ["O"]


@pytest.mark.parametrize("score", [float("nan"), float("inf"), 6, True, "5"])
def test_result_rejects_invalid_scores(score):
    with pytest.raises(ValueError, match="tonic_similarity"):
        EvaluationResult([score], [5], [5], [1])


def test_recorded_failure_remains_a_negative_vote():
    assert EvaluationResult([-1], [5], [3], [1]).verdicts == ["X"]
