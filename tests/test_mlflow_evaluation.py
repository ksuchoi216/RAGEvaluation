import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rag_evaluation.mlflow_evaluation import evaluate_with_mlflow


@pytest.fixture
def mlflow_boundary(monkeypatch):
    mlflow = ModuleType("mlflow")
    mlflow.active_run = lambda: object()
    mlflow.start_run = MagicMock()
    mlflow.models = SimpleNamespace(evaluate=MagicMock())
    metrics = ModuleType("mlflow.metrics.genai")
    metrics.answer_similarity = lambda **kwargs: ("similarity", kwargs)
    metrics.answer_correctness = lambda **kwargs: ("correctness", kwargs)
    pandas = ModuleType("pandas")
    pandas.DataFrame = lambda values: values
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.metrics.genai", metrics)
    monkeypatch.setitem(sys.modules, "pandas", pandas)
    return mlflow


def set_scores(mlflow, similarity, correctness):
    table = MagicMock()
    table.__len__.return_value = len(similarity)
    columns = {
        "answer_similarity/v1/score": SimpleNamespace(tolist=lambda: similarity),
        "answer_correctness/v1/score": SimpleNamespace(tolist=lambda: correctness),
    }
    table.__getitem__.side_effect = columns.__getitem__
    mlflow.models.evaluate.return_value = SimpleNamespace(
        tables={"eval_results_table": table}
    )


def test_mlflow_maps_columns_and_uses_nested_run(mlflow_boundary):
    set_scores(mlflow_boundary, [4], [2])
    scores = evaluate_with_mlflow(["q"], ["a"], ["r"], model="openai:/judge")
    assert scores == ([4.0], [2.0])
    mlflow_boundary.start_run.assert_called_once_with(nested=True)
    arguments = mlflow_boundary.models.evaluate.call_args.kwargs
    assert arguments["data"] == {
        "inputs": ["q"],
        "predictions": ["a"],
        "ground_truth": ["r"],
    }
    assert arguments["targets"] == "ground_truth"
    assert arguments["predictions"] == "predictions"
    assert arguments["extra_metrics"] == [
        ("similarity", {"model": "openai:/judge"}),
        ("correctness", {"model": "openai:/judge"}),
    ]


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), 0, 6])
def test_mlflow_invalid_scores_raise_or_record(mlflow_boundary, value):
    set_scores(mlflow_boundary, [value], [5])
    with pytest.raises(ValueError, match="answer_similarity.*row 0"):
        evaluate_with_mlflow(["q"], ["a"], ["r"], model="openai:/judge")
    assert evaluate_with_mlflow(
        ["q"], ["a"], ["r"], model="openai:/judge", on_error="record"
    ) == ([-1.0], [5.0])


def test_mlflow_incomplete_results_are_rejected(mlflow_boundary):
    set_scores(mlflow_boundary, [], [])
    with pytest.raises(ValueError, match="number of rows"):
        evaluate_with_mlflow(["q"], ["a"], ["r"], model="openai:/judge")


def test_mlflow_run_failure_is_not_hidden(mlflow_boundary):
    mlflow_boundary.models.evaluate.side_effect = RuntimeError("service failed")
    with pytest.raises(RuntimeError, match="service failed"):
        evaluate_with_mlflow(
            ["q"], ["a"], ["r"], model="openai:/judge", on_error="record"
        )
