import os
from unittest.mock import patch

import pytest

from rag_evaluation import RAGEvaluator


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "EVAL_OPENAI_MODEL",
        "EVAL_ANTHROPIC_MODEL",
        "EVAL_MLFLOW_MODEL",
        "MLFLOW_TRACKING_URI",
        "mlflow_url",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(os, "environ", os.environ.copy())


def test_loads_dotenv_from_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=test-openai\nANTHROPIC_API_KEY=test-anthropic\n"
        "EVAL_OPENAI_MODEL=openai-judge\nEVAL_ANTHROPIC_MODEL=claude-judge\n"
        "MLFLOW_TRACKING_URI=http://localhost:5000\n"
    )
    with patch.object(RAGEvaluator, "from_models") as factory:
        evaluator = RAGEvaluator.from_env(on_error="record")
    assert evaluator is factory.return_value
    factory.assert_called_once_with(
        openai_model="openai-judge",
        anthropic_model="claude-judge",
        mlflow_model="openai:/openai-judge",
        on_error="record",
    )
    assert os.environ["OPENAI_API_KEY"] == "test-openai"
    assert os.environ["MLFLOW_TRACKING_URI"] == "http://localhost:5000"


def test_environment_overrides_explicit_dotenv_path(tmp_path, monkeypatch):
    path = tmp_path / "custom.env"
    path.write_text(
        "OPENAI_API_KEY=file-key\nANTHROPIC_API_KEY=test-anthropic\n"
        "EVAL_OPENAI_MODEL=file-model\nEVAL_ANTHROPIC_MODEL=claude-judge\n"
        "EVAL_MLFLOW_MODEL=openai:/separate-judge\n"
    )
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    monkeypatch.setenv("EVAL_OPENAI_MODEL", "environment-model")
    with patch.object(RAGEvaluator, "from_models") as factory:
        RAGEvaluator.from_env(path)
    assert os.environ["OPENAI_API_KEY"] == "environment-key"
    assert factory.call_args.kwargs["openai_model"] == "environment-model"
    assert factory.call_args.kwargs["mlflow_model"] == "openai:/separate-judge"


def test_missing_settings_fail_before_model_creation(tmp_path):
    path = tmp_path / ".env"
    path.write_text("OPENAI_API_KEY=\nEVAL_OPENAI_MODEL=   \n")
    with patch.object(RAGEvaluator, "from_models") as factory:
        with pytest.raises(ValueError, match="OPENAI_API_KEY.*EVAL_OPENAI_MODEL"):
            RAGEvaluator.from_env(path)
    factory.assert_not_called()


def test_environment_only_works_without_dotenv_file(tmp_path, monkeypatch):
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "EVAL_OPENAI_MODEL",
        "EVAL_ANTHROPIC_MODEL",
    ):
        monkeypatch.setenv(name, "configured")
    with patch.object(RAGEvaluator, "from_models") as factory:
        RAGEvaluator.from_env(tmp_path / "missing.env")
    assert factory.call_args.kwargs["openai_model"] == "configured"


@pytest.mark.parametrize(
    "url, expected",
    [
        ("localhost:30001", "http://localhost:30001"),
        ("http://localhost:30001", "http://localhost:30001"),
        ("https://mlflow.example.com", "https://mlflow.example.com"),
    ],
)
def test_mlflow_url_becomes_tracking_uri(tmp_path, monkeypatch, url, expected):
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "EVAL_OPENAI_MODEL",
        "EVAL_ANTHROPIC_MODEL",
    ):
        monkeypatch.setenv(name, "configured")
    path = tmp_path / ".env"
    path.write_text(f"mlflow_url={url}\n")
    with patch.object(RAGEvaluator, "from_models"):
        RAGEvaluator.from_env(path)
    assert os.environ["MLFLOW_TRACKING_URI"] == expected


def test_explicit_tracking_uri_takes_priority(tmp_path, monkeypatch):
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "EVAL_OPENAI_MODEL",
        "EVAL_ANTHROPIC_MODEL",
    ):
        monkeypatch.setenv(name, "configured")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://existing-server:5000")
    path = tmp_path / ".env"
    path.write_text("mlflow_url=localhost:30001\n")
    with patch.object(RAGEvaluator, "from_models"):
        RAGEvaluator.from_env(path)
    assert os.environ["MLFLOW_TRACKING_URI"] == "http://existing-server:5000"
