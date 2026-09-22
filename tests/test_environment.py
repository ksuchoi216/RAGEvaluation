import os

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from rag_evaluation import RAGEvaluator


@pytest.fixture(autouse=True)
def clean_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
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
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=test-openai\nANTHROPIC_API_KEY=test-anthropic\n"
    )


def test_default_constructor_loads_keys_and_selects_claude():
    evaluator = RAGEvaluator()
    assert isinstance(evaluator.model, ChatAnthropic)
    assert evaluator.model.model == "claude-haiku-4-5"
    assert evaluator.model.anthropic_api_key.get_secret_value() == "test-anthropic"
    assert evaluator.mlflow_model == "anthropic:/claude-haiku-4-5"
    assert os.environ["OPENAI_API_KEY"] == "test-openai"


def test_openai_provider_does_not_require_anthropic_key(tmp_path):
    (tmp_path / ".env").write_text("OPENAI_API_KEY=test-openai\n")
    evaluator = RAGEvaluator(model_provider="openai")
    assert isinstance(evaluator.model, ChatOpenAI)
    assert evaluator.model.model_name == "gpt-4.1-nano"
    assert evaluator.mlflow_model == "openai:/gpt-4.1-nano"


def test_existing_environment_takes_priority(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "existing-key")
    evaluator = RAGEvaluator(model_provider="openai")
    assert evaluator.model.openai_api_key.get_secret_value() == "existing-key"


@pytest.mark.parametrize("option", ["model_provider"])
def test_local_provider_is_not_implemented(option):
    with pytest.raises(NotImplementedError, match=option):
        RAGEvaluator(**{option: "local"})


@pytest.mark.parametrize("option", ["model_provider"])
def test_unknown_provider_is_rejected(option):
    with pytest.raises(ValueError, match=option):
        RAGEvaluator(**{option: "unknown"})


def test_constructor_does_not_configure_mlflow_tracking(monkeypatch):
    monkeypatch.setenv("mlflow_url", "localhost:30001")
    RAGEvaluator()
    assert "MLFLOW_TRACKING_URI" not in os.environ


def test_claude_provider_does_not_require_openai_key(tmp_path):
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=test-anthropic\n")
    evaluator = RAGEvaluator()
    assert evaluator.model.anthropic_api_key.get_secret_value() == "test-anthropic"
    assert evaluator.mlflow_model.startswith("anthropic:/")


@pytest.mark.parametrize(
    "provider, model_class, model_uri",
    [
        ("claude", ChatAnthropic, "anthropic:/claude-haiku-4-5"),
        ("openai", ChatOpenAI, "openai:/gpt-4.1-nano"),
    ],
)
def test_selected_provider_drives_all_four_metrics(
    monkeypatch, provider, model_class, model_uri
):
    from langchain_core.messages import AIMessage

    responses = iter(["5", "1"])
    monkeypatch.setattr(
        model_class,
        "invoke",
        lambda *args, **kwargs: AIMessage(content=next(responses)),
    )

    def score_mlflow(
        questions, generated_answers, reference_answers, *, model, on_error
    ):
        assert model == model_uri
        assert (questions, generated_answers, reference_answers) == (
            ["q"],
            ["a"],
            ["r"],
        )
        return [5], [5]

    evaluator = RAGEvaluator(model_provider=provider, mlflow_evaluator=score_mlflow)
    result = evaluator.evaluate(["q"], ["a"], ["r"])
    assert result.verdicts == ["O"]
    assert result.tonic_similarity == [5]
    assert result.allganize_correctness == [1]


def test_explicit_env_file_is_loaded(tmp_path):
    path = tmp_path / "custom.env"
    path.write_text("OPENAI_API_KEY=custom-key\n")
    evaluator = RAGEvaluator(model_provider="openai", env_file=path)
    assert evaluator.model.openai_api_key.get_secret_value() == "custom-key"
