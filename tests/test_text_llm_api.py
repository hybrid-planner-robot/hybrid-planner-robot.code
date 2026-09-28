"""OpenAI-compatible text LLM adapter (no network in CI)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from planner.text_llm import (
    DEFAULT_API_TIMEOUT_S,
    DEFAULT_TEXT_LLM_TEMPERATURE,
    ENV_TEXT_LLM_API_KEY,
    ENV_TEXT_LLM_BASE_URL,
    ENV_TEXT_LLM_ENABLE_THINKING,
    ENV_TEXT_LLM_TEMPERATURE,
    ENV_TEXT_LLM_TIMEOUT_S,
    OpenAICompatibleClient,
    get_shared_text_client,
    reset_shared_text_client,
    resolve_api_timeout_s,
    resolve_enable_thinking,
    resolve_temperature,
    strip_think_tags,
)


def test_openai_compatible_complete_parses_choices():
    client = OpenAICompatibleClient(
        "gpt-test", base_url="https://example.invalid/v1", api_key="sk-test"
    )
    payload = {
        "choices": [{"message": {"content": '{"template": "manipulation_base"}'}}]
    }
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False
    with patch("urllib.request.urlopen", return_value=mock_resp) as mocked:
        text = client.complete("sys", "user")
    assert "manipulation_base" in text
    req = mocked.call_args[0][0]
    assert req.full_url.endswith("/chat/completions")
    assert req.get_header("Authorization") == "Bearer sk-test"
    body = json.loads(req.data.decode("utf-8"))
    assert "enable_thinking" not in body
    assert "chat_template_kwargs" not in body
    assert body["temperature"] == 0


def test_shared_client_reuses_weights_across_decode_budgets(monkeypatch):
    """Select (256) and R1 (768) must not load the 7B twice."""
    reset_shared_text_client()
    monkeypatch.delenv(ENV_TEXT_LLM_BASE_URL, raising=False)

    class Dummy:
        n_init = 0

        def __init__(self, model_id, *, max_new_tokens=256):
            type(self).n_init += 1
            self.model_id = model_id
            self.max_new_tokens = max_new_tokens
            self.last_budget = None

        def complete(self, system, user, *, max_new_tokens=None):
            self.last_budget = (
                int(max_new_tokens)
                if max_new_tokens is not None
                else self.max_new_tokens
            )
            return "ok"

    monkeypatch.setattr("planner.text_llm.TransformersLocalClient", Dummy)
    a = get_shared_text_client("Qwen/test", max_new_tokens=256)
    b = get_shared_text_client("Qwen/test", max_new_tokens=768)
    assert a is b
    assert Dummy.n_init == 1
    assert a.complete("s", "u", max_new_tokens=768) == "ok"
    assert a.last_budget == 768
    reset_shared_text_client()


def test_shared_client_uses_api_when_base_url_set(monkeypatch):
    reset_shared_text_client()
    monkeypatch.setenv(ENV_TEXT_LLM_BASE_URL, "http://127.0.0.1:8000/v1")
    monkeypatch.setenv(ENV_TEXT_LLM_API_KEY, "x")
    client = get_shared_text_client("local-model")
    assert isinstance(client, OpenAICompatibleClient)
    assert client.model_id == "local-model"
    reset_shared_text_client()


def test_reset_shared_client_closes_loaded_weights(monkeypatch):
    reset_shared_text_client()
    monkeypatch.delenv(ENV_TEXT_LLM_BASE_URL, raising=False)
    client = get_shared_text_client("dummy")
    closed = {"n": 0}

    def _close():
        closed["n"] += 1

    client.close = _close  # type: ignore[method-assign]
    reset_shared_text_client()
    assert closed["n"] == 1
    reset_shared_text_client()
    assert closed["n"] == 1


def _mock_completion(payload: dict):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False
    return mock_resp


def test_enable_thinking_false_is_sent():
    client = OpenAICompatibleClient(
        "m", base_url="http://127.0.0.1:8080/v1", enable_thinking=False
    )
    with patch(
        "urllib.request.urlopen",
        return_value=_mock_completion(
            {"choices": [{"message": {"content": "ok"}}]}
        ),
    ) as mocked:
        client.complete("s", "u")
    body = json.loads(mocked.call_args[0][0].data.decode("utf-8"))
    assert body["enable_thinking"] is False
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_temperature_defaults_to_zero(monkeypatch):
    monkeypatch.delenv(ENV_TEXT_LLM_TEMPERATURE, raising=False)
    assert resolve_temperature() == DEFAULT_TEXT_LLM_TEMPERATURE == 0.0
    monkeypatch.setenv(ENV_TEXT_LLM_TEMPERATURE, "0")
    assert resolve_temperature() == 0.0
    client = OpenAICompatibleClient("m", base_url="http://127.0.0.1:8080/v1")
    assert client.temperature == 0.0


def test_strips_think_tags_and_falls_back_to_reasoning():
    assert strip_think_tags("<think>secret</think>\n{\"a\":1}") == '{"a":1}'
    client = OpenAICompatibleClient("m", base_url="http://127.0.0.1:8080/v1")
    with patch(
        "urllib.request.urlopen",
        return_value=_mock_completion(
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning_content": "<think>x</think>goal",
                        }
                    }
                ]
            }
        ),
    ):
        assert client.complete("s", "u") == "goal"


def test_timeout_and_thinking_from_env(monkeypatch):
    monkeypatch.delenv(ENV_TEXT_LLM_TIMEOUT_S, raising=False)
    monkeypatch.delenv(ENV_TEXT_LLM_ENABLE_THINKING, raising=False)
    assert resolve_api_timeout_s() == DEFAULT_API_TIMEOUT_S
    assert resolve_enable_thinking() is None
    monkeypatch.setenv(ENV_TEXT_LLM_TIMEOUT_S, "900")
    monkeypatch.setenv(ENV_TEXT_LLM_ENABLE_THINKING, "0")
    assert resolve_api_timeout_s() == 900.0
    assert resolve_enable_thinking() is False
    reset_shared_text_client()
    monkeypatch.setenv(ENV_TEXT_LLM_BASE_URL, "http://127.0.0.1:8080/v1")
    client = get_shared_text_client("gguf")
    assert isinstance(client, OpenAICompatibleClient)
    assert client.timeout_s == 900.0
    assert client.enable_thinking is False
    reset_shared_text_client()
