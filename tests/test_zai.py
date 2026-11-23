import pytest
from fintech_support.zai import ZaiClient, ProviderError, Budget, BudgetExceeded


def test_native_json_and_usage():
    calls = []

    def transport(payload, timeout):
        calls.append(payload)
        return {
            "choices": [{"message": {"content": '{"answer":"ok"}'}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }

    client = ZaiClient(api_key="secret", transport=transport)
    assert client.chat_json([{"role": "user", "content": "hello"}]) == {"answer": "ok"}
    assert calls[0]["model"] == "glm-5.3-flash"
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert client.last_usage["completion_tokens"] == 20
    assert client.total_cost_usd > 0


def test_invalid_json_is_not_retried():
    calls = []

    def transport(payload, timeout):
        calls.append(1)
        return {"choices": [{"message": {"content": "not json"}}]}

    with pytest.raises(ProviderError):
        ZaiClient(api_key="secret", transport=transport).chat_json([])
    assert len(calls) == 1


def test_model_and_budget_are_enforced():
    with pytest.raises(ValueError):
        ZaiClient(api_key="secret", model="other")
    budget = Budget(0.000001)
    with pytest.raises(BudgetExceeded):
        ZaiClient(api_key="secret", budget=budget, transport=lambda *a: {}).chat_json([])


def test_retry_is_bounded_and_accounts_for_failed_attempt(monkeypatch):
    from urllib.error import HTTPError

    calls = []

    def transport(payload, timeout):
        calls.append(payload)
        if len(calls) == 1:
            raise HTTPError("https://example.invalid", 429, "secret body", {"Retry-After": "0"}, None)
        return {
            "choices": [{"message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    budget = Budget()
    client = ZaiClient(api_key="secret", transport=transport, budget=budget)
    assert client.chat_json([]) == {}
    assert len(calls) == 2
    assert budget.spent_usd > client.total_cost_usd
    assert budget.reserved_usd == 0


def test_auth_error_is_sanitized_and_not_retried():
    from urllib.error import HTTPError

    calls = []

    def transport(payload, timeout):
        calls.append(1)
        raise HTTPError("https://secret.example", 401, "sensitive provider body", {}, None)

    with pytest.raises(ProviderError, match="HTTP 401") as error:
        ZaiClient(api_key="secret", transport=transport).chat_json([])
    assert "sensitive" not in str(error.value)
    assert "secret" not in str(error.value)
    assert len(calls) == 1


def test_native_tool_message_preserves_calls_without_json_mode():
    calls = []
    message = {
        "content": None,
        "tool_calls": [{"id": "a", "function": {"name": "lookup", "arguments": "{}"}}],
    }

    def transport(payload, timeout):
        calls.append(payload)
        return {"choices": [{"message": message}]}

    assert (
        ZaiClient(api_key="s", transport=transport).chat_message([], tools=[{"type": "function"}]) == message
    )
    assert "response_format" not in calls[0]
    assert calls[0]["tools"] == [{"type": "function"}]
