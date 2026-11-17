"""Prove smoke checks reject stale releases and broken workflows."""
import io
import json

import pytest

from scripts.verify_release import verify


@pytest.fixture
def service(monkeypatch):
    bodies = {
        "/ready": {"status": "ready"},
        "/version": {"version": "1.0.0", "commit": "abc123"},
        "/health": {"status": "ok"},
        "/faqs": {"count": 9},
        "/demo-status/TX-CARD-001": {"status": "declined"},
        "/support": {"synthetic_only": True, "answer": "Your card is frozen.",
                     "intent": "declined_card_payment",
                     "tool_result": {"reason_code": "CARD_FROZEN"}},
    }

    def respond(request, timeout):
        path = request.full_url.removeprefix("http://localhost")
        if path == "/support":
            assert json.loads(request.data)["record_reference"] == "TX-CARD-001"
        return io.BytesIO(json.dumps(bodies[path]).encode())

    monkeypatch.setattr("scripts.verify_release.urlopen", respond)
    return bodies


def test_healthy_release(service):
    verify("http://localhost", "1.0.0", "abc123")


@pytest.mark.parametrize("field,value", [("version", "0.9.0"), ("commit", "stale")])
def test_rejects_stale_release(service, field, value):
    service["/version"][field] = value
    with pytest.raises(AssertionError, match="Unexpected"):
        verify("http://localhost", "1.0.0", "abc123")


def test_rejects_broken_support_lookup(service):
    service["/support"]["tool_result"]["reason_code"] = "WRONG"
    with pytest.raises(AssertionError, match="Support lookup failed"):
        verify("http://localhost", "1.0.0", "abc123")


def test_startup_connection_reset_is_retried(monkeypatch):
    import sys
    from scripts import verify_release

    calls = []

    def startup(*args):
        calls.append(args)
        if len(calls) == 1:
            raise ConnectionResetError("container port is starting")
    monkeypatch.setattr(verify_release, "verify", startup)
    monkeypatch.setattr(verify_release.time, "sleep", lambda _: None)
    monkeypatch.setattr(sys, "argv", ["verify", "--version", "1.0.0", "--commit", "abc", "--wait-seconds", "1"])
    verify_release.main()
    assert len(calls) == 2
