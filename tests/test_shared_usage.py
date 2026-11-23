"""Usage controls apply across workers and release slots after failures."""

import pytest


def test_local_turn_limiter_blocks_overlap_and_counts_requests():
    from fintech_support.limits import CapacityExceeded, TurnLimiter

    limiter = TurnLimiter(max_concurrent=1, requests_per_minute=2)
    with limiter.turn("DEMO-ALPHA"):
        with pytest.raises(CapacityExceeded):
            with limiter.turn("DEMO-BETA"):
                pass
    with limiter.turn("DEMO-ALPHA"):
        pass
    with pytest.raises(CapacityExceeded):
        with limiter.turn("DEMO-ALPHA"):
            pass


def test_failed_turn_releases_capacity():
    from fintech_support.limits import TurnLimiter

    limiter = TurnLimiter(max_concurrent=1)
    with pytest.raises(ValueError):
        with limiter.turn("DEMO-ALPHA"):
            raise ValueError("failure")
    with limiter.turn("DEMO-BETA"):
        pass


@pytest.fixture
def redis_client():
    import os
    import redis

    url = os.environ.get("REDIS_TEST_URL")
    if not url:
        pytest.skip("Set REDIS_TEST_URL to an isolated Redis test container")
    client = redis.Redis.from_url(url, decode_responses=True, socket_timeout=3)
    client.ping()
    return client


def test_redis_budget_is_shared_across_workers(redis_client):
    import uuid
    from fintech_support.zai import BudgetExceeded, RedisBudget

    key = "test:budget:" + uuid.uuid4().hex
    first = RedisBudget(redis_client, key=key, limit_usd=0.01)
    second = RedisBudget(redis_client, key=key, limit_usd=0.01)
    first.reserve(0.007)
    with pytest.raises(BudgetExceeded):
        second.reserve(0.004)
    first.settle(0.007, 0.005)
    second.reserve(0.004)
    second.settle(0.004, 0.004)
    with pytest.raises(BudgetExceeded):
        first.reserve(0.002)


def test_redis_limiter_is_shared_and_releases_owned_slot(redis_client):
    import uuid
    from fintech_support.limits import CapacityExceeded, RedisTurnLimiter

    prefix = "test:limiter:" + uuid.uuid4().hex
    first = RedisTurnLimiter(redis_client, prefix=prefix, max_concurrent=1, requests_per_minute=2)
    second = RedisTurnLimiter(redis_client, prefix=prefix, max_concurrent=1, requests_per_minute=2)
    with first.turn("DEMO-ALPHA"):
        with pytest.raises(CapacityExceeded):
            with second.turn("DEMO-BETA"):
                pass
    with second.turn("DEMO-ALPHA"):
        pass
    with pytest.raises(CapacityExceeded):
        with first.turn("DEMO-ALPHA"):
            pass


def test_redis_session_history_and_cases_cross_worker_boundaries(redis_client):
    from fintech_support.sessions import RedisSessionStore, SessionNotFound

    first = RedisSessionStore(client=redis_client)
    second = RedisSessionStore(client=redis_client)
    session = first.create(customer_id="DEMO-BETA")
    with first.turn(session["id"], session["token"]) as state:
        state["case_state"] = {"issue": "dispute", "active_reference": "TX-CARD-006"}
        state["cases"].append({"case_id": "CASE-TEST"})
    state = second.get(session["id"], session["token"])
    assert state["customer_id"] == "DEMO-BETA"
    assert state["case_state"]["active_reference"] == "TX-CARD-006"
    assert state["cases"][0]["case_id"] == "CASE-TEST"
    second.delete(session["id"], session["token"])
    with pytest.raises(SessionNotFound):
        first.get(session["id"], session["token"])
