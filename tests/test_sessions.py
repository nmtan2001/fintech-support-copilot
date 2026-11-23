import pytest
from fintech_support.sessions import InMemorySessionStore, SessionNotFound, SessionBusy, SessionLimit


def test_sessions_commit_isolation_expiry_and_delete():
    now = [100.0]
    store = InMemorySessionStore(clock=lambda: now[0], ttl=30)
    created = store.create()
    sid = created["id"]
    token = created["token"]
    with pytest.raises(SessionNotFound):
        store.get(sid, "wrong")
    with store.turn(sid, token) as state:
        state["messages"].append({"role": "user", "content": "hello"})
        with pytest.raises(SessionBusy):
            with store.turn(sid, token):
                pass
    assert store.get(sid, token)["turns"] == 1
    detached = store.get(sid, token)
    detached["messages"].clear()
    assert len(store.get(sid, token)["messages"]) == 1
    now[0] = 131
    with pytest.raises(SessionNotFound):
        store.get(sid, token)


def test_failed_turn_does_not_commit_and_limit():
    store = InMemorySessionStore(max_turns=1)
    c = store.create()
    with pytest.raises(ValueError):
        with store.turn(c["id"], c["token"]) as state:
            state["references"].append("CARD-001")
            raise ValueError("fail")
    assert not store.get(c["id"], c["token"])["references"]
    with store.turn(c["id"], c["token"]):
        pass
    with pytest.raises(SessionLimit):
        with store.turn(c["id"], c["token"]):
            pass
    store.delete(c["id"], c["token"])
    with pytest.raises(SessionNotFound):
        store.get(c["id"], c["token"])


def test_deletion_during_turn_cannot_resurrect_session():
    store = InMemorySessionStore()
    c = store.create()
    with store.turn(c["id"], c["token"]):
        with pytest.raises(SessionBusy):
            store.delete(c["id"], c["token"])
    store.delete(c["id"], c["token"])
    with pytest.raises(SessionNotFound):
        store.get(c["id"], c["token"])


def test_redis_unavailable_is_sanitized():
    from fintech_support.sessions import RedisSessionStore, SessionUnavailable

    class Broken:
        def get(self, *args):
            raise OSError("password=secret")

    with pytest.raises(SessionUnavailable) as error:
        RedisSessionStore(client=Broken()).get("valid", "token")
    assert "secret" not in str(error.value)
