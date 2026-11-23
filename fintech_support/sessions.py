"""Expiring capability-token sessions shared across API replicas."""

import copy
import hashlib
import hmac
import json
import secrets
import threading
import time
from contextlib import contextmanager


class SessionNotFound(RuntimeError):
    pass


class SessionBusy(RuntimeError):
    pass


class SessionLimit(RuntimeError):
    pass


class SessionUnavailable(RuntimeError):
    pass


def _hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


class InMemorySessionStore:
    """Explicit offline/test store; production uses RedisSessionStore."""

    def __init__(self, *, ttl=1800, max_turns=20, clock=time.time):
        self.ttl, self.max_turns, self.clock = ttl, max_turns, clock
        self._data = {}
        self._busy = set()
        self._lock = threading.RLock()

    def create(self, customer_id="DEMO-ALPHA"):
        sid, token = secrets.token_urlsafe(24), secrets.token_urlsafe(32)
        expires = self.clock() + self.ttl
        with self._lock:
            self._data[sid] = {
                "id": sid,
                "token_hash": _hash(token),
                "expires_at": expires,
                "messages": [],
                "references": [],
                "turns": 0,
                "customer_id": customer_id,
                "case_state": {},
                "cases": [],
            }
        return {"id": sid, "token": token, "expires_at": expires}

    def get(self, sid, token):
        with self._lock:
            state = self._data.get(sid)
            if (
                not state
                or state["expires_at"] <= self.clock()
                or not hmac.compare_digest(state["token_hash"], _hash(token))
            ):
                raise SessionNotFound("Conversation not found")
            return {k: copy.deepcopy(v) for k, v in state.items() if k != "token_hash"}

    def delete(self, sid, token):
        with self._lock:
            self.get(sid, token)
            if sid in self._busy:
                raise SessionBusy("Conversation is processing a turn")
            del self._data[sid]

    @contextmanager
    def turn(self, sid, token):
        with self._lock:
            state = self.get(sid, token)
            if sid in self._busy:
                raise SessionBusy("Conversation is processing a turn")
            if state["turns"] >= self.max_turns:
                raise SessionLimit("Conversation turn limit reached")
            self._busy.add(sid)
        try:
            yield state
            with self._lock:
                self.get(sid, token)
                state["turns"] += 1
                state["messages"] = state["messages"][-40:]
                state["references"] = list(dict.fromkeys(state["references"]))[-60:]
                state["expires_at"] = self.clock() + self.ttl
                self._data[sid] = {**copy.deepcopy(state), "token_hash": _hash(token)}
        finally:
            with self._lock:
                self._busy.discard(sid)


class RedisSessionStore:
    """Redis state with ownership-checked lease and atomic commit/release."""

    def __init__(self, url=None, *, client=None, ttl=1800, max_turns=20, clock=time.time):
        if client is None:
            import redis

            client = redis.Redis.from_url(
                url, decode_responses=True, socket_timeout=3, socket_connect_timeout=3
            )
        self.client, self.ttl, self.max_turns, self.clock = client, ttl, max_turns, clock

    def _key(self, sid):
        if (
            not isinstance(sid, str)
            or len(sid) > 80
            or not sid
            or not all(c.isalnum() or c in "-_" for c in sid)
        ):
            raise SessionNotFound("Conversation not found")
        return "copilot:conversation:" + sid

    def _call(self, method, *args, **kwargs):
        try:
            return getattr(self.client, method)(*args, **kwargs)
        except Exception:
            raise SessionUnavailable("Conversation storage unavailable") from None

    def create(self, customer_id="DEMO-ALPHA"):
        sid, token = secrets.token_urlsafe(24), secrets.token_urlsafe(32)
        expires = self.clock() + self.ttl
        state = {
            "id": sid,
            "token_hash": _hash(token),
            "expires_at": expires,
            "messages": [],
            "references": [],
            "turns": 0,
                "customer_id": customer_id,
                "case_state": {},
                "cases": [],
        }
        self._call("set", self._key(sid), json.dumps(state), ex=self.ttl)
        return {"id": sid, "token": token, "expires_at": expires}

    def get(self, sid, token):
        raw = self._call("get", self._key(sid))
        if not raw:
            raise SessionNotFound("Conversation not found")
        try:
            state = json.loads(raw)
            valid = state["expires_at"] > self.clock() and hmac.compare_digest(
                state["token_hash"], _hash(token)
            )
        except (ValueError, KeyError, TypeError):
            raise SessionUnavailable("Conversation state invalid") from None
        if not valid:
            raise SessionNotFound("Conversation not found")
        return {k: v for k, v in state.items() if k != "token_hash"}

    def delete(self, sid, token):
        self.get(sid, token)
        key = self._key(sid)
        script = "if redis.call('exists',KEYS[2])==1 then return 0 end; redis.call('del',KEYS[1]); return 1"
        if not self._call("eval", script, 2, key, key + ":lock"):
            raise SessionBusy("Conversation is processing a turn")

    @contextmanager
    def turn(self, sid, token):
        state = self.get(sid, token)
        key = self._key(sid)
        lease = secrets.token_urlsafe(32)
        if not self._call("set", key + ":lock", lease, nx=True, ex=120):
            raise SessionBusy("Conversation is processing a turn")
        try:
            state = self.get(sid, token)
            if state["turns"] >= self.max_turns:
                raise SessionLimit("Conversation turn limit reached")
            yield state
            state["turns"] += 1
            state["messages"] = state["messages"][-40:]
            state["references"] = list(dict.fromkeys(state["references"]))[-60:]
            state["expires_at"] = self.clock() + self.ttl
            state["token_hash"] = _hash(token)
            script = (
                "if redis.call('get',KEYS[2])~=ARGV[1] or redis.call('exists',KEYS[1])==0 then return 0 end; "
                "redis.call('set',KEYS[1],ARGV[2],'EX',ARGV[3]); return 1"
            )
            if not self._call("eval", script, 2, key, key + ":lock", lease, json.dumps(state), self.ttl):
                raise SessionNotFound("Conversation expired during turn")
        finally:
            self._call(
                "eval",
                "if redis.call('get',KEYS[1])==ARGV[1] then return redis.call('del',KEYS[1]) end; return 0",
                1,
                key + ":lock",
                lease,
            )
