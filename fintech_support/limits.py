"""Bound provider work globally and request frequency per fictional customer."""

import secrets
import threading
import time
from contextlib import contextmanager


class CapacityExceeded(RuntimeError):
    pass


class TurnLimiter:
    def __init__(self, *, max_concurrent=2, requests_per_minute=20, clock=time.time):
        self.maximum, self.rate, self.clock = max_concurrent, requests_per_minute, clock
        self._lock = threading.Lock()
        self._active = 0
        self._requests = {}

    @contextmanager
    def turn(self, customer):
        with self._lock:
            window = int(self.clock() // 60)
            self._requests = {key: value for key, value in self._requests.items() if key[1] == window}
            key = (customer, window)
            if self._active >= self.maximum or self._requests.get(key, 0) >= self.rate:
                raise CapacityExceeded("Showcase capacity reached. Wait before sending another message")
            self._requests[key] = self._requests.get(key, 0) + 1
            self._active += 1
        try:
            yield
        finally:
            with self._lock:
                self._active -= 1


class RedisTurnLimiter:
    def __init__(self, client, *, prefix="copilot:usage", max_concurrent=2, requests_per_minute=20):
        self.client, self.prefix = client, prefix
        self.maximum, self.rate = max_concurrent, requests_per_minute

    @contextmanager
    def turn(self, customer):
        lease = secrets.token_urlsafe(24)
        script = """
        local t=redis.call('TIME'); local now=tonumber(t[1])+tonumber(t[2])/1000000
        redis.call('ZREMRANGEBYSCORE',KEYS[1],'-inf',now)
        if redis.call('ZCARD',KEYS[1])>=tonumber(ARGV[1]) then return 0 end
        local window=tostring(math.floor(now/60)); local ratekey=KEYS[2]..':'..window
        if tonumber(redis.call('GET',ratekey) or '0')>=tonumber(ARGV[2]) then return 0 end
        redis.call('INCR',ratekey); redis.call('EXPIRE',ratekey,120)
        redis.call('ZADD',KEYS[1],now+90,ARGV[3]); redis.call('EXPIRE',KEYS[1],120)
        return 1
        """
        try:
            admitted = self.client.eval(script, 2, self.prefix + ":active", self.prefix + ":requests:" + customer,
                                        self.maximum, self.rate, lease)
        except Exception:
            raise CapacityExceeded("Showcase usage controls are unavailable. Try again later") from None
        if not admitted:
            raise CapacityExceeded("Showcase capacity reached. Wait before sending another message")
        try:
            yield
        finally:
            try:
                self.client.zrem(self.prefix + ":active", lease)
            except Exception:
                # The lease expires even if Redis becomes unavailable during cleanup.
                import logging
                logging.getLogger(__name__).warning("Could not release the showcase slot; its lease will expire")
