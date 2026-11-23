"""Bounded native Z.ai JSON calls. No prompt or credential logging."""

import json
import math
import os
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MODEL = "glm-5.3-flash"
ENDPOINT = "https://api.z.ai/api/paas/v4/chat/completions"
PRICING_DATE = "2026-11-19"
INPUT_PER_MILLION = 0.15
OUTPUT_PER_MILLION = 0.50


class ProviderError(RuntimeError):
    """Sanitized provider failure."""


class BudgetExceeded(ProviderError):
    """The worst-case request would exceed the run budget."""


class Budget:
    def __init__(self, limit_usd=5.0):
        if not 0 < limit_usd <= 5:
            raise ValueError("Paid evaluation budget must be in (0, 5] USD")
        self.limit_usd = limit_usd
        self.spent_usd = 0.0
        self.reserved_usd = 0.0
        self._lock = threading.Lock()

    def reserve(self, amount):
        with self._lock:
            if self.spent_usd + self.reserved_usd + amount > self.limit_usd:
                raise BudgetExceeded("Paid evaluation budget exhausted")
            self.reserved_usd += amount

    def settle(self, reserved, actual):
        with self._lock:
            self.reserved_usd -= reserved
            self.spent_usd += actual


class RedisBudget:
    def __init__(self, client, *, key="copilot:budget:runtime", limit_usd=5.0):
        if not 0 < limit_usd <= 5:
            raise ValueError("Showcase budget must be in (0, 5] USD")
        self.client, self.key = client, key
        self.limit = math.floor(limit_usd * 1_000_000)

    def reserve(self, amount):
        script = """
        local reserved=tonumber(redis.call('HGET',KEYS[1],'reserved') or '0')
        local spent=tonumber(redis.call('HGET',KEYS[1],'spent') or '0')
        if reserved+spent+tonumber(ARGV[1])>tonumber(ARGV[2]) then return 0 end
        redis.call('HINCRBY',KEYS[1],'reserved',ARGV[1]); redis.call('EXPIRE',KEYS[1],86400)
        return 1
        """
        try:
            accepted = self.client.eval(script, 1, self.key, math.ceil(amount * 1_000_000), self.limit)
        except Exception:
            raise BudgetExceeded("Showcase spending controls are unavailable") from None
        if not accepted:
            raise BudgetExceeded("Shared showcase budget exhausted")

    def settle(self, reserved, actual):
        script = """
        redis.call('HINCRBY',KEYS[1],'reserved',-tonumber(ARGV[1]))
        redis.call('HINCRBY',KEYS[1],'spent',ARGV[2]); return 1
        """
        try:
            self.client.eval(script, 1, self.key, math.ceil(reserved * 1_000_000), math.ceil(actual * 1_000_000))
        except Exception:
            raise BudgetExceeded("Could not record provider usage; generation is paused") from None


class ZaiClient:
    def __init__(self, api_key=None, *, model=MODEL, transport=None, budget=None):
        if model != MODEL:
            raise ValueError("Only glm-5.3-flash is permitted")
        self.api_key = api_key or os.environ.get("ZAI_API_KEY")
        self.model = model
        self.transport = transport or self._http
        self.budget = budget
        self.last_usage = {}
        self.total_cost_usd = 0.0

    def _http(self, payload, timeout):
        request = Request(
            ENDPOINT,
            data=json.dumps(payload).encode(),
            headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"},
        )
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ProviderError("Provider response exceeds size limit")
        return json.loads(raw)

    def _chat(self, messages, *, timeout=30, max_tokens=4096, json_mode=False, tools=None):
        if not self.api_key:
            raise ProviderError("ZAI_API_KEY is not configured")
        if not 1 <= max_tokens <= 4096 or not 0 < timeout <= 60:
            raise ValueError("Invalid provider request bounds")
        # UTF-8 bytes conservatively upper-bound normal tokenizer input length.
        input_bound = (
            len(json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False).encode()) + 1024
        )
        reserved = (input_bound * INPUT_PER_MILLION + max_tokens * OUTPUT_PER_MILLION) / 1e6
        payload = {"model": MODEL, "messages": messages, "stream": False, "max_tokens": max_tokens,
                   "thinking": {"type": "enabled"}, "reasoning_effort": "low"}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if tools:
            payload["tools"] = tools
        deadline = time.monotonic() + timeout
        for attempt in range(2):
            if self.budget:
                self.budget.reserve(reserved)
            try:
                response = self.transport(payload, max(0.001, deadline - time.monotonic()))
            except HTTPError as exc:
                # Failed requests may have consumed tokens; retain their reservation as spend.
                if self.budget:
                    self.budget.settle(reserved, reserved)
                retryable = exc.code == 429 or 500 <= exc.code <= 599
                try:
                    delay = max(0.0, float(exc.headers.get("Retry-After", "1")))
                except (ValueError, AttributeError):
                    delay = 1.0
                if attempt == 0 and retryable and time.monotonic() + delay < deadline:
                    time.sleep(delay)
                    continue
                raise ProviderError(f"Z.ai request failed (HTTP {exc.code})") from None
            except (URLError, OSError, ValueError, ProviderError):
                if self.budget:
                    self.budget.settle(reserved, reserved)
                raise ProviderError("Z.ai transport or response failed") from None
            usage = response.get("usage", {}) if isinstance(response, dict) else {}
            try:
                prompt = max(0, int(usage["prompt_tokens"]))
                completion = max(0, int(usage["completion_tokens"]))
                cost = (prompt * INPUT_PER_MILLION + completion * OUTPUT_PER_MILLION) / 1e6
            except (KeyError, ValueError, TypeError):
                prompt, completion, cost = input_bound, max_tokens, reserved
            if self.budget:
                self.budget.settle(reserved, cost)
            self.total_cost_usd += cost
            self.last_usage = {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "estimated_cost_usd": cost,
                "model": MODEL,
                "pricing_date": PRICING_DATE,
            }
            try:
                message = response["choices"][0]["message"]
                if not isinstance(message, dict):
                    raise ValueError()
                return message
            except (KeyError, IndexError, TypeError, ValueError):
                raise ProviderError("Z.ai returned invalid completion") from None
        raise ProviderError("Z.ai request failed")

    def chat_message(self, messages, *, tools=None, timeout=30, max_tokens=4096):
        return self._chat(messages, tools=tools, timeout=timeout, max_tokens=max_tokens)

    def chat_json(self, messages, *, timeout=30, max_tokens=4096):
        message = self._chat(messages, timeout=timeout, max_tokens=max_tokens, json_mode=True)
        try:
            value = json.loads(message.get("content"))
            if not isinstance(value, dict):
                raise ValueError()
            return value
        except (ValueError, TypeError):
            raise ProviderError("Z.ai returned invalid structured output") from None
