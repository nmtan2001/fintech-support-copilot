"""HTTP adapter preserving the workbench's existing support result contract."""

from urllib.parse import urlsplit

import requests

from fintech_support.copilot import SupportResult
from fintech_support.demo_tools import ToolResult
from fintech_support.retrieval import FAQ, Hit


class SupportAPIClient:
    def __init__(self, base_url, *, transport=None):
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("SUPPORT_API_URL must be an HTTP(S) URL without credentials")
        self.base_url = base_url.rstrip("/")
        self.transport = transport or requests.Session()
        self.request_id = None

    def _conversation_request(self, method, path, *, session=None, payload=None):
        headers = {"Authorization": "Bearer " + session["token"]} if session else {}
        try:
            response = self.transport.request(method, self.base_url + path, headers=headers, json=payload, timeout=90)
        except requests.RequestException as exc:
            raise RuntimeError("Conversation API is unreachable. Check the showcase cluster.") from exc
        self.request_id = response.headers.get("x-request-id")
        if response.status_code == 204:
            return None
        if response.status_code not in {200, 201}:
            if response.status_code == 404:
                raise RuntimeError("Conversation expired. Start a new conversation.")
            try:
                detail = response.json().get("detail", "Conversation request failed")
            except ValueError:
                detail = "Conversation request failed"
            raise RuntimeError(f"{detail} (HTTP {response.status_code})")
        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError("Conversation API returned an invalid response") from exc

    def demo_profiles(self):
        return self._conversation_request("GET", "/demo-customers")["customers"]

    def create_conversation(self, customer_id="DEMO-ALPHA"):
        return self._conversation_request("POST", "/conversations", payload={"customer_id": customer_id})

    def conversation_cases(self, session):
        return self._conversation_request("GET", f"/conversations/{session['id']}/cases", session=session)["cases"]

    def conversation_message(self, session, message):
        return self._conversation_request("POST", f"/conversations/{session['id']}/messages",
                                          session=session, payload={"message": message})

    def delete_conversation(self, session):
        self._conversation_request("DELETE", f"/conversations/{session['id']}", session=session)

    def respond(self, question, *, record_reference=None, min_confidence=0.0):
        self.request_id = None
        try:
            response = self.transport.post(self.base_url + "/support", json={
                "question": question, "record_reference": record_reference,
                "min_confidence": min_confidence,
            }, timeout=60)
        except requests.RequestException as exc:
            raise RuntimeError("Support API is unreachable. Check the showcase cluster and localhost service mapping.") from exc
        self.request_id = response.headers.get("x-request-id")
        if response.status_code in {400, 422}:
            raise ValueError(str(response.json().get("detail", "Invalid support request")))
        if response.status_code != 200:
            raise RuntimeError(f"Support API returned HTTP {response.status_code}. Check backend readiness.")
        try:
            body = response.json()
            if body["synthetic_only"] is not True:
                raise ValueError("Synthetic boundary missing")
            hits = tuple(Hit(FAQ(row["id"], row["title"], row["content"],
                                 tuple(row["intents"]), row["source"]), row["score"])
                         for row in body["retrieved_sources"])
            tool = ToolResult(**body["tool_result"]) if body["tool_result"] else None
            if tool:
                tool.validate_for(body["intent"], tool.reference)
            return SupportResult(body["intent"], body["confidence_uncalibrated"], body["answer"],
                                 hits, body["status"], tool, body["tool_routing"],
                                 body["generation_status"], tuple(body["grounding_issues"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Support API returned an incompatible response. Update the backend image.") from exc
