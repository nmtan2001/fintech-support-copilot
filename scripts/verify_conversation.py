"""Verify a cited answer and follow-up through the deployed conversation API."""

import argparse

from fintech_support.api_client import SupportAPIClient


def verify_conversation(url, *, require_generation=False):
    client = SupportAPIClient(url)
    session = client.create_conversation()
    try:
        for index, message in enumerate(("Why was my card payment TX-CARD-001 declined?", "What is its status now?")):
            response = client.conversation_message(session, message)
            if response.get("synthetic_only") is not True or not response.get("answer") or not response.get("citations"):
                raise RuntimeError("Conversation did not return attributed synthetic evidence")
            if require_generation:
                if index == 0 and response["generation_status"] != "accepted":
                    reason = response.get("fallback_reason", "unavailable")
                    raise RuntimeError(f"Z.ai generation was not accepted ({reason}). Check provider and grounding results")
                if not any(row["reference"] == "TX-CARD-001" for row in response["tool_results"]):
                    raise RuntimeError("The conversational follow-up did not retain the authorized record reference")
        return {"verified": True, "generation_required": require_generation}
    finally:
        client.delete_conversation(session)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--require-generation", action="store_true")
    args = parser.parse_args()
    print(verify_conversation(args.url, require_generation=args.require_generation))


if __name__ == "__main__":
    main()
