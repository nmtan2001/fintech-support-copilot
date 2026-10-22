"""Read-only lookups over an explicitly synthetic SQLite banking dataset."""

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_INTENTS = {
    "declined_card_payment": "transaction",
    "pending_transfer": "transaction",
    "transfer_not_received_by_recipient": "transaction",
    "lost_or_stolen_card": "card",
}
REFERENCE_PATTERN = re.compile(r"\b(?:TX-CARD|TX-TRF|CARD)-\d{3}\b", re.IGNORECASE)


@dataclass(frozen=True)
class ToolResult:
    name: str
    reference: str
    found: bool
    status: str | None
    reason_code: str | None

    def validate_for(self, intent: str, reference: str) -> None:
        expected_name = DemoBankingTools.validate(intent, reference)
        valid_data = (
            all(isinstance(value, str) and value.strip() and "\n" not in value
                for value in (self.status, self.reason_code))
            if self.found is True else self.status is None and self.reason_code is None
        )
        if (expected_name is None or self.name != expected_name or
                self.reference != reference.strip().upper() or type(self.found) is not bool or
                not valid_data):
            raise RuntimeError("Invalid tool result for the authorized demo lookup")

    def render(self) -> str:
        if not self.found:
            return f"Synthetic lookup: no demo record found for {self.reference}."
        return (
            f"Synthetic lookup for {self.reference}: status={self.status}, "
            f"reason_code={self.reason_code}. This is a simulated record, not a real account."
        )


def extract_reference(question: str) -> str | None:
    matches = {match.upper() for match in REFERENCE_PATTERN.findall(question)}
    if len(matches) > 1:
        raise ValueError("Provide only one demo reference per request")
    return next(iter(matches)) if matches else None


def initialize_demo_database(path: Path, fixture_path: Path) -> None:
    """Initialize once from a trusted fixture, preserving any existing database."""
    if path.exists():
        return
    records = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Demo records must be a nonempty list")
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS demo_records ("
            "reference TEXT PRIMARY KEY, kind TEXT NOT NULL CHECK (kind IN ('card','transaction')), "
            "status TEXT NOT NULL, reason_code TEXT NOT NULL)"
        )
        if connection.execute("SELECT COUNT(*) FROM demo_records").fetchone()[0]:
            return
        connection.executemany(
            "INSERT INTO demo_records(reference,kind,status,reason_code) VALUES(?,?,?,?)",
            [
                (row["reference"], row["kind"], row["status"], row["reason_code"])
                for row in records
            ],
        )


class DemoBankingTools:
    def __init__(self, database_path: Path):
        self.database_path = Path(database_path)

    @staticmethod
    def validate(intent: str, reference: str) -> str | None:
        kind = SUPPORTED_INTENTS.get(intent)
        if kind is None:
            return None
        reference = reference.strip().upper()
        if not REFERENCE_PATTERN.fullmatch(reference):
            raise ValueError("Use a demo reference such as TX-CARD-001, TX-TRF-001 or CARD-001")
        if kind == "card" and not reference.startswith("CARD-"):
            raise ValueError("Lost card lookup requires a CARD-xxx reference")
        if kind == "transaction" and not reference.startswith("TX-"):
            raise ValueError("Payment or transfer lookup requires a TX-xxx reference")
        if intent == "declined_card_payment" and not reference.startswith("TX-CARD-"):
            raise ValueError("Declined payment lookup requires a TX-CARD-xxx reference")
        if intent in {"pending_transfer", "transfer_not_received_by_recipient"} and not reference.startswith("TX-TRF-"):
            raise ValueError("Transfer lookup requires a TX-TRF-xxx reference")
        return "lookup_demo_card" if kind == "card" else "lookup_demo_transaction"

    def call(self, intent: str, reference: str) -> ToolResult | None:
        name = self.validate(intent, reference)
        if name is None:
            return None
        kind = SUPPORTED_INTENTS[intent]
        reference = reference.strip().upper()

        if not self.database_path.is_file():
            raise FileNotFoundError(f"Missing synthetic demo database: {self.database_path}")
        uri = f"{self.database_path.resolve().as_uri()}?mode=ro"
        try:
            with sqlite3.connect(uri, uri=True) as connection:
                row = connection.execute(
                    "SELECT status, reason_code FROM demo_records WHERE reference = ? AND kind = ?",
                    (reference, kind),
                ).fetchone()
        except sqlite3.Error as exc:
            raise RuntimeError("Could not read synthetic demo database; check its schema and integrity") from exc
        result = ToolResult(
            name=name,
            reference=reference,
            found=row is not None,
            status=row[0] if row else None,
            reason_code=row[1] if row else None,
        )
        result.validate_for(intent, reference)
        return result
