"""Conservative, extractive checks against attributed FAQ evidence."""

import re
from dataclasses import dataclass

from fintech_support.demo_tools import ToolResult
from fintech_support.retrieval import Hit


FALLBACK = (
    "This demo does not have a verified support article for your request. "
    "Please contact your financial provider through its official support channel."
)
CITATION_PATTERN = re.compile(r"\[([^\[\]\n]+)\]")


@dataclass(frozen=True)
class GroundingAssessment:
    citations_valid: bool
    faithful: bool
    issues: tuple[str, ...]


class GroundingError(RuntimeError):
    def __init__(self, answer: str, assessment: GroundingAssessment):
        self.answer = answer
        self.assessment = assessment
        super().__init__("LLM grounding failed: " + "; ".join(assessment.issues))


def assess_answer(
    answer: str, hits: tuple[Hit, ...], *, tool_result: ToolResult | None = None,
    allow_fallback: bool = False,
) -> GroundingAssessment:
    """Require complete FAQ sentences; paraphrase entailment is not estimated."""
    text = answer.strip()
    if tool_result and text.startswith(tool_result.render()):
        text = text[len(tool_result.render()):].strip()
    if allow_fallback and text == FALLBACK:
        return GroundingAssessment(True, True, ())
    if text.startswith("Demo guidance: "):
        text = text[len("Demo guidance: "):]
    sources = {hit.faq.id: hit.faq.content for hit in hits}
    issues = []
    citations_valid = True
    if not text:
        issues.append("empty answer without FAQ evidence")
        citations_valid = False
    for line in text.splitlines():
        if not line.strip():
            continue
        citations = CITATION_PATTERN.findall(line)
        if not citations:
            citations_valid = False
            issues.append("missing FAQ citation")
        if any(citation not in sources for citation in citations):
            citations_valid = False
            issues.append("citation outside the retrieved FAQ evidence")
        claim = " ".join(CITATION_PATTERN.sub("", line).split())
        supported = False
        for citation in citations:
            sentences = re.split(r"(?<=[.!?])\s+", sources.get(citation, ""))
            for start in range(len(sentences)):
                for end in range(start + 1, len(sentences) + 1):
                    if claim == " ".join(" ".join(sentences[start:end]).split()):
                        supported = True
        if not supported:
            citations_valid = False
            issues.append("claim is not a complete extract from its cited FAQ evidence")
    return GroundingAssessment(citations_valid, not issues, tuple(dict.fromkeys(issues)))
