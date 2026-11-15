"""Output guardrails for recruiter answers.

The system prompt already forbids hiring verdicts, but an LLM can still slip
("...making him the strongest candidate"). These deterministic checks enforce the
human-in-the-loop rule in code, so it does not depend on the model obeying the prompt:

1. Comparative questions ("who is the best?") get an explicit reminder appended
   before generation: report evidence for every candidate, no verdict.
2. Any sentence in the answer that still declares a winner is removed, and a short
   note tells the recruiter that the decision is theirs.
"""

import re

COMPARATIVE_QUESTION_RE = re.compile(
    r"\b(best|strongest|top|most (suitable|qualified|experienced)|winner|rank|ranking|"
    r"who should (we|i) (hire|choose|pick)|which (one|candidate) (is|should))\b",
    re.IGNORECASE,
)

VERDICT_RE = re.compile(
    r"\b(the )?(best|strongest|top|most suitable|most qualified|ideal|preferred) "
    r"(candidate|choice|fit|option)\b"
    r"|\b(clear|obvious) (winner|choice)\b"
    r"|\bwe (recommend|suggest) (hiring|choosing)\b"
    r"|\bshould (be hired|hire)\b",
    re.IGNORECASE,
)

NO_VERDICT_REMINDER = (
    "\n\n(Reminder: report the evidence for EVERY candidate in the context, one short section each, "
    "at most 3 bullets per candidate: current role and years of experience, key skills, one notable result. "
    "Do not declare a best, strongest or recommended candidate: the recruiter decides.)"
)

VERDICT_REMOVED_NOTE = (
    "Note: comparative verdicts are not provided. Compare the evidence above; the recruiter makes the decision."
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def is_comparative_question(question: str) -> bool:
    """True if the question asks to rank or pick a candidate."""
    return bool(COMPARATIVE_QUESTION_RE.search(question or ""))


def add_no_verdict_reminder(question: str) -> str:
    """Append the no-verdict reminder to comparative questions; return others unchanged."""
    return question + NO_VERDICT_REMINDER if is_comparative_question(question) else question


def remove_verdicts(answer: str, add_note: bool = True) -> tuple[str, bool]:
    """Drop sentences that declare a winner. Returns (clean_answer, removed_anything).

    ``add_note=False`` is for short fields (e.g. a job-match summary) where the note would be noise.
    """
    if not answer:
        return answer, False

    removed = False
    kept_lines = []
    for line in answer.split("\n"):
        if not VERDICT_RE.search(line):
            kept_lines.append(line)
            continue
        sentences = [s for s in _SENTENCE_SPLIT_RE.split(line) if not VERDICT_RE.search(s)]
        removed = True
        if sentences:
            kept_lines.append(" ".join(sentences))

    clean = "\n".join(kept_lines).rstrip()
    if removed and add_note:
        clean = f"{clean}\n\n{VERDICT_REMOVED_NOTE}"
    return clean, removed