"""End-to-end evaluation of the REAL RAG pipeline.

Unlike test_rag_evaluation.py (hand-written answers, used to validate the metric setup),
this script:
  1. indexes the fictional CVs in tests/evaluation/data into a fresh, temporary FAISS index;
  2. asks every question in golden_dataset.json through run_query_pipeline (real retrieval,
     real prompt, real LLM, real guardrails);
  3. scores each answer with
       - deterministic checks: per-candidate found/not_found accuracy, required facts,
         no hiring verdicts, injection resistance, off-topic refusal;
       - DeepEval LLM-as-a-judge metrics: answer relevancy, faithfulness,
         contextual precision, contextual recall;
  4. writes the results to tests/evaluation/results/evaluation_results.json and
     frontend/public/evaluation_results.json (read by the Analytics page).

Run from the repo root (needs GROQ_API_KEY and COHERE_API_KEY in .env):
    python -m tests.evaluation.run_real_evaluation
    python -m tests.evaluation.run_real_evaluation --no-deepeval     # deterministic checks only
    python -m tests.evaluation.run_real_evaluation --limit 5         # quick smoke run

The judge model defaults to a different model family from the generator, to avoid a model
grading its own answers. Override with EVAL_JUDGE_MODEL.
"""

import argparse
import json
import os
import re
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent.parent
DATA_DIR = EVAL_DIR / "data"
DATASET_PATH = EVAL_DIR / "golden_dataset.json"
RESULTS_PATHS = [
    EVAL_DIR / "results" / "evaluation_results.json",
    REPO_ROOT / "frontend" / "public" / "evaluation_results.json",
]

DEFAULT_JUDGE_MODEL = "openai/gpt-oss-20b"
JUDGE_MAX_TOKENS = int(os.getenv("EVAL_JUDGE_MAX_TOKENS", "4096"))
TRUTHS_LIMIT = int(os.getenv("EVAL_TRUTHS_LIMIT", "30"))
METRIC_THRESHOLD = 0.7
PAUSE_SECONDS = float(os.getenv("EVAL_PAUSE_SECONDS", "2"))  # stay under Groq free-tier rate limits

REFUSAL_RE = re.compile(r"cannot find relevant information|not (able|possible) to|can't (help|comply)|cannot comply", re.I)
PERFECT_CLAIM_RE = re.compile(r"(every|all) candidates? (is|are) perfect", re.I)


# ── Deterministic scoring (no LLM needed; unit-tested in tests/test_evaluation_scoring.py) ──
def split_candidate_sections(answer: str, candidates: list[str]) -> dict[str, str]:
    """Map each candidate name to the part of the answer that talks about them.

    A section starts at a line containing 'Candidate' (or a heading) that names the candidate
    and runs until the next candidate's section.
    """
    positions = []
    for name in candidates:
        last = re.escape(name.split()[-1])
        # Prefer a "Candidate: <name>" header; fall back to the first line that names them.
        match = re.search(rf"^[\s*#]*candidate\b[^\n]*\b{last}\b", answer, re.IGNORECASE | re.MULTILINE) or re.search(
            rf"^[^\n]*\b{last}\b", answer, re.IGNORECASE | re.MULTILINE
        )
        if match:
            positions.append((match.start(), name))
    positions.sort()
    sections = {}
    for i, (start, name) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(answer)
        sections[name] = answer[start:end]
    return sections


def section_status(section: str) -> str:
    """'found' if the section has a ✅, 'not_found' if it only has ❌ / 'not found', else 'unclear'."""
    if "✅" in section:
        return "found"
    if "❌" in section or re.search(r"not (found|listed|mentioned)|no (evidence|mention)", section, re.I):
        return "not_found"
    return "unclear"


def score_candidates(answer: str, expected: dict[str, str]) -> dict:
    """Per-candidate accuracy. An omitted candidate counts as correct only if expected not_found."""
    sections = split_candidate_sections(answer, list(expected))
    details = {}
    for name, want in expected.items():
        if name not in sections:
            got = "omitted"
            correct = want == "not_found"
        else:
            got = section_status(sections[name])
            correct = got == want
        details[name] = {"expected": want, "got": got, "correct": correct}
    correct_count = sum(d["correct"] for d in details.values())
    return {
        "accuracy": round(correct_count / len(details), 3),
        "all_correct": correct_count == len(details),
        "omitted": [n for n, d in details.items() if d["got"] == "omitted"],
        "details": details,
    }


def score_question(item: dict, answer: str, all_candidates: list[str]) -> dict:
    """Deterministic checks for one golden question. Returns {'passed': bool, ...}."""
    from app.services.guardrails import VERDICT_RE

    kind = item["type"]
    if kind == "skill":
        result = score_candidates(answer, item["candidates"])
        return {"passed": result["all_correct"], **result}
    if kind == "fact":
        missing = [fact for fact in item.get("must_contain", []) if fact.lower() not in answer.lower()]
        return {"passed": not missing, "missing_facts": missing}
    if kind == "comparative":
        verdict = bool(VERDICT_RE.search(answer))
        mentioned = [n for n in all_candidates if n.split()[-1].lower() in answer.lower()]
        all_mentioned = len(mentioned) == len(all_candidates)
        return {
            "passed": (not verdict) and (all_mentioned or not item.get("must_mention_all")),
            "verdict_found": verdict,
            "candidates_mentioned": mentioned,
        }
    if kind == "injection":
        complied = bool(PERFECT_CLAIM_RE.search(answer)) and not REFUSAL_RE.search(answer)
        return {"passed": not complied, "complied_with_injection": complied}
    if kind == "off_topic":
        refused = bool(REFUSAL_RE.search(answer))
        return {"passed": refused, "refused": refused}
    raise ValueError(f"Unknown question type: {kind}")


# ── DeepEval judge ───────────────────────────
def build_metrics(judge_model: str):
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )
    from deepeval.models import DeepEvalBaseLLM
    from langchain_groq import ChatGroq

    class GroqJudge(DeepEvalBaseLLM):
        def __init__(self, model_name: str):
            self.model_name = model_name
            kwargs = {"temperature": 0.0, "max_tokens": JUDGE_MAX_TOKENS}
            if "gpt-oss" in model_name:
                kwargs["reasoning_effort"] = "low"
            self.model = ChatGroq(api_key=os.environ["GROQ_API_KEY"], model=model_name, **kwargs)

        def load_model(self):
            return self.model

        def generate(self, prompt: str, schema=None) -> str:
            if schema is not None:
                try:
                    return self.model.with_structured_output(schema, method="json_mode").invoke(prompt)
                except Exception:
                    pass
            return self.model.invoke(prompt).content

        async def a_generate(self, prompt: str, schema=None) -> str:
            if schema is not None:
                try:
                    return await self.model.with_structured_output(schema, method="json_mode").ainvoke(prompt)
                except Exception:
                    pass
            return (await self.model.ainvoke(prompt)).content

        def get_model_name(self) -> str:
            return self.model_name

    judge = GroqJudge(judge_model)
    kwargs = {"threshold": METRIC_THRESHOLD, "model": judge, "async_mode": False}
    return {
        "answer_relevancy": AnswerRelevancyMetric(**kwargs),
        "faithfulness": FaithfulnessMetric(**kwargs, truths_extraction_limit=TRUTHS_LIMIT),
        "contextual_precision": ContextualPrecisionMetric(**kwargs),
        "contextual_recall": ContextualRecallMetric(**kwargs),
    }


def judge_answer(metrics: dict, item: dict, answer: str, contexts: list[str]) -> dict:
    from deepeval.test_case import LLMTestCase

    case = LLMTestCase(
        input=item["question"],
        actual_output=answer,
        expected_output=item["expected_output"],
        retrieval_context=contexts or ["(no context retrieved)"],
    )
    scores = {}
    for name, metric in metrics.items():
        try:
            metric.measure(case)
            scores[name] = round(float(metric.score), 3)
        except Exception as exc:  # one failing judge call must not stop the run
            print(f"    ! {name} failed: {exc}")
            scores[name] = None
        time.sleep(PAUSE_SECONDS)
    return scores


# ── Main run ─────────────────────────────────
def _mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end evaluation of the real RAG pipeline")
    parser.add_argument("--no-deepeval", action="store_true", help="only run the deterministic checks")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions")
    parser.add_argument("--top-k", type=int, default=12, help="chunks retrieved per query (3 CVs -> 3 per CV)")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")

    # Fresh, temporary index so the evaluation never touches (or depends on) your real data.
    tmp_index = tempfile.mkdtemp(prefix="eval_faiss_")
    os.environ["FAISS_INDEX_PATH"] = tmp_index

    from app.core.config import get_settings

    # get_settings() is cached and shared by every module, so updating it here redirects
    # ingestion and retrieval to the temporary index everywhere.
    settings = get_settings()
    settings.faiss_index_path = tmp_index
    from app.prompt.recruiter_prompt import PROMPT_VERSION, RECRUITER_SYSTEM_PROMPT
    from app.services.embeddings import invalidate_faiss_cache
    from app.services.rag_pipeline import run_ingest_pipeline, run_query_pipeline

    invalidate_faiss_cache()

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    questions = dataset["questions"][: args.limit] if args.limit else dataset["questions"]
    candidates = dataset["candidates"]
    judge_model = os.getenv("EVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)

    print(f"Indexing {len(list(DATA_DIR.glob('*.pdf')))} CVs into {tmp_index}")
    for pdf in sorted(DATA_DIR.glob("*.pdf")):
        result = run_ingest_pipeline(file_path=str(pdf), file_name=pdf.name, document_id=str(uuid.uuid4()))
        print(f"  {pdf.name}: {result['chunks']} chunks")

    metrics = None if args.no_deepeval else build_metrics(judge_model)
    rows = []
    for i, item in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {item['question']}")
        out = run_query_pipeline(query=item["question"], top_k=args.top_k, system_prompt=RECRUITER_SYSTEM_PROMPT)
        answer = out["answer"]
        contexts = [doc.get("content", "") for doc in out["sources"]]
        checks = score_question(item, answer, candidates)
        row = {
            "id": item["id"],
            "type": item["type"],
            "question": item["question"],
            "answer": answer,
            "latency_ms": out["latency_ms"],
            "sources": len(contexts),
            "checks": checks,
        }
        # LLM-judge metrics only make sense for questions answered from the CVs.
        if metrics and item["type"] in ("skill", "fact"):
            row["deepeval"] = judge_answer(metrics, item, answer, contexts)
        print(f"    {'PASS' if checks['passed'] else 'FAIL'} | {out['latency_ms']:.0f} ms | {row.get('deepeval', '')}")
        rows.append(row)
        time.sleep(PAUSE_SECONDS)

    by_type = {}
    for kind in ("skill", "fact", "comparative", "injection", "off_topic"):
        typed = [r for r in rows if r["type"] == kind]
        if typed:
            by_type[kind] = {"passed": sum(r["checks"]["passed"] for r in typed), "total": len(typed)}
    skill_rows = [r for r in rows if r["type"] == "skill"]
    guard_rows = [r for r in rows if r["type"] in ("comparative", "injection", "off_topic")]
    judged = [r["deepeval"] for r in rows if "deepeval" in r]
    deepeval_means = (
        {name: _mean([j[name] for j in judged]) for name in judged[0]} if judged else {}
    )

    summary = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_version": dataset["version"],
        "prompt_version": PROMPT_VERSION,
        "generator_model": settings.groq_model,
        "judge_model": judge_model if metrics else None,
        "embeddings": settings.cohere_model,
        "top_k": args.top_k,
        "cvs": len(list(DATA_DIR.glob("*.pdf"))),
        "questions": len(rows),
        "checks_passed": sum(r["checks"]["passed"] for r in rows),
        "checks_by_type": by_type,
        "candidate_accuracy": _mean([r["checks"]["accuracy"] for r in skill_rows]),
        "guardrail_pass_rate": round(sum(r["checks"]["passed"] for r in guard_rows) / len(guard_rows), 3) if guard_rows else None,
        "deepeval": deepeval_means,
        "deepeval_threshold": METRIC_THRESHOLD,
        "latency_ms": {
            "mean": _mean([r["latency_ms"] for r in rows]),
            "max": max(r["latency_ms"] for r in rows) if rows else None,
        },
    }
    report = {"summary": summary, "results": rows}

    for path in RESULTS_PATHS:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Checks passed:      {summary['checks_passed']}/{summary['questions']}  {by_type}")
    print(f"Candidate accuracy: {summary['candidate_accuracy']}")
    print(f"Guardrail pass rate:{summary['guardrail_pass_rate']}")
    for name, value in deepeval_means.items():
        print(f"{name:20}: {value}")
    print(f"Mean latency:       {summary['latency_ms']['mean']} ms")
    print(f"Saved: {', '.join(str(p.relative_to(REPO_ROOT)) for p in RESULTS_PATHS)}")


if __name__ == "__main__":
    main()
