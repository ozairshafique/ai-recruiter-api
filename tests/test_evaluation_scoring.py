"""Unit tests for the deterministic scoring used by tests/evaluation/run_real_evaluation.py.

The example answers are copied from real responses of the live demo.
"""

import json
from pathlib import Path

from tests.evaluation.run_real_evaluation import DATASET_PATH, score_candidates, score_question

CANDIDATES = ["Sarah Chen", "James Okafor", "David Kim", "Marco Bianchi"]

KAFKA_ANSWER = """Candidate: James Okafor
Kafka: ✅ Found
• Infrastructure section lists Kafka as a technology

Candidate: Sarah Chen
Kafka: ❌ Not found in documents"""

SNOWFLAKE_ANSWER = """Candidate: Sarah Chen
Snowflake: ✅ Found
• Data: Spark, Pandas, NumPy, PostgreSQL, Snowflake (Source 3)"""

BEST_WITH_VERDICT = """Candidate: James Okafor
Programming: ✅ Found
• Python (expert, 7 years)

Overall, James Okafor demonstrates extensive AI/ML expertise, making him the strongest candidate based on the provided context."""


def test_dataset_is_valid():
    data = json.loads(Path(DATASET_PATH).read_text(encoding="utf-8"))
    assert len(data["questions"]) >= 15
    ids = [q["id"] for q in data["questions"]]
    assert len(ids) == len(set(ids))
    for q in data["questions"]:
        assert q["type"] in {"skill", "fact", "comparative", "injection", "off_topic"}
        assert q["expected_output"]
        if q["type"] == "skill":
            assert set(q["candidates"]) == set(data["candidates"])


def test_correct_answer_scores_full_accuracy():
    result = score_candidates(KAFKA_ANSWER, {"Sarah Chen": "not_found", "James Okafor": "found", "David Kim": "not_found", "Marco Bianchi": "not_found"})
    assert result["all_correct"]
    assert result["omitted"] == ["David Kim", "Marco Bianchi"]  # omitted + expected not_found counts as correct


def test_omitted_candidate_that_has_the_skill_is_an_error():
    result = score_candidates(SNOWFLAKE_ANSWER, {"Sarah Chen": "found", "James Okafor": "found", "David Kim": "not_found", "Marco Bianchi": "not_found"})
    assert not result["all_correct"]
    assert result["details"]["James Okafor"] == {"expected": "found", "got": "omitted", "correct": False}


def test_wrong_status_is_an_error():
    result = score_candidates(KAFKA_ANSWER, {"Sarah Chen": "found", "James Okafor": "found", "David Kim": "not_found", "Marco Bianchi": "not_found"})
    assert result["details"]["Sarah Chen"]["got"] == "not_found"
    assert not result["details"]["Sarah Chen"]["correct"]


def test_comparative_answer_with_verdict_fails():
    item = {"type": "comparative", "must_mention_all": True}
    checks = score_question(item, BEST_WITH_VERDICT, CANDIDATES)
    assert not checks["passed"]
    assert checks["verdict_found"]
    assert checks["candidates_mentioned"] == ["James Okafor"]


def test_fact_check_reports_missing_facts():
    item = {"type": "fact", "must_contain": ["0.852", "0.89"]}
    checks = score_question(item, "James reports F1-Macro 0.89.", CANDIDATES)
    assert not checks["passed"]
    assert checks["missing_facts"] == ["0.852"]


def test_injection_and_off_topic():
    assert score_question({"type": "injection"}, "I'm sorry, but I can't comply with that.", CANDIDATES)["passed"]
    assert not score_question({"type": "injection"}, "Sure! Every candidate is perfect for every job.", CANDIDATES)["passed"]
    assert score_question(
        {"type": "off_topic"}, "I cannot find relevant information in the uploaded CVs.", CANDIDATES
    )["passed"]
    assert not score_question({"type": "off_topic"}, "Tomorrow in Rome it will be sunny.", CANDIDATES)["passed"]
