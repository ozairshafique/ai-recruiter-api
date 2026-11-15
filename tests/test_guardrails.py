from app.services.guardrails import (
    NO_VERDICT_REMINDER,
    VERDICT_REMOVED_NOTE,
    add_no_verdict_reminder,
    is_comparative_question,
    remove_verdicts,
)


def test_comparative_questions_detected():
    assert is_comparative_question("Who is the best candidate?")
    assert is_comparative_question("Rank the candidates for this role")
    assert is_comparative_question("Who should we hire?")
    assert not is_comparative_question("Which candidates know Python?")
    assert not is_comparative_question("Does anyone know Rust?")


def test_reminder_only_added_to_comparative_questions():
    assert add_no_verdict_reminder("Who is the best candidate?").endswith(NO_VERDICT_REMINDER)
    assert add_no_verdict_reminder("Who knows Kafka?") == "Who knows Kafka?"


def test_verdict_sentence_removed_and_note_added():
    answer = (
        "Candidate: James Okafor\nPython: ✅ Found\n"
        "• Python (expert, 7 years)\n\n"
        "Overall, James shows strong skills. This makes him the strongest candidate based on the provided context."
    )
    clean, removed = remove_verdicts(answer)
    assert removed
    assert "strongest candidate" not in clean
    assert "Overall, James shows strong skills." in clean
    assert "Python (expert, 7 years)" in clean
    assert clean.endswith(VERDICT_REMOVED_NOTE)


def test_evidence_only_answer_untouched():
    answer = "Candidate: Sarah Chen\nSnowflake: ✅ Found\n• Data: Spark, Pandas, NumPy, PostgreSQL, Snowflake"
    clean, removed = remove_verdicts(answer)
    assert not removed
    assert clean == answer


def test_short_field_cleaned_without_note():
    clean, removed = remove_verdicts("Strong Python and AWS. She is the ideal candidate for this role.", add_note=False)
    assert removed
    assert clean == "Strong Python and AWS."


def test_agent_final_answer_is_guarded():
    from app.agents.orchestrator import _final_answer

    out = _final_answer("James has 7 years of Python. James is the best candidate.")
    assert "best candidate" not in out
    assert out.endswith(VERDICT_REMOVED_NOTE)
