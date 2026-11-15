from unittest.mock import MagicMock, patch

from app.services import llm_service
from app.services.llm_service import EMPTY_ANSWER_FALLBACK, generate_answer, initialize_llm


def _fake_llm(content, finish_reason="length"):
    response = MagicMock()
    response.content = content
    response.response_metadata = {"finish_reason": finish_reason}
    llm = MagicMock()
    llm.invoke.return_value = response
    return llm


def test_empty_answer_returns_fallback_instead_of_empty_string():
    with patch("app.services.llm_service.ChatPromptTemplate.from_messages") as from_messages:
        from_messages.return_value.__or__.return_value = _fake_llm("")
        answer = generate_answer(question="Who is the best candidate?", context="ctx", llm=MagicMock())
    assert answer == EMPTY_ANSWER_FALLBACK


def test_normal_answer_is_returned_unchanged():
    with patch("app.services.llm_service.ChatPromptTemplate.from_messages") as from_messages:
        from_messages.return_value.__or__.return_value = _fake_llm("Candidate: Sarah Chen", "stop")
        answer = generate_answer(question="Who knows Snowflake?", context="ctx", llm=MagicMock())
    assert answer == "Candidate: Sarah Chen"


def test_reasoning_model_gets_token_budget_and_low_effort():
    with patch("app.services.llm_service.ChatGroq") as chat_groq:
        initialize_llm()
    kwargs = chat_groq.call_args.kwargs
    assert kwargs["max_tokens"] == llm_service.settings.llm_max_tokens >= 4096


def test_default_prompt_is_the_recruiter_prompt():
    from app.prompt.recruiter_prompt import RECRUITER_SYSTEM_PROMPT

    with patch("app.services.llm_service.ChatPromptTemplate.from_messages") as from_messages:
        from_messages.return_value.__or__.return_value = _fake_llm("Candidate: Sarah Chen", "stop")
        generate_answer(question="Who knows Snowflake?", context="ctx", llm=MagicMock())
    system_message = from_messages.call_args.args[0][0]
    assert system_message == ("system", RECRUITER_SYSTEM_PROMPT)


def test_recruiter_agent_uses_the_shared_pipeline():
    from app.agents.recruiter_agent import RecruiterAgent
    from app.prompt.recruiter_prompt import RECRUITER_SYSTEM_PROMPT

    fake = {"answer": "Candidate: James Okafor", "sources": [{"file_name": "james.pdf"}], "model": "m", "latency_ms": 5}
    with patch("app.agents.recruiter_agent.run_query_pipeline", return_value=fake) as pipeline:
        result = RecruiterAgent(top_k=8).run(question="Who knows Kafka?")
    pipeline.assert_called_once_with(
        query="Who knows Kafka?", top_k=8, document_id=None, system_prompt=RECRUITER_SYSTEM_PROMPT
    )
    assert result["answer"] == "Candidate: James Okafor"


def test_truncated_answer_is_flagged():
    from app.services.llm_service import TRUNCATED_ANSWER_NOTE

    with patch("app.services.llm_service.ChatPromptTemplate.from_messages") as from_messages:
        from_messages.return_value.__or__.return_value = _fake_llm("Candidate: Sarah Chen\n• reducing costs by 18", "length")
        answer = generate_answer(question="Who is the best candidate?", context="ctx", llm=MagicMock())
    assert answer.startswith("Candidate: Sarah Chen")
    assert answer.endswith(TRUNCATED_ANSWER_NOTE)