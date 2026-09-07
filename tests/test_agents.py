
import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

 #── CandidateExtractionAgent─────────
def test_candidate_extraction_agent_parse_valid():
    from app.agents.candidate_extraction_agent import CandidateExtractionAgent
    with patch("app.agents.candidate_extraction_agent.initialize_llm"):
        agent = CandidateExtractionAgent.__new__(CandidateExtractionAgent)

        raw = json.dumps({
            "full_name": "John Doe",
            "email": "jane@example.com",
            "phone": "123-456-7890",
            "skills": ["Python", "SQL"],
            "experience_years": 5.5,
            "experience_level": "senior",
            "education": "BSc Computer Science",
            "job_type": "full_time",
            "summary": "Experienced backend engineer."
        })

        profiles = agent._parse_profile(raw, document_id="doc1234")
        assert profiles.full_name == "John Doe"
        assert profiles.email == "jane@example.com"
        assert profiles.phone == "123-456-7890"
        assert profiles.experience_years == 5.5
        assert profiles.experience_level == "senior"
        assert "Python" in profiles.skills
        assert profiles.summary == "Experienced backend engineer."
        assert profiles.document_id == "doc1234"

def test_candidate_extraction_agent_enum_invalid():
    from app.agents.candidate_extraction_agent import CandidateExtractionAgent
    with patch("app.agents.candidate_extraction_agent.initialize_llm"):
        agent = CandidateExtractionAgent.__new__(CandidateExtractionAgent)

        raw = json.dumps({
        "full_name": "Jane Doe",
        "experience_level": "not_a_real_level",
        "job_type": "not_a_real_type",
        "skills": [],
        })

        profile = agent._parse_profile(raw, document_id="doc5678")
        assert profile.experience_level is None
        assert profile.job_type is None

def test_candidate_extraction_agent_fallback_on_invalid_json():
    from app.agents.candidate_extraction_agent import CandidateExtractionAgent
    with patch("app.agents.candidate_extraction_agent.initialize_llm"):
        agent = CandidateExtractionAgent.__new__(CandidateExtractionAgent)

        raw = "Name: John Smith\nSkills:\n- Python\n- SQL\n5 years experience"

        profile = agent._parse_profile(raw, document_id="doc-2")

        assert profile.document_id == "doc-2"
        assert profile.experience_years == 5.0

def test_candidate_extraction_agent_no_fallback():
    from app.agents.candidate_extraction_agent import CandidateExtractionAgent
    with patch("app.agents.candidate_extraction_agent.initialize_llm"), \
        patch("app.agents.candidate_extraction_agent.retrieve") as mock_retrieve:
        mock_retrieve.return_value = [{
                "document_id": "other_docs",
                "content":"unrelated content",
                "page": 1
            }]

        agent = CandidateExtractionAgent.__new__(CandidateExtractionAgent)
        agent.chain = MagicMock()
        profile = agent.run(document_id="doc-2")
        agent.chain.invoke.assert_not_called()
        assert profile.document_id == "doc-2"
        assert profile.full_name is None

#── Job Matching Agent Tests ─────────
def test_job_match_agent_parse_valid():
    from app.agents.job_match_agent import JobMatchAgent
    agent = JobMatchAgent.__new__(JobMatchAgent)
    raw = json.dumps({
            "full_name": "Alice Smith",
            "match_score": 0.85,
            "document_id": "doc-2",
            "matched_skills": ["Python", "SQL"],
            "candidate_experience_level": "senior",
            "summary": "Strong candidate with backend experience"
        })
    profile = agent._parse_match(raw, document_id="doc-2")
    assert profile.full_name == "Alice Smith"
    assert "Python" in profile.matched_skills
    assert profile.summary == "Strong candidate with backend experience"
    assert profile.match_score == 0.85
    assert profile.document_id == "doc-2"

def test_job_match_agent_clamps_mismatch():
    from app.agents.job_match_agent import JobMatchAgent
    agent = JobMatchAgent.__new__(JobMatchAgent)
    raw = json.dumps({
            "full_name": "Bob Johnson",
            "match_score": 1.5,
            "document_id": "doc-4",
            "matched_skills": ["Java", "Spring"],
            "summary": "Experienced Java developer"
    })
    profile = agent._parse_match(raw, document_id="doc-4")
    assert profile.full_name == "Bob Johnson"
    assert profile.match_score == 1.0  # Clamped to 1.0

def test_job_match_agent_degrades_on_malformed_json():
    from app.agents.job_match_agent import JobMatchAgent
    agent = JobMatchAgent.__new__(JobMatchAgent)

    results = agent._parse_match("This is not JSON", document_id="doc-5")
    assert results.match_score == 0.0
    assert results.summary == "Error parsing LLM response"
    assert results.document_id == "doc-5"
    assert results.full_name is None

def test_job_match_agent_not_duplicate_candidate_chunks():
    ''' Test that the JobMatchAgent does not return duplicate candidates when multiple chunks of the same candidate are present.'''
    from app.agents.job_match_agent import JobMatchAgent
    with patch("app.agents.job_match_agent.initialize_llm"), \
        patch("app.agents.job_match_agent.get_job_match_prompt"), \
            patch("app.agents.job_match_agent.get_all_document_ids") as mock_ids, \
                patch("app.agents.job_match_agent.retrieve") as mock_retrieve:
        mock_ids.return_value = ["docs-1"]
        mock_retrieve.return_value = [{
            "document_id": "docs-1", "content": "chunk A", "page": 1, "file_name": "file1.pdf"
        }, {"document_id": "docs-1", "content": "chunk B", "page": 2, "file_name": "file1.pdf"}]

        agent = JobMatchAgent.__new__(JobMatchAgent)
        agent.top_k = 5
        agent.chain = MagicMock()

        import asyncio
        async def fake_invoke(*args, **kwargs):
            response = MagicMock()
            response.content = json.dumps({
                "full_name": "Candidate A",
                "match_score": 0.9,
                "document_id": "docs-1",
                "matched_skills": ["Python"],
                "candidate_experience_level": "mid",
                "summary": "ok"
            })
            return response

        agent._invoke_with_retry = fake_invoke
        import app.agents.job_match_agent as jma
        agent._semaphore = jma.asyncio.Semaphore(4)
        results = asyncio.run(agent.arun(job_description="Looking for a Python developer", top_k=5))
        assert len(results) == 1
        assert results[0].document_id == "docs-1"


#── Job Matching Agent Tests with expereince level cap ─────────
@pytest.mark.parametrize("candidate_level, required_level, expected_score",[("senior", "senior", 0.95), # exact match, no cap, natural score preserved
("senior","mid", 0.6), # 1st tier cap, score capped at 0.6
("senior","junior", 0.3), # 2nd tier cap, score capped at 0.3
("junior","mid", 0.6), # 1st tier cap, score capped at 0.6
("mid","lead", 0.3), # 2nd tier cap, score capped at 0.3
("junior", "lead", 0.3), # 3 tiers off - still capped at 0.3 (not lower)
("entry", "lead", 0.3), # max distance - capped at 0.3
],)

def test_job_match_agent_experience_level_cap(candidate_level, required_level, expected_score):
    from app.agents.job_match_agent import _apply_experience_level_filter

    natural_score = 0.95

    capped_score, note = _apply_experience_level_filter(natural_score, candidate_level, required_level)
    assert capped_score == expected_score
    if candidate_level != required_level:
        assert note is not None
        assert "does not match required level" in note
        assert candidate_level in note
        assert required_level in note
        assert "Experience level" not in note

def test_experience_level_cap_does_not_raise_already_low_score():
    from app.agents.job_match_agent import _apply_experience_level_filter

    low_natural_score = 0.15
    capped_score, note = _apply_experience_level_filter(low_natural_score, "senior", "junior")
    assert capped_score == 0.15
    assert note is None

