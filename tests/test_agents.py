
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
