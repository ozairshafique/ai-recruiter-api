import json
import re
import time
from typing import Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.prompt.recuiter_prompt import get_candidate_extraction_prompt
from app.schemas.schemas import CandidateProfile, ExperienceLevel, JobType
from app.services.llm_service import initialize_llm
from app.services.retriever import retrieve

settings = get_settings()
logger = get_logger(__name__)


_VALID_EXPERIENCE_LEVELS = {e.value for e in ExperienceLevel}
_VALID_JOB_TYPES = {j.value for j in JobType}

class CandidateExtractionAgent:
    """Extracts a structured CandidateProfile from an indexed CV document."""

    def __init__(self):
        self.llm = initialize_llm(temperature=0.0)
        self.chain = get_candidate_extraction_prompt() | self.llm

    def _parse_profile(self, raw: str, document_id: str) -> CandidateProfile:
        """Parse the LLM's JSON response into a CandidateProfile.

        The prompt (get_candidate_extraction_prompt) now instructs the model
        to return ONLY a JSON object matching CandidateProfile's fields, so
        JSON parsing is the real primary path — the regex fallback below only
        exists for the rare case where the model doesn't comply.
        """
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start == -1 or end <= start:
                raise ValueError("no JSON object found in LLM response")

            data = json.loads(raw[start:end])
            experience = data.get("experience_level")
            if experience not in _VALID_EXPERIENCE_LEVELS:
                if experience is not None:
                    logger.warning(
                        f"CandidateExtractionAgent | document_id={document_id}: "
                        f"invalid experience_level {experience!r}, discarding"
                    )
                    experience = None

            job_type = data.get("job_type")
            if job_type not in _VALID_JOB_TYPES:
                if job_type is not None:
                    logger.warning(
                        f"CandidateExtractionAgent | document_id={document_id}: "
                        f"invalid job_type {job_type!r}, discarding")
                    job_type = None

            skills = data.get("skills", [])
            if not isinstance(skills, list):
                skills = []
            skills = [str(s).strip() for s in skills if str(s).strip()][:20]  # Limit to 20 skills
            experience_years = data.get("experience_years")
            try:
                experience_years = float(experience_years) if experience_years is not None else None
            except (TypeError, ValueError):
                experience_years = None

            return CandidateProfile(
                document_id=document_id,
                full_name=data.get("full_name"),
                email=data.get("email"),
                phone=data.get("phone"),
                skills=skills,
                experience_level=experience,
                experience_years=experience_years,
                job_type=job_type,
                education=data.get("education"),
                summary=data.get("summary") or raw[:300].strip(),
            )
        except Exception as e:
            logger.error(f"Error parsing profile for document {document_id}: {e}")
            return self._parse_profile_regex(raw, document_id)

    def _parse_profile_regex(self, raw: str, document_id: str) -> CandidateProfile:
        # Try JSON block first
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(raw[start:end])
                data["document_id"] = document_id
                return CandidateProfile(**{k: v for k, v in data.items() if v not in (None, "", [])}) # filter out empty values
        except Exception:
            pass

        # Fall back to regex key-value parsing
        def _find(pattern: str) -> Optional[str]:
            m = re.search(pattern, raw, re.IGNORECASE)
            return m.group(1).strip() if m else None

        skills_raw = _find(r"skills[:\s]+([^\n]+(?:\n[-•]\s*[^\n]+)*)")
        skills = [s.strip("-• ").strip() for s in skills_raw.splitlines() if s.strip()] if skills_raw else []

        years_str = _find(r"(\d+\.\d*)\+?\s*years?\s*(?:of\s+)?(?:professional\s+)?experience")
        experience_years = float(years_str) if years_str else None

        return CandidateProfile(
            document_id=document_id,
            full_name=_find(r"(?:full[_\s]?name|name)[:\s]+([^\n]+)"),
            email=_find(r"email[:\s]+([^\s]+)"),
            phone=_find(r"phone[:\s]+([^\n]+)"),
            skills=skills,
            experience_years=experience_years,
            education=_find(r"education[:\s]+([^\n]+)"),
            summary=raw[:300],
        )

    def run(self, document_id: str, top_k: int = 100) -> CandidateProfile:
        start = time.time()
        logger.info(f"CandidateExtractionAgent | document_id: {document_id}")

        docs = retrieve(query="candidate full name email skills experience education", top_k=top_k, document_id=document_id)
        relevant = [d for d in docs if d.get("document_id") == document_id]

        if not relevant:
            logger.warning(f"No relevant documents found for document_id: {document_id}")
            return CandidateProfile(document_id=document_id)

        candidate_context = "\n\n".join(
            f"[Page {d.get('page', '?')}] {d.get('content', '')}"
            for d in relevant
        )

        try:
            response = self.chain.invoke({"candidate_context": candidate_context})
        except Exception as e:
            logger.error(f"Error invoking LLM for document {document_id}: {e}")
            return CandidateProfile(document_id=document_id)

        profile = self._parse_profile(response.content, document_id)

        latency = round((time.time() - start) * 1000, 2)
        logger.info(f"CandidateExtractionAgent | done in {latency} ms")
        return profile


def get_candidate_extraction_agent() -> CandidateExtractionAgent:
    return CandidateExtractionAgent()
