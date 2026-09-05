import re
import time

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import json
import asyncio
from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.retriever import retrieve, get_all_document_ids
from app.prompt.recuiter_prompt import get_job_match_prompt
from app.schemas.schemas import JobMatchResult
from app.services.llm_service import initialize_llm

settings = get_settings()
logger = get_logger(__name__)


MAX_CONCURRENT_MATCHES = getattr(settings, "max_concurrent_matches", 8)
CHUNKS_PER_CANDIDATE = getattr(settings, "job_match_chunks_per_candidate", 8)


class LLMCallError(Exception):
    """Raised when the LLM call fails after retries are exhausted."""

class JobMatchAgent:
    """Scores indexed candidates against a job description and returns ranked results."""

    def __init__(self, top_k: int = None):
        self.top_k = top_k or settings.top_k
        self.llm = initialize_llm(temperature=0.0)
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_MATCHES)
        self.chain = get_job_match_prompt() | self.llm

    def _parse_match(self, raw: str, document_id: str) -> JobMatchResult:

        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            if start == -1 or end <= start:
                raise ValueError("No JSON object found in LLM response")


            data = json.loads(raw[start:end])
            score = float(data.get("match_score", 0.0))
            score = min(max(score, 0.0), 1.0)  # Clamp score between 0 and 1

            skills = data.get("matched_skills", [])
            if not isinstance(skills, list):
                skills = []
            skills = [s.strip() for s in skills if str(s).strip()][:10]

            summary = str(data.get("summary", "")).strip() or raw[:200].strip()

            full_name = data.get("full_name", None)

            return JobMatchResult(
                 match_score=min(max(score, 0.0), 1.0),
                 full_name=full_name,            matched_skills=skills,
                 summary=summary,
                 document_id=document_id,
                 )

        except ValueError as e:
                logger.error(f"Error parsing match for document {document_id}: {e}")

                return JobMatchResult(
                    match_score=0.0,
                    full_name=None,
                    matched_skills=[],
                    summary="Error parsing LLM response",
                    document_id=document_id,
                )

    # LLM call with retries and exponential backoff
    @retry(
        stop=stop_after_attempt(3),
        reraise=True,
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        retry=retry_if_exception_type(LLMCallError),
    )

    async def _invoke_with_retry(self, job_description: str, candidate_context: str):
        try:
            return await self.chain.ainvoke({
                "job_context": job_description,
                "candidate_context": candidate_context,
            })
        except Exception as exc:
            raise LLMCallError(str(exc)) from exc

    async def _score_candidate(self, job_description: str, document_id: str) -> JobMatchResult | None:

        chunks = retrieve(query=job_description, top_k=CHUNKS_PER_CANDIDATE, document_id=document_id)
        if not chunks:
            logger.warning(f"No candidate chunks found for document_id={document_id}")
            return None

        file_name = chunks[0].get("file_name", "N/A")
        candidate_context = "\n\n".join(f"[Page {c.get('page', 'N/A')}]\n{c.get('content', '')}" for c in chunks)
        candidate_context = f"File: {file_name} \n {candidate_context}"

        # Use semaphore to limit concurrent LLM calls
        async with self._semaphore:
            try:
                response = await self._invoke_with_retry(job_description, candidate_context)
            except Exception as e:
                logger.error(f"Error invoking LLM for document {document_id}: {e}")

                return JobMatchResult(
                    match_score=0.0,
                    full_name=None,
                    matched_skills=[],
                    summary="Error invoking LLM for this candidate.",
                    document_id=document_id,
                )

        return self._parse_match(response.content,document_id)

    # Asynchronous Public API
    async def arun(self, job_description: str, top_k: int = None) -> list[JobMatchResult]:
        top_k = top_k or self.top_k
        start = time.time()
        logger.info(f"JobMatchAgent | JD: {job_description[:60]}")

        try:
            document_ids = get_all_document_ids()
        except Exception as e:
            logger.error(f"Error retrieving document IDs: {e}")
            return []
        if not document_ids:
            logger.warning("JobMatchAgent | no candidate documents retrieved")
            return []

        # Use asyncio.gather to score all candidates concurrently, respecting the semaphore limit

        results = await asyncio.gather(*(self._score_candidate(job_description, doc_id) for doc_id in document_ids))
        results = [r for r in results if r is not None]
        results.sort(key=lambda r: r.match_score, reverse=True)
        results = results[:top_k]

        latency = round((time.time() - start) * 1000, 2)
        logger.info(f"JobMatchAgent | {len(results)} matches in {latency} ms")
        return results


    def run(self, job_description: str, top_k: int = None) -> list[JobMatchResult]:
        ''' Synchronous wrapper for the asynchronous run method.'''
        return asyncio.run(self.arun(job_description, top_k))

def get_job_match_agent(top_k: int = None) -> JobMatchAgent:
    return JobMatchAgent(top_k=top_k)
