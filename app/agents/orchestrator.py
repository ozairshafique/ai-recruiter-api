
from typing import Annotated, TypedDict
import asyncio
import time
from tenacity import retry, stop_after_attempt, wait_random_exponential
from app.core.logging import get_logger
from app.agents import (
    get_recruiter_agent,
    get_job_match_agent,
    get_candidate_extraction_agent,
    )
from langgraph.graph.message import add_message
from app.services.llm_service import initialize_llm
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import SystemMessage, UserMessage, HumanMessage, AnyMessage

logger = get_logger(__name__)

# Prevent infinite loops by limiting the number of steps an agent can take
MAX_AGENTS_STEPS = 6


class AgentLLMError(Exception):
    """Custom exception for errors related to the Agent LLM."""


@tool
async def match_candidate(job_description: str, top_k: int = 5) -> str:

    agent = get_job_match_agent(top_k=top_k)
    results = await agent.arun(job_description=job_description,top_k=top_k)

    if not results:
        raise AgentLLMError("No results returned from the JobMatchAgent.")

    return "\n".join({f"Candidate: {r.full_name}, Score: {r.score:.2f}, Summary: {r.summary or 'None'}" } for r in results)

@tool
def extract_candidate_info(document_id: str) -> str:
    """ Extract a structured profile (skills, experience level, education)
    for ONE candidate, given their document_id. If you don't have a
    document_id yet, use find_document_id first"""

    agent = get_candidate_extraction_agent()
    profile = agent.run(document_id=document_id)

    if profile.full_name is None or profile.skills is None:
        return f"No candidate information could be extracted for document_id: {document_id}"
    return f"Candidate Profile:{profile.full_name} \n Skills:{', '.join(profile.skills)}, Experience:{profile.experience_years}, Experience Level: {profile.experience_level}, Education: {profile.education}"