
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
from langchain_core.tools import tools
from langgraph.graph import StateGraph, END
from langgraph.graph.message import SystemMessage, UserMessage, HumanMessage, AnyMessage

logger = get_logger(__name__)

# Prevent infinite loops by limiting the number of steps an agent can take
MAX_AGENTS_STEPS = 6


class AgentLLMError(Exception):
    """Custom exception for errors related to the Agent LLM."""


async def match_candidate(job_description: str, top_k: int = 5) -> str:

    agent = get_job_match_agent(top_k=top_k)
    results = await agent.arun(job_description=job_description,top_k=top_k)

    if not results:
        raise AgentLLMError("No results returned from the JobMatchAgent.")

    return "\n".join({f"Candidate: {r.full_name}, Score: {r.score:.2f}, Summary: {r.summary or 'None'}" } for r in results)