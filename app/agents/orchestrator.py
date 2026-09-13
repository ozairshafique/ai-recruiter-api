
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
from langgraph.graph.message import add_messages
from app.services.llm_service import initialize_llm
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage

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
        return "No results returned from the JobMatchAgent."

    return "\n".join(f"Candidate: {r.full_name}, Score: {r.match_score:.2f}, Summary: {r.summary or 'N/A'}" for r in results)

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

@tool
def find_document_id(candidate_name_or_descriptions: str) -> str:
    """ Find a candidate's document_id by their name or a description of them. Use this BEFORE extract_candidate_profile when you only have a name, not an ID - this is the kind of multi-step chain that requires genuine reasoning: look up the ID, THEN use it in a follow-up call """

    agent = get_recruiter_agent()
    results = agent.run(question=f"Find the document_id for the candidate with the following name or description: {candidate_name_or_descriptions}")

    if not results.get("sources"):
        return f"No document_id found for candidate: {candidate_name_or_descriptions}"

    top_sources = results["sources"][0]
    doc_id = top_sources.get("document_id")
    file_name = top_sources.get("file_name", "N/A")

    # sanity check: does the matched file_name plausibly relate to what was asked? This won't catch every misrank, but it catches the clearest cases where semantic search returned something unrelate

    name_tokens = candidate_name_or_descriptions.lower().split()
    file_name_tokens = file_name.lower()
    plausible_match = any(token in file_name_tokens for token in name_tokens if len(token) > 2)

    if not plausible_match:
        logger.warning(f"Autnomous Agent | find_document_id: The top source file_name '{file_name}' does not seem to match the candidate name/description '{candidate_name_or_descriptions}'.")

        return f"Found possible matched document (document_id: {doc_id}, file_name: {file_name})," f" but file name does not match what was asked. Please verify the document_id before using it in extract_candidate_info"

    return f"Found document_id: {doc_id} for candidate: {candidate_name_or_descriptions} (file_name: {file_name})"

@tool
def ask_about_candidate(questions: str) -> str:
    """Ask a question about a specific candidate"""
    agent = get_recruiter_agent()
    results = agent.run(question=questions)
    return results["message"]

TOOLS = [match_candidate, extract_candidate_info, ask_about_candidate, find_document_id]
TOOL_BY_NAME = {t.name: t for t in TOOLS}

_agent_llm = None

 # Initialize the agent's LLM with the available tools
def _get_agent_llm():
    global _agent_llm
    if _agent_llm is None:
        _agent_llm = initialize_llm(temperature=0.0).bind_tools(TOOLS)
    return _agent_llm
