
from typing import Annotated, TypedDict
import asyncio
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.core.logging import get_logger
from app.agents import (
    get_recruiter_agent,
    get_job_match_agent,
    get_candidate_extraction_agent,
    )
from langgraph.graph.message import add_messages
from app.services.llm_service import initialize_llm
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage

logger = get_logger(__name__)

# Prevent infinite loops by limiting the number of steps an agent can take
MAX_AGENTS_STEPS = 6


class AgentLLMError(Exception):
    """Custom exception for errors related to the Agent LLM."""


@tool
async def match_candidate(job_description: str, top_k: int = 5) -> str:

    """ Match a candidate to a job description based on their skills and experience. """
    agent = get_job_match_agent(top_k=top_k)
    results = await agent.arun(job_description=job_description,top_k=top_k)

    if not results:
        return "No results returned from the JobMatchAgent."

    return "\n".join(f"Candidate: {r.full_name}, Score: {r.match_score:.2f}, Summary: {r.summary or 'N/A'}" for r in results)

@tool
def extract_candidate_info(document_id: str) -> str:
    """ Extract a structured profile (skills, experience level, education) for ONE candidate, given their document_id. If you don't have a document_id yet, use find_document_id first"""

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
    return results["answer"]

TOOLS = [match_candidate, extract_candidate_info, ask_about_candidate, find_document_id]
TOOL_BY_NAME = {t.name: t for t in TOOLS}

_agent_llm = None

 # Initialize the agent's LLM with the available tools
def _get_agent_llm():
    global _agent_llm
    if _agent_llm is None:
        _agent_llm = initialize_llm(temperature=0.0).bind_tools(TOOLS)
    return _agent_llm


AGENT_SYSTEM_PROMPT = """
You are an autonomous recruiting assistant with access to tools. Reason step by step:
1. Decide if you need a tool to answer the request, and which one.
2. Call it, then read its actual result before deciding what to do next.
3. If the result gives you what you need, answer the user directly - do not call more tools than necessary.
4. If you need another piece of information (e.g. you found a name but need its document_id before extracting a profile), call the
appropriate tool next - do not guess or fabricate an ID or a result.
5. Never state a hiring recommendation or declare one candidate "best" - report evidence only, the human recruiter decides.
"""

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int

@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
    retry=retry_if_exception_type(AgentLLMError),
    )
def _invoke_agent_llm(messages):
    try:
        return _get_agent_llm().invoke(messages)
    except Exception as e:
        raise AgentLLMError(str(e))

def agent_node(state: AgentState) -> AgentState:
    start = time.time()
    try:
        response = _invoke_agent_llm(state["messages"])
    except AgentLLMError as e:
        logger.error(f"AutonomousAgent | reasoning call failed after retries: {e}")
        response = HumanMessage(content="I encountered an error and cannot continue reasoning about this request")

    latency = round((time.time() - start) * 1000, 2)
    tools_calls = getattr(response, "tool_calls", None) or []
    logger.info(f"AutonomousAgent | agent_node | steps: {state['steps']}"
    f"decided to call {len(tools_calls)} tools | latency: {latency} ms")
    return {"messages": [response], "steps": state["steps"] + 1}

async def tools_node(state: AgentState) -> AgentState:
    """Executes whatever tool(s) the model just decided to call, and feeds the REAL result back as an observation - the model sees this on its next reasoning step and decides what to do with it """

    last_message = state["messages"][-1]
    tools_messages = []

    for call in last_message.tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]
        logger.info(f"AutonomousAgent | tools_node | calling tool: {tool_name} with args: {tool_args}")

        try:
            # Execute the selected tool asynchronously
            tool_selected = TOOL_BY_NAME[tool_name]
            results = await tool_selected.ainvoke(tool_args)
        except Exception as e:
            logger.error(f"AutonomousAgent | tools_node | error calling tool: {tool_name} with args: {tool_args} - {e}")
            results = f"Tools {tool_name} Error: {e}"
        tools_messages.append(ToolMessage(content=str(results), tool_call_id=call["id"]))

        # Return the updated state with the tool results
    return {"messages": tools_messages, "steps": state["steps"]}

def should_continue(state: AgentState) -> str:
    """Determines whether the agent should continue or end, based on the number of steps taken and the presence of tool calls in the last message"""
    if state["steps"] >= MAX_AGENTS_STEPS:
        logger.error("AutonomousAgent | should_continue | maximum steps {MAX_AGENTS_STEPS}, stopping")
        return "end"
    last_messages =state["messages"][-1]
    if getattr(last_messages, "tool_calls", None):
        return "continue"
    return "end"

def build_autnomous_agent(require_human_approval: bool = False):
    """Builds the autonomous agent with the specified configuration, including whether human approval is required"""
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue,{"continue": "tools", "end": END})
    graph.add_edge("tools", "agent")
    checkpoint = {"checkpointer": _get_memory_saver()}
    if require_human_approval:
        # Add an interrupt before the tools node
        checkpoint["interrupt_before"] = ["tools"]
    return graph.compile(**checkpoint) # Compile the graph with the specified checkpoint



# ── Cached singletons: checkpointer + compiled graphs ──────────
# One shared MemorySaver so thread_id-based memory actually persists across separate calls, not just within a single call's internal loop. Two separate compiled graphs (with/without human approval) since interrupt_before is baked in at compile time, not something you can
# toggle per-call on an already-compiled graphs

_memory_saver = None
_compiled_graph_with_approval = None
_compiled_graph_normal = None

def _get_memory_saver():
    global _memory_saver
    if _memory_saver is None:
        _memory_saver = MemorySaver()
    return _memory_saver

def _get_compiled_graph(require_human_approval: bool):
    global _compiled_graph_with_approval, _compiled_graph_normal
    if require_human_approval:
        if _compiled_graph_with_approval is None:
            _compiled_graph_with_approval = build_autnomous_agent(require_human_approval=True)
        return _compiled_graph_with_approval
    if _compiled_graph_normal is None:
        _compiled_graph_normal = build_autnomous_agent(require_human_approval=False)
    return _compiled_graph_normal