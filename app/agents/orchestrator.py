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
from app.services.guardrails import remove_verdicts
from app.services.retriever import find_documents_by_name, get_document_by_name
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

    matches = find_documents_by_name(candidate_name_or_descriptions)
    if matches and matches[0]["matched_words"] > 0:
        top = matches[0]
        tied = [m for m in matches if m["matched_words"] == top["matched_words"]]
        if len(tied) > 1:
            options = ", ".join(f"{m['document_id']} ({m['file_name']})" for m in tied)
            return f"Several documents match '{candidate_name_or_descriptions}': {options}. Ask the user which one."
        return f"Found document_id: {top['document_id']} for candidate: {candidate_name_or_descriptions} (file_name: {top['file_name']})"

    # Not a name (e.g. "the frontend engineer from Toronto"): fall back to semantic search.
    doc_id = get_document_by_name(candidate_name_or_descriptions)
    if not doc_id:
        return f"No document_id found for candidate: {candidate_name_or_descriptions}"
    logger.warning(f"AutonomousAgent | find_document_id: no name match, semantic guess {doc_id}")
    return (
        f"Possible match (semantic search, not a name match): document_id {doc_id}. "
        f"Verify it before using it in extract_candidate_info."
    )

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

def _final_answer(content: str) -> str:
    """Enforce the no-verdict rule on the agent's final answer, in code (not only in the prompt)."""
    clean, removed = remove_verdicts(content or "")
    if removed:
        logger.warning("AutonomousAgent | guardrail removed a hiring verdict from the final answer")
    return clean


FINAL_ANSWER_INSTRUCTION = (
    "Stop calling tools now. Using ONLY the tool results above, give the user your final answer. "
    "Report evidence only; never declare a best candidate."
)


def _tool_results_summary(messages: list, max_chars: int = 1500) -> str:
    """Deterministic fallback: the latest tool results, so the user is never left with an empty answer."""
    results = [m.content for m in messages if isinstance(m, ToolMessage) and m.content]
    if not results:
        return "The agent stopped without a final answer and no tool returned results. Please rephrase the request."
    text = "\n\n".join(results[-3:])
    return "The agent reached its step limit. Latest tool results:\n\n" + text[:max_chars]


def _resolve_final_answer(messages: list) -> str:
    """Return a non-empty final answer even when the agent stopped mid-way.

    The graph can end with an empty message or with unanswered tool calls (step limit).
    In that case, ask the LLM once more WITHOUT tools to answer from the tool results it
    already has; if that fails too, fall back to the raw tool results.
    """
    last = messages[-1]
    content = (getattr(last, "content", "") or "").strip()
    if content and not getattr(last, "tool_calls", None):
        return _final_answer(content)

    logger.warning("AutonomousAgent | no final answer (step limit or empty reply), synthesising from tool results")
    # Drop a trailing message whose tool calls were never executed: the API rejects unanswered tool calls.
    history = messages[:-1] if getattr(last, "tool_calls", None) else list(messages)
    try:
        reply = initialize_llm(temperature=0.0).invoke(history + [HumanMessage(content=FINAL_ANSWER_INSTRUCTION)])
        text = (getattr(reply, "content", "") or "").strip()
        if text:
            return _final_answer(text)
    except Exception as exc:
        logger.error(f"AutonomousAgent | final synthesis failed: {exc}")
    return _final_answer(_tool_results_summary(messages))


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

def build_autonomous_agent(require_human_approval: bool = False):
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
            _compiled_graph_with_approval = build_autonomous_agent(require_human_approval=True)
        return _compiled_graph_with_approval
    if _compiled_graph_normal is None:
        _compiled_graph_normal = build_autonomous_agent(require_human_approval=False)
    return _compiled_graph_normal


async def run_autonomous_agent(user_message: str, require_human_approval: bool = False, thread_id: str = None) -> dict:
    start = time.time()
    app = _get_compiled_graph(require_human_approval)
    config = {"configurable": {"thread_id": thread_id}}
    initialize_state = {
        "messages": [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            HumanMessage(content=user_message)],
        "steps": 0
    }

    final_results = await app.ainvoke(initialize_state, config=config)
    latency = round((time.time() - start) * 1000, 2)
    last_message = final_results["messages"][-1]
    pending_tools = getattr(last_message, "tool_calls", None) or []

    if require_human_approval and pending_tools:
        logger.info(f"Autonomous agent has pending tools: {pending_tools} for human review")
        return {
            "status": "awaiting_approval",
            "pending_tools": pending_tools,
            "thread_id": thread_id,
            "latency_ms": latency
        }
    logger.info(f"AutonomousAgent | thread_id: {thread_id} | completed in steps {final_results['steps']} | latency: {latency} ms")
    return {
        "status": "completed",
        "results": _resolve_final_answer(final_results["messages"]),
        "steps": final_results['steps'],
        "thread_id": thread_id,
        "latency_ms": latency
    }

async def approve_pending_action(thread_id: str, require_human_approval: bool = True):
    start = time.time()
    app = _get_compiled_graph(require_human_approval)
    config = {"configurable": {"thread_id": thread_id}}

    final_results = await app.ainvoke(None, config=config)
    latency = round((time.time() - start) * 1000, 2)
    last_message = final_results["messages"][-1]
    logger.info(f"Approved pending action for thread_id: {thread_id} | latency: {latency} ms")
    return {
        "status": "completed",
        "results": _resolve_final_answer(final_results["messages"]),
        "steps": final_results['steps'],
        "thread_id": thread_id,
        "latency_ms": latency
    }