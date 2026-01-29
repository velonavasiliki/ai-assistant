"""Conditional edge routing functions for the agent graph."""
from langchain_core.messages import AIMessage, ToolMessage

from state import AgentState, Intent


def yt_url_choice(state: AgentState) -> str:
    """Decide where to go after the greeter node based on the current_task."""
    if state["quit"]:
        return "quit"
    if state["current_task"] == Intent.youtube.value:
        return "youtube"
    if state["current_task"] == Intent.url.value:
        return "url"
    if state["current_task"] == Intent.library.value:
        return "library"
    if state["current_task"] == Intent.chat.value:
        return "chat"
    return "quit"


def start_over_or_quit(state: AgentState) -> str:
    """Decide whether to restart at the greeter or quit after QA."""
    if state["quit"]:
        return "quit"
    # If current_task was changed by user input inside QA, restart flow
    restart_tasks = [Intent.youtube.value, Intent.url.value, Intent.library.value, Intent.greeter.value]
    if state["current_task"] in restart_tasks:
        return "restart"
    return "continue"


def library_start_over_or_quit(state: AgentState) -> str:
    """Decide whether to restart at the greeter or quit after library management."""
    if state["quit"]:
        return "quit"
    if state["go_back"]:
        return "restart"
    return "quit"


def should_loop(state: AgentState) -> str:
    """
    Determines if the agent's last message contains a tool call.
    If so, we should go to the tool node. Otherwise, the agent has a final
    response and we should end the graph.
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "loop"
    if state["quit"]:
        return "quit"
    if state.get("go_back"):
        return "restart"
    return "continue"


def yttools_routing(state: AgentState) -> str:
    """Route from yttools based on which tool was executed."""
    # Check the last ToolMessage to see what was executed
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            # get_transcript_tool auto-vectorizes, so go to RAG for Q&A
            if msg.name == "get_transcript_tool":
                return "rag"
            else:
                return "youtube"
    return "youtube"


def yt_transcript_routing(state: AgentState) -> str:
    """Route from yt_transcript_node based on state."""
    if state["quit"]:
        return "quit"
    if state["go_back"]:
        return "restart"
    last_message = state["messages"][-1] if state["messages"] else None
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "loop"
    # Check if we have vectorized transcripts (tool completed successfully)
    for msg in reversed(state["messages"][-5:]):
        if isinstance(msg, ToolMessage) and msg.name == "get_transcript_tool":
            return "rag"
    return "continue"
