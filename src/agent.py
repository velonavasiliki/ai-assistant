"""Main agent module - graph construction and execution."""
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from config import Config
from state import AgentState
from models import youtube_tools, url_tools
from nodes import (
    greeter_intent_node,
    youtube_node,
    yt_transcript_node,
    url_node,
    rag_agent_node,
    library_node,
)
from routing import (
    yt_url_choice,
    start_over_or_quit,
    library_start_over_or_quit,
    should_loop,
    yttools_routing,
    yt_transcript_routing,
)

Config.setup_logging()

# ======= Building the Graph ======= #

graph = StateGraph(AgentState)

# Add nodes
graph.add_node("greeter_intent_node", greeter_intent_node)
graph.add_node("youtube_node", youtube_node)
graph.add_node("yt_transcript_node", yt_transcript_node)
graph.add_node("rag_agent_node", rag_agent_node)
graph.add_node("url_node", url_node)
graph.add_node("library_node", library_node)
graph.add_node("yttools", ToolNode(youtube_tools))
graph.add_node("vectorize_url_tool", ToolNode(url_tools))

# Add edges
graph.add_edge("vectorize_url_tool", "rag_agent_node")

graph.add_conditional_edges(
    "yttools",
    yttools_routing,
    {
        "youtube": "youtube_node",
        "transcript": "yt_transcript_node",
        "rag": "rag_agent_node"
    }
)

graph.set_entry_point("greeter_intent_node")

graph.add_conditional_edges(
    "greeter_intent_node",
    yt_url_choice,
    {
        "youtube": "youtube_node",
        "url": "url_node",
        "library": "library_node",
        "chat": "rag_agent_node",
        "quit": END
    }
)

graph.add_conditional_edges(
    "youtube_node",
    should_loop,
    {
        "loop": "yttools",
        "continue": "yt_transcript_node",
        "restart": "greeter_intent_node",
        "quit": END
    }
)

graph.add_conditional_edges(
    "yt_transcript_node",
    yt_transcript_routing,
    {
        "loop": "yttools",
        "continue": "yt_transcript_node",
        "restart": "youtube_node",
        "rag": "rag_agent_node",
        "quit": END
    }
)

graph.add_conditional_edges(
    "rag_agent_node",
    start_over_or_quit,
    {
        "restart": "greeter_intent_node",
        "continue": "rag_agent_node",
        "quit": END
    }
)

graph.add_conditional_edges(
    "library_node",
    library_start_over_or_quit,
    {
        "restart": "greeter_intent_node",
        "quit": END
    }
)

graph.add_conditional_edges(
    "url_node",
    should_loop,
    {
        "loop": "vectorize_url_tool",
        "continue": "rag_agent_node",
        "restart": "greeter_intent_node",
        "quit": END
    }
)

# Compile the graph
app = graph.compile()

# ======= Execution ======= #

if __name__ == "__main__":
    # Initial state
    state: AgentState = {
        "messages": [],
        "ytrecords": youtube_tools[0].yt_instance,
        "current_task": "",
        "quit": False,
        "go_back": False,
        "session_urls": [],
        "session_videos": [],
        "all_session_urls": [],
        "chat_scope": "all"
    }

    # Run the graph interactively
    while not state["quit"]:
        state = app.invoke(state)
    print("\n==== PERSONAL ASSISTANT FINISHED ====")
