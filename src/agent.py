import os
import operator
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Union
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
import html
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi
from modules.ytinteraction import ytinteraction

# ======= Environment variables ======= #

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ======= Tools ======= #

@tool
def youtube_search(query: str, order: str = 'viewCount', duration: str = 'medium', num_results: int = 1, before: str = None, after: str = None) -> str:
    """Search YouTube and return formatted results as a string."""

    results = ytinteraction().ytretriever(query=query, order=order, duration=duration, num_results=num_results, before=before, after=after)
    
    if not results:
        return f"No videos found for query: '{query}'"
    
    formatted_results = []
    for video_id, info in results.items():
        formatted_results.append(
            f"Title: {info['title']}\n"
            f"Channel: {info['channel']}\n"
            f"Published: {info['date']}\n"
            f"URL: https://youtube.com/watch?v={info['id']}\n"
        )
    
    return "\n".join(formatted_results)

@tool
def validate_date_format(date_str: str) -> bool:
    """Validate date string format. If False, return message to user."""
    try:
        datetime.strptime(date_str, "%m/%d/%Y")
        return True
    except ValueError:
        return False

# ======= Model definition ======= #

class AgentState(TypedDict):
    """The state of the agent, containing the conversation history."""
    messages: Annotated[List[BaseMessage], operator.add]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
model = llm.bind_tools([youtube_search, validate_date_format])

# ======= Nodes ======= #

def agent_node(state: AgentState):
    """
    The agent node that calls the LLM with the latest messages.
    """
    system_prompt = SystemMessage(content="""
        You are a personal AI assistant. Your purpose is to help the user search for videos on youtube.
        - You must use the `youtube_search` tool whenever the user asks for a video search.
        - Respond to the user in a friendly and helpful manner.
        - If the user asks for dates, you should verify that the date format is correct, that is, %m/%d/%Y,
            using the tool `validate_date_format`. If the tool returns `False`, tell the user:
            'I'm sorry, I don't understand which date you prefer. Could you tell me the date in "m/d/Y" format, e.g. 05/23/2025?'
        - Always provide a helpful response, don't just repeat what the user said.
    """)
    
    response = model.invoke([system_prompt] + state["messages"])
    
    return {"messages": [response]}

# ======= Codnitional Edges ======= #

def should_continue(state: AgentState) -> str:
    """
    Determines if the agent's last message contains a tool call.
    If so, we should go to the tool node. Otherwise, the agent has a final
    response and we should end the graph.
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "continue"
    return "end"

# ======= Building the Graph ======= #

graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode([youtube_search, validate_date_format]))
graph.set_entry_point("agent")

# Define conditional edges from the 'agent' node
graph.add_conditional_edges(
    "agent",  # From the 'agent' node...
    should_continue,  # ...use this function to decide where to go...
    {
        "continue": "tools", # ...if 'continue', go to the 'tools' node
        "end": END,          # ...if 'end', terminate the graph
    }
)

# After the tools node, always return to the agent to process the tool output
graph.add_edge("tools", "agent")

# Compile the graph
app = graph.compile()

# ======= Execution ======= #

if __name__ == "__main__":
    print("\nHello! I am your assistant for youtube! What would you like to search for?")
    print("\nType 'quit' or 'exit' to end the conversation.")

    messages = []

    while True:
        user_input = input("\nUSER: ")
        if user_input.lower() in ['quit', 'exit']:
            break

        current_messages = messages + [HumanMessage(content=user_input)]
        final_state = app.invoke({"messages": current_messages})
        
        final_message = final_state['messages'][-1]
        messages = final_state['messages']

        print(f"\nAI: {final_message.content}")

    print("\n==== YOUTUBE ASSISTANT FINISHED ====")
