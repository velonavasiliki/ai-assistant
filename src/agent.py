from typing import TypedDict, List
from datetime import datetime, timezone
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from tools.base_tools import *
from langchain_chroma import Chroma
import json
import enum
from pydantic import BaseModel 
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from config import Config

Config.setup_logging()
logger = logging.getLogger(__name__)

# ======= Model setup ======= #

class AgentState(TypedDict):
    """The state of the agent, containing the conversation history."""
    messages: List[BaseMessage]
    ytrecords: ytinteraction
    current_task: str    # options: 'youtube', 'url', 'q'
    quit: bool = False
    go_back: bool = False


def create_youtube_tools(yt_instance: ytinteraction) -> List[BaseTool]:
    """Create YouTube tools with the ytinteraction instance"""
    return [
        YouTubeSearchTool(yt_instance),
        ValidateDateFormatTool(),
        GetTranscriptTool(yt_instance),
        VectorizeYTTranscriptsTool()
    ]

try:
    llm = ChatGoogleGenerativeAI(
        model=Config.LLM_MODEL_NAME, 
        temperature=Config.LLM_TEMPERATURE,
        google_api_key=Config.GOOGLE_API_KEY
    )
except Exception as e:
    logger.critical(f'Failed to initialize LLM: {e}')
    raise RuntimeError('Cannot start application without LLM access') from e

url_tools = [VectorizeURLTool()]
url_model = llm.bind_tools(url_tools)

try:
    youtube_tools = create_youtube_tools(ytinteraction())
except Exception as e:
    logger.error(f'Failed to create YouTube tools: {e}')
    raise

yt_model = llm.bind_tools(youtube_tools)

class Intent(enum.Enum):
    """Coerces LLM output"""
    youtube_search = 'youtube'
    url_recovery = 'url'
    unsure = 'unsure'

class IntentClassification(BaseModel):
    intent: Intent
    confidence: float = 1.0
    reasoning: str = ""

# ======= Nodes definitions ======= #

def greeter_intent_node(state: AgentState):
    """Agent that greets and identifies the user's intention"""
    if not state['messages']:
        intro_prompt = "Hello! I am your personal assistant! Do you want to learn about a url or search youtube and discuss the results?"
    else:
        intro_prompt = "Do you want to learn about a url or search youtube and discuss the results?"
    user_input = input(intro_prompt + "\nUSER: ")
    state['messages'].extend(
        [AIMessage(content=intro_prompt), HumanMessage(content=user_input)])

    # Create structured LLM
    structured_llm = llm.with_structured_output(IntentClassification)

    system_prompt = """You are a personal AI Assistant.
    Classify the user's intention into one of these categories:
    - youtube_search: User wants to search YouTube videos or discuss video content
    - url_recovery: User wants to analyze/process a document from a URL. 
    - unsure: Intent is unclear or doesn't fit the above categories
    The users does not need to give an actual url or an actual search term for youtube. It is enough to set their intention.
    """

    response = structured_llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_input)])

    logger.debug(f"AI thinks: {response}")

    # Extract the enum value
    intent_value = response.intent
    if intent_value == Intent.youtube_search:
        state["current_task"] = 'youtube'
        state["messages"].append(
            AIMessage(content=f"Intent classified as: {intent_value}"))
    elif intent_value == Intent.url_recovery:
        state["current_task"] = 'url'
        state["messages"].append(
            AIMessage(content=f"Intent classified as: {intent_value}"))
    else:
        while True:
            new_input = input(
                "I'm sorry, I do not understand. Type 'youtube' for youtube search or type 'url' for url retrieval. \nType 'q' to quit.\nUSER: ")
            if new_input in ['youtube', 'url', 'q']:
                state["current_task"] = new_input
                state["quit"] = True if new_input == "q" else False
                state["messages"].append(
                    AIMessage(content=f"Intent classified as: {new_input}"))
                break
    print(f"AI: Intent classified as {intent_value}.")

    return state


def youtube_node(state: AgentState):
    """Agent node that makes queries on youtube."""

    system_prompt = SystemMessage(content="""
        You are a personal AI assistant helping users search for YouTube videos.
        
        - Use `yt_search_tool` to search for videos when the user is ready to search. 
        Parameters of `yt_search_tool`:
        -----------
        query : Search term for YouTube videos.
        order : date', 'rating', 'relevance', 'viewCount' (default: 'viewCount')
        duration : 'any', 'long' (20+ min), 'medium' (4-20 min), 'short' (<4 min) (default: 'medium')
        num_results : number of videos to retrieve (default: 1)
        before : Date upper limit in format %m/%d/%Y (optional)
        after : Date lower limit in format %m/%d/%Y (optional)
                                  
        - Use `validate_date_tool` to validate date is of the form %m/%d/%Y, if needed by the user's request. 
        - If `validate_date_tool` returns False, only then tell user to provide it in the required format.
                                  
        - Be helpful and polite. Do not repeat what the user says.
    """)

    tool_message_found = None
    for i, message in enumerate(reversed(state["messages"])):
        if isinstance(message, ToolMessage) and message.name == 'yt_search_tool':
            tool_message_found = message
            break
        if i > 5:
            break
    logger.debug(f"YouTube records info: {state['ytrecords'].info}")
    if tool_message_found:
        try:
            results = json.loads(tool_message_found.content)
            print(f"\n🎥 Found YouTube Videos:")
            print("=" * 50)

            if isinstance(results, dict):
                for video_id, video_info in results.items():
                    print(f"📺 Title: {video_info.get('title', 'N/A')}")
                    print(f"👤 Channel: {video_info.get('channel', 'N/A')}")
                    print(f"📅 Date: {video_info.get('date', 'N/A')}")
                    print(f"🔗 Video ID: {video_id}")
                    print("-" * 30)

                print(f"\nGreat! Found {len(results)} video(s).")
            else:
                print(f"Search results: {results}")

        except (json.JSONDecodeError, Exception) as e:
            print(f"Search completed. Results: {tool_message_found.content}")

        user_choice = input(
            "\nWould you like to get transcripts for these videos? (yes/no/search again): ")

        if user_choice.lower() in ['yes', 'y']:
            return state
        elif user_choice.lower() in ['no', 'n']:
            state["quit"] = True
            return state

    if not any(isinstance(msg, HumanMessage) for msg in state["messages"] if "youtube" not in msg.content.lower()):
        print("I can help you find YouTube videos! Tell me what you're looking for.")
    else:
        print("What else do you want to search for on youtube?")
    print("\nType 'q' to quit.")

    while not state["quit"]:
        user_input = input("\nUSER: ")

        if user_input.lower() == 'q':
            state["quit"] = True
            break

        state["messages"].append(HumanMessage(content=user_input))

        recent_messages = state["messages"][-4:]

        response = yt_model.invoke([system_prompt] + recent_messages)
        
        state["messages"].append(response)

        if hasattr(response, 'tool_calls') and response.tool_calls:
            logger.debug(f"\nAI TOOL CALL: {response.tool_calls}\n")
            return state

    return state


def yt_transcript_node(state: AgentState):
    """Agent that fetches, vectorizes, and stores locally transcripts from youtube."""

    sys_message = SystemMessage(content=f"""
    You are an agent that fetches youtube transcripts from videos requested by the user.
    - Available video information: {json.dumps(state['ytrecords'].info)}
    - Use `get_transcript_tool` to fetch transcripts for video IDs that the user requests.
    - Use `vectorize_ytt_tool` to store the transcripts after fetching them.
    - Always be polite. Do not repeat what the user says.
    - If user asks for transcripts, extract the video IDs from the available videos and call the appropriate tools.
    """)

    while not (state['quit'] or state['go_back']):
        next_action = input(
            "Which transcripts do you want to retrieve from the video results?\nIf you want to perform another search, type 'again'.\nIf you want to quit, type 'q'.\nUSER: ")

        if next_action == 'q':
            state["quit"] = True
            break
        elif next_action == 'again':
            state["go_back"] = True
            break
        else:
            state["messages"].append(HumanMessage(content=next_action))
            response = yt_model.invoke(
                [sys_message] + [HumanMessage(content=next_action)])
            state["messages"].append(response)

            if hasattr(response, 'tool_calls') and response.tool_calls:
                return state
            else:
                print(f"\nAI: {response.content}")

    return state


def url_node(state: AgentState):
    """
    Agent node that takes content from url, vectorizes it and stores it locally.
    """
    system_prompt = SystemMessage(content="""
    You are a personal AI assistant. Your purpose is to help the user process documents from URLs.
    - Use the `vectorize_url_tool` whenever the user provides a URL.
    - Be friendly and helpful in your responses.
    - Inform the user if the URL cannot be processed.
    """)

    print("\nPlease provide a URL to process and vectorize.")
    print("\nType 'q' to end the conversation.")
    while not state["quit"]:
        user_input = input("\nUSER: ")

        if user_input == 'q':
            state["quit"] = True
            break

        state["messages"].append(HumanMessage(content=user_input))

        response = url_model.invoke(
            [system_prompt] + [HumanMessage(content=user_input)])
        state["messages"].append(response)

        if hasattr(response, 'tool_calls') and response.tool_calls:
            return state

        print(f"\nAI: {response.content}")

    return state


def rag_agent_node(state: AgentState):
    """Agent node for Q&A about retrieved and vectorized documents from url and video transcripts."""

    # Load the existing persisted ChromaDB vector store
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL_NAME
        )
    except ValueError as e:
        logger.error(f'Invalid API key or model: {e}')
        print('Failed to initialize embeddings.')
        state['current_task'] = 'greeter'
        return state

    try:
        vector_store = Chroma(
            persist_directory=Config.PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
    except Exception as e:
        logger.error(f'Failed to load vector store: {e}', exc_info=True)
        print('Error accessing document database.')
        state['current_task'] = 'greeter'
        return state

    try:
        retriever = vector_store.as_retriever(
            search_type=Config.RETRIEVER_SEARCH_TYPE,
            search_kwargs={"k": Config.RETRIEVER_K}
        )
    except Exception as e:
        print(f"Error loading vector store: {e}")
        print("No vectorized documents found. Please process some documents first.")
        state["current_task"] = "greeter"
        return state

    system_prompt = SystemMessage(content="""
    You are a Q&A assistant that answers questions based on retrieved documents.
    Use only the provided context to answer questions. If the context doesn't contain 
    relevant information, say you don't have enough information to answer the question.
    """)

    while not state["quit"]:
        user_question = input(
            "\nWhat would you like to know about the retrieved documents?\nType 'q' to quit, 'back' to start over.\nUSER: ")

        if user_question.lower() == 'q':
            state["quit"] = True
            break
        elif user_question.lower() == 'back':
            state["current_task"] = "greeter"
            break

        try:
            # Retrieve relevant documents from existing vector store
            retrieved_docs = retriever.invoke(user_question)

            if not retrieved_docs:
                print(
                    "\nAI: I couldn't find relevant information for your question in the stored documents.")
                continue

            # Prepare context from retrieved documents
            context = "\n\n".join(
                [f"Source {i+1}: {doc.page_content}" for i, doc in enumerate(retrieved_docs)])

            context_prompt = f"""
            Context from retrieved documents:
            {context}

            Question: {user_question}

            Please answer based only on the provided context."""

            state["messages"].append(HumanMessage(content=user_question))

            ai_response = llm.invoke(
                [system_prompt, HumanMessage(content=context_prompt)])
            state["messages"].append(ai_response)

            print(f"\nAI: {ai_response.content}")

        except Exception as e:
            print(f"Error during retrieval: {e}")

    return state

# ======= Conditional Edges ======= #


def yt_url_choice(state: AgentState) -> str:
    """
    Decide where to go after the greeter node based on the current_task.
    """
    if state["quit"]:
        return "quit"
    if state["current_task"] == "youtube":
        return "youtube"
    if state["current_task"] == "url":
        return "url"
    return "quit"


def start_over_or_quit(state: AgentState) -> str:
    """
    Decide whether to restart at the greeter or quit after QA.
    """
    if state["quit"]:
        return "quit"
    # If current_task was changed by user input inside QA, restart flow
    if state["current_task"] in ["youtube", "url", "greeter"]:
        return "restart"    
    return "continue"  


def should_loop(state: AgentState) -> str:
    """
    Determines if the agent's last message contains a tool call.
    If so, we should go to the tool node. Otherwise, the agent has a final
    response and we should end the graph.
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "loop"
    if state["quit"] == True:
        return "quit"
    return "continue"


# ======= Building the Graph ======= #

graph = StateGraph(AgentState)

graph.add_node("greeter_intent_node", greeter_intent_node)
graph.add_node("youtube_node", youtube_node)
graph.add_node("yt_transcript_node", yt_transcript_node)
graph.add_node("rag_agent_node", rag_agent_node)
graph.add_node("url_node", url_node)
graph.add_node("yttools", ToolNode(youtube_tools))
graph.add_node("vectorize_url_tool", ToolNode(url_tools))

graph.add_edge("vectorize_url_tool", "rag_agent_node")
graph.add_edge("yttools", "youtube_node")
graph.set_entry_point("greeter_intent_node")

graph.add_conditional_edges(
    "greeter_intent_node",
    yt_url_choice,
    {
        "youtube": "youtube_node",
        "url": "url_node",
        "quit": END
    }
)
graph.add_conditional_edges(
    "youtube_node",
    should_loop,
    {
        "loop": "yttools",
        "continue": "yt_transcript_node",
        "quit": END
    }
)
graph.add_conditional_edges(
    "yt_transcript_node",
    should_loop,
    {
        "loop": "yttools",
        "continue": "yt_transcript_node",
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


graph.add_edge("yt_transcript_node", "rag_agent_node")
graph.add_conditional_edges(
    "url_node",
    should_loop,
    {
        "loop": "vectorize_url_tool",
        "continue": "rag_agent_node",
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
        "quit": False
    }

    # Run the graph interactively
    while not state["quit"]:
        state = app.invoke(state)
    print("\n==== PERSONAL ASSISTANT FINISHED ====")
