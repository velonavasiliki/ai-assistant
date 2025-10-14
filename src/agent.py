import os
import operator
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Union
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
import html
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi
from modules.ytinteraction import ytinteraction
from modules.vectorization import vectorization_url, vectorize_yt_transcripts
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import json
import enum
from pydantic import BaseModel 
import logging
from langchain_huggingface import HuggingFaceEmbeddings

# ======= Setup ======= #

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError('GOOGLE_API_KEY not found in the environment variables.')
PERSIST_DIRECTORY = "chroma_db_google"

logger = logging.getLogger(__name__)

# ======= Tools ======= #


class YouTubeSearchTool(BaseTool):
    """Tool to search YouTube videos and return results as JSON."""
    name: str = "yt_search_tool"
    description: str = "Search YouTube for videos with filtering options like duration, date range, and sort order."
    yt_instance: ytinteraction = None

    def __init__(self, yt_instance: ytinteraction):
        super().__init__()
        self.yt_instance = yt_instance

    def _run(self, query: str, order: str = 'viewCount', duration: str = 'medium',
             num_results: int = 1, before: str = None, after: str = None) -> str:
        """Search YouTube and return formatted results as a JSON string.

        Parameters:
        -----------
        query : str
            Search term for YouTube videos.
        order : str, default 'viewCount'
            Sort order: 'date', 'rating', 'relevance', 'viewCount'.
        duration : str, default 'medium'
            Video duration: 'any', 'long' (20+ min), 'medium' (4-20 min), 'short' (<4 min).
        num_results : int, default 1
            Maximum number of videos to retrieve.
        before : str, optional
            Date upper limit in format %m/%d/%Y.
        after : str, optional
            Date lower limit in format %m/%d/%Y.

        Returns:
        --------
        str
            JSON string containing video information, or error message if no results found.
        """
        results = self.yt_instance.ytretriever(query=query, order=order, duration=duration,
                                               num_results=num_results, before=before, after=after)
        if not results:
            return f"No videos found for query: '{query}'"

        return json.dumps(results)


class ValidateDateFormatTool(BaseTool):
    """Tool to validate if a date string matches the expected format."""
    name: str = "validate_date_tool"
    description: str = "Validate if a date string is in %m/%d/%Y format."

    def _run(self, date_str: str, format_str: str = "%m/%d/%Y") -> str:
        """Validate date string against specified format.

        Parameters:
        -----------
        date_str : str
            The date string to validate.
        format_str : str, default "%m/%d/%Y"
            Expected date format pattern.

        Returns:
        --------
        str
            JSON string with validation result and details.
        """
        try:
            datetime.strptime(date_str, format_str)
            return json.dumps({"valid": True, "message": "Date format is valid"})
        except ValueError as e:
            return json.dumps({"valid": False, "message": f"Invalid date format: {str(e)}"})


class GetTranscriptTool(BaseTool):
    """Tool to retrieve YouTube video transcripts from given list of video IDs."""
    name: str = "get_transcript_tool"
    description: str = "Retrieve transcripts for YouTube videos by their video IDs."
    yt_instance: ytinteraction = None

    def __init__(self, yt_instance: ytinteraction):
        super().__init__()
        self.yt_instance = yt_instance

    def _run(self, ids: list[str]) -> str:
        """
        Retrieve and assemble transcripts for a list of YouTube video IDs.

        Parameters:
        -----------
        ids : list[str]
            A list of YouTube video IDs.

        Returns:
        --------
        str
            JSON string containing video information and transcripts.
        """
        try:
            result = self.yt_instance.yttranscript(ids)
            return json.dumps(result)
        except Exception as e:
            error_result = {
                "error": f"Failed to retrieve transcripts: {str(e)}", "ids": ids}
            return json.dumps(error_result)


class VectorizeURLTool(BaseTool):
    """Tool to download and vectorize documents from URLs.

    Use this tool when the user provides a URL and asks to "process", "vectorize",
    "ingest", or "learn about" a document from a link. Supports PDF and HTML formats.
    """
    name: str = "vectorize_url_tool"
    description: str = "Download and vectorize a document (PDF or HTML) from a URL."

    def _run(self, document_url: str) -> str:
        """
        Download and vectorize a document from the given URL.

        Parameters:
        -----------
        document_url : str
            URL of the document to download and vectorize.

        Returns:
        --------
        str
            Status message indicating success or failure of the vectorization process.
        """
        try:
            logger.debug(f"\n--- Vectorizing document from: {document_url} ---")
            result = vectorization_url(document_url)
            logger.debug(f"--- Result: {result} ---")
            return result
        except Exception as e:
            error_msg = f"Failed to vectorize document from {
                document_url}: {str(e)}"
            logger.debug(f"--- Error: {error_msg} ---")
            return error_msg


class VectorizeYTTranscriptsTool(BaseTool):
    """Tool to vectorize YouTube transcripts and store them in ChromaDB.

    Takes transcript data in JSON format, splits into chunks, creates embeddings,
    and saves to persistent vector store for semantic search.
    """
    name: str = "vectorize_ytt_tool"
    description: str = "Vectorize and store YouTube transcripts in vector database."

    def _run(self, transcript_data: str) -> str:
        """
        Process YouTube transcript data into vector embeddings.

        Parameters:
        -----------
        transcript_data : str
            JSON string with video ID as key and dict value containing:
            - title: Video title
            - channel: Channel name
            - date: Publication date
            - transcript: Full transcript text
            - transcript_summary: Summary of transcript

        Returns:
        --------
        str
            Success message with details or error message if processing failed.
        """
        try:
            result = vectorize_yt_transcripts(transcript_data)
            if result:
                return "Successfully vectorized and stored YouTube transcripts."
            else:
                return "Failed to vectorize transcripts - no data processed."
        except Exception as e:
            return f"Error vectorizing transcripts: {str(e)}"

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
        model="gemini-2.5-flash", 
        temperature=0,
        google_api_key=GOOGLE_API_KEY
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
    print(state["ytrecords"].info)
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
    logger.debug(f"\n INFO: {state["ytrecords"].info}\n")
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
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    except ValueError as e:
        logger.error(f'Invalid API key or model: {e}')
        print('Failed to initialize mbeddings.')
        state['current_task'] = 'greeter'
        return state

    try:
        vector_store = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
    except Exception as e:
        logger.error(f'Failed to load vector store: {e}', exc_info=True)
        print('Error accessing document database.')
        state['current_task'] = 'greeter'
        return state

    try:
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5}
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
