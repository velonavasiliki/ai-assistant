"""YouTube search and transcript nodes."""
import json
import logging

from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage

from state import AgentState, Intent
from models import yt_model

logger = logging.getLogger(__name__)


def youtube_node(state: AgentState):
    """Agent node that makes queries on youtube."""
    # Reset go_back flag when entering this node
    state["go_back"] = False

    # Reset URL session content when switching to YouTube task
    state["session_urls"] = []

    system_prompt = SystemMessage(content="""
        You are a personal AI assistant helping users search for YouTube videos.

        - Use `yt_search_tool` to search for videos when the user is ready to search.
        Parameters of `yt_search_tool`:
        -----------
        query : Search term for YouTube videos.
        order : 'date', 'rating', 'relevance', 'viewCount' (default: 'viewCount')
        duration : 'any', 'long' (20+ min), 'medium' (4-20 min), 'short' (<4 min) (default: 'medium')
        num_results : number of videos to retrieve
        before : Date upper limit in format %m/%d/%Y (optional)
        after : Date lower limit in format %m/%d/%Y (optional)

        CRITICAL RULE - num_results parameter:
        - DO NOT include num_results in your tool call unless user explicitly asks for multiple videos
        - When user says "the latest", "a video", "find a video" → DO NOT pass num_results (defaults to 1)
        - Only when user says "find 5 videos", "show me multiple" → then pass num_results=5
        - NEVER pass num_results unless explicitly requested by user

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

        except (json.JSONDecodeError, Exception):
            print(f"Search completed. Results: {tool_message_found.content}")

        user_choice = input(
            "\nWould you like to get transcripts for these videos? (yes/no/search again): ")

        if user_choice.lower() in ['no', 'n']:
            state["go_back"] = True
            state["current_task"] = Intent.greeter.value
            return state
        elif user_choice.lower() in ['search again', 'again', 'search']:
            state["go_back"] = True
            # Fall through to search loop
        else:
            # Treat "yes", specific video requests, or any other input as wanting transcripts
            # Pass user's specific request to the transcript node via messages
            state["go_back"] = False  # Clear flag to ensure transcript node runs
            if user_choice.lower() not in ['yes', 'y']:
                state["messages"].append(HumanMessage(content=user_choice))
            return state

    # Check if there's been a previous YouTube search
    has_previous_search = any(isinstance(msg, ToolMessage) and msg.name == 'yt_search_tool'
                              for msg in state["messages"])
    if has_previous_search:
        print("What else do you want to search for on youtube? Type 'q' to quit.")
    else:
        print("I can help you find YouTube videos! Tell me what you're looking for. Type 'q' to quit.")

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
    import json

    sys_message = SystemMessage(content=f"""
    You are an agent that fetches youtube transcripts from videos requested by the user.
    - Available video information: {json.dumps(state['ytrecords'].info)}
    - Use `get_transcript_tool` to fetch transcripts for video IDs that the user requests.
    - Transcripts are automatically stored for Q&A after fetching.
    - Always be polite. Do not repeat what the user says.

    IMPORTANT - Video Selection:
    - If user says "yes", "get transcripts", "all of them", etc. → fetch transcripts for ALL available video IDs
    - If user specifies "the first one", "video 1", "the latest", "just one" → fetch ONLY the first video ID
    - If user specifies a particular video by title or number → fetch ONLY that specific video ID
    - When calling `get_transcript_tool`, pass a list containing only the requested video ID(s)

    Example: If there are 3 videos and user says "just the first one", only pass the first video ID like: ["video_id_1"]
    """)

    # Check if user already specified their preference from youtube_node
    pending_request = None
    if state["messages"] and isinstance(state["messages"][-1], HumanMessage):
        last_msg = state["messages"][-1].content.lower()
        # If the last message indicates a transcript request, process it directly
        if any(keyword in last_msg for keyword in ['transcript', 'video', 'title', 'first', 'all', 'yes']):
            pending_request = state["messages"][-1].content

    # If only one video, automatically fetch its transcript
    video_count = len(state['ytrecords'].info) if state.get('ytrecords') else 0
    if video_count == 1 and not pending_request:
        video_id = list(state['ytrecords'].info.keys())[0]
        video_title = state['ytrecords'].info[video_id].get('title', 'the video')
        print(f"\nFetching transcript for: {video_title}")
        pending_request = f"Get transcript for {video_id}"

    while not (state['quit'] or state['go_back']):
        skip_append = False
        if pending_request:
            next_action = pending_request
            pending_request = None  # Clear so we prompt on subsequent iterations
            skip_append = True  # Message was already added by youtube_node or auto-generated
        else:
            next_action = input(
                "Which transcripts do you want to retrieve from the video results?\nIf you want to perform another search, type 'again'.\nIf you want to quit, type 'q'.\nUSER: ")

        if next_action == 'q':
            state["quit"] = True
            break
        elif next_action == 'again':
            state["go_back"] = True
            break
        else:
            if not skip_append:
                state["messages"].append(HumanMessage(content=next_action))
            response = yt_model.invoke(
                [sys_message] + [HumanMessage(content=next_action)])
            state["messages"].append(response)

            if hasattr(response, 'tool_calls') and response.tool_calls:
                state["chat_scope"] = "session"  # Chat about newly fetched content
                return state
            else:
                print(f"\nAI: {response.content}")

    return state
