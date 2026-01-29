"""URL processing node."""
import re
from langchain_core.messages import HumanMessage, AIMessage

from state import AgentState


def _extract_urls(text: str) -> list[str]:
    """Extract URLs from text."""
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return re.findall(url_pattern, text)


def url_node(state: AgentState):
    """Agent node that takes content from urls, vectorizes them and stores locally."""
    from tools.vectorization import vectorization_url

    # Reset YouTube session content when switching to URL task
    if state.get('ytrecords'):
        state['ytrecords'].info = {}

    print("\nProvide URL(s) to process. You can enter multiple URLs (comma or space separated).")
    print("Type 'done' when finished adding URLs, 'q' to quit.\n")

    processed_urls = []

    while not state["quit"]:
        user_input = input("USER: ").strip()

        if user_input.lower() == 'q':
            state["quit"] = True
            break

        if user_input.lower() == 'done':
            if processed_urls:
                state["chat_scope"] = "session"
                # Route to RAG by not having tool_calls
                state["messages"].append(AIMessage(content=f"Processed {len(processed_urls)} URL(s). Ready to chat."))
                return state
            else:
                print("No URLs processed yet. Please provide at least one URL or type 'q' to quit.")
                continue

        # Extract URLs from input
        urls = _extract_urls(user_input)

        if not urls:
            print("No valid URL found. Please enter a URL starting with http:// or https://")
            continue

        # Process each URL
        for url in urls:
            print(f"\nProcessing: {url}")
            state["messages"].append(HumanMessage(content=f"Process URL: {url}"))

            try:
                result = vectorization_url(url)
                if result:
                    # Track URL for current task (chat filtering)
                    if 'session_urls' not in state:
                        state['session_urls'] = []
                    if url not in state['session_urls']:
                        state['session_urls'].append(url)

                    # Track URL for entire session (cleanup prompts)
                    if 'all_session_urls' not in state:
                        state['all_session_urls'] = []
                    if url not in state['all_session_urls']:
                        state['all_session_urls'].append(url)

                    processed_urls.append(url)
                    print(f"✓ Added successfully\n")
                else:
                    print(f"✗ Failed to process URL\n")
            except Exception as e:
                print(f"✗ Error processing URL: {e}\n")

        if processed_urls:
            print(f"URLs added so far: {len(processed_urls)}")
            print("Add more URLs or type 'done' to start chatting.\n")

    return state
