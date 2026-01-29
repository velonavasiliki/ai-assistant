"""Greeter node that identifies user intent."""
import logging

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from state import AgentState, Intent, IntentClassification
from models import llm

logger = logging.getLogger(__name__)


def greeter_intent_node(state: AgentState):
    """Agent that greets and identifies the user's intention."""
    # Reset go_back flag when entering this node
    state["go_back"] = False

    if not state['messages']:
        intro_prompt = "Hello! I am your personal assistant! Do you want to recover web content from url, recover youtube transcripts, manage your document library, or chat about stored documents?"
    else:
        intro_prompt = "Do you want to recover web content from url, recover youtube transcripts, manage your document library, or chat about stored documents?"
    user_input = input(intro_prompt + "\nUSER: ")

    # Check for quit before LLM classification
    if user_input.lower() == 'q':
        state["quit"] = True
        return state

    state['messages'].extend(
        [AIMessage(content=intro_prompt), HumanMessage(content=user_input)])

    # Create structured LLM
    structured_llm = llm.with_structured_output(IntentClassification)

    system_prompt = """You are a personal AI Assistant.
    Classify the user's intention into one of these categories:
    - youtube: User wants to search YouTube videos or fetch new transcripts
    - url: User wants to analyze/process a document from a URL
    - library: User wants to manage stored documents, delete stored transcripts or URLs
    - chat: User wants to chat about, ask questions about, or search through already stored documents
    - unsure: Intent is unclear or doesn't fit the above categories
    The users does not need to give an actual url or an actual search term for youtube. It is enough to set their intention.
    """

    response = structured_llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_input)])

    logger.debug(f"AI thinks: {response}")

    # Extract the enum value
    intent_value = response.intent
    if intent_value == Intent.youtube:
        state["current_task"] = Intent.youtube.value
        state["messages"].append(
            AIMessage(content=f"Intent classified as: {intent_value}"))
    elif intent_value == Intent.url:
        state["current_task"] = Intent.url.value
        state["messages"].append(
            AIMessage(content=f"Intent classified as: {intent_value}"))
    elif intent_value == Intent.library:
        state["current_task"] = Intent.library.value
        state["messages"].append(
            AIMessage(content=f"Intent classified as: {intent_value}"))
    elif intent_value == Intent.chat:
        state["current_task"] = Intent.chat.value
        state["chat_scope"] = "all"  # Explicit chat = all stored documents
        state["messages"].append(
            AIMessage(content=f"Intent classified as: {intent_value}"))
    else:
        while True:
            new_input = input(
                "I'm sorry, I do not understand. Type 'youtube', 'url', 'library', or 'chat'.\nType 'q' to quit.\nUSER: ")
            if new_input in ['youtube', 'url', 'library', 'chat', 'q']:
                state["current_task"] = new_input
                state["quit"] = True if new_input == "q" else False
                state["messages"].append(
                    AIMessage(content=f"Intent classified as: {new_input}"))
                break

    return state
