"""State definitions and enums for the agent."""
from typing import TypedDict, List
import enum

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from tools.base_tools import ytinteraction


class Intent(enum.Enum):
    """Coerces LLM output for intent classification."""
    youtube = 'youtube'
    url = 'url'
    library = 'library'
    chat = 'chat'
    greeter = 'greeter'
    unsure = 'unsure'


class IntentClassification(BaseModel):
    """Structured output model for intent classification."""
    intent: Intent
    confidence: float = 1.0
    reasoning: str = ""


class AgentState(TypedDict):
    """The state of the agent, containing the conversation history."""
    messages: List[BaseMessage]
    ytrecords: ytinteraction
    current_task: str
    quit: bool
    go_back: bool
    session_urls: List[str]  # URLs in current task (for chat filtering)
    session_videos: List[dict]  # All videos fetched this session (for cleanup prompts)
    all_session_urls: List[str]  # All URLs fetched this session (for cleanup prompts)
    chat_scope: str  # 'session' = only new content, 'all' = all stored documents
