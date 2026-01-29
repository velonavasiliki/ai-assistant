"""LLM and tool setup for the agent."""
from typing import List
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from tools.base_tools import (
    YouTubeSearchTool,
    ValidateDateFormatTool,
    GetTranscriptTool,
    VectorizeURLTool,
    ytinteraction,
    BaseTool,
)
from config import Config

logger = logging.getLogger(__name__)


def create_youtube_tools(yt_instance: ytinteraction) -> List[BaseTool]:
    """Create YouTube tools with the ytinteraction instance."""
    return [
        YouTubeSearchTool(yt_instance),
        ValidateDateFormatTool(),
        GetTranscriptTool(yt_instance),
    ]


def create_llm():
    """Create and return the LLM based on configuration."""
    try:
        if Config.LLM_PROVIDER == "groq":
            llm = ChatGroq(
                model=Config.LLM_MODEL_NAME_GROQ,
                temperature=Config.LLM_TEMPERATURE,
                api_key=Config.GROQ_API_KEY
            )
            logger.info(f"Using Groq with model: {Config.LLM_MODEL_NAME_GROQ}")
        else:
            llm = ChatGoogleGenerativeAI(
                model=Config.LLM_MODEL_NAME_GOOGLE,
                temperature=Config.LLM_TEMPERATURE,
                google_api_key=Config.GOOGLE_API_KEY
            )
            logger.info(f"Using Google with model: {Config.LLM_MODEL_NAME_GOOGLE}")
        return llm
    except Exception as e:
        logger.critical(f'Failed to initialize LLM: {e}')
        raise RuntimeError('Cannot start application without LLM access') from e


# Initialize LLM
llm = create_llm()

# Initialize tools
url_tools = [VectorizeURLTool()]
url_model = llm.bind_tools(url_tools)

try:
    youtube_tools = create_youtube_tools(ytinteraction())
except Exception as e:
    logger.error(f'Failed to create YouTube tools: {e}')
    raise

yt_model = llm.bind_tools(youtube_tools)
