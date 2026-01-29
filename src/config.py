"""
Configuration management.
This module provides a Config class that centralizes all configuration settings for the AI agent application.
""" 

import os
import logging
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv()

# Set default USER_AGENT to suppress warning from web libraries
if not os.getenv("USER_AGENT"):
    os.environ["USER_AGENT"] = "ai-assistant/1.0"

class Config:
    """Central configuration for the AI agent application."""

    # API Keys
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    YOUTUBE_API_KEY: Optional[str] = os.getenv("YOUTUBE_API_KEY")
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")

    # LLM Provider: "groq" or "google"
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")

    # Vector Store Configuration
    PERSIST_DIRECTORY: str = "chroma_db_google"

    # Model Configuration
    LLM_MODEL_NAME_GOOGLE: str = "gemini-2.5-flash-lite"
    LLM_MODEL_NAME_GROQ: str = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE: float = 0
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Text Splitting Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Retriever Configuration - chose maximal marginal relevance instead of similarity search
    RETRIEVER_SEARCH_TYPE: str = "mmr"
    RETRIEVER_K: int = 5

    # YouTube Search Defaults
    YT_DEFAULT_ORDER: str = "viewCount"
    YT_DEFAULT_DURATION: str = "medium"
    YT_DEFAULT_NUM_RESULTS: int = 1
    YT_DEFAULT_YEARS_BACK: int = 5

    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @classmethod
    def validate(cls) -> None:
        """Validate that all required configuration is present."""
        if cls.LLM_PROVIDER == "google" and not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        if cls.LLM_PROVIDER == "groq" and not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        if not cls.YOUTUBE_API_KEY:
            logging.warning("YOUTUBE_API_KEY not found - YouTube search will fail unless MOCK_MODE=1")

    @classmethod
    def setup_logging(cls) -> None:
        """Configure logging for the application."""
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL.upper()),
            format=cls.LOG_FORMAT
        )
        # Silence noisy third-party loggers
        for logger_name in [
            "chromadb",
            "sentence_transformers",
            "httpx",
            "google_genai",
            "googleapiclient",
            "posthog",
        ]:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        # Completely silence chromadb telemetry and segment warnings
        logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
        logging.getLogger("chromadb.segment.impl.vector.local_persistent_hnsw").setLevel(logging.ERROR)


# Validate configuration on import
try:
    Config.validate()
except ValueError as e:
    logging.warning(f"Configuration validation warning: {e}")
