"""
Configuration management for the Personal AI Agent.

This module centralizes all configuration settings including API keys,
model settings, and application parameters.
"""

import os
import logging
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv()

class Config:
    """Central configuration for the AI agent application."""

    # API Keys
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    YOUTUBE_API_KEY: Optional[str] = os.getenv("YOUTUBE_API_KEY")

    # Vector Store Configuration
    PERSIST_DIRECTORY: str = "chroma_db_google"

    # Model Configuration
    LLM_MODEL_NAME: str = "gemini-2.5-flash"
    LLM_TEMPERATURE: float = 0
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Text Splitting Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Retriever Configuration
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
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        if not cls.YOUTUBE_API_KEY:
            raise ValueError("YOUTUBE_API_KEY not found in environment variables")

    @classmethod
    def setup_logging(cls) -> None:
        """Configure logging for the application."""
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL.upper()),
            format=cls.LOG_FORMAT
        )


# Validate configuration on import
try:
    Config.validate()
except ValueError as e:
    logging.warning(f"Configuration validation warning: {e}")
