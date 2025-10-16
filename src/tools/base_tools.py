from tools.ytinteraction import ytinteraction
from tools.vectorization import vectorization_url, vectorize_yt_transcripts
from langchain_core.tools import BaseTool, tool
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

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
