import requests
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from config import Config
import os

def vectorization(documents: List[Document]):
    """
    Splits a list of Document objects into chunks, creates vector embeddings,
    and saves them to a persistent ChromaDB vector store.

    Parameters:
    -----------
    documents : List[Document]
        A list of LangChain Document objects to be vectorized.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL_NAME
    )

    # Create and persist the ChromaDB vector store
    print(
        f"Creating and persisting the ChromaDB vector store in '{Config.PERSIST_DIRECTORY}'...")
    try:
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=Config.PERSIST_DIRECTORY
        )
        print("Vectorization complete and saved to disk.")
        return True
    except Exception as e:
        print(f"Error creating or persisting the vector store: {e}")
        return False


def _extract_title_from_url(url: str, content: str = None) -> str:
    """Extract a title from URL or content."""
    # Try to get title from HTML content
    if content:
        import re
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', content, re.IGNORECASE)
        if title_match:
            return title_match.group(1).strip()

    # Fall back to extracting from URL path
    from urllib.parse import urlparse, unquote
    parsed = urlparse(url)
    path = unquote(parsed.path)

    # Get the last meaningful part of the path
    parts = [p for p in path.split('/') if p]
    if parts:
        # Clean up the last part (remove extensions, replace underscores)
        title = parts[-1]
        title = title.rsplit('.', 1)[0]  # Remove extension
        title = title.replace('_', ' ').replace('-', ' ')
        return title.title()

    # Fall back to domain name
    return parsed.netloc


def vectorization_url(document_url: str):
    """
    Downloads and parses a document from URL (either PDF or HTML),
    then calls the vectorization function to create vector embeddings
    and save them to a persistent ChromaDB vector store.

    Parameters:
    -----------
    document_url : str
    """
    print("Starting document fetching and vectorization process...")

    loader = None
    documents_from_url = []
    temp_file_path = None
    page_title = None

    headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36'
    }

    try:
        response_head = requests.head(document_url, headers=headers, allow_redirects=True)
        response_head.raise_for_status()
        content_type = response_head.headers.get('Content-Type', '').lower()
        print(f"Content-Type detected: {content_type}")
    except Exception as e:
        print(f"Error getting headers from URL: {e}")
        return False

    if 'application/pdf' in content_type:
        print(f"Detected PDF file from URL: {document_url}")
        print("Downloading PDF from URL...")
        try:
            response_get = requests.get(document_url, headers=headers, allow_redirects=True)
            response_get.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error downloading the PDF: {e}")
            return False

        # Use a temporary file to store the downloaded PDF
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(response_get.content)
            temp_file_path = tmp_file.name

        print(f"Successfully downloaded to a temporary file: {temp_file_path}")
        loader = PyPDFLoader(temp_file_path)
        page_title = _extract_title_from_url(document_url)

    elif 'text/html' in content_type:
        print(f"Detected web page from URL: {document_url}")
        print("Loading HTML content...")
        # Fetch content to extract title
        try:
            response_get = requests.get(document_url, headers=headers, allow_redirects=True)
            page_title = _extract_title_from_url(document_url, response_get.text)
        except Exception:
            page_title = _extract_title_from_url(document_url)
        loader = WebBaseLoader(
            web_paths=[document_url],
            header_template=headers
        )

    else:
        print(
            f"Unsupported content type: {content_type}. This agent only supports PDF and HTML.")
        return False

    try:
        documents_from_url = loader.load()
        if not documents_from_url:
            print("Error: The loader returned no documents. The URL may be invalid or the content could not be parsed.")
            return False

        # Add consistent metadata to all documents
        for doc in documents_from_url:
            doc.metadata['title'] = page_title
            doc.metadata['source_url'] = document_url
            doc.metadata['doc_type'] = 'url'

        vect = vectorization(documents_from_url)
    except Exception as e:
        print(f"Error loading the document: {e}")
        return False
    finally:
        # Clean up the temporary PDF file if one was created
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Cleaned up temporary file: {temp_file_path}")

    print(f"Document(s) fetched from URL. Ready for vectorization.")
    return vect


def vectorize_yt_transcripts(transcript_data: dict) -> bool:
    """
    Splits a YouTube transcript into chunks, creates vector embeddings, and saves them to a persistent ChromaDB vector store.

    Parameters:
        transcript_data dict: Dictionary with video IDs as keys and title, channel, date, transcript, transcript_summary as values.
    """
    documents = []
    for video_id, video_info in transcript_data.items():
        if video_info.get('transcript'):
            documents.append(Document(page_content=video_info['transcript'], metadata={
                'video_id': video_id,
                'title': video_info['title'],
                'doc_type': 'youtube'
            }))

    if not documents:
        print("No transcripts found to vectorize.")
        return False

    return vectorization(documents)


def list_stored_documents() -> dict:
    """
    Returns dict with 'videos' and 'urls' keys, each containing a list of stored documents.
    Videos: [{video_id, title}, ...]
    URLs: [{source_url, title}, ...]
    """
    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)

    result = {'videos': [], 'urls': []}

    # Check if persist directory exists
    if not os.path.exists(Config.PERSIST_DIRECTORY):
        return result

    try:
        vector_store = Chroma(
            persist_directory=Config.PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
        collection = vector_store._collection

        # Get all documents' metadata
        results = collection.get(include=["metadatas"])

        # Extract unique documents by type
        videos = {}
        urls = {}
        for metadata in results.get("metadatas", []):
            if not metadata:
                continue

            doc_type = metadata.get("doc_type", "")

            if doc_type == "youtube" or "video_id" in metadata:
                vid = metadata.get("video_id")
                if vid and vid not in videos:
                    videos[vid] = {
                        "video_id": vid,
                        "title": metadata.get("title", "Unknown")
                    }
            elif doc_type == "url" or "source_url" in metadata:
                source = metadata.get("source_url") or metadata.get("source", "")
                if source and source not in urls:
                    urls[source] = {
                        "source_url": source,
                        "title": metadata.get("title", "Unknown")
                    }

        result['videos'] = list(videos.values())
        result['urls'] = list(urls.values())
        return result
    except Exception as e:
        print(f"Error accessing vector store: {e}")
        return result


def list_stored_videos() -> list[dict]:
    """
    Returns list of unique videos stored in ChromaDB.
    Each dict contains: video_id, title
    """
    return list_stored_documents()['videos']


def delete_video_by_id(video_id: str) -> bool:
    """
    Deletes all chunks for a specific video from ChromaDB.
    Returns True if successful, False otherwise.
    """
    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)

    try:
        vector_store = Chroma(
            persist_directory=Config.PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
        collection = vector_store._collection

        # Delete by video_id metadata
        collection.delete(where={"video_id": {"$eq": video_id}})
        return True
    except Exception as e:
        print(f"Error deleting video: {e}")
        return False


def delete_url_documents(source_url: str) -> bool:
    """
    Deletes all chunks for a specific URL from ChromaDB.
    Returns True if successful, False otherwise.
    """
    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)

    try:
        vector_store = Chroma(
            persist_directory=Config.PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
        collection = vector_store._collection

        # Delete by source metadata (loaders store URL in 'source' field)
        collection.delete(where={"source": {"$eq": source_url}})
        return True
    except Exception as e:
        print(f"Error deleting URL documents: {e}")
        return False
