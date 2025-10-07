import os
from dotenv import load_dotenv
import requests
import tempfile
import chromadb
from urllib.parse import urlparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredURLLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema.document import Document
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
PERSIST_DIRECTORY = "chroma_db_google"

def vectorization(documents: List[Document]):
    """
    Splits a list of Document objects into chunks, creates vector embeddings,
    and saves them to a persistent ChromaDB vector store.

    Parameters:
    -----------
    documents : List[Document]
        A list of LangChain Document objects to be vectorized.
    """
    print("Splitting the document(s) into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    docs = text_splitter.split_documents(documents)
    print(f"Document(s) split into {len(docs)} chunks.")
    print("Initializing the embeddings model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create and persist the ChromaDB vector store
    print(
        f"Creating and persisting the ChromaDB vector store in '{PERSIST_DIRECTORY}'...")
    try:
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
        print("Vectorization complete and saved to disk.")
        return True
    except Exception as e:
        print(f"Error creating or persisting the vector store: {e}")
        return False


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

    elif 'text/html' in content_type:
        print(f"Detected web page from URL: {document_url}")
        print("Loading HTML content...")
        loader = UnstructuredURLLoader(urls=[document_url], headers=headers)

    else:
        print(
            f"Unsupported content type: {content_type}. This agent only supports PDF and HTML.")
        return False

    try:
        documents_from_url = loader.load()
        if not documents_from_url:
            print("Error: The loader returned no documents. The URL may be invalid or the content could not be parsed.")
            return False
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
                             'video_id': video_id, 'title': video_info['title']}))

    if not documents:
        print("No transcripts found to vectorize.")
        return False

    return vectorization(documents)
