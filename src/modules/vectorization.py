import os
import requests
import tempfile
import chromadb
from urllib.parse import urlparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredURLLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema.document import Document
from typing import List 
from dotenv import load_dotenv

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
    docs = text_splitter.split_documents(documents) # 'documents' now correctly refers to the parameter
    print(f"Document(s) split into {len(docs)} chunks.")
    print("Initializing the embeddings model...")

    if not GOOGLE_API_KEY:
        print("WARNING: GOOGLE_API_KEY is not set. Embeddings may fail.")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

    # Create and persist the ChromaDB vector store
    print(f"Creating and persisting the ChromaDB vector store in '{PERSIST_DIRECTORY}'...")
    try:
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
        vector_store.persist()
        print("Vectorization complete and saved to disk.")
        return True 
    except Exception as e:
        print(f"Error creating or persisting the vector store: {e}")
        return False 

def vectorization_url(document_url: str):
    """
    Downloads and parses a document from URL (either PDF or HTML),
    and then calls the vectorization function to create vector embeddings
    and save them to a persistent ChromaDB vector store.

    Parameters:
    -----------
    document_url : str
    """
    print("Starting document fetching and vectorization process...")

    loader = None
    documents_from_url = [] # Renamed to avoid confusion
    temp_file_path = None # Keep temp_file_path here

    try:
        response_head = requests.head(document_url, allow_redirects=True)
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
            response_get = requests.get(document_url, allow_redirects=True)
            response_get.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error downloading the PDF: {e}")
            return False 
        
        # Use a temporary file to store the downloaded PDF
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(response_get.content)
            temp_file_path = tmp_file.name

        print(f"   Successfully downloaded to a temporary file: {temp_file_path}")
        loader = PyPDFLoader(temp_file_path)

    elif 'text/html' in content_type:
        print(f"Detected web page from URL: {document_url}")
        print("Loading HTML content...")
        loader = UnstructuredURLLoader(urls=[document_url])

    else:
        print(f"Unsupported content type: {content_type}. This agent only supports PDF and HTML.")
        return False 
    
    try:
        documents_from_url = loader.load()
        if not documents_from_url:
            print("Error: The loader returned no documents. The URL may be invalid or the content could not be parsed.")
            return False 
    except Exception as e:
        print(f"Error loading the document: {e}")
        return False 
    finally:
        # Clean up the temporary PDF file if one was created
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Cleaned up temporary file: {temp_file_path}")

    print(f"Document(s) fetched from URL. Ready for vectorization.")
    # Call the separate vectorization function with the loaded documents
    return vectorization(documents_from_url) # Pass the list of documents