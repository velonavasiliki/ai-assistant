# AI Personal Assistant

An intelligent conversational agent built with LangChain and
LangGraph that helps users search YouTube videos, retrieve
transcripts, process documents from URLs, and answer questions
using Retrieval-Augmented Generation (RAG).

## Features

- **YouTube Integration**: Search YouTube videos with advanced
filters (date range, duration, sorting)
- **Transcript Retrieval**: Automatically fetch and process video
transcripts
- **URL Document Processing**: Extract and vectorize content from
web URLs (HTML, PDF)
- **RAG-Powered Q&A**: Ask questions about retrieved documents
using semantic search
- **Vector Storage**: Persistent storage using ChromaDB for
efficient retrieval
- **Interactive CLI**: Conversational interface with state
management using LangGraph

## Tech Stack

- **LLM**: Google Gemini 2.5 Flash
- **Framework**: LangChain + LangGraph for agent orchestration
- **Vector Database**: ChromaDB
- **Embeddings**: HuggingFace sentence-transformers
(all-MiniLM-L6-v2)
- **APIs**: YouTube Data API v3, Google Generative AI

## Architecture

The agent uses a state machine built with LangGraph, managing
different conversation flows:
- Intent classification (YouTube search vs URL processing)
- Tool execution (search, transcription, vectorization)
- RAG retrieval and question answering

## Installation

### Prerequisites
- Python 3.9+
- Google API Key (for Gemini)
- YouTube Data API Key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/velonavasiliki/ai-assistant.git
cd ai-assistant

2. Install dependencies:
pip install -r requirements.txt

3. Configure environment variables:
cp .env.example .env

Edit .env and add your API keys:
YOUTUBE_API_KEY='your_youtube_api_key_here'
GOOGLE_API_KEY='your_google_api_key_here'

Getting API Keys:
- Google API Key: https://aistudio.google.com/app/apikey
- YouTube API Key: https://console.cloud.google.com/ → Enable
YouTube Data API v3

Usage

Run the agent:
python3 src/agent.py

Example Workflow

1. YouTube Search:
- Agent asks if you want to search YouTube or process a URL
- Provide search query (e.g., "machine learning tutorials")
- Agent retrieves videos matching your criteria
- Choose videos to get transcripts
2. Document Processing:
- Provide a URL (HTML or PDF)
- Agent extracts and vectorizes content
3. Ask Questions:
- Query the retrieved documents
- Agent uses RAG to provide contextual answers

Project Structure

.
├── src/
│   ├── agent.py              # Main agent logic and LangGraph
workflow
│   ├── config.py             # Configuration management
│   └── tools/
│       ├── base_tools.py     # Custom LangChain tools
│       ├── ytinteraction.py  # YouTube API wrapper
│       └── vectorization.py  # Document processing and 
vectorization
├── .env.example              # Environment variables template
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
└── README.md                 # This file

Configuration

Key settings in src/config.py:
- LLM_MODEL_NAME: Default is "gemini-2.5-flash"
- EMBEDDING_MODEL_NAME: Sentence transformer model
- CHUNK_SIZE: Text chunking for vectorization (default: 1000)
- RETRIEVER_K: Number of documents to retrieve (default: 5)

Contributing

Contributions are welcome! Please feel free to submit issues or
pull requests.

License

This project is licensed under the MIT License - see the LICENSE
file for details.

Acknowledgments

- Built with https://python.langchain.com/ and
https://langchain-ai.github.io/langgraph/
- Powered by https://ai.google.dev/
- Vector storage with https://www.trychroma.com/
