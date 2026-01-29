# AI Assistant

A multi-agent system built with LangGraph for retrieving and querying YouTube transcripts and web documents.

## What it does

- Search YouTube videos with filters (date, duration, sort order)
- Fetch and store video transcripts
- Process web documents (HTML, PDF)
- Answer questions about stored content using RAG

## Tech Stack

- **LLM**: Google Gemini / Groq Llama
- **Framework**: LangChain + LangGraph
- **Vector Store**: ChromaDB
- **Embeddings**: HuggingFace sentence-transformers

## Architecture

```
User Input
    │
    ▼
┌─────────────────┐
│ Intent Classifier│
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│YouTube│ │  URL  │
│ Search│ │Process│
└───┬───┘ └───┬───┘
    │         │
    ▼         ▼
┌─────────────────┐
│  Vectorization  │
│   (ChromaDB)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   RAG Q&A       │
└─────────────────┘
```

## Setup

```bash
git clone https://github.com/velonavasiliki/ai-assistant.git
cd ai-assistant
pip install -r requirements.txt
```

Create `.env` with your API keys:
```
GOOGLE_API_KEY=your_key
YOUTUBE_API_KEY=your_key
GROQ_API_KEY=your_key        # optional
LLM_PROVIDER=google          # or groq
```

## Usage

```bash
python src/agent.py
```

Test without API keys:
```bash
MOCK_MODE=1 python src/agent.py
```

## Project Structure

```
src/
├── agent.py           # LangGraph state machine
├── config.py          # Configuration
├── state.py           # State definitions
├── routing.py         # Graph routing logic
├── models.py          # LLM and tool bindings
├── nodes/             # Agent nodes
│   ├── greeter.py     # Intent classification
│   ├── youtube.py     # YouTube search & transcripts
│   ├── url.py         # URL processing
│   ├── rag.py         # Q&A with retrieval
│   └── library.py     # Document management
└── tools/
    ├── base_tools.py      # LangChain tools
    ├── ytinteraction.py   # YouTube API
    └── vectorization.py   # ChromaDB storage
```

## License

MIT
