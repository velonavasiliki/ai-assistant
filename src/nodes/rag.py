"""RAG (Retrieval-Augmented Generation) Q&A node."""
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma

from state import AgentState, Intent
from models import llm
from config import Config

logger = logging.getLogger(__name__)


def _cleanup_session_content(state: AgentState):
    """Ask user about keeping/deleting content fetched in this session. Default is delete."""
    from tools.vectorization import delete_video_by_id, delete_url_documents

    # Handle videos
    session_videos_list = state.get('session_videos', [])
    if session_videos_list:
        print("\nYou fetched these transcripts in this session:")
        for i, vid in enumerate(session_videos_list, 1):
            print(f"  {i}. {vid['title']}")
        print("\nKeep them? (default: delete all)")
        print("  - 'yes' or 'all' to keep all")
        print("  - Enter numbers to KEEP specific ones (e.g., '1' or '1,2')")
        print("  - Press Enter or 'no' to delete all")
        keep_choice = input("\nYour choice: ").strip().lower()

        if keep_choice in ['yes', 'y', 'all']:
            print("Keeping all transcripts.")
        elif keep_choice in ['', 'no', 'n', 'none']:
            # Default: delete all
            for vid in session_videos_list:
                if delete_video_by_id(vid['id']):
                    print(f"Deleted: {vid['title']}")
                else:
                    print(f"Failed to delete: {vid['title']}")
        else:
            # Parse numbers to KEEP specific videos, delete the rest
            parts = keep_choice.replace(',', ' ').split()
            indices_to_keep = set()
            for part in parts:
                if part.isdigit():
                    idx = int(part) - 1
                    if 0 <= idx < len(session_videos_list):
                        indices_to_keep.add(idx)

            for idx, vid in enumerate(session_videos_list):
                if idx in indices_to_keep:
                    print(f"Keeping: {vid['title']}")
                else:
                    if delete_video_by_id(vid['id']):
                        print(f"Deleted: {vid['title']}")
                    else:
                        print(f"Failed to delete: {vid['title']}")

    # Handle URLs
    session_urls_list = state.get('all_session_urls', [])
    if session_urls_list:
        print("\nYou processed these URLs in this session:")
        for i, url in enumerate(session_urls_list, 1):
            print(f"  {i}. {url}")
        print("\nKeep them? (default: delete all)")
        print("  - 'yes' or 'all' to keep all")
        print("  - Enter numbers to KEEP specific ones (e.g., '1' or '1,2')")
        print("  - Press Enter or 'no' to delete all")
        keep_choice = input("\nYour choice: ").strip().lower()

        if keep_choice in ['yes', 'y', 'all']:
            print("Keeping all URLs.")
        elif keep_choice in ['', 'no', 'n', 'none']:
            # Default: delete all
            for url in session_urls_list:
                if delete_url_documents(url):
                    print(f"Deleted: {url}")
                else:
                    print(f"Failed to delete: {url}")
        else:
            # Parse numbers to KEEP specific URLs, delete the rest
            parts = keep_choice.replace(',', ' ').split()
            indices_to_keep = set()
            for part in parts:
                if part.isdigit():
                    idx = int(part) - 1
                    if 0 <= idx < len(session_urls_list):
                        indices_to_keep.add(idx)

            for idx, url in enumerate(session_urls_list):
                if idx in indices_to_keep:
                    print(f"Keeping: {url}")
                else:
                    if delete_url_documents(url):
                        print(f"Deleted: {url}")
                    else:
                        print(f"Failed to delete: {url}")

    # Clear session lists after cleanup
    state['session_videos'] = []
    state['all_session_urls'] = []


def rag_agent_node(state: AgentState):
    """Agent node for Q&A about retrieved and vectorized documents from url and video transcripts."""
    # Lazy import to avoid Python 3.9 compatibility error on startup
    from langchain_huggingface import HuggingFaceEmbeddings

    # Determine chat scope
    chat_scope = state.get("chat_scope", "all")

    # Get session content info for chat filtering (current task)
    session_video_ids = set()
    session_video_titles = []
    if state.get('ytrecords') and state['ytrecords'].info:
        for vid_id, vid_info in state['ytrecords'].info.items():
            if vid_info.get('transcript'):
                session_video_ids.add(vid_id)
                session_video_titles.append(vid_info.get('title', 'Unknown'))

                # Track for cleanup prompts (all session videos)
                if 'session_videos' not in state:
                    state['session_videos'] = []
                if not any(v['id'] == vid_id for v in state['session_videos']):
                    state['session_videos'].append({
                        'id': vid_id,
                        'title': vid_info.get('title', 'Unknown')
                    })

    session_urls = set(state.get('session_urls', []))

    has_session_content = len(session_video_ids) > 0 or len(session_urls) > 0

    # Load the existing persisted ChromaDB vector store
    print("\nLoading Q&A system...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL_NAME
        )
    except ValueError as e:
        logger.error(f'Invalid API key or model: {e}')
        print('Failed to initialize embeddings.')
        state['current_task'] = Intent.greeter.value
        return state

    try:
        vector_store = Chroma(
            persist_directory=Config.PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
    except Exception as e:
        logger.error(f'Failed to load vector store: {e}', exc_info=True)
        print('Error accessing document database.')
        state['current_task'] = Intent.greeter.value
        return state

    # Build metadata filter for session mode
    session_filter = None
    if chat_scope == "session" and has_session_content:
        conditions = []
        if session_video_ids:
            conditions.append({"video_id": {"$in": list(session_video_ids)}})
        if session_urls:
            # Check both source_url (our custom field) and source (WebBaseLoader's field)
            url_list = list(session_urls)
            conditions.append({"source_url": {"$in": url_list}})
            conditions.append({"source": {"$in": url_list}})

        if len(conditions) == 1:
            session_filter = conditions[0]
        elif len(conditions) > 1:
            session_filter = {"$or": conditions}

    try:
        search_kwargs = {"k": Config.RETRIEVER_K}
        if session_filter:
            search_kwargs["filter"] = session_filter

        retriever = vector_store.as_retriever(
            search_type=Config.RETRIEVER_SEARCH_TYPE,
            search_kwargs=search_kwargs
        )
    except Exception as e:
        print(f"Error loading vector store: {e}")
        print("No vectorized documents found. Please process some documents first.")
        state["current_task"] = Intent.greeter.value
        return state

    # Get list of all stored documents
    from tools.vectorization import list_stored_documents
    stored_docs = list_stored_documents()
    stored_videos = stored_docs['videos']
    stored_urls = stored_docs['urls']
    total_docs = len(stored_videos) + len(stored_urls)

    # Build context and display based on scope
    if chat_scope == "session" and has_session_content:
        # Session mode: only show newly fetched content
        print(f"\n📄 Chatting about newly fetched content:")
        if session_video_titles:
            print("   Videos:")
            for title in session_video_titles:
                print(f"     - {title}")
        if session_urls:
            print("   URLs:")
            for url in session_urls:
                print(f"     - {url}")

        docs_context = f"""
    You are chatting about newly fetched content in this session:
    {chr(10).join([f"- {t}" for t in session_video_titles])}
    {chr(10).join([f"- {u}" for u in session_urls])}
    Focus your answers on this content only.
    """
    else:
        # All mode: show all stored documents
        if total_docs > 0:
            print(f"\n📚 You have {total_docs} stored document(s):")
            if stored_videos:
                print("   Videos:")
                for doc in stored_videos:
                    print(f"     - {doc['title']}")
            if stored_urls:
                print("   URLs:")
                for doc in stored_urls:
                    print(f"     - {doc['title']}")
        else:
            print("\n📚 No documents stored yet. Fetch some transcripts or URLs first! Type 'q' to quit.")
            state["current_task"] = Intent.greeter.value
            return state

        stored_docs_list = []
        if stored_videos:
            for doc in stored_videos:
                stored_docs_list.append(f"- [Video] {doc['title']}")
        if stored_urls:
            for doc in stored_urls:
                stored_docs_list.append(f"- [URL] {doc['title']}")

        docs_context = f"""
    Stored documents in the library:
    {chr(10).join(stored_docs_list)}
    """

    system_prompt = SystemMessage(content=f"""
    You are a Q&A assistant that answers questions based on retrieved documents.
    Use only the provided context to answer questions. If the context doesn't contain
    relevant information, say you don't have enough information to answer the question.

    When the user asks "which documents are about X" or "what do I have about X",
    look at the retrieved context and tell them which documents mention that topic.
    {docs_context}
    """)

    prompt_text = "\nAsk me anything about your documents. Type 'q' to quit, 'back' to go back.\nUSER: "

    while not state["quit"]:
        user_question = input(prompt_text)

        if user_question.lower() in ['q', 'back']:
            # Ask about keeping content fetched in this session
            _cleanup_session_content(state)

            if user_question.lower() == 'q':
                state["quit"] = True
            else:
                state["current_task"] = Intent.greeter.value
            break

        try:
            # Retrieve relevant documents from existing vector store
            # (session filtering is now done via metadata filter in the retriever)
            retrieved_docs = retriever.invoke(user_question)

            if not retrieved_docs:
                print(
                    "\nAI: I couldn't find relevant information for your question in the documents.")
                continue

            # Prepare context from retrieved documents
            context = "\n\n".join(
                [f"Source {i+1} ({doc.metadata.get('title', 'Unknown')}): {doc.page_content}"
                 for i, doc in enumerate(retrieved_docs)])

            context_prompt = f"""
            Context from retrieved documents:
            {context}

            Question: {user_question}

            Please answer based only on the provided context."""

            state["messages"].append(HumanMessage(content=user_question))

            ai_response = llm.invoke(
                [system_prompt, HumanMessage(content=context_prompt)])
            state["messages"].append(ai_response)

            print(f"\nAI: {ai_response.content}")

        except Exception as e:
            print(f"Error during retrieval: {e}")

    return state
