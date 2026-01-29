"""Library management node."""
from state import AgentState


def library_node(state: AgentState):
    """Node for managing stored documents (video transcripts and URLs)."""
    from tools.vectorization import list_stored_documents, delete_video_by_id, delete_url_documents

    while not state["quit"]:
        docs = list_stored_documents()
        videos = docs['videos']
        urls = docs['urls']
        total_docs = len(videos) + len(urls)

        if total_docs == 0:
            print("\nNo documents stored in the library.")
            input("Press Enter to go back...")
            state["go_back"] = True
            return state

        # Build a unified list for selection
        all_items = []
        print("\n=== Stored Documents ===")
        idx = 1

        if videos:
            print("Videos:")
            for video in videos:
                print(f"  {idx}. {video['title']}")
                all_items.append({'type': 'video', 'data': video})
                idx += 1

        if urls:
            print("URLs:")
            for url_doc in urls:
                print(f"  {idx}. {url_doc['title']}")
                all_items.append({'type': 'url', 'data': url_doc})
                idx += 1

        print("\nOptions:")
        print("  - Enter number(s) to delete (e.g., '1' or '1 2 3' or '1,2,3')")
        print("  - Type 'all' to delete all documents")
        print("  - Type 'q' to quit")
        print("  - Type 'back' to return to main menu")

        choice = input("\nUSER: ").strip().lower()

        if choice == 'q':
            state["quit"] = True
        elif choice == 'back':
            state["go_back"] = True
            return state
        elif choice == 'all':
            confirm = input(f"Delete ALL {total_docs} document(s)? (yes/no): ")
            if confirm.lower() in ['yes', 'y']:
                for item in all_items:
                    if item['type'] == 'video':
                        if delete_video_by_id(item['data']['video_id']):
                            print(f"Deleted: {item['data']['title']}")
                        else:
                            print(f"Failed to delete: {item['data']['title']}")
                    else:
                        if delete_url_documents(item['data']['source_url']):
                            print(f"Deleted: {item['data']['title']}")
                        else:
                            print(f"Failed to delete: {item['data']['title']}")
        else:
            # Parse multiple numbers (space or comma separated)
            parts = choice.replace(',', ' ').split()
            indices = []
            for part in parts:
                if part.isdigit():
                    indices.append(int(part) - 1)

            # Filter valid indices and get items to delete
            to_delete = []
            for idx in indices:
                if 0 <= idx < len(all_items):
                    to_delete.append(all_items[idx])

            if not to_delete:
                print("Invalid selection.")
            else:
                titles = ', '.join([item['data']['title'] for item in to_delete])
                confirm = input(f"Delete {len(to_delete)} document(s): {titles}? (yes/no): ")
                if confirm.lower() in ['yes', 'y']:
                    for item in to_delete:
                        if item['type'] == 'video':
                            if delete_video_by_id(item['data']['video_id']):
                                print(f"Deleted: {item['data']['title']}")
                            else:
                                print(f"Failed to delete: {item['data']['title']}")
                        else:
                            if delete_url_documents(item['data']['source_url']):
                                print(f"Deleted: {item['data']['title']}")
                            else:
                                print(f"Failed to delete: {item['data']['title']}")

    return state
