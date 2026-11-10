"""Build the Pinecone index from the local handbook files.

Run once, and again only when the handbook files change:
    python build_index.py           # add or update all chunks
    python build_index.py --reset   # first delete everything in the namespace

The app never runs this; answering questions only reads from Pinecone.
"""

import argparse

from pinecone.exceptions import NotFoundException

from src.load_documents import load_markdown_documents, split_documents
from src.pinecone_store import NAMESPACE, get_vector_store, make_chunk_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Index the handbook in Pinecone.")
    parser.add_argument("--reset", action="store_true", help="delete all stored chunks first")
    args = parser.parse_args()

    # Steps 1-2: load the Markdown files and split them into chunks.
    documents, skipped_files = load_markdown_documents()
    chunks = split_documents(documents)
    print(f"Loaded {len(documents)} files (skipped {len(skipped_files)}), {len(chunks)} chunks.")

    # Connect to Pinecone. The index is created automatically on the first run.
    vector_store = get_vector_store(create_index=True)

    if args.reset:
        answer = input(f"Delete ALL vectors in namespace '{NAMESPACE}'? Type 'yes': ")
        if answer.strip().lower() != "yes":
            print("Reset cancelled. Nothing was changed.")
            return
        try:
            vector_store.delete(delete_all=True)
        except NotFoundException:
            pass  # the namespace is already empty
        print(f"Cleared namespace '{NAMESPACE}'.")

    # Steps 3-4: embed the chunks and store them in Pinecone.
    ids = vector_store.add_documents(chunks, ids=make_chunk_ids(chunks))
    print(f"Stored {len(ids)} chunks in Pinecone (namespace '{NAMESPACE}').")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")
