"""Connect to Pinecone: hosted embeddings + the vector store that holds the chunks.

RAG step 3 (embedding): PineconeEmbeddings turns text into a vector of 1024 numbers
using Pinecone's hosted llama-text-embed-v2 model (nothing is downloaded locally).

RAG step 4 (storing and searching): PineconeVectorStore saves each chunk's vector,
text and metadata, and finds the chunks most similar to a question.

Try a search (after running build_index.py):
    python -m src.pinecone_store "What benefits do employees get?"
"""

import os
import sys

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()  # read the settings from the .env file

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "askhr-handbook")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "demo")
EMBEDDING_MODEL = os.getenv("PINECONE_EMBEDDING_MODEL", "llama-text-embed-v2")


def get_vector_store(create_index: bool = False) -> PineconeVectorStore:
    """Connect LangChain to the Pinecone index.

    Only build_index.py passes create_index=True, so asking a question
    can never create an index by accident.
    """
    api_key = os.getenv("PINECONE_API_KEY", "")
    if not api_key or api_key == "replace_me":
        raise ValueError("PINECONE_API_KEY is missing. Add your key to the .env file.")

    # The embedding model runs on Pinecone's servers. LangChain automatically
    # embeds chunks as "passages" and questions as "queries".
    embeddings = PineconeEmbeddings(model=EMBEDDING_MODEL, pinecone_api_key=api_key)
    pinecone_client = Pinecone(api_key=api_key)

    if not pinecone_client.has_index(INDEX_NAME):
        if not create_index:
            raise ValueError(f"Pinecone index '{INDEX_NAME}' not found. Run: python build_index.py")
        print(f"Creating Pinecone index '{INDEX_NAME}'...")
        pinecone_client.create_index(
            name=INDEX_NAME,
            dimension=embeddings.dimension,  # 1024 numbers per vector for this model
            metric="cosine",  # how "closeness" between vectors is measured
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),  # only region on the free plan
        )

    return PineconeVectorStore(
        index=pinecone_client.Index(INDEX_NAME),
        embedding=embeddings,
        namespace=NAMESPACE,
    )


def make_chunk_ids(chunks: list[Document]) -> list[str]:
    """Give each chunk a stable ID such as '03benefits/02leave.md#chunk-3'.

    The same files always get the same IDs, so re-running build_index.py
    overwrites the stored chunks instead of adding duplicates.
    """
    ids = []
    chunks_seen_per_file = {}
    for chunk in chunks:
        source = chunk.metadata["source"]
        number = chunks_seen_per_file.get(source, 0)
        chunks_seen_per_file[source] = number + 1
        ids.append(f"{source}#chunk-{number}")
    return ids


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What benefits do employees get?"
    results = get_vector_store().similarity_search_with_score(question, k=4)

    print(f"Question: {question}")
    for document, score in results:  # higher score = more similar
        print(f"\n{score:.3f}  {document.metadata['source']} ({document.metadata['title']})")
        print(document.page_content[:150], "...")
