"""Load the handbook Markdown files and split them into chunks.

RAG step 1 (loading): read each local .md file into a LangChain Document,
which is the text plus metadata (source file, title, link to GitHub).

RAG step 2 (chunking): split the Documents into smaller, overlapping chunks,
because search works best on short, focused passages.

Check the result in the terminal:
    python -m src.load_documents
"""

import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path("data/handbook")
SOURCE_REPO_URL = "https://github.com/OsioLabs/emphandbook/blob/main"
MIN_CONTENT_CHARS = 50  # files with less text than this are skipped
CHUNK_SIZE = 800  # maximum characters per chunk
CHUNK_OVERLAP = 120  # characters repeated between neighbouring chunks

# Every handbook page ends with links like "[Next page](02values.md)".
NAV_LINK = re.compile(r"^\[(next page|previous page|table of contents)\]", re.IGNORECASE)


def clean_markdown(text: str) -> str:
    """Remove '---' divider lines and the page-navigation links."""
    kept_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "---" or NAV_LINK.match(stripped):
            continue
        kept_lines.append(line.rstrip())
    return "\n".join(kept_lines).strip()


def load_markdown_documents(data_dir: Path = DATA_DIR) -> tuple[list[Document], list[str]]:
    """Read every .md file under data_dir (including subfolders) into a Document.

    Returns (documents, skipped_files). Empty or near-empty files are skipped.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Handbook folder not found: {data_dir}. See README.md.")

    documents = []
    skipped_files = []

    for path in sorted(data_dir.rglob("*.md")):
        source = path.relative_to(data_dir).as_posix()  # e.g. "03benefits/02leave.md"
        text = clean_markdown(path.read_text(encoding="utf-8"))

        if len(text) < MIN_CONTENT_CHARS:
            skipped_files.append(source)
            continue

        # The title is the first Markdown heading ("# Leave" -> "Leave"), else the file name.
        heading = re.search(r"^#+\s*(.+)", text, re.MULTILINE)
        title = heading.group(1).strip() if heading else path.stem

        metadata = {
            "source": source,
            "title": title,
            "source_url": f"{SOURCE_REPO_URL}/{source}",
        }
        documents.append(Document(page_content=text, metadata=metadata))

    return documents, skipped_files


def split_documents(documents: list[Document]) -> list[Document]:
    """Split Documents into overlapping chunks; each chunk keeps its file's metadata.

    The splitter cuts at paragraphs first, then lines, then words,
    so chunks usually end at a natural boundary.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(documents)


if __name__ == "__main__":
    documents, skipped_files = load_markdown_documents()
    chunks = split_documents(documents)

    print(f"Loaded files: {len(documents)} | skipped: {skipped_files} | chunks: {len(chunks)}")
    for chunk in chunks[:3]:
        print(f"\n{chunk.metadata['source']} ({chunk.metadata['title']})")
        print(chunk.page_content[:150], "...")
