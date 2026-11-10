"""Tests for loading and chunking the handbook Markdown files."""

from src.load_documents import clean_markdown, load_markdown_documents, split_documents

POLICY_TEXT = "# Paid Time Off\n\nEmployees can take paid time off after asking their team lead."
NO_HEADING_TEXT = "During your first week you will meet the team and set up your laptop."


def write_file(base_dir, relative_path, text):
    """Create a small test file inside the temporary handbook folder."""
    path = base_dir / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_loads_only_markdown(tmp_path):
    write_file(tmp_path, "03benefits/02leave.md", POLICY_TEXT)
    write_file(tmp_path, "02welcome/details/02first_week.md", NO_HEADING_TEXT)
    write_file(tmp_path, "03benefits/logo.png", "not really an image")
    write_file(tmp_path, "notes.txt", POLICY_TEXT)

    documents, _ = load_markdown_documents(tmp_path)
    by_source = {doc.metadata["source"]: doc.metadata for doc in documents}

    # Nested .md files are found; other file types are ignored.
    assert sorted(by_source) == ["02welcome/details/02first_week.md", "03benefits/02leave.md"]
    # The title comes from the first heading, or from the file name if there is none.
    assert by_source["03benefits/02leave.md"]["title"] == "Paid Time Off"
    assert by_source["02welcome/details/02first_week.md"]["title"] == "02first_week"
    assert by_source["03benefits/02leave.md"]["source_url"].endswith("/03benefits/02leave.md")


def test_skips_empty_files(tmp_path):
    write_file(tmp_path, "good.md", POLICY_TEXT)
    write_file(tmp_path, "empty.md", "   \n\n  ")
    write_file(tmp_path, "tiny.md", "# Hi")

    documents, skipped_files = load_markdown_documents(tmp_path)

    assert [doc.metadata["source"] for doc in documents] == ["good.md"]
    assert sorted(skipped_files) == ["empty.md", "tiny.md"]


def test_removes_navigation_links():
    raw_text = (
        "---\n# What We Do\n\nWe teach open source tools.\n\n---\n"
        "[Next page](02values.md)  \n"
        "[Previous page](../01who_we_are)  \n"
        "[Table of Contents](../README.md#table-of-contents)\n"
    )

    assert clean_markdown(raw_text) == "# What We Do\n\nWe teach open source tools."


def test_chunks_keep_metadata(tmp_path):
    paragraph = "Employees should communicate openly with their team. " * 10
    long_text = "# Conduct\n\n" + "\n\n".join([paragraph] * 5)
    write_file(tmp_path, "04emp_conduct/01standards.md", long_text)

    documents, _ = load_markdown_documents(tmp_path)
    chunks = split_documents(documents)

    assert len(chunks) > 1
    for chunk in chunks:
        assert 0 < len(chunk.page_content.strip()) <= 800
        assert chunk.metadata["source"] == "04emp_conduct/01standards.md"
        assert chunk.metadata["title"] == "Conduct"
