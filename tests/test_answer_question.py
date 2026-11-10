"""Tests for the answer chain, using a fake vector store and a fake chat model."""

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.answer_question import NOT_FOUND_MESSAGE, answer_question, format_context

LEAVE_CHUNK = Document(
    page_content="Employees get 5 weeks of paid leave after 7 years.",
    metadata={"source": "03benefits/02leave.md", "title": "Leave"},
)


class FakeVectorStore:
    """Stands in for PineconeVectorStore and returns fixed search results."""

    def __init__(self, results):
        self.results = results

    def similarity_search(self, question, k=4):
        return self.results[:k]


class ModelThatMustNotBeCalled(FakeListChatModel):
    def invoke(self, *args, **kwargs):
        raise AssertionError("The model should not be called")


def test_context_includes_sources():
    context = format_context([LEAVE_CHUNK])

    assert "[Source 1: 03benefits/02leave.md (Leave)]" in context
    assert "Employees get 5 weeks of paid leave after 7 years." in context


def test_no_results_skips_llm():
    result = answer_question(
        "What is the policy on company cars?",
        vector_store=FakeVectorStore(results=[]),
        llm=ModelThatMustNotBeCalled(responses=["unused"]),
    )

    assert result == {"answer": NOT_FOUND_MESSAGE, "sources": []}


def test_returns_answer_and_sources():
    result = answer_question(
        "How much paid leave do long-serving employees get?",
        vector_store=FakeVectorStore(results=[LEAVE_CHUNK]),
        llm=FakeListChatModel(responses=["  You get 5 weeks of paid leave.  "]),
    )

    assert result["answer"] == "You get 5 weeks of paid leave."
    assert result["sources"] == [LEAVE_CHUNK]
