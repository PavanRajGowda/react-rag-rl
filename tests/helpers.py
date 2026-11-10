"""Synthetic article and question fixtures."""


def article():
    """Build a small source-article fixture with the requested identity and text."""
    return {
        "title": "Aster launch",
        "author": None,
        "source": "Example News",
        "published_at": "2023-10-01T00:00:00+00:00",
        "category": "technology",
        "url": "https://example.org/a",
        "body": "Aster launched its product in October. Revenue increased. " * 50,
    }


def question():
    """Build a question fixture with reference evidence metadata for preparation/retrieval
    tests.
    """
    return {
        "query": "Who launched a product?",
        "answer": "Aster",
        "question_type": "inference_query",
        "evidence_list": [
            {**article(), "fact": "Aster launched its product in October."}
        ],
    }


def sample():
    """Return the shared minimal article/question fixture set."""
    return article()


def passage():
    """Build a deterministic retrieval-document fixture for search and evidence tests."""
    return {
        "context_id": "a-0",
        "document_id": "a",
        "title": "Aster launch",
        "source": "Example News",
        "published_at": "2023-10-01T00:00:00+00:00",
        "author": None,
        "category": "technology",
        "url": "https://example.org/a",
        "chunk_index": 0,
        "start_char": 0,
        "end_char": 14,
        "token_count": 3,
        "text": "Aster launched",
    }
