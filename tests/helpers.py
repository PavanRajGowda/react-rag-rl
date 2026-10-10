"""Synthetic financial context used by component tests."""


def sample(context_id="c1", ticker="AAPL", cik=320193):
    return {
        "id": "q1",
        "context_id": context_id,
        "context": "Revenue (in millions)\n| year | value |\n| 2001 | (12) |\n* footnote",
        "table": "| year | value |\n| 2001 | (12) |",
        "pre_text": "Revenue (in millions)",
        "post_text": "* footnote",
        "company_name": "Apple Inc.",
        "company_symbol": ticker,
        "company_cik": cik,
        "report_year": "2002",
        "page_number": "23",
        "file_name": "pdf/AAPL/2002/page_23.pdf",
        "question": "Revenue?",
        "program_answer": "-12",
        "original_answer": "-12",
    }
