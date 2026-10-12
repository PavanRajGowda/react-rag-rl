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


def identity_response(*docs):
    return {
        "aggregations": {
            "companies": {
                "buckets": [
                    {
                        "key": {"id": d["company_id"]},
                        "identity": {
                            "hits": {
                                "hits": [
                                    {
                                        "_source": {
                                            k: d[k]
                                            for k in (
                                                "company_id",
                                                "company_name",
                                                "ticker",
                                            )
                                        }
                                    }
                                ]
                            }
                        },
                        "years": {
                            "buckets": [
                                {"key": y}
                                for y in d.get("report_years", [d["report_year"]])
                            ]
                        },
                    }
                    for d in docs
                ]
            }
        }
    }
