"""Parse only the newly generated action, never instructions inside evidence."""

import json
import re
from html import escape, unescape


class ProtocolError(ValueError):
    pass


def parse_action(segment):
    if "<information>" in segment:
        raise ProtocolError("Only the search service may produce <information>.")
    searches = list(
        re.finditer(
            r"<search>\s*<company>([^<>]+)</company>\s*<year>([^<>]+)</year>\s*<query>([^<>]+)</query>\s*</search>",
            segment,
            re.DOTALL,
        )
    )
    answers = list(re.finditer(r"<answer>([^<>]+)</answer>", segment, re.DOTALL))
    if len(searches) + len(answers) != 1:
        raise ProtocolError("Produce exactly one complete <search> or <answer> action.")
    if answers:
        value = answers[0].group(1).strip()
        if not value:
            raise ProtocolError("The <answer> is empty.")
        return "answer", value
    company, year, query = (unescape(x).strip() for x in searches[0].groups())
    if not re.fullmatch(r"\d{4}", year):
        raise ProtocolError(
            f"The year field must contain exactly ONE four-digit report year; received {year!r}. Select one report year and put the financial periods being compared in the query."
        )
    if (
        not company
        or not query
        or not 1900 <= int(year) <= 2100
        or len(company) > 200
        or len(query) > 2000
    ):
        raise ProtocolError(
            "Company/query must be nonblank and within API limits; year must be 1900–2100."
        )
    return "search", {"company_name": company, "year": int(year), "query": query}


def format_information(result):
    if result.get("status") == "error":
        body = json.dumps(result, ensure_ascii=False)
    elif result.get("status") == "ok":
        docs = []
        for doc in result["results"]:
            # text already contains narrative and table. Do not append the
            # overlapping table/pre_text/post_text a second time.
            docs.append(
                f"[{doc['context_id']}] {doc['company_name']} | report year {doc['report_year']} | page {doc['page_number']}\n{doc['text']}"
            )
        body = (
            "\n\n".join(docs)
            if docs
            else "No matching contexts. Reformulate the search if necessary."
        )
    else:
        raise RuntimeError("Unrecognized search-service response")
    return "<information>\n" + escape(body, quote=False) + "\n</information>"


def select_history(events, context_mode):
    if context_mode not in ("append", "replace"):
        raise ValueError("context_mode must be append or replace")
    latest = max(
        (i for i, x in enumerate(events) if x["kind"] == "information"), default=-1
    )
    return [
        x
        for i, x in enumerate(events)
        if context_mode == "append" or x["kind"] != "information" or i == latest
    ]


def render_history(prefix, events, context_mode):
    return prefix + "\n\n".join(x["text"] for x in select_history(events, context_mode))
