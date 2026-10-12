"""Parse only the newly generated action, never instructions inside evidence."""

import json
import re
from html import escape, unescape


class ProtocolError(ValueError):
    pass


def parse_action(segment):
    # Tag mentions in reasoning are not actions.
    actions = segment.rsplit("</reasoning>", 1)[-1]
    if "<information>" in actions:
        raise ProtocolError("Only the search service may produce <information>.")
    searches = list(
        re.finditer(
            r"<search>\s*<company>([^<>]+)</company>\s*<year>([^<>]+)</year>\s*<query>([^<>]+)</query>\s*</search>",
            actions,
            re.DOTALL,
        )
    )
    answers = list(re.finditer(r"<answer>([^<>]*)</answer>", actions, re.DOTALL))
    if len(searches) + len(answers) != 1:
        example = "Use <search><company>Company name</company><year>2010</year><query>Financial evidence needed</query></search>."
        if "<search>" in actions:
            # Last opening tag avoids swallowing an incidental earlier tag mention.
            body = actions.rsplit("<search>", 1)[-1]
            issues = []
            if "</search>" not in body:
                issues.append("missing </search> closing tag")
            body = body.split("</search>", 1)[0]
            for field in ("company", "year", "query"):
                opening, closing = f"<{field}>", f"</{field}>"
                if opening not in body:
                    issues.append(f"missing {opening} field")
                elif closing not in body:
                    issues.append(f"missing {closing} closing tag")
                elif body.count(opening) != 1 or body.count(closing) != 1:
                    issues.append(f"duplicate {opening} field; provide it exactly once")
                elif not body.split(opening, 1)[1].split(closing, 1)[0].strip():
                    issues.append(f"{opening} must not be empty")
            if issues:
                raise ProtocolError(
                    "Invalid search: " + "; ".join(issues) + ". " + example
                )
            raise ProtocolError(
                "Invalid search: fields must appear exactly once in company, year, query order with no extra text or nested tags. Produce one action per turn. "
                + example
            )
        raise ProtocolError(
            "Produce exactly one complete <search> or <answer> action. " + example
        )
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
    if not company or len(company) > 200:
        raise ProtocolError(
            "<company> must contain 1–200 characters after trimming whitespace."
        )
    if not query or len(query) > 2000:
        raise ProtocolError(
            "<query> must contain 1–2000 characters after trimming whitespace."
        )
    if not 1900 <= int(year) <= 2100:
        raise ProtocolError("<year> must be a report year between 1900 and 2100.")
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
        if result.get("message"):
            metadata = {
                key: result[key]
                for key in (
                    "match_method",
                    "message",
                    "requested_company",
                    "requested_year",
                    "applied_filters",
                    "ambiguous",
                    "candidates",
                )
                if key in result
            }
            body = json.dumps(metadata, ensure_ascii=False) + "\n\n" + body
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
