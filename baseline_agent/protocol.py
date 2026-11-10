"""Parse only the newly generated action, never instructions inside evidence."""

import json
import re
from html import escape, unescape

ACTION_STOP_SEQUENCES = ("</search>", "</answer>")
ACTION_MARKUP = re.compile(r"</?(?:search|answer)>")


class ProtocolError(ValueError):
    """A generated action failed the model-facing search/answer syntax contract."""

    pass  # noqa: PIE790 -- keep the original body in this documentation-only edit.


def action_complete(segment):
    """Return whether the generated prefix contains one protocol-valid action."""
    if "</reasoning>" not in segment:
        return False
    try:
        parse_action(segment)
    except ProtocolError:
        return False
    return True


def parse_action(segment):
    """Parse one flat search or answer action after the last reasoning close tag. Returns
    ("search", {"query": text}) or ("answer", text). Rejects missing closes, multiple
    actions, child tags, empty content and invalid search text with ProtocolError.
    """
    if "<reasoning>" in segment and "</reasoning>" not in segment:
        raise ProtocolError(
            "Close </reasoning> before emitting one <search>query text</search> or <answer>short answer</answer> action."
        )
    if "</reasoning>" in segment:
        reasoning, actions = segment.split("</reasoning>", 1)
        if ACTION_MARKUP.search(reasoning):
            raise ProtocolError(
                "Search/answer tags are reserved for actions and must not appear inside reasoning."
            )
        actions = actions.strip()
    else:
        actions = segment.strip()
    match = re.fullmatch(r"<(search|answer)>([^<>]*)</\1>", actions, re.DOTALL)
    if not match:
        raise ProtocolError(
            "Emit exactly one complete action after </reasoning>: <search>natural-language query</search> or <answer>short answer</answer>. Search is flat text; no company, year, query child tags or extra text."
        )
    kind, value = match.group(1), unescape(match.group(2)).strip()
    if not value:
        raise ProtocolError(f"<{kind}> must not be empty.")
    if kind == "search":
        if len(value) > 2000:
            raise ProtocolError("Search query must contain 1Ã¢â‚¬â€œ2000 characters.")
        if re.search(
            r"\b(none needed|no search needed|no further search)\b",
            value,
            re.IGNORECASE,
        ):
            raise ProtocolError(
                "Search requests evidence. If sufficient, emit <answer> instead."
            )
        return kind, {"query": value}
    return kind, value


def format_information(result):
    """Render a search response inside escaped information tags for the next prompt.
    Preserves document identifiers and readable evidence while escaping embedded markup;
    unknown response statuses raise RuntimeError.
    """
    if result.get("status") == "error":
        body = json.dumps(result, ensure_ascii=False)
    elif result.get("status") == "ok":
        docs = []
        for doc in result["results"]:
            docs.append(
                (
                    f"[{doc['context_id']}] document={doc['document_id']} | {doc['title']}"
                    if "context_id" in doc
                    else f"[{doc['document_id']}] {doc['title']}"
                )
                + (f" | source={doc['source']}" if doc.get("source") else "")
                + (
                    f" | published={doc['published_at']}"
                    if doc.get("published_at")
                    else ""
                )
                + f"\n{doc['text']}"
            )
        body = (
            result.get("message", "")
            + "\n\n"
            + (
                "\n\n".join(docs)
                if docs
                else result.get(
                    "empty_message", "No matching passages. Reformulate the query."
                )
            )
        )
    else:
        raise RuntimeError("Unrecognized search-service response")
    return "<information>\n" + escape(body, quote=False) + "\n</information>"


def select_history(events, context_mode):
    """Select prompt-visible events for append, replace or legacy bounded context mode.
    Append retains every event; other modes retain only the latest information block
    while preserving model/feedback events.
    """
    if context_mode not in ("append", "replace", "bounded"):
        raise ValueError("context_mode must be append, replace or bounded")
    latest = max(
        (i for i, x in enumerate(events) if x["kind"] == "information"), default=-1
    )
    return [
        x
        for i, x in enumerate(events)
        if context_mode == "append" or x["kind"] != "information" or i == latest
    ]


def render_history(prefix, events, context_mode):
    """Join the selected event text onto an initial prompt prefix. This text renderer is
    for legacy/history views; exact append-mode sampling uses TokenConversation IDs
    instead of re-encoding this string.
    """
    return prefix + "\n\n".join(x["text"] for x in select_history(events, context_mode))
