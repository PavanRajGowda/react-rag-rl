"""Validate retrieval responses without implementing an agent controller."""

from copy import deepcopy

from baseline_agent.protocol import format_information

DOMAIN_ERRORS = frozenset({"invalid_request"})


class SearchContractError(RuntimeError):
    """Malformed/mismatched backend output must not become model feedback."""


def checked_response(response, config):
    """Validate a search response against the episode retrieval contract.

    Returns (copied_response, rendered_information). Checks status, corpus/schema
    identity, result count, document fields and renderability. SearchContractError
    marks malformed/mismatched output as a technical failure for retry handling.
    """
    if not isinstance(response, dict):
        raise SearchContractError("Search response must be an object")
    if response.get("status") == "error":
        if response.get("code") not in DOMAIN_ERRORS or not isinstance(
            response.get("message"), str
        ):
            raise SearchContractError("Unknown or malformed search error")
    elif response.get("status") == "ok":
        if (
            response.get("corpus_version") != config.corpus_version
            or type(response.get("search_schema_version")) is not int
            or response.get("search_schema_version") != config.search_schema_version
        ):
            raise SearchContractError("Search corpus/schema disagrees with experiment")
        docs = response.get("results")
        if not isinstance(docs, list) or len(docs) > config.top_k:
            raise SearchContractError(
                "Search results must be a list within requested K"
            )
        for doc in docs:
            if not isinstance(doc, dict) or any(
                not isinstance(doc.get(k), str) or not doc[k].strip()
                for k in (
                    "document_id",
                    "title",
                    "text",
                )
            ):
                raise SearchContractError("Search document lacks required text fields")
    else:
        raise SearchContractError("Unrecognized search response status")
    # Validate rendering before recording successful evidence or feeding the model.
    observation = format_information(response)
    return deepcopy(response), observation
