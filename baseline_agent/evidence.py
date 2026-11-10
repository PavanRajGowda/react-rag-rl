"""Append-only evidence memory plus legacy bounded-history support."""

import hashlib
import json
from copy import deepcopy

from baseline_agent.protocol import format_information


def evidence_id(document):
    """Canonical evidence key; accept chunk IDs when reading historical records."""
    return document.get("context_id") or document["document_id"]


class EvidenceStore:
    """Ordered evidence collection keyed by stable document identity. Tracks content hashes
    to prevent silent evidence replacement; append and legacy bounded modes have
    separate retention behavior.
    """
    def __init__(self, max_passages, max_chars):
        """Initialize an ordered deduplicated evidence store with legacy count/character
        budgets. Append mode intentionally does not evict earlier documents using these
        budgets.
        """
        self.max_passages, self.max_chars = max_passages, max_chars
        self._documents = {}
        self._fingerprints = {}

    @staticmethod
    def fingerprint(document):
        """Hash identity and content fields to detect changed text for a previously seen
        evidence ID.
        """
        content = {
            k: document.get(k)
            for k in (
                "context_id",
                "document_id",
                "title",
                "source",
                "published_at",
                "text",
            )
        }
        return hashlib.sha256(
            json.dumps(content, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()

    def visible(self):
        """Return copied document identifiers and content hashes for the current retained
        evidence collection. These audit entries describe visibility without duplicating
        full document text.
        """
        return [
            {
                "document_id": doc["document_id"],
                **({"context_id": key} if "context_id" in doc else {}),
                "content_sha256": self._fingerprints[key],
            }
            for key, doc in self._documents.items()
        ]

    def append_response(self, response):
        """Append only new passages; never alter or evict earlier observations."""
        new_documents = []
        for doc in response.get("results", []):
            key, digest = evidence_id(doc), self.fingerprint(doc)
            # Stable identity must imply stable content; silently replacing text
            # would make the recorded evidence-visible-to-generation audit false.
            if key in self._fingerprints and self._fingerprints[key] != digest:
                raise ValueError(
                    "Passage content changed for an existing evidence identifier"
                )
            self._fingerprints[key] = digest
            if key not in self._documents:
                self._documents[key] = deepcopy(doc)
                new_documents.append(deepcopy(doc))
        rendered = deepcopy(response)
        if response.get("status") == "ok":
            rendered["results"] = new_documents
            if not new_documents:
                rendered["message"] = (
                    "This search supplied no new passages. Earlier evidence remains in the conversation."
                )
                rendered["empty_message"] = (
                    "Refine the query if more evidence is needed."
                )
        return {
            "kind": "information",
            "text": format_information(rendered),
            "visible_evidence": [
                item
                for item in self.visible()
                if evidence_id(item) in {evidence_id(doc) for doc in new_documents}
            ],
            "evicted_document_ids": [],
        }

    def update(self, response):
        """Merge returned evidence and evict oldest whole documents to satisfy legacy
        bounded-mode budgets. Returns an information event with current visible IDs and
        explicit evictions; append mode uses append_response instead.
        """
        legacy = any("context_id" in doc for doc in self._documents.values()) or any(
            "context_id" in doc for doc in response.get("results", [])
        )
        evicted = []
        for doc in response.get("results", []):
            key, digest = evidence_id(doc), self.fingerprint(doc)
            # Stable identity must imply stable content; silently replacing text
            # would make the recorded evidence-visible-to-generation audit false.
            if key in self._fingerprints and self._fingerprints[key] != digest:
                raise ValueError(
                    "Passage content changed for an existing evidence identifier"
                )
            self._fingerprints[key] = digest
            if key not in self._documents:
                self._documents[key] = deepcopy(doc)

        # Bound rendered evidence, including headers/metadata and escaped text.
        def rendered():
            """Render the current evidence collection so bounded-mode limits include
            headers and escaped text, not only raw document characters.
            """
            return format_information(
                {"status": "ok", "results": list(self._documents.values())}
            )

        while self._documents and (
            len(self._documents) > self.max_passages or len(rendered()) > self.max_chars
        ):
            key = next(iter(self._documents))
            evicted.append(key)
            del self._documents[key]
        message = "Current deduplicated evidence collection."
        if response.get("status") == "error":
            message += " Search feedback: " + response.get("message", "")
        elif not response.get("results"):
            message += " Latest search found no matching passages; refine the query."
        if evicted:
            message += " Whole passages evicted by the evidence budget: " + ", ".join(
                evicted
            )
        return {
            "kind": "information",
            "text": format_information(
                {
                    "status": "ok",
                    "message": message,
                    "empty_message": "The current evidence collection is empty.",
                    "results": list(self._documents.values()),
                }
            ),
            "visible_evidence": self.visible(),
            ("evicted_context_ids" if legacy else "evicted_document_ids"): evicted,
        }


def visible_evidence(events, context_mode):
    """Return the evidence inventory actually selected for the next prompt."""
    from baseline_agent.protocol import select_history

    documents = {}
    for event in select_history(events, context_mode):
        for item in event.get("visible_evidence", []):
            documents[evidence_id(item)] = item
    return deepcopy(list(documents.values()))


def information_event(response, store, context_mode):
    """Convert a search response into the context-mode-specific information event. Append
    mode emits only new documents; bounded mode updates the retained collection; legacy
    replacement mode renders the current response.
    """
    if context_mode == "append":
        return store.append_response(response)
    if context_mode == "bounded":
        return store.update(response)
    return {
        "kind": "information",
        "text": format_information(response),
        "visible_evidence": [
            {
                "document_id": d["document_id"],
                **({"context_id": d["context_id"]} if "context_id" in d else {}),
                "content_sha256": store.fingerprint(d),
            }
            for d in response.get("results", [])
        ],
    }
