"""Prepare official HotpotQA fullwiki questions and the verified Wikipedia abstracts.

Stream nested bz2 shards without extracting them. One official introductory
paragraph is one retrieval passage; supporting sentence IDs remain in labels.
"""

import argparse
import bz2
import hashlib
import io
import json
import tarfile
import unicodedata
from collections import Counter
from pathlib import Path

from retrieval_service.dataset import (
    CONFIG,
    CORPUS_VERSION,
    DATA_DIR,
    DATASET,
    REVISION,
    file_sha256,
)
from retrieval_service.debug_subset import select_debug_subset

ARCHIVE_MD5 = "01edf64cd120ecc03a2745352779514c"
ARCHIVE_BYTES = 1553565403


def title_key(title):
    """Normalize an article title to Unicode NFC for matching official supporting-fact
    titles. Does not perform fuzzy matching or infer missing evidence.
    """
    return unicodedata.normalize("NFC", title)


def write_json(path, value):
    """Write a readable UTF-8 preparation artifact to the given Path. Parent creation and
    publication order are controlled by prepare.
    """
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def verify_archive(path):
    """Check the downloaded abstracts archive against the official byte count and MD5.
    Streams the file; raises ValueError on a wrong or incomplete download before
    conversion begins.
    """
    digest = hashlib.md5()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if Path(path).stat().st_size != ARCHIVE_BYTES or digest.hexdigest() != ARCHIVE_MD5:
        raise ValueError(
            "Archive does not match the official fullwiki abstracts checksum"
        )


def wiki_rows(path):
    """Yield Wikipedia page dictionaries from nested bz2 members of a tar.bz2 archive.
    Processes members sequentially without extracting the archive to disk; each member
    is decoded as UTF-8 JSONL.
    """
    with tarfile.open(path, "r|bz2") as archive:
        for member in archive:
            if member.isfile() and member.name.endswith(".bz2"):
                with bz2.open(
                    io.BytesIO(archive.extractfile(member).read()),
                    "rt",
                    encoding="utf-8",
                ) as stream:
                    for line in stream:
                        yield json.loads(line)


def source_questions(cache_dir=None):
    """Yield (internal_split, question_row) pairs from the pinned Hugging Face HotpotQA
    dataset. Maps validation to dev and preserves the official train/test memberships.
    May download missing source data into cache_dir.
    """
    from datasets import load_dataset

    dataset = load_dataset(
        DATASET,
        CONFIG,
        revision=REVISION,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    for source, split in (("train", "train"), ("validation", "dev"), ("test", "test")):
        for row in dataset[source]:
            yield split, row


def fact_pairs(row):
    """Return supporting (article title, sentence index) pairs from either supported source
    annotation layout. Strict parallel-array alignment detects malformed labels.
    """
    facts = row.get("supporting_facts") or {}
    if isinstance(facts, dict):
        return list(zip(facts.get("title", []), facts.get("sent_id", []), strict=True))
    return [tuple(fact) for fact in facts]


def paragraph(row):
    """Convert one official abstract row into a document-only retrieval record. Joins its
    sentence array verbatim and returns document_id, title, text and URL; creates no
    overlapping chunks or passage IDs.
    """
    sentences = row["text"]
    if not isinstance(sentences, list) or any(
        not isinstance(s, str) for s in sentences
    ):
        raise ValueError("Expected the official flat abstract sentence array")
    document_id = "wiki-" + str(row["id"])
    return {
        "document_id": document_id,
        "title": row["title"],
        "text": "".join(sentences),
        "url": row["url"],
    }


def prepare(archive, questions, output, *, verify=True):
    """Convert official abstracts/questions into a versioned corpus and fixed question
    artifacts. archive is the downloaded tar.bz2; questions yields official split/row
    pairs; output is a new directory. Streams contexts.jsonl, maps evidence titles to
    document IDs, audits sentence issues, writes labels/test/debug files and publishes
    manifest.json last. Refuses to overwrite a completed corpus; verify controls the
    initial archive checksum check.
    """
    output = Path(output)
    if (output / "manifest.json").exists():
        raise FileExistsError(
            "Prepared corpus already exists; use a new output directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    if verify:
        verify_archive(archive)
    source = list(questions)
    # Keep only reference-title metadata in memory, not all Wikipedia pages.
    # Corpus documents stream to disk independently of question-label mapping.
    needed = {
        title_key(title)
        for split, row in source
        if split != "test"
        for title, _ in fact_pairs(row)
    }
    found, count = {}, 0
    temporary = output / "contexts.jsonl.partial"
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in wiki_rows(archive):
            doc = paragraph(row)
            stream.write(
                json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            key = title_key(row["title"])
            if key in needed:
                if key in found and found[key][0]["document_id"] != doc["document_id"]:
                    raise ValueError(
                        "Ambiguous supporting article title: " + row["title"]
                    )
                found[key] = (doc, row["text"])
            count += 1
            if count % 100000 == 0:
                print(f"Prepared {count:,} abstracts", flush=True)
    # Publication of the corpus file is separate from completion of preparation.
    # manifest.json is the final readiness signal after labels/checksums are written.
    temporary.replace(output / "contexts.jsonl")
    labels, predictions, assignments, issues, seen = [], [], [], [], set()
    for index, (split, row) in enumerate(source):
        key = row.get("id", row.get("_id"))
        if not key or key in seen:
            raise ValueError("Missing/duplicate official question ID")
        seen.add(key)
        basic = {
            "id": key,
            "question": row["question"],
            "split": split,
            "question_type": row.get("type") or "unknown",
            "source_index": index,
        }
        assignments.append({"id": key, "split": split})
        if split == "test":
            predictions.append(basic)  # No fabricated gold labels for hidden test.
            continue
        if not isinstance(row.get("answer"), str) or not row["answer"].strip():
            raise ValueError("Labeled split is missing an answer")
        evidence = []
        contexts = row.get("context") or {}
        context_by_title = (
            dict(
                zip(
                    contexts.get("title", []),
                    contexts.get("sentences", []),
                    strict=True,
                )
            )
            if isinstance(contexts, dict)
            else dict(contexts)
        )
        for title, sentence_id in fact_pairs(row):
            match = found.get(title_key(title))
            passages, fact = [], None
            if match:
                doc, sentences = match
                original = context_by_title.get(title)
                expected = (
                    original[sentence_id]
                    if original and 0 <= sentence_id < len(original)
                    else None
                )
                fact = (
                    sentences[sentence_id]
                    if 0 <= sentence_id < len(sentences)
                    else None
                )
                # Sentence identity is checked against the supplied context when available.
                if fact is not None and (
                    expected is None
                    or " ".join(expected.split()) == " ".join(fact.split())
                ):
                    passages = [doc["document_id"]]
            if not passages:
                issues.append(
                    {
                        "id": key,
                        "title": title,
                        "sentence_id": sentence_id,
                        "reason": "missing_or_mismatched_support",
                    }
                )
            evidence.append(
                {
                    "document_id": match[0]["document_id"]
                    if match
                    else "missing-"
                    + hashlib.sha256(title_key(title).encode()).hexdigest()[:24],
                    "title": title,
                    "sentence_id": sentence_id,
                    "fact": fact,
                    "sentence_mapping_valid": bool(passages),
                }
            )
        labels.append(
            basic
            | {
                "answer": row["answer"],
                "evidence": evidence,
                "gold_document_ids": sorted({e["document_id"] for e in evidence}),
            }
        )
    with (output / "labels.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in labels:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "test_questions.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(
        output / "split_assignments.json",
        {"policy": "official-fullwiki-splits", "assignments": assignments},
    )
    write_json(output / "debug_subset.json", select_debug_subset(labels, seed=42))
    write_json(
        output / "partition_audit.json",
        {
            "policy": "official-fullwiki-splits; no repartitioning",
            "split_counts": dict(Counter(s for s, _ in source)),
            "shared_searchable_corpus": True,
        },
    )
    write_json(output / "evidence_audit.json", issues)
    manifest = {
        "dataset": DATASET,
        "revision": REVISION,
        "dataset_config": CONFIG,
        "corpus_version": CORPUS_VERSION,
        "sha256": file_sha256(output / "contexts.jsonl"),
        "unique_contexts": count,
        "unique_documents": count,
        "questions": len(source),
        "split_questions": dict(Counter(s for s, _ in source)),
        "labeled_splits": ["train", "dev"],
        "hidden_test_labels": True,
        "archive_md5": ARCHIVE_MD5,
        "archive_bytes": ARCHIVE_BYTES,
        "license": "CC-BY-SA-4.0",
        "chunking": {"method": "official-introductory-paragraph", "overlap": 0},
        "supporting_facts": "Title/sentence annotations retained for diagnostics; rewards use gold document IDs",
        "evidence_identity": "document_id",
        "search_schema_version": 6,
        "evidence_issues": len(issues),
    }
    for name in (
        "labels",
        "split_assignments",
        "debug_subset",
        "partition_audit",
        "test_questions",
    ):
        manifest[name + "_sha256"] = file_sha256(
            output
            / (name + (".jsonl" if name in ("labels", "test_questions") else ".json"))
        )
    write_json(output / "manifest.json", manifest)  # Publish readiness last.
    return manifest


def main():
    """Parse archive/output/cache arguments and run the one-time local HotpotQA conversion.
    Preparation may download question sources but does not deploy or index
    Elasticsearch.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--cache-dir", type=Path, help="Optional Hugging Face cache directory"
    )
    parser.add_argument("--output", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.archive, source_questions(args.cache_dir), args.output),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
