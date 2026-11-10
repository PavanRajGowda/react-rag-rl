"""Immutable, supported Elasticsearch snapshots stored on Modal Volumes.

Active Lucene files and the writable snapshot repository use local disk. Only a
completed repository archive and its readiness marker are published to a Volume.
"""

import json
import os
import tarfile
import time
from pathlib import Path

from retrieval_service.dataset import file_sha256
from retrieval_service.search import index_name

REPOSITORY = "corpus"
SNAPSHOT = "corpus"


def check_marker(directory, manifest):
    """Load ready.json and verify its corpus hash, index name, document count and
    Elasticsearch version against manifest. Returns the marker or raises ValueError; a
    marker is required before serving a snapshot.
    """
    directory = Path(directory)
    marker = json.loads((directory / "ready.json").read_text(encoding="utf-8"))
    if (
        marker["corpus_sha256"] != manifest["sha256"]
        or marker["index"] != index_name(manifest)
        or marker["documents"] != manifest["unique_contexts"]
        or marker["elasticsearch_version"] != "8.19.3"
    ):
        raise ValueError("Persisted snapshot identity disagrees with corpus/runtime")
    return marker


def restore_index(es, runtime, directory, manifest):
    """Verify and extract a completed snapshot archive, then restore its index into a local
    Elasticsearch node. es is the live client; runtime is writable local disk; directory
    is persisted archive storage. Rejects unsafe member paths/checksum mismatches/failed
    shards and returns the verified marker. Does not rebuild from raw corpus text.
    """
    directory, runtime = Path(directory), Path(runtime)
    marker = check_marker(directory, manifest)
    archive_path = directory / "repository.tar"
    if file_sha256(archive_path) != marker["archive_sha256"]:
        raise ValueError("Persisted Elasticsearch snapshot checksum mismatch")
    started = time.perf_counter()
    with tarfile.open(archive_path, "r:") as archive:
        for member in archive.getmembers():
            if member.name != "repository" and not member.name.startswith("repository/"):
                raise ValueError("Unexpected snapshot archive entry")
        archive.extractall(runtime, filter="data")
    repository = runtime / "repository"
    # The safe tar extraction filter intentionally drops stored uid/gid values.
    for path in [repository, *repository.rglob("*")]:
        os.chown(path, 1000, 1000)
    es.snapshot.create_repository(
        name=REPOSITORY,
        repository={
            "type": "fs",
            "settings": {"location": str(repository), "readonly": True},
        },
    )
    response = es.options(request_timeout=900).snapshot.restore(
        repository=REPOSITORY,
        snapshot=SNAPSHOT,
        body={
            "indices": index_name(manifest),
            "include_global_state": False,
            "include_aliases": False,
        },
        wait_for_completion=True,
    )
    if response["snapshot"]["shards"]["failed"]:
        raise RuntimeError("Snapshot restore reported failed shards")
    print(
        f"Restored existing index in {time.perf_counter() - started:.1f}s; no corpus indexing",
        flush=True,
    )
    return marker


def create_snapshot(es, runtime, manifest):
    """Ask a live Elasticsearch node to create a completed filesystem snapshot on local
    runtime disk. Returns the repository Path only after the API reports SUCCESS; failed
    snapshots raise RuntimeError.
    """
    repository = Path(runtime) / "repository"
    repository.mkdir(exist_ok=True)
    os.chown(repository, 1000, 1000)
    es.snapshot.create_repository(
        name=REPOSITORY,
        repository={"type": "fs", "settings": {"location": str(repository)}},
    )
    response = es.options(request_timeout=1800).snapshot.create(
        repository=REPOSITORY,
        snapshot=SNAPSHOT,
        body={
            "indices": index_name(manifest),
            "include_global_state": False,
            "metadata": {"corpus_sha256": manifest["sha256"]},
        },
        wait_for_completion=True,
    )
    if response["snapshot"]["state"] != "SUCCESS":
        raise RuntimeError("Elasticsearch snapshot was not successful")
    return repository


def publish_repository(repository, directory, manifest):
    """Archive a completed snapshot repository and publish its readiness marker last.
    directory is the destination, typically a mounted Modal Volume; completed targets
    are immutable. Returns marker metadata including archive checksum/size. The caller
    owns committing the volume.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "ready.json").exists():
        raise FileExistsError("Completed index snapshot is immutable")
    partial = directory / "repository.tar.partial"
    with tarfile.open(partial, "w:") as archive:
        archive.add(repository, arcname="repository")
    digest = file_sha256(partial)
    partial.replace(directory / "repository.tar")
    marker = {
        "index": index_name(manifest),
        "corpus_sha256": manifest["sha256"],
        "documents": manifest["unique_contexts"],
        "archive_sha256": digest,
        "archive_bytes": (directory / "repository.tar").stat().st_size,
        "elasticsearch_version": "8.19.3",
        "method": "elasticsearch-snapshot-api",
    }
    # Publish readiness last. A partial archive without this verified marker
    # must never be mistaken for a serving-ready Elasticsearch index.
    temporary = directory / "ready.json.partial"
    temporary.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    temporary.replace(directory / "ready.json")
    return marker
