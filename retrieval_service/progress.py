"""Monitor an existing HotpotQA conversion without restarting it.

Counts complete JSONL records incrementally; does not parse/load the full corpus.
The first scan establishes the existing count. Subsequent rate measurements
reflect newly written records, not the speed of that initial scan.
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from retrieval_service.dataset import DATA_DIR


class ConversionMonitor:
    """Incremental observer of local corpus conversion output. Tracks byte position and
    complete JSONL records without loading the full corpus or restarting preparation.
    """
    def __init__(self, directory):
        """Initialize a read-only conversion monitor with a zero count and byte cursor.
        Does not start or restart the conversion writer.
        """
        self.directory = Path(directory)
        self.count = 0
        self.offset = 0
        self.path = None

    def snapshot(self):
        """Count newly observed JSONL newline records and return conversion phase/progress
        metadata. Resets its cursor if the observed file changes or shrinks; a completed
        manifest supplies the final total. This measures local conversion, not
        Elasticsearch indexing or a reliable writer health signal.
        """
        final = self.directory / "contexts.jsonl"
        path = self.directory / "contexts.jsonl.partial"
        if not path.exists() and final.exists():
            path = final
        if path.exists():
            if path != self.path or path.stat().st_size < self.offset:
                self.count = self.offset = 0
            self.path = path
            with path.open("rb") as stream:
                stream.seek(self.offset)
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    self.count += block.count(b"\n")
                    self.offset += len(block)
        manifest_path = self.directory / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else None
        )
        phase = (
            "completed"
            if manifest
            else "finalizing_labels_and_checksums"
            if final.exists()
            else "converting_abstracts"
            if self.path
            else "waiting_for_output"
        )
        return {
            "phase": phase,
            "abstracts_converted": self.count,
            "total_abstracts": manifest["unique_contexts"] if manifest else None,
            "output_bytes": self.offset,
            "observed_at": datetime.now(UTC).isoformat(),
            "evidence_mapping_issues": manifest.get("evidence_issues")
            if manifest
            else None,
            "note": "Observes local conversion output, not Elasticsearch indexing. A stalled writer cannot be diagnosed from this count alone.",
        }


def save_snapshot(directory, status):
    """Atomically publish observed_progress.json only when the output directory already
    exists. Does not create a missing conversion directory or alter the corpus.
    """
    path = Path(directory) / "observed_progress.json"
    if path.parent.exists():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)


def main():
    """Display a one-shot or continuously refreshed conversion-progress report. Watch mode
    reads new output every two seconds and exits after manifest publication.
    """
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DATA_DIR)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    monitor = ConversionMonitor(args.directory)
    status = monitor.snapshot()
    save_snapshot(args.directory, status)
    if not args.watch:
        print(json.dumps(status, indent=2))
        return
    print(
        "Monitoring converted abstracts; total/percentage and ETA are unknown until conversion finishes."
    )
    with tqdm(
        initial=status["abstracts_converted"],
        total=status["total_abstracts"],
        unit="abstract",
        unit_scale=True,
        desc="Converted",
    ) as bar:
        while True:
            if status["abstracts_converted"] < bar.n:
                bar.reset()
            bar.update(status["abstracts_converted"] - bar.n)
            bar.set_postfix(phase=status["phase"])
            if status["phase"] == "completed":
                bar.total = status["total_abstracts"]
                bar.refresh()
                break
            time.sleep(2)
            status = monitor.snapshot()
            save_snapshot(args.directory, status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
