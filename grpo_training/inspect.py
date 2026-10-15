"""Offline Commit-1 inspection. Does not import a trainer, model or cloud client."""

import argparse
import json
from pathlib import Path

from grpo_training.config import TrainingConfig
from grpo_training.data import load_local_splits
from grpo_training.provenance import ROOT, build_preflight, write_preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--corpus-dir", type=Path, default=ROOT / "data/finqa")
    parser.add_argument("--train-count", type=int)
    parser.add_argument("--dev-count", type=int)
    parser.add_argument(
        "--output-root", type=Path, help="Optional local preflight record directory"
    )
    args = parser.parse_args()
    config = (
        TrainingConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
        if args.config
        else TrainingConfig()
    )
    train, dev, hashes = load_local_splits(
        args.corpus_dir, config, args.train_count, args.dev_count
    )
    record = build_preflight(config, train, dev, hashes)
    summary = {
        "kind": record["kind"],
        "experiment_id": record["experiment_id"],
        "execution_status": record["execution_status"],
        "train_questions": len(train.questions),
        "dev_questions": len(dev.questions),
        "skipped_blank": len(train.skipped_blank_ids) + len(dev.skipped_blank_ids),
        "logical_episodes_per_update": config.episodes_per_update,
        "environment_validation": record["environment_validation"],
    }
    if args.output_root:
        summary["preflight_file"] = str(
            write_preflight(record, args.output_root).resolve()
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
