"""Local read-only live training dashboard and optional Modal Volume mirror."""

import argparse
import csv
import io
import json
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from grpo_training.metrics import distribution, merge_rates, pair
from grpo_training.observability import atomic_json


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


class ModalMirror:
    """One background reader; live GPU writes are explicitly committed by observer."""

    def __init__(self, run_id, root, *, volume=None):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
            raise ValueError("Invalid run ID")
        if volume is None:
            import modal

            volume = modal.Volume.from_name("finance-rag-grpo-runs")
        self.run_id, self.root, self.volume = run_id, Path(root), volume
        self.status, self.lock = {}, threading.Lock()

    def download(self, name, *, mutable=False):
        if not re.fullmatch(
            r"(?:index|live)\.json|(?:steps|reference)/[0-9]{8}\.json|(?:rollouts|traces)/[0-9a-f]{24}\.json",
            name,
        ):
            raise ValueError("Invalid mirror path")
        path = self.root / name
        if mutable or not path.exists():
            data = b"".join(
                self.volume.read_file(f"{self.run_id}/observability/{name}")
            )
            atomic_json(path, json.loads(data))
        return read(path)

    def sync(self):
        with self.lock:
            try:
                index = self.download("index.json", mutable=True)
                live = self.download("live.json", mutable=True)
                for row in index["steps"]:
                    if row["step"] <= index["committed_step"]:
                        self.download(f"steps/{row['step']:08d}.json")
                        self.download(f"rollouts/{row['rollout_id']}.json")
                for step in index["references"]:
                    if step <= index["committed_step"]:
                        self.download(f"reference/{step:08d}.json")
                if live.get("rollout_id"):
                    self.download(f"rollouts/{live['rollout_id']}.json")
                # Event files are append-only; re-fetch only files whose size changed.
                for entry in self.volume.iterdir(
                    f"{self.run_id}/tensorboard", recursive=False
                ):
                    filename = entry.path.rsplit("/", 1)[-1]
                    if not re.fullmatch(
                        r"events\.out\.tfevents\.[A-Za-z0-9._-]+", filename
                    ):
                        continue
                    path = self.root.parent / "tensorboard" / filename
                    if not path.exists() or path.stat().st_size != entry.size:
                        data = b"".join(self.volume.read_file(entry.path))
                        path.parent.mkdir(parents=True, exist_ok=True)
                        temporary = path.with_suffix(".tmp")
                        temporary.write_bytes(data)
                        temporary.replace(path)
                self.status = {"last_sync_unix": time.time(), "error": None}
            except Exception as exc:  # noqa: BLE001 -- surface errors to the live UI, retain last good snapshot.
                self.status = self.status | {"error": f"{type(exc).__name__}: {exc}"}

    def trace(self, key):
        with self.lock:
            return self.download(f"traces/{key}.json")


class Reader:
    def __init__(self, root):
        self.root, self.cache = Path(root), {}

    def file(self, name):
        path = self.root / name
        if not path.exists():
            return None
        stamp = path.stat().st_mtime_ns
        if name not in self.cache or self.cache[name][0] != stamp:
            self.cache[name] = stamp, read(path)
        return self.cache[name][1]

    def snapshot(self, window=20, episode_limit=500):
        index = self.file("index.json") or {
            "steps": [],
            "references": [],
            "committed_step": 0,
        }
        steps = [
            self.file(f"steps/{r['step']:08d}.json")
            for r in sorted(index["steps"], key=lambda r: r["step"])
            if r["step"] <= index["committed_step"]
        ]
        steps = [s for s in steps if s]
        selected = steps[-window:] if window else steps
        ids = list(dict.fromkeys(s["rollout_id"] for s in selected))
        rollouts = [self.file(f"rollouts/{key}.json") for key in ids]
        rollouts = [r for r in rollouts if r]
        rates = {
            name: merge_rates(
                [r["rates"][name] for r in rollouts if name in r["rates"]]
            )
            for name in {k for r in rollouts for k in r["rates"]}
        }
        tokens = sum(s["tokens"] for s in selected)
        reference = [
            self.file(f"reference/{step:08d}.json")
            for step in index["references"]
            if step <= index["committed_step"]
        ]
        reference = [r for r in reference if r]
        series = []
        cumulative_loss, cumulative_tokens = [0.0], [0]
        for i, step in enumerate(steps):
            cumulative_loss.append(cumulative_loss[-1] + step["loss"] * step["tokens"])
            cumulative_tokens.append(cumulative_tokens[-1] + step["tokens"])
            first = max(0, i - window + 1) if window else 0
            count = cumulative_tokens[i + 1] - cumulative_tokens[first]
            series.append(
                {
                    "step": step["step"],
                    "loss": step["loss"],
                    "gradient_norm": step["gradient_norm"],
                    "rolling_loss": (cumulative_loss[i + 1] - cumulative_loss[first])
                    / count
                    if count
                    else None,
                }
            )
        policy = [s["policy_metrics"] for s in selected if s.get("policy_metrics")]
        episodes = [e for r in rollouts for e in r["episodes"]]
        hits = [e for e in episodes if e.get("first_hit_rank")]
        return {
            "live": self.file("live.json"),
            "committed_step": index["committed_step"],
            "steps": selected,
            "series": series[-500:],
            "reference": reference,
            "rates": rates,
            "rollouts": len(rollouts),
            "episodes": episodes[-episode_limit:] if episode_limit else episodes,
            "episode_count": len(episodes),
            "first_hit": {
                "searches": distribution(e["first_hit_search"] for e in hits),
                "generations": distribution(
                    e["first_hit_turn"] for e in hits if e["first_hit_turn"] is not None
                ),
                "reciprocal_rank": pair(
                    sum(1 / e["first_hit_rank"] for e in hits), len(hits)
                ),
                "first_search_hit": pair(
                    sum(e["first_hit_search"] == 1 for e in episodes), len(episodes)
                ),
            },
            "lengths": {
                k: distribution(e[k] for e in episodes if e.get(k) is not None)
                for k in (
                    "generations",
                    "searches",
                    "input_tokens",
                    "output_tokens",
                    "peak_input_tokens",
                    "seconds",
                    "search_seconds",
                )
            },
            "groups": [g for r in rollouts for g in r["groups"]],
            "weighted_loss": pair(
                sum(s["loss"] * s["tokens"] for s in selected), tokens
            ),
            "policy_rates": {
                name: merge_rates([p[name] for p in policy])
                for name in (
                    "ratio_outside_bounds",
                    "objective_clipped_learning_tokens",
                    "objective_clipped_all_tokens",
                    "old_to_current_kl_sample_estimate",
                )
            },
            "sampling": [r["sampling"] for r in rollouts],
            "rollout_details": [
                {k: v for k, v in r.items() if k not in ("episodes", "groups")}
                for r in rollouts
            ],
            "definitions": {
                "window": "optimizer steps; fresh rollouts deduplicated by rollout_id",
                "loss": "pre-update; token-weighted DAPO loss",
                "reference_kl": "post-update; fixed initial-generation train prefixes, full vocabulary, nats",
                "reference_warning": "A probe diagnostic, not trajectory-wide KL or a penalty; beta remains zero",
                "ratios": "current/old: optimizer drift; old/vLLM: backend mismatch; distinct from fixed-reference KL",
                "timings": "generation times are batch wall times; episode time includes batching/tool queue waits; unavailable stays null",
            },
        }


def create_app(root, *, mirror=None):
    reader = Reader(root)
    stop = threading.Event()

    @asynccontextmanager
    async def lifespan(app):
        def refresh():
            while not stop.is_set():
                mirror.sync()
                stop.wait(5)

        thread = threading.Thread(target=refresh, daemon=True) if mirror else None
        if thread:
            thread.start()
        yield
        stop.set()
        if thread:
            thread.join(timeout=1)

    app = FastAPI(title="FinanceRAG training", lifespan=lifespan)

    @app.get("/")
    def home():
        return FileResponse(Path(__file__).with_suffix(".html"))

    @app.get("/api/snapshot")
    def snapshot(window: int = 20):
        if not 0 <= window <= 10000:
            raise HTTPException(400, "Invalid window")
        return reader.snapshot(window) | {"mirror": mirror.status if mirror else None}

    @app.get("/api/traces/{key}")
    def trace(key: str):
        if not re.fullmatch(r"[0-9a-f]{24}", key):
            raise HTTPException(400, "Invalid trace key")
        result = mirror.trace(key) if mirror else reader.file(f"traces/{key}.json")
        if result is None:
            raise HTTPException(404, "Trace not available")
        return result

    @app.get("/api/export/{kind}")
    def export(kind: str):
        if kind not in ("steps", "episodes", "groups"):
            raise HTTPException(400, "Unknown export")
        rows = reader.snapshot(0, episode_limit=None)[kind]
        fields = sorted({key for row in rows for key in row})

        def generate():
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=fields)
            writer.writeheader()
            yield buffer.getvalue()
            for row in rows:
                buffer.seek(0)
                buffer.truncate(0)
                writer.writerow(
                    {
                        k: json.dumps(v) if isinstance(v, (dict, list)) else v
                        for k, v in row.items()
                    }
                )
                yield buffer.getvalue()

        return StreamingResponse(
            generate(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{kind}.csv"'},
        )

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Local run directory or mirror destination",
    )
    parser.add_argument(
        "--modal-run-id", help="Read-only live mirror from finance-rag-grpo-runs"
    )
    parser.add_argument("--port", type=int, default=8502)
    args = parser.parse_args()
    root = args.run_dir.resolve() / "observability"
    mirror = ModalMirror(args.modal_run_id, root) if args.modal_run_id else None
    import uvicorn

    uvicorn.run(create_app(root, mirror=mirror), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
