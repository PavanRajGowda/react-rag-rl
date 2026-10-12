"""Optional local TensorBoard scalar and text events; no Torch required."""

import json
import time
from pathlib import Path


def write_dashboard(rows, directory, start_step=0):
    from tensorboard.compat.proto import event_pb2, summary_pb2, tensor_pb2, types_pb2
    from tensorboard.plugins.text.metadata import create_summary_metadata
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    writer = EventFileWriter(str(directory))
    try:
        for index, row in enumerate(rows, start_step):
            values = [
                summary_pb2.Summary.Value(tag=tag, simple_value=float(row[field]))
                for tag, field in [
                    ("latency/inference_seconds", "inference_seconds"),
                    ("latency/generation_seconds", "generation_seconds"),
                    ("latency/search_seconds", "search_seconds"),
                    ("usage/search_attempts", "search_attempts"),
                    ("usage/generated_tokens", "generated_tokens"),
                ]
            ]
            if row["score"] is not None:
                values.append(
                    summary_pb2.Summary.Value(
                        tag="evaluation/answer_match", simple_value=row["score"]
                    )
                )
            text = f"### {row['id']}\n\n{row['question']}\n\nGold: {row['correct_answer']}  \nPrediction: {row['our_answer']}  \nStatus: {row['status']}  \nScore: {row['score']}  \nInference: {row['inference_seconds']:.2f}s\n\nTrace: {row['trace_file']}"
            trace = json.loads(Path(row["trace_file"]).read_text(encoding="utf-8"))
            for turn in trace["turns"]:
                text += f"\n\n#### Generation {turn.get('turn', '?')}\n\n```text\n{turn.get('model_output', turn.get('error', ''))}\n```"
            for search in trace["searches"]:
                text += (
                    "\n\n#### Search\n\n```json\n"
                    + json.dumps(search.get("request", {}), indent=2)
                    + "\n```"
                )
            values.append(
                summary_pb2.Summary.Value(
                    tag="examples/result",
                    tensor=tensor_pb2.TensorProto(
                        dtype=types_pb2.DT_STRING, string_val=[text.encode()]
                    ),
                    metadata=create_summary_metadata(
                        display_name="Question results",
                        description="Gold, prediction and timing for each question",
                    ),
                )
            )
            writer.add_event(
                event_pb2.Event(
                    wall_time=time.time(),
                    step=index,
                    summary=summary_pb2.Summary(value=values),
                )
            )
        writer.flush()
    finally:
        writer.close()
