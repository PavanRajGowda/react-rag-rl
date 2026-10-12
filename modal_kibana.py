"""Optional, password-protected FinQA Kibana dashboard on Modal."""

import json
from pathlib import Path

import modal

runtime_options = {
    "image": modal.Image.debian_slim(python_version="3.11"),
    "secrets": [modal.Secret.from_dict({})],
}
if modal.is_local():
    root = Path(__file__).parent
    credentials = root / "data/kibana-credentials.json"
    if not credentials.exists():
        raise RuntimeError("Run uv run python -m finance_rag.kibana_setup first")
    image = (
        modal.Image.from_dockerfile(
            root / "docker/Kibana.Dockerfile",
            context_dir=root / "docker",
            add_python="3.11",
        )
        .pip_install_from_requirements(str(root / "requirements.lock.txt"))
        .env({"ES_REPLICAS": "0"})
        .add_local_python_source("finance_rag")
        .add_local_file(root / "data/finqa/contexts.jsonl", "/corpus/contexts.jsonl")
        .add_local_file(root / "data/finqa/manifest.json", "/corpus/manifest.json")
    )
    runtime_options = {
        "image": image,
        "secrets": [modal.Secret.from_dict(json.loads(credentials.read_text()))],
    }
app = modal.App("finance-rag-kibana")


@app.cls(
    **runtime_options,
    cpu=2,
    memory=6144,
    min_containers=0,
    max_containers=1,
    scaledown_window=60,
    startup_timeout=600,
    timeout=600,
)
@modal.concurrent(max_inputs=32)
class Explorer:
    @modal.enter()
    def start(self):
        from finance_rag.kibana import KibanaStack

        self.stack = KibanaStack(Path("/corpus"))
        self.stack.start()

    @modal.web_server(5601, startup_timeout=600)
    def dashboard(self):
        pass

    @modal.exit()
    def stop(self):
        if hasattr(self, "stack"):
            self.stack.stop()
