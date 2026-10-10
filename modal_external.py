"""Alternative HTTP deployment with externally hosted Elasticsearch."""

from pathlib import Path

import modal

root = Path(__file__).parent
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(root / "requirements.lock.txt"))
    .env({"FINQA_DATA_DIR": "/corpus"})
    .add_local_python_source("finance_rag")
    .add_local_file(root / "data/finqa/contexts.jsonl", "/corpus/contexts.jsonl")
    .add_local_file(root / "data/finqa/manifest.json", "/corpus/manifest.json")
)
app = modal.App("finance-rag-search-http")


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("finance-search")],
    cpu=1,
    memory=1024,
    min_containers=1,
    max_containers=4,
)
@modal.concurrent(max_inputs=32)
@modal.asgi_app()
def search_api():
    from finance_rag.api import create_app

    return create_app()
