import asyncio
import hashlib
import json
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from elasticsearch import AsyncElasticsearch
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from finance_rag.retrieval import retrieve
from finance_rag.search import (
    SEARCH_SCHEMA_VERSION,
    SearchError,
    SearchRequest,
    index_name,
)


def create_app(directory=None, es_client=None):
    directory = Path(directory or os.environ.get("FINQA_DATA_DIR", "data/finqa"))

    @asynccontextmanager
    async def lifespan(app):
        token = os.environ.get("SEARCH_API_TOKEN")
        if not token:
            raise RuntimeError("SEARCH_API_TOKEN must be configured")
        app.state.token = token
        app.state.manifest = json.loads((directory / "manifest.json").read_text())
        app.state.index = index_name(app.state.manifest)
        digest = hashlib.sha256((directory / "contexts.jsonl").read_bytes()).hexdigest()
        if digest != app.state.manifest["sha256"]:
            raise RuntimeError("Local contexts do not match corpus manifest")
        app.state.es = es_client or AsyncElasticsearch(
            os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200"),
            api_key=os.environ.get("ELASTIC_API_KEY"),
            request_timeout=10,
            max_retries=1,
            retry_on_timeout=True,
        )
        app.state.limit = asyncio.Semaphore(32)
        try:
            mapping = await app.state.es.indices.get_mapping(index=app.state.index)
            if (
                mapping[app.state.index]["mappings"]["_meta"]["sha256"]
                != app.state.manifest["sha256"]
            ):
                raise RuntimeError("Remote index does not match local corpus manifest")
            if (
                mapping[app.state.index]["mappings"]["_meta"].get(
                    "search_schema_version"
                )
                != SEARCH_SCHEMA_VERSION
            ):
                raise RuntimeError("Remote index has an incompatible search schema")
            count = await app.state.es.count(index=app.state.index)
            if count["count"] != app.state.manifest["unique_contexts"]:
                raise RuntimeError("Remote index is incomplete")
            yield
        finally:
            if es_client is None:
                await app.state.es.close()

    app = FastAPI(title="FinQA Search", version="1.0", lifespan=lifespan)
    bearer = HTTPBearer(auto_error=False)

    def auth(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ):
        if credentials is None or not secrets.compare_digest(
            credentials.credentials.encode("utf-8"), app.state.token.encode("utf-8")
        ):
            raise HTTPException(401, detail={"code": "unauthorized"})

    @app.get("/health", dependencies=[Depends(auth)])
    async def health():
        if not await app.state.es.ping():
            raise HTTPException(503, detail={"code": "search_unavailable"})
        return {"status": "ok", "corpus_version": app.state.manifest["corpus_version"]}

    @app.post("/v1/search", dependencies=[Depends(auth)])
    async def search(request: SearchRequest):
        async def execute(**body):
            async with app.state.limit:
                return await app.state.es.search(index=app.state.index, **body)

        try:
            return await retrieve(
                request, execute, app.state.manifest["corpus_version"]
            )
        except SearchError as exc:
            raise HTTPException(exc.status, detail=exc.as_dict()) from exc
        except Exception as exc:
            raise HTTPException(503, detail={"code": "search_unavailable"}) from exc

    return app


app = create_app()
