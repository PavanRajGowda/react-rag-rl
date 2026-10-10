"""An isolated Elasticsearch process and index per Modal container."""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from elasticsearch import Elasticsearch
from pydantic import ValidationError

from finance_rag.index import client, index_corpus
from finance_rag.search import (
    Registry,
    SearchError,
    SearchRequest,
    load_docs,
    query_body,
)


class EmbeddedSearch:
    def __init__(self, directory, bootstrap_password=None):
        self.directory = Path(directory)
        self.bootstrap_password = bootstrap_password
        self.process = None
        self.es = None
        self.workspace = None
        self.log = None

    def start(self):
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        digest = hashlib.sha256(
            (self.directory / "contexts.jsonl").read_bytes()
        ).hexdigest()
        if digest != self.manifest["sha256"]:
            raise RuntimeError("Corpus does not match manifest")
        self.registry = Registry(load_docs(self.directory))
        self.workspace = tempfile.TemporaryDirectory(prefix="finqa-es-")
        runtime = Path(self.workspace.name)
        os.chown(runtime, 1000, 1000)
        self.log = (runtime / "startup.log").open("w")
        es_env = os.environ | {"ES_JAVA_OPTS": "-Xms1g -Xmx1g -XX:+StartAttachListener"}
        if self.bootstrap_password:
            conf = runtime / "config"
            shutil.copytree("/usr/share/elasticsearch/config", conf)
            for path in [conf, *conf.rglob("*")]:
                os.chown(path, 1000, 1000)
            es_env["ES_PATH_CONF"] = str(conf)
            subprocess.run(
                [
                    "/usr/share/elasticsearch/bin/elasticsearch-keystore",
                    "add",
                    "-xf",
                    "bootstrap.password",
                ],
                input=self.bootstrap_password + "\n",
                text=True,
                check=True,
                env=es_env,
                user=1000,
                group=1000,
                extra_groups=(),
                stdout=subprocess.DEVNULL,
                stderr=self.log,
            )
        self.process = subprocess.Popen(
            [
                "/usr/share/elasticsearch/bin/elasticsearch",
                "-Ediscovery.type=single-node",
                "-Enetwork.host=127.0.0.1",
                f"-Expack.security.enabled={str(bool(self.bootstrap_password)).lower()}",
                "-Expack.security.http.ssl.enabled=false",
                "-Expack.security.autoconfiguration.enabled=false",
                "-Expack.ml.enabled=false",
                "-Eingest.geoip.downloader.enabled=false",
                "-Enode.store.allow_mmap=false",
                f"-Epath.data={runtime / 'data'}",
                f"-Epath.logs={runtime / 'logs'}",
            ],
            cwd="/usr/share/elasticsearch",
            # Start the attach socket eagerly: gVisor does not expose the
            # process signal state expected by Java's lazy attach handshake.
            env=es_env,
            user=1000,
            group=1000,
            extra_groups=(),
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        try:
            self.es = (
                Elasticsearch(
                    "http://127.0.0.1:9200",
                    basic_auth=("elastic", self.bootstrap_password),
                    request_timeout=15,
                )
                if self.bootstrap_password
                else client()
            )
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError("Elasticsearch exited during startup")
                if self.es.ping():
                    break
                time.sleep(1)
            else:
                raise TimeoutError("Elasticsearch startup exceeded 180 seconds")
            index_corpus(self.directory, es=self.es)
        except Exception:
            self.log.flush()
            print((runtime / "startup.log").read_text()[-8000:])
            self.stop()
            raise

    def search(self, payload):
        try:
            request = SearchRequest.model_validate(payload)
            company = self.registry.resolve(request)
        except ValidationError as exc:
            return {"status": "error", "code": "invalid_request", "message": str(exc)}
        except SearchError as exc:
            return {"status": "error", "code": exc.code, "message": exc.message}
        response = self.es.search(
            index=self.manifest["corpus_version"], **query_body(request, company)
        )
        if response.get("timed_out") or response.get("_shards", {}).get("failed", 0):
            raise RuntimeError("Incomplete Elasticsearch response")
        return {
            "status": "ok",
            "corpus_version": self.manifest["corpus_version"],
            "resolved_company": company,
            "year": request.year,
            "total_hits": response["hits"]["total"]["value"],
            "results": [
                x["_source"] | {"score": x["_score"]} for x in response["hits"]["hits"]
            ],
        }

    def stop(self):
        if self.es is not None:
            self.es.close()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.log is not None:
            self.log.close()
        if self.workspace is not None:
            self.workspace.cleanup()
