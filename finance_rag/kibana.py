"""Authenticated Kibana plus a local copy of the fixed FinQA corpus."""

import json
import os
import subprocess
import time
from pathlib import Path

import requests

from finance_rag.embedded import EmbeddedSearch
from finance_rag.search import index_name


class KibanaStack:
    def __init__(self, directory):
        self.engine = EmbeddedSearch(
            directory, bootstrap_password=os.environ["ELASTIC_BOOTSTRAP_PASSWORD"]
        )
        self.process = None
        self.log = None

    def start(self):
        try:
            self.engine.start()
            es = self.engine.es
            es.security.change_password(
                username="elastic", password=os.environ["ELASTIC_BOOTSTRAP_PASSWORD"]
            )
            es.security.change_password(
                username="kibana_system", password=os.environ["KIBANA_SYSTEM_PASSWORD"]
            )
            es.security.put_role(
                name="finqa_inspector",
                cluster=["monitor"],
                indices=[
                    {
                        "names": ["finqa-*"],
                        "privileges": ["read", "view_index_metadata"],
                    }
                ],
            )
            es.security.put_user(
                username=os.environ["KIBANA_USERNAME"],
                password=os.environ["KIBANA_PASSWORD"],
                roles=["finqa_inspector", "kibana_admin"],
                full_name="FinQA researcher",
            )
            runtime = Path(self.engine.workspace.name)
            config = runtime / "kibana.yml"
            settings = {
                "server.host": "0.0.0.0",
                "server.port": 5601,
                "server.name": "FinQA Explorer",
                "server.shutdownTimeout": "5s",
                "path.data": str(runtime / "kibana-data"),
                "elasticsearch.hosts": ["http://127.0.0.1:9200"],
                "elasticsearch.username": "kibana_system",
                "elasticsearch.password": os.environ["KIBANA_SYSTEM_PASSWORD"],
                "xpack.security.secureCookies": True,
                "xpack.security.encryptionKey": os.environ["KIBANA_SECURITY_KEY"],
                "xpack.encryptedSavedObjects.encryptionKey": os.environ[
                    "KIBANA_SAVED_OBJECTS_KEY"
                ],
                "xpack.reporting.encryptionKey": os.environ["KIBANA_REPORTING_KEY"],
                "xpack.fleet.enabled": False,
                "telemetry.enabled": False,
                "telemetry.optIn": False,
            }
            config.write_text(
                "\n".join(
                    f"{key}: {json.dumps(value)}" for key, value in settings.items()
                )
                + "\n"
            )
            config.chmod(0o600)
            os.chown(config, 1000, 1000)
            self.log = (runtime / "kibana.log").open("w")
            self.process = subprocess.Popen(
                ["/usr/share/kibana/bin/kibana", "--config", str(config)],
                cwd="/usr/share/kibana",
                env=os.environ | {"NODE_OPTIONS": "--max-old-space-size=1536"},
                user=1000,
                group=1000,
                extra_groups=(),
                stdout=self.log,
                stderr=subprocess.STDOUT,
            )
            base = "http://127.0.0.1:5601"
            auth = (os.environ["KIBANA_USERNAME"], os.environ["KIBANA_PASSWORD"])
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError("Kibana exited during startup")
                try:
                    response = requests.get(base + "/api/status", auth=auth, timeout=5)
                    if (
                        response.ok
                        and response.json()
                        .get("status", {})
                        .get("overall", {})
                        .get("level")
                        == "available"
                    ):
                        break
                except (requests.RequestException, ValueError):
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("Kibana startup exceeded 240 seconds")
            headers = {"kbn-xsrf": "bootstrap"}
            response = requests.post(
                base + "/api/data_views/data_view",
                auth=auth,
                headers=headers,
                json={
                    "data_view": {
                        "id": "finqa-contexts",
                        "title": "finqa-*",
                        "name": "FinQA contexts",
                    }
                },
                timeout=30,
            )
            response.raise_for_status()
            response = requests.post(
                base + "/api/data_views/default",
                auth=auth,
                headers=headers,
                json={"data_view_id": "finqa-contexts", "force": True},
                timeout=30,
            )
            response.raise_for_status()
            # Browser login must protect the document APIs even though Modal
            # exposes the web server publicly over HTTPS.
            response = requests.get(base + "/api/data_views", timeout=10)
            if response.status_code != 401:
                raise RuntimeError("Kibana data API is not protected by authentication")
            count = es.count(index=index_name(self.engine.manifest))["count"]
            if count != self.engine.manifest["unique_contexts"]:
                raise RuntimeError("FinQA index count mismatch")
            print(
                json.dumps(
                    {
                        "kibana": "ready",
                        "documents": count,
                        "corpus_version": self.engine.manifest["corpus_version"],
                        "data_view": "FinQA contexts",
                        "unauthenticated_api_status": response.status_code,
                    }
                )
            )
        except Exception:
            if self.log is not None:
                self.log.flush()
                # Avoid logging configuration or credentials.
                print(
                    (Path(self.engine.workspace.name) / "kibana.log").read_text()[
                        -6000:
                    ]
                )
            self.stop()
            raise

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.log is not None:
            self.log.close()
        self.engine.stop()
