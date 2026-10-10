import runpy
from pathlib import Path

import modal


def test_remote_definition_does_not_read_local_credentials(monkeypatch, tmp_path):
    module = Path(__file__).resolve().parents[1] / "modal_kibana.py"
    monkeypatch.setattr(modal, "is_local", lambda: False)
    monkeypatch.chdir(tmp_path)
    definition = runpy.run_path(str(module))
    assert definition["app"].name == "finance-rag-kibana"
