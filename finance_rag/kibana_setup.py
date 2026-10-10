"""Generate local, ignored credentials; never print passwords."""

import json
import secrets
from pathlib import Path


def main():
    path = Path("data/kibana-credentials.json")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        credentials = {
            "KIBANA_USERNAME": "researcher",
            "KIBANA_PASSWORD": secrets.token_urlsafe(24),
            "ELASTIC_BOOTSTRAP_PASSWORD": secrets.token_urlsafe(32),
            "KIBANA_SYSTEM_PASSWORD": secrets.token_urlsafe(32),
            "KIBANA_SECURITY_KEY": secrets.token_hex(32),
            "KIBANA_SAVED_OBJECTS_KEY": secrets.token_hex(32),
            "KIBANA_REPORTING_KEY": secrets.token_hex(32),
        }
        path.write_text(json.dumps(credentials, indent=2), encoding="utf-8")
    print(f"Credentials available in {path.resolve()} (ignored by Git).")
    credentials = json.loads(path.read_text(encoding="utf-8"))
    login = path.parent / "kibana-login.txt"
    login.write_text(
        f"Username: {credentials['KIBANA_USERNAME']}\nPassword: {credentials['KIBANA_PASSWORD']}\n",
        encoding="utf-8",
    )
    print(f"Browser login available in {login.resolve()} (ignored by Git).")


if __name__ == "__main__":
    main()
