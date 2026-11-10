"""Active dataset identity shared by preparation, retrieval and model runners."""

from pathlib import Path

DATASET = "hotpotqa/hotpot_qa"
REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
CONFIG = "fullwiki"
CORPUS_VERSION = "hotpotqa-fullwiki-20171001-v2"
DATA_DIR = Path(__file__).resolve().parents[1] / "data/hotpotqa-documents"
SEARCH_APP = "hotpotqa-search"


def file_sha256(path):
    """Stream a file in bounded binary blocks and return its SHA-256 digest. Used to verify
    corpus and snapshot identity without loading multi-gigabyte files into memory.
    """
    import hashlib

    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()
