import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blue.config import load_config  # noqa: E402
from blue.data import prepare_splits  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    """The project config, loaded once for the tests."""
    return load_config()


@pytest.fixture(scope="session")
def splits(cfg):
    """Train, test and edge-case CSVs (created if missing)."""
    return prepare_splits(cfg)


@pytest.fixture()
def tmp_cfg(cfg, tmp_path):
    """Config whose artifacts and LLM cache live in a temp dir (so tests never touch the repo's own outputs)."""
    c = copy.deepcopy(cfg)
    c["paths"]["artifacts_dir"] = str(tmp_path / "artifacts")
    c["paths"]["cache_dir"] = str(tmp_path / "llm_cache")
    return c
