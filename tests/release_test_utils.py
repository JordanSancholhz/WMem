"""Temporary directories shared by training regression tests."""
from contextlib import contextmanager
from pathlib import Path
import shutil
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def test_directory():
    base = (ROOT / "tmp" / "release-tests").resolve()
    directory = base / uuid4().hex
    directory.mkdir(parents=True)
    try:
        yield str(directory)
    finally:
        target = directory.resolve()
        if target == base or not target.is_relative_to(base):
            raise RuntimeError("Refusing to remove a path outside the test directory")
        shutil.rmtree(target)
