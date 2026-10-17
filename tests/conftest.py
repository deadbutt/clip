"""Keep Python temporary files and their sibling app state inside each test."""
from __future__ import annotations

import tempfile

import pytest


@pytest.fixture(autouse=True)
def isolated_temporary_directory(tmp_path, monkeypatch):
    # Some Windows environments fall back to cwd when the system temp is not
    # writable. App config/media live beside runs, so reserve a nested temp
    # root to prevent test apps from touching the real project's siblings.
    directory = tmp_path / "temporary"
    directory.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(directory))
