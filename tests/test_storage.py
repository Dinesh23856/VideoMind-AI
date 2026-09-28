"""Tests for LocalStorageProvider and S3StorageProvider fail-fast behavior."""

import tempfile
from pathlib import Path
import pytest

from app.storage.provider import LocalStorageProvider, S3StorageProvider


def test_local_storage_crud_and_traversal_protection():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorageProvider(base_dir=tmpdir)

        # 1. Save and read
        key = "lectures/math/notes.txt"
        data = b"Integral calculus foundations."
        saved_path = storage.save(key, data)
        assert Path(saved_path).exists()
        assert storage.exists(key)
        assert storage.read(key) == data

        # 2. Save file
        src_file = Path(tmpdir) / "source.txt"
        src_file.write_bytes(b"Derivative derivations.")
        saved_file = storage.save_file("lectures/math/derived.txt", str(src_file))
        assert storage.exists("lectures/math/derived.txt")
        assert storage.read("lectures/math/derived.txt") == b"Derivative derivations."

        # 3. Path traversal attack protection
        traversal_key = "../../../outside_secret.txt"
        saved_traversal = storage.save(traversal_key, b"Attacker data")
        # Ensure target file was contained strictly within tmpdir
        assert Path(saved_traversal).resolve().is_relative_to(Path(tmpdir).resolve())

        # 4. Delete
        assert storage.delete(key) is True
        assert storage.exists(key) is False

        # Nonexistent read raises FileNotFoundError
        with pytest.raises(FileNotFoundError):
            storage.read(key)


def test_s3_storage_fails_fast_when_unconfigured():
    provider = S3StorageProvider()
    # When S3 credentials/bucket are missing, S3 operations must fail fast rather than silently write to disk
    with pytest.raises(RuntimeError) as exc_info:
        provider.save("test.txt", b"content")
    assert "S3 storage is not properly configured" in str(exc_info.value)
