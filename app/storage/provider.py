"""Storage provider implementations for local disk and S3-compatible cloud storage."""

from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from config.settings import get_settings


class StorageProvider(ABC):
    """Abstract interface for object and media file storage."""

    @abstractmethod
    def save(self, key: str, data: bytes) -> str:
        """Store bytes and return the target storage path or URI."""
        pass

    @abstractmethod
    def save_file(self, key: str, local_path: str) -> str:
        """Copy local file to storage and return the target path."""
        pass

    @abstractmethod
    def read(self, key: str) -> bytes:
        """Retrieve stored bytes."""
        pass

    @abstractmethod
    def stream(self, key: str, chunk_size: int = 64 * 1024):
        """Stream stored bytes in bounded chunks without loading whole file into RAM."""
        pass

    @abstractmethod
    def get_local_path(self, key: str) -> Optional[Path]:
        """Return local filesystem path if available, or None for remote-only stores."""
        pass

    @abstractmethod
    def get_size(self, key: str) -> int:
        """Return file size in bytes."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if file exists in storage."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Remove file from storage."""
        pass

    @abstractmethod
    def get_url(self, key: str, expiry_seconds: int = 3600) -> str:
        """Get accessible download URL (or local file URI)."""
        pass


class LocalStorageProvider(StorageProvider):
    """Local filesystem storage provider with strict path & symlink isolation."""

    def __init__(self, base_dir: Optional[str] = None):
        settings = get_settings()
        self.base_dir = Path(base_dir or settings.scratch_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        # Sanitize key and strictly reject path traversal attempts
        clean_key = str(key).replace("\\", "/")
        if clean_key.startswith("/") or (len(clean_key) > 1 and clean_key[1] == ":"):
            raise PermissionError(f"Path traversal detected: '{key}' is an absolute path.")

        parts = [p for p in clean_key.split("/") if p]
        if any(p == ".." for p in parts) or "/../" in f"/{clean_key}/":
            raise PermissionError(f"Path traversal detected: '{key}' contains directory traversal sequences.")

        target = self.base_dir.joinpath(*parts)

        # Verify strict containment within base_dir (including symlink resolution)
        resolved_base = self.base_dir.resolve()
        resolved_target = target.resolve()

        # If target doesn't exist yet, check its resolved parent
        check_path = resolved_target if target.exists() else resolved_target.parent
        try:
            check_path.relative_to(resolved_base)
        except ValueError:
            raise PermissionError(f"Path traversal detected: '{key}' attempts to escape storage root.")

        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def save(self, key: str, data: bytes) -> str:
        target = self._resolve(key)
        target.write_bytes(data)
        return str(target)

    def save_file(self, key: str, local_path: str) -> str:
        target = self._resolve(key)
        shutil.copy2(local_path, target)
        return str(target)

    def read(self, key: str) -> bytes:
        target = self._resolve(key)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"File not found in storage: {key}")
        return target.read_bytes()

    def stream(self, key: str, chunk_size: int = 64 * 1024):
        target = self._resolve(key)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"File not found in storage: {key}")
        with open(target, "rb") as f:
            while chunk := f.read(chunk_size):
                yield chunk

    def get_local_path(self, key: str) -> Optional[Path]:
        try:
            target = self._resolve(key)
            if target.exists() and target.is_file():
                return target
        except Exception:
            return None
        return None

    def get_size(self, key: str) -> int:
        target = self._resolve(key)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"File not found in storage: {key}")
        return target.stat().st_size

    def exists(self, key: str) -> bool:
        try:
            target = self._resolve(key)
            return target.exists() and target.is_file()
        except Exception:
            return False

    def delete(self, key: str) -> bool:
        try:
            target = self._resolve(key)
            if target.exists():
                target.unlink()
                return True
        except Exception:
            pass
        return False

    def get_url(self, key: str, expiry_seconds: int = 3600) -> str:
        return f"/api/v1/storage/{str(key).replace('\\', '/').lstrip('/')}"


class S3StorageProvider(StorageProvider):
    """S3-compatible object storage provider with graceful local fallback."""

    def __init__(self):
        settings = get_settings()
        self.bucket = settings.s3_bucket
        self._s3_client = None

        if self.bucket and settings.aws_access_key_id and settings.aws_secret_access_key:
            try:
                import boto3  # type: ignore
                self._s3_client = boto3.client(
                    "s3",
                    endpoint_url=settings.s3_endpoint_url,
                    aws_access_key_id=settings.aws_access_key_id,
                    aws_secret_access_key=settings.aws_secret_access_key,
                )
            except Exception as exc:
                raise RuntimeError(f"Failed to initialize configured S3 client: {exc}") from exc

    def save(self, key: str, data: bytes) -> str:
        if not self._s3_client or not self.bucket:
            raise RuntimeError("S3 storage is not properly configured.")
        self._s3_client.put_object(Bucket=self.bucket, Key=key, Body=data)
        return f"s3://{self.bucket}/{key}"

    def save_file(self, key: str, local_path: str) -> str:
        if not self._s3_client or not self.bucket:
            raise RuntimeError("S3 storage is not properly configured.")
        self._s3_client.upload_file(local_path, self.bucket, key)
        return f"s3://{self.bucket}/{key}"

    def read(self, key: str) -> bytes:
        if not self._s3_client or not self.bucket:
            raise RuntimeError("S3 storage is not properly configured.")
        resp = self._s3_client.get_object(Bucket=self.bucket, Key=key)
        return resp["Body"].read()

    def stream(self, key: str, chunk_size: int = 64 * 1024):
        if not self._s3_client or not self.bucket:
            raise RuntimeError("S3 storage is not properly configured.")
        resp = self._s3_client.get_object(Bucket=self.bucket, Key=key)
        body = resp["Body"]
        while chunk := body.read(chunk_size):
            yield chunk

    def get_local_path(self, key: str) -> Optional[Path]:
        return None

    def get_size(self, key: str) -> int:
        if not self._s3_client or not self.bucket:
            raise RuntimeError("S3 storage is not properly configured.")
        resp = self._s3_client.head_object(Bucket=self.bucket, Key=key)
        return int(resp.get("ContentLength", 0))

    def exists(self, key: str) -> bool:
        if not self._s3_client or not self.bucket:
            raise RuntimeError("S3 storage is not properly configured.")
        try:
            self._s3_client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def delete(self, key: str) -> bool:
        if not self._s3_client or not self.bucket:
            raise RuntimeError("S3 storage is not properly configured.")
        self._s3_client.delete_object(Bucket=self.bucket, Key=key)
        return True

    def get_url(self, key: str, expiry_seconds: int = 3600) -> str:
        if not self._s3_client or not self.bucket:
            raise RuntimeError("S3 storage is not properly configured.")
        return self._s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expiry_seconds,
        )


def get_storage_provider() -> StorageProvider:
    settings = get_settings()
    if settings.s3_bucket and settings.aws_access_key_id:
        return S3StorageProvider()
    return LocalStorageProvider()
