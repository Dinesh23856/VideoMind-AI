"""Base interface for all educational artifact exporters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


class BaseExporter(ABC):
    """Abstract interface that every artifact exporter must satisfy."""

    @property
    @abstractmethod
    def extension(self) -> str:
        """File extension without leading period (e.g. 'md', 'pdf', 'apkg')."""
        pass

    @property
    @abstractmethod
    def mime_type(self) -> str:
        """Standard MIME type string for HTTP streaming."""
        pass

    @abstractmethod
    def export(
        self,
        title: str,
        data: Dict[str, Any],
        output_path: str | Path,
    ) -> str:
        """
        Executes the export and writes to output_path.
        Returns the final string path to the generated artifact.
        """
        pass
