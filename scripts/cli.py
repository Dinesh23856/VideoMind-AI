#!/usr/bin/env python3
"""Local CLI for running the pipeline without the HTTP layer."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.orchestration.tasks import process_video_job  # noqa: E402

app = typer.Typer(help="Video-to-Knowledge Pipeline CLI")
console = Console()


@app.command()
def process(
    source: str = typer.Argument(..., help="YouTube URL or local media file path"),
    title: str = typer.Option(None, help="Override lecture title"),
    output: str = typer.Option("./output", help="Directory for generated artifacts"),
):
    """Run the full pipeline synchronously (useful for debugging / local use)."""
    job_id = str(uuid.uuid4())
    source_type = "url" if source.startswith("http") else "file"
    out = Path(output) / job_id
    out.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Job[/bold] {job_id}")
    console.print(f"[bold]Source[/bold] {source} ({source_type})")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running pipeline…", total=None)
        try:
            result = process_video_job(
                job_id=job_id,
                source=source,
                source_type=source_type,
                title=title,
                output_dir=str(out),
            )
            progress.update(task, description="Done")
        except Exception as exc:
            console.print(f"[red]Failed:[/red] {exc}")
            raise typer.Exit(1)

    console.print_json(json.dumps(result, indent=2, default=str))
    console.print(f"\n[green]Artifacts written to[/green] {out}")


if __name__ == "__main__":
    app()
