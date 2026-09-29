#!/usr/bin/env python3
"""VideoMind-AI Command Line Interface: process, search, ask AI tutor, export artifacts, courses, and admin."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import List, Optional

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.orchestration.tasks import process_video_job  # noqa: E402
from app.retrieval.hybrid_search import HybridSearchEngine  # noqa: E402
from app.rag.engine import RAGEngine  # noqa: E402
from app.tutor.engine import AITutorEngine, TutorMode  # noqa: E402
from app.knowledge.graph import KnowledgeGraph  # noqa: E402
from app.domain.models import ConceptNode, TranscriptSegment  # noqa: E402
from app.artifacts.exporters import (  # noqa: E402
    AnkiExporter,
    DocxExporter,
    JsonExporter,
    MarkdownExporter,
    SrtExporter,
    TypstPdfExporter,
    VttExporter,
)

app = typer.Typer(help="VideoMind AI — Video-to-Knowledge OS CLI")
console = Console(legacy_windows=False)


@app.command()
def process(
    source: str = typer.Argument(..., help="YouTube URL or local media file path"),
    title: Optional[str] = typer.Option(None, help="Override lecture title"),
    output: str = typer.Option("./output", help="Directory for generated artifacts"),
    json_output: bool = typer.Option(False, "--json", help="Output pure JSON format"),
):
    """Run the complete multimodal pipeline synchronously."""
    job_id = str(uuid.uuid4())
    source_type = "url" if source.startswith("http") else "file"
    out = Path(output) / job_id
    out.mkdir(parents=True, exist_ok=True)

    if not json_output:
        console.print(f"[bold cyan]VideoMind OS[/bold cyan] Job: {job_id}")
        console.print(f"[bold]Source[/bold]: {source} ({source_type})")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        disable=json_output,
    ) as progress:
        task = progress.add_task("Running knowledge pipeline...", total=None)
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
            if not json_output:
                console.print(f"[red]Failed:[/red] {exc}")
            raise typer.Exit(1)

    if json_output:
        print(json.dumps(result, indent=2, default=str))
    else:
        console.print_json(json.dumps(result, indent=2, default=str))
        console.print(f"\n[green]Artifacts written to[/green] {out}")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query terms or technical concept"),
    job_dir: Optional[str] = typer.Option(None, help="Path to job output directory with transcript.json"),
    k: int = typer.Option(5, help="Number of results to retrieve"),
):
    """Hybrid search across transcript segments with timestamp citations."""
    engine = HybridSearchEngine()
    if job_dir and Path(job_dir).exists():
        tr_path = Path(job_dir) / "transcript.json"
        if tr_path.exists():
            data = json.loads(tr_path.read_text(encoding="utf-8"))
            segs = [TranscriptSegment(**s) for s in data]
            engine.index_segments("job_local", "Lecture", segs)

    results = engine.search(query, k=k)
    table = Table(title=f"Search Results for '{query}'")
    table.add_column("Score", style="cyan", width=8)
    table.add_column("Timestamp", style="green", width=18)
    table.add_column("Snippet", style="white")

    for r in results:
        table.add_row(f"{r.score:.1f}", r.timestamp_formatted, r.snippet[:120] + "...")

    console.print(table)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask AI Tutor"),
    job_dir: Optional[str] = typer.Option(None, help="Path to job directory"),
    mode: str = typer.Option("Intermediate", help="Tutor mode: Beginner, Intermediate, Developer, Interview, Exam"),
):
    """Ask the Socratic AI Tutor grounded questions."""
    engine = HybridSearchEngine()
    if job_dir and Path(job_dir).exists():
        tr_path = Path(job_dir) / "transcript.json"
        if tr_path.exists():
            data = json.loads(tr_path.read_text(encoding="utf-8"))
            segs = [TranscriptSegment(**s) for s in data]
            engine.index_segments("job_local", "Lecture", segs)

    rag = RAGEngine(engine)
    tutor = AITutorEngine(rag)
    t_mode = TutorMode(mode) if mode in TutorMode.__members__ else TutorMode.INTERMEDIATE
    res = tutor.interact(question, mode=t_mode)

    console.print(f"\n[bold cyan]AI Tutor ({t_mode.value}):[/bold cyan]\n")
    console.print(res.tutor_reply)
    if res.suggested_follow_ups:
        console.print("\n[bold yellow]Suggested Follow-ups:[/bold yellow]")
        for f in res.suggested_follow_ups:
            console.print(f"- {f}")


@app.command()
def export(
    job_dir: str = typer.Argument(..., help="Path to job output directory with notes.md and transcript.json"),
    format: str = typer.Option("pdf", help="Target format: pdf, anki, srt, vtt, docx, json"),
):
    """Export artifacts to a specific format."""
    j_path = Path(job_dir)
    notes_path = j_path / "notes.md"
    tr_path = j_path / "transcript.json"

    md_content = notes_path.read_text(encoding="utf-8") if notes_path.exists() else ""
    segments = json.loads(tr_path.read_text(encoding="utf-8")) if tr_path.exists() else []

    data = {"markdown": md_content, "segments": segments}
    exporters = {
        "pdf": (TypstPdfExporter(), "notes.pdf"),
        "anki": (AnkiExporter(), "deck.apkg"),
        "srt": (SrtExporter(), "subtitles.srt"),
        "vtt": (VttExporter(), "subtitles.vtt"),
        "docx": (DocxExporter(), "notes.docx"),
        "json": (JsonExporter(), "knowledge.json"),
    }

    if format not in exporters:
        console.print(f"[red]Unknown format '{format}'.[/red] Choices: {list(exporters.keys())}")
        raise typer.Exit(1)

    exporter, filename = exporters[format]
    out_file = j_path / filename
    exporter.export("Lecture", data, out_file)
    console.print(f"[green]Successfully exported:[/green] {out_file}")


@app.command()
def course(
    title: str = typer.Argument(..., help="Course title"),
    video_ids: List[str] = typer.Option([], "--video-id", help="Video IDs to include in course"),
):
    """Synthesize a structured multi-unit course curriculum."""
    from app.learning.course import CourseGenerator
    generator = CourseGenerator()
    course_obj = generator.generate_course(
        course_id=f"course_{uuid.uuid4().hex[:6]}",
        title=title,
        video_ids=video_ids or ["vid_default"],
    )
    console.print(f"[bold green]Course Generated:[/bold green] {course_obj.title} ({course_obj.difficulty})")
    for mod in course_obj.modules:
        console.print(f"  • Module: [bold]{mod.title}[/bold] ({len(mod.lessons)} lessons)")


@app.command()
def graph(
    concept: Optional[str] = typer.Option(None, help="Concept to inspect prerequisites for"),
):
    """Display Knowledge Graph information or prerequisite learning paths."""
    from app.knowledge.graph import KnowledgeGraph
    kg = KnowledgeGraph()
    kg.add_concept(ConceptNode(id="c1", name="Calculus", definition="Study of continuous change"))
    kg.add_concept(ConceptNode(id="c2", name="Gradient Descent", definition="Optimization algorithm"))
    kg.add_edge("c2", "c1", relation_type="PREREQUISITE_OF")

    if concept:
        prereqs = kg.get_prerequisites(concept)
        console.print(f"Prerequisites for [bold cyan]{concept}[/bold cyan]: {[p.name for p in prereqs]}")
    else:
        path = kg.get_topological_learning_path()
        console.print(f"Topological Learning Path: {[c.name for c in path]}")


@app.command()
def admin():
    """Display operational telemetry and system health."""
    from app.database.session import SessionLocal, init_db
    from app.admin.console import AdminConsoleService

    init_db()
    db = SessionLocal()
    try:
        service = AdminConsoleService()
        telemetry = service.get_system_telemetry(db)
        console.print_json(json.dumps(telemetry, indent=2))
    finally:
        db.close()


if __name__ == "__main__":
    app()
