"""Project Mode Synthesizer: Converts instructional video lectures into real-world project specifications."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.domain.models import ConceptNode


class ProjectMilestone(BaseModel):
    milestone_number: int
    title: str
    description: str
    deliverables: List[str] = Field(default_factory=list)
    acceptance_criteria: List[str] = Field(default_factory=list)
    starter_code_hint: Optional[str] = None


class RubricCriterion(BaseModel):
    category: str  # Correctness, Architecture, Testing, Documentation
    description: str
    weight_percent: int


class ProjectSpecification(BaseModel):
    project_id: str
    title: str
    overview: str
    difficulty: str = "INTERMEDIATE"  # BEGINNER, INTERMEDIATE, ADVANCED
    prerequisites: List[str] = Field(default_factory=list)
    recommended_tech_stack: List[str] = Field(default_factory=list)
    milestones: List[ProjectMilestone] = Field(default_factory=list)
    rubric: List[RubricCriterion] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


class ProjectModeSynthesizer:
    """Transforms abstract educational concepts and code snippets into an end-to-end practical project."""

    def synthesize_project(
        self,
        topic: str,
        concepts: List[ConceptNode],
        extracted_code: Optional[List[Dict[str, Any]]] = None,
    ) -> ProjectSpecification:
        concept_names = [c.name for c in concepts]

        # Milestone 1: Environment & Core Data Structures
        m1 = ProjectMilestone(
            milestone_number=1,
            title=f"Core Foundations: Setup and Model Scaffold for {topic}",
            description="Establish module structure, type definitions, and core abstractions.",
            deliverables=["Directory structure", "Base interfaces", "Unit test scaffolding"],
            acceptance_criteria=["All imports resolve cleanly", "Base interface passes contract tests"],
            starter_code_hint="class BaseModule:\n    def forward(self, x):\n        raise NotImplementedError",
        )

        # Milestone 2: Algorithmic Implementation
        core_c = concept_names[0] if concept_names else topic
        m2 = ProjectMilestone(
            milestone_number=2,
            title=f"Engine Implementation: {core_c}",
            description=f"Implement the fundamental algorithms and transformations discussed in the lecture for {core_c}.",
            deliverables=[f"{core_c} transformation engine", "Boundary condition validation"],
            acceptance_criteria=["Handles edge cases without unhandled exceptions", "Mathematical invariants hold"],
            starter_code_hint=extracted_code[0].get("code") if extracted_code else None,
        )

        # Milestone 3: End-to-End Pipeline & Testing
        m3 = ProjectMilestone(
            milestone_number=3,
            title="Integration, Verification & Benchmarking",
            description="Tie components into an executable pipeline and benchmark throughput and correctness.",
            deliverables=["Integration test suite", "Benchmark script", "README and API docs"],
            acceptance_criteria=["Full pipeline executes end-to-end", "Test coverage >= 80%"],
        )

        rubric = [
            RubricCriterion(category="Correctness", description="Implementation accurately models target concept algorithms", weight_percent=40),
            RubricCriterion(category="Code Quality & Architecture", description="Clean separation of concerns, modularity, type hints", weight_percent=25),
            RubricCriterion(category="Test Coverage", description="Thorough unit and integration tests with edge cases", weight_percent=20),
            RubricCriterion(category="Documentation", description="Clear README, API usage instructions, and explanation", weight_percent=15),
        ]

        return ProjectSpecification(
            project_id=f"proj_{int(time.time()*1000)}",
            title=f"Hands-On Master Project: Building {topic}",
            overview=f"An applied engineering project designed to solidify your mastery of {', '.join(concept_names[:4])}.",
            difficulty="INTERMEDIATE" if len(concepts) > 2 else "BEGINNER",
            prerequisites=concept_names[:3],
            recommended_tech_stack=["Python 3.12+", "PyTorch / NumPy", "Pytest", "FastAPI"],
            milestones=[m1, m2, m3],
            rubric=rubric,
        )
