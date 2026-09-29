"""Tests for Autonomous Agent Platform, Supervisor, Plan-Execute-Verify, Research Agent, and Project Mode."""

import pytest
from app.agents.contracts import AgentRole, AgentState, AgentStep
from app.agents.supervisor import AgentSupervisor, SupervisorPlan
from app.agents.research import AutonomousResearchAgent
from app.agents.project_mode import ProjectModeSynthesizer
from app.domain.models import ConceptNode


def test_agent_supervisor_plan_and_execution():
    supervisor = AgentSupervisor(max_repairs=1)
    plan = supervisor.create_plan("Research multi-head attention vs state space models")

    assert len(plan.steps) == 4
    assert plan.status == AgentState.PLANNING

    # Mock role executors
    executors = {
        AgentRole.SEARCH: lambda action, inp: {"retrieved_docs": ["doc_mha", "doc_mamba"]},
        AgentRole.KNOWLEDGE: lambda action, inp: {"claims": ["MHA is O(N^2)", "Mamba is O(N)"]},
        AgentRole.RESEARCH: lambda action, inp: {"summary": "Comparative study between attention and SSMs"},
        AgentRole.EVALUATION: lambda action, inp: {"citations_verified": True},
    }

    executed_plan = supervisor.execute_plan(plan, executors)
    assert executed_plan.status == AgentState.DONE
    assert executed_plan.final_output["citations_verified"] is True
    assert len(executed_plan.final_output["claims"]) == 2


def test_agent_supervisor_self_repair():
    supervisor = AgentSupervisor(max_repairs=2)
    plan = supervisor.create_plan("Build project on convolutional neural networks")

    attempt = 0

    def mock_executor(action, inp):
        nonlocal attempt
        attempt += 1
        if inp.get("repair_mode"):
            return {"status": "repaired_output"}
        return {"status": "unverified_output"}

    executors = {
        AgentRole.KNOWLEDGE: mock_executor,
        AgentRole.COURSE: mock_executor,
        AgentRole.EVALUATION: mock_executor,
    }

    # Verifier that rejects first course step output to trigger repair
    def verifier(step: AgentStep) -> bool:
        if step.role == AgentRole.COURSE and not step.input_data.get("repair_mode"):
            return False
        return True

    executed_plan = supervisor.execute_plan(plan, executors, verifier=verifier)
    assert executed_plan.status == AgentState.DONE
    assert executed_plan.repair_count == 1


def test_autonomous_research_agent():
    agent = AutonomousResearchAgent()

    passages = [
        {"source_id": "v1", "title": "Attention Lecture", "text": "Self-attention computes dot-product query-key affinities.", "timestamp": 12.0},
        {"source_id": "v2", "title": "Transformer Deep Dive", "text": "Self-attention computes dot-product query-key affinities.", "timestamp": 45.0},
        {"source_id": "v3", "title": "Linear Attention", "text": "Linear attention avoids quadratic matrix multiplication.", "timestamp": 90.0},
    ]

    report = agent.conduct_research("Transformer Attention Complexity", passages)
    assert report.topic == "Transformer Attention Complexity"
    assert len(report.resolved_claims) >= 2
    assert len(report.citations) == 3
    assert len(report.open_questions) > 0


def test_project_mode_synthesizer():
    synthesizer = ProjectModeSynthesizer()

    concepts = [
        ConceptNode(id="c1", name="ResNet", definition="Deep residual learning with skip connections"),
        ConceptNode(id="c2", name="Batch Normalization", definition="Accelerating deep network training"),
    ]

    code_snippet = [{"code": "class ResidualBlock(nn.Module): pass"}]

    proj = synthesizer.synthesize_project("Deep Residual Networks", concepts, code_snippet)
    assert proj.title == "Hands-On Master Project: Building Deep Residual Networks"
    assert len(proj.milestones) == 3
    assert proj.milestones[0].milestone_number == 1
    assert proj.milestones[1].starter_code_hint == "class ResidualBlock(nn.Module): pass"
    assert len(proj.rubric) == 4
    total_rubric_weight = sum(r.weight_percent for r in proj.rubric)
    assert total_rubric_weight == 100
