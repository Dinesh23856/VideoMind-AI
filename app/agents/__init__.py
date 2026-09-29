"""Autonomous Multi-Agent Platform Subsystem."""

from app.agents.contracts import AgentMessage, AgentRole, AgentState, AgentStep
from app.agents.supervisor import AgentSupervisor, SupervisorPlan
from app.agents.research import AutonomousResearchAgent, ResearchReport
from app.agents.project_mode import ProjectModeSynthesizer, ProjectSpecification

__all__ = [
    "AgentRole",
    "AgentState",
    "AgentStep",
    "AgentMessage",
    "AgentSupervisor",
    "SupervisorPlan",
    "AutonomousResearchAgent",
    "ResearchReport",
    "ProjectModeSynthesizer",
    "ProjectSpecification",
]
