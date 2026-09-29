"""Agent Supervisor: Hierarchical multi-agent coordinator with self-repair verification loop."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

from app.agents.contracts import AgentRole, AgentState, AgentStep

logger = logging.getLogger(__name__)


class SupervisorPlan(BaseModel):
    plan_id: str
    goal: str
    steps: List[AgentStep] = Field(default_factory=list)
    current_index: int = 0
    status: AgentState = AgentState.IDLE
    repair_count: int = 0
    max_repairs: int = 2
    final_output: Dict[str, Any] = Field(default_factory=dict)


class AgentSupervisor:
    """Oversees autonomous sub-agent execution, validating outputs and managing repair loops."""

    def __init__(self, max_repairs: int = 2) -> None:
        self.max_repairs = max_repairs

    def create_plan(self, goal: str) -> SupervisorPlan:
        goal_lower = goal.lower()
        steps: List[AgentStep] = []

        if "research" in goal_lower:
            steps = [
                AgentStep(role=AgentRole.SEARCH, action="retrieve_sources", input_data={"goal": goal}),
                AgentStep(role=AgentRole.KNOWLEDGE, action="extract_claims", input_data={}),
                AgentStep(role=AgentRole.RESEARCH, action="synthesize_findings", input_data={}),
                AgentStep(role=AgentRole.EVALUATION, action="verify_citations", input_data={}),
            ]
        elif "project" in goal_lower:
            steps = [
                AgentStep(role=AgentRole.KNOWLEDGE, action="extract_technologies", input_data={"goal": goal}),
                AgentStep(role=AgentRole.COURSE, action="generate_milestones", input_data={}),
                AgentStep(role=AgentRole.EVALUATION, action="verify_rubric", input_data={}),
            ]
        else:
            # Standard video-to-knowledge pipeline
            steps = [
                AgentStep(role=AgentRole.TRANSCRIPTION, action="process_audio", input_data={"goal": goal}),
                AgentStep(role=AgentRole.VISION, action="extract_frames", input_data={}),
                AgentStep(role=AgentRole.KNOWLEDGE, action="extract_concepts", input_data={}),
                AgentStep(role=AgentRole.EVALUATION, action="verify_knowledge", input_data={}),
            ]

        return SupervisorPlan(
            plan_id=f"plan_{int(time.time()*1000)}",
            goal=goal,
            steps=steps,
            status=AgentState.PLANNING,
            max_repairs=self.max_repairs,
        )

    def execute_plan(
        self,
        plan: SupervisorPlan,
        role_executors: Dict[AgentRole, Callable[[str, Dict[str, Any]], Dict[str, Any]]],
        verifier: Optional[Callable[[AgentStep], bool]] = None,
    ) -> SupervisorPlan:
        plan.status = AgentState.EXECUTING
        accumulated_context: Dict[str, Any] = {"goal": plan.goal}

        for idx, step in enumerate(plan.steps):
            plan.current_index = idx
            executor = role_executors.get(step.role)

            if not executor:
                step.state = AgentState.FAILED
                step.error = f"No registered executor for agent role {step.role.value}"
                plan.status = AgentState.FAILED
                return plan

            # Prepare inputs with accumulated context
            inputs = {**step.input_data, **accumulated_context}
            step.state = AgentState.EXECUTING
            start = time.time()

            try:
                out = executor(step.action, inputs)
                step.output_data = out
                step.duration_seconds = time.time() - start

                # Verification step
                if verifier:
                    step.state = AgentState.VERIFYING
                    is_valid = verifier(step)
                    if not is_valid and plan.repair_count < plan.max_repairs:
                        # Enter repair
                        step.state = AgentState.REPAIRING
                        plan.repair_count += 1
                        logger.warning(f"Step {step.step_id} failed verification. Repairing (attempt {plan.repair_count})...")
                        # Retry with repair prompt context
                        repair_inputs = {**inputs, "repair_mode": True}
                        out = executor(step.action, repair_inputs)
                        step.output_data = out

                step.state = AgentState.DONE
                accumulated_context.update(out or {})

            except Exception as ex:
                step.state = AgentState.FAILED
                step.error = str(ex)
                step.duration_seconds = time.time() - start
                plan.status = AgentState.FAILED
                return plan

        plan.status = AgentState.DONE
        plan.final_output = accumulated_context
        return plan
