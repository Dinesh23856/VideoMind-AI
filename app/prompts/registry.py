"""Versioned Prompt Template Registry with validation and fallback."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PromptTemplate(BaseModel):
    template_name: str
    version: int = 1
    system_template: str
    user_template: str
    input_variables: List[str] = Field(default_factory=list)
    description: str = ""
    is_active: bool = True

    def render(self, variables: Dict[str, Any]) -> tuple[str, str]:
        # Validate variables
        for var in self.input_variables:
            if var not in variables:
                raise ValueError(f"Missing variable '{var}' required by prompt template {self.template_name} v{self.version}")

        # Safe replace {var}
        sys_rendered = self.system_template
        usr_rendered = self.user_template
        for k, v in variables.items():
            sys_rendered = sys_rendered.replace(f"{{{k}}}", str(v))
            usr_rendered = usr_rendered.replace(f"{{{k}}}", str(v))

        return sys_rendered, usr_rendered


class PromptRegistry:
    """Manages versioned, testable, and rollable prompt templates."""

    def __init__(self) -> None:
        self._templates: Dict[str, Dict[int, PromptTemplate]] = {}
        self._init_defaults()

    def _init_defaults(self) -> None:
        self.register(
            PromptTemplate(
                template_name="concept_explanation",
                version=1,
                system_template="You are VideoMind AI Tutor. Provide accurate, evidence-backed explanations.",
                user_template="Explain concept {concept_name} in context of video {video_title}.",
                input_variables=["concept_name", "video_title"],
            )
        )
        self.register(
            PromptTemplate(
                template_name="concept_explanation",
                version=2,
                system_template="You are VideoMind AI 2.0 Senior Academic Tutor. Cite timestamps and provide LaTeX formulas where applicable.",
                user_template="Explain concept {concept_name} from lecture {video_title} with mathematical precision.",
                input_variables=["concept_name", "video_title"],
            )
        )

    def register(self, template: PromptTemplate) -> None:
        if template.template_name not in self._templates:
            self._templates[template.template_name] = {}
        self._templates[template.template_name][template.version] = template

    def get(self, template_name: str, version: Optional[int] = None) -> PromptTemplate:
        if template_name not in self._templates:
            raise KeyError(f"Template '{template_name}' not found in registry.")

        versions = self._templates[template_name]
        if version is not None:
            if version not in versions:
                raise KeyError(f"Version {version} not found for template '{template_name}'.")
            return versions[version]

        # Return latest active version
        latest_v = max(versions.keys())
        return versions[latest_v]

    def render(self, template_name: str, variables: Dict[str, Any], version: Optional[int] = None) -> tuple[str, str]:
        template = self.get(template_name, version)
        return template.render(variables)
