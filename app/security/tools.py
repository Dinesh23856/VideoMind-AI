"""Tool Execution Sandbox for safe, deterministic agent tool dispatch."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field


class ToolPermission(str, Enum):
    READ_ONLY = "READ_ONLY"
    SEARCH = "SEARCH"
    KNOWLEDGE_READ = "KNOWLEDGE_READ"
    MATH_RENDER = "MATH_RENDER"


class ToolSecurityError(Exception):
    pass


class SafeToolDefinition(BaseModel):
    name: str
    description: str
    permission: ToolPermission
    required_params: List[str] = Field(default_factory=list)


class ToolExecutionSandbox:
    """Sandboxes and dispatches agent tool calls with strict permission boundaries and input isolation."""

    # Disallowed function names and substrings
    FORBIDDEN_CALLS = {"exec", "eval", "os.system", "subprocess", "shutil.rmtree", "open"}

    def __init__(self) -> None:
        self._tools: Dict[str, tuple[SafeToolDefinition, Callable[..., Any]]] = {}

    def register_tool(
        self,
        name: str,
        description: str,
        permission: ToolPermission,
        func: Callable[..., Any],
        required_params: Optional[List[str]] = None,
    ) -> None:
        # Check forbidden
        func_name = getattr(func, "__name__", "")
        if func_name in self.FORBIDDEN_CALLS or any(f in name.lower() for f in ("eval", "exec", "shell", "bash")):
            raise ToolSecurityError(f"Cannot register unsafe tool: {name}")

        definition = SafeToolDefinition(
            name=name,
            description=description,
            permission=permission,
            required_params=required_params or [],
        )
        self._tools[name] = (definition, func)

    def execute_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        allowed_permissions: Optional[List[ToolPermission]] = None,
    ) -> Any:
        if tool_name not in self._tools:
            raise ToolSecurityError(f"Tool '{tool_name}' is not registered in the sandbox.")

        definition, func = self._tools[tool_name]

        # Permission check
        if allowed_permissions is not None and definition.permission not in allowed_permissions:
            raise ToolSecurityError(
                f"Permission denied: Tool '{tool_name}' requires {definition.permission.value}."
            )

        # Parameter validation
        for req in definition.required_params:
            if req not in params:
                raise ValueError(f"Missing required parameter '{req}' for tool '{tool_name}'.")

        # Execute safely
        try:
            return func(**params)
        except Exception as ex:
            raise RuntimeError(f"Tool execution failed in sandbox for '{tool_name}': {ex}") from ex

    def list_tools(self) -> List[SafeToolDefinition]:
        return [defn for defn, _ in self._tools.values()]
