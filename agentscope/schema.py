"""
schema.py - The data contract for AgentScope traces.

Every other module either writes into these structures (tracer, runner)
or reads from them (exporter, analysis). Nothing else in the codebase
should define its own trace format.

Hierarchy:
    RunTrace
        └── StepTrace   (one per agent decision cycle)
                └── ToolCallTrace  (zero or more per step)
"""

from dataclasses import dataclass, field
from typing import Optional
import time
import uuid


@dataclass
class ToolCallTrace:
    """Records a single tool invocation within an agent step.

    Note: store summaries of inputs/outputs, not the raw values.
    Raw values can be arbitrarily large and are not serializable in
    general. A short descriptive string is enough for analysis and
    is safe to log.
    """
    tool_name: str
    input_summary: str
    output_summary: str
    success: bool
    latency_ms: float
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "success": self.success,
            "latency_ms": self.latency_ms,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ToolCallTrace":
        return cls(**data)


@dataclass
class StepTrace:
    """Records one agent decision cycle: observe -> think -> act.

    A step is NOT the same as an LLM call. One step may involve
    multiple LLM calls if the agent retries. Track retries here,
    not as separate steps.

    context_window_tokens: the total tokens in context at THIS step,
    including all prior conversation history carried forward. This is
    the field that reveals context bloat over time.
    """
    step_index: int
    prompt_tokens: int
    completion_tokens: int
    context_window_tokens: int
    latency_ms: float
    retries: int = 0
    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    step_type: str = "standard"  # "standard" | "tool_use" | "final"
    notes: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        return {
            "step_index": self.step_index,
            "step_type": self.step_type,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "context_window_tokens": self.context_window_tokens,
            "latency_ms": self.latency_ms,
            "retries": self.retries,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StepTrace":
        tool_calls = [ToolCallTrace.from_dict(tc) for tc in data.get("tool_calls", [])]
        return cls(
            step_index=data["step_index"],
            step_type=data.get("step_type", "standard"),
            prompt_tokens=data["prompt_tokens"],
            completion_tokens=data["completion_tokens"],
            context_window_tokens=data["context_window_tokens"],
            latency_ms=data["latency_ms"],
            retries=data.get("retries", 0),
            tool_calls=tool_calls,
            notes=data.get("notes"),
        )


@dataclass
class RunTrace:
    """Top-level record for one complete agent task run.

    run_id is a UUID generated at creation time. It uniquely identifies
    this run across all experiments and trials.

    agent_config stores whatever key/value pairs describe the agent that
    produced this run: model name, max steps, temperature, tool list, etc.
    Keeping it as a plain dict means the schema does not need to change
    when you add new config fields.

    start_time and end_time are Unix timestamps (seconds). Duration is
    derived, not stored, to avoid inconsistency.
    """
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_config: dict = field(default_factory=dict)
    task_description: str = ""
    steps: list[StepTrace] = field(default_factory=list)
    outcome: str = "unknown"  # "success" | "failure" | "partial" | "unknown"
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    notes: Optional[str] = None

    @property
    def total_duration_ms(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    @property
    def total_tokens(self) -> int:
        return sum(s.total_tokens for s in self.steps)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(s.prompt_tokens for s in self.steps)

    @property
    def total_completion_tokens(self) -> int:
        return sum(s.completion_tokens for s in self.steps)

    @property
    def total_retries(self) -> int:
        return sum(s.retries for s in self.steps)

    @property
    def total_tool_calls(self) -> int:
        return sum(len(s.tool_calls) for s in self.steps)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "agent_config": self.agent_config,
            "task_description": self.task_description,
            "outcome": self.outcome,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration_ms": self.total_duration_ms,
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_retries": self.total_retries,
            "total_tool_calls": self.total_tool_calls,
            "notes": self.notes,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RunTrace":
        steps = [StepTrace.from_dict(s) for s in data.get("steps", [])]
        run = cls(
            run_id=data["run_id"],
            agent_config=data.get("agent_config", {}),
            task_description=data.get("task_description", ""),
            outcome=data.get("outcome", "unknown"),
            start_time=data["start_time"],
            end_time=data.get("end_time"),
            notes=data.get("notes"),
        )
        run.steps = steps
        return run
