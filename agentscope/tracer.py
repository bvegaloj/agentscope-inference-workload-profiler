"""
tracer.py - Records agent behavior into schema objects at runtime.

The Tracer is a stateful observer. It sits alongside the agent and is
called by the agent (or runner) to report what happened at each step.
Timing is handled internally - the agent never touches a stopwatch.

The Tracer knows nothing about how to save traces. That is the
exporter's job. When the run is complete, call get_trace() and pass
the result to an exporter.

Usage pattern:

    tracer = Tracer()
    tracer.start_run(config={"model": "gpt-4o"}, task="summarize document")

    for i in range(max_steps):
        tracer.start_step(i)

        result = provider.complete(prompt)           # agent calls LLM

        if agent_used_a_tool:
            tracer.record_tool_call(ToolCallTrace(...))

        tracer.end_step(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            context_window_tokens=result.context_window_tokens,
        )

        if done:
            break

    tracer.end_run(outcome="success")
    trace = tracer.get_trace()
"""

import time
from typing import Optional

from agentscope.schema import RunTrace, StepTrace, ToolCallTrace


class Tracer:
    """
    Stateful recorder for a single agent run.

    One Tracer instance = one run. Do not reuse a Tracer across runs.
    Instantiate a fresh Tracer for each experiment trial.
    """

    def __init__(self) -> None:
        self._run: Optional[RunTrace] = None
        self._step_start: Optional[float] = None   # perf_counter timestamp
        self._step_index: Optional[int] = None
        self._step_type: str = "standard"
        self._step_tool_calls: list[ToolCallTrace] = []

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(self, config: dict, task_description: str = "") -> None:
        """Begin a new run. Must be called before any step methods."""
        if self._run is not None:
            raise RuntimeError(
                "A run is already active. Use a new Tracer instance for each run."
            )
        self._run = RunTrace(
            agent_config=config,
            task_description=task_description,
        )

    def end_run(self, outcome: str = "unknown", notes: Optional[str] = None) -> None:
        """Finalize the run. Must be called after all steps are closed."""
        self._require_active_run("end_run")
        if self._step_start is not None:
            raise RuntimeError(
                f"Step {self._step_index} is still open. "
                "Call end_step() before end_run()."
            )
        self._run.end_time = time.time()
        self._run.outcome = outcome
        self._run.notes = notes

    # ------------------------------------------------------------------
    # Step lifecycle
    # ------------------------------------------------------------------

    def start_step(self, step_index: int, step_type: str = "standard") -> None:
        """
        Open a new step and start the latency timer.

        step_type should be one of: "standard", "tool_use", "final".
        The agent decides which label is appropriate.
        """
        self._require_active_run("start_step")
        if self._step_start is not None:
            raise RuntimeError(
                f"Step {self._step_index} is still open. "
                "Call end_step() before starting a new step."
            )
        self._step_index = step_index
        self._step_type = step_type
        self._step_tool_calls = []
        self._step_start = time.perf_counter()   # high-resolution timer for latency

    def end_step(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        context_window_tokens: int,
        retries: int = 0,
        notes: Optional[str] = None,
    ) -> None:
        """
        Close the current step.

        Computes latency automatically from when start_step() was called.
        Token counts must come from the LLM provider response.

        context_window_tokens is the total tokens in the context at this
        step, including all prior history. Pass the actual value from the
        provider, not just prompt_tokens - they diverge as the run progresses.
        """
        self._require_active_run("end_step")
        if self._step_start is None:
            raise RuntimeError("No step is open. Call start_step() first.")

        latency_ms = (time.perf_counter() - self._step_start) * 1000

        step = StepTrace(
            step_index=self._step_index,
            step_type=self._step_type,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            context_window_tokens=context_window_tokens,
            latency_ms=latency_ms,
            retries=retries,
            tool_calls=self._step_tool_calls,
            notes=notes,
        )
        self._run.steps.append(step)

        # Reset step state
        self._step_start = None
        self._step_index = None
        self._step_type = "standard"
        self._step_tool_calls = []

    # ------------------------------------------------------------------
    # Tool call recording
    # ------------------------------------------------------------------

    def record_tool_call(self, tool_call: ToolCallTrace) -> None:
        """
        Attach a tool call to the currently open step.

        The agent is responsible for constructing the ToolCallTrace,
        including measuring the tool's own latency. The tracer only
        stores it.
        """
        self._require_active_run("record_tool_call")
        if self._step_start is None:
            raise RuntimeError(
                "No step is open. Call start_step() before recording tool calls."
            )
        self._step_tool_calls.append(tool_call)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def get_trace(self) -> RunTrace:
        """
        Return the completed RunTrace.

        Only valid after end_run() has been called.
        """
        self._require_active_run("get_trace")
        if self._step_start is not None:
            raise RuntimeError(
                "A step is still open. Call end_step() before get_trace()."
            )
        if self._run.end_time is None:
            raise RuntimeError(
                "Run has not been ended. Call end_run() before get_trace()."
            )
        return self._run

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_active_run(self, caller: str) -> None:
        if self._run is None:
            raise RuntimeError(
                f"{caller}() called with no active run. Call start_run() first."
            )
