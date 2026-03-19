"""
Tests for agentscope/tracer.py.

Tests are organized into three groups:
1. Happy path - normal usage produces correct RunTrace objects.
2. Latency - the tracer measures time automatically and non-negatively.
3. Error cases - calling methods out of order raises clear RuntimeErrors.

We do not mock time because the latency tests only assert >= 0. Asserting
exact values would require mocking and would make the tests fragile.
"""

import pytest
from agentscope.tracer import Tracer
from agentscope.schema import ToolCallTrace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONFIG = {"model": "gpt-4o", "max_steps": 5}
TASK = "Summarize recent AI research papers"


def run_single_step(tracer: Tracer, step_index: int = 0, retries: int = 0) -> None:
    """Drive the tracer through one complete step."""
    tracer.start_step(step_index)
    tracer.end_step(
        prompt_tokens=512,
        completion_tokens=128,
        context_window_tokens=512 + step_index * 128,
        retries=retries,
    )


def make_tool_call() -> ToolCallTrace:
    return ToolCallTrace(
        tool_name="web_search",
        input_summary="query: AI 2025",
        output_summary="5 results returned",
        success=True,
        latency_ms=180.0,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestTracerHappyPath:
    def test_single_step_produces_run_trace(self):
        tracer = Tracer()
        tracer.start_run(CONFIG, TASK)
        run_single_step(tracer)
        tracer.end_run(outcome="success")
        trace = tracer.get_trace()

        assert trace.outcome == "success"
        assert trace.task_description == TASK
        assert trace.agent_config == CONFIG
        assert len(trace.steps) == 1

    def test_step_tokens_are_recorded(self):
        tracer = Tracer()
        tracer.start_run(CONFIG, TASK)
        run_single_step(tracer, step_index=0)
        tracer.end_run(outcome="success")
        trace = tracer.get_trace()

        step = trace.steps[0]
        assert step.prompt_tokens == 512
        assert step.completion_tokens == 128
        assert step.total_tokens == 640

    def test_multiple_steps_all_recorded(self):
        tracer = Tracer()
        tracer.start_run(CONFIG, TASK)
        for i in range(4):
            run_single_step(tracer, step_index=i)
        tracer.end_run(outcome="success")
        trace = tracer.get_trace()

        assert len(trace.steps) == 4
        for i, step in enumerate(trace.steps):
            assert step.step_index == i

    def test_step_indices_preserved_in_order(self):
        tracer = Tracer()
        tracer.start_run(CONFIG, TASK)
        for i in range(3):
            run_single_step(tracer, step_index=i)
        tracer.end_run(outcome="success")
        trace = tracer.get_trace()

        indices = [s.step_index for s in trace.steps]
        assert indices == [0, 1, 2]

    def test_retries_are_passed_through(self):
        tracer = Tracer()
        tracer.start_run(CONFIG, TASK)
        run_single_step(tracer, step_index=0, retries=2)
        tracer.end_run(outcome="success")
        trace = tracer.get_trace()

        assert trace.steps[0].retries == 2
        assert trace.total_retries == 2

    def test_run_id_is_set(self):
        tracer = Tracer()
        tracer.start_run(CONFIG, TASK)
        run_single_step(tracer)
        tracer.end_run(outcome="success")
        trace = tracer.get_trace()
        assert trace.run_id is not None and len(trace.run_id) > 0

    def test_end_time_is_set_after_end_run(self):
        tracer = Tracer()
        tracer.start_run(CONFIG, TASK)
        run_single_step(tracer)
        tracer.end_run(outcome="success")
        trace = tracer.get_trace()
        assert trace.end_time is not None
        assert trace.total_duration_ms is not None
        assert trace.total_duration_ms >= 0

    def test_outcome_is_recorded(self):
        for outcome in ("success", "failure", "partial", "unknown"):
            tracer = Tracer()
            tracer.start_run(CONFIG, TASK)
            run_single_step(tracer)
            tracer.end_run(outcome=outcome)
            assert tracer.get_trace().outcome == outcome

    def test_notes_on_run(self):
        tracer = Tracer()
        tracer.start_run(CONFIG, TASK)
        run_single_step(tracer)
        tracer.end_run(outcome="success", notes="ran out of context at step 10")
        assert tracer.get_trace().notes == "ran out of context at step 10"


# ---------------------------------------------------------------------------
# Tool calls
# ---------------------------------------------------------------------------

class TestTracerToolCalls:
    def test_tool_call_attached_to_step(self):
        tracer = Tracer()
        tracer.start_run(CONFIG, TASK)
        tracer.start_step(0, step_type="tool_use")
        tracer.record_tool_call(make_tool_call())
        tracer.end_step(prompt_tokens=512, completion_tokens=64, context_window_tokens=512)
        tracer.end_run(outcome="success")
        trace = tracer.get_trace()

        assert len(trace.steps[0].tool_calls) == 1
        assert trace.steps[0].tool_calls[0].tool_name == "web_search"

    def test_multiple_tool_calls_in_one_step(self):
        tracer = Tracer()
        tracer.start_run(CONFIG, TASK)
        tracer.start_step(0, step_type="tool_use")
        tracer.record_tool_call(make_tool_call())
        tracer.record_tool_call(make_tool_call())
        tracer.end_step(prompt_tokens=512, completion_tokens=64, context_window_tokens=512)
        tracer.end_run(outcome="success")
        trace = tracer.get_trace()

        assert len(trace.steps[0].tool_calls) == 2

    def test_tool_calls_not_bleed_between_steps(self):
        """Tool calls from step 0 must not appear in step 1."""
        tracer = Tracer()
        tracer.start_run(CONFIG, TASK)

        tracer.start_step(0, step_type="tool_use")
        tracer.record_tool_call(make_tool_call())
        tracer.end_step(prompt_tokens=512, completion_tokens=64, context_window_tokens=512)

        tracer.start_step(1)
        tracer.end_step(prompt_tokens=640, completion_tokens=64, context_window_tokens=640)

        tracer.end_run(outcome="success")
        trace = tracer.get_trace()

        assert len(trace.steps[0].tool_calls) == 1
        assert len(trace.steps[1].tool_calls) == 0


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------

class TestTracerLatency:
    def test_step_latency_is_non_negative(self):
        tracer = Tracer()
        tracer.start_run(CONFIG, TASK)
        run_single_step(tracer)
        tracer.end_run(outcome="success")
        trace = tracer.get_trace()
        assert trace.steps[0].latency_ms >= 0

    def test_latency_recorded_per_step(self):
        tracer = Tracer()
        tracer.start_run(CONFIG, TASK)
        for i in range(3):
            run_single_step(tracer, step_index=i)
        tracer.end_run(outcome="success")
        trace = tracer.get_trace()
        for step in trace.steps:
            assert step.latency_ms >= 0


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestTracerErrors:
    def test_start_run_twice_raises(self):
        tracer = Tracer()
        tracer.start_run(CONFIG, TASK)
        with pytest.raises(RuntimeError, match="already active"):
            tracer.start_run(CONFIG, TASK)

    def test_end_run_without_start_raises(self):
        tracer = Tracer()
        with pytest.raises(RuntimeError, match="no active run"):
            tracer.end_run()

    def test_start_step_without_start_run_raises(self):
        tracer = Tracer()
        with pytest.raises(RuntimeError, match="no active run"):
            tracer.start_step(0)

    def test_end_step_without_start_step_raises(self):
        tracer = Tracer()
        tracer.start_run(CONFIG, TASK)
        with pytest.raises(RuntimeError, match="No step is open"):
            tracer.end_step(prompt_tokens=1, completion_tokens=1, context_window_tokens=1)

    def test_start_step_twice_without_end_raises(self):
        tracer = Tracer()
        tracer.start_run(CONFIG, TASK)
        tracer.start_step(0)
        with pytest.raises(RuntimeError, match="still open"):
            tracer.start_step(1)

    def test_end_run_with_open_step_raises(self):
        tracer = Tracer()
        tracer.start_run(CONFIG, TASK)
        tracer.start_step(0)
        with pytest.raises(RuntimeError, match="still open"):
            tracer.end_run(outcome="success")

    def test_get_trace_before_end_run_raises(self):
        tracer = Tracer()
        tracer.start_run(CONFIG, TASK)
        run_single_step(tracer)
        with pytest.raises(RuntimeError, match="not been ended"):
            tracer.get_trace()

    def test_record_tool_call_without_open_step_raises(self):
        tracer = Tracer()
        tracer.start_run(CONFIG, TASK)
        with pytest.raises(RuntimeError, match="No step is open"):
            tracer.record_tool_call(make_tool_call())
