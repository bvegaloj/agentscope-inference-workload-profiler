"""
Tests for agentscope/schema.py.

These tests do three things:
1. Verify that schema objects can be constructed correctly.
2. Verify that derived properties compute the right values.
3. Verify that round-tripping through to_dict() / from_dict()
   produces an equivalent object.

If any of these fail, the schema is broken before anything else is built.
"""

from agentscope.schema import ToolCallTrace, StepTrace, RunTrace


# ---------------------------------------------------------------------------
# Fixtures: reusable schema objects
# ---------------------------------------------------------------------------

def make_tool_call(success: bool = True) -> ToolCallTrace:
    return ToolCallTrace(
        tool_name="web_search",
        input_summary="query: climate change 2025",
        output_summary="3 results returned",
        success=success,
        latency_ms=240.5,
        error_message=None if success else "connection timeout",
    )


def make_step(step_index: int = 0, retries: int = 0, include_tool: bool = False) -> StepTrace:
    tool_calls = [make_tool_call()] if include_tool else []
    return StepTrace(
        step_index=step_index,
        prompt_tokens=512,
        completion_tokens=128,
        context_window_tokens=512 + step_index * 128,  # grows each step
        latency_ms=1100.0,
        retries=retries,
        tool_calls=tool_calls,
        step_type="tool_use" if include_tool else "standard",
    )


def make_run(num_steps: int = 2) -> RunTrace:
    run = RunTrace(
        task_description="Research and summarize climate policy",
        agent_config={"model": "gpt-4o", "max_steps": 10},
        outcome="success",
    )
    for i in range(num_steps):
        run.steps.append(make_step(step_index=i, retries=i, include_tool=(i % 2 == 0)))
    run.end_time = run.start_time + 5.0
    return run


# ---------------------------------------------------------------------------
# ToolCallTrace tests
# ---------------------------------------------------------------------------

class TestToolCallTrace:
    def test_construction(self):
        tc = make_tool_call()
        assert tc.tool_name == "web_search"
        assert tc.success is True
        assert tc.error_message is None

    def test_failed_tool_call_has_error_message(self):
        tc = make_tool_call(success=False)
        assert tc.success is False
        assert tc.error_message == "connection timeout"

    def test_round_trip(self):
        tc = make_tool_call()
        restored = ToolCallTrace.from_dict(tc.to_dict())
        assert restored.tool_name == tc.tool_name
        assert restored.success == tc.success
        assert restored.latency_ms == tc.latency_ms
        assert restored.error_message == tc.error_message


# ---------------------------------------------------------------------------
# StepTrace tests
# ---------------------------------------------------------------------------

class TestStepTrace:
    def test_total_tokens_is_sum(self):
        step = make_step()
        assert step.total_tokens == step.prompt_tokens + step.completion_tokens
        assert step.total_tokens == 640

    def test_tool_calls_default_is_empty_list(self):
        step = make_step()
        assert step.tool_calls == []
        # Verify it is not the same list object across instances (mutable default bug)
        step2 = make_step()
        step.tool_calls.append(make_tool_call())
        assert step2.tool_calls == []

    def test_round_trip_without_tools(self):
        step = make_step()
        restored = StepTrace.from_dict(step.to_dict())
        assert restored.step_index == step.step_index
        assert restored.prompt_tokens == step.prompt_tokens
        assert restored.completion_tokens == step.completion_tokens
        assert restored.retries == step.retries
        assert restored.tool_calls == []

    def test_round_trip_with_tools(self):
        step = make_step(include_tool=True)
        restored = StepTrace.from_dict(step.to_dict())
        assert len(restored.tool_calls) == 1
        assert restored.tool_calls[0].tool_name == "web_search"

    def test_to_dict_includes_total_tokens(self):
        step = make_step()
        d = step.to_dict()
        assert "total_tokens" in d
        assert d["total_tokens"] == 640


# ---------------------------------------------------------------------------
# RunTrace tests
# ---------------------------------------------------------------------------

class TestRunTrace:
    def test_run_id_is_unique(self):
        run1 = RunTrace()
        run2 = RunTrace()
        assert run1.run_id != run2.run_id

    def test_total_duration_ms(self):
        run = make_run()
        assert run.total_duration_ms == 5000.0

    def test_total_duration_none_before_end(self):
        run = RunTrace()
        assert run.total_duration_ms is None

    def test_aggregated_token_counts(self):
        run = make_run(num_steps=2)
        # Each step: prompt=512, completion=128
        assert run.total_prompt_tokens == 1024
        assert run.total_completion_tokens == 256
        assert run.total_tokens == 1280

    def test_total_retries(self):
        run = make_run(num_steps=2)
        # step 0 retries=0, step 1 retries=1
        assert run.total_retries == 1

    def test_total_tool_calls(self):
        run = make_run(num_steps=2)
        # step 0 has a tool (index 0 is even), step 1 does not (index 1 is odd)
        assert run.total_tool_calls == 1

    def test_steps_default_is_empty_list(self):
        run1 = RunTrace()
        run2 = RunTrace()
        run1.steps.append(make_step())
        assert run2.steps == []

    def test_round_trip(self):
        original = make_run(num_steps=2)
        restored = RunTrace.from_dict(original.to_dict())

        assert restored.run_id == original.run_id
        assert restored.outcome == original.outcome
        assert restored.task_description == original.task_description
        assert restored.agent_config == original.agent_config
        assert restored.start_time == original.start_time
        assert restored.end_time == original.end_time
        assert len(restored.steps) == len(original.steps)

    def test_round_trip_preserves_step_details(self):
        original = make_run(num_steps=2)
        restored = RunTrace.from_dict(original.to_dict())
        for i, (orig_step, rest_step) in enumerate(zip(original.steps, restored.steps)):
            assert rest_step.step_index == orig_step.step_index
            assert rest_step.prompt_tokens == orig_step.prompt_tokens
            assert rest_step.retries == orig_step.retries
            assert len(rest_step.tool_calls) == len(orig_step.tool_calls)
