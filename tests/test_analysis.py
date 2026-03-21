"""
Tests for agentscore/analysis.py

These tests build trace data programmatically rather than running the full pipeline.
This keeps the tests fast and focust. Each test controls
exactly what data it feeds in, so failures are easy to diagnose. We can also easily test edge cases like empty runs or steps with no tools.
"""

import pytest
from agentscope.schema import RunTrace, StepTrace
from agentscope.exporters import json_exporter
from agentscope import analysis

# ------------------------
# Helpers

def make_run(
    outcome: str = "success",
    num_steps: int = 3,
    prompt_tokens_start: int = 512,
    completion_tokens: int = 128,
    retries_per_step: int = 0,
) -> RunTrace:
    """
    Build a RunTrace with predictable, controlled values
    Context grows each step to simulate real agent behavior
    """
    run = RunTrace(
        task_description="test task",
        agent_config={"model": "synthetic"},
        outcome=outcome,
    )
    for i in range(num_steps):
        prompt = prompt_tokens_start + i * completion_tokens
        run.steps.append(StepTrace(
            step_index=i,
            prompt_tokens=prompt,
            completion_tokens=completion_tokens,
            context_window_tokens=prompt + completion_tokens,
            latency_ms=100.0 + i * 50,
            retries=retries_per_step,
        ))
    run.end_time = run.start_time + 3.0
    return run
    
#3------------------------
# load_traces

class TestLoadTraces:
    def test_loads_matching_files(self, tmp_path):
        run = make_run()
        json_exporter.save(run, tmp_path, "baseline")
        traces = analysis.load_traces(tmp_path, "baseline")
        assert len(traces) == 1

    def test_loads_multiple_files(self, tmp_path):
        for _ in range(3):
            json_exporter.save(make_run(), tmp_path, "baseline")
        traces = analysis.load_traces(tmp_path, "baseline")
        assert len(traces) == 3

    def test_ignores_other_experiments(self, tmp_path):
        json_exporter.save(make_run(), tmp_path, "baseline")
        json_exporter.save(make_run(), tmp_path, "high_temp")
        traces = analysis.load_traces(tmp_path, "baseline")
        assert len(traces) == 1

    def test_returns_empty_list_when_no_files(self, tmp_path):
        traces = analysis.load_traces(tmp_path, "baseline")
        assert traces == []

    def test_returns_run_trace_objects(self, tmp_path):
        json_exporter.save(make_run(), tmp_path, "baseline")
        traces = analysis.load_traces(tmp_path, "baseline")
        assert isinstance(traces[0], RunTrace)

#------------------------
# summarize_runs

class TestSummarizeRuns:
    def test_empty_input_returns_empty_dict(self):
        assert analysis.summarize_runs([]) == {}

    def test_num_runs(self):
        traces = [make_run() for _ in range(4)]
        result = analysis.summarize_runs(traces)
        assert result["num_runs"] == 4

    def test_success_rate_all_success(self):
        traces = [make_run(outcome="success") for _ in range(3)]
        result = analysis.summarize_runs(traces)
        assert result["success_rate"] == 1.0

    def test_success_rate_all_failure(self):
        traces = [make_run(outcome="failure") for _ in range(3)]
        result = analysis.summarize_runs(traces)
        assert result["success_rate"] == 0.0

    def test_success_rate_mixed(self):
        traces = [make_run(outcome="success")] * 2 + [make_run(outcome="failure")]
        result = analysis.summarize_runs(traces)
        assert abs(result["success_rate"] - 2/3) < 1e-9

    def test_avg_tokens(self):
        # 3 steps: prompt 512+0*128=512, 512+1*128=640, 512+2*128=768
        # completion 128 each
        # total tokens = (512+128) + (640+128) + (768+128) = 640+768+896 = 2304
        run = make_run(num_steps=3, prompt_tokens_start=512, completion_tokens=128)
        result = analysis.summarize_runs([run])
        assert result["avg_tokens"] == 2304

    def test_avg_steps(self):
        traces = [make_run(num_steps=4), make_run(num_steps=2)]
        result = analysis.summarize_runs(traces)
        assert result["avg_steps"] == 3.0

    def test_avg_retries(self):
        traces = [make_run(retries_per_step=0), make_run(retries_per_step=2, num_steps=3)]
        result = analysis.summarize_runs(traces)
        # run 0: 0 retries, run 1: 3*2=6 retries, avg = 3.0
        assert result["avg_retries"] == 3.0

    def test_avg_duration_ms(self):
        run = make_run()
        result = analysis.summarize_runs([run])
        assert result["avg_duration_ms"] == pytest.approx(3000.0, abs=1.0)
        
# ------------------------
# context_growth

class TestContextGrowth:        
    def test_empty_input_returns_empty_list(self):
        assert analysis.context_growth([]) == []

    def test_step_indices_in_order(self):
        run = make_run(num_steps=4)
        result = analysis.context_growth([run])
        indices = [r["step_index"] for r in result]
        assert indices == [0, 1, 2, 3]

    def test_prompt_tokens_grow_across_steps(self):
        run = make_run(num_steps=3, prompt_tokens_start=512, completion_tokens=128)
        result = analysis.context_growth([run])
        # step 0: 512, step 1: 640, step 2: 768
        assert result[0]["avg_prompt_tokens"] == 512
        assert result[1]["avg_prompt_tokens"] == 640
        assert result[2]["avg_prompt_tokens"] == 768

    def test_num_runs_reflects_how_many_reached_each_step(self):
        # run A has 3 steps, run B has 2 steps
        # step 0 and 1 seen by both, step 2 seen by only run A
        run_a = make_run(num_steps=3)
        run_b = make_run(num_steps=2)
        result = analysis.context_growth([run_a, run_b])
        assert result[0]["num_runs"] == 2
        assert result[1]["num_runs"] == 2
        assert result[2]["num_runs"] == 1

    def test_avg_across_multiple_runs(self):
        run_a = make_run(num_steps=1, prompt_tokens_start=400, completion_tokens=100)
        run_b = make_run(num_steps=1, prompt_tokens_start=600, completion_tokens=100)
        result = analysis.context_growth([run_a, run_b])
        # step 0: (400 + 600) / 2 = 500
        assert result[0]["avg_prompt_tokens"] == 500.0
        
# --------------------
# estimate_cost

class TestEstimateCost:
    def test_empty_input_returns_empty_dict(self):
        assert analysis.estimate_cost([], 0.005, 0.015) == {}

    def test_total_cost_single_run(self):
        # 1 step: prompt=1000, completion=1000
        run = make_run(num_steps=1, prompt_tokens_start=1000, completion_tokens=1000)
        result = analysis.estimate_cost([run], prompt_price_per_1k=1.0, completion_price_per_1k=2.0)
        # prompt: (1000/1000)*1.0 = 1.0
        # completion: (1000/1000)*2.0 = 2.0
        # total: 3.0
        assert result["total_cost_usd"] == pytest.approx(3.0)

    def test_avg_cost_equals_total_for_single_run(self):
        run = make_run(num_steps=1, prompt_tokens_start=1000, completion_tokens=1000)
        result = analysis.estimate_cost([run], prompt_price_per_1k=1.0, completion_price_per_1k=2.0)
        assert result["avg_cost_usd"] == result["total_cost_usd"]

    def test_cost_per_run_length(self):
        traces = [make_run() for _ in range(3)]
        result = analysis.estimate_cost(traces, 0.005, 0.015)
        assert len(result["cost_per_run"]) == 3

    def test_zero_price_gives_zero_cost(self):
        run = make_run(num_steps=3)
        result = analysis.estimate_cost([run], 0.0, 0.0)
        assert result["total_cost_usd"] == 0.0
        