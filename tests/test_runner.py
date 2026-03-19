"""
Tests for agentscope/runner.py

The runner is the integration point between the agent, tracer, and
exporter. These tests use the SyntheticAgent so no API key is needed

It checks: 
- Correct number of trace files is produced per experiment
- Each trial produces an independent trace with a unique run_id
- Agent outcomes are recorder correctly
- An agent that raises an exception does not crash the runner
"""

from pathlib import Path

import pytest

from agentscope.agents.synthetic_agent import SyntheticAgent
from agentscope.exporters import json_exporter
from agentscope.runner import run_experiment

CONFIG = {"model": "synthetic", "max_steps": 5}
TASK = "Summarize recent AI papers"


class TestRunExperimentOutputs:
    def test_returns_one_path_per_trial(self, tmp_path):
        agent = SyntheticAgent(num_steps=2)
        paths = run_experiment(agent, CONFIG, TASK, "baseline", tmp_path, num_trials=3)
        assert len(paths) == 3
        
    def test_all_returned_paths_exist(self, tmp_path):
        agent = SyntheticAgent(num_steps=2)
        paths = run_experiment(agent, CONFIG, TASK, "baseline", tmp_path, num_trials=2)
        for path in paths:
            assert path.exists()
            
    def test_trace_files_contain_experiment_name(self, tmp_path):
        agent = SyntheticAgent(num_steps=2)
        paths = run_experiment(agent, CONFIG, TASK, "my_experiment", tmp_path, num_trials=2)
        for path in paths:
            assert "my_experiment" in path.name
            
    def test_each_trial_has_unique_run_id(self, tmp_path):
        agent = SyntheticAgent(num_steps=2)
        paths = run_experiment(agent, CONFIG, TASK, "baseline", tmp_path, num_trials=3)
        traces = [json_exporter.load(p) for p in paths]
        run_ids = [t.run_id for t in traces]
        assert len(set(run_ids)) == 3
        
class TestRunExperimentTraceContent:
    def test_successful_agent_produces_success_outcome(self, tmp_path):
        agent = SyntheticAgent(num_steps=3)
        paths = run_experiment(agent, CONFIG, TASK, "baseline", tmp_path)
        trace = json_exporter.load(paths[0])
        assert trace.outcome == "success"

    def test_failing_agent_produces_failure_outcome(self, tmp_path):
        agent = SyntheticAgent(num_steps=5, fail_at_step=1)
        paths = run_experiment(agent, CONFIG, TASK, "baseline", tmp_path)
        trace = json_exporter.load(paths[0])
        assert trace.outcome == "failure"

    def test_failing_agent_records_steps_up_to_failure(self, tmp_path):
        agent = SyntheticAgent(num_steps=5, fail_at_step=1)
        paths = run_experiment(agent, CONFIG, TASK, "baseline", tmp_path)
        trace = json_exporter.load(paths[0])
        # fail_at_step=1 means steps 0 and 1 are recorded, then the loop exits
        assert len(trace.steps) == 2

    def test_trace_has_correct_step_count(self, tmp_path):
        agent = SyntheticAgent(num_steps=4)
        paths = run_experiment(agent, CONFIG, TASK, "baseline", tmp_path)
        trace = json_exporter.load(paths[0])
        assert len(trace.steps) == 4

    def test_agent_config_is_recorded(self, tmp_path):
        agent = SyntheticAgent(num_steps=2)
        paths = run_experiment(agent, CONFIG, TASK, "baseline", tmp_path)
        trace = json_exporter.load(paths[0])
        assert trace.agent_config == CONFIG

    def test_task_description_is_recorded(self, tmp_path):
        agent = SyntheticAgent(num_steps=2)
        paths = run_experiment(agent, CONFIG, TASK, "baseline", tmp_path)
        trace = json_exporter.load(paths[0])
        assert trace.task_description == TASK

    def test_end_time_is_set(self, tmp_path):
        agent = SyntheticAgent(num_steps=2)
        paths = run_experiment(agent, CONFIG, TASK, "baseline", tmp_path)
        trace = json_exporter.load(paths[0])
        assert trace.end_time is not None
        assert trace.total_duration_ms > 0

class TestRunExperimentResilience:
    def test_crashing_agent_does_not_stop_runner(self, tmp_path):
        """An agent that raises must not prevent subsequent trials"""
        
        class CrashingAgent(SyntheticAgent):
            def run(self, task, tracer):
                tracer.start_step(0)
                tracer.end_step(
                    prompt_tokens=100,
                    completion_tokens=50,
                    context_window_tokens=100,
                )
                raise RuntimeError("LLM API timed out")
            
        agent = CrashingAgent(num_steps=1)
        paths = run_experiment(agent, CONFIG, TASK, "baseline", tmp_path, num_trials=3)
        assert len(paths) == 3
    
    def test_crashing_agent_produces_failure_outcome(self, tmp_path):
            class CrashingAgent(SyntheticAgent):
                def run(self, task, tracer):
                    tracer.start_step(0)
                    tracer.end_step(
                        prompt_tokens=100,
                        completion_tokens=50,
                        context_window_tokens=100,
                    )
                    raise RuntimeError("LLM API timed out")
                
            agent = CrashingAgent(num_steps=1)
            paths = run_experiment(agent, CONFIG, TASK, "baseline", tmp_path)
            trace = json_exporter.load(paths[0])
            assert trace.outcome == "failure"
            
    def test_crashing_agent_records_exception_in_notes(self, tmp_path):
            class CrashingAgent(SyntheticAgent):
                def run(self, task, tracer):
                    tracer.start_step(0)
                    tracer.end_step(
                        prompt_tokens=100,
                        completion_tokens=50,
                        context_window_tokens=100,
                    )
                    raise RuntimeError("LLM API timed out")
                
            agent = CrashingAgent(num_steps=1)
            paths = run_experiment(agent, CONFIG, TASK, "baseline", tmp_path)
            trace = json_exporter.load(paths[0])
            assert trace.notes is not None
            assert "LLM API timed out" in trace.notes