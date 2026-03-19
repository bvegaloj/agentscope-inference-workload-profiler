"""
Tests for agentscope/exporters/json_exporter.py

These tests write real files to a temporary directory provided by
pytest's tmp_path fixture. No mocking needed - the exporter is simple 
enough to test against the real file system

It checks
 - a file is created with the correct name
 - the saved content round-trips back to an equivalent RunTrace 
 - Loading a non-existent file raises FileNotFoundError
"""

import json
from pathlib import Path

import pytest

from agentscope.schema import RunTrace, StepTrace
from agentscope.exporters import json_exporter

def make_trace() -> RunTrace:
    run = RunTrace(
        task_description="Test task",
        agent_config={"model": "gpt-4o"},
        outcome="success",
    )
    run.steps.append(StepTrace(
        step_index=0,
        prompt_tokens=256,
        completion_tokens=64,
        context_window_tokens=256,
        latency_ms=500.0,
    ))
    run.end_time = run.start_time + 2.0
    return run

class TestJsonExporterSave:
    def test_file_is_created(self, tmp_path):
        trace = make_trace()
        filepath = json_exporter.save(trace, tmp_path, "baseline")
        assert filepath.exists()
    
    def test_filename_contains_experiment_name(self, tmp_path):
        trace = make_trace()
        filepath = json_exporter.save(trace, tmp_path, "baseline")
        assert "baseline" in filepath.name
        
    def test_filename_contains_run_id(self, tmp_path):
        trace = make_trace()
        filepath = json_exporter.save(trace, tmp_path, "baseline")
        assert trace.run_id in filepath.name
        
    def test_filename_is_json(self, tmp_path):
        trace = make_trace()
        filepath = json_exporter.save(trace, tmp_path, "baseline")
        assert filepath.suffix == ".json"
        
    def test_output_dir_is_created_if_missing(self, tmp_path):
        trace = make_trace()
        nested_dir = tmp_path / "experiments" / "traces"
        assert not nested_dir.exists()
        json_exporter.save(trace, nested_dir, "baseline")
        assert nested_dir.exists()
        
    def test_saved_file_is_valid_json(self, tmp_path):
        trace = make_trace()
        filepath = json_exporter.save(trace, tmp_path, "baseline")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f) # raises if file is not valid JSON
            assert isinstance(data, dict)
    
    def test_returns_path_object(self, tmp_path):
        trace = make_trace()
        result = json_exporter.save(trace, tmp_path, "baseline")
        assert isinstance(result, Path)
        
        
class TestJsonExporterLoad:
    def test_load_returns_run_trace(self, tmp_path):
        trace = make_trace()
        filepath = json_exporter.save(trace, tmp_path, "baseline")
        loaded = json_exporter.load(filepath)
        assert isinstance(loaded, RunTrace)
        
    def test_load_preserves_run_id(self, tmp_path):
        trace = make_trace()
        filepath = json_exporter.save(trace, tmp_path, "baseline")
        loaded = json_exporter.load(filepath)
        assert loaded.run_id == trace.run_id
        
    def test_load_preserves_outcome(self, tmp_path):
        trace = make_trace()
        filepath = json_exporter.save(trace, tmp_path, "baseline")
        loaded = json_exporter.load(filepath)
        assert loaded.outcome == trace.outcome
        
    def test_load_preserves_steps(self, tmp_path):
        trace = make_trace()
        filepath = json_exporter.save(trace, tmp_path, "baseline")
        loaded = json_exporter.load(filepath)
        assert len(loaded.steps) == 1
        assert loaded.steps[0].prompt_tokens == 256
        
    def test_load_accepts_string_path(self, tmp_path):
        trace = make_trace()
        filepath = json_exporter.save(trace, tmp_path, "baseline")
        loaded = json_exporter.load(str(filepath))
        assert loaded.run_id == trace.run_id
        
    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            json_exporter.load(tmp_path / "does_not_exist.json")