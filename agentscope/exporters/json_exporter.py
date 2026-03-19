"""
json_exporter.py - Saves and loads RunTrace objects as JSON

The exporter knows about the file system and the schema.
It knows nothing about the tracer, runner, or agents.

Two responsibilities:
    save(trace, output_dir, experiment_name) -> path of the written file
    load(filepath) -> RunTrace
"""

import json
from pathlib import Path

from agentscope.schema import RunTrace

def save(trace: RunTrace, output_dir: str | Path, experiment_name: str) -> Path:
    """
    Write a RunTrace to a JSON file
    
    The file is named: {experiment_name}_{run_id}.json
    Output directory is created if does not exist
    
    Returns the Path of the written file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename= f"{experiment_name}_{trace.run_id}.json"
    filepath = output_dir / filename
    
    with open (filepath, "w", encoding="utf-8") as f:
        json.dump(trace.to_dict(), f, indent=2)
        
    return filepath

def load(filepath: str | Path) -> RunTrace:
    """
    Read a JSON file and return a RunTrace object
    
    Raises FileNotFoundError if the file does not exist"""
    filepath = Path(filepath)
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    return RunTrace.from_dict(data)