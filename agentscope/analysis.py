"""
analysis.py = Loads saved traces and computes statistics.

This module knows about:
- the schema (RunTrace, StepTrace) - reads fields and properties
- the exporter's load() function - to deserialize traces from disk
- the file system - to find trace files in a directory

it knows nothing about the tracer, runner, or agents
"""

import statistics
from pathlib import Path

from agentscope.exporters import json_exporter
from agentscope.schema import RunTrace

def load_traces(traces_dir: str | Path, experiment_name: str) -> list[RunTrace]:
    """
    Load all trace files for a given experiment from a directory.
    
    Looks for files matching: {experiment_name}_*.json
    Return an empty list if no matching files are found.
    """
    traces_dir = Path(traces_dir)
    pattern = f"{experiment_name}_*.json"
    paths = sorted(traces_dir.glob(pattern))
    return [json_exporter.load(p) for p in paths]

def summarize_runs(traces: list[RunTrace]) -> dict:
    """
    Compute aggrefate statistics across a list of runs
    
    Returns a dic with:
    
    num_runs        - total number of runs
    success_rate    - fraction of runs with outcome "success"
    avg_tokens      - mean total tokens per run
    avg_prompt_tokens
    avg_completion_tokens
    avg_duration_ms - mean wall-clock duration per run
    avg_steps       - mean number of steps per run
    avg_retries     - mean total retries per run
    avg_tool_calls  - mean total tool calls per run
    """
    if not traces:
        return {}
    
    num_runs = len(traces)
    successes = sum(1 for t in traces if t.outcome == "success")
    
    total_tokens = [t.total_tokens for t in traces]
    prompt_tokens = [t.total_prompt_tokens for t in traces]
    completion_tokens = [t.total_completion_tokens for t in traces]
    durations = [t.total_duration_ms for t in traces if t.total_duration_ms is not None]
    steps = [len(t.steps) for t in traces]
    retries = [t.total_retries for t in traces]
    tool_calls = [t.total_tool_calls for t in traces]
    
    return {
        "num_runs": num_runs,
        "success_rate": successes / num_runs,
        "avg_tokens": statistics.mean(total_tokens),
        "avg_prompt_tokens": statistics.mean(prompt_tokens),
        "avg_completion_tokens": statistics.mean(completion_tokens),
        "avg_duration_ms": statistics.mean(durations) if durations else None,
        "avg_steps": statistics.mean(steps),
        "avg_retries": statistics.mean(retries),
        "avg_tool_calls": statistics.mean(tool_calls),
    }
    
def context_growth(traces: list[RunTrace]) -> list[dict]:
    """
    Compute average prompt tokens per step index across all runs
    
    This reveals context bloat: how much the prompt grows with each step
    as prior history accumulates in the context window
    
    Returns a list of dicts, one per step index
        step_index      - the step number (0-based)
        avg_prompt_tokens
        num_runs        - how many runs reached this step 
    """
    if not traces:
        return []
    
    # Group prompt_tokens by step_index across all runs
    step_tokens: dict[int, list[int]] = {}
    for trace in traces:
        for step in trace.steps:
            step_tokens.setdefault(step.step_index, []).append(step.prompt_tokens)
            
    return [
        {
                "step_index": idx,
                "avg_prompt_tokens": statistics.mean(tokens),
                "num_runs": len(tokens),
        }
        for idx, tokens in sorted(step_tokens.items())
    ]
    
def estimate_cost(traces: list[RunTrace], prompt_price_per_1k: float, completion_price_per_1k: float) -> dict:
    """
    Estimate the total and average cost of a set of runs
    
    Prices are in dollars per 1000 tokens (the standard unit LLM
    providers use)
    
    Returns a dict
        total_cost_usd    - total cost across all runs
        avg_cost_usd      - average cost per run
        cost_per_run      - list of individual run costs, in order
    """ 
    if not traces:
        return {}
    
    def run_cost(trace: RunTrace) -> float:
        prompt_cost = (trace.total_prompt_tokens / 1000) * prompt_price_per_1k
        completion_cost = (trace.total_completion_tokens / 1000) * completion_price_per_1k
        return prompt_cost + completion_cost
    
    costs = [run_cost(t) for t in traces]
    
    return {
        "total_cost_usd": sum(costs),
        "avg_cost_usd": statistics.mean(costs),
        "cost_per_run": costs,
    }