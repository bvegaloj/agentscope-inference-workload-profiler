"""
runner.py - Executes an agent configuration multiple times and saves traces

The runner is the onlu component that knows about all the others
the agent, tracer, and exporter. Its job is to wire them together and run N trials of a given configuration

Each trial produces one trace file. Trials are independent.
a failure in one trial does not stop the others.
"""

from pathlib import Path

from agentscope.agents.base_agent import BaseAgent
from agentscope.exporters import json_exporter
from agentscope.tracer import Tracer

def run_experiment(
    agent: BaseAgent,
    config: dict,
    task: str,
    experiment_name: str,
    output_dir: str | Path,
    num_trials: int = 1,
) -> list[Path]:
    """
    Run an agent on a task for num_trials trials
    
    Each trial is independent. The tracer and exporter are created
    fresh for every trial. Saves one JSON trace file per trial.
    
    Returns a list of Paths to the saved trace files
    """
    output_dir = Path(output_dir)
    saved_paths = []
    
    for trial in range(num_trials):
        print(f"[{experiment_name}] trial {trial + 1}/{num_trials} ...", end=" ")
        
        tracer = Tracer()
        tracer.start_run(config=config, task_description=task)
        
        try:
            outcome = agent.run(task=task, tracer=tracer)
        except Exception as e:
            outcome = "failure"
            if tracer.has_open_step:
                tracer.end_step(
                    prompt_tokens=0,
                    completion_tokens=0,
                    context_window_tokens=0,
                    notes=f"step abandoned due to exception: {e}",
                )
            tracer.end_run(outcome=outcome, notes=f"exception: {e}")
        else:
            tracer.end_run(outcome=outcome)
        
        trace = tracer.get_trace()
        path = json_exporter.save(trace, output_dir, experiment_name)
        saved_paths.append(path)
        
        print(f"outcome={trace.outcome} tokens={trace.total_tokens} steps={len(trace.steps)}")
        
    return saved_paths