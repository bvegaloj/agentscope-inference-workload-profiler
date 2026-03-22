"""
experiments/run_high_context.py - High-context failure experiment

Runs a 15-step SyntheticAgent that fails at step 10 with a larger
starting context window. Demonstrates token cost accrued before failure

Run from project root:
    python experiments/run_high_context.py
"""

from pathlib import Path

from agentscope.agents.synthetic_agent import SyntheticAgent
from agentscope import analysis
from agentscope.runner import run_experiment

import yaml

CONFIG_FILE = Path("experiments/configs/high_context.yaml")

with open(CONFIG_FILE) as f:
    cfg = yaml.safe_load(f)

EXPERIMENT_NAME = cfg["experiment_name"]
TASK = cfg["task"]
NUM_TRIALS = cfg["num_trials"]
TRACES_DIR = Path("experiments/traces")
CONFIG = cfg["agent"]
PROMPT_PRICE_PER_1K = cfg["pricing"]["prompt_price_per_1k"]
COMPLETION_PRICE_PER_1K = cfg["pricing"]["completion_price_per_1k"]

agent = SyntheticAgent(
    num_steps=CONFIG["num_steps"],
    use_tool_at_step=CONFIG["use_tool_at_step"],
    fail_at_step=CONFIG["fail_at_step"],
    prompt_tokens_start=CONFIG["prompt_tokens_start"],
    completion_tokens=CONFIG["completion_tokens"],
)

# Helpers

def bar(value: float, max_value: float, width: int = 20) -> str:
    """
    Render a simple ASCII progress bar.
    """
    filled = round((value / max_value) * width) if max_value > 0 else 0
    return "█" * filled + "░" * (width - filled)

def section(title: str) -> None:
    print(f"\n{'-' * 60}")
    print(title)
    print("-" * 60)
    
# main

def main() -> None:
    print("=" * 60)
    print(f"AgentScope Profiler  |  Experiment: {EXPERIMENT_NAME}")
    print("=" * 60)
    
    print("\nConfig")
    for k, v in CONFIG.items():
        print(f"  {k:<20}: {v if v is not None else 'none'}")
    print(f"\n  {'task':<20}: {TASK}")
    print(f"  {'trials':<20}: {NUM_TRIALS}")
    print(f"  {'output':<20}: {TRACES_DIR}")
    
    agent = SyntheticAgent(
        num_steps=CONFIG["num_steps"],
        use_tool_at_step=CONFIG["use_tool_at_step"],
        fail_at_step=CONFIG["fail_at_step"],
        prompt_tokens_start=CONFIG["prompt_tokens_start"],
        completion_tokens=CONFIG["completion_tokens"],
    )

    section("Running trials")
    run_experiment(
        agent, CONFIG, TASK, EXPERIMENT_NAME, TRACES_DIR, num_trials=NUM_TRIALS
    )

    traces = analysis.load_traces(TRACES_DIR, EXPERIMENT_NAME)
    summary = analysis.summarize_runs(traces)
    growth = analysis.context_growth(traces)
    cost = analysis.estimate_cost(traces, PROMPT_PRICE_PER_1K, COMPLETION_PRICE_PER_1K)

    section(f"Run Summary  ({summary['num_runs']} trials)")
    print(f"  {'Success rate':<26} {summary['success_rate'] * 100:.1f}%")
    print(f"  {'Avg tokens / run':<26} {summary['avg_tokens']:.0f}")
    print(f"    {'Avg prompt tokens':<24} {summary['avg_prompt_tokens']:.0f}")
    print(f"    {'Avg completion tokens':<24} {summary['avg_completion_tokens']:.0f}")
    print(f"  {'Avg steps / run':<26} {summary['avg_steps']:.1f}")
    print(f"  {'Avg duration':<26} {summary['avg_duration_ms']:.0f} ms")
    print(f"  {'Avg retries / run':<26} {summary['avg_retries']:.2f}")
    print(f"  {'Avg tool calls / run':<26} {summary['avg_tool_calls']:.2f}")

    section("Context Growth  (avg prompt tokens per step)")
    max_tokens = max(r["avg_prompt_tokens"] for r in growth) if growth else 1
    for row in growth:
        b = bar(row["avg_prompt_tokens"], max_tokens)
        print(
            f"  Step {row['step_index']:<4} "
            f"{row['avg_prompt_tokens']:>6.0f} tokens  {b}  "
            f"({row['num_runs']} runs)"
        )

    section(
        f"Cost Estimate  "
        f"(prompt: ${PROMPT_PRICE_PER_1K}/1k  "
        f"completion: ${COMPLETION_PRICE_PER_1K}/1k)"
    )
    print(f"  {'Total (' + str(summary['num_runs']) + ' runs)':<26} ${cost['total_cost_usd']:.4f}")
    print(f"  {'Avg per run':<26} ${cost['avg_cost_usd']:.4f}")
    print(f"  {'Note':<26} all runs failed at step {CONFIG['fail_at_step']}; cost above is wasted spend")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()   