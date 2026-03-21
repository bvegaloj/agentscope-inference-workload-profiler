# AgentScope: Inference Workload Profiler

A Python system for profiling long-horizon agent workloads. Measures token usage, context growth, retries, tool calls, and cost across multi-step agent runs.

## Why This Exists

LLM agent systems fail in ways that are invisible without instrumentation:

- **Context bloat**: prompt tokens grow on every step as prior history accumulates. By step 20, you may be paying 5x more per step that step 1.
- **Silent retries**: failed LLM calls are retried automatically. Without tracing, you never know how often this happens or what it costs.
- **Unpredictable duration**: a 10-step agent task can take 2 seconds or 40 seconds depending on tool latency and retry behavior.

AgentScope makes all of this visible by attaching a tracer to the agent loop and recording structured data at every step.

## What It Measures

| Metric | Description |
|---|---|
| Prompt tokens per step | Grows as context accumulates |
| Completion tokens per step | Output verbosity |
| Context window tokens | Total tokens in context at each step |
| Step latency | Time per agent decision cycle |
| Total task duration | Wall-clock cost of a complete run |
| Retries | Per step and across the full run |
| Tool calls | Count and success rate |
| Estimated cost | Prompt + completion cost at configurable pricing |
| Outcome | success / failure / partial |

## Architecture

Five components, each with a single responsibility:

- **schema** - dataclasses defining the trace data contract
- **tracer** - stateful recorder; instruments the agent loop at runtime
- **exporter** - serializes completed traces to JSON files
- **runner** - orchestrates N trials of an agent config; wires the above together
- **analysis** - loads saved traces and computes statistics

The tracer and exporter are decoupled: the tracer holds data in memory, the exporter decides where it goes. Swapping JSON for a database requires only changing the exporter.

## Quick Start

```bash
git clone https://github.com/bvegaloj/agentscope-inference-workload-profiler
cd agentscope-inference-workload-profiler
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"
python experiments/run_baseline.py
```

## Sample Output

```text
============================================================
AgentScope Profiler  |  Experiment: baseline
============================================================

Config
  model               : synthetic
  max_steps           : 8
  fail_at_step        : none
  use_tool_at_step    : 2

  task                : Summarize recent AI research papers
  trials              : 5
  output              : experiments\traces

------------------------------------------------------------
Run Summary  (5 trials)
------------------------------------------------------------
  Success rate               100.0%
  Avg tokens / run           8704
    Avg prompt tokens        7680
    Avg completion tokens    1024
  Avg steps / run            8.0
  Avg duration               82 ms
  Avg retries / run          0.00
  Avg tool calls / run       1.00

------------------------------------------------------------
Context Growth  (avg prompt tokens per step)
------------------------------------------------------------
  Step 0       512 tokens  ███████░░░░░░░░░░░░░  (5 runs)
  Step 1       640 tokens  █████████░░░░░░░░░░░  (5 runs)
  Step 2       768 tokens  ███████████░░░░░░░░░  (5 runs)
  Step 3       896 tokens  █████████████░░░░░░░  (5 runs)
  Step 4      1024 tokens  ███████████████░░░░░  (5 runs)
  Step 5      1152 tokens  ████████████████░░░░  (5 runs)
  Step 6      1280 tokens  ██████████████████░░  (5 runs)
  Step 7      1408 tokens  ████████████████████  (5 runs)

------------------------------------------------------------
Cost Estimate  (prompt: $0.005/1k  completion: $0.015/1k)
------------------------------------------------------------
  Total (5 runs)             $0.2688
  Avg per run                $0.0538
```

## Running Tests

```bash
python -m pytest -v
```

86 tests across schema, tracer, exporter, runner, and analysis. No external
dependencies, no API key required.

## Project Structure

```
agentscope/
  schema.py          # RunTrace, StepTrace, ToolCallTrace dataclasses
  tracer.py          # runtime recorder; instruments the agent loop
  runner.py          # experiment orchestration; N trials of a config
  analysis.py        # load traces, compute stats, estimate cost
  agents/
    base_agent.py    # abstract interface all agents must implement
    synthetic_agent.py  # deterministic agent for testing; no API needed
  exporters/
    json_exporter.py # save/load traces as JSON files
experiments/
  configs/
    baseline.yaml    # experiment parameters
  traces/            # output directory (gitignored)
  run_baseline.py    # end-to-end experiment script
tests/
  test_schema.py
  test_tracer.py
  test_json_exporter.py
  test_runner.py
  test_analysis.py
```

## Planned Extensions
- Second experiment config: long-horizon run with failure injection to compare
cost and reliability across configurations
- Go REST API: serve trace data over HTTP so analysis can be queried rather
than run as a CLI script
- Real LLM provider: OpenAI provider implementation behind the BaseAgent
interface; no changes to tracer, runner, or analysis required