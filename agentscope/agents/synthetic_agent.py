"""
synthetic_agent.py - A deterministic agent for testing the profiling pipeline

This agent does not call any LLM or external API. It simulates
a multi-step agent run with configurable behavior like number of steps,
tokens per step, whether a tool is used, and if to simulate a failure

The purpose is to let us run the full pipeline (runner -> tracer -> exporter)
in tests and experiments without needing an API key
"""

import time

from agentscope.agents.base_agent import BaseAgent
from agentscope.schema import ToolCallTrace
from agentscope.tracer import Tracer

class SyntheticAgent(BaseAgent):
    """
    A fake agent that produces predictable, configurable trace data
    
    Parameters:
    - num_steps: int, how many steps the agent will take before finishing
    - prompt_tokens_start: int, Prompt token count at step 0. Grows by this amount each step to
    simulate context accumulation
    - completion_tokens: int, Completion token count, same for every step
    - use_tool_at_step: int | None, if set, the agent records a tool call at this step index
    - fail_at_step: int | None, if set, agent returns "failure" and stops at this step
    - step_duration_ms: float, how long to sleep per step in milliseconds, keeps 
    latency realistic without being slow. Default is 10ms
    """
    
    def __init__(
        self,
        num_steps: int = 3,
        prompt_tokens_start: int = 512,
        completion_tokens: int = 128,
        use_tool_at_step: int | None = None,
        fail_at_step: int | None = None,
        step_duration_ms: float = 10.0,
    ) -> None:
        self.num_steps = num_steps
        self.prompt_tokens_start = prompt_tokens_start
        self.completion_tokens = completion_tokens
        self.use_tool_at_step = use_tool_at_step
        self.fail_at_step = fail_at_step
        self.step_duration_ms = step_duration_ms
        
    def run(self, task: str, tracer: Tracer) -> str:
        for i in range(self.num_steps):
            step_type = "tool_use" if i == self.use_tool_at_step else "standard"
            if i == self.num_steps - 1:
                step_type = "final"
            
            tracer.start_step(i, step_type=step_type)
            
            # Simulate agent doing work (LLM call latency)
            time.sleep(self.step_duration_ms / 1000)
            
            # Context grows each step, prior response is added to the prompt
            prompt_tokens = self.prompt_tokens_start + i * self.completion_tokens
            context_window_tokens = prompt_tokens + self.completion_tokens
            
            if i == self.use_tool_at_step:
                tracer.record_tool_call(ToolCallTrace(
                    tool_name="web_search",
                    input_summary=f"step {i} search query",
                    output_summary="3 results returned",
                    success=True,
                    latency_ms=50.0,
                ))

            tracer.end_step(
                prompt_tokens=prompt_tokens,
                completion_tokens=self.completion_tokens,
                context_window_tokens=context_window_tokens,
            )
            
            if i == self.fail_at_step:
                return "failure"
            
        return "success"
