"""
base_agent.py - The interface contract every agent must follow

The runner only knows about this interface. Any agent must implement the
run() method with this exact signature to be usable by the runner

This uses Python's ABC (Abstract Base Class) module to enforce the contract
If a subclass forgets to implement run(), Python will raise
a TypeError at instantiation time, not silently at call time
"""

from abc import ABC, abstractmethod

from agentscope.tracer import Tracer

class BaseAgent(ABC):
    
    @abstractmethod
    def run(self, task: str, tracer: Tracer) -> str:
        """
        Execute the task and record all steps into tracer
        
        The agent is responsible for calling tracer.start_step() and 
        tracer.end_step() for every step it takes. 
        The runner calls tracer.start_run() and tracer.end_run()
        The agent must not call those
        
        Returns the outcome as a string:
            "success", "failure", "partial"
        """
        ...