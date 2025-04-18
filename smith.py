from langsmith import Client
import os
from dotenv import load_dotenv
from langchain_core.tracers.langchain import LangChainTracer
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
from typing import Dict, Any, List, Callable, Optional
import json
import traceback

# Load environment variables
load_dotenv()

# Initialize LangSmith client
langsmith_client = Client(
    api_key=os.getenv("LANGCHAIN_API_KEY"),
    project_name=os.getenv("LANGCHAIN_PROJECT", "srs_generator_project")
)

class LangSmithLogger:
    """
    A utility class for logging workflow execution to LangSmith.
    """
    
    def __init__(self, project_name: str = None, run_name: str = None):
        self.project_name = project_name or os.getenv("LANGCHAIN_PROJECT", "srs_generator_project")
        self.run_name = run_name or "SRS Processing Run"
        self.tracer = LangChainTracer(project_name=self.project_name)
        self.run_id = None
    
    def start_trace(self, metadata: Dict[str, Any] = None):
        """Start a new trace."""
        trace_metadata = metadata or {}
        self.run_id = self.tracer.new_trace(self.run_name, metadata=trace_metadata)
        return self.run_i