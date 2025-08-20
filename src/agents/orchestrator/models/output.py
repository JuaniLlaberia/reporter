from enum import Enum
from typing import List
from pydantic import BaseModel, Field

class ReportType(str, Enum):
    DETAILED = "detailed"
    SUMMARY = "summary"

class OrchestratorOutput(BaseModel):
    topic: str = Field(..., description="Report topic mentioned in the prompt")
    focus_keys: List[str] = Field(..., min_items=1, max_items=5, description="Sub-topics to focus report on")
    type: ReportType = Field(..., description="Report type")
    plan_queries: List[str] = Field(..., min_items=1, max_items=10, description="Queries based on prompt to retrieve basic chunks")