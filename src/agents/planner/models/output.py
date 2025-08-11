from typing import List, Optional
from pydantic import BaseModel, Field

class Section(BaseModel):
    name: str = Field(..., description="Section name or temporal title")
    expected_format: List[str] = Field(min_length=2, max_length=3, description="Section format can be narrative, bullets or table. And can be combined")
    goal: str = Field(..., description="Section goal")
    queries: List[str] = Field(..., min_length=3, max_length=5, description="Queries to retrieve section data")
    documents: Optional[List[str]] = None

class PlannerOutput(BaseModel):
    sections: List[Section] = Field(..., min_length=4, max_length=20, description="Sections that will be part of the final report")

class EvaluationOutput(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Score between 0 and 1 on how good is the plan")
    improvements: List[str] = Field(..., min_length=0, max_length=5, description="Key points to improve the plan")