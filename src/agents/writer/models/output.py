from typing import List
from pydantic import BaseModel, Field
from .content import ContentItem

class WriterOutput(BaseModel):
    section_title: str = Field(..., description="Section title")
    section_content: List[ContentItem] = Field(..., min_length=2, max_length=4, description="Content for this section that will be part of the final report")

class ValidatorOutput(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Score between 0 and 1 on how good section content is")
    improvements: List[str] = Field(..., min_length=0, max_length=5, description="Key points to improve the content")