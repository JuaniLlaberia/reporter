from typing import List, Literal
from pydantic import BaseModel

class ContentItem(BaseModel):
    order: int
    content_type: Literal["narrative", "bullets", "table"]
    text: str = ""
    items: List[str] = []
    headers: List[str] = []
    rows: List[List[str]] = []

class SectionContent(BaseModel):
    section_title: str
    content: List[ContentItem]
