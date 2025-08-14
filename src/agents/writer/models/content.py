from typing import List, Literal
from pydantic import BaseModel

class ContentItem(BaseModel):
    order: int
    content_type: Literal["narrative", "bullets", "table"]
    text: str = ""
    items_subtitle: str = ""
    items: List[str] = []
    headers: List[str] = []
    rows: List[List[str]] = []
    table_subtitle: str = ""
    table_footer: str = ""

class SectionContent(BaseModel):
    section_title: str
    section_content: List[ContentItem]
