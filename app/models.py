from typing import Optional, List
from sqlmodel import SQLModel, Field
import datetime


class Document(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str
    original_name: Optional[str]
    uploaded_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    num_pages: Optional[int] = None
    text_path: Optional[str] = None 