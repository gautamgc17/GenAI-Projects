from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


# Define message schema
class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    stream: bool = True