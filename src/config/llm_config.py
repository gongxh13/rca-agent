from typing import Optional, Any, Dict
from pydantic import BaseModel

class LLMConfig(BaseModel):
    type: str  # provider, e.g., "openai", "deepseek"
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    setting: Optional[Dict[str, Any]] = None
