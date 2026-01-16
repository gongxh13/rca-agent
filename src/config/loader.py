import yaml
import os
from pathlib import Path
from typing import List, Optional, Any
from pydantic import BaseModel
from .llm_config import LLMConfig

class Config(BaseModel):
    llm: List[LLMConfig]

def _expand_env(obj: Any) -> Any:
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    return obj

def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from a YAML file."""
    # Search in current directory, or project root
    paths_to_try = [
        Path(config_path),
        Path(os.getcwd()) / config_path,
        Path(__file__).parent.parent.parent / config_path
    ]
    
    found_path = None
    for p in paths_to_try:
        if p.exists():
            found_path = p
            break
            
    if not found_path:
        raise FileNotFoundError(f"Config file not found. Tried: {paths_to_try}")
        
    with open(found_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data = _expand_env(data)
    return Config(**data)
