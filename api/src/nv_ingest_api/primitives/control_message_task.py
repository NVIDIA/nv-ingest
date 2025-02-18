from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict


class ControlMessageTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    id: str
    properties: Dict[str, Any] = Field(default_factory=dict)
