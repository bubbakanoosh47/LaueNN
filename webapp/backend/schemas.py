from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

class RunRequest(BaseModel):
    cli_args: List[str]


class WorkflowRunRequest(BaseModel):
    step: str
    workflow_id: Optional[str] = None
    material: Optional[str] = None
    overrides: Dict[str, Any] = Field(default_factory=dict)
    output_root: str = "project"
    data_root: str = "project"
