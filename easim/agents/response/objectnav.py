from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ObjectNavResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class ObjectNavAction(BaseModel):
        model_config = ConfigDict(extra="forbid")
        action: str = Field(..., description="The action to take: MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, LOOK_UP, LOOK_DOWN, or STOP")
        reasoning: str = Field(..., description="Brief explanation of why this action was chosen")
        confidence: float = Field(..., description="Confidence level in this decision (0.0 to 1.0)")

    action_response: ObjectNavAction = Field(..., description="The action decision and reasoning")
    
    target_visible: bool = Field(..., description="Whether the target object is currently visible in the scene")
    
    observed_objects: Optional[List[str]] = Field(None, description="List of objects currently visible in the scene")