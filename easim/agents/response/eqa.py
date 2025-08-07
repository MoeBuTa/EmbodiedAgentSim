from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class EQAResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class EQAAction(BaseModel):
        model_config = ConfigDict(extra="forbid")
        action_type: str = Field(..., description="Type of action: NAVIGATE or ANSWER")
        action: str = Field(..., description="The specific action: MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, LOOK_UP, LOOK_DOWN, or ANSWER")
        reasoning: str = Field(..., description="Explanation of why this action was chosen")
        confidence: float = Field(..., description="Confidence level in this decision (0.0 to 1.0)")

    class AnswerDetails(BaseModel):
        model_config = ConfigDict(extra="forbid")
        answer: str = Field(..., description="The answer to the question (e.g., 'red', 'kitchen', 'yes')")
        answer_id: int = Field(..., description="Numeric ID for the answer (0-9 for evaluation)")
        evidence: Optional[str] = Field(None, description="Visual evidence supporting this answer")

    action_response: EQAAction = Field(..., description="The action decision and reasoning")
    
    question_understood: bool = Field(..., description="Whether the agent understands what the question is asking")
    
    can_answer: bool = Field(..., description="Whether enough information is visible to answer the question")
    
    answer_details: Optional[AnswerDetails] = Field(None, description="Answer details if action_type is ANSWER")
    
    observed_relevant_info: Optional[str] = Field(None, description="Any relevant information observed that relates to the question")