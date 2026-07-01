"""
Pydantic models for the SHL Assessment Recommender API.
Schema is non-negotiable — any deviation breaks the automated evaluator.
"""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text content")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("user", "assistant", "system"):
            raise ValueError(f"role must be 'user', 'assistant', or 'system', got: {v}")
        return v


class ChatRequest(BaseModel):
    messages: List[Message] = Field(
        ...,
        min_length=1,
        description="Full conversation history; stateless — caller must pass everything",
    )


class Recommendation(BaseModel):
    name: str = Field(..., description="Exact assessment name from the SHL catalog")
    url: str = Field(..., description="Canonical catalog URL (must be from SHL catalog)")
    test_type: str = Field(
        ...,
        description=(
            "Short letter code for primary test type: "
            "A=Ability&Aptitude, B=Biodata&SJT, C=Competencies, "
            "D=Development&360, E=AssessmentExercises, "
            "K=Knowledge&Skills, P=Personality&Behavior, S=Simulations"
        ),
    )


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Agent's natural language reply to the user")
    recommendations: List[Recommendation] = Field(
        default_factory=list,
        description=(
            "Empty when still gathering context or refusing. "
            "1-10 items when committed to a shortlist."
        ),
    )
    end_of_conversation: bool = Field(
        default=False,
        description="True only when the agent considers the task complete",
    )


class HealthResponse(BaseModel):
    status: str = "ok"
