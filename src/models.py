from pydantic import BaseModel, Field, validator
from typing import List, Literal


class TicketRequest(BaseModel):
    """
    Request model for support ticket resolution.
    Follows MCP (Model Context Protocol) for structured input.
    """
    ticket_text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The support ticket text to analyze"
    )
    
    @validator('ticket_text')
    def validate_ticket_text(cls, v):
        """Validate ticket text is not just whitespace"""
        if not v or not v.strip():
            raise ValueError('ticket_text cannot be empty or whitespace only')
        if '\x00' in v:
            raise ValueError('ticket_text contains invalid null bytes')
        return v.strip()


class TicketResponse(BaseModel):
    """
    Response model for support ticket resolution.
    Follows MCP (Model Context Protocol) for structured output.
    """
    answer: str = Field(
        ...,
        description="Helpful answer based strictly on provided documentation"
    )
    references: List[str] = Field(
        ...,
        description="Source documents with page/line numbers cited"
    )
    action_required: Literal[
        "none",
        "escalate_to_abuse_team",
        "escalate_to_billing",
        "escalate_to_technical"
    ] = Field(
        ...,
        description="Required action based on the ticket analysis"
    )


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: str | None = None