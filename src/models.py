from pydantic import BaseModel, Field, validator
from typing import List, Literal

class TicketRequest(BaseModel):
    """a structured model for ticket creation requests"""
    ticket_text: str = Field(...,min_length=1,max_length=5000,description="the support ticket text")
    
    @validator('ticket_text')
    def validate_ticket_text(cls,v):
        """check that ticket is not just blank spaces"""
        if not v.strip():
            raise ValueError("ticket text cannot be blank or just spaces")
        if '\x00' in v:
            raise ValueError("ticket text cannot contain null characters")
        return v.strip()
    
class TicketResponse(BaseModel):
    """a structured model for response for that ticket"""
    answer:str=Field(...,description="the answer to the support ticket")
    references: List[str] = Field(...,description="list of reference documents used to generate the response with page / section number")
    action_required:Literal["none",
                            "escalate_to_abuse_team",
                            "escalate_to_billing_team",
                            "escalate_to_technical_support"
                            ] = Field(...,description="Required action based on the ticket analysis")

class ErrorResponse(BaseModel):
    """Error Response Model"""
    error:str
    detail:str