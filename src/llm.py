import json
from src.models import TicketResponse
from src.prompts import PromptManager
from src.config import get_settings

settings = get_settings()


class LLMService:
    """
    LLM Service using OpenAI for reliable JSON responses.
    Uses json_object mode for guaranteed valid JSON format.
    """
    
    def __init__(self):
        """Initialize OpenAI client"""
        from openai import OpenAI
        
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
        print(f"LLM Service initialized: OpenAI ({self.model})")
        
    def generate_response(self, query: str, context: str) -> TicketResponse:
        """
        Generate structured response from LLM.
        
        Args:
            query: User support ticket text
            context: Relevant document context
            
        Returns:
            TicketResponse: Validated structured response
        """
        messages = PromptManager.build_messages(query, context)
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=800,
                seed=42
            )
            
            content = completion.choices[0].message.content
            parsed = json.loads(content)
            response = TicketResponse(**parsed)
            return response
            
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {e}")
            return self._fallback_response()
            
        except Exception as e:
            print(f"LLM Error: {e}")
            return self._fallback_response()
        
    def _fallback_response(self) -> TicketResponse:
        """Safe fallback response when LLM fails"""
        return TicketResponse(
            answer="I encountered an error processing your request. Please contact support directly for assistance.",
            references=[],
            action_required="escalate_to_technical"
        )


def test_llm():
    """Test LLM service with sample context"""
    llm = LLMService()
    
    sample_context = """Document 1: domain_suspension_policy.txt
Domains may be suspended for missing WHOIS information. To reactivate, update your WHOIS details within 15 days.

Document 2: billing_faq.txt
For refund requests, contact billing@support.com within 30 days of purchase."""
    
    test_query = "My domain was suspended, how do I fix it?"
    
    print(f"Test Query: {test_query}")
    
    response = llm.generate_response(test_query, sample_context)
    
    print(f"Answer: {response.answer}")
    print(f"References: {response.references}")
    print(f"Action Required: {response.action_required}")


if __name__ == "__main__":
    test_llm()