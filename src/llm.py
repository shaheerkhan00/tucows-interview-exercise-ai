import json
from typing import Dict
from src.models import TicketResponse
from src.prompts import PromptManager
from src.config import get_settings


settings = get_settings()

class LLMService:
    """
    LLM Service 
    can switch between Groq and OpenAI based on configuration
    both support json_object for guranteed valid json format
    
    """
    def __init__(self):
        #self.provider = settings.LLM_PROVIDER
        
        from openai import OpenAI
        if not settings.OPENAI_API_KEY:
            raise ValueError("OpenAI API key is not set in environment variables.")
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
        print(f"Using OpenAI model: {self.model}")
        
        
    def generate_response(self,query:str,context:str)->TicketResponse:
        """
        Generate structured response from llm based on query and context
        args:
            query (str): user support ticket
            context (str): relevant document context
        returns:
            TicketResponse: structured response model
        """
        messages = PromptManager.build_messages(query,context)
        
        try:
            completion = self.client.chat.completions.create(
                model = self.model,
                messages = messages,
                temperature=0.2,
                response_format={"type":"json_object"},
                max_tokens=500,
            )
            content = completion.choices[0].message.content
            parsed = json.loads(content)
            response = TicketResponse(**parsed)
            return response
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response from LLM: {e}")
            return self._fallback_response()
        except Exception as e:
            print(f"Error generating response from LLM: {e}")
            return self._fallback_response()
        
    def _fallback_response(self)->TicketResponse:
        """Return a default fallback response in case of errors"""
        return TicketResponse(
            answer="I'm sorry, but I couldn't process your request at this time.",
            references=[],
            action_required="escalate_to_technical_support"
        )

def test_llm():
        """Test LLM service with sample context"""
        llm = LLMService()
    
        sample_context = """Document 1: domain_suspension_policy.txt
        Domains may be suspended for missing WHOIS information. To reactivate, update your WHOIS details within 15 days.

        Document 2: billing_faq.txt
        For refund requests, contact billing@support.com within 30 days of purchase."""
    
        test_query = "My domain was suspended, how do I fix it?"
    
        print(f"\nTest Query: {test_query}")
        
        response = llm.generate_response(test_query, sample_context)
    
        print(f"\nAnswer: {response.answer}")
        print(f"References: {response.references}")
        print(f"Action Required: {response.action_required}")

if __name__ == "__main__":
    test_llm()
    