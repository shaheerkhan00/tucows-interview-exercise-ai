from typing import List

class PromptManager:
    """Prompt management following MCP (Model Context Protocol). """
    SYSTEM_PROMPT = """
    You are a technical support assistant for a domain registrar company.
    Your role is to:
    1. Analyze support tickets based ONLY on the provided documentation
    2. Provide accurate, helpful responses
    3. Cite specific sources for your answers
    4. Recommend escalation when appropriate
    CRITICAL RULES:
    - Base your answer STRICTLY on the provided context
    - If information is not in the context, explicitly say so
    - Include specific source references in your response
    - Be concise (2-4 sentences maximum)
    - Always return valid JSON in the exact format specified
    - Never make up information not present in the context

    """
    
    FEW_SHOT_EXAMPLES = [
        {
            "role": "user",
            "content": """Context:
             Document 1: domain_suspension_policy.pdf, Page 2
             Domains are suspended when WHOIS information is missing or invalid. To reactivate, update your WHOIS details within 15 days and contact support.

             Query: My domain was suspended without any notice. How do I fix it?"""
        },
        {
            "role": "assistant",
            "content": """{
             "answer": "Your domain was likely suspended due to missing or invalid WHOIS information. To reactivate it, you need to update your WHOIS details within 15 days and contact support.",
             "references": ["domain_suspension_policy.pdf, Page 2"],
            "action_required": "escalate_to_abuse"
                 }"""
        },
        {
            "role": "user",
            "content": """Context:
            Document 1: billing_faq.txt
            For billing issues, contact our billing department at billing@example.com

            Query: How do I train my pet dragon?"""
        },
        {
            "role": "assistant",
            "content": """{
             "answer": "I couldn't find relevant information about that topic in our documentation. This appears to be outside the scope of our support system.",
             "references": [],
             "action_required": "escalate_to_technical"
            }"""
        }
    ]
    @staticmethod
    def build_messages(query:str,context:str)->List[dict]:
        """Build messages for LLM interaction"""
        user_message=f"""Context: 
        {context}
        Query:
        {query}
        Respond with ONLY valid JSON in this exact format (no markdown,nobackticks, no preamble):
        {{
            "answer":"Your detailed answer based on the context (2-4 sentences)",
            "references":["list of reference documents with page/section numbers"],
            "action_required":"one of [none, escalate_to_abuse, escalate_to_billing, escalate_to_technical]"
            
        }}
        IMPORTANT REMINDERS:
        - If the context doesn't contain relevant information, say so clearly in the answer
        - action_required should be "none" unless escalation is clearly needed based on context
        - references must cite specific sources from the context above
        - Return ONLY the JSON object, no other text
        """
        messages = [{"role":"system","content":PromptManager.SYSTEM_PROMPT},
                    *PromptManager.FEW_SHOT_EXAMPLES,
                    {"role":"user","content":user_message}
                    ]
        return messages