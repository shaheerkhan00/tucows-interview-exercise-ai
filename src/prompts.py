import json
from typing import List, Dict, Any

class PromptManager:
    """
    MCP-compliant prompt manager for a RAG-based OpenAI application.
    """

    SYSTEM_PROMPT = """You are a technical support assistant for a domain registrar company.

Your responsibilities:
- Answer strictly and only using the provided documents
- Do not rely on outside knowledge
- Cite exact document sources
- Follow the escalation policy exactly
- Return ONLY valid JSON matching the required schema

CONFLICT RESOLUTION RULES:
- If two documents conflict, prioritize the "Handbook" or "Policy Manual" over "FAQs".
- Specific rules (e.g., "Premium Domains") always override general rules (e.g., "Refunds").

If the documents do not contain the answer, state this explicitly.
Do not include reasoning or any text outside the JSON response.
"""

    ESCALATION_POLICY = {
        "escalate_to_abuse_team": [
            "spam", "phishing", "malware", "abuse", "content violations"
        ],
        "escalate_to_billing": [
            "payment dispute", "double charge", "refund", "invoice"
        ],
        "escalate_to_technical": [
            "dns", "ssl", "system failure", "api error", "propagation"
        ],
        "escalate_to_legal": [
            "lawsuit", "subpoena", "udrp", "lawyer", "death of registrant"
        ],
        "escalate_to_privacy": [
            "gdpr", "right to be forgotten", "data deletion"
        ],
        "escalate_to_security": [
            "account hijacking", "stolen domain", "unauthorized access"
        ]
    }

    FEW_SHOT_EXAMPLES = [
        {
            "role": "user",
            "content": json.dumps({
                "documents": [
                    {
                        "id": "domain_suspension_policy.txt",
                        "page": 2,
                        "content": "Domains are suspended when WHOIS information is missing or invalid. To reactivate, update your WHOIS details within 15 days and contact support."
                    }
                ],
                "query": "My domain was suspended. How do I fix it?"
            })
        },
        {
            "role": "assistant",
            "content": json.dumps({
                "answer": "Your domain was likely suspended due to missing or invalid WHOIS information. To reactivate it, update your WHOIS details within 15 days and contact support.",
                "references": ["domain_suspension_policy.txt, Page 2"],
                "action_required": "none"
            })
        }
    ]

    @staticmethod
    def build_messages(query: str, context: str) -> List[Dict[str, Any]]:
        """
        Build MCP-compliant messages for the OpenAI API.
        The provided context is preserved verbatim and treated as data.
        """

        # Wrap the raw context string into a structured document object
        documents = [
            {
                "id": "context_bundle",
                "content": context
            }
        ]

        # Construct the structured payload
        user_payload = {
            "documents": documents,
            "query": query,
            "output_schema": {
                "answer": "string (2–4 sentences)",
                "references": "array of document citations",
                "action_required": "String. Must be exactly one of: none, escalate_to_abuse_team, escalate_to_billing, escalate_to_technical, escalate_to_legal, escalate_to_privacy, escalate_to_security"
            },
            "escalation_policy": PromptManager.ESCALATION_POLICY
        }

        messages = [
            {"role": "system", "content": PromptManager.SYSTEM_PROMPT}
        ]

        # Add few-shot examples (already strings)
        messages.extend(PromptManager.FEW_SHOT_EXAMPLES)

        # Add user query (Convert Dict -> JSON String)
        messages.append({
            "role": "user",
            "content": json.dumps(user_payload)
        })

        return messages