import json
from pydantic import BaseModel, Field
from typing import Dict, Any
from state import SupportState
from utils.llm import get_llm

class ClassificationOutput(BaseModel):
    category: str = Field(description="The category of the user query. Must be one of: billing, technical, delivery, general.")
    intent: str = Field(description="The specific intent of the user query. Must be chosen from the allowed list for the category.")
    entities: Dict[str, Any] = Field(default_factory=dict, description="Extracted entities such as order IDs, emails, subscription types, etc. if present.")

CLASSIFIER_PROMPT = """You are a query classifier for a customer support system.
Your job is to analyze the customer's query and extract the category, the intent, and any relevant entities.

You MUST choose the category and intent from the following allowed categories and intents:

1. Category: billing
   Intents:
   - double_charge
   - refund_request
   - invoice_request
   - cancel_subscription

2. Category: technical
   Intents:
   - reset_password
   - account_locked
   - app_crash
   - login_issue

3. Category: delivery
   Intents:
   - order_not_arrived
   - wrong_item
   - damaged_item
   - tracking_request

4. Category: general
   Intents:
   - fallback

INSTRUCTIONS:
- You must choose ONLY from the list above. Do NOT invent new categories or intents.
- If the query does not fit any of the billing, technical, or delivery categories, classify it as category: "general" and intent: "fallback".
- Extract any relevant entities (e.g., email addresses, order IDs, account IDs, names) as a key-value dictionary. If no entities are present, return an empty dictionary.

Customer Query: "{query}"
"""

def classify_query(state: SupportState) -> Dict[str, Any]:
    """
    Takes the user query and classifies it into a category and intent, extracting entities.
    Returns the update to the state.
    """
    query = state.get("user_query", "")
    
    # We use a low temperature of 0.1 for highly consistent classification and extraction
    llm = get_llm(temperature=0.1)
    
    # Set up structured output
    try:
        structured_llm = llm.with_structured_output(ClassificationOutput)
        prompt = CLASSIFIER_PROMPT.format(query=query)
        result = structured_llm.invoke(prompt)
        
        category = result.category.strip().lower()
        intent = result.intent.strip().lower()
        entities = result.entities if isinstance(result.entities, dict) else {}
        
        # Validate category and intent
        valid_map = {
            "billing": ["double_charge", "refund_request", "invoice_request", "cancel_subscription"],
            "technical": ["reset_password", "account_locked", "app_crash", "login_issue"],
            "delivery": ["order_not_arrived", "wrong_item", "damaged_item", "tracking_request"],
            "general": ["fallback"]
        }
        
        if category not in valid_map:
            category = "general"
            intent = "fallback"
        elif intent not in valid_map[category]:
            # If category is valid but intent is not, we can either default to fallback or look for substring
            intent = "fallback" if category == "general" else valid_map[category][0] # use first valid intent as default
            
        return {
            "category": category,
            "intent": intent,
            "entities": entities
        }
    except Exception as e:
        # Graceful degradation on failure (e.g., API issues, validation errors, parsing errors)
        return {
            "category": "general",
            "intent": "fallback",
            "entities": {}
        }
