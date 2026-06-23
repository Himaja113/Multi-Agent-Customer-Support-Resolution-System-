import hashlib
import time
from typing import Dict, Any
from state import SupportState

def handle_escalation(state: SupportState) -> Dict[str, Any]:
    """
    Handles queries that cannot be resolved automatically by:
    1. Identifying the escalation reason.
    2. Generating an internal system log (printed for visibility).
    3. Generating a polite customer-facing message with a reference ID.
    """
    category = state.get("category", "general")
    intent = state.get("intent", "fallback")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    low_confidence = state.get("low_confidence", False)
    generated_response = state.get("generated_response")
    quality_decision = state.get("quality_decision")
    user_query = state.get("user_query", "")
    
    # 1. Identify Escalation Reason
    if retry_count > max_retries:
        reason = "max_retries_exceeded"
    elif low_confidence:
        reason = "low_confidence"
    elif not generated_response or len(generated_response) < 20:
        reason = "empty_response"
    elif quality_decision == "escalate":
        reason = "critic_decision"
    else:
        reason = "critic_decision"
        
    # 2. Compute Priority and SLA Details based on category
    cat_details = {
        "billing": {"priority": "HIGH", "sla": "12-24 hours", "team": "Billing Specialist Team"},
        "technical": {"priority": "MEDIUM", "sla": "24-48 hours", "team": "Technical Engineering Team"},
        "delivery": {"priority": "MEDIUM", "sla": "24-48 hours", "team": "Logistics & Delivery Team"},
        "general": {"priority": "LOW", "sla": "48-72 hours", "team": "Customer Support Team"}
    }
    
    details = cat_details.get(category, cat_details["general"])
    
    # 3. Generate Reference ID
    hash_input = f"{user_query}{time.time()}"
    ref_id = f"REF-{hashlib.md5(hash_input.encode('utf-8')).hexdigest()[:8].upper()}"
    
    # 4. Formulate Part 1: Internal Log (printed for developer/system visibility)
    internal_log = f"""
================ SYSTEM ESCALATION LOG ================
[Reference ID]     {ref_id}
[Reason]           {reason}
[Category/Intent]  {category} / {intent}
[Priority]         {details['priority']}
[Attempts Made]    {retry_count}
[Original Query]   "{user_query}"
[Retrieved Context] {state.get('retrieved_context', 'None')[:150]}...
[Last Response]    "{generated_response if generated_response else 'None'}"
======================================================
"""
    # Print the log so it appears clearly in the terminal run
    print(internal_log)
    
    # 5. Formulate Part 2: Customer-Facing Message
    customer_message = (
        f"Thank you for contacting us. Your request has been received and escalated to our "
        f"{details['team']}. A ticket has been created with Reference ID: {ref_id}.\n\n"
        f"Category: {category.capitalize()}\n"
        f"Expected Response Time: within {details['sla']}\n"
        f"A support agent will follow up with you directly."
    )
    
    return {
        "escalated": True,
        "escalation_reason": reason,
        "final_answer": customer_message
    }
