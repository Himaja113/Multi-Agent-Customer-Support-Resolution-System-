from typing import Dict, Any
from state import SupportState
from utils.llm import get_llm

RESPONDER_PROMPT = """You are a professional and empathetic customer support agent.
Answer the customer's query using ONLY the information provided in the context below.
Do not add information or make up facts that are not in the context.
Be clear, warm, and provide specific next steps.

[Retrieved Context]
{retrieved_context}

[Customer Query]
{user_query}
"""

RETRY_FEEDBACK_PROMPT = """
Your previous response was rejected.
Reason: {quality_feedback}
Please address this feedback specifically in your new response.
"""

def generate_response(state: SupportState) -> Dict[str, Any]:
    """
    Generates a professional response to the user query based on the retrieved context.
    Increments retry_count. Handles garbage responses and LLM failures by routing directly to escalation.
    """
    # 1. Increment retry count
    current_retry = state.get("retry_count", 0)
    new_retry_count = current_retry + 1
    
    user_query = state.get("user_query", "")
    retrieved_context = state.get("retrieved_context", "")
    quality_feedback = state.get("quality_feedback")
    
    # Check if we should abort due to lack of query
    if not user_query:
        return {
            "generated_response": None,
            "quality_decision": "escalate",
            "retry_count": new_retry_count
        }

    # 2. Formulate prompt
    prompt = RESPONDER_PROMPT.format(
        retrieved_context=retrieved_context,
        user_query=user_query
    )
    
    if current_retry > 0 and quality_feedback:
        prompt += RETRY_FEEDBACK_PROMPT.format(quality_feedback=quality_feedback)
        
    try:
        # Get LLM with temperature 0.3 (moderate creativity while retaining grounding)
        llm = get_llm(temperature=0.3)
        response = llm.invoke(prompt)
        response_text = response.content.strip()
        
        # 3. Check for empty or garbage response (under 20 characters)
        if not response_text or len(response_text) < 20:
            return {
                "generated_response": None,
                "quality_decision": "escalate",
                "retry_count": new_retry_count
            }
            
        return {
            "generated_response": response_text,
            "retry_count": new_retry_count,
            # Reset decision on successful generation so critic can evaluate
            "quality_decision": None 
        }
        
    except Exception as e:
        # Log error internally and fail gracefully to escalation
        return {
            "generated_response": None,
            "quality_decision": "escalate",
            "retry_count": new_retry_count
        }
