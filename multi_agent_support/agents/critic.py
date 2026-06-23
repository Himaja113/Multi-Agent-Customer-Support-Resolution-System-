from typing import Dict, Any
from pydantic import BaseModel, Field
from state import SupportState
from utils.llm import get_llm

class CriticOutput(BaseModel):
    decision: str = Field(description="The decision: 'accept', 'regenerate', or 'escalate'. Must be lowercased.")
    feedback: str = Field(description="Detailed feedback explaining the decision and what to improve if regenerating.")

CRITIC_PROMPT = """You are a Quality Check (Critic) Agent for a customer support system.
Your job is to evaluate the generated response against the customer query and the retrieved context.

Evaluate the response based on these three criteria:
1. Correctness: Is the response fully grounded in the retrieved context? Does it avoid adding outside facts or contradicting the context?
2. Completeness: Does it address the customer's query completely? Does it include specific next steps as outlined in the context?
3. Tone: Is it professional, polite, warm, and empathetic? Does it feel human rather than robotic or dismissive?

Decide on the action to take:
- "accept": If the response is correct, complete, and has the right tone.
- "regenerate": If the response has minor, fixable issues in completeness or tone (e.g., missed a step, too robotic, forgot a greeting). You MUST provide clear, actionable feedback on what needs to be fixed.
- "escalate": If the response is completely wrong, contradicts the context, contains hallucinated information, or fails to address the query.

Retrieved Context:
{retrieved_context}

Generated Response:
{generated_response}

Customer Query:
{user_query}
"""

def quality_check(state: SupportState) -> Dict[str, Any]:
    """
    Evaluates the generated response for correctness, completeness, and tone.
    Routes to accept, regenerate, or escalate.
    """
    # 1. Fail-fast if escalation is already triggered by Responder
    if state.get("quality_decision") == "escalate":
        return {
            "quality_decision": "escalate",
            "quality_feedback": "Responder generation failed or produced invalid content.",
            "final_answer": None
        }
        
    # 2. Check low confidence flag. If retrieval was fallback, we escalate immediately.
    # Regeneration won't help if we don't have the facts.
    if state.get("low_confidence", False):
        return {
            "quality_decision": "escalate",
            "quality_feedback": "Knowledge base retrieval returned the fallback context (low confidence). Escalation required.",
            "final_answer": None
        }
        
    generated_response = state.get("generated_response")
    retrieved_context = state.get("retrieved_context", "")
    user_query = state.get("user_query", "")
    
    if not generated_response:
        return {
            "quality_decision": "escalate",
            "quality_feedback": "No response was generated.",
            "final_answer": None
        }
        
    # 3. Call LLM to evaluate the response
    # We use temperature 0.1 for strict, deterministic evaluation
    llm = get_llm(temperature=0.1)
    
    try:
        structured_llm = llm.with_structured_output(CriticOutput)
        prompt = CRITIC_PROMPT.format(
            retrieved_context=retrieved_context,
            generated_response=generated_response,
            user_query=user_query
        )
        
        result = structured_llm.invoke(prompt)
        
        decision = result.decision.strip().lower()
        feedback = result.feedback.strip()
        
        if decision not in ["accept", "regenerate", "escalate"]:
            decision = "escalate"
            
        final_answer = generated_response if decision == "accept" else None
        
        return {
            "quality_decision": decision,
            "quality_feedback": feedback,
            "final_answer": final_answer
        }
        
    except Exception as e:
        # Fallback to escalate on evaluation exception
        return {
            "quality_decision": "escalate",
            "quality_feedback": f"Critic agent evaluation failed: {str(e)}",
            "final_answer": None
        }
