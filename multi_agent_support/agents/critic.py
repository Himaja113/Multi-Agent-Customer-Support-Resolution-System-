from typing import Dict, Any
from pydantic import BaseModel, Field
from state import SupportState
from utils.llm import get_llm

class CriticOutput(BaseModel):
    correctness: int = Field(description="Score from 1 to 5 for correctness (accuracy and grounding in retrieved context). Must be an integer.")
    completeness: int = Field(description="Score from 1 to 5 for completeness (addressing the query fully with next steps). Must be an integer.")
    tone: int = Field(description="Score from 1 to 5 for tone (empathy, politeness, warmth, and professionalism). Must be an integer.")
    feedback: str = Field(description="Detailed evaluation feedback focusing on correctness, completeness, and tone.")

CRITIC_PROMPT = """You are a Quality Check (Critic) Agent for a customer support system.
Your job is to evaluate the generated response against the customer query and the retrieved context by scoring three criteria from 1 to 5:

1. Correctness: Is the response fully grounded in the retrieved context? Does it avoid adding outside facts or contradicting the context? (1 = completely ungrounded or contradictory, 5 = fully grounded)
2. Completeness: Does it address the customer's query completely? Does it include specific next steps as outlined in the context? (1 = does not answer query, 5 = fully answers all aspects of query)
3. Tone: Is it professional, polite, warm, and empathetic? Does it feel human rather than robotic or dismissive? (1 = rude/cold/robotic, 5 = extremely polite, warm, and professional)

Retrieved Context:
{retrieved_context}

Generated Response:
{generated_response}

Customer Query:
{user_query}
"""

def quality_check(state: SupportState) -> Dict[str, Any]:
    """
    Evaluates the generated response for correctness, completeness, and tone using a weighted scoring rubric.
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
        
        # Clamp scores to 1-5 range just in case
        correctness = max(1, min(5, result.correctness))
        completeness = max(1, min(5, result.completeness))
        tone = max(1, min(5, result.tone))
        feedback = result.feedback.strip()
        
        # Calculate weighted average score
        weighted_score = (correctness * 0.4) + (completeness * 0.4) + (tone * 0.2)
        
        # Determine decision based on weighted score
        if weighted_score >= 4.0:
            decision = "accept"
        elif weighted_score >= 2.5:
            decision = "regenerate"
        else:
            decision = "escalate"
            
        # Log evaluation details to stdout for developer visibility
        print(f"\n[Critic Scoring Evaluator]")
        print(f"  - Correctness (40%):  {correctness}/5")
        print(f"  - Completeness (40%): {completeness}/5")
        print(f"  - Tone (20%):         {tone}/5")
        print(f"  - Weighted Score:     {weighted_score:.2f}/5.00")
        print(f"  - Decision:           {decision.upper()}")
        
        # If regenerating, target the lowest scoring criterion in the feedback
        if decision == "regenerate":
            scores = {
                "Correctness": correctness,
                "Completeness": completeness,
                "Tone": tone
            }
            lowest_criterion = min(scores, key=scores.get)
            lowest_score = scores[lowest_criterion]
            feedback = f"[Target: {lowest_criterion} (Score: {lowest_score}/5)] {feedback}"
            
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
