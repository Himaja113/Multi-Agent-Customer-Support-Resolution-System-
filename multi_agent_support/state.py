from typing import TypedDict, Optional, Dict, Any

class SupportState(TypedDict):
    user_query: str                       # Raw input — never modified after set
    category: Optional[str]                # Set by Classifier
    intent: Optional[str]                  # Set by Classifier
    entities: Dict[str, Any]               # Set by Classifier (order IDs, emails, etc.)
    retrieved_context: Optional[str]      # Set by Retriever
    low_confidence: bool                  # Set by Retriever — True if only fallback found
    generated_response: Optional[str]     # Set by Responder
    quality_decision: Optional[str]       # Set by Critic: "accept"/"regenerate"/"escalate"
    quality_feedback: Optional[str]       # Set by Critic — passed back to Responder on retry
    retry_count: int                      # Incremented by Responder, starts at 0
    max_retries: int                      # Default 2 — set at initialization, lives in state
    escalated: bool                       # Set by Escalation Handler
    escalation_reason: Optional[str]      # Set by Escalation Handler
    final_answer: Optional[str]           # Set by Critic (accept) or Escalation Handler
