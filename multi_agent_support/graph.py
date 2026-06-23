from langgraph.graph import StateGraph, END
from state import SupportState
from agents.classifier import classify_query
from agents.retriever import retrieve_knowledge
from agents.responder import generate_response
from agents.critic import quality_check
from agents.escalator import handle_escalation

def critic_router(state: SupportState) -> str:
    """
    Determines the next node from the Critic.
    Acts as the controller/router of the graph.
    """
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    quality_decision = state.get("quality_decision")
    
    # Graph-level hard cap check on retries
    if retry_count > max_retries:
        return "escalate"
        
    if quality_decision == "accept":
        return "accept"
    elif quality_decision == "regenerate":
        return "regenerate"
    elif quality_decision == "escalate":
        return "escalate"
    else:
        # Default fallback route
        return "escalate"

def build_graph() -> StateGraph:
    """Assembles and compiles the multi-agent LangGraph workflow."""
    workflow = StateGraph(SupportState)
    
    # 1. Add Nodes
    workflow.add_node("classify", classify_query)
    workflow.add_node("retrieve", retrieve_knowledge)
    workflow.add_node("respond", generate_response)
    workflow.add_node("critic", quality_check)
    workflow.add_node("escalate", handle_escalation)
    
    # 2. Set Entry Point
    workflow.set_entry_point("classify")
    
    # 3. Add Fixed Edges
    workflow.add_edge("classify", "retrieve")
    workflow.add_edge("retrieve", "respond")
    workflow.add_edge("respond", "critic")
    workflow.add_edge("escalate", END)
    
    # 4. Add Conditional Edge from Critic
    workflow.add_conditional_edges(
        "critic",
        critic_router,
        {
            "accept": END,
            "regenerate": "respond",
            "escalate": "escalate"
        }
    )
    
    return workflow.compile()
