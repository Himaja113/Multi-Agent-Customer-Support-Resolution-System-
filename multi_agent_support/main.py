import os
from dotenv import load_dotenv
from graph import build_graph

# Load environment variables (such as GROQ_API_KEY from .env)
load_dotenv()

DEMO_QUERIES = [
    {"query": "I was charged twice for my subscription."},
    {"query": "How do I reset my password."},
    {"query": "My order hasn't arrived yet."},
    {"query": "My account is having problems."},
    {"query": "What is your business address?"},  # Guarantees low_confidence escalation
    {"query": "Can I cancel my subscription?"}    # Guarantees max_retries_exceeded escalation
]

def run_demo():
    print("="*70)
    print("Starting Multi-Agent Support System Demo (LangGraph + Llama3/Groq)")
    print("="*70)
    
    # Check that we have a model API key or Ollama set up
    has_groq = bool(os.environ.get("GROQ_API_KEY"))
    if has_groq:
        print("[LLM Backend] Using Groq (Llama 3 8B) for fast, key-based execution.")
    else:
        print("[LLM Backend] GROQ_API_KEY not found. Attempting local Ollama (Llama 3) connection.")
    print("="*70)
    
    app = build_graph()
    
    for i, query_data in enumerate(DEMO_QUERIES, 1):
        query = query_data["query"]
        max_retries = query_data.get("max_retries", 2)
        print(f"\n\n{'#'*70}")
        print(f"RUNNING DEMO QUERY {i}: \"{query}\"")
        print(f"{'#'*70}")
        
        # Initial State as defined in state.py and Phase 8.5
        initial_state = {
            "user_query": query,
            "category": None,
            "intent": None,
            "entities": {},
            "retrieved_context": None,
            "low_confidence": False,
            "generated_response": None,
            "quality_decision": None,
            "quality_feedback": None,
            "retry_count": 0,
            "max_retries": max_retries,
            "escalated": False,
            "escalation_reason": None,
            "final_answer": None
        }
        
        # Run graph streaming to monitor routing and agent decisions in real-time
        current_state = dict(initial_state)
        
        for event in app.stream(initial_state):
            for node_name, state_update in event.items():
                print(f"\n--- [Node: {node_name}] ---")
                
                # Check and print update fields for each node
                if "category" in state_update:
                    print(f"  Category:           {state_update.get('category')}")
                    print(f"  Intent:             {state_update.get('intent')}")
                    print(f"  Entities:           {state_update.get('entities')}")
                
                if "retrieved_context" in state_update:
                    ctx = state_update.get("retrieved_context") or ""
                    print(f"  Retrieved Context:  {ctx[:100]}...")
                    print(f"  Low Confidence:    {state_update.get('low_confidence')}")
                
                if "generated_response" in state_update:
                    resp = state_update.get("generated_response") or ""
                    print(f"  Generated Response: {resp[:150]}...")
                    print(f"  Retry Count:        {state_update.get('retry_count')}")
                
                if "quality_decision" in state_update:
                    print(f"  Quality Decision:   {state_update.get('quality_decision')}")
                    print(f"  Quality Feedback:   {state_update.get('quality_feedback')}")
                
                if "escalated" in state_update:
                    print(f"  Escalated:          {state_update.get('escalated')}")
                    print(f"  Escalation Reason:  {state_update.get('escalation_reason')}")
                
                # Accumulate the changes into the current state
                current_state.update(state_update)
        
        # Print the Final Output trace
        print("\n" + "="*50)
        print("FINAL RESULTS:")
        print(f"Query:      {current_state['user_query']}")
        print(f"Escalated:  {current_state['escalated']}")
        if current_state['escalated']:
            print(f"Reason:     {current_state['escalation_reason']}")
        print(f"Final Answer:\n{current_state['final_answer']}")
        print("="*50)

if __name__ == "__main__":
    run_demo()
