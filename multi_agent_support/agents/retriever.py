import os
import json
from typing import Dict, Any
from state import SupportState

def load_knowledge_base() -> Dict[str, Any]:
    """Loads the kb.json knowledge base file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # kb.json is in ../knowledge_base/kb.json relative to this file
    kb_path = os.path.join(current_dir, "..", "knowledge_base", "kb.json")
    
    with open(kb_path, "r", encoding="utf-8") as f:
        return json.load(f)

def retrieve_knowledge(state: SupportState) -> Dict[str, Any]:
    """
    Retrieves facts from the static kb.json mock knowledge base
    based on the state's category and intent.
    """
    kb = load_knowledge_base()
    category = state.get("category")
    intent = state.get("intent")
    
    # Defaults in case of missing input
    if not category or not intent:
        fallback_text = kb.get("general", {}).get("fallback", "Please contact support.")
        return {
            "retrieved_context": fallback_text,
            "low_confidence": True
        }
    
    category = category.lower().strip()
    intent = intent.lower().strip()
    
    # If category or intent indicates general fallback, mark as low confidence
    if category == "general" or intent == "fallback":
        fallback_text = kb.get("general", {}).get("fallback", "Please contact support.")
        return {
            "retrieved_context": fallback_text,
            "low_confidence": True
        }
    
    # Level 1: Exact Match
    if category in kb and intent in kb[category]:
        return {
            "retrieved_context": kb[category][intent],
            "low_confidence": False
        }
        
    # Level 2: Substring/Word-Overlap Match
    if category in kb:
        intent_words = set(w.strip() for w in intent.replace("_", " ").split() if w.strip())
        for key in kb[category].keys():
            key_words = set(kw.strip() for kw in key.replace("_", " ").split() if kw.strip())
            # If any word from the intent matches a word in the KB key
            if intent_words.intersection(key_words):
                return {
                    "retrieved_context": kb[category][key],
                    "low_confidence": False
                }
                
    # Level 3: General Fallback
    fallback_text = kb.get("general", {}).get("fallback", "Please contact support.")
    return {
        "retrieved_context": fallback_text,
        "low_confidence": True
    }
