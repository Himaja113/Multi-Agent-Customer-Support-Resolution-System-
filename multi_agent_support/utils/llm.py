import os
from dotenv import load_dotenv

load_dotenv()

def get_llm(temperature: float = 0.3):
    """
    Returns an LLM instance. Uses ChatGroq if GROQ_API_KEY is present in the environment,
    otherwise falls back to ChatOllama (local).
    """
    if os.environ.get("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        # Fallback to Groq cloud API
        return ChatGroq(model="llama-3.1-8b-instant", temperature=temperature)
    else:
        from langchain_ollama import ChatOllama
        # Default model is llama3.2:3b
        model_name = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
        return ChatOllama(model=model_name, temperature=temperature)
