import os
from dotenv import load_dotenv

load_dotenv()

def get_llm(temperature: float = 0.3):
    """
    Returns an LLM instance. Uses ChatGroq if GROQ_API_KEY is present in the environment
    (for immediate out-of-the-box demo execution), otherwise falls back to ChatOllama.
    """
    if os.environ.get("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        # We use llama-3.1-8b-instant as recommended in the Build Plan (Llama3 8B)
        return ChatGroq(model="llama-3.1-8b-instant", temperature=temperature)
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(model="llama3", temperature=temperature)
