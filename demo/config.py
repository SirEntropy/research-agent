import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

LLM_CONFIG = {
    "model": "gpt-4.1",
    "api_key": OPENAI_API_KEY,
    "temperature": 0.5,
    "max_tokens": 2000,
}

# arXiv API Configuration
ARXIV_MAX_RESULTS = 10  # Limit for demo purposes
ARXIV_CATEGORIES = [
    "cs.AI",  # Artificial Intelligence
    "cs.LG",  # Machine Learning
    "cs.CV",  # Computer Vision
    "cs.CL",  # Computation and Language
    "cs.NE",  # Neural and Evolutionary Computing
]

# Demo Settings
DEMO_MODE = True
VERBOSE = True
