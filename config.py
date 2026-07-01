"""
Configuration — loads from environment variables or .env file.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present (for local dev)
load_dotenv(Path(__file__).parent / ".env")

# ── Groq ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

# ── Catalog ───────────────────────────────────────────────────────────────────
CATALOG_PATH: Path = Path(__file__).parent / "CATALOGUE.json"

# ── Agent behaviour ───────────────────────────────────────────────────────────
MAX_RECOMMENDATIONS: int = 10
MIN_RECOMMENDATIONS: int = 1
MAX_TURNS: int = 8          # Hard cap from spec
FORCE_RECOMMEND_TURN: int = 6  # Force a recommendation after this many user turns

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K: int = 30   # TF-IDF candidates before LLM re-ranks
FINAL_TOP_K: int = 10       # Max items returned to user

# ── Safety ────────────────────────────────────────────────────────────────────
REFUSED_TOPICS = [
    "legal", "lawsuit", "discrimination", "gdpr", "compliance",
    "salary", "pay", "compensation", "visa", "immigration",
    "how to cheat", "bypass", "ignore previous", "ignore instructions",
    "forget everything", "jailbreak", "act as", "pretend you are",
    "you are now", "disregard",
]

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")

# ── Test type key mapping ──────────────────────────────────────────────────────
KEY_TO_TYPE_CODE: dict[str, str] = {
    "Ability & Aptitude": "A",
    "Assessment Exercises": "E",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

TYPE_CODE_TO_KEY: dict[str, str] = {v: k for k, v in KEY_TO_TYPE_CODE.items()}
