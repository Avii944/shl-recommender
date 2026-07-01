# SHL Assessment Recommender

A conversational FastAPI service that helps hiring managers find the right SHL
Individual Test Solutions through dialogue.

## Quick Start (Local)

```bash
# 1. Clone / navigate to the project
cd shl_recommender

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment
copy .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Run the server
python main.py
```

Server starts at http://localhost:8000

## API Reference

### GET /health
Returns `{"status": "ok"}` with HTTP 200.

### POST /chat
Stateless conversational endpoint. Send full history every call.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "What seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are 5 assessments that fit a mid-level Java dev with stakeholder needs.",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

## Deploy to Render (Free Tier)

1. Push this folder to a GitHub repository
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Set `GROQ_API_KEY` as an environment variable (secret) in the Render dashboard
5. Render will auto-detect `render.yaml` and deploy

The free tier has a cold-start delay (~60s). The evaluator's `/health` timeout
of 2 minutes accommodates this.

## Project Structure

```
shl_recommender/
├── main.py          # FastAPI app, /health, /chat endpoints
├── agent.py         # LLM agent logic (Groq), prompt templates
├── catalog.py       # Catalog loading, cleaning, indexing
├── retriever.py     # TF-IDF + keyword boost retrieval
├── models.py        # Pydantic request/response models
├── config.py        # Settings from env vars
├── CATALOGUE.json   # SHL product catalog (377 assessments)
├── requirements.txt
├── render.yaml      # Render.com deployment config
└── .env.example
```

## Test the endpoints locally

```bash
# Health
curl http://localhost:8000/health

# Vague query (should clarify)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I need an assessment"}]}'

# Specific role (should recommend)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [
    {"role": "user", "content": "Hiring a Java developer"},
    {"role": "assistant", "content": "What seniority level?"},
    {"role": "user", "content": "Mid-level, 4 years experience, works with stakeholders"}
  ]}'

# Comparison
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is the difference between OPQ32r and GSA?"}]}'

# Off-topic (should refuse)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Give me legal advice on hiring"}]}'
```
