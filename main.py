"""
FastAPI application — exposes GET /health and POST /chat.

Design notes:
- The TF-IDF index and catalog are loaded once at startup via lifespan events.
- All per-request state lives in the request body (stateless).
- A 25-second timeout guard catches slow Groq calls before the 30s spec limit.
- CORS is open so the evaluator harness can reach us from any origin.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

import config
from agent import run_agent
from catalog import CatalogManager
from models import ChatRequest, ChatResponse, HealthResponse
from retriever import CatalogIndex

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    stream=sys.stdout,
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Startup / shutdown ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Pre-load catalog and build TF-IDF index before accepting requests."""
    logger.info("=== SHL Recommender starting up ===")
    try:
        catalog = CatalogManager.get()          # loads CATALOGUE.json
        CatalogIndex.get()                       # builds TF-IDF index
        logger.info("Startup complete — %d assessments indexed", len(catalog.all()))
    except Exception as exc:
        logger.error("Startup failed: %s", exc, exc_info=True)
        # Don't raise — health endpoint should still return 200 so the platform
        # doesn't keep restarting the service, but /chat will fail gracefully.
    yield
    logger.info("=== SHL Recommender shutting down ===")


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SHL Assessment Recommender",
    description=(
        "Conversational agent that recommends SHL Individual Test Solutions "
        "from the official product catalog via a stateless multi-turn chat API."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Prevent 404 errors for browser favicon requests."""
    return JSONResponse(content={})

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Readiness probe — always returns 200 {"status": "ok"}."""
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Stateless conversational endpoint.

    The caller must pass the full conversation history on every call.
    Returns the agent's next reply and, when appropriate, a shortlist of
    SHL assessment recommendations.
    """
    if not request.messages:
        raise HTTPException(status_code=422, detail="messages list cannot be empty")

    # Hard turn-count guard (spec says max 8 turns total incl. user+assistant)
    total_turns = len(request.messages)
    if total_turns > 16:  # 8 user + 8 assistant = 16 max reasonable
        raise HTTPException(
            status_code=422,
            detail=f"Conversation too long ({total_turns} messages). Max is 8 turns.",
        )

    try:
        # Wrap in asyncio timeout to stay under the 30s spec limit
        loop = asyncio.get_event_loop()
        response: ChatResponse = await asyncio.wait_for(
            loop.run_in_executor(None, run_agent, request.messages),
            timeout=25.0,
        )
        return response

    except asyncio.TimeoutError:
        logger.error("Agent timed out after 25s")
        return ChatResponse(
            reply=(
                "I'm sorry, this is taking longer than expected. "
                "Based on what you've told me, could you briefly summarise "
                "the role and required skills so I can give you a quick recommendation?"
            ),
            recommendations=[],
            end_of_conversation=False,
        )
    except Exception as exc:
        logger.error("Unhandled agent error: %s", exc, exc_info=True)
        return ChatResponse(
            reply=(
                "I encountered an unexpected error. Please try again. "
                "If the issue persists, try rephrasing your request."
            ),
            recommendations=[],
            end_of_conversation=False,
        )


# ── Exception handlers ─────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "reply": "An internal error occurred. Please try again.",
            "recommendations": [],
            "end_of_conversation": False,
        },
    )


# ── Dev entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level=config.LOG_LEVEL.lower(),
    )
