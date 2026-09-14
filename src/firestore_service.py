"""
Firestore integration service for Macro Pulse (hf-macro-dashboard).

Backs per-user personalized watchlists for the MCP server. Same project
as src/bigquery_service.py (agentichackathon-506620) but a separate GCP
service — Firestore is a point read/write NoSQL store, a much better fit
for "look up/save one small document per user" than BigQuery, which is
built for analytical scans over large tables.

user_id is currently just a plain string tool argument — there is no
real authentication behind it yet (see mcp_server.py's docstrings).
This is intentional: Alexa+ hasn't published how it passes account
identity to a self-hosted MCP server, so this ships trust-based for now
and should move to validating a real identity (e.g. an OAuth subject
claim) once that's documented, without changing the storage shape.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional

DEFAULT_GCP_PROJECT = os.getenv("GCP_PROJECT", os.getenv("GOOGLE_CLOUD_PROJECT", "agentichackathon-506620"))
COLLECTION = "watchlists"
MAX_TICKERS = 50  # keep a single user's list bounded

_FIRESTORE_AVAILABLE = False
try:
    from google.cloud import firestore
    _FIRESTORE_AVAILABLE = True
except ImportError:
    firestore = None  # type: ignore

_client = None  # lazily constructed, reused across calls in this process


def is_firestore_available() -> bool:
    """Return True if google-cloud-firestore is installed."""
    return _FIRESTORE_AVAILABLE


def get_client(project_id: Optional[str] = None):
    """Obtain (and cache) an authenticated Firestore client, or None if
    the package isn't installed or credentials aren't available."""
    global _client
    if not _FIRESTORE_AVAILABLE:
        return None
    if _client is not None:
        return _client
    try:
        _client = firestore.Client(project=project_id or DEFAULT_GCP_PROJECT)
        return _client
    except Exception:
        return None


def _normalize(tickers: list[str]) -> list[str]:
    seen = []
    for t in tickers:
        t = t.strip().upper()
        if t and t not in seen:
            seen.append(t)
    return seen[:MAX_TICKERS]


def get_watchlist(user_id: str) -> list[str]:
    """Return the user's saved tickers, or [] if they have none saved yet
    (or Firestore isn't reachable — callers should fall back gracefully,
    not treat this as an error)."""
    client = get_client()
    if client is None or not user_id:
        return []
    try:
        doc = client.collection(COLLECTION).document(user_id).get()
        if not doc.exists:
            return []
        return list(doc.to_dict().get("tickers", []))
    except Exception:
        return []


def save_watchlist(user_id: str, tickers: list[str]) -> list[str]:
    """Overwrite the user's watchlist with exactly this set of tickers.
    Returns the normalized (deduped, uppercased, capped) list that was
    actually saved. Raises on a real Firestore error — callers decide
    how to surface that, unlike the read path which fails soft."""
    if not user_id:
        raise ValueError("user_id is required")
    client = get_client()
    if client is None:
        raise RuntimeError("Firestore is not available (not installed, or no GCP credentials)")

    normalized = _normalize(tickers)
    client.collection(COLLECTION).document(user_id).set({
        "tickers": normalized,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    return normalized


def add_ticker(user_id: str, ticker: str) -> list[str]:
    """Add one ticker to the user's watchlist (no-op if already present).
    Returns the resulting list."""
    current = get_watchlist(user_id)
    return save_watchlist(user_id, current + [ticker])


def remove_ticker(user_id: str, ticker: str) -> list[str]:
    """Remove one ticker from the user's watchlist (no-op if absent).
    Returns the resulting list."""
    ticker = ticker.strip().upper()
    current = get_watchlist(user_id)
    return save_watchlist(user_id, [t for t in current if t != ticker])
