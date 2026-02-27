"""
supabase_db.py  —  Supabase database abstraction for custom profiles and ticker overrides.

Provides a unified interface to Supabase PostgreSQL with graceful fallback to local JSON files
if Supabase is unavailable. Handles credential loading from Streamlit secrets or environment.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

try:
    from supabase import create_client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Supabase Client Initialization
# ─────────────────────────────────────────────────────────────────────────────

_supabase_client = None
_supabase_initialized = False


def _get_supabase_client():
    """
    Lazily initialize and return Supabase client from Streamlit secrets.
    Returns None if credentials unavailable or Supabase not installed.
    """
    global _supabase_client, _supabase_initialized

    if _supabase_initialized:
        return _supabase_client

    _supabase_initialized = True

    if not SUPABASE_AVAILABLE:
        logger.debug("Supabase not installed, using local JSON fallback")
        return None

    try:
        import streamlit as st
        if "supabase" not in st.secrets:
            logger.debug("Supabase secrets not configured, using local JSON fallback")
            return None

        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]

        _supabase_client = create_client(url, key)
        logger.info("Connected to Supabase")
        return _supabase_client
    except Exception as e:
        logger.warning(f"Failed to connect to Supabase: {e}. Using local JSON fallback.")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Local JSON Fallback (for when Supabase unavailable)
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_PROFILES_FILE = Path.home() / ".trader-bot" / "custom_profiles.json"


def _load_json_data() -> dict:
    """Load custom profiles and ticker overrides from local JSON file."""
    try:
        if CUSTOM_PROFILES_FILE.exists():
            with open(CUSTOM_PROFILES_FILE, "r") as f:
                data = json.load(f)
            data.setdefault("profiles", {})
            data.setdefault("ticker_overrides", {})
            return data
    except Exception as e:
        logger.warning(f"Failed to load local JSON: {e}")
    return {"profiles": {}, "ticker_overrides": {}}


def _save_json_data(data: dict) -> None:
    """Persist custom profiles and ticker overrides to local JSON file."""
    try:
        CUSTOM_PROFILES_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CUSTOM_PROFILES_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save local JSON: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Public API: Custom Profiles
# ─────────────────────────────────────────────────────────────────────────────


def get_custom_profiles() -> dict[str, dict]:
    """Return all user-created custom profiles keyed by name."""
    client = _get_supabase_client()

    if client is None:
        # Fallback to local JSON
        return _load_json_data()["profiles"]

    try:
        response = client.table("custom_profiles").select("*").execute()
        profiles = {}
        for row in response.data:
            name = row["profile_name"]
            profiles[name] = {
                "category": row["category"],
                "rsi_bull_min": row["rsi_bull_min"],
                "rsi_bull_max": row["rsi_bull_max"],
                "adx_threshold": float(row["adx_threshold"]),
                "sl_pct": float(row["sl_pct"]),
                "tp_pct": float(row["tp_pct"]),
                "description": row.get("description") or "",
            }
        return profiles
    except Exception as e:
        logger.warning(f"Failed to fetch profiles from Supabase: {e}. Falling back to JSON.")
        return _load_json_data()["profiles"]


def save_custom_profile(name: str, profile: dict) -> None:
    """Create or update a custom profile by name."""
    client = _get_supabase_client()

    if client is None:
        # Fallback to local JSON
        data = _load_json_data()
        data["profiles"][name] = profile
        _save_json_data(data)
        return

    try:
        client.table("custom_profiles").upsert(
            {
                "profile_name": name,
                "category": profile.get("category", name),
                "rsi_bull_min": int(profile["rsi_bull_min"]),
                "rsi_bull_max": int(profile["rsi_bull_max"]),
                "adx_threshold": float(profile["adx_threshold"]),
                "sl_pct": float(profile["sl_pct"]),
                "tp_pct": float(profile["tp_pct"]),
                "description": profile.get("description", ""),
            }
        ).execute()
        logger.info(f"Saved profile '{name}' to Supabase")
    except Exception as e:
        logger.warning(f"Failed to save profile to Supabase: {e}. Falling back to JSON.")
        data = _load_json_data()
        data["profiles"][name] = profile
        _save_json_data(data)


def delete_custom_profile(name: str) -> None:
    """Delete a custom profile and remove any ticker overrides pointing to it."""
    client = _get_supabase_client()

    if client is None:
        # Fallback to local JSON
        data = _load_json_data()
        data["profiles"].pop(name, None)
        data["ticker_overrides"] = {
            t: k for t, k in data["ticker_overrides"].items() if k != name
        }
        _save_json_data(data)
        return

    try:
        # Cascade delete via foreign key (ticker_overrides will auto-delete)
        client.table("custom_profiles").delete().eq("profile_name", name).execute()
        logger.info(f"Deleted profile '{name}' from Supabase")
    except Exception as e:
        logger.warning(f"Failed to delete profile from Supabase: {e}. Falling back to JSON.")
        data = _load_json_data()
        data["profiles"].pop(name, None)
        data["ticker_overrides"] = {
            t: k for t, k in data["ticker_overrides"].items() if k != name
        }
        _save_json_data(data)


# ─────────────────────────────────────────────────────────────────────────────
# Public API: Ticker Overrides
# ─────────────────────────────────────────────────────────────────────────────


def get_ticker_overrides() -> dict[str, str]:
    """Return the user-defined ticker → profile_key override map."""
    client = _get_supabase_client()

    if client is None:
        # Fallback to local JSON
        return _load_json_data()["ticker_overrides"]

    try:
        response = client.table("ticker_overrides").select("*").execute()
        overrides = {}
        for row in response.data:
            overrides[row["ticker"]] = row["profile_name"]
        return overrides
    except Exception as e:
        logger.warning(f"Failed to fetch ticker overrides from Supabase: {e}. Falling back to JSON.")
        return _load_json_data()["ticker_overrides"]


def set_ticker_override(ticker: str, profile_key_str: str) -> None:
    """Override the profile used for a specific ticker symbol."""
    client = _get_supabase_client()
    ticker = ticker.upper()

    if client is None:
        # Fallback to local JSON
        data = _load_json_data()
        data["ticker_overrides"][ticker] = profile_key_str
        _save_json_data(data)
        return

    try:
        client.table("ticker_overrides").upsert(
            {
                "ticker": ticker,
                "profile_name": profile_key_str,
            }
        ).execute()
        logger.info(f"Set ticker override: {ticker} → {profile_key_str}")
    except Exception as e:
        logger.warning(f"Failed to set ticker override in Supabase: {e}. Falling back to JSON.")
        data = _load_json_data()
        data["ticker_overrides"][ticker] = profile_key_str
        _save_json_data(data)


def remove_ticker_override(ticker: str) -> None:
    """Remove a ticker override, reverting to auto-detection."""
    client = _get_supabase_client()
    ticker = ticker.upper()

    if client is None:
        # Fallback to local JSON
        data = _load_json_data()
        data["ticker_overrides"].pop(ticker, None)
        _save_json_data(data)
        return

    try:
        client.table("ticker_overrides").delete().eq("ticker", ticker).execute()
        logger.info(f"Removed ticker override: {ticker}")
    except Exception as e:
        logger.warning(f"Failed to remove ticker override from Supabase: {e}. Falling back to JSON.")
        data = _load_json_data()
        data["ticker_overrides"].pop(ticker, None)
        _save_json_data(data)


# ─────────────────────────────────────────────────────────────────────────────
# Migration Helper: Sync local JSON to Supabase
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Public API: Watchlist
# ─────────────────────────────────────────────────────────────────────────────


def get_watchlist() -> list[str]:
    """Return the user's watchlist of tickers from Supabase or local JSON fallback."""
    client = _get_supabase_client()

    if client is None:
        # Fallback to local JSON
        return _load_json_data().get("watchlist", [])

    try:
        response = client.table("watchlist").select("tickers").limit(1).execute()
        if response.data and len(response.data) > 0:
            tickers = response.data[0].get("tickers", [])
            return tickers if isinstance(tickers, list) else []
        return []
    except Exception as e:
        logger.warning(f"Failed to fetch watchlist from Supabase: {e}. Falling back to JSON.")
        return _load_json_data().get("watchlist", [])


def save_watchlist(tickers: list[str]) -> None:
    """Save the user's watchlist to Supabase or local JSON fallback."""
    client = _get_supabase_client()

    if client is None:
        # Fallback to local JSON
        data = _load_json_data()
        data["watchlist"] = tickers
        _save_json_data(data)
        return

    try:
        # Upsert: update if exists, insert if not
        client.table("watchlist").upsert(
            {"id": 1, "tickers": tickers}  # Always use id=1 for single watchlist
        ).execute()
        logger.info(f"Saved watchlist with {len(tickers)} tickers to Supabase")
    except Exception as e:
        logger.warning(f"Failed to save watchlist to Supabase: {e}. Falling back to JSON.")
        data = _load_json_data()
        data["watchlist"] = tickers
        _save_json_data(data)


def migrate_json_to_supabase() -> bool:
    """
    One-time migration: sync local JSON to Supabase if Supabase is available.
    Returns True if migration succeeded, False otherwise.
    """
    client = _get_supabase_client()
    if client is None:
        return False

    try:
        json_data = _load_json_data()
        local_profiles = json_data.get("profiles", {})
        local_overrides = json_data.get("ticker_overrides", {})
        local_watchlist = json_data.get("watchlist", [])

        # Check if Supabase is empty
        profiles_count = (
            client.table("custom_profiles").select("id").execute().data
        )
        if profiles_count:
            logger.debug("Supabase already has profiles, skipping migration")
            return True

        # Migrate profiles
        for name, profile in local_profiles.items():
            save_custom_profile(name, profile)

        # Migrate ticker overrides
        for ticker, profile_key in local_overrides.items():
            set_ticker_override(ticker, profile_key)

        # Migrate watchlist
        if local_watchlist:
            save_watchlist(local_watchlist)

        logger.info(f"Migrated {len(local_profiles)} profiles, {len(local_overrides)} overrides, and watchlist to Supabase")
        return True
    except Exception as e:
        logger.warning(f"Migration failed: {e}")
        return False
