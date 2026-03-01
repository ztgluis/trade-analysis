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


def _is_nested_profiles(profiles_data: dict) -> bool:
    """
    Detect whether profiles data uses the new nested format.

    New format: {"default": {"profile_name": {...}}, "workspace_id": {...}}
    Old format: {"profile_name": {...}}

    We detect old format when none of the top-level values is a dict-of-dicts.
    """
    if not profiles_data:
        return True  # empty — treat as new format
    for v in profiles_data.values():
        if isinstance(v, dict):
            # If the value contains profile keys (rsi_bull_min etc.), it is old flat format
            if "rsi_bull_min" in v or "category" in v:
                return False
    return True


def _get_profiles_for_workspace(data: dict, workspace_id: str) -> dict:
    """Extract profiles dict for a given workspace, handling old flat format."""
    profiles_raw = data.get("profiles", {})
    if _is_nested_profiles(profiles_raw):
        return profiles_raw.get(workspace_id, {})
    else:
        # Old flat format — belongs to "default" workspace
        if workspace_id == "default":
            return profiles_raw
        return {}


def _get_overrides_for_workspace(data: dict, workspace_id: str) -> dict:
    """Extract ticker_overrides dict for a given workspace, handling old flat format."""
    overrides_raw = data.get("ticker_overrides", {})
    if not overrides_raw:
        return {}
    # Detect old flat format: values are strings (profile keys), not dicts
    first_val = next(iter(overrides_raw.values()), None)
    if isinstance(first_val, str):
        # Old flat format — belongs to "default" workspace
        if workspace_id == "default":
            return overrides_raw
        return {}
    return overrides_raw.get(workspace_id, {})


def _get_watchlist_for_workspace(data: dict, workspace_id: str) -> list:
    """Extract watchlist for a given workspace, handling old flat format."""
    watchlist_raw = data.get("watchlist", {})
    if isinstance(watchlist_raw, list):
        # Old flat format — belongs to "default" workspace
        if workspace_id == "default":
            return watchlist_raw
        return []
    return watchlist_raw.get(workspace_id, [])


def _get_strategies_for_workspace(data: dict, workspace_id: str) -> list:
    """Extract saved strategies list for a given workspace from JSON data."""
    strats_raw = data.get("saved_strategies", {})
    if isinstance(strats_raw, dict):
        return strats_raw.get(workspace_id, [])
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Public API: Custom Profiles
# ─────────────────────────────────────────────────────────────────────────────


def get_custom_profiles(workspace_id: str = "default") -> dict[str, dict]:
    """Return all user-created custom profiles keyed by name."""
    client = _get_supabase_client()

    if client is None:
        # Fallback to local JSON
        return _get_profiles_for_workspace(_load_json_data(), workspace_id)

    try:
        response = (
            client.table("custom_profiles")
            .select("*")
            .eq("workspace_id", workspace_id)
            .execute()
        )
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
        return _get_profiles_for_workspace(_load_json_data(), workspace_id)


def save_custom_profile(name: str, profile: dict, workspace_id: str = "default") -> None:
    """Create or update a custom profile by name."""
    client = _get_supabase_client()

    if client is None:
        # Fallback to local JSON
        data = _load_json_data()
        profiles_raw = data.get("profiles", {})
        # Migrate to nested format if needed
        if not _is_nested_profiles(profiles_raw):
            data["profiles"] = {"default": profiles_raw}
        data["profiles"].setdefault(workspace_id, {})
        data["profiles"][workspace_id][name] = profile
        _save_json_data(data)
        return

    try:
        client.table("custom_profiles").upsert(
            {
                "profile_name": name,
                "workspace_id": workspace_id,
                "category": profile.get("category", name),
                "rsi_bull_min": int(profile["rsi_bull_min"]),
                "rsi_bull_max": int(profile["rsi_bull_max"]),
                "adx_threshold": float(profile["adx_threshold"]),
                "sl_pct": float(profile["sl_pct"]),
                "tp_pct": float(profile["tp_pct"]),
                "description": profile.get("description", ""),
            }
        ).execute()
        logger.info(f"Saved profile '{name}' to Supabase (workspace={workspace_id})")
    except Exception as e:
        logger.warning(f"Failed to save profile to Supabase: {e}. Falling back to JSON.")
        data = _load_json_data()
        profiles_raw = data.get("profiles", {})
        if not _is_nested_profiles(profiles_raw):
            data["profiles"] = {"default": profiles_raw}
        data["profiles"].setdefault(workspace_id, {})
        data["profiles"][workspace_id][name] = profile
        _save_json_data(data)


def delete_custom_profile(name: str, workspace_id: str = "default") -> None:
    """Delete a custom profile and remove any ticker overrides pointing to it."""
    client = _get_supabase_client()

    if client is None:
        # Fallback to local JSON
        data = _load_json_data()
        profiles_raw = data.get("profiles", {})
        if _is_nested_profiles(profiles_raw):
            data["profiles"].setdefault(workspace_id, {})
            data["profiles"][workspace_id].pop(name, None)
        else:
            # Old flat format — only "default" workspace exists
            if workspace_id == "default":
                data["profiles"].pop(name, None)

        # Clean up ticker overrides for this workspace
        overrides_raw = data.get("ticker_overrides", {})
        first_val = next(iter(overrides_raw.values()), None) if overrides_raw else None
        if isinstance(first_val, str):
            # Old flat format
            if workspace_id == "default":
                data["ticker_overrides"] = {
                    t: k for t, k in overrides_raw.items() if k != name
                }
        else:
            ws_overrides = overrides_raw.get(workspace_id, {})
            data["ticker_overrides"][workspace_id] = {
                t: k for t, k in ws_overrides.items() if k != name
            }
        _save_json_data(data)
        return

    try:
        # Cascade delete via workspace-scoped filter
        client.table("custom_profiles").delete().eq("profile_name", name).eq("workspace_id", workspace_id).execute()
        logger.info(f"Deleted profile '{name}' from Supabase (workspace={workspace_id})")
    except Exception as e:
        logger.warning(f"Failed to delete profile from Supabase: {e}. Falling back to JSON.")
        data = _load_json_data()
        profiles_raw = data.get("profiles", {})
        if _is_nested_profiles(profiles_raw):
            data["profiles"].setdefault(workspace_id, {})
            data["profiles"][workspace_id].pop(name, None)
        else:
            if workspace_id == "default":
                data["profiles"].pop(name, None)

        overrides_raw = data.get("ticker_overrides", {})
        first_val = next(iter(overrides_raw.values()), None) if overrides_raw else None
        if isinstance(first_val, str):
            if workspace_id == "default":
                data["ticker_overrides"] = {
                    t: k for t, k in overrides_raw.items() if k != name
                }
        else:
            ws_overrides = overrides_raw.get(workspace_id, {})
            data["ticker_overrides"][workspace_id] = {
                t: k for t, k in ws_overrides.items() if k != name
            }
        _save_json_data(data)


# ─────────────────────────────────────────────────────────────────────────────
# Public API: Ticker Overrides
# ─────────────────────────────────────────────────────────────────────────────


def get_ticker_overrides(workspace_id: str = "default") -> dict[str, str]:
    """Return the user-defined ticker → profile_key override map."""
    client = _get_supabase_client()

    if client is None:
        # Fallback to local JSON
        return _get_overrides_for_workspace(_load_json_data(), workspace_id)

    try:
        response = (
            client.table("ticker_overrides")
            .select("*")
            .eq("workspace_id", workspace_id)
            .execute()
        )
        overrides = {}
        for row in response.data:
            overrides[row["ticker"]] = row["profile_name"]
        return overrides
    except Exception as e:
        logger.warning(f"Failed to fetch ticker overrides from Supabase: {e}. Falling back to JSON.")
        return _get_overrides_for_workspace(_load_json_data(), workspace_id)


def set_ticker_override(ticker: str, profile_key_str: str, workspace_id: str = "default") -> None:
    """Override the profile used for a specific ticker symbol."""
    client = _get_supabase_client()
    ticker = ticker.upper()

    if client is None:
        # Fallback to local JSON
        data = _load_json_data()
        overrides_raw = data.get("ticker_overrides", {})
        first_val = next(iter(overrides_raw.values()), None) if overrides_raw else None
        if isinstance(first_val, str):
            # Old flat format — migrate to nested
            data["ticker_overrides"] = {"default": overrides_raw}
        data["ticker_overrides"].setdefault(workspace_id, {})
        data["ticker_overrides"][workspace_id][ticker] = profile_key_str
        _save_json_data(data)
        return

    try:
        client.table("ticker_overrides").upsert(
            {
                "ticker": ticker,
                "profile_name": profile_key_str,
                "workspace_id": workspace_id,
            }
        ).execute()
        logger.info(f"Set ticker override: {ticker} → {profile_key_str} (workspace={workspace_id})")
    except Exception as e:
        logger.warning(f"Failed to set ticker override in Supabase: {e}. Falling back to JSON.")
        data = _load_json_data()
        overrides_raw = data.get("ticker_overrides", {})
        first_val = next(iter(overrides_raw.values()), None) if overrides_raw else None
        if isinstance(first_val, str):
            data["ticker_overrides"] = {"default": overrides_raw}
        data["ticker_overrides"].setdefault(workspace_id, {})
        data["ticker_overrides"][workspace_id][ticker] = profile_key_str
        _save_json_data(data)


def remove_ticker_override(ticker: str, workspace_id: str = "default") -> None:
    """Remove a ticker override, reverting to auto-detection."""
    client = _get_supabase_client()
    ticker = ticker.upper()

    if client is None:
        # Fallback to local JSON
        data = _load_json_data()
        overrides_raw = data.get("ticker_overrides", {})
        first_val = next(iter(overrides_raw.values()), None) if overrides_raw else None
        if isinstance(first_val, str):
            # Old flat format
            if workspace_id == "default":
                data["ticker_overrides"].pop(ticker, None)
        else:
            data["ticker_overrides"].setdefault(workspace_id, {})
            data["ticker_overrides"][workspace_id].pop(ticker, None)
        _save_json_data(data)
        return

    try:
        client.table("ticker_overrides").delete().eq("ticker", ticker).eq("workspace_id", workspace_id).execute()
        logger.info(f"Removed ticker override: {ticker} (workspace={workspace_id})")
    except Exception as e:
        logger.warning(f"Failed to remove ticker override from Supabase: {e}. Falling back to JSON.")
        data = _load_json_data()
        overrides_raw = data.get("ticker_overrides", {})
        first_val = next(iter(overrides_raw.values()), None) if overrides_raw else None
        if isinstance(first_val, str):
            if workspace_id == "default":
                data["ticker_overrides"].pop(ticker, None)
        else:
            data["ticker_overrides"].setdefault(workspace_id, {})
            data["ticker_overrides"][workspace_id].pop(ticker, None)
        _save_json_data(data)


# ─────────────────────────────────────────────────────────────────────────────
# Public API: Watchlist
# ─────────────────────────────────────────────────────────────────────────────


def get_watchlist(workspace_id: str = "default") -> list[str]:
    """Return the user's watchlist of tickers from Supabase or local JSON fallback."""
    client = _get_supabase_client()

    if client is None:
        logger.debug("Supabase not available, loading watchlist from local JSON")
        return _get_watchlist_for_workspace(_load_json_data(), workspace_id)

    try:
        response = (
            client.table("watchlist")
            .select("tickers")
            .eq("workspace_id", workspace_id)
            .limit(1)
            .execute()
        )
        if response.data and len(response.data) > 0:
            tickers = response.data[0].get("tickers", [])
            logger.info(f"Loaded watchlist from Supabase (workspace={workspace_id}): {tickers}")
            return tickers if isinstance(tickers, list) else []
        logger.debug("Watchlist table is empty, returning empty list")
        return []
    except Exception as e:
        logger.warning(f"Failed to fetch watchlist from Supabase: {e}. Falling back to JSON.")
        return _get_watchlist_for_workspace(_load_json_data(), workspace_id)


def save_watchlist(tickers: list[str], workspace_id: str = "default") -> None:
    """Save the user's watchlist to Supabase or local JSON fallback."""
    client = _get_supabase_client()

    if client is None:
        # Fallback to local JSON
        data = _load_json_data()
        data.setdefault("watchlist", {})
        if isinstance(data["watchlist"], list):
            # Old flat format — migrate to nested
            data["watchlist"] = {"default": data["watchlist"]}
        data["watchlist"][workspace_id] = tickers
        _save_json_data(data)
        return

    try:
        # Upsert keyed on workspace_id (not a hardcoded id=1)
        client.table("watchlist").upsert(
            {"workspace_id": workspace_id, "tickers": tickers}
        ).execute()
        logger.info(f"Saved watchlist with {len(tickers)} tickers to Supabase (workspace={workspace_id})")
    except Exception as e:
        logger.warning(f"Failed to save watchlist to Supabase: {e}. Falling back to JSON.")
        data = _load_json_data()
        data.setdefault("watchlist", {})
        if isinstance(data["watchlist"], list):
            data["watchlist"] = {"default": data["watchlist"]}
        data["watchlist"][workspace_id] = tickers
        _save_json_data(data)


# ─────────────────────────────────────────────────────────────────────────────
# Migration Helper: Sync local JSON to Supabase
# ─────────────────────────────────────────────────────────────────────────────


def migrate_json_to_supabase(workspace_id: str = "default") -> bool:
    """
    One-time migration: sync local JSON to Supabase if Supabase is available.
    Returns True if migration succeeded, False otherwise.
    """
    client = _get_supabase_client()
    if client is None:
        return False

    try:
        json_data = _load_json_data()
        local_profiles = _get_profiles_for_workspace(json_data, workspace_id)
        local_overrides = _get_overrides_for_workspace(json_data, workspace_id)
        local_watchlist = _get_watchlist_for_workspace(json_data, workspace_id)

        # Check if Supabase is empty for this workspace
        profiles_count = (
            client.table("custom_profiles").select("id").eq("workspace_id", workspace_id).execute().data
        )
        if profiles_count:
            logger.debug("Supabase already has profiles, skipping migration")
            return True

        # Migrate profiles
        for name, profile in local_profiles.items():
            save_custom_profile(name, profile, workspace_id)

        # Migrate ticker overrides
        for ticker, profile_key in local_overrides.items():
            set_ticker_override(ticker, profile_key, workspace_id)

        # Migrate watchlist
        if local_watchlist:
            save_watchlist(local_watchlist, workspace_id)

        logger.info(
            f"Migrated {len(local_profiles)} profiles, {len(local_overrides)} overrides, "
            f"and watchlist to Supabase (workspace={workspace_id})"
        )
        return True
    except Exception as e:
        logger.warning(f"Migration failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Public API: Saved Strategies
# ─────────────────────────────────────────────────────────────────────────────


def get_saved_strategies(workspace_id: str = "default") -> list[dict]:
    """Return all saved strategies for this workspace, newest first."""
    client = _get_supabase_client()

    if client is None:
        return _get_strategies_for_workspace(_load_json_data(), workspace_id)

    try:
        response = (
            client.table("saved_strategies")
            .select("*")
            .eq("workspace_id", workspace_id)
            .order("updated_at", desc=True)
            .execute()
        )
        return response.data or []
    except Exception as e:
        logger.warning(f"Failed to fetch saved strategies from Supabase: {e}. Falling back to JSON.")
        return _get_strategies_for_workspace(_load_json_data(), workspace_id)


def save_strategy(
    name: str,
    code: str,
    metadata: dict,
    workspace_id: str = "default",
) -> None:
    """
    Save (upsert) a generated Pine Script strategy.

    Parameters
    ----------
    name         : human-readable strategy name (unique per workspace)
    code         : full Pine Script v6 source code
    metadata     : dict with keys profile_name, indicators, entry_mode, timeframe
    workspace_id : workspace scope
    """
    client = _get_supabase_client()

    row = {
        "workspace_id": workspace_id,
        "name":         name,
        "code":         code,
        "profile_name": metadata.get("profile_name", ""),
        "indicators":   metadata.get("indicators", []),
        "entry_mode":   metadata.get("entry_mode", "all_signals"),
        "timeframe":    metadata.get("timeframe", "D"),
    }

    if client is None:
        data  = _load_json_data()
        strats = data.setdefault("saved_strategies", {}).setdefault(workspace_id, [])
        for i, s in enumerate(strats):
            if s.get("name") == name:
                strats[i] = row
                _save_json_data(data)
                return
        strats.append(row)
        _save_json_data(data)
        return

    try:
        client.table("saved_strategies").upsert(
            row, on_conflict="name,workspace_id"
        ).execute()
        logger.info(f"Saved strategy '{name}' to Supabase (workspace={workspace_id})")
    except Exception as e:
        logger.warning(f"Failed to save strategy to Supabase: {e}. Falling back to JSON.")
        data  = _load_json_data()
        strats = data.setdefault("saved_strategies", {}).setdefault(workspace_id, [])
        for i, s in enumerate(strats):
            if s.get("name") == name:
                strats[i] = row
                _save_json_data(data)
                return
        strats.append(row)
        _save_json_data(data)


def delete_strategy(name: str, workspace_id: str = "default") -> None:
    """Delete a saved strategy by name."""
    client = _get_supabase_client()

    if client is None:
        data = _load_json_data()
        strats = data.get("saved_strategies", {}).get(workspace_id, [])
        data.setdefault("saved_strategies", {})[workspace_id] = [
            s for s in strats if s.get("name") != name
        ]
        _save_json_data(data)
        return

    try:
        client.table("saved_strategies").delete().eq("name", name).eq(
            "workspace_id", workspace_id
        ).execute()
        logger.info(f"Deleted strategy '{name}' (workspace={workspace_id})")
    except Exception as e:
        logger.warning(f"Failed to delete strategy from Supabase: {e}. Falling back to JSON.")
        data = _load_json_data()
        strats = data.get("saved_strategies", {}).get(workspace_id, [])
        data.setdefault("saved_strategies", {})[workspace_id] = [
            s for s in strats if s.get("name") != name
        ]
        _save_json_data(data)


def rename_strategy(
    old_name: str, new_name: str, workspace_id: str = "default"
) -> None:
    """Rename a saved strategy."""
    client = _get_supabase_client()

    if client is None:
        data = _load_json_data()
        for s in data.get("saved_strategies", {}).get(workspace_id, []):
            if s.get("name") == old_name:
                s["name"] = new_name
        _save_json_data(data)
        return

    try:
        client.table("saved_strategies").update({"name": new_name}).eq(
            "name", old_name
        ).eq("workspace_id", workspace_id).execute()
        logger.info(f"Renamed strategy '{old_name}' → '{new_name}' (workspace={workspace_id})")
    except Exception as e:
        logger.warning(f"Failed to rename strategy in Supabase: {e}. Falling back to JSON.")
        data = _load_json_data()
        for s in data.get("saved_strategies", {}).get(workspace_id, []):
            if s.get("name") == old_name:
                s["name"] = new_name
        _save_json_data(data)
