"""
analysis/asset_profiles.py  —  Per-asset parameter profiles for Trade Analysis.

Why this matters
────────────────
The default GrowthSignalBot was tuned on large-cap growth tech (GOOG 5y).
Different asset classes behave differently:

  • GLD/SLV   — slow-trending commodities: RSI rarely hits 70+, ADX 18 is plenty
  • TSLA/NVDA — momentum rockets: RSI stays at 60-75 during bull runs, wider stops needed
  • SNAP/RIVN — small-cap volatile: large swings, ADX needs to be higher to confirm trends
  • SPY/QQQ   — index ETFs: tight RSI, lower ADX bar, small SL/TP

Usage
─────
    from analysis.asset_profiles import get_profile
    profile = get_profile("TSLA")
    strat = GrowthSignalBot(
        rsi_bull_min  = profile["rsi_bull_min"],
        rsi_bull_max  = profile["rsi_bull_max"],
        adx_threshold = profile["adx_threshold"],
        sl_pct        = profile["sl_pct"],
        tp_pct        = profile["tp_pct"],
    )

Custom profiles
───────────────
Users can create named custom profiles that persist in ~/.trader-bot/custom_profiles.json.
Custom profiles appear everywhere built-in profiles do (deep-dive dropdown, ticker overrides, etc).

Ticker override priority
────────────────────────
  1. User's custom ticker_overrides map  (set_ticker_override)
  2. Built-in TICKER_MAP                 (e.g. GLD → _metals)
  3. DEFAULT_PROFILE fallback            (_growth)
"""
from __future__ import annotations

import json
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_PROFILES_FILE = Path.home() / ".trader-bot" / "custom_profiles.json"


def _load_custom_data() -> dict:
    """Load custom profiles and ticker overrides from disk.
    Returns empty structure on any failure (missing file, bad JSON, etc)."""
    try:
        if CUSTOM_PROFILES_FILE.exists():
            with open(CUSTOM_PROFILES_FILE, "r") as f:
                data = json.load(f)
            data.setdefault("profiles", {})
            data.setdefault("ticker_overrides", {})
            return data
    except Exception:
        pass
    return {"profiles": {}, "ticker_overrides": {}}


def _save_custom_data(data: dict) -> None:
    """Persist custom profiles and ticker overrides to disk."""
    CUSTOM_PROFILES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CUSTOM_PROFILES_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Built-in profile templates  (read-only)
# ─────────────────────────────────────────────────────────────────────────────

PROFILES: dict[str, dict] = {

    # ── Large-cap growth tech  (tuned on GOOG 5y — our baseline) ───────────
    "_growth": dict(
        category      = "Large-Cap Growth",
        rsi_bull_min  = 42,
        rsi_bull_max  = 62,
        adx_threshold = 20.0,
        sl_pct        = 5.0,
        tp_pct        = 15.0,
    ),

    # ── Precious metals / commodities ───────────────────────────────────────
    # Slower-moving, trend-following assets.
    # RSI rarely reaches 70+; ADX 18 catches even gentle trends.
    # Tighter SL/TP to match lower daily ATR.
    "_metals": dict(
        category      = "Precious Metal",
        rsi_bull_min  = 40,
        rsi_bull_max  = 65,
        adx_threshold = 18.0,
        sl_pct        = 4.0,
        tp_pct        = 12.0,
    ),

    # ── High-volatility / momentum tech ────────────────────────────────────
    # RSI routinely stays 60-75+ during bull runs.
    # Wider stops are REQUIRED — tight stops get shaken out on normal noise.
    "_highvol": dict(
        category      = "High-Vol Tech",
        rsi_bull_min  = 45,
        rsi_bull_max  = 72,
        adx_threshold = 25.0,
        sl_pct        = 8.0,
        tp_pct        = 24.0,
    ),

    # ── Small / mid-cap volatile ────────────────────────────────────────────
    # Wide RSI swings, needs higher ADX to confirm real moves vs noise.
    # Larger stops to handle volatility.
    "_smallvol": dict(
        category      = "Small-Cap Volatile",
        rsi_bull_min  = 38,
        rsi_bull_max  = 68,
        adx_threshold = 25.0,
        sl_pct        = 8.0,
        tp_pct        = 20.0,
    ),

    # ── Index ETFs ──────────────────────────────────────────────────────────
    # Liquid, mean-reverting tendencies. Lower ADX bar.
    # Tight SL/TP because lower volatility than individual stocks.
    "_etf": dict(
        category      = "Index ETF",
        rsi_bull_min  = 42,
        rsi_bull_max  = 62,
        adx_threshold = 15.0,
        sl_pct        = 3.0,
        tp_pct        = 9.0,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Built-in ticker → profile mapping
# Add new tickers here as needed; anything not listed falls back to _growth.
# ─────────────────────────────────────────────────────────────────────────────

TICKER_MAP: dict[str, str] = {

    # Precious metals / commodity ETFs
    "GLD":  "_metals",
    "SLV":  "_metals",
    "GDX":  "_metals",
    "GDXJ": "_metals",
    "IAU":  "_metals",
    "SIVR": "_metals",

    # High-volatility momentum tech
    "TSLA": "_highvol",
    "NVDA": "_highvol",
    "AMD":  "_highvol",
    "PLTR": "_highvol",
    "MSTR": "_highvol",
    "COIN": "_highvol",
    "ARM":  "_highvol",
    "SMCI": "_highvol",

    # Small / mid-cap volatile
    "SNAP": "_smallvol",
    "RIVN": "_smallvol",
    "LCID": "_smallvol",
    "SOFI": "_smallvol",
    "HOOD": "_smallvol",
    "RBLX": "_smallvol",
    "BYND": "_smallvol",

    # Index ETFs
    "SPY":  "_etf",
    "QQQ":  "_etf",
    "IWM":  "_etf",
    "DIA":  "_etf",
    "VTI":  "_etf",
    "XLF":  "_etf",
    "XLE":  "_etf",
    "XLK":  "_etf",

    # Large-cap growth (explicit — same as default, listed for clarity)
    "GOOG":  "_growth",
    "GOOGL": "_growth",
    "META":  "_growth",
    "AMZN":  "_growth",
    "MSFT":  "_growth",
    "AAPL":  "_growth",
    "NFLX":  "_growth",
    "UBER":  "_growth",
    "LYFT":  "_growth",
    "SPOT":  "_growth",
    "CRM":   "_growth",
    "NOW":   "_growth",
    "SHOP":  "_growth",
}

DEFAULT_PROFILE = "_growth"


# ─────────────────────────────────────────────────────────────────────────────
# Public read API
# ─────────────────────────────────────────────────────────────────────────────

def get_custom_profiles() -> dict[str, dict]:
    """Return only the user-created custom profiles keyed by name."""
    return _load_custom_data()["profiles"]


def get_ticker_overrides() -> dict[str, str]:
    """Return the user-defined ticker → profile_key override map."""
    return _load_custom_data()["ticker_overrides"]


def get_all_profiles() -> dict[str, dict]:
    """Return merged dict of built-in + custom profiles.

    Custom profiles are keyed by their user-chosen name string.
    If a custom profile shares a key with a built-in, it overrides it.
    """
    combined = dict(PROFILES)
    combined.update(get_custom_profiles())
    return combined


def get_profile(ticker: str) -> dict:
    """Return the parameter profile dict for a given ticker symbol.

    Priority:
      1. User's custom ticker_overrides
      2. Built-in TICKER_MAP
      3. DEFAULT_PROFILE (_growth)
    """
    t            = ticker.upper()
    all_profiles = get_all_profiles()
    overrides    = get_ticker_overrides()

    # 1. Custom ticker override
    if t in overrides:
        key = overrides[t]
        if key in all_profiles:
            return all_profiles[key]

    # 2. Built-in TICKER_MAP
    key = TICKER_MAP.get(t, DEFAULT_PROFILE)
    return PROFILES.get(key, PROFILES[DEFAULT_PROFILE])


def profile_key(ticker: str) -> str:
    """Return the profile key string (e.g. '_metals') for a ticker.

    Priority mirrors get_profile(): overrides → TICKER_MAP → DEFAULT_PROFILE.
    """
    t         = ticker.upper()
    overrides = get_ticker_overrides()
    if t in overrides:
        return overrides[t]
    return TICKER_MAP.get(t, DEFAULT_PROFILE)


# ─────────────────────────────────────────────────────────────────────────────
# Public write API  (custom profiles + ticker overrides)
# ─────────────────────────────────────────────────────────────────────────────

def save_custom_profile(name: str, profile: dict) -> None:
    """Create or update a custom profile by name.

    profile should contain at minimum:
        category, rsi_bull_min, rsi_bull_max, adx_threshold, sl_pct, tp_pct
    """
    data = _load_custom_data()
    data["profiles"][name] = profile
    _save_custom_data(data)


def delete_custom_profile(name: str) -> None:
    """Delete a custom profile and remove any ticker overrides pointing to it."""
    data = _load_custom_data()
    data["profiles"].pop(name, None)
    # Prune orphaned ticker overrides
    data["ticker_overrides"] = {
        t: k for t, k in data["ticker_overrides"].items() if k != name
    }
    _save_custom_data(data)


def set_ticker_override(ticker: str, profile_key_str: str) -> None:
    """Override the profile used for a specific ticker symbol."""
    data = _load_custom_data()
    data["ticker_overrides"][ticker.upper()] = profile_key_str
    _save_custom_data(data)


def remove_ticker_override(ticker: str) -> None:
    """Remove a ticker override, reverting to auto-detection."""
    data = _load_custom_data()
    data["ticker_overrides"].pop(ticker.upper(), None)
    _save_custom_data(data)
