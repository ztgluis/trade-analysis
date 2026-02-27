"""
app.py  —  Trade Analysis · Trading Dashboard
Run with:  streamlit run backend/app.py
"""
from __future__ import annotations
import sys
import json
import uuid
import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from analysis.decision_engine import analyze
from analysis.asset_profiles  import (
    PROFILES, TICKER_MAP, DEFAULT_PROFILE,
    get_profile, get_all_profiles, get_custom_profiles,
    get_ticker_overrides,
    save_custom_profile, delete_custom_profile,
    set_ticker_override, remove_ticker_override,
)
from analysis import supabase_db
from strategies.pine_generator import PineScriptGenerator
from backtester.data   import fetch_ohlcv as _bt_fetch
from backtester.engine import BacktestEngine
from backtester        import metrics as bt_m
from strategies.growth_signal_bot import GrowthSignalBot

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trade Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_WATCHLIST = ["META", "AAPL", "AMZN", "NFLX", "GOOG", "SPY", "QQQ", "DIA", "GLD", "SLV"]

VERDICT_COLORS = {
    "green":  "#44dd88",
    "lime":   "#aaff44",
    "red":    "#ff5555",
    "grey":   "#888888",
    "yellow": "#ffdd44",
}

REGIME_EMOJI = {"bull": "🐂", "bear": "🐻", "neutral": "🔀"}

# Built-in profile badge colours  {category: (fg, bg)}
PROFILE_BADGE = {
    "Large-Cap Growth":   ("#4488ff", "#0a1a3a"),
    "Precious Metal":     ("#ffaa00", "#2a1a00"),
    "High-Vol Tech":      ("#ff4466", "#2a0010"),
    "Small-Cap Volatile": ("#ff8800", "#2a1200"),
    "Index ETF":          ("#44ddaa", "#002a20"),
}
CUSTOM_BADGE = ("#cc88ff", "#1a0030")   # purple pill for any user-created profile

HORIZON_MAP = {
    "1 week  (5td)":    5,
    "2 weeks (10td)":   10,
    "1 month (21td)":   21,
    "2 months (42td)":  42,
    "3 months (63td)":  63,
    "6 months (126td)": 126,
}

INDICATOR_TEMPLATES: dict[str, list[str]] = {
    "Momentum Only":    ["rsi", "macd"],
    "Trend + Momentum": ["sma50", "sma200", "rsi", "macd"],
    "Full Strategy":    ["rsi", "macd", "adx", "ema20", "sma50", "sma200", "vwap", "atr", "fib", "volume"],
    "Custom":           [],
}

ALL_INDICATORS = ["rsi", "macd", "adx", "ema20", "sma50", "sma200", "vwap", "atr", "fib", "volume"]

# ─────────────────────────────────────────────────────────────────────────────
# Built-in (default) strategies — loaded from pine-scripts/ at startup
# ─────────────────────────────────────────────────────────────────────────────

_PINE_DIR = Path(__file__).parent.parent / "pine-scripts"

_DEFAULT_STRATEGY_META: list[tuple] = [
    # (display name,                        filename,                         profile,             entry_mode,     timeframe, indicators,                                                                         is_indicator)
    ("Long Signal Strategy v4 (Daily)",    "growth_signal_bot_v4.pine",     "Large-Cap Growth",  "All Signals",  "1D",
     ["rsi", "macd", "adx", "ema20", "sma50", "sma200", "vwap", "atr", "fib", "volume"],  False),
    ("Long Signal Strategy v1 (1H)",       "growth_signal_bot_1h_v1.pine",  "Large-Cap Growth",  "All Signals",  "1H",
     ["rsi", "macd", "adx", "ema20", "sma50", "sma200", "vwap", "atr", "fib", "volume"],  False),
    ("Swing Signal Strategy v1",           "swing_signal_bot_v1.pine",      "Large-Cap Growth",  "All Signals",  "1D",
     ["rsi", "macd", "ema20", "sma50", "sma200", "atr", "volume"],                         False),
    ("Triple MA [Indicator]",              "triple_ma_v1.pine",             "—",                 "—",            "1D",
     ["ema20", "sma50", "sma200"],                                                          True),
    ("VIX Spike Warning [Indicator]",      "vix_spike_warning_v1.pine",     "—",                 "—",            "1D",
     ["vwap"],                                                                              True),
]


def _load_default_strategies() -> list[dict]:
    """Read the pre-built Pine Script files from pine-scripts/ and return them
    as strategy dicts compatible with the Library tab's row format."""
    result = []
    for name, filename, profile, entry_mode, timeframe, indicators, is_indicator in _DEFAULT_STRATEGY_META:
        path = _PINE_DIR / filename
        try:
            code = path.read_text(encoding="utf-8")
        except Exception:
            code = f"// Could not load {filename}"
        result.append({
            "name":         name,
            "code":         code,
            "profile_name": profile,
            "entry_mode":   entry_mode,
            "timeframe":    timeframe,
            "indicators":   indicators,
            "is_builtin":   True,
            "is_indicator": is_indicator,
        })
    return result


DEFAULT_STRATEGIES: list[dict] = _load_default_strategies()


# ─────────────────────────────────────────────────────────────────────────────
# Workspace ID management
# ─────────────────────────────────────────────────────────────────────────────

def get_workspace_id() -> str:
    """
    Get or create the workspace ID for this session.
    Stored in URL query params as 'w'. Falls back to session state.
    On first visit, generates a new UUID and writes it to the URL.
    """
    params = st.query_params
    if "w" in params:
        wid = params["w"]
        st.session_state["workspace_id"] = wid
        return wid
    if "workspace_id" in st.session_state:
        return st.session_state["workspace_id"]
    # Generate new workspace
    wid = str(uuid.uuid4())[:8]  # short 8-char ID is friendlier
    st.session_state["workspace_id"] = wid
    st.query_params["w"] = wid
    return wid


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic profile helpers  (include custom profiles)
# ─────────────────────────────────────────────────────────────────────────────

def get_profile_options() -> dict[str, dict]:
    """Return {display_name: profile_dict} for all profiles (built-in + custom).

    Keys are the profile's 'category' field (or the profile name for custom ones).
    """
    return {p["category"]: p for p in get_all_profiles().values()}


# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .verdict-green  { background:#1a4a1a; border:1px solid #44dd88; border-radius:8px;
                    padding:12px 18px; font-size:1.4rem; font-weight:700; color:#44dd88; }
  .verdict-lime   { background:#1a3a00; border:1px solid #aaff44; border-radius:8px;
                    padding:12px 18px; font-size:1.4rem; font-weight:700; color:#aaff44; }
  .verdict-red    { background:#4a1a1a; border:1px solid #ff5555; border-radius:8px;
                    padding:12px 18px; font-size:1.4rem; font-weight:700; color:#ff5555; }
  .verdict-grey   { background:#2a2a2a; border:1px solid #888888; border-radius:8px;
                    padding:12px 18px; font-size:1.4rem; font-weight:700; color:#aaaaaa; }
  .verdict-yellow { background:#3a3a00; border:1px solid #ffdd44; border-radius:8px;
                    padding:12px 18px; font-size:1.4rem; font-weight:700; color:#ffdd44; }
  .action-text    { color:#cccccc; font-size:0.95rem; margin-top:8px; }
  .profile-badge  { display:inline-block; border-radius:12px; padding:3px 10px;
                    font-size:0.8rem; font-weight:600; margin-top:6px; }
  .metric-label   { color:#888; font-size:0.8rem; }
  .metric-val     { font-size:1.1rem; font-weight:600; }
  div[data-testid="stDataFrame"] { border-radius:8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def pct(v: float, d: int = 1) -> str:
    if v is None: return "—"
    return f"{'+'if v>=0 else''}{v*100:.{d}f}%"

def dol(v: float | None) -> str:
    if v is None: return "—"
    return f"${v:,.2f}"

def verdict_css_class(color: str) -> str:
    return f"verdict-{color}"

def score_bar_html(score: int, max_score: int = 10, color: str = "#44dd88") -> str:
    pct_val = score / max_score
    return (
        f'<div style="background:#333;border-radius:4px;height:14px;width:100%;">'
        f'<div style="background:{color};border-radius:4px;height:14px;'
        f'width:{pct_val*100:.0f}%;"></div></div>'
    )

def profile_badge_html(profile_name: str) -> str:
    fg, bg = PROFILE_BADGE.get(profile_name, CUSTOM_BADGE)
    return (f'<span class="profile-badge" '
            f'style="color:{fg};background:{bg};border:1px solid {fg}55;">'
            f'{profile_name}</span>')


# ─────────────────────────────────────────────────────────────────────────────
# Plotly chart builders
# ─────────────────────────────────────────────────────────────────────────────

SIGNAL_STYLES = {
    "🔺 BUY":         dict(symbol="triangle-up",   color="#00ff88", size=14),
    "⬤ BOUNCE":       dict(symbol="circle",         color="#00ccff", size=10),
    "◆ VWAP BOUNCE":  dict(symbol="diamond",        color="#cc44ff", size=10),
    "🔻 SELL":         dict(symbol="triangle-down",  color="#ff4444", size=14),
    "✕ Bull Div":      dict(symbol="x",              color="#88ff44", size=10),
    "✕ Bear Div":      dict(symbol="x",              color="#ff88ff", size=10),
}

def build_price_chart(r: dict) -> go.Figure:
    df = r["df_chart"]
    if df is None or df.empty:
        return go.Figure()

    dates = df.index
    fig   = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=df["close"], name="Price",
        line=dict(color="#ffffff", width=2),
        hovertemplate="%{x|%b %d}<br>$%{y:,.2f}<extra></extra>",
    ))

    for col, name, color, dash in [
        ("sma200", "SMA200", "#4488ff", "dot"),
        ("sma50",  "SMA50",  "#ff8844", "dot"),
        ("ema20",  "EMA20",  "#44dd88", "dash"),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=dates, y=df[col], name=name,
                line=dict(color=color, width=1.5, dash=dash),
                hovertemplate=f"{name} $%{{y:,.2f}}<extra></extra>",
            ))

    signal_col_map = {
        "buy_signal":    "🔺 BUY",
        "bounce_signal": "⬤ BOUNCE",
        "vwap_bounce":   "◆ VWAP BOUNCE",
        "sell_signal":   "🔻 SELL",
        "bull_div":      "✕ Bull Div",
        "bear_div":      "✕ Bear Div",
    }
    for col, label in signal_col_map.items():
        if col not in df.columns:
            continue
        sig_df = df[df[col] == True]
        if sig_df.empty:
            continue
        style = SIGNAL_STYLES[label]
        fig.add_trace(go.Scatter(
            x=sig_df.index, y=sig_df["close"],
            mode="markers", name=label,
            marker=dict(symbol=style["symbol"], color=style["color"],
                        size=style["size"], line=dict(width=1, color="#000")),
            hovertemplate=f"{label}<br>%{{x|%b %d}}<br>$%{{y:,.2f}}<extra></extra>",
        ))

    fig.update_layout(
        template="plotly_dark", height=400,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        xaxis=dict(showgrid=True, gridcolor="#333"),
        yaxis=dict(showgrid=True, gridcolor="#333", tickprefix="$"),
        hovermode="x unified",
    )
    return fig


def build_signal_timeline(r: dict) -> go.Figure | None:
    df = r.get("df_chart")
    if df is None or df.empty:
        return None

    signal_col_map = {
        "buy_signal":    "🔺 BUY",
        "bounce_signal": "⬤ BOUNCE",
        "vwap_bounce":   "◆ VWAP BOUNCE",
        "sell_signal":   "🔻 SELL",
        "bull_div":      "✕ Bull Div",
        "bear_div":      "✕ Bear Div",
    }
    timeline_sigs: list[tuple] = []
    for col, label in signal_col_map.items():
        if col not in df.columns:
            continue
        for ts, row in df[df[col] == True].iterrows():
            timeline_sigs.append((ts, label, float(row["close"])))
    if not timeline_sigs:
        return None

    timeline_sigs.sort(key=lambda x: x[0])
    fig = go.Figure()
    added: set = set()
    for ts, label, price in timeline_sigs:
        style = SIGNAL_STYLES.get(label, dict(symbol="circle", color="#888", size=8))
        fig.add_trace(go.Scatter(
            x=[ts], y=[0],
            mode="markers+text", name=label,
            showlegend=label not in added,
            marker=dict(symbol=style["symbol"], color=style["color"],
                        size=style["size"] + 2, line=dict(width=1, color="#000")),
            text=[f"${price:,.0f}"],
            textposition="top center",
            textfont=dict(size=9, color=style["color"]),
            hovertemplate=f"{label}<br>{pd.Timestamp(ts).strftime('%b %d, %Y')}<br>${price:,.2f}<extra></extra>",
        ))
        added.add(label)

    fig.update_layout(
        template="plotly_dark", height=130,
        margin=dict(l=0, r=0, t=10, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=10)),
        xaxis=dict(showgrid=False),
        yaxis=dict(visible=False, range=[-0.5, 1.5]),
        hovermode="closest",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(workspace_id: str = "default") -> None:
    st.sidebar.title("📈 Trade Analysis")
    st.sidebar.caption("Decision Dashboard · v1.0")
    st.sidebar.divider()

    # ── Workspace ──────────────────────────────────────────────────────────
    with st.sidebar.expander("🔑 Workspace", expanded=False):
        st.caption("Your personal workspace token. Bookmark this URL or share it to access your data from any device.")
        st.code(workspace_id, language=None)

        # Show full URL with workspace param
        # Can't easily get the full URL in Streamlit, so just show the token
        st.caption("Add `?w=YOUR_TOKEN` to the app URL to resume your workspace.")

        st.divider()
        new_wid = st.text_input("Switch to another workspace:", placeholder="Enter token...", key="switch_workspace_input")
        if st.button("Switch", key="switch_workspace_btn") and new_wid.strip():
            clean = new_wid.strip()
            st.session_state["workspace_id"] = clean
            st.query_params["w"] = clean
            # Clear cached session data so it reloads from new workspace
            for key in ["watchlist", "results", "migration_attempted"]:
                st.session_state.pop(key, None)
            st.rerun()

    st.sidebar.divider()

    # Get page from query params or default to dashboard
    _PAGE_MAP = {
        "📊 Dashboard":  "dashboard",
        "⚙️ Profiles":   "profiles",
        "🎛️ Generator":  "generator",
    }
    _REVERSE_PAGE_MAP = {v: k for k, v in _PAGE_MAP.items()}

    params = st.query_params
    current_page = params.get("p", "dashboard")
    default_label = _REVERSE_PAGE_MAP.get(current_page, "📊 Dashboard")

    page_label = st.sidebar.radio(
        "Navigation",
        ["📊 Dashboard", "⚙️ Profiles", "🎛️ Generator"],
        label_visibility="collapsed",
        key="nav_radio",
        index=["📊 Dashboard", "⚙️ Profiles", "🎛️ Generator"].index(default_label),
    )
    st.session_state["page"] = _PAGE_MAP[page_label]
    st.query_params["p"] = st.session_state["page"]


# ─────────────────────────────────────────────────────────────────────────────
# Watchlist Dashboard
# ─────────────────────────────────────────────────────────────────────────────

def render_dashboard(results: dict) -> None:
    """Render watchlist summary table. Clicking a row selects that ticker."""
    if not results:
        st.info("Click **🔄 Run All Analysis** in the sidebar to load your watchlist.")
        return

    rows       = []
    row_tickers = []   # parallel list to map row index → ticker
    for ticker, r in results.items():
        if r.get("error"):
            rows.append({"Symbol": ticker, "Price": "—", "Verdict": "⚠ Error",
                         "Long": "—", "Short": "—", "R/R": "—",
                         "Regime": "—", "RSI": "—", "ADX": "—",
                         "1mo vs SPY": "—", "Last Signal": "—"})
            row_tickers.append(ticker)
            continue

        last_sig = "—"
        if r.get("recent_signals"):
            ts, lbl, _ = r["recent_signals"][-1]
            last_sig = f"{lbl}  {pd.Timestamp(ts).strftime('%b %d')}"

        alpha_str = pct(r.get("alpha_1m")) if r.get("alpha_1m") is not None else "—"
        rr        = r.get("rr_ratio")
        rr_str    = f"{rr:.1f}:1" if rr is not None else "—"

        rows.append({
            "Symbol":      ticker,
            "Price":       f"${r['price']:,.2f}",
            "Verdict":     r["verdict"],
            "Long":        f"{r['long_score']}/10",
            "Short":       f"{r['short_score']}/10",
            "R/R":         rr_str,
            "Regime":      f"{REGIME_EMOJI.get(r['regime'], '?')} {r['regime'].upper()}",
            "RSI":         f"{r['rsi']:.0f}",
            "ADX":         f"{r['adx']:.0f}{'✅' if r['adx_ok'] else ''}",
            "1mo vs SPY":  alpha_str,
            "Last Signal": last_sig,
        })
        row_tickers.append(ticker)

    df_table = pd.DataFrame(rows)

    def style_verdict(val: str) -> str:
        if "STRONG LONG" in val or "LEAN LONG" in val: return "color:#44dd88;font-weight:bold"
        if "BOUNCE" in val:                            return "color:#aaff44;font-weight:bold"
        if "SHORT" in val:                             return "color:#ff5555;font-weight:bold"
        if "WAIT" in val or "Error" in val:            return "color:#888888"
        return ""

    def style_alpha(val: str) -> str:
        if val.startswith("+"): return "color:#44dd88"
        if val.startswith("-"): return "color:#ff7777"
        return ""

    def style_rr(val: str) -> str:
        if val == "—": return "color:#666666"
        try:
            ratio = float(val.split(":")[0])
            if ratio >= 2.5: return "color:#44dd88;font-weight:bold"
            if ratio >= 1.5: return "color:#ffdd44"
            return "color:#ff7777"
        except Exception:
            return ""

    styled = (df_table.style
              .map(style_verdict, subset=["Verdict"])
              .map(style_alpha,   subset=["1mo vs SPY"])
              .map(style_rr,      subset=["R/R"]))

    event = st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=38 * len(rows) + 40,
        on_select="rerun",
        selection_mode="single-row",
        key="watchlist_df",
    )

    # Handle row click → navigate to deep dive
    if event.selection.rows:
        row_idx = event.selection.rows[0]
        if row_idx < len(row_tickers):
            clicked = row_tickers[row_idx]
            if clicked != st.session_state.get("selected_ticker"):
                st.session_state["selected_ticker"] = clicked

    sel = st.session_state.get("selected_ticker")
    if sel and sel in results and not results[sel].get("error"):
        st.caption(f"↓ Showing deep dive for **{sel}** — click any other row to switch")
    else:
        st.caption("↑ Click any row to open detailed analysis below")


# ─────────────────────────────────────────────────────────────────────────────
# Deep Dive
# ─────────────────────────────────────────────────────────────────────────────

def render_deep_dive(r: dict) -> None:
    if r.get("error"):
        st.error(f"⚠ {r['error']}")
        return

    ticker  = r["ticker"]
    verdict = r["verdict"]
    color   = r["color"]
    action  = r["action"]

    # ── Header: verdict + profile badge ──────────────────────────────────────
    h1, h2, h3, h4 = st.columns([3, 1, 1, 1])

    profile_name = r.get("profile", "Large-Cap Growth")
    badge_html   = profile_badge_html(profile_name)

    h1.markdown(
        f'<div class="{verdict_css_class(color)}">{verdict}</div>'
        f'<div class="action-text">{action}</div>'
        f'<div style="margin-top:8px;">{badge_html}</div>',
        unsafe_allow_html=True,
    )

    regime_icon = REGIME_EMOJI.get(r["regime"], "?")
    h2.metric("Price",     dol(r["price"]))
    h3.metric("Regime",    f"{regime_icon} {r['regime'].upper()}")
    h4.metric("RSI / ADX", f"{r['rsi']:.0f}  /  {r['adx']:.0f}")

    # ── Profile override selector ─────────────────────────────────────────────
    profile_opts = get_profile_options()
    profile_categories = list(profile_opts.keys())
    current_idx = profile_categories.index(profile_name) if profile_name in profile_categories else 0
    chosen_profile_name = st.selectbox(
        "🔬 Analyze with profile:",
        profile_categories,
        index=current_idx,
        key=f"profile_override_{ticker}",
        help="Switch profiles to see how this ticker's signals change with different parameters.",
    )
    if chosen_profile_name != profile_name:
        op = profile_opts[chosen_profile_name]
        st.info(
            f"**Auto-detected:** {profile_name}  →  "
            f"**Override active:** {chosen_profile_name}  "
            f"*(RSI {op['rsi_bull_min']}–{op['rsi_bull_max']}, "
            f"ADX≥{op['adx_threshold']:.0f}, "
            f"SL {op['sl_pct']}% / TP {op['tp_pct']}%)*"
        )

    # ── Score bars ────────────────────────────────────────────────────────────
    st.markdown("---")
    sc1, sc2 = st.columns(2)
    with sc1:
        ls = r["long_score"]
        st.markdown(f"**LONG  {ls}/10** {'🟢' if ls >= 7 else '🟡' if ls >= 5 else '🔴'}")
        st.markdown(score_bar_html(ls, color="#44dd88"), unsafe_allow_html=True)
    with sc2:
        ss = r["short_score"]
        st.markdown(f"**SHORT  {ss}/10** {'🔴' if ss >= 7 else '🟡' if ss >= 5 else '⚪'}")
        st.markdown(score_bar_html(ss, color="#ff5555"), unsafe_allow_html=True)

    # ── Conditions expander (expanded by default) ─────────────────────────────
    with st.expander("📋 Condition Details", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Long conditions ({r['long_score']}/10)**")
            for ok, desc in r["long_conditions"]:
                st.markdown(f"{'✅' if ok else '❌'}  {desc}")
        with c2:
            st.markdown(f"**Short conditions ({r['short_score']}/10)**")
            for ok, desc in r["short_conditions"]:
                st.markdown(f"{'✅' if ok else '❌'}  {desc}")

    # ── Price chart ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"📊 {ticker} — Last 6 Months")

    active = r.get("active_signals", [])
    if active:
        st.success(f"**Signal on last bar:** {' · '.join(active)}")

    st.plotly_chart(build_price_chart(r), use_container_width=True)

    # ── Signal timeline ───────────────────────────────────────────────────────
    tl_fig = build_signal_timeline(r)
    if tl_fig:
        st.markdown("**Signal Timeline (last 6 months)**")
        st.plotly_chart(tl_fig, use_container_width=True)

    # ── Bottom panels ─────────────────────────────────────────────────────────
    st.markdown("---")
    p1, p2, p3 = st.columns([2, 2, 1.5])

    # Base rates
    with p1:
        direction = r["base_rates_dir"]
        st.markdown(f"**📈 Base Rates ({direction.upper()} perspective)**")
        n = r["base_rates"]["_n"]
        st.caption(f"{n} historical instances in same state")
        h_lbl = {5: "1wk", 10: "2wk", 21: "1mo", 42: "2mo", 63: "3mo", 126: "6mo"}
        horizon_td = r.get("horizon", 21)
        br_rows = []
        for h, lbl in h_lbl.items():
            s = r["base_rates"].get(h, {})
            if s and s.get("n", 0) > 0:
                fav     = "✅" if s["win"] > 0.55 else ("🟡" if s["win"] > 0.45 else "🔴")
                current = " ◀" if h == horizon_td else ""
                br_rows.append({
                    "Horizon":   lbl + current,
                    "n":         s["n"],
                    "Win Rate":  pct(s["win"]),
                    "Median":    pct(s["median"]),
                    "":          fav,
                })
        if br_rows:
            st.dataframe(pd.DataFrame(br_rows), hide_index=True, use_container_width=True)
        else:
            st.caption("Insufficient historical data")

    # Key levels
    with p2:
        st.markdown("**📏 Key Levels**")
        price = r["price"]
        level_rows = []
        for name, val in r["levels"]:
            dist = (val - price) / price
            role = ""
            if direction == "long":
                role = "resistance" if dist > 0 else "support"
            else:
                role = "stop-loss" if dist > 0 else "cover target"
            level_rows.append({
                "Level": name, "Price": f"${val:,.2f}",
                "Dist": pct(dist), "Role": role,
            })
        st.dataframe(pd.DataFrame(level_rows), hide_index=True, use_container_width=True)

    # Earnings + Macro
    with p3:
        st.markdown("**🌍 Earnings & Macro**")
        earnings = r.get("earnings")
        if earnings:
            td   = earnings["trading_days"]
            flag = "🚨" if td <= 3 else ("⚠️" if td <= 10 else "📅")
            st.markdown(f"{flag} **Earnings:** {earnings['date_str']}")
            st.markdown(f"~{td} trading days away")
        else:
            st.markdown("📅 No earnings scheduled")

        macro = r.get("macro", {})
        if macro:
            st.markdown("---")
            for tkr, info in macro.items():
                reg   = REGIME_EMOJI.get(info["regime"], "?")
                arrow = "↑" if info["sma200_rising"] else "↓"
                ret   = pct(info.get("ret_1m")) if info.get("ret_1m") is not None else "—"
                st.markdown(f"**{tkr}** {reg} {info['regime'].upper()}  SMA200{arrow}  {ret}")

        alpha = r.get("alpha_1m")
        if alpha is not None:
            sign      = "+" if alpha >= 0 else ""
            color_str = "green" if alpha > 0 else "red"
            st.markdown(f"**vs SPY 1mo:** :{color_str}[{sign}{alpha*100:.1f}%]")

    # ── Action Plan ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🎯 Action Plan")
    sma200    = r["sma200"]
    sma50     = r["sma50"]
    rsi_val   = r["rsi"]
    sl_price  = r.get("sl_price")
    tp_price  = r.get("tp_price")
    rr_ratio  = r.get("rr_ratio")
    entry_lo  = r.get("entry_lo")
    entry_hi  = r.get("entry_hi")
    sl_pct    = r.get("sl_pct", 5.0)
    tp_pct    = r.get("tp_pct", 15.0)
    rr_str    = f"{rr_ratio:.1f}:1" if rr_ratio is not None else "—"

    if "STRONG LONG" in verdict or "LEAN LONG" in verdict:
        ac1, ac2, ac3, ac4 = st.columns(4)
        ac1.metric("Entry Zone",  f"{dol(entry_lo)} – {dol(entry_hi)}")
        ac2.metric("Stop-Loss",   dol(sl_price), delta=f"-{sl_pct:.0f}%", delta_color="inverse")
        ac3.metric("Target",      dol(tp_price), delta=f"+{tp_pct:.0f}%")
        ac4.metric("R/R Ratio",   rr_str)
        st.caption(f"Invalidation: close below {dol(sma200)} SMA200 → exit long")

    elif "BOUNCE" in verdict:
        ema20_val = r.get("ema20") or r["price"] * 1.04
        ac1, ac2, ac3, ac4 = st.columns(4)
        ac1.metric("Entry Zone",  f"≈ {dol(r['price'])}")
        ac2.metric("Stop-Loss",   dol(sl_price), help="Below 52w low / recent swing low")
        ac3.metric("Target 1",    dol(ema20_val), help="EMA20 — first resistance")
        ac4.metric("R/R Ratio",   rr_str)
        st.caption(f"Target 2: {dol(sma50)} (SMA50 — larger recovery)")
        st.warning("⚠ **REDUCED SIZE** — counter-trend bounce play. Tight stop.")

    elif "SHORT" in verdict:
        ac1, ac2, ac3, ac4 = st.columns(4)
        ac1.metric("Entry Zone",  f"{dol(entry_lo)} – {dol(entry_hi)}")
        ac2.metric("Stop-Loss",   dol(sl_price), help="Above nearest resistance")
        ac3.metric("Target",      dol(tp_price), delta="-7%", delta_color="normal")
        ac4.metric("R/R Ratio",   rr_str)
        st.caption(f"RSI gate: must be > 30 before entering (currently {rsi_val:.0f})")
        st.caption(f"Invalidation: close above {dol(sma200)} SMA200 → cover short")

    else:
        st.info("⚪ **No entry recommended.** Wait for long/short score to diverge.")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Watch: RSI",    f"{rsi_val:.0f}")
        mc2.metric("Watch: SMA200", dol(sma200))
        mc3.metric("Watch: MACD",   "🟢 above" if r["macd_bull"] else "🔴 below")


# ─────────────────────────────────────────────────────────────────────────────
# Library helpers
# ─────────────────────────────────────────────────────────────────────────────

def _lib_backtest_expander(strat: dict, key_pfx: str) -> None:
    """Render the 🧪 Backtest expander for any strategy dict.
    Module-level so Streamlit doesn't re-create it on every rerun."""
    with st.expander("🧪 Backtest", expanded=False):
        _bt_ticker = st.text_input(
            "Ticker", value="AAPL",
            key=f"{key_pfx}_bt_ticker", placeholder="e.g. TSLA",
        ).upper().strip()
        _bt_period = st.selectbox(
            "History", ["1y", "2y", "3y", "5y", "max"],
            index=1, key=f"{key_pfx}_bt_period",
        )
        if st.button(
            "▶ Run Backtest", key=f"{key_pfx}_bt_run",
            type="primary", use_container_width=True,
        ):
            if not _bt_ticker:
                st.error("Enter a ticker first.")
            else:
                _em_raw = strat.get("entry_mode") or "All Signals"
                _em_map = {
                    "all_signals": "All Signals", "buy_only": "Buy Only",
                    "strong_buy_only": "Strong Buy Only",
                    "All Signals": "All Signals", "Buy Only": "Buy Only",
                    "Strong Buy Only": "Strong Buy Only",
                }
                _bt_em = _em_map.get(_em_raw, "All Signals")
                _bt_prof_name = strat.get("profile_name") or ""
                _bt_prof = PROFILES.get("_growth", {})
                for _p in PROFILES.values():
                    if _p.get("category") == _bt_prof_name:
                        _bt_prof = _p
                        break
                with st.spinner(f"Running backtest on {_bt_ticker}…"):
                    try:
                        _df_bt = _bt_fetch(_bt_ticker, period=_bt_period, interval="1d")
                        if _df_bt is None or len(_df_bt) < 60:
                            st.error(f"Not enough data for {_bt_ticker}.")
                        else:
                            _sbt = GrowthSignalBot(
                                entry_mode    = _bt_em,
                                sl_pct        = _bt_prof.get("sl_pct",        5.0),
                                tp_pct        = _bt_prof.get("tp_pct",        15.0),
                                rsi_bull_min  = _bt_prof.get("rsi_bull_min",  42),
                                rsi_bull_max  = _bt_prof.get("rsi_bull_max",  62),
                                adx_threshold = _bt_prof.get("adx_threshold", 20.0),
                            )
                            _eng = BacktestEngine(
                                strategy=_sbt, data=_df_bt,
                                initial_capital=10_000, commission_pct=0.001,
                            )
                            _res    = _eng.run()
                            _eq     = _res.equity
                            _trades = _res.trades
                            _prices = _df_bt["close"]
                            _n_t    = len(_trades)
                            _n_w    = sum(1 for t in _trades if (t.pnl_pct or 0) > 0)
                            _r_tot  = bt_m.total_return(_eq)
                            _r_bah  = bt_m.buy_hold_return(_prices)
                            _r_cagr = bt_m.cagr(_eq)
                            _r_dd   = bt_m.max_drawdown(_eq)
                            _r_sh   = bt_m.sharpe_ratio(_eq)
                            _r_wr   = bt_m.win_rate(_trades)
                            _r_pf   = bt_m.profit_factor(_trades)
                            st.markdown("**Results**")
                            _mc1, _mc2, _mc3 = st.columns(3)
                            _mc1.metric("Total Return", f"{_r_tot*100:+.1f}%",
                                        delta=f"B&H {_r_bah*100:+.1f}%")
                            _mc2.metric("CAGR",   f"{_r_cagr*100:+.1f}%")
                            _mc3.metric("Sharpe", f"{_r_sh:.2f}")
                            _mc4, _mc5, _mc6 = st.columns(3)
                            _mc4.metric("Max Drawdown",  f"{_r_dd*100:.1f}%")
                            _mc5.metric("Win Rate", f"{_r_wr*100:.1f}%",
                                        delta=f"{_n_w}W / {_n_t - _n_w}L")
                            _mc6.metric("Profit Factor", f"{_r_pf:.2f}")
                            st.caption(
                                f"{_n_t} trades · {_bt_period} · "
                                f"{_bt_prof_name or 'default profile'} · {_bt_em}"
                            )
                    except Exception as _bt_err:
                        st.error(f"Backtest error: {_bt_err}")


# ─────────────────────────────────────────────────────────────────────────────
# Strategies Page
# ─────────────────────────────────────────────────────────────────────────────

def render_strategies_page(workspace_id: str = "default") -> None:
    st.title("🎛️ Generator")

    # Get generator tab from query params, default to "generate"
    params = st.query_params
    gen_tab = params.get("gt", "generate")  # gt = generator tab
    gen_tab_index = 0 if gen_tab == "generate" else 1

    tab_labels = ["⚡ Generate", "📚 Library"]
    tab_gen, tab_lib = st.tabs(tab_labels)

    # ══════════════════════════════════════════════════════════════════════
    # Tab 1 — Generate
    # ══════════════════════════════════════════════════════════════════════
    with tab_gen:
        st.caption("Generate a TradingView Pine Script strategy or indicator from any profile.")

        # ── Phase 1: Template selection ───────────────────────────────────────
        st.markdown("#### Step 1 — Choose a Template")

        # Initialise template session state on first load
        if "gen_template" not in st.session_state:
            st.session_state["gen_template"] = "Full Strategy"

        template_choice = st.radio(
            "Template",
            options=list(INDICATOR_TEMPLATES.keys()),
            index=list(INDICATOR_TEMPLATES.keys()).index(st.session_state["gen_template"]),
            horizontal=True,
            key="gen_template_radio",
        )

        # When the template radio changes, update session state and sync indicator defaults
        if template_choice != st.session_state["gen_template"]:
            st.session_state["gen_template"] = template_choice
            # Reset per-indicator checkbox states to match the new template
            for ind in ALL_INDICATORS:
                st.session_state[f"gen_ind_{ind}"] = ind in INDICATOR_TEMPLATES[template_choice]
            st.rerun()

        # Show info about the current template
        tmpl_inds = INDICATOR_TEMPLATES[template_choice]
        if tmpl_inds:
            st.info(f"**{template_choice}** uses: {', '.join(tmpl_inds).upper()}")
        else:
            st.info("**Custom** — select indicators manually below.")

        # Seed indicator checkbox defaults on first load (before any radio change)
        for ind in ALL_INDICATORS:
            key = f"gen_ind_{ind}"
            if key not in st.session_state:
                st.session_state[key] = ind in INDICATOR_TEMPLATES[st.session_state["gen_template"]]

        st.markdown("---")

        # ── Script Type ───────────────────────────────────────────────────────
        st.markdown("#### Script Type")
        script_type_label = st.radio(
            "Script type",
            ["📈 Strategy", "📊 Indicator"],
            horizontal=True,
            key="gen_script_type",
            label_visibility="collapsed",
            captions=[
                "Full strategy with entry/exit logic, SL/TP, and signal shapes",
                "Indicator-only script — calculations + plots, no trade orders",
            ],
        )
        is_indicator_gen = script_type_label == "📊 Indicator"

        st.markdown("---")

        # ── Phase 2: Customisation ────────────────────────────────────────────
        st.markdown("#### Step 2 — Customise")

        # Profile dropdown (all profiles)
        all_profiles_for_gen = get_all_profiles(workspace_id)
        profile_display = {p["category"]: k for k, p in all_profiles_for_gen.items()}
        selected_profile_name = st.selectbox(
            "Analysis Profile",
            options=list(profile_display.keys()),
            key="gen_profile_sel",
        )
        selected_profile_key  = profile_display[selected_profile_name]
        selected_profile_data = all_profiles_for_gen[selected_profile_key]

        # ── Indicator checkboxes in 3 columns, grouped by category ────────────
        st.markdown("**Select Indicators**")

        INDICATOR_GROUPS = {
            "Momentum":            [("rsi",   "RSI"),       ("macd",  "MACD")],
            "Trend":               [("ema20", "Fast EMA"),  ("sma50", "Mid SMA"), ("sma200", "Slow SMA")],
            "Filters":             [("adx",   "ADX"),       ("atr",   "ATR")],
            "Support/Resistance":  [("vwap",  "VWAP"),      ("fib",   "Fibonacci Levels"), ("volume", "Volume")],
        }

        # Human-readable labels for expander titles
        IND_LABELS = {
            "rsi": "RSI", "macd": "MACD", "adx": "ADX", "atr": "ATR",
            "ema20": "Fast EMA", "sma50": "Mid SMA", "sma200": "Slow SMA",
            "vwap": "VWAP", "fib": "Fibonacci", "volume": "Volume",
        }

        cb_col1, cb_col2, cb_col3 = st.columns(3)
        group_items = list(INDICATOR_GROUPS.items())

        col_map = {0: cb_col1, 1: cb_col2, 2: cb_col3}
        col_idx = 0
        for group_name, indicators in group_items:
            with col_map[col_idx % 3]:
                st.markdown(f"**{group_name}**")
                for ind_key, ind_label in indicators:
                    st.checkbox(ind_label, key=f"gen_ind_{ind_key}")
            col_idx += 1

        # Collect which indicators are currently checked
        chosen_indicators = [ind for ind in ALL_INDICATORS if st.session_state.get(f"gen_ind_{ind}", False)]

        # ── Per-indicator parameter expanders ────────────────────────────────
        if chosen_indicators:
            st.markdown("**Indicator Parameters**")

            indicator_params: dict[str, dict] = {}

            for ind in chosen_indicators:
                with st.expander(IND_LABELS.get(ind, ind.upper()), expanded=False):
                    params: dict = {}
                    if ind == "rsi":
                        params["length"] = st.number_input("Length", min_value=2, max_value=50,
                                                           value=14, key="gen_param_rsi_length")
                    elif ind == "macd":
                        params["fast"]   = st.number_input("Fast",   min_value=2, max_value=50,
                                                           value=12, key="gen_param_macd_fast")
                        params["slow"]   = st.number_input("Slow",   min_value=5, max_value=200,
                                                           value=26, key="gen_param_macd_slow")
                        params["signal"] = st.number_input("Signal", min_value=2, max_value=50,
                                                           value=9,  key="gen_param_macd_signal")
                    elif ind == "adx":
                        params["length"]    = st.number_input("Length",    min_value=2, max_value=50,
                                                              value=14, key="gen_param_adx_length")
                        params["smoothing"] = st.number_input("Smoothing", min_value=2, max_value=50,
                                                              value=14, key="gen_param_adx_smoothing")
                    elif ind == "ema20":
                        params["length"] = st.number_input("Length", min_value=2, max_value=200,
                                                           value=20, key="gen_param_ema20_length")
                    elif ind == "sma50":
                        params["length"] = st.number_input("Length", min_value=2, max_value=200,
                                                           value=50, key="gen_param_sma50_length")
                    elif ind == "sma200":
                        params["length"] = st.number_input("Length", min_value=2, max_value=500,
                                                           value=200, key="gen_param_sma200_length")
                    elif ind == "atr":
                        params["length"]     = st.number_input("Length",     min_value=2, max_value=50,
                                                               value=14, key="gen_param_atr_length")
                        params["multiplier"] = st.number_input("Multiplier", min_value=0.1,
                                                               max_value=10.0, value=1.5, step=0.1,
                                                               key="gen_param_atr_multiplier")
                    elif ind == "vwap":
                        params["anchor"] = st.selectbox(
                            "Anchor Period",
                            options=["Session", "Week", "Month", "Quarter", "Year"],
                            index=1,  # Week default
                            key="gen_param_vwap_anchor",
                            help="Session resets daily (uses ta.vwap built-in). "
                                 "Week/Month/Quarter/Year use a cumulative manual reset.",
                        )
                        params["source"] = st.selectbox(
                            "Source",
                            options=["HLC3", "HL2", "Close", "OHLC4"],
                            index=0,  # HLC3 (typical price) default
                            key="gen_param_vwap_source",
                            help="HLC3 = (H+L+C)/3 · HL2 = (H+L)/2 · OHLC4 = (O+H+L+C)/4",
                        )
                    elif ind == "fib":
                        params["swing"] = st.number_input("Swing Lookback", min_value=5, max_value=200,
                                                          value=50, key="gen_param_fib_swing")
                    elif ind == "volume":
                        params["ma_length"] = st.number_input("MA Length", min_value=2, max_value=200,
                                                              value=20, key="gen_param_volume_ma_length")
                    indicator_params[ind] = params

        else:
            indicator_params = {}

        # ── Entry mode (strategy only) ────────────────────────────────────────
        entry_mode_labels = {
            "All Signals":      "all_signals",
            "Buy Only":         "buy_only",
            "Strong Buy Only":  "strong_buy_only",
        }
        if not is_indicator_gen:
            st.markdown("**Entry Mode**")
            entry_mode_label = st.radio(
                "Entry mode",
                options=list(entry_mode_labels.keys()),
                captions=[
                    "Enter on any buy signal or EMA bounce",
                    "Enter only on scored buy signals",
                    "Enter only when buy signal AND bull score is strong",
                ],
                key="gen_entry_mode",
                label_visibility="collapsed",
            )
            entry_mode = entry_mode_labels[entry_mode_label]
        else:
            entry_mode = "all_signals"  # unused for indicator scripts

        gen_col1, gen_col2 = st.columns(2)
        with gen_col1:
            _default_script_name = "My Generated Indicator" if is_indicator_gen else "My Generated Strategy"
            strategy_name = st.text_input(
                "Script Name",
                value=_default_script_name,
                key="gen_strategy_name",
            )
        with gen_col2:
            timeframe = st.selectbox(
                "Timeframe",
                options=["D", "60", "240", "W"],
                format_func=lambda x: {"D": "Daily (D)", "60": "1 Hour (60)",
                                        "240": "4 Hour (240)", "W": "Weekly (W)"}[x],
                key="gen_timeframe",
            )

        st.markdown("---")

        # ── Generate / Save buttons ───────────────────────────────────────────
        _output_type = "indicator" if is_indicator_gen else "strategy"
        _btn_gen_label = "Generate Pine Script" if not is_indicator_gen else "Generate Indicator Script"
        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            gen_clicked = st.button(
                _btn_gen_label,
                type="primary",
                key="gen_generate_btn",
                use_container_width=True,
            )
        with btn_c2:
            save_clicked = st.button(
                "💾 Save to Library",
                key="gen_save_direct_btn",
                use_container_width=True,
            )

        if gen_clicked or save_clicked:
            if not chosen_indicators:
                st.error("Select at least one indicator before generating.")
            else:
                with st.spinner("Generating Pine Script…"):
                    gen = PineScriptGenerator(
                        profile=selected_profile_data,
                        indicators=chosen_indicators,
                        indicator_params=indicator_params,
                        strategy_name=strategy_name,
                        entry_mode=entry_mode,
                        timeframe=timeframe,
                        output_type=_output_type,
                    )
                    code = gen.generate()
                    st.session_state["gen_code"] = code

                if save_clicked:
                    name_clean = strategy_name.strip()
                    if name_clean:
                        metadata = {
                            "profile_name": selected_profile_name if not is_indicator_gen else "—",
                            "indicators":   chosen_indicators,
                            "entry_mode":   "—" if is_indicator_gen else entry_mode,
                            "timeframe":    timeframe,
                        }
                        supabase_db.save_strategy(name_clean, code, metadata, workspace_id)
                        _saved_type = "indicator" if is_indicator_gen else "strategy"
                        st.success(f"✅ Saved **{name_clean}** ({_saved_type}) to the Library.")
                    else:
                        st.error("Enter a script name before saving.")

        # ── Show results if code has been generated ───────────────────────────
        if st.session_state.get("gen_code"):
            code = st.session_state["gen_code"]
            _gen_label = "Generated Indicator Script" if is_indicator_gen else "Generated Pine Script"
            st.markdown(f"#### {_gen_label}")

            # Validate / lint
            gen_for_validate = PineScriptGenerator(
                profile=selected_profile_data,
                indicators=chosen_indicators,
                indicator_params=indicator_params,
                strategy_name=strategy_name,
                entry_mode=entry_mode,
                timeframe=timeframe,
                output_type=_output_type,
            )
            is_valid, messages = gen_for_validate.validate()

            if is_valid:
                st.success("Lint passed — no issues found.")
            else:
                for msg in messages:
                    if "error" in msg.lower():
                        st.error(msg)
                    else:
                        st.warning(msg)

            st.code(code, language="")

            dl_name = (strategy_name.strip().replace(" ", "_") or "strategy") + ".pine"
            st.download_button(
                label="⬇️ Download .pine file",
                data=code,
                file_name=dl_name,
                mime="text/plain",
                key="gen_download_btn",
            )

    # ══════════════════════════════════════════════════════════════════════
    # Tab 2 — Library
    # ══════════════════════════════════════════════════════════════════════
    with tab_lib:

        # Split DEFAULT_STRATEGIES and user strategies by type
        bi_strats = [s for s in DEFAULT_STRATEGIES if not s.get("is_indicator")]
        bi_inds   = [s for s in DEFAULT_STRATEGIES if s.get("is_indicator")]

        all_saved = supabase_db.get_saved_strategies(workspace_id)
        # Indicators stored with entry_mode="—" (set by the Generate tab)
        usr_strats = [s for s in all_saved if (s.get("entry_mode") or "—") not in ("—",)]
        usr_inds   = [s for s in all_saved if (s.get("entry_mode") or "—") in ("—",)]

        # Get library sub-tab from query params
        lib_tab = params.get("lt", "strategies")  # lt = library tab
        lib_tab_index = 0 if lib_tab == "strategies" else 1

        lib_strat_tab, lib_ind_tab = st.tabs([
            f"📈 Strategies ({len(bi_strats) + len(usr_strats)})",
            f"📊 Indicators ({len(bi_inds) + len(usr_inds)})",
        ])

        # ── helper: build a row dict from a strategy/indicator dict ──────────
        def _lib_row(s: dict) -> dict:
            inds = s.get("indicators", [])
            inds_str = ", ".join(i.upper() for i in inds) if isinstance(inds, list) else str(inds)
            return {
                "Name":       s.get("name", "—"),
                "Profile":    s.get("profile_name", "—"),
                "Indicators": inds_str or "—",
                "Entry":      (s.get("entry_mode", "—") or "—").replace("_", " ").title(),
                "TF":         s.get("timeframe", "—") or "—",
            }

        def _lib_saved_date(s: dict) -> str:
            created = s.get("created_at", "") or s.get("updated_at", "") or ""
            if created:
                try:
                    from datetime import datetime as _dt
                    return _dt.fromisoformat(
                        created.replace("Z", "+00:00")
                    ).strftime("%Y-%m-%d")
                except Exception:
                    pass
            return "—"

        def _lib_user_crud(strategies: list[dict], key_prefix: str, show_backtest: bool = True) -> None:
            """Render a user-library CRUD table for strategies or indicators."""
            if not strategies:
                st.caption("Nothing saved here yet. Use **⚡ Generate** to create and save scripts.")
                return

            rows = []
            for s in strategies:
                row = _lib_row(s)
                row["Saved"] = _lib_saved_date(s)
                rows.append(row)

            lib_event = st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key=f"lib_df_{key_prefix}",
            )

            sel_rows = lib_event.selection.rows if lib_event.selection else []
            if sel_rows:
                sel_idx   = sel_rows[0]
                sel_strat = strategies[sel_idx]
                sel_name  = sel_strat.get("name", "")
                sel_code  = sel_strat.get("code", "")

                st.markdown(f"**Selected:** {sel_name}")
                n_cols = 4 if show_backtest else 3
                cols = st.columns(n_cols)

                with cols[0]:
                    with st.expander("📋 View Code", expanded=False):
                        st.code(sel_code, language="")
                        dl_fn = (sel_name.replace(" ", "_") or "script") + ".pine"
                        st.download_button(
                            "⬇️ Download .pine",
                            data=sel_code, file_name=dl_fn,
                            mime="text/plain", key=f"{key_prefix}_dl_{sel_idx}",
                        )

                with cols[1]:
                    with st.expander("✏️ Rename", expanded=False):
                        new_name = st.text_input(
                            "New name", value=sel_name,
                            key=f"{key_prefix}_rename_input_{sel_idx}",
                        )
                        if st.button("✅ Apply", key=f"{key_prefix}_rename_btn_{sel_idx}"):
                            new_name_clean = new_name.strip()
                            if new_name_clean and new_name_clean != sel_name:
                                supabase_db.rename_strategy(sel_name, new_name_clean, workspace_id)
                                st.success(f"Renamed to **{new_name_clean}**")
                                st.rerun()
                            elif not new_name_clean:
                                st.error("Name cannot be empty.")
                            else:
                                st.info("Name is unchanged.")

                with cols[2]:
                    with st.expander("🗑️ Delete", expanded=False):
                        st.warning(f"Permanently delete **{sel_name}**?")
                        if st.button(
                            "🗑️ Confirm Delete",
                            key=f"{key_prefix}_del_btn_{sel_idx}",
                            type="primary",
                        ):
                            supabase_db.delete_strategy(sel_name, workspace_id)
                            st.success(f"Deleted **{sel_name}**.")
                            st.rerun()

                if show_backtest:
                    with cols[3]:
                        _lib_backtest_expander(sel_strat, f"{key_prefix}_{sel_idx}")
            else:
                st.caption("↑ Click a row to view, rename, or delete a script.")

        # ══════════════════════════════════════════════════════════════════════
        # Strategies tab
        # ══════════════════════════════════════════════════════════════════════
        with lib_strat_tab:

            # ── Built-in Strategies ───────────────────────────────────────────
            st.markdown("#### 🔒 Built-in Strategies")
            st.caption(
                "Pre-built strategies shipped with the app — read-only.  "
                "Use **🧪 Backtest** or **📋 View Code** to copy into TradingView."
            )

            if bi_strats:
                bi_s_rows = [_lib_row(s) for s in bi_strats]
                bi_s_event = st.dataframe(
                    pd.DataFrame(bi_s_rows),
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="lib_bi_strat_df",
                )
                bi_s_sel = bi_s_event.selection.rows if bi_s_event.selection else []
                if bi_s_sel:
                    bi_s = bi_strats[bi_s_sel[0]]
                    st.markdown(f"**Selected:** {bi_s['name']}  🔒 *built-in*")
                    bsc1, bsc2 = st.columns(2)
                    with bsc1:
                        with st.expander("📋 View Code", expanded=False):
                            st.code(bi_s["code"], language="")
                            st.download_button(
                                "⬇️ Download .pine",
                                data=bi_s["code"],
                                file_name=bi_s["name"].replace(" ", "_") + ".pine",
                                mime="text/plain",
                                key=f"bi_s_dl_{bi_s_sel[0]}",
                            )
                    with bsc2:
                        _lib_backtest_expander(bi_s, f"bi_s_{bi_s_sel[0]}")
                else:
                    st.caption("↑ Click a built-in strategy to view its code or run a backtest.")
            else:
                st.caption("No built-in strategies available.")

            st.markdown("---")

            # ── Your Strategies ───────────────────────────────────────────────
            st.markdown("#### 📚 Your Strategies")
            _lib_user_crud(usr_strats, key_prefix="ust", show_backtest=True)

        # ══════════════════════════════════════════════════════════════════════
        # Indicators tab
        # ══════════════════════════════════════════════════════════════════════
        with lib_ind_tab:

            # ── Built-in Indicators ───────────────────────────────────────────
            st.markdown("#### 🔒 Built-in Indicators")
            st.caption(
                "Pre-built indicator scripts shipped with the app — read-only.  "
                "Use **📋 View Code** to copy into TradingView."
            )

            if bi_inds:
                bi_i_rows = [_lib_row(s) for s in bi_inds]
                # Drop the Entry column — not relevant for indicators
                bi_i_df = pd.DataFrame(bi_i_rows).drop(columns=["Entry"], errors="ignore")
                bi_i_event = st.dataframe(
                    bi_i_df,
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="lib_bi_ind_df",
                )
                bi_i_sel = bi_i_event.selection.rows if bi_i_event.selection else []
                if bi_i_sel:
                    bi_i = bi_inds[bi_i_sel[0]]
                    st.markdown(f"**Selected:** {bi_i['name']}  🔒 *built-in*")
                    with st.expander("📋 View Code", expanded=False):
                        st.code(bi_i["code"], language="")
                        st.download_button(
                            "⬇️ Download .pine",
                            data=bi_i["code"],
                            file_name=bi_i["name"].replace(" ", "_") + ".pine",
                            mime="text/plain",
                            key=f"bi_i_dl_{bi_i_sel[0]}",
                        )
                else:
                    st.caption("↑ Click a built-in indicator to view its code.")
            else:
                st.caption("No built-in indicators available.")

            st.markdown("---")

            # ── Your Indicators ───────────────────────────────────────────────
            st.markdown("#### 📊 Your Indicators")
            _lib_user_crud(usr_inds, key_prefix="uid", show_backtest=False)


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard Page
# ─────────────────────────────────────────────────────────────────────────────

def render_dashboard_page(workspace_id: str = "default") -> None:
    # A. Ensure watchlist is loaded
    if "watchlist" not in st.session_state:
        wl = supabase_db.get_watchlist(workspace_id)
        st.session_state.watchlist = wl if wl else DEFAULT_WATCHLIST.copy()

    # B. Horizon — read from session state BEFORE the selectbox renders at the
    #    bottom.  Streamlit writes widget values to session_state before the
    #    script runs, so this always reflects the user's current selection.
    _default_horizon = "1 month (21td)"
    st.session_state.setdefault("horizon_select", _default_horizon)
    horizon_label = st.session_state["horizon_select"]
    if horizon_label not in HORIZON_MAP:
        horizon_label = _default_horizon
    horizon_td = HORIZON_MAP[horizon_label]

    # C. Analysis logic
    # "▶ Run Analysis" (rendered below) sets _do_run_all=True and calls st.rerun().
    # On the next render this flag is popped and the analysis block fires.
    run_all = bool(st.session_state.pop("_do_run_all", False))
    results = st.session_state.get("results", {})

    if run_all or not results:
        results = {}
        if st.session_state.watchlist:
            prog = st.progress(0, text="Running analysis…")
            for i, ticker in enumerate(st.session_state.watchlist):
                prog.progress(
                    (i + 1) / len(st.session_state.watchlist),
                    text=f"Analyzing {ticker}…",
                )
                results[ticker] = analyze(ticker, horizon_td=horizon_td)
            prog.empty()
        st.session_state["results"]  = results
        st.session_state["horizon"]  = horizon_td
        st.session_state["last_run"] = datetime.datetime.now().strftime("%H:%M:%S")

    if st.session_state.get("horizon") != horizon_td and results:
        st.session_state.pop("results", None)
        st.session_state.pop("selected_ticker", None)
        st.rerun()

    # D. Last-run caption + results grid
    if st.session_state.get("last_run"):
        st.caption(f"Last run: {st.session_state['last_run']}")

    render_dashboard(results)

    # E. Remove-selected action strip — appears below the table when a row is selected
    sel_now = st.session_state.get("selected_ticker")
    if sel_now and sel_now in st.session_state.watchlist:
        rm_col, _ = st.columns([2, 6])
        with rm_col:
            if st.button(f"✕ Remove {sel_now}", key="btn_remove_selected"):
                st.session_state.watchlist.remove(sel_now)
                supabase_db.save_watchlist(st.session_state.watchlist, workspace_id)
                st.session_state.get("results", {}).pop(sel_now, None)
                st.session_state.pop("selected_ticker", None)
                st.rerun()

    # F. Bottom controls bar — styled as a table footer
    with st.container(border=True):
        col_add, col_run, col_hz = st.columns([4, 2, 2])

        with col_add:
            with st.form("add_ticker_form", clear_on_submit=True):
                c1, c2 = st.columns([5, 1])
                new_sym = c1.text_input(
                    "", placeholder="Add ticker…  e.g. AAPL",
                    label_visibility="collapsed", key="add_ticker_input",
                )
                if c2.form_submit_button("＋", use_container_width=True):
                    sym = new_sym.upper().strip()
                    if sym and sym not in st.session_state.watchlist:
                        st.session_state.watchlist.append(sym)
                        supabase_db.save_watchlist(
                            st.session_state.watchlist, workspace_id
                        )
                        # Auto-run analysis for the new ticker immediately
                        cur = st.session_state.get("results", {})
                        with st.spinner(f"Analyzing {sym}…"):
                            cur[sym] = analyze(sym, horizon_td=horizon_td)
                        st.session_state["results"]  = cur
                        st.session_state["selected_ticker"] = sym
                        st.session_state["last_run"] = datetime.datetime.now().strftime(
                            "%H:%M:%S"
                        )
                        st.rerun()

        with col_run:
            if st.button(
                "▶ Run Analysis", type="primary",
                use_container_width=True, key="btn_run_all",
            ):
                st.session_state["_do_run_all"] = True
                st.rerun()

        with col_hz:
            st.selectbox(
                "Horizon",
                list(HORIZON_MAP.keys()),
                index=list(HORIZON_MAP.keys()).index(horizon_label),
                key="horizon_select",
                label_visibility="collapsed",
            )

    # G. Deep dive
    st.divider()
    sel = st.session_state.get("selected_ticker")
    if not sel:
        return
    r_base = results.get(sel, {})
    if r_base.get("error"):
        st.error(r_base["error"])
        return

    # Profile override cache
    chosen_profile_name = st.session_state.get(f"profile_override_{sel}", "(auto)")
    all_p = get_all_profiles(workspace_id)
    override_profile = None
    if chosen_profile_name != "(auto)":
        override_profile = all_p.get(chosen_profile_name)

    cache_key = f"oc_{sel}_{chosen_profile_name}_{horizon_td}"
    if override_profile is not None:
        if st.session_state.get("override_cache", {}).get("key") != cache_key:
            r_show = analyze(sel, horizon_td=horizon_td, profile_override=override_profile)
            st.session_state["override_cache"] = {"key": cache_key, "result": r_show}
        else:
            r_show = st.session_state["override_cache"]["result"]
    else:
        r_show = r_base

    render_deep_dive(r_show)


# ─────────────────────────────────────────────────────────────────────────────
# Profile Settings Page
# ─────────────────────────────────────────────────────────────────────────────

def render_profiles_page(workspace_id: str = "default") -> None:
    st.title("⚙️ Profiles")

    st.caption(
        "Profiles control how the analysis engine interprets RSI, ADX, and sets stop-loss / "
        "take-profit levels for each asset type. Built-in profiles are read-only; create custom "
        "ones below and optionally pin specific tickers to them."
    )

    tab1, tab2, tab3 = st.tabs(["📊 Built-in Profiles", "✏️ Custom Profiles", "🎯 Ticker Overrides"])

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 1 — Built-in profiles (read-only)
    # ─────────────────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Built-in Profile Templates")
        st.caption("Read-only. These ship with the bot and cannot be modified.")

        rows = []
        for key, p in PROFILES.items():
            rr_implied = f"{p['tp_pct']/p['sl_pct']:.1f}:1"
            rows.append({
                "Key":        key,
                "Category":   p["category"],
                "RSI Min":    p["rsi_bull_min"],
                "RSI Max":    p["rsi_bull_max"],
                "ADX ≥":      p["adx_threshold"],
                "SL %":       p["sl_pct"],
                "TP %":       p["tp_pct"],
                "Implied R/R": rr_implied,
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        st.markdown("---")
        st.subheader("Auto-detection Map")
        st.caption(
            "Tickers in this map are assigned a built-in profile automatically. "
            "Anything not listed defaults to Large-Cap Growth. "
            "Use the **Ticker Overrides** tab to change individual assignments."
        )
        map_rows = []
        for ticker_sym, key in sorted(TICKER_MAP.items()):
            p = PROFILES[key]
            map_rows.append({
                "Ticker":  ticker_sym,
                "Profile": p["category"],
                "Key":     key,
            })
        st.dataframe(pd.DataFrame(map_rows), hide_index=True, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 2 — Custom profiles (create / delete)
    # ─────────────────────────────────────────────────────────────────────────
    with tab2:
        custom = get_custom_profiles(workspace_id)

        if custom:
            st.subheader(f"Your Custom Profiles  ({len(custom)})")
            for cname, p in list(custom.items()):
                rr_implied = f"{p['tp_pct']/p['sl_pct']:.1f}:1"
                label = (f"**{cname}** — "
                         f"RSI {p['rsi_bull_min']}–{p['rsi_bull_max']}, "
                         f"ADX≥{p['adx_threshold']:.0f}, "
                         f"SL {p['sl_pct']}% / TP {p['tp_pct']}%  "
                         f"(R/R {rr_implied})")
                with st.expander(label, expanded=False):
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("RSI Min",   p["rsi_bull_min"])
                    m2.metric("RSI Max",   p["rsi_bull_max"])
                    m3.metric("ADX ≥",     p["adx_threshold"])
                    m4.metric("SL %",      p["sl_pct"])
                    m5.metric("TP %",      p["tp_pct"])
                    if p.get("description"):
                        st.caption(p["description"])
                    if st.button(f"🗑 Delete  '{cname}'", key=f"del_profile_{cname}",
                                 type="secondary"):
                        delete_custom_profile(cname, workspace_id)
                        st.toast(f"Deleted profile '{cname}'", icon="🗑")
                        st.rerun()
            st.markdown("---")
        else:
            st.info("No custom profiles yet — create one below.")

        # ── Create new profile form ───────────────────────────────────────────
        st.subheader("Create New Profile")

        # We split into two visual columns: name/desc/template on left, sliders on right
        left, right = st.columns([1, 1])

        with left:
            p_name = st.text_input("Profile name *", placeholder="e.g. My Crypto")
            p_desc = st.text_input("Description (optional)",
                                   placeholder="e.g. BTC, ETH, high-momentum crypto")

            base_options: dict[str, str] = {p["category"]: k for k, p in PROFILES.items()}
            base_label = st.selectbox("Copy settings from built-in template", list(base_options.keys()))
            base = PROFILES[base_options[base_label]]

        with right:
            p_rsi_min = st.slider("RSI Bull Min",      20, 60,   int(base["rsi_bull_min"]),  key="np_rsi_min")
            p_rsi_max = st.slider("RSI Bull Max",      50, 85,   int(base["rsi_bull_max"]),  key="np_rsi_max")
            p_adx     = st.slider("ADX Threshold",     10.0, 40.0, float(base["adx_threshold"]),
                                  step=0.5, key="np_adx")
            p_sl      = st.slider("Stop-Loss %",       1.0, 20.0, float(base["sl_pct"]),
                                  step=0.5, key="np_sl")
            p_tp      = st.slider("Take-Profit %",     3.0, 60.0, float(base["tp_pct"]),
                                  step=1.0, key="np_tp")

            if p_sl > 0:
                st.caption(f"Implied R/R: **{p_tp/p_sl:.1f}:1**")

        # Save button below both columns
        if st.button("✅ Save Profile", type="primary", key="save_new_profile"):
            name_clean = p_name.strip()
            if not name_clean:
                st.error("Profile name is required.")
            elif p_rsi_min >= p_rsi_max:
                st.error("RSI Min must be less than RSI Max.")
            elif name_clean in PROFILES:
                st.error(f"'{name_clean}' conflicts with a built-in profile key. Choose a different name.")
            else:
                save_custom_profile(name_clean, {
                    "category":      name_clean,
                    "rsi_bull_min":  p_rsi_min,
                    "rsi_bull_max":  p_rsi_max,
                    "adx_threshold": p_adx,
                    "sl_pct":        p_sl,
                    "tp_pct":        p_tp,
                    "description":   p_desc.strip(),
                }, workspace_id)
                st.toast(f"Profile '{name_clean}' saved!", icon="✅")
                st.rerun()

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 3 — Ticker overrides (assign any ticker to any profile)
    # ─────────────────────────────────────────────────────────────────────────
    with tab3:
        overrides  = get_ticker_overrides(workspace_id)
        all_p      = get_all_profiles(workspace_id)

        # ── Current overrides table ───────────────────────────────────────────
        if overrides:
            st.subheader(f"Active Ticker Overrides  ({len(overrides)})")
            st.caption("These take precedence over the auto-detection map.")

            override_rows = []
            for t_sym, key in sorted(overrides.items()):
                auto_key  = TICKER_MAP.get(t_sym, DEFAULT_PROFILE)
                auto_name = PROFILES.get(auto_key, PROFILES[DEFAULT_PROFILE])["category"]
                over_name = all_p.get(key, {}).get("category", key)
                override_rows.append({
                    "Ticker":            t_sym,
                    "Override Profile":  over_name,
                    "Auto-detected":     auto_name,
                    "Changed":           over_name != auto_name,
                })
            st.dataframe(pd.DataFrame(override_rows), hide_index=True, use_container_width=True)

            # Remove an override
            st.markdown("**Remove an override**")
            rc1, rc2 = st.columns([3, 1])
            rm_ticker = rc1.selectbox("Ticker to remove", sorted(overrides.keys()),
                                      key="rm_override_sel")
            if rc2.button("Remove", key="rm_override_btn"):
                remove_ticker_override(rm_ticker, workspace_id)
                st.toast(f"Removed override for {rm_ticker}", icon="🗑")
                st.rerun()

            st.markdown("---")
        else:
            st.info("No overrides set — all tickers use auto-detection.")

        # ── Add a new override ────────────────────────────────────────────────
        st.subheader("Add / Update Ticker Override")

        # Build dropdown: show category name, map back to key
        all_profile_names = {p["category"]: k for k, p in all_p.items()}

        ac1, ac2, ac3 = st.columns([2, 3, 1])
        ov_ticker  = ac1.text_input("Ticker", placeholder="e.g. NVDA, BTC-USD",
                                    key="ov_ticker_in").upper().strip()
        ov_profile = ac2.selectbox("Assign to profile", list(all_profile_names.keys()),
                                   key="ov_profile_sel")
        add_btn    = ac3.button("Save", type="primary", key="ov_add_btn")

        if add_btn:
            if not ov_ticker:
                st.error("Ticker is required.")
            else:
                ov_key = all_profile_names[ov_profile]
                set_ticker_override(ov_ticker, ov_key, workspace_id)
                st.toast(f"Override saved: {ov_ticker} → {ov_profile}", icon="✅")
                st.rerun()

        # Show current auto-detection for the entered ticker (live preview)
        if ov_ticker:
            auto_key  = TICKER_MAP.get(ov_ticker, DEFAULT_PROFILE)
            auto_name = PROFILES.get(auto_key, PROFILES[DEFAULT_PROFILE])["category"]
            current_override = overrides.get(ov_ticker)
            if current_override:
                cur_name = all_p.get(current_override, {}).get("category", current_override)
                st.caption(f"ℹ️ {ov_ticker} currently overridden → **{cur_name}** "
                           f"(auto-detect: {auto_name})")
            else:
                st.caption(f"ℹ️ {ov_ticker} currently uses auto-detect → **{auto_name}**")



# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    workspace_id = get_workspace_id()

    if "migration_attempted" not in st.session_state:
        if supabase_db.migrate_json_to_supabase(workspace_id="default"):
            st.toast("✅ Synced local profiles to cloud", icon="🔄")
        st.session_state.migration_attempted = True

    render_sidebar(workspace_id)  # sets st.session_state["page"] via radio

    page = st.session_state.get("page", "dashboard")
    if page == "profiles":
        render_profiles_page(workspace_id)
    elif page == "generator":
        render_strategies_page(workspace_id)
    else:
        render_dashboard_page(workspace_id)


if __name__ == "__main__":
    main()
