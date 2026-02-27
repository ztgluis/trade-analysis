"""
app.py  —  Trade Analysis · Trading Dashboard
Run with:  streamlit run backend/app.py
"""
from __future__ import annotations
import sys
import json
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
DEFAULT_WATCHLIST = ["GOOG", "META", "GLD", "SLV", "NFLX", "SNAP"]

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

def render_sidebar() -> tuple[list[str], int, bool]:
    st.sidebar.title("📈 Trade Analysis")
    st.sidebar.caption("Decision Dashboard · v1.0")
    st.sidebar.divider()

    # ── Watchlist ──────────────────────────────────────────────────────────
    st.sidebar.subheader("📋 Watchlist")
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = DEFAULT_WATCHLIST.copy()

    to_remove = None
    for sym in st.session_state.watchlist:
        r = st.session_state.get("results", {}).get(sym)
        verdict_emoji = "⚪"
        price_str     = ""
        if r and not r.get("error"):
            color_key = r.get("color", "grey")
            verdict_emoji = {"green": "🟢", "lime": "🟡", "red": "🔴",
                             "grey": "⚪", "yellow": "🟡"}.get(color_key, "⚪")
            price_str = f"  ${r['price']:,.0f}"

        c1, c2 = st.sidebar.columns([5, 1])
        # Clickable name → navigate to deep dive
        if c1.button(f"{verdict_emoji} {sym}{price_str}", key=f"nav_{sym}",
                     use_container_width=True, help=r.get("verdict", "") if r else ""):
            st.session_state["selected_ticker"] = sym
            st.session_state["page"] = "dashboard"
            st.rerun()
        if c2.button("✕", key=f"rm_{sym}", help=f"Remove {sym}"):
            to_remove = sym

    if to_remove:
        st.session_state.watchlist.remove(to_remove)
        st.session_state.get("results", {}).pop(to_remove, None)
        if st.session_state.get("selected_ticker") == to_remove:
            st.session_state.pop("selected_ticker", None)
        st.rerun()

    # Add symbol form
    with st.sidebar.form("add_symbol", clear_on_submit=True):
        new_sym = st.text_input("Add symbol", placeholder="e.g. AAPL").upper().strip()
        if st.form_submit_button("+ Add") and new_sym and new_sym not in st.session_state.watchlist:
            st.session_state.watchlist.append(new_sym)
            st.rerun()

    st.sidebar.divider()

    # ── Horizon ────────────────────────────────────────────────────────────
    st.sidebar.subheader("⏱ Horizon")
    horizon_label = st.sidebar.selectbox(
        "Analysis horizon", list(HORIZON_MAP.keys()),
        index=2,   # default: 1 month
        key="horizon_select",
    )
    horizon_td = HORIZON_MAP[horizon_label]

    st.sidebar.divider()

    # ── Actions ────────────────────────────────────────────────────────────
    run_all = st.sidebar.button("🔄 Run All Analysis", type="primary",
                                use_container_width=True)
    if "results" in st.session_state and st.session_state.results:
        last_run = st.session_state.get("last_run", "")
        st.sidebar.caption(f"Last run: {last_run}")

    if st.sidebar.button("⚙️ Profile Settings", use_container_width=True):
        st.session_state["page"] = "settings"
        st.rerun()

    return st.session_state.watchlist, horizon_td, run_all


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
# Profile Settings Page
# ─────────────────────────────────────────────────────────────────────────────

def render_profiles_page() -> None:
    # ── Header ────────────────────────────────────────────────────────────────
    col_back, col_title = st.columns([1, 6])
    with col_back:
        if st.button("← Dashboard", type="secondary"):
            st.session_state["page"] = "dashboard"
            st.rerun()
    with col_title:
        st.title("⚙️ Profile Settings")

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
        custom = get_custom_profiles()

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
                        delete_custom_profile(cname)
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
                })
                st.toast(f"Profile '{name_clean}' saved!", icon="✅")
                st.rerun()

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 3 — Ticker overrides (assign any ticker to any profile)
    # ─────────────────────────────────────────────────────────────────────────
    with tab3:
        overrides  = get_ticker_overrides()
        all_p      = get_all_profiles()

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
                remove_ticker_override(rm_ticker)
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
                set_ticker_override(ov_ticker, ov_key)
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
    watchlist, horizon_td, run_all = render_sidebar()

    # ── Page routing ──────────────────────────────────────────────────────────
    page = st.session_state.get("page", "dashboard")

    if page == "settings":
        render_profiles_page()
        return

    # ── Run analysis ─────────────────────────────────────────────────────────
    if run_all or "results" not in st.session_state or not st.session_state.get("results"):
        results: dict = {}
        prog = st.progress(0, text="Running analysis…")
        for i, ticker in enumerate(watchlist):
            prog.progress((i + 1) / len(watchlist), text=f"Analyzing {ticker}…")
            results[ticker] = analyze(ticker, horizon_td=horizon_td)
        prog.empty()
        st.session_state["results"]  = results
        st.session_state["horizon"]  = horizon_td
        st.session_state["last_run"] = datetime.datetime.now().strftime("%H:%M:%S")
    else:
        results = st.session_state.get("results", {})
        # Re-run if horizon changed
        if st.session_state.get("horizon") != horizon_td and results:
            st.session_state["results"] = {}
            st.session_state["override_cache"] = {}
            st.rerun()

    # ── Watchlist table (always visible) ─────────────────────────────────────
    st.title("📈 Trade Analysis")
    render_dashboard(results)

    # ── Deep dive (shown when a ticker is selected) ───────────────────────────
    sel = st.session_state.get("selected_ticker")
    if not sel or sel not in results or results[sel].get("error"):
        return

    st.divider()

    # Profile override handling
    profile_opts        = get_profile_options()
    chosen_profile_name = st.session_state.get(f"profile_override_{sel}")
    auto_profile        = results[sel].get("profile", "Large-Cap Growth")

    if chosen_profile_name and chosen_profile_name != auto_profile:
        # Check cache first
        cache_key = f"oc_{sel}_{chosen_profile_name}_{horizon_td}"
        if cache_key not in st.session_state.get("override_cache", {}):
            with st.spinner(f"Re-running {sel} with '{chosen_profile_name}' profile…"):
                override_profile = profile_opts.get(chosen_profile_name)
                r_show = analyze(sel, horizon_td=horizon_td, profile_override=override_profile)
            if "override_cache" not in st.session_state:
                st.session_state["override_cache"] = {}
            st.session_state["override_cache"][cache_key] = r_show
        else:
            r_show = st.session_state["override_cache"][cache_key]
    else:
        r_show = results[sel]

    render_deep_dive(r_show)


if __name__ == "__main__":
    main()
