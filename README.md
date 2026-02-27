# Trade Analysis 📈

A decision-making engine for swing traders — analyzes bull/bear regimes, identifies entry setups, and calculates risk/reward ratios across multiple asset classes with tuned per-ticker profiles.

## Features

- **Real-time Decision Analysis**: Long/short scoring (0-10) with actionable verdicts (STRONG LONG, LEAN SHORT, BOUNCE, WAIT)
- **Asset Class Profiles**: 5 built-in profiles (Large-Cap Growth, High-Vol Tech, Small-Cap Volatile, Precious Metals, ETFs) + unlimited custom profiles
- **Per-Ticker Overrides**: Pin specific tickers to custom profiles; changes persist automatically
- **Risk/Reward Analysis**: Entry zones, stop-loss, take-profit targets with implied R/R ratios
- **Base Rate Analysis**: Historical win rates at different time horizons (1 week → 6 months)
- **Regime Detection**: Bull/bear/neutral classification with SMA200 trend confirmation
- **Macro Context**: SPY/QQQ regime + alpha vs benchmark (1-month returns)
- **Technical Signals**: BUY, BOUNCE, SELL, Bull Div, Bear Div with timeline visualization
- **Live Price Data**: Real-time OHLCV fetching via yfinance with automatic caching

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Run the Dashboard

```bash
streamlit run app.py
```

Then visit `http://localhost:8501`

### 3. Try Analysis Commands

```bash
# Unified long/short verdict — same engine as the Dashboard (recommended)
python run_analyze.py TSLA
python run_analyze.py GOOG META NVDA          # multi-symbol

# Deep-dive long analysis only (detailed buy/hold/sell checklist)
python run_analyze_long.py META
python run_analyze_long.py META --horizon 21  # 1-month horizon

# Deep-dive short analysis only
python run_analyze_short.py SNAP

# Raw indicator scan — dumps current state of every indicator (no verdict)
python run_scan.py META
python run_scan.py META --tf 1h               # 1-hour timeframe

# Multi-ticker backtest comparison table
python run_backtest_multi.py GOOG META NVDA TSLA

# Single-ticker backtest
python run_backtest.py --ticker TSLA --period 3y
```

## Configuration

### Supabase Setup (Optional but Recommended)

Custom profiles and ticker overrides can persist across both local development and Streamlit Cloud deployments using Supabase (a PostgreSQL backend). Without setup, profiles are stored locally in `~/.trader-bot/custom_profiles.json` (local-only, lost on cloud redeploy).

**Step 1: Create Supabase Project**
1. Sign up at https://supabase.com (free tier available)
2. Create a new project
3. Go to **Settings → API** and copy:
   - Project URL (e.g., `https://xxx.supabase.co`)
   - Anon Public Key (the `eyJ...` token)

**Step 2: Create Database Tables**

In Supabase **SQL Editor**, run:

```sql
CREATE TABLE custom_profiles (
  id BIGSERIAL PRIMARY KEY,
  profile_name TEXT NOT NULL,
  workspace_id TEXT NOT NULL DEFAULT 'default',
  category TEXT NOT NULL,
  rsi_bull_min INTEGER NOT NULL,
  rsi_bull_max INTEGER NOT NULL,
  adx_threshold DECIMAL NOT NULL,
  sl_pct DECIMAL NOT NULL,
  tp_pct DECIMAL NOT NULL,
  description TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW(),
  CONSTRAINT custom_profiles_name_workspace_key UNIQUE (profile_name, workspace_id)
);

CREATE TABLE ticker_overrides (
  id BIGSERIAL PRIMARY KEY,
  ticker TEXT NOT NULL,
  profile_name TEXT NOT NULL,
  workspace_id TEXT NOT NULL DEFAULT 'default',
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW(),
  CONSTRAINT ticker_overrides_ticker_workspace_key UNIQUE (ticker, workspace_id),
  CONSTRAINT ticker_overrides_profile_workspace_fkey
    FOREIGN KEY (profile_name, workspace_id)
    REFERENCES custom_profiles (profile_name, workspace_id)
    ON DELETE CASCADE
);

CREATE TABLE watchlist (
  id BIGSERIAL PRIMARY KEY,
  workspace_id TEXT NOT NULL DEFAULT 'default',
  tickers JSONB NOT NULL DEFAULT '[]',
  updated_at TIMESTAMP DEFAULT NOW(),
  CONSTRAINT watchlist_workspace_key UNIQUE (workspace_id)
);

CREATE TABLE saved_strategies (
  id           BIGSERIAL  PRIMARY KEY,
  workspace_id TEXT       NOT NULL DEFAULT 'default',
  name         TEXT       NOT NULL,
  code         TEXT       NOT NULL,
  profile_name TEXT,
  indicators   JSONB      DEFAULT '[]',
  entry_mode   TEXT       DEFAULT 'all_signals',
  timeframe    TEXT       DEFAULT 'D',
  created_at   TIMESTAMP  DEFAULT NOW(),
  updated_at   TIMESTAMP  DEFAULT NOW(),
  CONSTRAINT saved_strategies_name_workspace_key UNIQUE (name, workspace_id)
);
```

> **Existing installation?** If you already have these tables without `workspace_id`, run the migration script at `docs/supabase_workspace_migration.sql` instead. The `saved_strategies` table uses `CREATE TABLE IF NOT EXISTS` so it is safe to re-run on any install.

**Step 3: Local Development Setup**

Create `~/.streamlit/secrets.toml`:

```toml
[supabase]
url = "https://your-project.supabase.co"
key = "your-anon-public-key"
```

Then restart Streamlit — you'll see "✅ Synced local profiles to cloud" on first run.

**Step 4: Streamlit Cloud Deployment**

In your Streamlit Cloud app dashboard:
1. Click **Settings → Secrets**
2. Paste the same credentials:
   ```toml
   [supabase]
   url = "https://your-project.supabase.co"
   key = "your-anon-public-key"
   ```
3. Click **Save** and redeploy

Now both local and cloud environments use the same Supabase database — profiles are synced automatically.

**Fallback Behavior**: If Supabase credentials are missing or unavailable, the app gracefully falls back to local JSON files (`~/.trader-bot/custom_profiles.json`).

---

### Multi-User Workspaces

Each visitor gets an isolated **workspace** — their own watchlist, custom profiles, and ticker overrides — with no login required.

**How it works:**
- On first visit, an 8-character token is auto-generated and added to the URL: `https://yourapp.streamlit.app/?w=a3f9c2d1`
- Bookmarking that URL resumes your workspace on any device
- All data (watchlist, profiles, overrides, saved strategies) is stored in Supabase scoped to your token
- Other users visiting the plain URL get their own fresh workspace

**Workspace expander** (in the sidebar under 🔑 Workspace):
- Shows your current token
- "Switch to another workspace" input — paste a token to share data between devices or team members

**Claiming legacy data**: If you had data before workspaces were added (stored in the `default` workspace), enter `default` as your token in the Switch input.

**No Supabase?** Workspaces still work — data is stored in local JSON keyed by workspace ID. Not shared across devices in that case.

## App Navigation

The app has three sections, accessible via the left sidebar radio:

### 📊 Dashboard
- **Results grid**: Summary table showing verdict, scores, regime, and risk/reward for all watchlist tickers. Click any row to open the deep dive below.
- **✕ Remove**: When a row is selected, a remove button appears below the table to delete that ticker from your watchlist.
- **Add ticker** (bottom bar): Type a symbol and press ＋ — the new ticker is immediately analyzed and selected.
- **▶ Run Analysis** (bottom bar): Re-runs analysis on all watchlist tickers.
- **⏱ Horizon** (bottom bar): Select analysis horizon (1 week → 6 months) — affects scoring calibration and base rates.
- **Deep Dive** (click any row): Per-ticker detailed analysis panel below the table.

### ⚙️ Profiles

**Tab 1 — Built-in Profiles**
- Read-only comparison table of all 5 templates
- Auto-detection map showing which tickers use which profile

**Tab 2 — Custom Profiles**
- Create profiles by adjusting 5 sliders:
  - RSI Bull Min/Max (entry range)
  - ADX Threshold (trend confirmation)
  - Stop-Loss % / Take-Profit % (risk/reward)
- Implied R/R ratio auto-calculated
- Delete custom profiles (orphaned ticker overrides auto-cleaned)
- Saved to Supabase (or `~/.trader-bot/custom_profiles.json` as fallback)

**Tab 3 — Ticker Overrides**
- Pin any ticker (TSLA, RIVN, BTC-USD, etc.) to any profile
- Override takes precedence over auto-detection
- Live preview of current auto-detected profile
- Overrides persist automatically

### 🎛️ Generator

**Tab 1 — ⚡ Generate**
- Choose a **template** (Momentum Only / Trend+Momentum / Full Strategy / Custom)
- Select **Script Type**:
  - **📈 Strategy** — full `strategy()` script with entry/exit logic, stop-loss, take-profit, and signal shapes
  - **📊 Indicator** — `indicator()` script with calculations + plots only (no trade orders); hides entry mode options
- Toggle individual **indicators** (RSI, MACD, ADX, Fast EMA, Mid SMA, Slow SMA, VWAP, ATR, Fibonacci, Volume) and configure their parameters
  - **VWAP** supports anchor period (Session / Week / Month / Quarter / Year) and source (HLC3 / HL2 / Close / OHLC4)
  - EMA/SMA lengths are user-configurable; labels reflect role (Fast/Mid/Slow) rather than fixed period
- Set **entry mode** (All Signals / Buy Only / Strong Buy Only) — strategy scripts only
- Set script name and timeframe
- Preview generated code + lint validation, then **download as `.pine` file** or **💾 Save to Library**

**Tab 2 — 📚 Library**

Split into two sub-tabs:

**📈 Strategies**
- **🔒 Built-in Strategies** (read-only): 3 pre-built strategy scripts shipped with the app
  - *Long Signal Strategy v4 (Daily)* — full indicator suite, daily timeframe
  - *Long Signal Strategy v1 (1H)* — full indicator suite, 1-hour timeframe
  - *Swing Signal Strategy v1* — swing-optimised subset, daily timeframe
- **📚 Your Strategies** (full CRUD): user-generated strategy scripts
  - Click a row to reveal: **📋 View Code**, **✏️ Rename**, **🗑️ Delete**, **🧪 Backtest**
  - Backtest runs `LongSignalStrategy` configured with the strategy's profile (RSI zone, ADX threshold, SL/TP %) and entry mode; shows 6 metrics: Total Return vs B&H, CAGR, Sharpe, Max Drawdown, Win Rate, Profit Factor

**📊 Indicators**
- **🔒 Built-in Indicators** (read-only): 2 pre-built indicator scripts
  - *Triple MA [Indicator]* — EMA20, SMA50, SMA200 with regime background
  - *VIX Spike Warning [Indicator]* — VWAP-based volatility overlay
- **📊 Your Indicators** (full CRUD): user-generated indicator scripts
  - Click a row to reveal: **📋 View Code**, **✏️ Rename**, **🗑️ Delete**

All saved scripts are workspace-scoped — each workspace has its own library.

### Deep Dive (Per-Ticker)
- **Verdict + Profile Badge**: Current recommendation with applied profile
- **Score Bars**: Long (0-16) and short scores
- **Condition Details**: Breakdown of why each score is what it is
- **Price Chart**: Last 6 months with moving averages (SMA200, SMA50, EMA20) and signal markers
- **Signal Timeline**: All signals (BUY, BOUNCE, SELL, divergences) chronologically
- **Base Rates**: Win rate at different horizons (1 week, 2 weeks, 1 month, etc.)
- **Key Levels**: Support/resistance with distance from current price
- **Action Plan**: Entry zones, stop-loss, target, R/R ratio
- **Earnings & Macro**: Next earnings date, macro regime (SPY/QQQ), alpha vs benchmark

## Project Structure

```
trader-bot/
├── backend/
│   ├── analysis/
│   │   ├── decision_engine.py       # Core analysis: buy/sell scoring, verdicts
│   │   ├── asset_profiles.py        # Profile system + per-ticker overrides
│   │   └── supabase_db.py           # Supabase persistence layer (workspace-scoped)
│   │
│   ├── backtester/
│   │   ├── engine.py                # Backtest runner
│   │   ├── data.py                  # yfinance data fetching + caching
│   │   ├── indicators.py            # Technical indicators (RSI, ADX, MACD, etc)
│   │   ├── metrics.py               # Performance metrics (Sharpe, drawdown, etc)
│   │   └── portfolio.py             # Portfolio simulation
│   │
│   ├── strategies/
│   │   ├── base.py                  # Abstract BaseStrategy interface
│   │   ├── long_signal_strategy.py  # LongSignalStrategy v4.3 (main strategy)
│   │   ├── growth_signal_bot.py     # Backwards-compat shim → LongSignalStrategy
│   │   └── pine_generator.py        # Pine Script v6 code generator
│   │
│   ├── app.py                       # Streamlit dashboard (Dashboard / Profiles / Strategies)
│   │
│   ├── run_analyze.py               # Unified long+short verdict (CLI wrapper for decision_engine)
│   ├── run_analyze_long.py          # Deep-dive LONG analysis with base rates + checklist
│   ├── run_analyze_short.py         # Deep-dive SHORT analysis with base rates + checklist
│   ├── run_scan.py                  # Raw indicator scan — current state dump, no verdict
│   ├── run_backtest_multi.py        # Multi-ticker backtest comparison table
│   ├── run_backtest.py              # Single-ticker backtest runner
│   └── requirements.txt
│
├── pine-scripts/                    # TradingView Pine Script strategies
│   ├── growth_signal_bot_v4.pine    # Daily strategy v4.3
│   ├── growth_signal_bot_1h_v1.pine # 1H strategy
│   ├── swing_signal_bot_v1.pine
│   └── lint_pine.py                 # Pine Script linter / validator
│
├── docs/
│   └── supabase_workspace_migration.sql  # Run if upgrading from pre-workspace schema
│
└── README.md
```

## Architecture

### Decision Engine (`analysis/decision_engine.py`)

Analyzes a ticker and returns:

```python
{
  "ticker": "TSLA",
  "verdict": "STRONG LONG",
  "color": "green",
  "action": "Enter on dip to $245; target $345",
  "long_score": 9,
  "short_score": 2,
  "long_conditions": [(True, "RSI in bull range"), ...],
  "short_conditions": [(False, "RSI not overbought"), ...],
  "price": 275.50,
  "rsi": 58.2,
  "adx": 22.5,
  "regime": "bull",
  "profile": "High-Vol Tech",  # Applied profile name
  "sl_price": 250.00,         # Stop-loss
  "tp_price": 345.00,         # Take-profit
  "rr_ratio": 1.9,            # Risk/Reward
  "entry_lo": 268.00,
  "entry_hi": 285.00,
  "df_chart": <DataFrame>,    # OHLCV + indicators for charting
  "base_rates": {...},        # Win rate at each horizon
  ...
}
```

### Asset Profiles (`analysis/asset_profiles.py`)

Each profile defines parameter ranges:

```python
{
  "category": "High-Vol Tech",
  "rsi_bull_min": 45,
  "rsi_bull_max": 72,
  "adx_threshold": 25.0,
  "sl_pct": 8.0,
  "tp_pct": 24.0,
}
```

**Resolution priority** for a given ticker:
1. User's custom ticker override (e.g., TSLA → "my_tsla_profile")
2. Built-in TICKER_MAP (e.g., TSLA → "_highvol")
3. DEFAULT_PROFILE ("_growth")

**Built-in Profiles**:
- `_growth` — Large-cap tech (GOOG, META, MSFT, AAPL, etc.)
- `_highvol` — High-momentum (TSLA, NVDA, PLTR, COIN, etc.)
- `_smallvol` — Small/mid-cap volatile (SNAP, RIVN, LCID, SOFI, etc.)
- `_metals` — Precious metals (GLD, SLV, etc.)
- `_etf` — Index ETFs (SPY, QQQ, IWM, etc.)

### Backtester (`backtester/`)

Simulates historical trades using the GrowthSignalBot strategy:

```bash
python run_backtest.py --symbol GOOG --start 2020-01-01
```

Returns win rate, profit factor, max drawdown, Sharpe ratio.

## Data

- **Source**: yfinance (live, real-time)
- **Caching**: `.cache/ohlcv/` — pickled OHLCV DataFrames by symbol
- **Refresh**: Automatic on each analysis run (fetches latest bars)
- **History**: Pulls maximum available history (max 50 years), then recent analysis uses last 252 trading days (1 year)

## Configuration

### Horizon (Time Perspective)

Set on the **Dashboard** page via the **⏱ Horizon** dropdown. Controls:
- How many trading days to analyze (e.g., 1 month = 21 trading days)
- Base rate lookback (win rate at this horizon)
- **Does NOT change the data analyzed** — always uses 1 year of OHLCV, but scores/verdicts are calibrated for the selected horizon

Options: 1 week, 2 weeks, 1 month (default), 2 months, 3 months, 6 months

### Analysis Horizon vs Data Analyzed

- **Horizon**: The expected duration of a trade if it plays out
- **Data**: Always 252 trading days of historical OHLCV, plus 5+ years for base rate calculations

## Git Automation

Changes are auto-committed and pushed to GitHub after each edit:

```
auto: checkpoint
```

Manual commits still work; just run `git commit` as usual.

## Troubleshooting

### "No data for symbol"
- yfinance may not recognize the ticker or it doesn't have sufficient history
- Try ticker variations (e.g., BTC-USD instead of BTC)

### Earnings data not showing
- Earnings come from yfinance, which may not have future earnings for all tickers
- Check manually on Seeking Alpha if needed

### Override not appearing in dropdown
- Make sure the custom profile is created in Tab 2 first
- Custom profile name must match exactly

### Slow analysis on first run
- First fetch downloads 5+ years of data; subsequent runs use cache
- Expect 2–5 seconds per ticker after cache is warm

## Future Improvements

- Live broker integration (Schwab API)
- Email/Discord alerts on new signals
- Portfolio backtester (multi-leg positions)
- Machine learning regime detection
- Web API (FastAPI) instead of just CLI + Streamlit
