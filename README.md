# Growth Signal Bot 📈

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
# Single ticker analysis (long setup)
python run_decide.py TSLA

# Short setup analysis
python run_decide_short.py NFLX

# Both long + short, best recommendation
python run_decide_unified.py GLD SNAP META

# Multi-ticker scan
python run_multi.py GOOG META NVDA TSLA
```

## Dashboard Features

### Watchlist View
- Summary table showing all tickers with verdict, scores, regime, and risk/reward
- Click any row to open detailed analysis
- Add/remove symbols on the fly

### Deep Dive (Per-Ticker)
- **Verdict + Profile Badge**: Current recommendation with applied profile
- **Score Bars**: Long (0-10) and short (0-10) scores
- **Condition Details**: Breakdown of why each score is what it is
- **Price Chart**: Last 6 months with moving averages (SMA200, SMA50, EMA20) and signal markers
- **Signal Timeline**: All signals (BUY, BOUNCE, SELL, divergences) chronologically
- **Base Rates**: Win rate at different horizons (1 week, 2 weeks, 1 month, etc.)
- **Key Levels**: Support/resistance with distance from current price
- **Action Plan**: Entry zones, stop-loss, target, R/R ratio
- **Earnings & Macro**: Next earnings date, macro regime (SPY/QQQ), alpha vs benchmark

### Profile Settings

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
- Saved to `~/.trader-bot/custom_profiles.json`

**Tab 3 — Ticker Overrides**
- Pin any ticker (TSLA, RIVN, BTC-USD, etc.) to any profile
- Override takes precedence over auto-detection
- Live preview of current auto-detected profile
- Overrides persist automatically

## Project Structure

```
trader-bot/
├── backend/
│   ├── analysis/
│   │   ├── decision_engine.py       # Core analysis: buy/sell scoring, verdicts
│   │   └── asset_profiles.py        # Profile system + per-ticker overrides
│   │
│   ├── backtester/
│   │   ├── engine.py                # Backtest runner
│   │   ├── data.py                  # yfinance data fetching + caching
│   │   ├── indicators.py            # Technical indicators (RSI, ADX, MACD, etc)
│   │   ├── metrics.py               # Performance metrics (Sharpe, drawdown, etc)
│   │   └── portfolio.py             # Portfolio simulation
│   │
│   ├── strategies/
│   │   └── growth_signal_bot.py     # Growth stock signal strategy
│   │
│   ├── app.py                       # Streamlit dashboard
│   │
│   ├── run_decide.py                # Long decision analysis
│   ├── run_decide_short.py          # Short decision analysis
│   ├── run_decide_unified.py        # Both + best recommendation
│   ├── run_multi.py                 # Multi-ticker scan
│   ├── run_backtest.py              # Backtester runner
│   └── requirements.txt
│
├── pine-scripts/                    # TradingView Pine Script indicators (archived)
│   ├── abnb_signal_bot_v3.pine
│   └── swing_signal_bot_v1.pine
│
├── docs/
│   └── (strategy notes)
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

Set in sidebar under **⏱ Horizon**. Controls:
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
