# Trader Bot

Signal generation system for swing/day trading.
Target stocks: TSLA, RIVN, NVDA, NET, SLV, GOOG, ABNB, NFLX, SQ, SPX, NDAQ, GOLD, SILVER

## Project Structure

```
trader-bot/
├── pine-scripts/               # TradingView Pine Script indicators
│   ├── abnb_signal_bot_v3.pine     ← ABNB-tuned: VWAP + scoring (active)
│   └── swing_signal_bot_v1.pine    ← Generic swing signal template
│
├── backend/                    # Python webhook server (Phase 2)
│   └── (coming soon)
│
└── docs/                       # Notes, strategy decisions
    └── (coming soon)
```

## Pine Scripts

### abnb_signal_bot_v3.pine (active)
- **Type**: `strategy()` — works as both chart indicator AND backtester
- **Signals**: STRONG BUY 🔥, BUY ✅, SELL 🔴, BOUNCE 🔵, VWAP BOUNCE 💧
- **Indicators**: 20 EMA, 50 SMA, 200 SMA, Weekly VWAP, Monthly VWAP, RSI, MACD, ATR
- **Scoring**: 13-point system (trend, momentum, volume, VWAP, quality)
- **Fakeout filters**: ATR buffer, N-candle confirmation, volume, RSI divergence, weekly MTF
- **Alerts**: `alert()` function — create ONE alert in TradingView, all signal types fire through it

### swing_signal_bot_v1.pine (generic template)
- **Type**: `indicator()`
- Generic version — use as starting point for other tickers (NVDA, TSLA, etc.)

## Roadmap

- [x] Phase 1 — Pine Script signal generation (ABNB)
- [ ] Phase 1b — Port to other tickers (NVDA, TSLA, GOOG, SPX)
- [ ] Phase 2 — Python webhook server (FastAPI)
- [ ] Phase 2b — Discord/email alerts + signal log dashboard (Streamlit)
- [ ] Phase 3 — Schwab API integration (auto-execution)

## Backtesting (TradingView Strategy Tester)

1. Paste `abnb_signal_bot_v3.pine` into TradingView Pine Script editor
2. Add to ABNB chart on Daily timeframe
3. Open **Strategy Tester** tab at bottom
4. Adjust `Stop Loss %` and `Take Profit %` in Settings

Target metrics:
- Win rate > 45%
- Profit factor > 1.5
- Max drawdown < 20%
