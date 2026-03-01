#!/usr/bin/env python3
"""
run_optimize.py — Sweep signal config parameters and find optimal weights.

Usage:
    python run_optimize.py --config configs/signals.yaml --ticker GOOG --period 5y \\
        --sweep "regime.weight=2:8:2,rsi_zone.weight=0:4:1"

    python run_optimize.py --ticker GOOG --period 3y \\
        --sweep "regime.weight=1:5:1,adx.weight=0:3:1"

Sweep format: "component.param=min:max:step,..."
  - component.weight  → override component weight
  - exits.sl_pct      → override stop-loss
  - exits.tp_pct      → override take-profit
"""
import argparse
import copy
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backtester.data import fetch_ohlcv
from backtester.engine import BacktestEngine
from signals.config import SignalConfig
from strategies.configurable_strategy import ConfigurableStrategy


def parse_sweep(sweep_str: str) -> dict[str, list[float]]:
    """Parse sweep spec into {param_path: [values]}."""
    result = {}
    for item in sweep_str.split(","):
        item = item.strip()
        if not item:
            continue
        path, range_str = item.split("=")
        parts = range_str.split(":")
        if len(parts) == 3:
            lo, hi, step = float(parts[0]), float(parts[1]), float(parts[2])
            values = []
            v = lo
            while v <= hi + 1e-9:
                values.append(round(v, 4))
                v += step
        elif len(parts) == 1:
            values = [float(parts[0])]
        else:
            raise ValueError(f"Invalid sweep range: {range_str}")
        result[path.strip()] = values
    return result


def apply_param(config: SignalConfig, path: str, value: float) -> None:
    """Apply a parameter override to a config in-place.

    Supported paths:
      component.weight            → override component weight
      component.params.key        → override component parameter
      exits.sl_pct / exits.tp_pct → override exit params
      thresholds.score_*_frac     → override threshold fractions
      entry.signal_name.min_score_frac → override entry signal threshold
    """
    parts = path.split(".")
    if len(parts) == 2 and parts[1] == "weight":
        comp_name = parts[0]
        if comp_name in config.components:
            config.components[comp_name].weight = value
        else:
            raise ValueError(f"Unknown component: {comp_name}")
    elif len(parts) == 3 and parts[1] == "params":
        comp_name, _, key = parts
        if comp_name in config.components:
            # Use int if value is a whole number and param looks like a length
            if value == int(value) and any(k in key for k in ("len", "lookback", "swing")):
                config.components[comp_name].params[key] = int(value)
            else:
                config.components[comp_name].params[key] = value
        else:
            raise ValueError(f"Unknown component: {comp_name}")
    elif parts[0] == "exits":
        attr = parts[1]
        if hasattr(config.exits, attr):
            setattr(config.exits, attr, value)
        else:
            raise ValueError(f"Unknown exit param: {attr}")
    elif parts[0] == "thresholds":
        config.thresholds[parts[1]] = value
    elif parts[0] == "entry" and len(parts) == 3:
        sig_name, attr = parts[1], parts[2]
        if sig_name in config.entry_signals:
            setattr(config.entry_signals[sig_name], attr, value)
        else:
            raise ValueError(f"Unknown entry signal: {sig_name}")
    else:
        raise ValueError(f"Unknown param path: {path}")


def deep_copy_config(config: SignalConfig) -> SignalConfig:
    """Create a deep copy of a SignalConfig."""
    return SignalConfig.from_dict(config.to_dict())


def main():
    parser = argparse.ArgumentParser(description="Sweep signal config parameters")
    parser.add_argument("--config", type=str, default=None,
                        help="Base config YAML (default: legacy v4.3)")
    parser.add_argument("--ticker", type=str, default="GOOG",
                        help="Ticker to backtest (default: GOOG)")
    parser.add_argument("--period", type=str, default="5y",
                        help="Data period (default: 5y)")
    parser.add_argument("--interval", type=str, default="1d",
                        help="Bar interval (default: 1d)")
    parser.add_argument("--sweep", type=str, required=True,
                        help='Sweep spec: "comp.weight=min:max:step,..."')
    parser.add_argument("--sort", type=str, default="sharpe",
                        choices=["sharpe", "cagr", "win_rate", "profit_factor", "max_drawdown"],
                        help="Sort results by (default: sharpe)")
    parser.add_argument("--save-best", type=str, default=None,
                        help="Save best config to this YAML path")
    parser.add_argument("--capital", type=float, default=10_000)
    parser.add_argument("--commission", type=float, default=0.001)
    args = parser.parse_args()

    # Load base config
    if args.config:
        base_config = SignalConfig.from_yaml(args.config)
    else:
        base_config = SignalConfig.from_legacy()

    # Parse sweep
    sweep = parse_sweep(args.sweep)
    param_names = list(sweep.keys())
    param_values = list(sweep.values())
    combos = list(itertools.product(*param_values))

    print(f"Base config: {base_config.name}")
    print(f"Ticker: {args.ticker}  |  Period: {args.period}")
    print(f"Sweep: {len(combos)} combinations")
    for name, vals in sweep.items():
        print(f"  {name}: {vals}")

    # Fetch data once
    ticker = args.ticker.upper()
    df = fetch_ohlcv(ticker, period=args.period, interval=args.interval)
    if df is None or len(df) < 220:
        print(f"[!] {ticker}: insufficient data")
        sys.exit(1)
    print(f"Data: {len(df)} bars\n")

    # Run baseline
    baseline_strategy = ConfigurableStrategy(base_config)
    baseline_engine = BacktestEngine(baseline_strategy, df,
                                     initial_capital=args.capital,
                                     commission_pct=args.commission)
    baseline_results = baseline_engine.run()
    baseline = baseline_results.summary()

    # Run all combos
    results = []
    for i, combo in enumerate(combos):
        cfg = deep_copy_config(base_config)
        label_parts = []
        for name, val in zip(param_names, combo):
            apply_param(cfg, name, val)
            short_name = name.split(".")[-1] if "." in name else name
            label_parts.append(f"{short_name}={val}")
        label = ", ".join(label_parts)

        strategy = ConfigurableStrategy(cfg)
        engine = BacktestEngine(strategy, df,
                                initial_capital=args.capital,
                                commission_pct=args.commission)
        r = engine.run()
        s = r.summary()
        s["label"] = label
        s["_config"] = cfg
        s["_combo"] = dict(zip(param_names, combo))
        results.append(s)

    # Sort
    sort_key = args.sort
    reverse = sort_key != "max_drawdown"
    results.sort(key=lambda x: x.get(sort_key, 0) or 0, reverse=reverse)

    # Print results table
    print(f"\n{'#':>3} {'Config':<40} {'Trades':>6} {'WR':>6} {'PF':>6} "
          f"{'Return':>8} {'CAGR':>7} {'MaxDD':>7} {'Sharpe':>7}")
    print("-" * 95)

    # Baseline row
    print(f"{'BL':>3} {'[BASELINE]':<40} "
          f"{baseline['n_trades']:>6} "
          f"{baseline['win_rate']:>5.1%} "
          f"{baseline['profit_factor']:>6.2f} "
          f"{baseline['total_return']:>+7.1%} "
          f"{baseline['cagr']:>+6.1%} "
          f"{baseline['max_drawdown']:>+6.1%} "
          f"{baseline['sharpe']:>7.2f}")
    print("-" * 95)

    for i, r in enumerate(results):
        label = r["label"][:40]
        print(
            f"{i+1:>3} {label:<40} "
            f"{r['n_trades']:>6} "
            f"{r['win_rate']:>5.1%} "
            f"{r['profit_factor']:>6.2f} "
            f"{r['total_return']:>+7.1%} "
            f"{r['cagr']:>+6.1%} "
            f"{r['max_drawdown']:>+6.1%} "
            f"{r['sharpe']:>7.2f}"
        )

    # Delta table (top 5 vs baseline)
    print(f"\n--- Top {min(5, len(results))} vs Baseline (deltas) ---")
    for i, r in enumerate(results[:5]):
        deltas = []
        for k in ['sharpe', 'cagr', 'win_rate', 'profit_factor', 'max_drawdown']:
            d = (r.get(k, 0) or 0) - (baseline.get(k, 0) or 0)
            sign = "+" if d >= 0 else ""
            if k in ('cagr', 'win_rate', 'max_drawdown'):
                deltas.append(f"{k}:{sign}{d:.1%}")
            else:
                deltas.append(f"{k}:{sign}{d:.2f}")
        print(f"  {i+1}. {r['label']}")
        print(f"     {', '.join(deltas)}")

    # Save best config
    if args.save_best and results:
        best = results[0]
        best_config = best["_config"]
        best_config.name = f"{base_config.name} (optimized)"
        best_config.description = f"Best by {sort_key}: {best['label']}"
        best_config.to_yaml(args.save_best)
        print(f"\nBest config saved to: {args.save_best}")


if __name__ == "__main__":
    main()
