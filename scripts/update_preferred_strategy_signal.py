#!/usr/bin/env python
"""Update Yahoo Finance data and report the current preferred QQQ/TQQQ signal.

Preferred strategy as of 2026-06-04:
- Signal source: QQQ 60-minute bars.
- Exposure: synthetic +3x QQQ (QQQ_3X_CALC) for research evaluation.
- Entry: QQQ hourly SMA-MACD histogram > 0 and QQQ hourly close > QQQ hourly 200-day MA.
  Preferred MACD option: 12/24/9 trading-day windows.
- Exit: QQQ hourly close < QQQ hourly 200-day MA.
- Profit lock: +300% synthetic-3x trade gain -> 75%; +400% -> 50%.
- Dynamic mean-reversion trim: after a trade first reaches +110%, learn the
  maximum QQQ/200MA distance seen up to that point; later trim to 50% if QQQ
  revisits/exceeds that learned distance, and re-add when QQQ touches its 20-day MA.
- Best robustness bear filter: if the QQQ hourly 200-day MA slope is negative,
  delay a new entry unless QQQ is at least 1% above its hourly 200-day MA, the
  QQQ 50MA slope over 30 trading days is positive, and QQQ 20MA > QQQ 50MA.
- Stop: exit if synthetic QQQ_3X_CALC falls 40% from the current trade peak.
- No daily regime gate, max one trade per day.

This updater intentionally uses Yahoo Finance through yfinance, not Alpha
Vantage. It refreshes newest available Yahoo data and merges it into local
history files. It also saves a local Yahoo-reported QQQ P/E snapshot history.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_tqqq_daily_gate_ablation import no_daily_gate_hourly_ma_gate_signal  # noqa: E402
from run_tqqq_entry_signal_comparison import executable_weights  # noqa: E402
from trend_following.bear_whipsaw import (  # noqa: E402
    bear_market_features,
    bear_reentry_filter_raw,
)
from trend_following.config import load_config  # noqa: E402
from trend_following.data_validation import read_price_file  # noqa: E402
from trend_following.risk_overlays import (  # noqa: E402
    apply_cap,
    dynamic_pre100_distance_trim_rebuy_cap,
    qqq_mean_reversion_features,
)
from trend_following.synthetic_leverage import (  # noqa: E402
    synthetic_daily_leveraged_ohlcv,
    synthetic_intraday_leveraged_ohlcv,
)
from trend_following.utils import ensure_directory, resolve_path  # noqa: E402

PROFIT_LOCK_SCHEME: list[tuple[float, float]] = [(3.0, 0.75), (4.0, 0.50)]
PEAK_STOP_DRAWDOWN = 0.40
DYNAMIC_TRIM_ACTIVATION_GAIN = 1.10
DYNAMIC_TRIM_DISTANCE_QUANTILE = 1.0
DYNAMIC_TRIM_TARGET_WEIGHT = 0.50
BEAR_FILTER_DISTANCE_BUFFER = 0.01
BEAR_FILTER_SLOPE_DAYS = 30
BEAR_FILTER_REQUIRE_SHORT_GT_MEDIUM = True
PREFERRED_MACD_FAST_DAYS = 12.0
PREFERRED_MACD_SLOW_DAYS = 24.0
PREFERRED_MACD_SIGNAL_DAYS = 9.0
RETAINED_MACD_OPTIONS = "12/24/9 preferred; retained near-equivalent options: 12/26/9 and 12/26/8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/regime_hourly_qqq.yaml")
    parser.add_argument("--ticker", default="QQQ")
    parser.add_argument("--actual-tqqq-ticker", default="TQQQ")
    parser.add_argument("--target-ticker", default="QQQ_3X_CALC")
    parser.add_argument("--interval", default="60m", choices=["60m", "60min", "30m", "30min", "15m", "15min"])
    parser.add_argument("--intraday-period", default="730d")
    parser.add_argument("--daily-period", default="max")
    parser.add_argument("--intraday-raw-dir", default="data/raw/yfinance_60min")
    parser.add_argument("--actual-tqqq-raw-dir", default=None)
    parser.add_argument("--daily-raw-dir", default="data/raw/yfinance_daily")
    parser.add_argument("--synthetic-daily-dir", default="data/raw/synthetic_yfinance_3x_1d")
    parser.add_argument("--synthetic-intraday-dir", default="data/raw/synthetic_yfinance_3x_60min")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-actual-tqqq", action="store_true")
    parser.add_argument("--skip-pe", action="store_true")
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=None,
        help=argparse.SUPPRESS,  # Backward-compatible no-op from Alpha Vantage updater.
    )
    parser.add_argument("--pe-processed-dir", default="data/processed/valuation")
    parser.add_argument("--pe-tables-dir", default="reports/tables")
    # Yahoo 60m bars normally have seven regular-session bars per full day:
    # 09:30, 10:30, ..., 15:30. This differs from the Alpha Vantage cache used
    # in earlier research, which had six 60-minute bars per full day.
    parser.add_argument("--bars-per-day", type=int, default=7)
    parser.add_argument("--average-type", choices=["sma", "ema"], default="sma")
    parser.add_argument("--macd-unit", choices=["days", "bars"], default="days")
    parser.add_argument("--macd-fast-days", type=float, default=PREFERRED_MACD_FAST_DAYS)
    parser.add_argument("--macd-slow-days", type=float, default=PREFERRED_MACD_SLOW_DAYS)
    parser.add_argument("--macd-signal-days", type=float, default=PREFERRED_MACD_SIGNAL_DAYS)
    parser.add_argument("--entry-confirm-bars", type=int, default=2)
    parser.add_argument("--exit-confirm-bars", type=int, default=3)
    parser.add_argument("--exit-ma-days", type=float, default=200.0)
    parser.add_argument("--output-path", default="reports/tables/preferred_strategy_current_signal.csv")
    parser.add_argument(
        "--history-path",
        default="reports/tables/preferred_strategy_current_signal_history.csv",
    )
    return parser.parse_args()


def _yf_interval(interval: str) -> str:
    return {"60min": "60m", "30min": "30m", "15min": "15m"}.get(interval, interval)


def _write_price_frame(frame: pd.DataFrame, path: Path) -> None:
    ensure_directory(path.parent)
    out = frame.sort_index().reset_index()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out.to_parquet(path, index=False)


def _merge_price_file(path: Path, new_frame: pd.DataFrame) -> pd.DataFrame:
    new_indexed = new_frame.set_index("date") if "date" in new_frame.columns else new_frame.copy()
    if path.exists():
        old = read_price_file(path).sort_index()
        combined = pd.concat([old, new_indexed])
    else:
        combined = new_indexed
    combined.index = pd.to_datetime(combined.index).tz_localize(None)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    _write_price_frame(combined, path)
    return combined


def _normalize_yfinance_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize yfinance OHLCV output while preserving New York bar labels.

    ``normalize_downloaded_frame`` is fine for daily data, but for tz-aware
    intraday yfinance bars it can convert labels to UTC before dropping the
    timezone. For signal monitoring we want the local exchange-time labels
    shown by yfinance, e.g. 09:30, 10:30, ..., 15:30 America/New_York.
    """
    if frame.empty:
        raise ValueError(f"No rows downloaded for {ticker}")
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        columns = out.columns
        if ticker in columns.get_level_values(-1):
            out = out.xs(ticker, axis=1, level=-1, drop_level=True)
        elif ticker in columns.get_level_values(0):
            out = out.xs(ticker, axis=1, level=0, drop_level=True)
        else:
            out.columns = ["_".join(str(part) for part in col if part) for col in columns]

    index = pd.DatetimeIndex(pd.to_datetime(out.index))
    if index.tz is not None:
        dates = index.tz_convert("America/New_York").tz_localize(None)
    else:
        dates = index.tz_localize(None) if getattr(index, "tz", None) is not None else index

    rename = {}
    for column in out.columns:
        normalized = str(column).strip().lower().replace("_", " ")
        if normalized == "open":
            rename[column] = "open"
        elif normalized == "high":
            rename[column] = "high"
        elif normalized == "low":
            rename[column] = "low"
        elif normalized == "close":
            rename[column] = "close"
        elif normalized in {"adj close", "adj_close"}:
            rename[column] = "adj_close"
        elif normalized == "volume":
            rename[column] = "volume"
    out = out.rename(columns=rename)
    out.insert(0, "date", dates)
    if "adj_close" not in out.columns and "close" in out.columns:
        out["adj_close"] = out["close"]
    ordered_columns = [
        column
        for column in ["date", "open", "high", "low", "close", "adj_close", "volume"]
        if column in out.columns
    ]
    return out[ordered_columns].sort_values("date").reset_index(drop=True)


def _download_yfinance_price(
    ticker: str,
    *,
    interval: str,
    period: str,
    path: Path,
) -> pd.DataFrame:
    frame = yf.download(
        ticker,
        period=period,
        interval=_yf_interval(interval),
        auto_adjust=False,
        actions=False,
        prepost=False,
        progress=False,
        threads=False,
        multi_level_index=True,
    )
    normalized = _normalize_yfinance_frame(frame, ticker)
    if "adj_close" not in normalized.columns:
        normalized["adj_close"] = normalized["close"]
    return _merge_price_file(path, normalized)


def _update_yfinance_data(
    args: argparse.Namespace,
    *,
    qqq_intraday_path: Path,
    tqqq_intraday_path: Path,
    qqq_daily_path: Path,
) -> None:
    _download_yfinance_price(
        args.ticker,
        interval=args.interval,
        period=args.intraday_period,
        path=qqq_intraday_path,
    )
    if not args.skip_actual_tqqq:
        _download_yfinance_price(
            args.actual_tqqq_ticker,
            interval=args.interval,
            period=args.intraday_period,
            path=tqqq_intraday_path,
        )
    _download_yfinance_price(
        args.ticker,
        interval="1d",
        period=args.daily_period,
        path=qqq_daily_path,
    )


def _parse_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        result = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _append_yfinance_pe_history(args: argparse.Namespace, *, root: Path) -> dict[str, Any]:
    if args.skip_pe:
        return {}

    processed_dir = resolve_path(root, args.pe_processed_dir)
    tables_dir = resolve_path(root, args.pe_tables_dir)
    ensure_directory(processed_dir)
    ensure_directory(tables_dir)

    snapshot_ts = pd.Timestamp.now(tz="America/New_York")
    snapshot_date = snapshot_ts.date().isoformat()
    usable_from = (pd.Timestamp(snapshot_date) + BDay(1)).date().isoformat()

    info = yf.Ticker(args.ticker).get_info()
    row = {
        "snapshot_datetime_local": snapshot_ts.isoformat(timespec="seconds"),
        "snapshot_date": snapshot_date,
        "usable_from_date_no_lookahead": usable_from,
        "symbol": args.ticker,
        "source": "Yahoo Finance via yfinance Ticker.get_info()",
        "method": "vendor_reported_etf_trailing_pe_snapshot",
        "qqq_pe_yfinance_trailing_pe": _parse_float(info.get("trailingPE")),
        "qqq_forward_pe_yfinance": _parse_float(info.get("forwardPE")),
        "qqq_nav_price_yfinance": _parse_float(info.get("navPrice")),
        "qqq_regular_market_price_yfinance": _parse_float(info.get("regularMarketPrice")),
        "qqq_previous_close_yfinance": _parse_float(info.get("previousClose")),
        "quote_type": info.get("quoteType", ""),
        "long_name": info.get("longName", ""),
        "notes": (
            "Yahoo Finance ETF P/E is a vendor snapshot, not a transparent holdings-level "
            "harmonic calculation. Do not assume it is identical to Alpha Vantage Option-B P/E."
        ),
    }

    snapshot_history_path = processed_dir / "qqq_pe_yfinance_snapshot_history.parquet"
    daily_history_path = processed_dir / "qqq_pe_yfinance_daily_history.parquet"
    if snapshot_history_path.exists():
        snapshot_history = pd.read_parquet(snapshot_history_path)
        snapshot_history = pd.concat([snapshot_history, pd.DataFrame([row])], ignore_index=True)
        snapshot_history = snapshot_history.drop_duplicates(
            subset=["snapshot_datetime_local"], keep="last"
        )
    else:
        snapshot_history = pd.DataFrame([row])
    snapshot_history.to_parquet(snapshot_history_path, index=False)
    snapshot_history.to_csv(tables_dir / "qqq_pe_yfinance_snapshot_history.csv", index=False)

    if daily_history_path.exists():
        daily_history = pd.read_parquet(daily_history_path)
        daily_history = pd.concat([daily_history, pd.DataFrame([row])], ignore_index=True)
    else:
        daily_history = pd.DataFrame([row])
    daily_history = daily_history.sort_values("snapshot_datetime_local").drop_duplicates(
        subset=["snapshot_date"], keep="last"
    )
    daily_history.to_parquet(daily_history_path, index=False)
    daily_history.to_csv(tables_dir / "qqq_pe_yfinance_daily_history.csv", index=False)
    pd.DataFrame([row]).to_csv(tables_dir / "qqq_pe_yfinance_latest_summary.csv", index=False)
    row["pe_update_status"] = "downloaded_yfinance_snapshot"
    return row


def _latest_yfinance_pe_from_history(history_path: Path) -> dict[str, Any]:
    if not history_path.exists():
        return {}
    history = pd.read_parquet(history_path)
    if history.empty:
        return {}
    latest = history.sort_values("snapshot_datetime_local").iloc[-1].to_dict()
    latest["pe_update_status"] = "latest_yfinance_history_only"
    return latest


def _rebuild_synthetic(
    *,
    daily_path: Path,
    intraday_path: Path,
    synthetic_daily_path: Path,
    synthetic_intraday_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_underlying = read_price_file(daily_path).sort_index()
    intraday_underlying = read_price_file(intraday_path).sort_index()
    daily_synth = synthetic_daily_leveraged_ohlcv(
        daily_underlying,
        leverage=3.0,
        initial_price=100.0,
    )
    intraday_synth = synthetic_intraday_leveraged_ohlcv(
        intraday_underlying=intraday_underlying,
        daily_underlying=daily_underlying,
        daily_synthetic=daily_synth,
        leverage=3.0,
    )
    _write_price_frame(daily_synth, synthetic_daily_path)
    _write_price_frame(intraday_synth, synthetic_intraday_path)
    return daily_synth, intraday_synth


def _raw_with_peak_drawdown_stop(
    base_raw: pd.Series,
    traded_price: pd.Series,
    *,
    stop_drawdown: float,
) -> tuple[pd.Series, pd.DataFrame]:
    """Force raw signal to cash if trade-level peak drawdown breaches threshold.

    The stop is observed on a completed bar and then passed through
    ``executable_weights`` below, preserving the no-lookahead timing convention.
    """
    base = base_raw.fillna(0.0).astype(float)
    price = traded_price.reindex(base.index).astype(float)
    threshold = -abs(float(stop_drawdown))
    in_trade = False
    stopped_until_base_exit = False
    peak = np.nan
    values: list[float] = []
    peaks: list[float] = []
    drawdowns: list[float] = []
    triggers: list[bool] = []

    for base_signal, current_price in zip(base, price, strict=True):
        current_price = float(current_price) if np.isfinite(current_price) else np.nan
        trigger = False
        if base_signal <= 0.0 or not np.isfinite(current_price):
            in_trade = False
            stopped_until_base_exit = False
            peak = np.nan
            value = 0.0
            drawdown = np.nan
        else:
            if not in_trade:
                in_trade = True
                stopped_until_base_exit = False
                peak = current_price
            else:
                peak = max(float(peak), current_price)
            drawdown = current_price / peak - 1.0 if peak > 0 else np.nan
            if stopped_until_base_exit:
                value = 0.0
            elif np.isfinite(drawdown) and drawdown <= threshold:
                trigger = True
                stopped_until_base_exit = True
                value = 0.0
            else:
                value = 1.0
        values.append(value)
        peaks.append(peak)
        drawdowns.append(drawdown)
        triggers.append(trigger)

    stopped = pd.Series(values, index=base.index, name=base_raw.name, dtype=float)
    diagnostics = pd.DataFrame(
        {
            "base_raw": base,
            "stopped_raw": stopped,
            "trade_peak_price": peaks,
            "trade_peak_drawdown": drawdowns,
            "stop_trigger": triggers,
        },
        index=base.index,
    )
    return stopped, diagnostics


def _trade_profit_lock_tiers(
    base_raw: pd.Series,
    price: pd.Series,
    *,
    thresholds_to_weights: list[tuple[float, float]],
) -> pd.Series:
    """Reduce target size within a trade after unrealized-gain thresholds hit."""
    thresholds_to_weights = sorted(thresholds_to_weights)
    in_trade = False
    entry_price = np.nan
    current_weight = 0.0
    values: list[float] = []

    for signal, current_price in zip(base_raw.fillna(0.0), price.reindex(base_raw.index), strict=True):
        if signal <= 0.0 or not np.isfinite(current_price):
            in_trade = False
            entry_price = np.nan
            current_weight = 0.0
            values.append(0.0)
            continue

        if not in_trade:
            in_trade = True
            entry_price = float(current_price)
            current_weight = 1.0

        gain = float(current_price) / entry_price - 1.0 if entry_price > 0 else 0.0
        for threshold, weight in thresholds_to_weights:
            if gain >= threshold:
                current_weight = min(current_weight, weight)
        values.append(current_weight)

    return pd.Series(values, index=base_raw.index, name=base_raw.name, dtype=float)


def _last_open_transition_time(weights: pd.Series) -> pd.Timestamp | pd.NaT:
    transitions = weights[(weights.shift(1).fillna(0.0).le(0.0)) & weights.gt(0.0)]
    if transitions.empty:
        return pd.NaT
    return pd.Timestamp(transitions.index[-1])


def _last_close_transition_time(weights: pd.Series) -> pd.Timestamp | pd.NaT:
    transitions = weights[(weights.shift(1).fillna(0.0).gt(0.0)) & weights.le(0.0)]
    if transitions.empty:
        return pd.NaT
    return pd.Timestamp(transitions.index[-1])


def _last_reduce_transition_time(weights: pd.Series) -> pd.Timestamp | pd.NaT:
    previous = weights.shift(1).fillna(0.0)
    transitions = weights[weights.gt(0.0) & previous.gt(0.0) & weights.lt(previous)]
    if transitions.empty:
        return pd.NaT
    return pd.Timestamp(transitions.index[-1])


def _float_or_nan(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _format_position(value: float) -> str:
    return f"long synthetic TQQQ exposure at {value:.0%} target weight" if value > 0.0 else "cash / out of market"


def _full_days_bar_count(index: pd.DatetimeIndex) -> int:
    if index.empty:
        return 0
    counts = pd.Series(1, index=index).groupby(index.normalize()).sum()
    return int(counts.mode().iloc[0]) if not counts.empty else 0


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = config.root

    intraday_dir = resolve_path(root, args.intraday_raw_dir)
    actual_tqqq_dir = resolve_path(root, args.actual_tqqq_raw_dir or args.intraday_raw_dir)
    daily_dir = resolve_path(root, args.daily_raw_dir)
    synthetic_daily_dir = resolve_path(root, args.synthetic_daily_dir)
    synthetic_intraday_dir = resolve_path(root, args.synthetic_intraday_dir)
    qqq_intraday_path = intraday_dir / f"{args.ticker}.parquet"
    tqqq_intraday_path = actual_tqqq_dir / f"{args.actual_tqqq_ticker}.parquet"
    qqq_daily_path = daily_dir / f"{args.ticker}.parquet"
    synthetic_daily_path = synthetic_daily_dir / f"{args.target_ticker}.parquet"
    synthetic_intraday_path = synthetic_intraday_dir / f"{args.target_ticker}.parquet"

    if not args.skip_download:
        _update_yfinance_data(
            args,
            qqq_intraday_path=qqq_intraday_path,
            tqqq_intraday_path=tqqq_intraday_path,
            qqq_daily_path=qqq_daily_path,
        )
        pe_info = _append_yfinance_pe_history(args, root=root)
    else:
        pe_info = _latest_yfinance_pe_from_history(
            resolve_path(root, args.pe_processed_dir) / "qqq_pe_yfinance_snapshot_history.parquet"
        )

    _, intraday_synth = _rebuild_synthetic(
        daily_path=qqq_daily_path,
        intraday_path=qqq_intraday_path,
        synthetic_daily_path=synthetic_daily_path,
        synthetic_intraday_path=synthetic_intraday_path,
    )

    qqq_frame = read_price_file(qqq_intraday_path).sort_index()
    qqq_close = qqq_frame["adj_close"].astype(float).rename(args.ticker)
    raw, diag = no_daily_gate_hourly_ma_gate_signal(
        entry_price=qqq_close,
        exit_price=qqq_close,
        output_index=qqq_close.index,
        bars_per_day=args.bars_per_day,
        average_type=args.average_type,
        macd_unit=args.macd_unit,
        entry_confirm_bars=args.entry_confirm_bars,
        exit_confirm_bars=args.exit_confirm_bars,
        exit_ma_days=args.exit_ma_days,
        macd_fast_days=args.macd_fast_days,
        macd_slow_days=args.macd_slow_days,
        macd_signal_days=args.macd_signal_days,
    )
    synthetic_close = intraday_synth["adj_close"].astype(float)
    stopped_raw, stop_diag = _raw_with_peak_drawdown_stop(
        raw.rename(args.target_ticker),
        synthetic_close,
        stop_drawdown=PEAK_STOP_DRAWDOWN,
    )
    raw_profit_locked_weight = _trade_profit_lock_tiers(
        stopped_raw.rename(args.target_ticker),
        synthetic_close,
        thresholds_to_weights=PROFIT_LOCK_SCHEME,
    ).rename(args.target_ticker)
    qqq_features = qqq_mean_reversion_features(qqq_close, bars_per_day=args.bars_per_day)
    dynamic_trim = dynamic_pre100_distance_trim_rebuy_cap(
        raw_profit_locked_weight.gt(0.0).astype(float),
        synthetic_close,
        qqq_features,
        activation_gain=DYNAMIC_TRIM_ACTIVATION_GAIN,
        threshold_quantile=DYNAMIC_TRIM_DISTANCE_QUANTILE,
        trim_weight=DYNAMIC_TRIM_TARGET_WEIGHT,
        reentry_rule="ma20",
    )
    raw_dynamic_trimmed_weight = apply_cap(raw_profit_locked_weight, dynamic_trim.weights).rename(
        args.target_ticker
    )
    bear_features = bear_market_features(
        qqq_close,
        bars_per_day=args.bars_per_day,
        slope_days=(BEAR_FILTER_SLOPE_DAYS,),
    )
    bear_filter = bear_reentry_filter_raw(
        raw_dynamic_trimmed_weight.gt(0.0).astype(float),
        bear_features,
        distance_buffer=BEAR_FILTER_DISTANCE_BUFFER,
        slope_days=BEAR_FILTER_SLOPE_DAYS,
        require_short_gt_medium=BEAR_FILTER_REQUIRE_SHORT_GT_MEDIUM,
    )
    raw_bear_filtered_weight = apply_cap(
        raw_dynamic_trimmed_weight, bear_filter.weights
    ).rename(args.target_ticker)
    raw_weights = raw_bear_filtered_weight.to_frame(args.target_ticker)
    weights = executable_weights(raw_weights, config=config)[args.target_ticker]

    latest = pd.Timestamp(qqq_close.index[-1])
    prev = pd.Timestamp(qqq_close.index[-2]) if len(qqq_close) > 1 else pd.NaT
    latest_diag = diag.loc[latest]
    latest_weight = _float_or_nan(weights.loc[latest])
    prev_weight = _float_or_nan(weights.loc[prev]) if pd.notna(prev) else float("nan")
    latest_base_raw = _float_or_nan(raw.loc[latest])
    previous_base_raw = _float_or_nan(raw.loc[prev]) if pd.notna(prev) else float("nan")
    latest_stopped_raw = _float_or_nan(stopped_raw.loc[latest])
    previous_stopped_raw = _float_or_nan(stopped_raw.loc[prev]) if pd.notna(prev) else float("nan")
    latest_profit_locked_weight = _float_or_nan(raw_profit_locked_weight.loc[latest])
    previous_profit_locked_weight = (
        _float_or_nan(raw_profit_locked_weight.loc[prev]) if pd.notna(prev) else float("nan")
    )
    latest_raw_weight = _float_or_nan(raw_dynamic_trimmed_weight.loc[latest])
    previous_raw_weight = (
        _float_or_nan(raw_dynamic_trimmed_weight.loc[prev]) if pd.notna(prev) else float("nan")
    )
    latest_bear_filtered_weight = _float_or_nan(raw_bear_filtered_weight.loc[latest])
    previous_bear_filtered_weight = (
        _float_or_nan(raw_bear_filtered_weight.loc[prev]) if pd.notna(prev) else float("nan")
    )
    latest_dynamic_trim = dynamic_trim.diagnostics.loc[latest]
    latest_bear_filter = bear_filter.diagnostics.loc[latest]
    latest_bear_features = bear_features.loc[latest]
    qqq_latest = _float_or_nan(qqq_close.loc[latest])
    qqq_ma = _float_or_nan(latest_diag.get("exit_ma"))
    distance_to_exit = qqq_latest / qqq_ma - 1.0 if qqq_ma and np.isfinite(qqq_ma) else float("nan")

    tqqq_frame = read_price_file(tqqq_intraday_path).sort_index() if tqqq_intraday_path.exists() else pd.DataFrame()
    tqqq_close = (
        tqqq_frame["adj_close"].astype(float)
        if not tqqq_frame.empty and "adj_close" in tqqq_frame.columns
        else pd.Series(dtype=float)
    )
    synthetic_latest = (
        _float_or_nan(synthetic_close.reindex([latest]).iloc[0])
        if latest in synthetic_close.index
        else float("nan")
    )
    last_buy_time = _last_open_transition_time(weights)
    last_sell_time = _last_close_transition_time(weights)
    last_reduce_time = _last_reduce_transition_time(weights)
    synthetic_at_last_buy = (
        _float_or_nan(synthetic_close.reindex([last_buy_time]).iloc[0])
        if pd.notna(last_buy_time) and last_buy_time in synthetic_close.index
        else float("nan")
    )
    qqq_at_last_buy = (
        _float_or_nan(qqq_close.reindex([last_buy_time]).iloc[0])
        if pd.notna(last_buy_time) and last_buy_time in qqq_close.index
        else float("nan")
    )
    trade_unrealized_synth = (
        synthetic_latest / synthetic_at_last_buy - 1.0
        if latest_weight >= 0.5 and synthetic_at_last_buy and np.isfinite(synthetic_at_last_buy)
        else float("nan")
    )
    actual_tqqq_latest_time = pd.Timestamp(tqqq_close.index[-1]) if not tqqq_close.empty else pd.NaT
    actual_tqqq_latest_close = _float_or_nan(tqqq_close.iloc[-1]) if not tqqq_close.empty else float("nan")
    actual_tqqq_at_last_buy = (
        _float_or_nan(tqqq_close.reindex([last_buy_time]).iloc[0])
        if pd.notna(last_buy_time) and last_buy_time in tqqq_close.index
        else float("nan")
    )
    trade_unrealized_actual_tqqq = (
        actual_tqqq_latest_close / actual_tqqq_at_last_buy - 1.0
        if (
            latest_weight >= 0.5
            and actual_tqqq_at_last_buy
            and np.isfinite(actual_tqqq_at_last_buy)
            and np.isfinite(actual_tqqq_latest_close)
        )
        else float("nan")
    )
    qqq_pe_latest = _float_or_nan(pe_info.get("qqq_pe_yfinance_trailing_pe"))

    action = "hold long / buy if currently out" if latest_weight > 0.0 else "hold cash / sell if currently long"
    if np.isfinite(prev_weight) and latest_weight != prev_weight:
        if prev_weight <= 0.0 and latest_weight > 0.0:
            action = f"BUY signal became executable; target weight {latest_weight:.0%}"
        elif prev_weight > 0.0 and latest_weight <= 0.0:
            action = "SELL signal became executable"
        elif latest_weight < prev_weight:
            action = f"REDUCE exposure signal became executable; target weight {latest_weight:.0%}"
        else:
            action = f"INCREASE exposure signal became executable; target weight {latest_weight:.0%}"

    bar_count_mode = _full_days_bar_count(qqq_close.index)
    inconsistency_notes = (
        "Yahoo Finance/yfinance update mode. Yahoo 60m bars usually have 7 bars/day "
        "(09:30...15:30), while prior Alpha Vantage research used 6 bars/day "
        "(10:00...15:00). This script uses bars_per_day="
        f"{args.bars_per_day}; signal values may differ from Alpha Vantage-based backtests. "
        "Yahoo intraday history is limited (typically about 730 days), so it is suitable for "
        "current monitoring but not for reproducing full long-history backtests. Yahoo ETF P/E "
        "is a vendor-reported snapshot and may differ from the prior Alpha Vantage holdings-based PE."
    )

    daily_frame = read_price_file(qqq_daily_path)
    row = {
        "generated_at_local": pd.Timestamp.now().isoformat(timespec="seconds"),
        "data_source": "yfinance",
        "asof_intraday_bar": latest,
        "previous_intraday_bar": prev,
        "strategy_label": "preferred_qqq_hourly_200ma_macd_slow24_profit_lock_300_400_stop40_q110_best_bear_filter",
        "preferred_macd_option": f"{args.macd_fast_days:g}/{args.macd_slow_days:g}/{args.macd_signal_days:g}",
        "retained_macd_options": RETAINED_MACD_OPTIONS,
        "signal_source": args.ticker,
        "target_exposure": args.target_ticker,
        "base_raw_position_latest_bar": latest_base_raw,
        "base_raw_position_previous_bar": previous_base_raw,
        "stopped_raw_position_latest_bar": latest_stopped_raw,
        "stopped_raw_position_previous_bar": previous_stopped_raw,
        "profit_locked_position_latest_bar": latest_profit_locked_weight,
        "profit_locked_position_previous_bar": previous_profit_locked_weight,
        "dynamic_trimmed_position_latest_bar": latest_raw_weight,
        "dynamic_trimmed_position_previous_bar": previous_raw_weight,
        "raw_desired_position_latest_bar": latest_bear_filtered_weight,
        "raw_desired_position_previous_bar": previous_bear_filtered_weight,
        "executable_position_latest_bar": latest_weight,
        "executable_position_previous_bar": prev_weight,
        "position_interpretation": _format_position(latest_weight),
        "action_if_following_strategy": action,
        "pending_raw_vs_executable_difference": bool(latest_bear_filtered_weight != latest_weight),
        "qqq_latest_close": qqq_latest,
        "qqq_200d_hourly_ma": qqq_ma,
        "distance_to_exit_trigger_pct": distance_to_exit,
        "sell_trigger_rule": (
            f"exit after {args.exit_confirm_bars} confirmed hourly closes below QQQ 200-day hourly MA, "
            f"or after synthetic QQQ_3X_CALC falls {PEAK_STOP_DRAWDOWN:.0%} from its current trade peak"
        ),
        "approx_qqq_sell_trigger_close": qqq_ma,
        "profit_lock_rule": "+300% synthetic-3x trade gain -> 75%; +400% -> 50%",
        "profit_lock_first_threshold": PROFIT_LOCK_SCHEME[0][0],
        "profit_lock_first_target_weight": PROFIT_LOCK_SCHEME[0][1],
        "profit_lock_second_threshold": PROFIT_LOCK_SCHEME[1][0],
        "profit_lock_second_target_weight": PROFIT_LOCK_SCHEME[1][1],
        "dynamic_trim_rule": (
            "after +110% synthetic-3x trade gain, learn q100=max QQQ distance above hourly "
            "200-day MA through the first +110% bar; later cap to 50% if QQQ revisits/exceeds "
            "that learned distance; re-add when QQQ touches its 20-day MA"
        ),
        "dynamic_trim_activation_gain": DYNAMIC_TRIM_ACTIVATION_GAIN,
        "dynamic_trim_distance_quantile": DYNAMIC_TRIM_DISTANCE_QUANTILE,
        "dynamic_trim_target_weight": DYNAMIC_TRIM_TARGET_WEIGHT,
        "dynamic_trim_cap_raw_latest": _float_or_nan(latest_dynamic_trim.get("overlay_cap")),
        "dynamic_trim_trigger_raw_latest": bool(latest_dynamic_trim.get("overlay_trigger")),
        "dynamic_trim_reentry_raw_latest": bool(latest_dynamic_trim.get("overlay_reentry")),
        "dynamic_trim_trade_gain_raw_latest": _float_or_nan(latest_dynamic_trim.get("trade_gain")),
        "dynamic_trim_learned_threshold_latest": _float_or_nan(
            latest_dynamic_trim.get("dynamic_distance_threshold")
        ),
        "dynamic_trim_threshold_learned_latest": bool(latest_dynamic_trim.get("threshold_learned")),
        "bear_filter_rule": (
            "if QQQ hourly 200-day MA slope is negative, delay new entries until QQQ is at least "
            "1% above its hourly 200-day MA, QQQ 50MA slope over 30 trading days is positive, "
            "and QQQ 20MA > QQQ 50MA"
        ),
        "bear_filter_distance_buffer": BEAR_FILTER_DISTANCE_BUFFER,
        "bear_filter_slope_days": BEAR_FILTER_SLOPE_DAYS,
        "bear_filter_require_20ma_gt_50ma": BEAR_FILTER_REQUIRE_SHORT_GT_MEDIUM,
        "bear_filter_blocked_entry_raw_latest": bool(latest_bear_filter.get("blocked_entry")),
        "bear_filter_release_confirmation_raw_latest": bool(
            latest_bear_filter.get("release_confirmation")
        ),
        "bear_filter_distance_to_200ma_latest": _float_or_nan(
            latest_bear_features.get("distance_to_long_ma")
        ),
        "bear_filter_200ma_slope_30d_latest": _float_or_nan(
            latest_bear_features.get(f"long_ma_slope_{BEAR_FILTER_SLOPE_DAYS}d")
        ),
        "bear_filter_50ma_slope_30d_latest": _float_or_nan(
            latest_bear_features.get(f"medium_ma_slope_{BEAR_FILTER_SLOPE_DAYS}d")
        ),
        "bear_filter_20ma_gt_50ma_latest": bool(latest_bear_features.get("short_gt_medium")),
        "peak_stop_drawdown_threshold": -PEAK_STOP_DRAWDOWN,
        "trade_peak_price_raw_latest": _float_or_nan(stop_diag.loc[latest].get("trade_peak_price")),
        "trade_peak_drawdown_raw_latest": _float_or_nan(stop_diag.loc[latest].get("trade_peak_drawdown")),
        "peak_stop_trigger_raw_latest": bool(stop_diag.loc[latest].get("stop_trigger")),
        "macd_fast_days": args.macd_fast_days,
        "macd_slow_days": args.macd_slow_days,
        "macd_signal_days": args.macd_signal_days,
        "macd_hist_latest": _float_or_nan(latest_diag.get("macd_hist")),
        "entry_flag_latest": bool(_float_or_nan(latest_diag.get("entry_flag")) >= 0.5),
        "above_200ma_latest": bool(_float_or_nan(latest_diag.get("above_exit_ma")) >= 0.5),
        "exit_flag_latest": bool(_float_or_nan(latest_diag.get("price_exit")) >= 0.5),
        "synthetic_3x_latest_close": synthetic_latest,
        "actual_tqqq_latest_bar": actual_tqqq_latest_time,
        "actual_tqqq_latest_close": actual_tqqq_latest_close,
        "last_buy_time": last_buy_time,
        "last_sell_time": last_sell_time,
        "last_reduce_time": last_reduce_time,
        "qqq_at_last_buy": qqq_at_last_buy,
        "synthetic_3x_at_last_buy": synthetic_at_last_buy,
        "actual_tqqq_at_last_buy": actual_tqqq_at_last_buy,
        "current_trade_unrealized_pct_synthetic": trade_unrealized_synth,
        "current_trade_unrealized_pct_actual_tqqq": trade_unrealized_actual_tqqq,
        "intraday_rows": int(len(qqq_close)),
        "intraday_start": qqq_close.index.min(),
        "intraday_end": qqq_close.index.max(),
        "yfinance_intraday_period_requested": args.intraday_period,
        "observed_full_day_bar_count_mode": bar_count_mode,
        "bars_per_day_used_for_strategy": args.bars_per_day,
        "actual_tqqq_rows": int(len(tqqq_close)),
        "actual_tqqq_start": tqqq_close.index.min() if not tqqq_close.empty else pd.NaT,
        "actual_tqqq_end": tqqq_close.index.max() if not tqqq_close.empty else pd.NaT,
        "daily_rows": int(len(daily_frame)),
        "daily_end": daily_frame.index.max(),
        "qqq_pe_snapshot_datetime_local": pe_info.get("snapshot_datetime_local", ""),
        "qqq_pe_snapshot_date": pe_info.get("snapshot_date", ""),
        "qqq_pe_usable_from_no_lookahead": pe_info.get("usable_from_date_no_lookahead", ""),
        "qqq_pe_yfinance_trailing_pe": qqq_pe_latest,
        "qqq_forward_pe_yfinance": _float_or_nan(pe_info.get("qqq_forward_pe_yfinance")),
        "qqq_nav_price_yfinance": _float_or_nan(pe_info.get("qqq_nav_price_yfinance")),
        "qqq_regular_market_price_yfinance": _float_or_nan(
            pe_info.get("qqq_regular_market_price_yfinance")
        ),
        "qqq_pe_update_status": pe_info.get("pe_update_status", ""),
        "inconsistency_notes": inconsistency_notes,
        "note": "Research signal only; not financial advice. Position uses no-lookahead executable shift.",
    }

    output_path = resolve_path(root, args.output_path)
    history_path = resolve_path(root, args.history_path)
    ensure_directory(output_path.parent)
    pd.DataFrame([row]).to_csv(output_path, index=False)

    if history_path.exists():
        history = pd.read_csv(history_path)
        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
        history = history.drop_duplicates(subset=["data_source", "asof_intraday_bar"], keep="last")
    else:
        history = pd.DataFrame([row])
    history.to_csv(history_path, index=False)

    print(pd.DataFrame([row]).T.to_string(header=False))
    print(f"\nSaved current signal to {output_path}")
    print(f"Saved signal history to {history_path}")


if __name__ == "__main__":
    main()
