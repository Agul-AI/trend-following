"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from trend_following.utils import as_list, ensure_directory, resolve_path


@dataclass(frozen=True)
class DataConfig:
    source: str = "yfinance"
    tickers: list[str] = field(default_factory=list)
    start_date: str = "1990-01-01"
    end_date: str | None = None
    interval: str = "1d"
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    alignment: str = "inner"
    suspicious_gap_days: int = 7


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 1.0
    transaction_cost_bps: float = 1.0
    slippage_bps: float = 1.0
    annualization: int = 252
    return_convention: str = "close_to_close"
    execution_delay_days: int = 1
    max_trades_per_day: int | None = 1
    portfolio_mode: str = "equal_sleeves"
    train_end_date: str = "2018-12-31"


@dataclass(frozen=True)
class VolatilityTargetConfig:
    enabled: bool = False
    target_vol: float = 0.10
    lookback: int = 63
    max_leverage: float = 1.0


@dataclass(frozen=True)
class StrategiesConfig:
    sma_trend: dict[str, Any] = field(default_factory=lambda: {"window": 200})
    sma_crossover: dict[str, Any] = field(
        default_factory=lambda: {"short_window": 50, "long_window": 200}
    )
    tsmom: dict[str, Any] = field(default_factory=lambda: {"lookback": 252})
    donchian_breakout: dict[str, Any] = field(
        default_factory=lambda: {"entry_lookback": 252, "exit_lookback": 126}
    )
    regression_slope: dict[str, Any] = field(
        default_factory=lambda: {"window": 126, "min_r_squared": 0.0}
    )
    kalman_trend: dict[str, Any] = field(
        default_factory=lambda: {
            "process_level_var": 1e-5,
            "process_trend_var": 1e-7,
            "observation_var": 1e-3,
            "min_periods": 20,
        }
    )
    cross_sectional_momentum: dict[str, Any] = field(
        default_factory=lambda: {
            "lookback": 126,
            "top_n": 3,
            "require_positive": True,
            "portfolio_mode": "active_equal",
        }
    )
    regime_switch: dict[str, Any] = field(
        default_factory=lambda: {
            "target_ticker": "QQQ",
            "regime_ticker": "QQQ",
            "sma_window": 200,
            "sma_slope_window": 20,
            "variance_window": 63,
            "variance_horizon": 5,
            "use_variance_ratio_for_trend": False,
            "trend_variance_ratio_threshold": 1.05,
            "mean_reversion_variance_ratio_threshold": 0.98,
            "volatility_window": 20,
            "volatility_percentile_window": 252,
            "volatility_percentile_threshold": 0.80,
            "zscore_window": 20,
            "entry_zscore": -1.5,
            "exit_zscore": 0.0,
            "trend_short_window": 50,
            "trend_long_window": 200,
        }
    )
    volatility_targeting: VolatilityTargetConfig = field(default_factory=VolatilityTargetConfig)


@dataclass(frozen=True)
class ExperimentsConfig:
    sma_windows: list[int] = field(default_factory=lambda: [50, 100, 150, 200, 250])
    tsmom_lookbacks: list[int] = field(default_factory=lambda: [63, 126, 189, 252])
    crossover_short_windows: list[int] = field(default_factory=lambda: [20, 50, 100])
    crossover_long_windows: list[int] = field(default_factory=lambda: [100, 150, 200, 250])
    breakout_entry_lookbacks: list[int] = field(default_factory=lambda: [63, 126, 252])
    breakout_exit_lookbacks: list[int] = field(default_factory=lambda: [21, 63, 126])
    regression_windows: list[int] = field(default_factory=lambda: [63, 126, 189, 252])
    cross_sectional_lookbacks: list[int] = field(default_factory=lambda: [63, 126, 252])
    cross_sectional_top_ns: list[int] = field(default_factory=lambda: [2, 3, 5])


@dataclass(frozen=True)
class ReportsConfig:
    figures_dir: Path = Path("reports/figures")
    tables_dir: Path = Path("reports/tables")
    summary_path: Path = Path("reports/summary.md")


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    root: Path
    data: DataConfig
    backtest: BacktestConfig
    strategies: StrategiesConfig
    experiments: ExperimentsConfig
    reports: ReportsConfig


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return loaded


def _parse_date_string(value: str | None, field_name: str) -> str | None:
    if value in (None, "", "null"):
        return None
    try:
        date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be ISO format YYYY-MM-DD, got {value!r}") from exc
    return str(value)


def load_config(config_path: str | Path) -> ProjectConfig:
    """Load a YAML config and return a typed project config.

    Relative paths in the YAML file are resolved relative to the repository root,
    inferred as the parent of the config directory.
    """
    load_dotenv()
    path = Path(config_path).expanduser().resolve()
    raw = _read_yaml(path)
    root = path.parent.parent.resolve()

    project = raw.get("project", {})
    data_raw = raw.get("data", {})
    backtest_raw = raw.get("backtest", {})
    strategies_raw = raw.get("strategies", {})
    experiments_raw = raw.get("experiments", {})
    reports_raw = raw.get("reports", {})

    tickers = as_list(data_raw.get("tickers", []))
    if not tickers:
        raise ValueError("Config must include at least one ticker under data.tickers")

    data = DataConfig(
        source=str(data_raw.get("source", "yfinance")).lower(),
        tickers=tickers,
        start_date=_parse_date_string(data_raw.get("start_date", "1990-01-01"), "start_date")
        or "1990-01-01",
        end_date=_parse_date_string(data_raw.get("end_date"), "end_date"),
        interval=str(data_raw.get("interval", "1d")).lower(),
        raw_dir=resolve_path(root, data_raw.get("raw_dir", "data/raw")),
        processed_dir=resolve_path(root, data_raw.get("processed_dir", "data/processed")),
        alignment=str(data_raw.get("alignment", "inner")).lower(),
        suspicious_gap_days=int(data_raw.get("suspicious_gap_days", 7)),
    )
    if data.alignment not in {"inner", "outer"}:
        raise ValueError("data.alignment must be either 'inner' or 'outer'")
    if data.source not in {"yfinance", "alpha_vantage", "stooq"}:
        raise ValueError("data.source must be one of: yfinance, alpha_vantage, stooq")
    supported_intervals = {
        "1m",
        "1min",
        "2m",
        "5m",
        "5min",
        "15m",
        "15min",
        "30m",
        "30min",
        "60min",
        "60m",
        "90m",
        "1h",
        "1d",
        "d",
        "daily",
        "5d",
        "1wk",
        "1mo",
        "3mo",
    }
    if data.interval not in supported_intervals:
        raise ValueError(f"Unsupported data interval: {data.interval}")

    backtest = BacktestConfig(
        initial_capital=float(backtest_raw.get("initial_capital", 1.0)),
        transaction_cost_bps=float(backtest_raw.get("transaction_cost_bps", 1.0)),
        slippage_bps=float(backtest_raw.get("slippage_bps", 1.0)),
        annualization=int(backtest_raw.get("annualization", 252)),
        return_convention=str(backtest_raw.get("return_convention", "close_to_close")),
        execution_delay_days=int(backtest_raw.get("execution_delay_days", 1)),
        max_trades_per_day=(
            None
            if backtest_raw.get("max_trades_per_day", 1) in {None, "null"}
            else int(backtest_raw.get("max_trades_per_day", 1))
        ),
        portfolio_mode=str(backtest_raw.get("portfolio_mode", "equal_sleeves")),
        train_end_date=str(backtest_raw.get("train_end_date", "2018-12-31")),
    )
    if backtest.transaction_cost_bps < 0 or backtest.slippage_bps < 0:
        raise ValueError("Transaction cost and slippage assumptions must be non-negative")
    if backtest.execution_delay_days < 0:
        raise ValueError("execution_delay_days must be non-negative")
    if backtest.max_trades_per_day is not None and backtest.max_trades_per_day < 0:
        raise ValueError("max_trades_per_day must be non-negative or null")

    vol_raw = strategies_raw.get("volatility_targeting", {})
    volatility_targeting = VolatilityTargetConfig(
        enabled=bool(vol_raw.get("enabled", False)),
        target_vol=float(vol_raw.get("target_vol", 0.10)),
        lookback=int(vol_raw.get("lookback", 63)),
        max_leverage=float(vol_raw.get("max_leverage", 1.0)),
    )

    strategies = StrategiesConfig(
        sma_trend=dict(strategies_raw.get("sma_trend", {"window": 200})),
        sma_crossover=dict(
            strategies_raw.get("sma_crossover", {"short_window": 50, "long_window": 200})
        ),
        tsmom=dict(strategies_raw.get("tsmom", {"lookback": 252})),
        donchian_breakout=dict(
            strategies_raw.get("donchian_breakout", {"entry_lookback": 252, "exit_lookback": 126})
        ),
        regression_slope=dict(
            strategies_raw.get("regression_slope", {"window": 126, "min_r_squared": 0.0})
        ),
        kalman_trend=dict(
            strategies_raw.get(
                "kalman_trend",
                {
                    "process_level_var": 1e-5,
                    "process_trend_var": 1e-7,
                    "observation_var": 1e-3,
                    "min_periods": 20,
                },
            )
        ),
        cross_sectional_momentum=dict(
            strategies_raw.get(
                "cross_sectional_momentum",
                {
                    "lookback": 126,
                    "top_n": 3,
                    "require_positive": True,
                    "portfolio_mode": "active_equal",
                },
            )
        ),
        regime_switch=dict(
            strategies_raw.get(
                "regime_switch",
                {
                    "target_ticker": "QQQ",
                    "regime_ticker": "QQQ",
                    "sma_window": 200,
                    "sma_slope_window": 20,
                    "variance_window": 63,
                    "variance_horizon": 5,
                    "use_variance_ratio_for_trend": False,
                    "trend_variance_ratio_threshold": 1.05,
                    "mean_reversion_variance_ratio_threshold": 0.98,
                    "volatility_window": 20,
                    "volatility_percentile_window": 252,
                    "volatility_percentile_threshold": 0.80,
                    "zscore_window": 20,
                    "entry_zscore": -1.5,
                    "exit_zscore": 0.0,
                    "trend_short_window": 50,
                    "trend_long_window": 200,
                },
            )
        ),
        volatility_targeting=volatility_targeting,
    )

    experiments = ExperimentsConfig(
        sma_windows=[int(x) for x in experiments_raw.get("sma_windows", [50, 100, 150, 200, 250])],
        tsmom_lookbacks=[
            int(x) for x in experiments_raw.get("tsmom_lookbacks", [63, 126, 189, 252])
        ],
        crossover_short_windows=[
            int(x) for x in experiments_raw.get("crossover_short_windows", [20, 50, 100])
        ],
        crossover_long_windows=[
            int(x) for x in experiments_raw.get("crossover_long_windows", [100, 150, 200, 250])
        ],
        breakout_entry_lookbacks=[
            int(x) for x in experiments_raw.get("breakout_entry_lookbacks", [63, 126, 252])
        ],
        breakout_exit_lookbacks=[
            int(x) for x in experiments_raw.get("breakout_exit_lookbacks", [21, 63, 126])
        ],
        regression_windows=[
            int(x) for x in experiments_raw.get("regression_windows", [63, 126, 189, 252])
        ],
        cross_sectional_lookbacks=[
            int(x) for x in experiments_raw.get("cross_sectional_lookbacks", [63, 126, 252])
        ],
        cross_sectional_top_ns=[
            int(x) for x in experiments_raw.get("cross_sectional_top_ns", [2, 3, 5])
        ],
    )

    reports = ReportsConfig(
        figures_dir=resolve_path(root, reports_raw.get("figures_dir", "reports/figures")),
        tables_dir=resolve_path(root, reports_raw.get("tables_dir", "reports/tables")),
        summary_path=resolve_path(root, reports_raw.get("summary_path", "reports/summary.md")),
    )

    ensure_directory(data.raw_dir)
    ensure_directory(data.processed_dir)
    ensure_directory(reports.figures_dir)
    ensure_directory(reports.tables_dir)
    ensure_directory(reports.summary_path.parent)

    return ProjectConfig(
        name=str(project.get("name", "trend-following-real-data")),
        root=root,
        data=data,
        backtest=backtest,
        strategies=strategies,
        experiments=experiments,
        reports=reports,
    )
