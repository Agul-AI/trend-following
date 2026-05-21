"""Convert validated raw OHLCV files into adjusted price and return panels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trend_following.config import ProjectConfig
from trend_following.data_validation import (
    read_price_file,
    validate_price_frame,
    validation_has_fatal_issue,
)
from trend_following.utils import ensure_directory


def _load_and_validate_raw(config: ProjectConfig, ticker: str) -> pd.DataFrame:
    path = config.data.raw_dir / f"{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing raw data for {ticker}: {path}")
    frame = read_price_file(path)
    report = validate_price_frame(
        frame,
        ticker=ticker,
        suspicious_gap_days=config.data.suspicious_gap_days,
    )
    if validation_has_fatal_issue(report):
        raise ValueError(f"Fatal data validation issue for {ticker}: {report['messages']}")
    return frame.sort_index()


def _adjusted_open(frame: pd.DataFrame) -> pd.Series:
    """Approximate adjusted open using the adjusted-close/close adjustment factor."""
    adjustment_factor = frame["adj_close"] / frame["close"]
    adjustment_factor = adjustment_factor.replace([np.inf, -np.inf], np.nan)
    return frame["open"] * adjustment_factor


def build_adjusted_panels(
    config: ProjectConfig,
    tickers: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Create adjusted close/open, returns, and volume panels from raw data.

    No forward fill is applied. With the default ``inner`` alignment, dates with
    any missing adjusted close across the selected universe are dropped. With
    ``outer`` alignment, missing prices remain as NaN and downstream users must
    handle them explicitly.
    """
    requested = tickers or config.data.tickers
    if not requested:
        raise ValueError("No tickers provided for processing")

    adjusted_close: dict[str, pd.Series] = {}
    adjusted_open: dict[str, pd.Series] = {}
    volume: dict[str, pd.Series] = {}

    for ticker in requested:
        frame = _load_and_validate_raw(config, ticker)
        adjusted_close[ticker] = frame["adj_close"].astype(float)
        adjusted_open[ticker] = _adjusted_open(frame).astype(float)
        volume[ticker] = (
            frame["volume"].astype(float) if "volume" in frame else pd.Series(index=frame.index)
        )

    close_panel = pd.DataFrame(adjusted_close).sort_index()
    open_panel = pd.DataFrame(adjusted_open).sort_index()
    volume_panel = pd.DataFrame(volume).sort_index()

    # Keep only business weekdays. Market holidays are naturally absent; we do
    # not create synthetic rows or forward-fill around holidays.
    close_panel = close_panel.loc[close_panel.index.weekday < 5]
    open_panel = open_panel.reindex(close_panel.index)
    volume_panel = volume_panel.reindex(close_panel.index)

    if config.data.alignment == "inner":
        valid_index = close_panel.dropna(how="any").index
        close_panel = close_panel.loc[valid_index]
        open_panel = open_panel.loc[valid_index]
        volume_panel = volume_panel.loc[valid_index]
    elif config.data.alignment == "outer":
        # Do not forward-fill. Missing values remain visible in the panel.
        pass
    else:  # guarded by config validation
        raise ValueError(f"Unsupported alignment mode: {config.data.alignment}")

    returns = close_panel.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    if not returns.empty:
        returns.iloc[0] = 0.0

    ensure_directory(config.data.processed_dir)
    close_panel.to_parquet(config.data.processed_dir / "adjusted_close.parquet")
    open_panel.to_parquet(config.data.processed_dir / "adjusted_open.parquet")
    returns.to_parquet(config.data.processed_dir / "returns.parquet")
    volume_panel.to_parquet(config.data.processed_dir / "volume.parquet")

    return {
        "adjusted_close": close_panel,
        "adjusted_open": open_panel,
        "returns": returns,
        "volume": volume_panel,
    }


def load_processed_panels(processed_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load processed panels from parquet."""
    base = Path(processed_dir)
    required = {
        "adjusted_close": base / "adjusted_close.parquet",
        "adjusted_open": base / "adjusted_open.parquet",
        "returns": base / "returns.parquet",
        "volume": base / "volume.parquet",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing processed panel(s): {missing}")
    return {name: pd.read_parquet(path).sort_index() for name, path in required.items()}
