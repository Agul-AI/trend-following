"""Download and cache daily OHLCV data from yfinance."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from trend_following.config import ProjectConfig
from trend_following.utils import ensure_directory

YFINANCE_RENAME = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj close": "adj_close",
    "adj_close": "adj_close",
    "volume": "volume",
}


def _flatten_yfinance_columns(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize possible yfinance MultiIndex columns for a single ticker."""
    if isinstance(frame.columns, pd.MultiIndex):
        columns = frame.columns
        # yfinance commonly returns columns like ('Close', 'SPY'). Drop the ticker level.
        if ticker in columns.get_level_values(-1):
            frame = frame.xs(ticker, axis=1, level=-1, drop_level=True)
        elif ticker in columns.get_level_values(0):
            frame = frame.xs(ticker, axis=1, level=0, drop_level=True)
        else:
            frame.columns = ["_".join(str(part) for part in col if part) for col in columns]
    return frame


def normalize_downloaded_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Convert yfinance output into a stable raw parquet schema."""
    if frame.empty:
        raise ValueError(f"No rows downloaded for {ticker}")
    frame = _flatten_yfinance_columns(frame.copy(), ticker)
    frame = frame.reset_index()
    rename = {}
    for column in frame.columns:
        normalized = str(column).strip().lower().replace("_", " ")
        if normalized == "date" or normalized == "datetime":
            rename[column] = "date"
        elif normalized in YFINANCE_RENAME:
            rename[column] = YFINANCE_RENAME[normalized]
    frame = frame.rename(columns=rename)
    if "date" not in frame.columns:
        raise ValueError(f"Downloaded data for {ticker} has no date column")
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    ordered_columns = [
        column
        for column in ["date", "open", "high", "low", "close", "adj_close", "volume"]
        if column in frame.columns
    ]
    frame = frame[ordered_columns].sort_values("date").reset_index(drop=True)
    return frame


def _download_with_retries(
    ticker: str,
    start_date: str,
    end_date: str | None,
    retries: int = 3,
    pause_seconds: float = 2.0,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            frame = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
                multi_level_index=True,
            )
            normalized = normalize_downloaded_frame(frame, ticker)
            if normalized.empty:
                raise ValueError(f"No rows returned for {ticker}")
            return normalized
        except Exception as exc:  # pragma: no cover - network failures are environment-specific
            last_error = exc
            if attempt < retries:
                time.sleep(pause_seconds * attempt)
    raise RuntimeError(f"Failed to download {ticker} after {retries} attempts: {last_error}")


def summarize_frame(frame: pd.DataFrame, ticker: str, source: str) -> str:
    """Build a concise download/cache summary string."""
    if frame.empty:
        return f"{ticker}: {source}, empty"
    dates = pd.to_datetime(frame["date"] if "date" in frame.columns else frame.index)
    return f"{ticker}: {source}, {len(frame):,} rows, {dates.min().date()} -> {dates.max().date()}"


def download_ticker(config: ProjectConfig, ticker: str, force: bool = False) -> Path:
    """Download one ticker to ``data/raw/{ticker}.parquet`` unless cached."""
    if config.data.source != "yfinance":
        raise ValueError(f"Unsupported data source: {config.data.source}")
    ticker = ticker.upper()
    ensure_directory(config.data.raw_dir)
    output_path = config.data.raw_dir / f"{ticker}.parquet"
    if output_path.exists() and not force:
        cached = pd.read_parquet(output_path)
        print(summarize_frame(cached, ticker, "cached"))
        return output_path

    downloaded = _download_with_retries(ticker, config.data.start_date, config.data.end_date)
    downloaded.to_parquet(output_path, index=False)
    print(summarize_frame(downloaded, ticker, "downloaded"))
    return output_path


def download_universe(
    config: ProjectConfig, tickers: list[str] | None = None, force: bool = False
) -> list[Path]:
    """Download all requested tickers and return saved parquet paths."""
    requested = tickers or config.data.tickers
    saved_paths = []
    for ticker in requested:
        try:
            saved_paths.append(download_ticker(config, ticker, force=force))
        except Exception as exc:  # pragma: no cover - network failures are environment-specific
            print(f"ERROR downloading {ticker}: {exc}")
            raise
    return saved_paths
