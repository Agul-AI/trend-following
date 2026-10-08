"""Download and cache OHLCV data from supported market-data sources."""

from __future__ import annotations

import os
import time
from csv import DictWriter
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
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

ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
ALPHA_VANTAGE_INTRADAY_INTERVALS = {
    "1m": "1min",
    "1min": "1min",
    "5m": "5min",
    "5min": "5min",
    "15m": "15min",
    "15min": "15min",
    "30m": "30min",
    "30min": "30min",
    "60m": "60min",
    "60min": "60min",
    "1h": "60min",
}
ALPHA_VANTAGE_DAILY_INTERVALS = {"1d", "d", "daily"}
_ALPHA_VANTAGE_LAST_REQUEST_START: float | None = None

STOOQ_URL = "https://stooq.com/q/d/l/"
STOOQ_INTERVALS = {
    "5m": "5",
    "5min": "5",
    "60m": "60",
    "60min": "60",
    "1h": "60",
    "1d": "d",
    "d": "d",
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
    interval: str = "1d",
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
                interval=interval,
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


def _month_starts(start_date: str, end_date: str | None) -> list[pd.Timestamp]:
    start = pd.Timestamp(start_date).normalize().replace(day=1)
    end = pd.Timestamp(end_date).normalize() if end_date else pd.Timestamp.today().normalize()
    end = end.replace(day=1)
    return list(pd.date_range(start=start, end=end, freq="MS"))


def _alpha_vantage_interval(interval: str) -> str:
    normalized = interval.lower()
    if normalized not in ALPHA_VANTAGE_INTRADAY_INTERVALS:
        supported = ", ".join(sorted(ALPHA_VANTAGE_INTRADAY_INTERVALS))
        raise ValueError(f"Alpha Vantage intraday interval must be one of: {supported}")
    return ALPHA_VANTAGE_INTRADAY_INTERVALS[normalized]


def _normalize_alpha_vantage_csv(csv_text: str, ticker: str) -> pd.DataFrame:
    frame = pd.read_csv(StringIO(csv_text))
    if frame.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume"])

    rename = {column: str(column).strip().lower() for column in frame.columns}
    frame = frame.rename(columns=rename)
    if "timestamp" not in frame.columns:
        raise ValueError(f"Alpha Vantage response for {ticker} did not include timestamp data")

    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Alpha Vantage response for {ticker} missing columns: {missing}")

    frame = frame.rename(columns={"timestamp": "date"})
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    # The endpoint is queried with adjusted=true, so close is already adjusted.
    frame["adj_close"] = frame["close"]
    return frame[["date", "open", "high", "low", "close", "adj_close", "volume"]]


def _normalize_alpha_vantage_daily_adjusted_csv(csv_text: str, ticker: str) -> pd.DataFrame:
    """Normalize Alpha Vantage daily adjusted CSV into the raw parquet schema."""
    frame = pd.read_csv(StringIO(csv_text))
    if frame.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume"])

    rename = {column: str(column).strip().lower() for column in frame.columns}
    frame = frame.rename(columns=rename)
    if "timestamp" not in frame.columns:
        raise ValueError(
            f"Alpha Vantage daily response for {ticker} did not include timestamp data"
        )

    frame = frame.rename(
        columns={
            "timestamp": "date",
            "adjusted_close": "adj_close",
        }
    )
    required = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Alpha Vantage daily response for {ticker} missing columns: {missing}")

    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "dividend_amount",
        "split_coefficient",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    ordered_columns = [
        column
        for column in [
            "date",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "dividend_amount",
            "split_coefficient",
        ]
        if column in frame.columns
    ]
    return frame[ordered_columns].sort_values("date").reset_index(drop=True)


def _raise_for_alpha_vantage_message(text: str, ticker: str, label: str) -> None:
    stripped = text.strip()
    if not stripped.startswith("{"):
        return
    try:
        payload = requests.models.complexjson.loads(stripped)
    except ValueError:
        return
    message = (
        payload.get("Error Message")
        or payload.get("Information")
        or payload.get("Note")
        or payload.get("message")
    )
    if message:
        raise RuntimeError(f"Alpha Vantage message for {ticker} {label}: {message}")


def _alpha_vantage_uses_start_time_rate_limiter() -> bool:
    """Return True when request-start pacing should replace post-response sleeps."""
    mode = os.getenv("ALPHA_VANTAGE_RATE_LIMIT_MODE", "start_time").strip().lower()
    return mode in {"start_time", "start", "start-time"}


def _alpha_vantage_min_start_interval_seconds() -> float:
    """Minimum seconds between Alpha Vantage request starts.

    For a 75 requests/minute plan, the theoretical interval is 0.80 seconds.
    The project uses 0.85 seconds by default during Premium bootstrap to leave
    a small rate-limit buffer while not adding response latency to the throttle.
    """
    configured = os.getenv(
        "ALPHA_VANTAGE_REQUEST_START_INTERVAL_SECONDS",
        os.getenv("ALPHA_VANTAGE_PAUSE_SECONDS", "12"),
    )
    return max(float(configured), 0.0)


def _wait_for_alpha_vantage_start_slot() -> None:
    """Throttle Alpha Vantage by request start time rather than response finish time."""
    global _ALPHA_VANTAGE_LAST_REQUEST_START
    min_interval = _alpha_vantage_min_start_interval_seconds()
    if min_interval <= 0:
        _ALPHA_VANTAGE_LAST_REQUEST_START = time.monotonic()
        return

    now = time.monotonic()
    if _ALPHA_VANTAGE_LAST_REQUEST_START is not None:
        elapsed = now - _ALPHA_VANTAGE_LAST_REQUEST_START
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
    _ALPHA_VANTAGE_LAST_REQUEST_START = time.monotonic()


def _alpha_vantage_get(
    params: dict[str, str],
    ticker: str,
    label: str,
    retries: int | None = None,
    timeout_seconds: float | None = None,
) -> requests.Response:
    """GET an Alpha Vantage endpoint with retry handling for long bulk downloads."""
    max_retries = retries if retries is not None else int(os.getenv("ALPHA_VANTAGE_RETRIES", "5"))
    timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else float(os.getenv("ALPHA_VANTAGE_TIMEOUT_SECONDS", "120"))
    )
    retry_pause = float(os.getenv("ALPHA_VANTAGE_RETRY_PAUSE_SECONDS", "5"))
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            if _alpha_vantage_uses_start_time_rate_limiter():
                _wait_for_alpha_vantage_start_slot()
            response = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=timeout)
            response.raise_for_status()
            _raise_for_alpha_vantage_message(response.text, ticker=ticker, label=label)
            return response
        except requests.RequestException as exc:
            last_error = exc
        except RuntimeError as exc:
            # Retry temporary rate-limit informational responses, but fail fast for
            # endpoint entitlement or symbol errors.
            if "rate limit" not in str(exc).lower() and "frequency" not in str(exc).lower():
                raise
            last_error = exc

        if attempt < max_retries:
            time.sleep(retry_pause * attempt)

    raise RuntimeError(
        f"Alpha Vantage request failed for {ticker} {label} after {max_retries} attempts: "
        f"{last_error}"
    )


def _download_alpha_vantage_month(
    ticker: str,
    month: str,
    interval: str,
    api_key: str,
    adjusted: bool = True,
    extended_hours: bool = False,
) -> pd.DataFrame:
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": ticker,
        "interval": interval,
        "adjusted": str(adjusted).lower(),
        "extended_hours": str(extended_hours).lower(),
        "month": month,
        "outputsize": "full",
        "datatype": "csv",
        "apikey": api_key,
    }
    response = _alpha_vantage_get(params, ticker=ticker, label=month)
    return _normalize_alpha_vantage_csv(response.text, ticker=ticker)


def _alpha_vantage_should_skip_invalid_month(error: Exception) -> bool:
    """Return True for isolated Alpha Vantage intraday monthly-slice errors.

    Alpha Vantage's historical intraday endpoint is queried one month at a time.
    Some ETF inception-edge months, and occasionally isolated historical months,
    return a JSON ``Invalid API call`` response even when surrounding months are
    available. For research bootstrapping it is better to log and skip only that
    month than to discard the whole ticker.
    """
    if os.getenv("ALPHA_VANTAGE_SKIP_INVALID_MONTHS", "true").strip().lower() in {
        "0",
        "false",
        "no",
    }:
        return False
    message = str(error).lower()
    return "invalid api call" in message and "time_series_intraday" in message


def _record_alpha_vantage_skipped_month(
    ticker: str,
    interval: str,
    month: str,
    error: Exception,
) -> None:
    """Append a skipped-month record when ALPHA_VANTAGE_SKIPPED_MONTHS_PATH is set."""
    path_value = os.getenv("ALPHA_VANTAGE_SKIPPED_MONTHS_PATH")
    if not path_value:
        return

    output_path = Path(path_value)
    ensure_directory(output_path.parent)
    fieldnames = ["ticker", "interval", "month", "reason"]
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", newline="") as file:
        writer = DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "ticker": ticker,
                "interval": interval,
                "month": month,
                "reason": str(error),
            }
        )


def _filter_date_range(
    frame: pd.DataFrame,
    start_date: str,
    end_date: str | None,
) -> pd.DataFrame:
    filtered = frame.copy()
    start_ts = pd.Timestamp(start_date)
    filtered = filtered[filtered["date"] >= start_ts]
    if end_date:
        end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        filtered = filtered[filtered["date"] < end_ts]
    return filtered.reset_index(drop=True)


def _download_alpha_vantage_daily_adjusted(
    ticker: str,
    start_date: str,
    end_date: str | None,
    api_key: str,
) -> pd.DataFrame:
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "outputsize": "full",
        "datatype": "csv",
        "apikey": api_key,
    }
    response = _alpha_vantage_get(params, ticker=ticker, label="daily_adjusted")
    normalized = _normalize_alpha_vantage_daily_adjusted_csv(response.text, ticker=ticker)
    filtered = _filter_date_range(normalized, start_date=start_date, end_date=end_date)
    if filtered.empty:
        raise ValueError(f"No Alpha Vantage daily rows returned for {ticker}")
    return filtered


def _download_alpha_vantage(
    ticker: str,
    start_date: str,
    end_date: str | None,
    interval: str,
    api_key: str,
    pause_seconds: float = 12.0,
) -> pd.DataFrame:
    av_interval = _alpha_vantage_interval(interval)
    monthly_frames: list[pd.DataFrame] = []
    months = _month_starts(start_date, end_date)

    for month_start in months:
        month = month_start.strftime("%Y-%m")
        try:
            frame = _download_alpha_vantage_month(ticker, month, av_interval, api_key)
        except RuntimeError as exc:
            if not _alpha_vantage_should_skip_invalid_month(exc):
                raise
            _record_alpha_vantage_skipped_month(ticker, av_interval, month, exc)
            print(f"WARNING: skipping {ticker} {av_interval} {month}: {exc}")
            frame = pd.DataFrame()
        if not frame.empty:
            monthly_frames.append(frame)
        if (
            not _alpha_vantage_uses_start_time_rate_limiter()
            and pause_seconds > 0
            and month_start != months[-1]
        ):
            time.sleep(pause_seconds)

    if not monthly_frames:
        raise ValueError(f"No Alpha Vantage rows returned for {ticker}")

    combined = pd.concat(monthly_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["date"]).sort_values("date")
    return _filter_date_range(combined, start_date=start_date, end_date=end_date)


def _stooq_interval(interval: str) -> str:
    normalized = interval.lower()
    if normalized not in STOOQ_INTERVALS:
        supported = ", ".join(sorted(STOOQ_INTERVALS))
        raise ValueError(f"Stooq interval must be one of: {supported}")
    return STOOQ_INTERVALS[normalized]


def _stooq_symbol(ticker: str) -> str:
    """Map a US ticker to Stooq's symbol convention."""
    if "." in ticker:
        return ticker.lower()
    return f"{ticker.lower()}.us"


def _date_to_stooq(value: str | None) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value).strftime("%Y%m%d")


def _raise_for_stooq_message(text: str, ticker: str) -> None:
    stripped = text.strip()
    lower = stripped.lower()
    if lower.startswith("get your apikey"):
        raise RuntimeError(
            "Stooq CSV downloads require STOOQ_API_KEY. Open "
            f"https://stooq.com/q/d/?s={_stooq_symbol(ticker)}&get_apikey, solve the captcha, "
            "copy the apikey from the generated CSV link, and save it in .env as STOOQ_API_KEY."
        )
    if lower.startswith("no data") or "symbol not found" in lower:
        raise ValueError(f"No Stooq data returned for {ticker}: {stripped[:200]}")


def _normalize_stooq_csv(csv_text: str, ticker: str) -> pd.DataFrame:
    frame = pd.read_csv(StringIO(csv_text))
    if frame.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume"])

    rename = {column: str(column).strip().lower() for column in frame.columns}
    frame = frame.rename(columns=rename)

    if "datetime" in frame.columns:
        frame["date"] = pd.to_datetime(frame["datetime"])
    elif "date" in frame.columns and "time" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"].astype(str) + " " + frame["time"].astype(str))
    elif "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
    else:
        raise ValueError(f"Stooq response for {ticker} did not include a date column")

    required = ["open", "high", "low", "close"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Stooq response for {ticker} missing columns: {missing}")

    if "volume" not in frame.columns:
        frame["volume"] = pd.NA
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # Stooq's CSV endpoint does not expose a separate adjusted-close column.
    # Keep the raw close as adj_close so downstream validation/processing can run,
    # and document the vendor-specific adjustment limitation in the README.
    frame["adj_close"] = frame["close"]
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    return (
        frame[["date", "open", "high", "low", "close", "adj_close", "volume"]]
        .sort_values("date")
        .reset_index(drop=True)
    )


def _download_stooq(
    ticker: str,
    start_date: str,
    end_date: str | None,
    interval: str,
    api_key: str,
) -> pd.DataFrame:
    params = {
        "s": _stooq_symbol(ticker),
        "i": _stooq_interval(interval),
        "d1": _date_to_stooq(start_date),
        "apikey": api_key,
    }
    if end_date:
        params["d2"] = _date_to_stooq(end_date)

    response = requests.get(
        STOOQ_URL,
        params=params,
        timeout=60,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    _raise_for_stooq_message(response.text, ticker=ticker)
    normalized = _normalize_stooq_csv(response.text, ticker=ticker)
    if normalized.empty:
        raise ValueError(f"No Stooq rows returned for {ticker}")
    return normalized


def summarize_frame(frame: pd.DataFrame, ticker: str, source: str) -> str:
    """Build a concise download/cache summary string."""
    if frame.empty:
        return f"{ticker}: {source}, empty"
    dates = pd.to_datetime(frame["date"] if "date" in frame.columns else frame.index)
    return f"{ticker}: {source}, {len(frame):,} rows, {dates.min().date()} -> {dates.max().date()}"


def download_ticker(config: ProjectConfig, ticker: str, force: bool = False) -> Path:
    """Download one ticker to ``data/raw/{ticker}.parquet`` unless cached."""
    ticker = ticker.upper()
    ensure_directory(config.data.raw_dir)
    output_path = config.data.raw_dir / f"{ticker}.parquet"
    if output_path.exists() and not force:
        cached = pd.read_parquet(output_path)
        print(summarize_frame(cached, ticker, "cached"))
        return output_path

    if config.data.source == "yfinance":
        downloaded = _download_with_retries(
            ticker,
            config.data.start_date,
            config.data.end_date,
            interval=config.data.interval,
        )
    elif config.data.source == "alpha_vantage":
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ALPHA_VANTAGE_API_KEY is not set. Add it to .env or the shell environment."
            )
        pause_seconds = float(os.getenv("ALPHA_VANTAGE_PAUSE_SECONDS", "12"))
        if config.data.interval in ALPHA_VANTAGE_DAILY_INTERVALS:
            downloaded = _download_alpha_vantage_daily_adjusted(
                ticker,
                config.data.start_date,
                config.data.end_date,
                api_key=api_key,
            )
        else:
            downloaded = _download_alpha_vantage(
                ticker,
                config.data.start_date,
                config.data.end_date,
                interval=config.data.interval,
                api_key=api_key,
                pause_seconds=pause_seconds,
            )
    elif config.data.source == "stooq":
        api_key = os.getenv("STOOQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "STOOQ_API_KEY is not set. Open a Stooq download URL with get_apikey, "
                "solve the captcha, and add the apikey to .env."
            )
        downloaded = _download_stooq(
            ticker,
            config.data.start_date,
            config.data.end_date,
            interval=config.data.interval,
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unsupported data source: {config.data.source}")

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
