"""Immutable future-data input packages for a frozen-rule simulation, not live fills.

The Yahoo adapter uses completed regular-session bar-end closes, restores physical
share units from split-adjusted downloaded prices and preserves the unmodified
provider frames. Missing bars are reported, never padded or replaced by daily bars.
Economic action availability is a disclosed ex-date assumption; retrieval times
are separate and are not represented as contemporaneous execution evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ASSETS = ("QQQ", "TQQQ")
ACTION_COLUMNS = [
    "action_id",
    "asset",
    "kind",
    "effective_at",
    "available_at",
    "split_ratio",
    "amount_per_share",
    "pay_at",
    "character",
    "availability_basis",
    "source_observed_at",
]


def utc(value: Any, name: str = "timestamp") -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if pd.isna(ts) or ts.tzinfo is None:
        raise ValueError(f"{name} must be a nonmissing timezone-aware timestamp")
    return ts.tz_convert("UTC")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_file(root: Path, relative: str) -> Path:
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or symlink input: {relative}")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("input path escapes package") from error
    # Also disallow a symlink in any parent directory beneath the input root.
    if any(
        (root / Path(*Path(relative).parts[:n])).is_symlink()
        for n in range(1, len(Path(relative).parts) + 1)
    ):
        raise ValueError("symlink package path is not allowed")
    return path


def read_calendar(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"session", "market_open", "market_close", "is_early_close"}
    if not required.issubset(frame):
        raise ValueError("calendar lacks required fields")
    if frame.empty or frame.session.duplicated().any() or not frame.session.is_monotonic_increasing:
        raise ValueError("calendar must be nonempty, chronological and unique")
    for column in ["market_open", "market_close"]:
        frame[column] = [utc(v, column) for v in frame[column]]
    early = frame.is_early_close.astype(str).str.lower()
    if not early.isin(["true", "false"]).all():
        raise ValueError("invalid early-close flags")
    frame["is_early_close"] = early.eq("true")
    for row in frame.itertuples():
        op, cl = (
            row.market_open.tz_convert("America/New_York"),
            row.market_close.tz_convert("America/New_York"),
        )
        if (
            op.strftime("%Y-%m-%d") != row.session
            or cl.date() != op.date()
            or op.strftime("%H:%M") != "09:30"
            or cl.strftime("%H:%M") != ("13:00" if row.is_early_close else "16:00")
        ):
            raise ValueError("unsupported or inconsistent scheduled session")
    return frame


def expected_bars(calendar: pd.DataFrame, start_session: str, end_session: str) -> pd.DataFrame:
    selected = calendar.loc[calendar.session.between(start_session, end_session)]
    rows = []
    for row in selected.itertuples():
        for start in pd.date_range(row.market_open, row.market_close, freq="h", inclusive="left"):
            rows.append(
                {
                    "bar_start": start,
                    "bar_end": min(start + pd.Timedelta(hours=1), row.market_close),
                    "session": row.session,
                }
            )
    return pd.DataFrame(rows, columns=["bar_start", "bar_end", "session"])


def validate_panel(prices: pd.DataFrame) -> pd.DataFrame:
    if (
        not isinstance(prices.index, pd.DatetimeIndex)
        or prices.index.tz is None
        or prices.empty
        or prices.index.has_duplicates
        or not prices.index.is_monotonic_increasing
        or list(prices.columns) != list(ASSETS)
    ):
        raise ValueError("prices require aware unique sorted bar ends and columns QQQ,TQQQ")
    values = prices.astype(float)
    if not np.isfinite(values.to_numpy()).all() or values.le(0).any().any():
        raise ValueError("all prices must be finite and positive")
    return values


def coverage_report(prices: pd.DataFrame, calendar: pd.DataFrame, *, as_of: pd.Timestamp) -> dict:
    prices = validate_panel(prices)
    now = utc(as_of)
    local = prices.index.tz_convert("America/New_York")
    grid = expected_bars(calendar, local[0].strftime("%Y-%m-%d"), local[-1].strftime("%Y-%m-%d"))
    expected = pd.DatetimeIndex(grid.bar_end)
    actual = prices.index.tz_convert("UTC")
    missing, extra = expected.difference(actual), actual.difference(expected)
    if len(actual) and actual[-1] > now:
        raise ValueError("package includes incomplete or future bars")
    return {
        "expected_bars": len(expected),
        "observed_bars": len(actual),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "missing_bar_ends_utc": [x.isoformat() for x in missing],
        "extra_bar_ends_utc": [x.isoformat() for x in extra],
        "start": actual[0].isoformat(),
        "end": actual[-1].isoformat(),
        "complete": not len(missing) and not len(extra),
    }


def _provider_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
    if not isinstance(out.index, pd.DatetimeIndex):
        if "date" not in out:
            raise ValueError("provider frame needs an aware timestamp index/date")
        out.index = pd.DatetimeIndex(pd.to_datetime(out.pop("date"), utc=True))
    if out.index.tz is None or out.index.has_duplicates or not out.index.is_monotonic_increasing:
        raise ValueError("provider timestamps must be aware, sorted and unique")
    out.index = out.index.tz_convert("UTC")
    if "close" not in out or out.close.isna().any() or out.close.le(0).any():
        raise ValueError("provider close missing or nonpositive")
    for col in ["dividends", "stock_splits", "capital_gains"]:
        if col not in out:
            raise ValueError(f"provider action field missing: {col}")
        if not np.isfinite(out[col]).all() or out[col].lt(0).any():
            raise ValueError("provider action fields must be finite and nonnegative")
    return out


def normalize_yahoo_frames(
    hourly: dict[str, pd.DataFrame],
    daily: dict[str, pd.DataFrame],
    calendar: pd.DataFrame,
    distribution_calendar: pd.DataFrame,
    *,
    observed_at: pd.Timestamp,
    fallback_30min: dict[str, pd.DataFrame] | None = None,
    fallback_daily: dict[str, pd.DataFrame] | None = None,
    fallback_15min: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Normalize as-of-download split units; never hide absent scheduled closes.

    The adapter contract is Yahoo Close split-adjusted to the END of the request,
    auto_adjust=False/back_adjust=False. All requests must share that same end.
    Physical historical quotes and dividend amounts multiply splits strictly AFTER
    their date, using only actions already effective by download time.
    """
    now = utc(observed_at)
    required = {"asset", "ex_date", "pay_date", "source_url"}
    if not required.issubset(distribution_calendar):
        raise ValueError("distribution calendar requires asset/ex_date/pay_date/source_url")
    if distribution_calendar.duplicated(["asset", "ex_date"]).any():
        raise ValueError("duplicate distribution calendar entries")
    actions, series, basis_records = [], {}, []
    shared_range = None
    for asset in ASSETS:
        h, d = _provider_frame(hourly[asset]), _provider_frame(daily[asset])
        dates = h.index.tz_convert("America/New_York").strftime("%Y-%m-%d")
        daily_dates = d.index.tz_convert("America/New_York").strftime("%Y-%m-%d")
        if h.empty or d.empty or max(daily_dates) != max(dates):
            raise ValueError("hourly/daily frames must have the same last request session")
        request_range = (min(dates), max(dates))
        if shared_range is not None and shared_range != request_range:
            raise ValueError("both symbols must share the same request/session range")
        shared_range = request_range
        expected_daily = calendar.loc[calendar.session.between(*request_range), "session"].tolist()
        actual_daily = list(
            daily_dates[(daily_dates >= request_range[0]) & (daily_dates <= request_range[1])]
        )
        if actual_daily != expected_daily:
            raise ValueError("daily action source must cover every scheduled session exactly once")
        for field in ["dividends", "stock_splits", "capital_gains"]:
            for stamp, value in h.loc[h[field].ne(0), field].items():
                day = stamp.tz_convert("America/New_York").strftime("%Y-%m-%d")
                daily_value = d.loc[daily_dates == day, field]
                if len(daily_value) != 1 or float(daily_value.iloc[0]) != float(value):
                    raise ValueError("hourly action disagrees with complete daily action source")
        latest_session = calendar.loc[calendar.session.eq(max(dates))]
        if len(latest_session) != 1 or latest_session.iloc[0].market_close > now:
            raise ValueError("acquisition must end at a fully completed scheduled session")
        split_rows = [
            (str(day), float(ratio))
            for day, ratio in zip(daily_dates, d.stock_splits, strict=True)
            if ratio
        ]
        if any(
            day > now.tz_convert("America/New_York").strftime("%Y-%m-%d") for day, _ in split_rows
        ):
            raise ValueError("future split in downloaded action inputs")

        def restore_factor(day, splits=tuple(split_rows)):
            return float(np.prod([ratio for ex, ratio in splits if ex > day]))

        grid = expected_bars(calendar, *shared_range)
        end_by_start = dict(zip(grid.bar_start, grid.bar_end, strict=True))
        ends = []
        for stamp in h.index:
            if stamp not in end_by_start:
                raise ValueError(f"unexpected/out-of-session hourly timestamp: {asset} {stamp}")
            ends.append(end_by_start[stamp])
        series[asset] = pd.Series(
            [price * restore_factor(day) for price, day in zip(h.close, dates, strict=True)],
            index=pd.DatetimeIndex(ends),
            name=asset,
        )
        for day, row in zip(daily_dates, d.to_dict("records"), strict=True):
            if day < min(dates) or day > max(dates):
                continue
            session = calendar.loc[calendar.session.eq(day)]
            if len(session) != 1:
                if row["dividends"] or row["stock_splits"] or row["capital_gains"]:
                    raise ValueError("action on unknown scheduled session")
                continue
            effective = session.iloc[0].market_open
            common = {
                "asset": asset,
                "effective_at": effective,
                "available_at": effective,
                "availability_basis": "assumed_public_by_ex_date",
                "source_observed_at": now,
            }
            if row["stock_splits"]:
                actions.append(
                    {
                        **common,
                        "action_id": f"{asset}:{day}:split",
                        "kind": "split",
                        "split_ratio": float(row["stock_splits"]),
                        "amount_per_share": 0.0,
                        "character": "unknown",
                        "pay_at": pd.NaT,
                    }
                )
            if row["dividends"] or row["capital_gains"]:
                matches = distribution_calendar.loc[
                    distribution_calendar.asset.eq(asset) & distribution_calendar.ex_date.eq(day)
                ]
                if len(matches) != 1:
                    raise ValueError(f"missing verified payment date for {asset} {day}")
                record = matches.iloc[0]
                pay = pd.Timestamp(record.pay_date).tz_localize("America/New_York") + pd.Timedelta(
                    hours=16
                )
                # Conservative spendable-cash convention: 16:00 NY on declared pay date.
                amount = float(row["dividends"] + row["capital_gains"]) * restore_factor(day)
                character = (
                    "long_capital_gain"
                    if row["capital_gains"] and not row["dividends"]
                    else "unknown"
                )
                actions.append(
                    {
                        **common,
                        "action_id": f"{asset}:{day}:distribution",
                        "kind": "distribution",
                        "split_ratio": 1.0,
                        "amount_per_share": amount,
                        "pay_at": pay.tz_convert("UTC"),
                        "character": character,
                    }
                )
                basis_records.append(
                    {
                        "asset": asset,
                        "ex_date": day,
                        "source_url": str(record.source_url),
                        "amount_source": "Yahoo daily action; request-end split units restored",
                        "pay_cash_convention": "16:00 America/New_York on stated pay date",
                    }
                )
    fallback_records = []
    if (fallback_30min is None) != (fallback_daily is None):
        raise ValueError(
            "AV fallback needs both 30-minute closes and matching daily adjustment factors"
        )
    if fallback_30min is not None and fallback_15min is None:
        raise ValueError("AV fallback requires preserved 15min nested-close timing evidence")
    if fallback_30min is not None:
        grid = expected_bars(calendar, min(dates), max(dates))
        for day, session_grid in grid.groupby("session", sort=True):
            ends = pd.DatetimeIndex(session_grid.bar_end)
            if all(ends.isin(series[asset].index).all() for asset in ASSETS):
                continue
            session = calendar.loc[calendar.session.eq(day)].iloc[0]
            expected_subbars = pd.date_range(
                session.market_open, session.market_close, freq="30min", inclusive="left"
            )
            for asset in ASSETS:
                sub = fallback_30min[asset].copy()
                sub.columns = [str(c).lower().replace(" ", "_") for c in sub.columns]
                index = (
                    pd.DatetimeIndex(pd.to_datetime(sub.pop("date")))
                    if "date" in sub
                    else sub.index
                )
                if not isinstance(index, pd.DatetimeIndex) or index.has_duplicates:
                    raise ValueError("AV subbar timestamps must be unique")
                # Explicit, independently checked convention of THIS licensed cache.
                index = index.tz_localize("America/New_York") if index.tz is None else index
                sub.index = index.tz_convert("UTC")
                if not expected_subbars.isin(sub.index).all():
                    raise ValueError(
                        f"AV fallback missing complete observed subbars: {asset} {day}"
                    )
                finer = fallback_15min[asset].copy()
                finer.columns = [str(c).lower().replace(" ", "_") for c in finer.columns]
                finer_index = (
                    pd.DatetimeIndex(pd.to_datetime(finer.pop("date")))
                    if "date" in finer
                    else finer.index
                )
                if not isinstance(finer_index, pd.DatetimeIndex) or finer_index.has_duplicates:
                    raise ValueError("15min proof requires unique timestamp observations")
                finer_index = (
                    finer_index.tz_localize("America/New_York")
                    if finer_index.tz is None
                    else finer_index
                )
                finer.index = finer_index.tz_convert("UTC")
                proof_ends = expected_subbars + pd.Timedelta(minutes=15)
                if not proof_ends.isin(finer.index).all() or not np.array_equal(
                    sub.loc[expected_subbars, "close"].astype(float).to_numpy(),
                    finer.loc[proof_ends, "close"].astype(float).to_numpy(),
                ):
                    raise ValueError(
                        "30min start-label timing proof disagrees with preserved 15min endpoints"
                    )
                factor_frame = fallback_daily[asset].copy()
                factor_frame.columns = [
                    str(c).lower().replace(" ", "_") for c in factor_frame.columns
                ]
                factor_dates = pd.DatetimeIndex(pd.to_datetime(factor_frame["date"])).strftime(
                    "%Y-%m-%d"
                )
                factors = factor_frame.loc[factor_dates == day]
                if len(factors) != 1:
                    raise ValueError(
                        f"AV raw/adjusted daily factor missing or duplicated: {asset} {day}"
                    )
                raw, adjusted = float(factors.iloc[0].close), float(factors.iloc[0].adj_close)
                if not np.isfinite([raw, adjusted]).all() or min(raw, adjusted) <= 0:
                    raise ValueError("invalid AV daily restoration factor")
                restoration = raw / adjusted
                replacement = pd.Series(
                    [
                        float(sub.loc[end - pd.Timedelta(minutes=30), "close"]) * restoration
                        for end in ends
                    ],
                    index=ends,
                    name=asset,
                )
                if not np.isfinite(replacement).all() or replacement.le(0).any():
                    raise ValueError("AV fallback closes must be finite and positive")
                old = series[asset].reindex(ends).dropna()
                overlap = (replacement.reindex(old.index) / old - 1).abs() * 10000
                day_mask = (
                    series[asset].index.tz_convert("America/New_York").strftime("%Y-%m-%d") == day
                )
                series[asset] = pd.concat([series[asset].loc[~day_mask], replacement]).sort_index()
                same_day = sub.index.tz_convert("America/New_York").strftime("%Y-%m-%d") == day
                outside = sub.index[same_day & ~sub.index.isin(expected_subbars)]
                fallback_records.append(
                    {
                        "asset": asset,
                        "session": day,
                        "observations": len(ends),
                        "source": "licensed_local_AlphaVantage_30min_plus_matching_daily_factor",
                        "raw_restoration_factor": restoration,
                        "valid_30min_subbars": len(expected_subbars),
                        "nested_15min_closes_verified": len(expected_subbars),
                        "out_of_core_session_provider_rows_excluded": len(outside),
                        "overlapping_Yahoo_quotes": len(old),
                        "maximum_absolute_overlap_difference_bps": float(overlap.max())
                        if len(overlap)
                        else None,
                        "timestamp_basis": "cache_start_label_verified_by_15min_nested_closes_not_universal_vendor_claim",
                    }
                )
    if not series["QQQ"].index.equals(series["TQQQ"].index):
        raise ValueError("QQQ/TQQQ observed bar coverage differs; no inner join deletion")
    prices = validate_panel(pd.concat([series[a] for a in ASSETS], axis=1))
    action_frame = pd.DataFrame(actions, columns=ACTION_COLUMNS)
    return (
        prices,
        action_frame,
        {
            "action_sources": basis_records,
            "full_session_fallbacks": fallback_records,
            "price_basis": "physical raw quotes restored from request-end split-adjusted Close",
            "action_availability": "assumed public by ex-date; not measured real-time availability",
            "signal_basis": "causal cumulative-split-normalized price index; no dividend reinvestment",
        },
    )


def write_input_package(
    output: Path,
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    observed_at: pd.Timestamp,
    provenance: dict,
    raw_files: dict[str, Path] | None = None,
) -> dict:
    from trend_following.research_oos_accounting import normalize_oos_actions

    prices = validate_panel(prices)
    actions = normalize_oos_actions(actions)
    now = utc(observed_at)
    if len(actions) and actions.source_observed_at.max() > now:
        raise ValueError("action observed_at cannot exceed package acquisition time")
    quality = coverage_report(prices, calendar, as_of=now)
    output.mkdir(parents=True, exist_ok=False)
    prices.to_csv(output / "prices.csv", index_label="bar_end")
    actions.to_csv(output / "actions.csv", index=False)
    (output / "quality.json").write_text(json.dumps(quality, indent=2) + "\n")
    if raw_files:
        import shutil

        (output / "raw").mkdir()
        for label, path in raw_files.items():
            if Path(label).name != label or path.is_symlink():
                raise ValueError("safe raw filename and nonsymlink source required")
            shutil.copyfile(path, output / "raw" / label)
    files = [
        {
            "path": p.relative_to(output).as_posix(),
            "sha256": file_hash(p),
            "bytes": p.stat().st_size,
        }
        for p in sorted(output.rglob("*"))
        if p.is_file()
    ]
    acquisition = provenance.get("source_acquisition", {})
    stamps = [utc(x.get("observed_at_utc", x.get("observed_at"))) for x in acquisition.values()]
    if stamps and max(stamps) > now:
        raise ValueError("original source acquisition cannot occur after packaging")
    vintage = max(stamps) if stamps else None
    uses_fallback = bool(provenance.get("full_session_fallbacks"))
    if uses_fallback:
        vintage = max(vintage, now) if vintage is not None else now
    meta = {
        "schema_version": 1,
        "package_kind": "frozen_policy_batch_prices_not_live_execution",
        "observed_at_utc": now.isoformat(),
        "files": files,
        "quality": quality,
        "source_vintage_at_utc": vintage.isoformat() if vintage is not None else None,
        "source_vintage_basis": "Yahoo retrieval plus local AV cache read at packaging; AV original acquisition unknown"
        if uses_fallback
        else "actual Yahoo retrieval",
        "provenance": provenance,
    }
    (output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def read_input_package(package: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    meta_path = safe_file(package, "metadata.json")
    meta = json.loads(meta_path.read_text())
    if (
        meta.get("schema_version") != 1
        or meta.get("package_kind") != "frozen_policy_batch_prices_not_live_execution"
    ):
        raise ValueError("unsupported or incomplete input package")
    utc(meta["observed_at_utc"])
    records = meta["files"]
    paths = [record["path"] for record in records]
    if len(paths) != len(set(paths)) or not {"prices.csv", "actions.csv", "quality.json"}.issubset(
        paths
    ):
        raise ValueError("invalid package file inventory")
    if package.is_symlink() or any(p.is_symlink() for p in package.rglob("*")):
        raise ValueError("package tree may not contain symlinks")
    actual_files = {p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_file()}
    if actual_files != {*paths, "metadata.json"}:
        raise ValueError("unlisted or missing package files")
    for record in records:
        path = safe_file(package, record["path"])
        if path.stat().st_size != record["bytes"] or file_hash(path) != record["sha256"]:
            raise ValueError(f"package input changed: {record['path']}")
    prices = pd.read_csv(package / "prices.csv", float_precision="round_trip")
    prices.index = pd.DatetimeIndex(pd.to_datetime(prices.pop("bar_end"), utc=True))
    prices = validate_panel(prices)
    actions = pd.read_csv(package / "actions.csv", float_precision="round_trip")
    from trend_following.research_oos_accounting import normalize_oos_actions

    actions = normalize_oos_actions(actions)
    if prices.index[-1] > utc(meta["observed_at_utc"]):
        raise ValueError("package prices occur after acquisition")
    if len(actions) and actions.source_observed_at.max() > utc(meta["observed_at_utc"]):
        raise ValueError("package actions occur after acquisition")
    return prices, actions, meta


def verify_package_derivation(package: Path, calendar: pd.DataFrame) -> dict:
    """Production guard: replay normalization from preserved original provider files.

    Generic hand-created normalized packages remain useful test inputs, but cannot
    initialize or score a production study without the required source evidence.
    This binds a transformation, not independent verification of a data vendor.
    """
    prices, actions, meta = read_input_package(package)
    provenance = meta["provenance"]
    if provenance.get("adapter") != "qqq_oos_v2_Yahoo_AV_full_session_v1":
        raise ValueError("production input package requires reviewed source adapter provenance")
    hourly, daily, actual_acquisition = {}, {}, {}
    for asset in ASSETS:
        hourly[asset] = pd.read_parquet(safe_file(package, f"raw/{asset}_hourly.parquet"))
        daily[asset] = pd.read_parquet(safe_file(package, f"raw/{asset}_daily.parquet"))
        info = json.loads(safe_file(package, f"raw/{asset}_metadata.json").read_text())
        request = info.get("request", {})
        expected = {
            "period": "2y",
            "auto_adjust": False,
            "back_adjust": False,
            "actions": True,
            "prepost": False,
            "repair": False,
            "intervals": ["60m", "1d"],
        }
        if any(request.get(k) != v for k, v in expected.items()):
            raise ValueError("unreviewed source request/basis")
        observed = utc(info.get("observed_at_utc", info.get("observed_at")))
        if observed > utc(meta["observed_at_utc"]) or observed < prices.index[-1]:
            raise ValueError("source acquisition does not support completed package quotes")
        actual_acquisition[asset] = info
    if actual_acquisition != provenance.get("source_acquisition"):
        raise ValueError("acquisition provenance disagrees with original source records")
    vintage = max(
        utc(x.get("observed_at_utc", x.get("observed_at"))) for x in actual_acquisition.values()
    )
    if provenance.get("full_session_fallbacks"):
        vintage = max(vintage, utc(meta["observed_at_utc"]))
    if meta.get("source_vintage_at_utc") != vintage.isoformat():
        raise ValueError("source vintage differs from its declared retrieval/cache-read basis")
    reference = pd.read_csv(safe_file(package, "raw/distribution_calendar.csv"))
    fallback_30, fallback_daily, fallback_15 = None, None, None
    if provenance.get("fallback_supplied"):
        fallback_30 = {
            a: pd.read_parquet(safe_file(package, f"raw/{a}_av30.parquet")) for a in ASSETS
        }
        fallback_daily = {
            a: pd.read_parquet(safe_file(package, f"raw/{a}_avdaily.parquet")) for a in ASSETS
        }
        fallback_15 = {
            a: pd.read_parquet(safe_file(package, f"raw/{a}_av15.parquet")) for a in ASSETS
        }
    reproduced_prices, reproduced_actions, basis = normalize_yahoo_frames(
        hourly,
        daily,
        calendar,
        reference,
        observed_at=utc(meta["observed_at_utc"]),
        fallback_30min=fallback_30,
        fallback_daily=fallback_daily,
        fallback_15min=fallback_15,
    )
    from trend_following.research_oos_accounting import normalize_oos_actions

    try:
        pd.testing.assert_frame_equal(
            prices, reproduced_prices, check_freq=False, check_exact=True, check_names=False
        )
        pd.testing.assert_frame_equal(
            actions, normalize_oos_actions(reproduced_actions), check_exact=True
        )
    except AssertionError as error:
        raise ValueError("normalized input differs from preserved source replay") from error
    for key, value in basis.items():
        if provenance.get(key) != value:
            raise ValueError("normalized provenance differs from raw source replay")
    return {
        "verified": True,
        "source_vintage_at_utc": vintage.isoformat(),
        "source_authenticity_limit": "source bytes/retrieval attestations are not independent market-data notarization",
    }


def merge_packages(packages: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """First-vintage wins; preserve and report overlap revisions, never mutate inputs."""
    if not packages:
        raise ValueError("at least one input package required")
    panels, acts, records = [], [], []
    previous_observed = None
    differences = []
    first_prices: dict[pd.Timestamp, dict] = {}
    first_actions: dict[str, dict] = {}
    for package in packages:
        prices, actions, meta = read_input_package(package)
        observed = utc(meta.get("source_vintage_at_utc") or meta["observed_at_utc"])
        if previous_observed is not None and observed < previous_observed:
            raise ValueError(
                "packages must be ordered by actual retrieval vintage, not preparation time"
            )
        previous_observed = observed
        records.append(
            {
                "metadata_sha256": file_hash(package / "metadata.json"),
                "observed_at_utc": observed.isoformat(),
                "quality": meta["quality"],
            }
        )
        for stamp, row in prices.iterrows():
            values = row.to_dict()
            if stamp in first_prices:
                if values != first_prices[stamp]:
                    differences.append({"kind": "price_revision", "timestamp": stamp.isoformat()})
            else:
                first_prices[stamp] = values
        covered_start, covered_end = prices.index[0], prices.index[-1]
        incoming = actions.to_dict("records")
        incoming_ids = {a["action_id"] for a in incoming}
        for key, old in first_actions.items():
            if (
                covered_start.normalize()
                <= old["effective_at"].normalize()
                <= covered_end.normalize()
                and key not in incoming_ids
            ):
                raise ValueError(
                    "withdrawn historical corporate action revision; cannot silently rewrite an experiment"
                )
        for action in incoming:
            key = action["action_id"]
            if key in first_actions:
                economic = [
                    "asset",
                    "kind",
                    "effective_at",
                    "split_ratio",
                    "amount_per_share",
                    "pay_at",
                    "character",
                ]
                same = all(
                    (pd.isna(action[k]) and pd.isna(first_actions[key][k]))
                    or action[k] == first_actions[key][k]
                    for k in economic
                )
                if not same:
                    raise ValueError(
                        "revised historical corporate action; requires explicit failed-study/correction review"
                    )
            else:
                for prior in panels:
                    local_day = action["effective_at"].tz_convert("America/New_York").date()
                    if (
                        prior.index[0].tz_convert("America/New_York").date()
                        <= local_day
                        <= prior.index[-1].tz_convert("America/New_York").date()
                    ):
                        raise ValueError(
                            "late-added historical corporate action; cannot change frozen history"
                        )
                first_actions[key] = action
        panels.append(prices)
        acts.append(actions)
    merged = (
        pd.DataFrame.from_dict(first_prices, orient="index").sort_index().reindex(columns=ASSETS)
    )
    merged.index = pd.DatetimeIndex(merged.index)
    from trend_following.research_oos_accounting import normalize_oos_actions

    combined = normalize_oos_actions(list(first_actions.values()))
    return (
        validate_panel(merged),
        combined,
        {
            "packages": records,
            "revision_policy": "first_vintage_wins",
            "overlap_revisions": differences,
        },
    )
