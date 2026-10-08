"""Strict research price panels and explicit daily-reset reconstruction.

Vendor caches are retrospective, adjusted proxies. They are never relabeled as
unadjusted tax-lot histories or a point-in-time dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trend_following.data_validation import read_price_file


def load_research_price(path: str | Path) -> pd.Series:
    frame = read_price_file(path)
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError(f"nonunique/nonmonotonic prices: {path}")
    price = frame["adj_close"].astype(float)
    if price.isna().any() or not np.isfinite(price).all() or (price <= 0).any():
        raise ValueError(f"invalid prices: {path}")
    if price.index.tz is None:
        price.index = price.index.tz_localize("America/New_York")
    else:
        price.index = price.index.tz_convert("America/New_York")
    return price.rename(Path(path).stem)


def audit_sessions(price: pd.Series, *, bars_per_day: int = 6) -> pd.DataFrame:
    """Report partial/missing/extra sessions; no silent zero-return padding."""
    groups = price.groupby(price.index.normalize())
    rows = []
    for session, values in groups:
        rows.append(
            {
                "session": session,
                "bars": len(values),
                "first_bar": values.index[0],
                "last_bar": values.index[-1],
                "status": "full" if len(values) == bars_per_day else "partial_or_vendor_difference",
                "max_intraday_gap_minutes": values.index.to_series()
                .diff()
                .dt.total_seconds()
                .div(60)
                .max(),
            }
        )
    return pd.DataFrame(rows)


def synthetic_session_reset_price(
    qqq: pd.Series, *, leverage: float = 3.0, initial: float = 100.0
) -> pd.Series:
    """Recompute leverage on each observed session, not each hourly interval.

    Bootstrap convention: reset at the last observed QQQ bar of the preceding
    session. This is explicitly different from an actual fund's 16:00 close
    when a vendor cache ends at 15:00. It is not spliced into historical TQQQ.
    Nonpositive NAV raises rather than silently clipping bankruptcy away.
    """
    if qqq.empty or leverage <= 0 or initial <= 0:
        raise ValueError("positive nonempty input required")
    result = np.empty(len(qqq), dtype=float)
    days = qqq.index.normalize()
    anchor_q = float(qqq.iloc[0])
    anchor_nav = initial
    previous_day = days[0]
    values = qqq.to_numpy(dtype=float)
    for i, (day, value) in enumerate(zip(days, values, strict=True)):
        if day != previous_day:
            anchor_q = values[i - 1]
            anchor_nav = result[i - 1]
            previous_day = day
        result[i] = anchor_nav * (1 + leverage * (value / anchor_q - 1))
        if not np.isfinite(result[i]) or result[i] <= 0:
            raise ValueError(f"nonpositive synthetic NAV at {qqq.index[i]}")
    return pd.Series(result, index=qqq.index, name="synthetic_3x_session_reset")


def calendar_year_fraction(index: pd.DatetimeIndex) -> pd.Series:
    seconds = index.to_series().diff().dt.total_seconds().fillna(0)
    if (seconds < 0).any():
        raise ValueError("timestamps must be ordered")
    return (seconds / (365 * 86400)).rename("calendar_year_fraction")


def stationary_day_bootstrap(
    qqq: pd.Series, *, block_days: int, seed: int, paired_series: dict[str, pd.Series] | None = None
) -> tuple[pd.Series, dict[str, pd.Series], np.ndarray]:
    """Resample whole observed day vectors and preserve paired rate regimes.

    Source full and partial sessions are matched to target-length sessions, not
    padded/truncated. Selecting a block across a different session length
    forces a restart. Overnight interval returns remain the first day value.
    """
    if block_days < 1:
        raise ValueError("block_days must be positive")
    returns = qqq.pct_change().fillna(0)
    days = qqq.index.normalize()
    unique = days.unique()
    groups = [np.flatnonzero(days == day) for day in unique]
    by_length: dict[int, list[int]] = {}
    for i, g in enumerate(groups):
        if i:  # first observed day lacks its source overnight return
            by_length.setdefault(len(g), []).append(i)
    rng = np.random.default_rng(seed)
    samples = np.empty(len(groups), dtype=int)
    source = -1
    for i, target in enumerate(groups):
        advance = (source + 1) % len(groups)
        if (
            source < 0
            or rng.random() < 1 / block_days
            or advance == 0
            or len(groups[advance]) != len(target)
        ):
            source = int(rng.choice(by_length[len(target)]))
        else:
            source = advance
        samples[i] = source
    order = np.concatenate([groups[s] for s in samples])
    sampled_returns = returns.to_numpy()[order].copy()
    sampled_returns[0] = 0.0
    price = pd.Series(100 * np.cumprod(1 + sampled_returns), index=qqq.index, name="QQQ")
    paired = {
        k: pd.Series(v.reindex(qqq.index).to_numpy()[order], index=qqq.index, name=k)
        for k, v in (paired_series or {}).items()
    }
    if any(v.isna().any() for v in paired.values()):
        raise ValueError("paired series must cover all sampled timestamps")
    return price, paired, samples


def distribution_aware_panel(
    hourly_path: str | Path, daily_path: str | Path
) -> tuple[pd.Series, pd.Series, pd.DataFrame, dict]:
    """Reconstruct unadjusted hourly quotes from the vendor adjustment factor.

    Ex-date cash credit is an explicit hypothetical timing convention, not the
    actual dividend payment date. Unknown distributions are ordinary-income
    scenarios. Data lacking daily corporate actions is excluded and disclosed,
    never silently treated as no distributions. The daily close factor is a
    retrospective reconstruction, not a point-in-time feed.
    """
    hourly = load_research_price(hourly_path)
    daily = read_price_file(daily_path)
    required = {"close", "adj_close", "dividend_amount", "split_coefficient"}
    if not required.issubset(daily):
        raise ValueError("daily split/distribution data required")
    daily.index = (
        daily.index.tz_localize("America/New_York")
        if daily.index.tz is None
        else daily.index.tz_convert("America/New_York")
    ).normalize()
    if (
        daily[["close", "adj_close"]].le(0).any().any()
        or not np.isfinite(daily[["close", "adj_close"]].to_numpy()).all()
    ):
        raise ValueError("daily prices/adjustment factors must be finite and positive")
    valid = hourly.index.normalize().isin(daily.index)
    original_end = hourly.index[-1]
    covered = hourly.index[valid]
    if (
        len(covered)
        and ((~valid) & (hourly.index >= covered[0]) & (hourly.index <= covered[-1])).any()
    ):
        raise ValueError("missing interior daily corporate-action coverage")
    hourly = hourly.loc[valid]
    if hourly.empty:
        raise ValueError("no corporate-action-covered hourly observations")
    factors = (daily.adj_close / daily.close).reindex(hourly.index.normalize()).to_numpy()
    raw = pd.Series(hourly.to_numpy() / factors, index=hourly.index, name="risk")
    rows = []
    split_factors = pd.Series(1.0, index=raw.index)
    for session, values in raw.groupby(raw.index.normalize()):
        record = daily.loc[session]
        split = float(record.split_coefficient)
        dividend = float(record.dividend_amount)
        if not np.isfinite(split) or split <= 0 or not np.isfinite(dividend) or dividend < 0:
            raise ValueError("invalid corporate action")
        if split != 1 or dividend:
            stamp = values.index[0]
            rows.append(
                dict(
                    timestamp=stamp,
                    asset="risk",
                    split_ratio=split,
                    distribution_per_share=dividend,
                    tax_kind="ordinary",
                )
            )
            split_factors.loc[stamp] = split
    interval_returns = raw.pct_change().add(1).mul(split_factors).sub(1).fillna(0)
    actions = pd.DataFrame(
        rows, columns=["timestamp", "asset", "split_ratio", "distribution_per_share", "tax_kind"]
    )
    info = {
        "source": "vendor_daily_factor_reconstructed_unadjusted_quotes",
        "hourly_original_end": str(original_end),
        "corporate_action_covered_end": str(raw.index[-1]),
        "excluded_hourly_rows": int((~valid).sum()),
        "distribution_cash_credit": "first_bar_ex_date_not_actual_paydate",
        "tax_character": "ordinary_scenario_not_verified_distribution_qualification",
        "vintage": "retrospective_not_point_in_time",
    }
    return raw, interval_returns, actions, info
