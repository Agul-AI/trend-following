"""Fed hiking-cycle helpers for no-lookahead regime overlays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass(frozen=True)
class FedCycle:
    """A named Fed policy window."""

    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    source: str = ""
    tradability: str = "manual"


def load_cycle_config(path: str | Path) -> dict[str, list[FedCycle]]:
    """Load Fed cycle windows from a YAML config file."""
    with Path(path).open() as fh:
        raw = yaml.safe_load(fh) or {}
    groups = raw.get("fed_hiking_cycles", raw)
    result: dict[str, list[FedCycle]] = {}
    for group_name, rows in groups.items():
        cycles: list[FedCycle] = []
        for row in rows or []:
            cycles.append(
                FedCycle(
                    name=str(row["name"]),
                    start=pd.Timestamp(row["start"]).normalize(),
                    end=pd.Timestamp(row["end"]).normalize(),
                    source=str(row.get("source", "")),
                    tradability=str(row.get("tradability", group_name)),
                )
            )
        result[str(group_name)] = cycles
    return result


def cycle_flag(
    index: pd.DatetimeIndex,
    cycles: list[FedCycle],
    *,
    lag_days: int = 1,
    name: str = "fed_cycle_known",
) -> pd.Series:
    """Return an intraday flag for cycle status known before execution.

    The raw daily cycle label is shifted by ``lag_days`` observed trading dates
    before being mapped back to intraday bars.  This avoids using a same-day
    regime label that would not have been known before execution.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("cycle_flag requires a DatetimeIndex")
    if index.empty:
        return pd.Series(dtype=bool, name=name)

    normalized = pd.DatetimeIndex(index.normalize())
    unique_days = pd.DatetimeIndex(sorted(normalized.unique()))
    raw_daily = pd.Series(False, index=unique_days)
    for cycle in cycles:
        raw_daily |= (raw_daily.index >= cycle.start) & (raw_daily.index <= cycle.end)
    known_daily = raw_daily.shift(lag_days, fill_value=False)
    return pd.Series(known_daily.reindex(normalized).to_numpy(dtype=bool), index=index, name=name)


def monthly_pe_known(
    index: pd.DatetimeIndex,
    monthly_pe: pd.DataFrame,
    *,
    pe_column: str = "qqq_pe",
    lag_months: int = 1,
) -> pd.Series:
    """Map monthly QQQ P/E proxy to intraday bars with a month lag.

    With ``lag_months=1``, bars in June use the May monthly P/E value.  This is
    a conservative no-lookahead alignment for monthly valuation proxies.
    """
    if "month" not in monthly_pe.columns or pe_column not in monthly_pe.columns:
        raise ValueError("monthly_pe must contain 'month' and the requested pe_column")
    pe = monthly_pe[["month", pe_column]].copy()
    pe["month"] = pd.PeriodIndex(pe["month"].astype(str), freq="M")
    pe = pe.drop_duplicates("month", keep="last").set_index("month")[pe_column].astype(float)

    periods = pd.PeriodIndex(pd.DatetimeIndex(index).to_period("M"), freq="M") - lag_months
    values = pe.reindex(periods).to_numpy()
    return pd.Series(values, index=index, name=f"{pe_column}_known_lag_{lag_months}m")


def cycles_to_frame(cycle_groups: dict[str, list[FedCycle]]) -> pd.DataFrame:
    """Convert loaded cycles to a table."""
    rows: list[dict[str, Any]] = []
    for group, cycles in cycle_groups.items():
        for cycle in cycles:
            rows.append(
                {
                    "cycle_group": group,
                    "name": cycle.name,
                    "start": cycle.start.date().isoformat(),
                    "end": cycle.end.date().isoformat(),
                    "source": cycle.source,
                    "tradability": cycle.tradability,
                }
            )
    return pd.DataFrame(rows)
