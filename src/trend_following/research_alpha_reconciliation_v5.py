"""Alpha-only price-unit and quote-role reconciliation, never a backtest.

A provider's daily close and its last 16:00 intraday trade are not definitionally
identical. QQQ historically traded to16:15 on AMEX; exchange feeds also have
separate official, consolidated and last-sale closing values. Reconciliation
therefore distinguishes feature history from executable/marking quotes. It
never edits prices to force equality, loosens the old numeric tolerance, fills
missing bars, or calls reconstructed observations a native raw capture.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trend_following.research_hourly_data_v5 import (
    OHLCV,
    DataContractError,
    SourceMetadata,
    _provider_bars,
    _session_strings,
    audit_source_coverage,
    build_xnys_calendar,
    canonical_half_hour_grid,
    normalize_daily_actions,
    sha256_file,
    utc,
    validate_calendar,
    validate_cash_observations,
)

PRIMARY_SOURCES = {
    "alpha_price_basis": "https://www.alphavantage.co/documentation/",
    "qqq_listing_transfer": "https://ir.nasdaq.com/news-releases/news-release-details/qqq-list-nasdaq",
    "qqq_2004_broker_session_notice": "https://www.fidelity.com/products/stocksbonds/content/qqq.shtml",
    "separate_closing_trade_fields": "https://www.nasdaqtrader.com/content/technicalsupport/specifications/utp/utdfspecification.pdf",
    "closing_price_definition_filing": "https://listingcenter.nasdaq.com/assets/rulebook/nasdaq/filings/SR-NASDAQ-2014-076.pdf",
}
JSON_FIELDS = {
    "1. open": "open",
    "2. high": "high",
    "3. low": "low",
    "4. close": "close",
    "5. adjusted close": "adj_close",
    "6. volume": "volume",
    "7. dividend amount": "dividend_amount",
    "8. split coefficient": "split_coefficient",
}
DERIVED_PRICE_BASIS = "derived_as_traded_from_Alpha_adjustment_factors_NOT_native_raw_capture"


@dataclass(frozen=True)
class QuoteRolePolicy:
    """Explicit profile change; no implicit migration from frozen strict equality."""

    profile: str = "strict_daily_last_trade_identity"
    approval_reason: str = ""
    close_rtol: float = 0.001
    close_atol: float = 0.02
    quoted_decimal_places: int = 4
    minimum_anchor_days_per_month: int = 10

    def validate(self) -> None:
        if self.profile not in (
            "strict_daily_last_trade_identity",
            "distinct_daily_feature_and_rth_book_quotes",
        ):
            raise DataContractError("unknown quote-role reconciliation profile")
        if (
            self.profile == "distinct_daily_feature_and_rth_book_quotes"
            and not self.approval_reason.strip()
        ):
            raise DataContractError(
                "distinct quote roles require explicit source-evidenced approval_reason"
            )
        if (
            not np.isfinite([self.close_rtol, self.close_atol]).all()
            or min(self.close_rtol, self.close_atol) < 0
        ):
            raise DataContractError(
                "unchanged closing diagnostic tolerances must be finite and nonnegative"
            )
        if (
            isinstance(self.minimum_anchor_days_per_month, (bool, np.bool_))
            or not isinstance(self.minimum_anchor_days_per_month, (int, np.integer))
            or self.minimum_anchor_days_per_month < 10
        ):
            raise DataContractError(
                "source-unit proof requires at least10 independent anchor days per provider month"
            )
        if self.quoted_decimal_places != 4:
            raise DataContractError(
                "Alpha native cache precision proof requires declared four-decimal quotes"
            )


STRICT_POLICY = QuoteRolePolicy()
ROOT_APPROVED_ROLE_POLICY = QuoteRolePolicy(
    profile="distinct_daily_feature_and_rth_book_quotes",
    approval_reason="Source-evidenced explicit role revision approved before performance: provider daily quotes need not equal the last16:00 intraday trade; no price replacement or tolerance relaxation",
)


@dataclass
class AlphaReconciliationResult:
    """In-memory derived quotes; only metadata is persisted by the audit writer."""

    daily: pd.DataFrame
    reconstructed_30min: pd.DataFrame
    calendar: pd.DataFrame
    coverage: pd.DataFrame
    quote_relationships: pd.DataFrame
    observation_coverage: pd.DataFrame
    blackouts: pd.DataFrame
    audit: dict[str, Any]


def _daily(frame: pd.DataFrame) -> pd.DataFrame:
    needed = {"date", *JSON_FIELDS.values()}
    if frame.empty or not needed.issubset(frame):
        raise DataContractError(
            "Alpha daily cache requires full native OHLCV/adjusted-close/action fields"
        )
    out = frame.copy().reset_index(drop=True)
    out["session"] = _session_strings(out.date)
    if out.session.duplicated().any() or not out.session.is_monotonic_increasing:
        raise DataContractError("Alpha daily cache must be unique and chronologically ordered")
    for name in JSON_FIELDS.values():
        out[name] = pd.to_numeric(out[name], errors="coerce")
    if (
        not np.isfinite(out[list(JSON_FIELDS.values())].to_numpy()).all()
        or out[["open", "high", "low", "close", "adj_close", "split_coefficient"]].le(0).any().any()
        or out[["volume", "dividend_amount"]].lt(0).any().any()
    ):
        raise DataContractError("invalid Alpha daily quote/action values")
    if (
        out.high.lt(out[["open", "close", "low"]].max(axis=1)).any()
        or out.low.gt(out[["open", "close", "high"]].min(axis=1)).any()
    ):
        raise DataContractError("inconsistent Alpha daily OHLC envelope")
    return out


def verify_native_daily_json(daily: pd.DataFrame, native_json: dict[str, Any]) -> dict[str, Any]:
    """Verify cache serialization against an original SAME-provider response."""
    meta = native_json.get("Meta Data", {})
    if meta.get("2. Symbol") != "QQQ" or "Daily" not in meta.get("1. Information", ""):
        raise DataContractError("native Alpha daily JSON must identify QQQ daily data")
    series = native_json.get("Time Series (Daily)")
    if not isinstance(series, dict) or not series:
        raise DataContractError("native Alpha daily JSON time series missing")
    native = pd.DataFrame.from_dict(series, orient="index").rename(columns=JSON_FIELDS)
    native["session"] = _session_strings(native.index).to_numpy()
    if not set(JSON_FIELDS.values()).issubset(native):
        raise DataContractError("native Alpha JSON lacks daily quote/action fields")
    native = native.set_index("session").sort_index()
    cache = _daily(daily).set_index("session")
    if list(native.index) != list(cache.index):
        raise DataContractError(
            "native Alpha JSON and daily cache have different session coverage/vintages"
        )
    mismatch = {}
    for name in JSON_FIELDS.values():
        expected = pd.to_numeric(native[name], errors="coerce")
        match = np.isclose(cache[name], expected, rtol=1e-9, atol=1e-8)
        mismatch[name] = int((~match).sum())
    if any(mismatch.values()):
        raise DataContractError(f"native Alpha daily JSON/cache value mismatch: {mismatch}")
    if meta.get("3. Last Refreshed") != cache.index[-1]:
        raise DataContractError(
            "native Alpha daily refreshed date does not match cached terminal date"
        )
    return {
        "provider": "AlphaVantage",
        "symbol": "QQQ",
        "verified_sessions": len(cache),
        "verified_fields": list(JSON_FIELDS.values()),
        "mismatch_counts": mismatch,
        "last_refreshed": meta["3. Last Refreshed"],
        "point_in_time_vintage": False,
    }


def factor_action_proof(daily: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    """Observable inverse price factors must move only at reported actions.

    Diagnostic reinvestment-convention comparisons are NOT used to fit factors.
    The actually observed close/adjusted-close quotient is retained verbatim.
    """
    d = _daily(daily).set_index("session")
    factor = (d.close / d.adj_close).rename("raw_unit_multiplier")
    if not np.isfinite(factor).all() or factor.le(0).any():
        raise DataContractError("Alpha inverse price factors must be positive and finite")
    if not np.isclose(factor.iloc[-1], 1.0, rtol=1e-12, atol=1e-12):
        raise DataContractError(
            "latest native Alpha daily raw/adjusted closes are not the same terminal adjustment anchor"
        )
    ratio = factor.shift(1) / factor
    actions = d.dividend_amount.ne(0) | d.split_coefficient.ne(1)
    unexplained = ~actions & ratio.notna() & ~np.isclose(ratio, 1.0, rtol=1e-12, atol=1e-12)
    if unexplained.any():
        raise DataContractError(
            f"unexplained adjustment-factor changes without Alpha corporate actions: {list(d.index[unexplained])[:8]}"
        )
    split_only = d.split_coefficient.ne(1) & d.dividend_amount.eq(0) & ratio.notna()
    if not np.isclose(
        ratio[split_only], d.split_coefficient[split_only], rtol=1e-10, atol=1e-10
    ).all():
        raise DataContractError(
            "Alpha split-only price-factor jump does not match declared split ratio"
        )
    convention = d.split_coefficient * (1 + d.dividend_amount / d.close)
    action_mask = actions & ratio.notna()
    error = (ratio[action_mask] / convention[action_mask] - 1).abs()
    return factor, {
        "formula": "same_session_native_daily_close / native_daily_adjusted_close",
        "positive_finite": True,
        "action_transitions": int(action_mask.sum()),
        "nonaction_transitions": int((~actions & ratio.notna()).sum()),
        "unexplained_nonaction_factor_changes": 0,
        "split_only_transitions_exact": int(split_only.sum()),
        "cash_convention_diagnostic": "split_ratio * (1 + dividend_amount / current_ex_date_raw_close)",
        "cash_convention_max_relative_error": float(error.max()) if len(error) else 0.0,
        "cash_convention_not_used_to_fit_or_replace_factors": True,
        "normalization_is_retrospective_unit_reversal_not_a_signal_feature": True,
    }


def reconcile_alpha_frames(
    daily_frame: pd.DataFrame,
    adjusted_30min: pd.DataFrame,
    adjusted_15min: pd.DataFrame,
    native_daily_json: dict[str, Any],
    *,
    policy: QuoteRolePolicy = STRICT_POLICY,
    bar_start_session: str = "2001-01-02",
) -> AlphaReconciliationResult:
    """Source-validated unit restoration plus explicit book/feature quote roles.

    Returned quotes are derived—not a native adjusted=false capture. No missing
    observations are invented. The old strict closing-equality gate remains the
    default; a separately approved role profile is needed to remove that false
    identity assumption. Coverage/halts remain independent hard blockers.
    """
    policy.validate()
    d = _daily(daily_frame)
    native_proof = verify_native_daily_json(daily_frame, native_daily_json)
    factor, action_proof = factor_action_proof(daily_frame)
    source_meta = SourceMetadata(
        "AlphaVantage",
        "QQQ",
        30,
        "start",
        "America/New_York",
        "split_dividend_adjusted",
        True,
        pd.Timestamp.now(tz="UTC").isoformat(),
        "native cache produced with adjusted=true; reconstruction is explicitly derived",
    )
    thirty = _provider_bars(adjusted_30min, source_meta)
    fifteen = _provider_bars(
        adjusted_15min, SourceMetadata(**{**source_meta.__dict__, "interval_minutes": 15})
    )
    cutoff = min(
        d.session.iloc[-1],
        thirty.bar_start.iloc[-1].tz_convert("America/New_York").strftime("%Y-%m-%d"),
    )
    cal = build_xnys_calendar(d.session.iloc[0], cutoff)
    normalize_daily_actions(d, cal)  # Coverage/action validation; retain ALL native OHLC below.
    daily = d.loc[d.session.le(cutoff)].reset_index(drop=True).copy()
    daily["daily_price_available_at"] = pd.DatetimeIndex(
        [
            pd.Timestamp(session + " 17:00", tz="America/New_York").tz_convert("UTC")
            for session in daily.session
        ]
    )
    daily.attrs["source_metadata"] = {
        "provider": "AlphaVantage",
        "symbol": "QQQ",
        "price_basis": "raw_as_traded",
        "price_field": "provider_daily_close",
        "verification": "original native Alpha daily JSON agrees with every cached OHLCV/action field",
        "availability_policy": "17:00 America/New_York assumption; daily features prior completed sessions only",
    }
    thirty["session"] = thirty.bar_start.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    thirty = thirty.loc[thirty.session.le(cutoff)].reset_index(drop=True)
    if bar_start_session not in set(cal.session):
        raise DataContractError("reconciliation bar start must be a declared calendar session")
    future_actions = d.loc[d.session.gt(cutoff), ["dividend_amount", "split_coefficient"]]
    if future_actions.dividend_amount.ne(0).any() or future_actions.split_coefficient.ne(1).any():
        raise DataContractError(
            "different terminal adjustment vintages contain intervening actions; explicit intraday vintage evidence required"
        )
    indexed_cal = cal.set_index("session")
    inside = thirty.bar_start.ge(thirty.session.map(indexed_cal.market_open)) & thirty.bar_start.lt(
        thirty.session.map(indexed_cal.market_close)
    )
    rth = thirty.loc[inside].copy()
    native_precision = np.abs(
        rth[list(OHLCV[:4])].to_numpy() - rth[list(OHLCV[:4])].round(4).to_numpy()
    )
    if native_precision.max() > 1e-10:
        raise DataContractError(
            "native Alpha30 quote precision differs from declared four decimals"
        )
    grouped = (
        fifteen.assign(bucket=fifteen.bar_start.dt.floor("30min"))
        .groupby("bucket")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            count=("bar_start", "size"),
        )
    )
    pairs = grouped.loc[grouped["count"].eq(2)].join(
        thirty.set_index("bar_start")[list(OHLCV[:4])], how="inner", rsuffix="_30"
    )
    match = np.isclose(
        pairs[list(OHLCV[:4])], pairs[[f"{c}_30" for c in OHLCV[:4]]], rtol=1e-6, atol=1e-4
    ).all(axis=1)
    if not len(pairs) or not match.all():
        raise DataContractError(
            "same-provider Alpha15/30 quote windows disagree; adjustment vintages/alignment require review"
        )
    units = rth.session.map(factor)
    if units.isna().any():
        raise DataContractError("Alpha30 timestamps lack native daily unit-factor coverage")
    restored = rth.copy()
    restored[list(OHLCV[:4])] = restored[list(OHLCV[:4])].mul(units, axis=0)
    # Preserve reported volume; this field is NOT falsely relabelled raw physical
    # share volume. It is not used in the fixed indicators or 1bp execution model.
    restored["volume_reported"] = restored.volume
    restored.attrs["price_basis"] = DERIVED_PRICE_BASIS
    restored.attrs["volume_basis"] = "vendor_reported_adjusted_volume_not_verified_raw_share_volume"
    aggregate = restored.groupby("session").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last")
    )
    reference = d.set_index("session").reindex(aggregate.index)
    day_factor = factor.reindex(aggregate.index)
    precision_bound = 0.5 * 10 ** (-policy.quoted_decimal_places) * (day_factor + 1) + 1e-9
    independent = (
        aggregate[["open", "high", "low"]]
        .sub(reference[["open", "high", "low"]])
        .abs()
        .le(precision_bound, axis=0)
    )
    anchors = independent.any(axis=1)
    monthly = anchors.groupby(anchors.index.str.slice(0, 7)).agg(["size", "sum"])
    unanchored = monthly.index[monthly["sum"].lt(policy.minimum_anchor_days_per_month)].tolist()
    if unanchored:
        raise DataContractError(
            f"monthly Alpha30 adjustment-vintage/unit factors lack at least10 independent daily price-anchor days: {unanchored}"
        )
    close_match = np.isclose(
        aggregate.close, reference.close, rtol=policy.close_rtol, atol=policy.close_atol
    )
    relationships = pd.DataFrame(
        {
            "session": aggregate.index,
            "independent_unit_anchor": anchors.to_numpy(),
            "old_strict_close_match": close_match,
            "absolute_relative_close_difference": (aggregate.close / reference.close - 1)
            .abs()
            .to_numpy(),
            "listing_era": np.where(
                aggregate.index < "2004-12-01", "AMEX_pre_transfer", "Nasdaq_post_transfer"
            ),
        }
    )
    relationships["classification"] = np.where(
        close_match,
        "distinct_quote_roles_numerically_close",
        "different_daily_and_RTH_quote_roles_not_per_session_cause_proven",
    )
    relationships["per_session_cause_proven"] = False
    selected = relationships.loc[relationships.session.ge(bar_start_session)]
    source_calendar = cal.loc[cal.session.ge(thirty.session.iloc[0])].reset_index(drop=True)
    original_for_coverage = adjusted_30min.loc[
        pd.to_datetime(adjusted_30min.date).dt.strftime("%Y-%m-%d").le(cutoff)
    ]
    coverage = audit_source_coverage(original_for_coverage, source_calendar, source_meta)
    required_missing = coverage.loc[
        coverage.session.ge(bar_start_session) & coverage.issue.eq("missing_subbar")
    ]
    halted = required_missing.session.eq("2013-08-22")
    grid = canonical_half_hour_grid(source_calendar)
    observed_columns = restored.drop(
        columns=["bar_end", "duration_minutes", "is_full_hour"], errors="ignore"
    )
    restored = observed_columns.merge(
        grid[["bar_start", "bar_end", "duration_minutes", "is_full_hour"]],
        on="bar_start",
        how="left",
        validate="one_to_one",
    )
    if restored.bar_end.isna().any():
        raise DataContractError(
            "restored Alpha30 start labels are not on the frozen session-open30m grid"
        )
    full_coverage, blackouts = classify_observation_coverage(grid, restored)
    restored = restored.merge(
        full_coverage[["bar_start", "status", "signal_eligible", "fill_eligible"]],
        on="bar_start",
        how="left",
        validate="one_to_one",
    )
    restored.attrs["price_basis"] = DERIVED_PRICE_BASIS
    restored.attrs["volume_basis"] = "vendor_reported_adjusted_volume_not_verified_raw_share_volume"
    role_approved = policy.profile == "distinct_daily_feature_and_rth_book_quotes"
    audit = {
        "schema": "alpha_price_unit_quote_role_reconciliation_v1",
        "provider": "AlphaVantage",
        "symbol": "QQQ",
        "cutoff_session": cutoff,
        "bar_start_session": bar_start_session,
        "sampling_interval_minutes": 30,
        "daily_indicator_periods_changed": False,
        "performance_generated": False,
        "new_provider_calls": 0,
        "source_caches_modified": False,
        "native_raw_capture": False,
        "price_basis": DERIVED_PRICE_BASIS,
        "unit_mapping_validation": "source_convention_and_cache_precision_supported_not_independent_raw_feed_verification",
        "quote_role_profile": policy.profile,
        "quote_role_approval_reason": policy.approval_reason,
        "daily_price_role": "Alpha_provider_daily_raw_close_for_finalized_daily_feature_history",
        "daily_available_for_use": "conservative17:00NY policy assumption; indicator history strictly prior completed sessions; not contemporaneous publication proof",
        "book_price_role": "Alpha30_factor_restored_RTH_trade_close_and_next_bar_trade_open",
        "last_16_00_book_close_forced_to_daily_provider_close": False,
        "physical_split_dividend_model_changed": False,
        "volume_basis": restored.attrs["volume_basis"],
        "volume_used_by_frozen_rules_or_fixed_cost_model": False,
        "native_daily_json_proof": native_proof,
        "daily_action_factor_proof": action_proof,
        "same_provider_15_30_alignment": {
            "windows": len(pairs),
            "all_ohlc_matches": int(match.sum()),
        },
        "independent_price_anchor_proof": {
            "sessions": len(anchors),
            "anchored_sessions": int(anchors.sum()),
            "months": len(monthly),
            "unanchored_months": unanchored,
            "minimum_required_anchor_days_per_month": policy.minimum_anchor_days_per_month,
            "minimum_anchor_days_per_month": int(monthly["sum"].min()),
            "precision_bound_formula": "0.5 * 10^-4 * (daily_inverse_adjustment_factor + 1) + 1e-9",
            "daily_close_not_used_as_an_anchor": True,
        },
        "old_identity_diagnostic": {
            "rtol": policy.close_rtol,
            "atol": policy.close_atol,
            "all_observed_sessions": len(relationships),
            "all_mismatches": int((~close_match).sum()),
            "required_observed_sessions": len(selected),
            "required_mismatches": int((~selected.old_strict_close_match).sum()),
            "required_mismatch_with_independent_anchor": int(
                (~selected.old_strict_close_match & selected.independent_unit_anchor).sum()
            ),
            "mismatch_is_corrupt_quote_or_bad_factor_proof": False,
            "strict_identity_gate_still_required": not role_approved,
        },
        "coverage": {
            "required_missing_expected_timestamps": len(required_missing),
            "required_missing_by_session": required_missing.groupby("session").size().to_dict(),
            "known_halt_consistent_missing": int(halted.sum()),
            "unresolved_nonhalt_missing": int((~halted).sum()),
            "out_of_RTH_rows_excluded_not_repaired": int((~inside).sum()),
            "holes_or_halt_prices_fabricated": False,
        },
        "quote_relationship_reconciled_by_explicit_roles": role_approved,
        "unit_mapping_proof_passed": True,
        "market_backtest_ready": False,
        "remaining_blocks": [
            "no missing-session/opening-bar or halt execution policy is implicitly waived",
            "derived reconstruction must be explicitly accepted and hash-bound instead of labelled native raw capture",
            "vendor-reported volume is not verified physical-share volume; must remain unused or separately reconciled",
        ],
        "primary_sources": PRIMARY_SOURCES,
        "interpretation": "documented timing/trade-condition differences invalidate universal daily-close/last-RTH-trade equality; exact causes of individual residuals are not proven",
    }
    if not role_approved and (~selected.old_strict_close_match).any():
        audit["remaining_blocks"].append(
            "old strict identity profile remains active; explicit quote-role profile approval is required"
        )
    audit["data_bindings"] = {
        "daily_session_hashes": _session_hashes(daily, DAILY_BINDING_COLUMNS),
        "bar_session_hashes": _session_hashes(restored, BAR_BINDING_COLUMNS),
        "calendar_session_hashes": _session_hashes(cal, CALENDAR_BINDING_COLUMNS),
        "coverage_session_hashes": _session_hashes(full_coverage, COVERAGE_BINDING_COLUMNS),
        "blackout_row_hashes": [
            _row_hash(row, BLACKOUT_BINDING_COLUMNS) for _, row in blackouts.iterrows()
        ],
    }
    return AlphaReconciliationResult(
        daily, restored, cal, coverage, relationships, full_coverage, blackouts, audit
    )


def reconcile_alpha_cache(
    source_root: str | Path,
    *,
    policy: QuoteRolePolicy = STRICT_POLICY,
    bar_start_session: str = "2001-01-02",
    output_dir: str | Path | None = None,
) -> AlphaReconciliationResult:
    """Read only original Alpha files; save metadata/relationship flags, not quotes."""
    root = Path(source_root) / "data/raw"
    paths = {
        "daily": root / "alpha_vantage_daily_adjusted/QQQ.parquet",
        "30min": root / "alpha_vantage_30min/QQQ.parquet",
        "15min": root / "alpha_vantage_15min/QQQ.parquet",
        "native_daily_json": root / "valuation/alpha_vantage_daily_adjusted_QQQ.json",
    }
    if output_dir is not None and Path(output_dir).exists():
        raise FileExistsError("immutable Alpha reconciliation audit already exists")
    if any(not path.is_file() or path.is_symlink() for path in paths.values()):
        raise DataContractError(
            "original nonsymlink Alpha daily/30/15/native-JSON source files required"
        )
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    result = reconcile_alpha_frames(
        pd.read_parquet(paths["daily"]),
        pd.read_parquet(paths["30min"]),
        pd.read_parquet(paths["15min"]),
        json.loads(paths["native_daily_json"].read_text()),
        policy=policy,
        bar_start_session=bar_start_session,
    )
    result.audit["source_files"] = [
        {
            "role": name,
            "provider": "AlphaVantage",
            "path": str(path.resolve()),
            "sha256": hashes[name],
        }
        for name, path in paths.items()
    ]
    if any(sha256_file(path) != hashes[name] for name, path in paths.items()):
        raise DataContractError("Alpha source file changed while reconciliation was running")
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=False)
        (output / "reconciliation.json").write_text(
            json.dumps(result.audit, indent=2, sort_keys=True) + "\n"
        )
        # Only relative differences, booleans, dates and classifications; no OHLC prices.
        result.quote_relationships.to_csv(output / "quote_relationship_flags.csv", index=False)
        result.coverage.to_csv(output / "coverage_flags.csv", index=False)
        result.observation_coverage.to_csv(output / "observation_coverage.csv", index=False)
        result.blackouts.to_csv(output / "blackouts.csv", index=False)
    return result


def classify_observation_coverage(
    grid: pd.DataFrame, observed: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """No prices here: explicit safe flags and precise blackout boundaries.

    The12:00 open precedes the12:23 halt and may execute an already-confirmed
    order. Its overlapping bar cannot confirm. A15:00 label cannot backdate a
    quote whose first post-halt trade was15:25. First fully safe post-halt bar
    is15:30–16:00. Missing observations never receive synthetic OHLC values.
    """
    required = {"session", "bar_start", "bar_end"}
    if not required.issubset(grid) or not {"bar_start", "bar_end"}.issubset(observed):
        raise DataContractError("coverage classification requires aware start/end metadata")
    out = grid[["session", "bar_start", "bar_end"]].copy().reset_index(drop=True)
    exists = out.bar_start.isin(observed.bar_start)
    hs = pd.Timestamp("2013-08-22 12:23", tz="America/New_York").tz_convert("UTC")
    he = pd.Timestamp("2013-08-22 15:25", tz="America/New_York").tz_convert("UTC")
    overlap = out.bar_start.lt(he) & out.bar_end.gt(hs)
    open_in_halt = out.bar_start.ge(hs) & out.bar_start.lt(he)
    out["status"] = np.where(
        overlap, "known_halt_overlap", np.where(exists, "observed", "unknown_missing")
    )
    out["signal_eligible"] = exists & ~overlap
    out["fill_eligible"] = exists & ~open_in_halt
    rows = []
    if overlap.any():
        rows.append(
            {
                "start": hs,
                "end": he,
                "kind": "known_halt",
                "reason": "Nasdaq regulatory halt allTapeC; exact halt12:23/resumption15:25ET",
                "source_url": "https://www.nasdaqtrader.com/TraderNews.aspx?id=ETA2013-77",
            }
        )
    unknown = out.loc[out.status.eq("unknown_missing")]
    active = None
    for row in unknown.itertuples():
        if active is not None and row.bar_start == active["end"]:
            active["end"] = row.bar_end
        else:
            active = {
                "start": row.bar_start,
                "end": row.bar_end,
                "kind": "unknown_missing",
                "reason": "Alpha source contains no observation; no signal, fill, or synthetic quote",
                "source_url": "",
            }
            rows.append(active)
    blackouts = pd.DataFrame(rows, columns=["start", "end", "kind", "reason", "source_url"])
    if not blackouts.empty:
        blackouts = blackouts.sort_values("start").reset_index(drop=True)
    return out, blackouts


def reconstruct_alpha_raw_observations(
    daily_path: str | Path,
    intraday_30min_path: str | Path,
    intraday_15min_path: str | Path,
    daily_json_path: str | Path,
    *,
    policy: QuoteRolePolicy = ROOT_APPROVED_ROLE_POLICY,
    bar_start_session: str = "2001-01-02",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return observed physical-price-unit30m quotes plus source-unit proof.

    This name means reconstructed as-traded PRICE units, not native raw capture.
    It does not write quotes, prepare a packet, waive missing intervals, or run
    accounting/performance. For daily/coverage/blackout tables use the richer
    reconcile_alpha_cache/reconcile_alpha_frames result with the same policy.
    """
    paths = {
        "daily": Path(daily_path),
        "30min": Path(intraday_30min_path),
        "15min": Path(intraday_15min_path),
        "native_daily_json": Path(daily_json_path),
    }
    if any(not p.is_file() or p.is_symlink() for p in paths.values()):
        raise DataContractError("original nonsymlink Alpha source paths required")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    result = reconcile_alpha_frames(
        pd.read_parquet(paths["daily"]),
        pd.read_parquet(paths["30min"]),
        pd.read_parquet(paths["15min"]),
        json.loads(paths["native_daily_json"].read_text()),
        policy=policy,
        bar_start_session=bar_start_session,
    )
    result.audit["source_files"] = [
        {
            "role": name,
            "provider": "AlphaVantage",
            "path": str(path.resolve()),
            "sha256": hashes[name],
        }
        for name, path in paths.items()
    ]
    if any(sha256_file(path) != hashes[name] for name, path in paths.items()):
        raise DataContractError("source changed during Alpha unit reconstruction")
    result.reconstructed_30min.attrs["reconciliation_audit"] = result.audit
    return result.reconstructed_30min, result.audit


DAILY_BINDING_COLUMNS = (
    "session",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "dividend_amount",
    "split_coefficient",
    "daily_price_available_at",
)
BAR_BINDING_COLUMNS = (
    "session",
    "bar_start",
    "bar_end",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "duration_minutes",
    "is_full_hour",
    "status",
    "signal_eligible",
    "fill_eligible",
)
CALENDAR_BINDING_COLUMNS = ("session", "market_open", "market_close", "is_early_close")
COVERAGE_BINDING_COLUMNS = (
    "session",
    "bar_start",
    "bar_end",
    "status",
    "signal_eligible",
    "fill_eligible",
)
BLACKOUT_BINDING_COLUMNS = ("start", "end", "kind", "reason", "source_url")
QUALITY_MODEL = {
    "sampling_interval_minutes": 30,
    "observation_interval_minutes": 30,
    "confirmation_checks_formula": "1+2*wait_hours",
    "wait_hours_grid": [i / 2 for i in range(14)],
    "all_completed_half_hours_eligible_except_explicit_blackout_flags": True,
    "daily_indicator_periods_unchanged": True,
    "daily_price_role": "provider_daily_close",
    "daily_price_available_clock_NY": "17:00",
    "daily_features_use_prior_completed_sessions_only": True,
    "daily_valuation_clock_NY": "17:00",
    "terminal_execution_role": "last_verified_RTH_trade_close",
    "physical_split_dividend_accounting": True,
    "cash_dividend_Drip_only_safe_observed_open": True,
    "source_price_origin": DERIVED_PRICE_BASIS,
    "native_raw_capture": False,
    "security_price_provider": "AlphaVantage",
    "symbol": "QQQ",
    "benchmark_metrics_end_clock_NY": "17:00",
}


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _canonical_scalar(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return {"float64_hex": float(value).hex()}
    if isinstance(value, pd.Timestamp):
        return utc(value).isoformat() if value.tzinfo is not None else value.isoformat()
    return str(value)


def _row_hash(row: pd.Series, columns: tuple[str, ...]) -> str:
    return _json_hash({column: _canonical_scalar(row[column]) for column in columns})


def _session_hashes(frame: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, str]:
    if not set(columns).issubset(frame):
        raise DataContractError(
            f"source binding frame lacks required fields: {sorted(set(columns) - set(frame))}"
        )
    # pandas propagates attrs through group frames and every iterrows Series.
    # A full master unit proof in attrs would therefore be deep-copied per row.
    # A fresh public DataFrame keeps the identical data/dtypes without metadata;
    # the caller's attrs and backing values remain untouched.
    numeric_frame = pd.DataFrame(frame, copy=False)
    return {
        str(session): _json_hash([_row_hash(row, columns) for _, row in part.iterrows()])
        for session, part in numeric_frame.groupby("session", sort=False)
    }


@dataclass
class ReconciledBundle:
    daily: pd.DataFrame
    bars: pd.DataFrame
    calendar: pd.DataFrame
    coverage: pd.DataFrame
    blackouts: pd.DataFrame
    unit_proof: dict[str, Any]
    source_metadata: dict[str, Any]
    source_files: list[dict[str, Any]]


def build_reconciled_alpha_bundle(
    source_root: str | Path,
    *,
    bar_start_session: str = "2001-01-02",
    cutoff_session: str = "2026-05-28",
    policy: QuoteRolePolicy = ROOT_APPROVED_ROLE_POLICY,
) -> ReconciledBundle:
    """Build audited classified data, not a simulation or missing-bar waiver."""
    result = reconcile_alpha_cache(source_root, policy=policy, bar_start_session=bar_start_session)
    if result.audit["cutoff_session"] != cutoff_session:
        raise DataContractError(
            "native Alpha common source endpoint differs from explicitly declared cutoff"
        )
    if not result.audit["quote_relationship_reconciled_by_explicit_roles"]:
        raise DataContractError(
            "quality packet requires explicitly approved distinct daily/book quote roles"
        )
    metadata = {
        "provider": "AlphaVantage",
        "symbol": "QQQ",
        "price_basis": DERIVED_PRICE_BASIS,
        "price_origin": DERIVED_PRICE_BASIS,
        "native_raw_capture": False,
        "adjusted_source": True,
        "normalized_physical_price_units": True,
        "volume_basis": result.audit["volume_basis"],
        "quote_role_profile": policy.profile,
        "coverage_start_session": result.observation_coverage.session.iloc[0],
        "required_study_start_session": bar_start_session,
        "coverage_end_session": cutoff_session,
        "availability": "retrospective_latest_cache_not_point_in_time_evidence",
    }
    return ReconciledBundle(
        result.daily,
        result.reconstructed_30min,
        result.calendar,
        result.observation_coverage,
        result.blackouts,
        result.audit,
        metadata,
        result.audit["source_files"],
    )


def _proof_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DataContractError(f"{name} must be a proof mapping")
    return value


def _proof_integer(value: Any, name: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    # Integers are finite by construction. Never coerce floats, NaN, strings or
    # booleans into proof counts; Python bool is otherwise a subclass of int.
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise DataContractError(f"{name} must be a finite non-bool integer")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        raise DataContractError(f"{name} is outside consistent proof-count bounds")
    return result


def _proof_number(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise DataContractError(f"{name} must be finite nonnegative non-bool numeric data")
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError):
        raise DataContractError(f"{name} must be finite nonnegative numeric data") from None
    if not np.isfinite(converted) or converted < 0:
        raise DataContractError(f"{name} must be finite nonnegative non-bool numeric data")
    return converted


def _validate_unit_proof(proof: dict[str, Any]) -> None:
    proof = _proof_mapping(proof, "unit proof")
    if (
        proof.get("schema") != "alpha_price_unit_quote_role_reconciliation_v1"
        or proof.get("provider") != "AlphaVantage"
        or proof.get("symbol") != "QQQ"
        or proof.get("price_basis") != DERIVED_PRICE_BASIS
        or proof.get("native_raw_capture") is not False
        or proof.get("unit_mapping_proof_passed") is not True
        or proof.get("quote_relationship_reconciled_by_explicit_roles") is not True
        or proof.get("quote_role_profile") != "distinct_daily_feature_and_rth_book_quotes"
    ):
        raise DataContractError(
            "unit proof must attest tested Alpha-only derived units and explicit quote roles"
        )
    native = _proof_mapping(proof.get("native_daily_json_proof"), "native daily JSON proof")
    native_sessions = _proof_integer(
        native.get("verified_sessions"), "native verified_sessions", minimum=1
    )
    mismatches = _proof_mapping(native.get("mismatch_counts"), "native mismatch counts")
    fields = native.get("verified_fields")
    if (
        not isinstance(fields, list)
        or len(fields) != len(JSON_FIELDS)
        or not all(isinstance(field, str) for field in fields)
        or set(mismatches) != set(JSON_FIELDS.values())
        or set(fields) != set(JSON_FIELDS.values())
    ):
        raise DataContractError(
            "native daily JSON proof must verify every native OHLCV/action field"
        )
    for field, count in mismatches.items():
        if _proof_integer(count, "native mismatch_count " + field, maximum=native_sessions) != 0:
            raise DataContractError("native daily JSON proof has mismatches")
    factor = _proof_mapping(proof.get("daily_action_factor_proof"), "action factor proof")
    if factor.get("positive_finite") is not True:
        raise DataContractError("unit factor proof must establish positive finite multipliers")
    if (
        _proof_integer(
            factor.get("unexplained_nonaction_factor_changes"), "unexplained factor changes"
        )
        != 0
    ):
        raise DataContractError("unexplained adjustment factor changes")
    action_count = _proof_integer(
        factor.get("action_transitions"), "action transitions", maximum=native_sessions - 1
    )
    nonaction_count = _proof_integer(
        factor.get("nonaction_transitions"), "nonaction transitions", maximum=native_sessions - 1
    )
    if action_count + nonaction_count != native_sessions - 1:
        raise DataContractError(
            "action/nonaction counts do not cover every native daily transition"
        )
    _proof_integer(
        factor.get("split_only_transitions_exact"), "split-only transitions", maximum=action_count
    )
    _proof_number(factor.get("cash_convention_max_relative_error"), "cash convention error")
    alignment = _proof_mapping(proof.get("same_provider_15_30_alignment"), "15/30 alignment proof")
    windows = _proof_integer(alignment.get("windows"), "15/30 window count", minimum=1)
    if (
        _proof_integer(alignment.get("all_ohlc_matches"), "15/30 match count", maximum=windows)
        != windows
    ):
        raise DataContractError("unit proof lacks complete same-provider15/30 window alignment")
    anchors = _proof_mapping(
        proof.get("independent_price_anchor_proof"), "independent price anchor proof"
    )
    required = _proof_integer(
        anchors.get("minimum_required_anchor_days_per_month"),
        "minimum required anchor days",
        minimum=10,
    )
    observed_min = _proof_integer(
        anchors.get("minimum_anchor_days_per_month"),
        "minimum observed anchor days",
        minimum=required,
    )
    sessions = _proof_integer(
        anchors.get("sessions"), "anchor sessions", minimum=1, maximum=native_sessions
    )
    anchored = _proof_integer(
        anchors.get("anchored_sessions"), "anchored sessions", minimum=1, maximum=sessions
    )
    months = _proof_integer(anchors.get("months"), "anchor months", minimum=1, maximum=sessions)
    if (
        anchors.get("unanchored_months") != []
        or anchors.get("daily_close_not_used_as_an_anchor") is not True
        or months * observed_min > anchored
    ):
        raise DataContractError(
            "unit proof fails consistent minimum10 nonclosing anchor days in every native month"
        )
    diag = _proof_mapping(proof.get("old_identity_diagnostic"), "closing role diagnostic")
    if (
        _proof_integer(diag.get("all_observed_sessions"), "all observed diagnostic sessions")
        != sessions
    ):
        raise DataContractError("anchor and closing diagnostic session counts differ")
    _proof_integer(diag.get("all_mismatches"), "all closing mismatches", maximum=sessions)
    required_sessions = _proof_integer(
        diag.get("required_observed_sessions"),
        "required diagnostic sessions",
        minimum=1,
        maximum=sessions,
    )
    required_mismatches = _proof_integer(
        diag.get("required_mismatches"), "required closing mismatches", maximum=required_sessions
    )
    _proof_integer(
        diag.get("required_mismatch_with_independent_anchor"),
        "required anchored mismatches",
        maximum=min(required_mismatches, anchored),
    )
    _proof_number(diag.get("rtol"), "closing diagnostic rtol")
    _proof_number(diag.get("atol"), "closing diagnostic atol")
    coverage = _proof_mapping(proof.get("coverage"), "source coverage proof")
    total_missing = _proof_integer(
        coverage.get("required_missing_expected_timestamps"), "required missing count"
    )
    known_missing = _proof_integer(
        coverage.get("known_halt_consistent_missing"),
        "known-halt missing count",
        maximum=total_missing,
    )
    unknown_missing = _proof_integer(
        coverage.get("unresolved_nonhalt_missing"), "unknown missing count", maximum=total_missing
    )
    by_session = _proof_mapping(
        coverage.get("required_missing_by_session"), "missing count by session"
    )
    if (
        sum(_proof_integer(v, "missing session count") for v in by_session.values())
        != total_missing
        or known_missing + unknown_missing != total_missing
    ):
        raise DataContractError("known/unknown/per-session missing counts are inconsistent")
    _proof_integer(
        coverage.get("out_of_RTH_rows_excluded_not_repaired"), "excluded out-of-RTH rows"
    )
    bindings = _proof_mapping(proof.get("data_bindings"), "numeric prefix data bindings")
    for name in (
        "daily_session_hashes",
        "bar_session_hashes",
        "calendar_session_hashes",
        "coverage_session_hashes",
    ):
        values = _proof_mapping(bindings.get(name), name)
        if not values:
            raise DataContractError("unit proof lacks source-normalized numeric-prefix bindings")
        for value in values.values():
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise DataContractError("invalid source-normalized numeric-prefix hash")
    if not isinstance(bindings.get("blackout_row_hashes"), list):
        raise DataContractError("blackout source bindings must be a list")


def _implementation_hashes() -> dict[str, str]:
    here = Path(__file__).parent
    names = (
        "research_alpha_reconciliation_v5.py",
        "research_hourly_data_v5.py",
        "research_daily_hourly_v5.py",
        "research_quality_engine_v5.py",
    )
    if any(not (here / name).is_file() or (here / name).is_symlink() for name in names):
        raise DataContractError("quality/counter implementation source files unavailable")
    return {name: sha256_file(here / name) for name in names}


def _validate_bundle(bundle: ReconciledBundle) -> dict[str, Any]:
    _validate_unit_proof(bundle.unit_proof)
    meta = bundle.source_metadata
    if (
        meta.get("provider") != "AlphaVantage"
        or meta.get("symbol") != "QQQ"
        or meta.get("price_origin") != DERIVED_PRICE_BASIS
        or meta.get("native_raw_capture") is not False
        or meta.get("price_basis") != DERIVED_PRICE_BASIS
        or meta.get("adjusted_source") is not True
        or meta.get("normalized_physical_price_units") is not True
    ):
        raise DataContractError(
            "source metadata must explicitly identify Alpha-only derived-not-native prices"
        )
    if bundle.unit_proof.get("source_files") != bundle.source_files:
        raise DataContractError(
            "native source custody differs from the independently generated master unit proof"
        )
    expected_roles = {"daily", "30min", "15min", "native_daily_json"}
    if {row.get("role") for row in bundle.source_files} != expected_roles:
        raise DataContractError(
            "source custody requires daily parquet,30m,15m and original daily JSON"
        )
    for row in bundle.source_files:
        path = Path(row["path"])
        if (
            row.get("provider") != "AlphaVantage"
            or not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != row.get("sha256")
        ):
            raise DataContractError("Alpha native source file missing, changed or unsafe")
    cal = validate_calendar(bundle.calendar)
    d = _daily(bundle.daily)
    if list(d.session) != list(cal.session):
        raise DataContractError("quality daily raw history must exactly cover full calendar prefix")
    if "daily_price_available_at" not in bundle.daily:
        raise DataContractError("daily provider quote lacks explicit conservative17NY availability")
    available = pd.DatetimeIndex([utc(value) for value in bundle.daily.daily_price_available_at])
    expected_available = pd.DatetimeIndex(
        [
            pd.Timestamp(session + " 17:00", tz="America/New_York").tz_convert("UTC")
            for session in d.session
        ]
    )
    if not available.equals(expected_available):
        raise DataContractError("daily price availability must match frozen17NY assumption")
    bars = bundle.bars
    if bars.empty or not set(BAR_BINDING_COLUMNS).issubset(bars):
        raise DataContractError(
            "quality bars require observed30m OHLC with bound eligibility flags"
        )
    if (
        bars.bar_start.duplicated().any()
        or not bars.bar_start.is_monotonic_increasing
        or not bars.duration_minutes.eq(30).all()
        or bars.is_full_hour.any()
    ):
        raise DataContractError("quality book quotes must be sorted unique exact30m observations")
    _provider_bars(
        bars,
        SourceMetadata(
            "AlphaVantage",
            "QQQ",
            30,
            "start",
            "UTC",
            DERIVED_PRICE_BASIS,
            False,
            pd.Timestamp.now(tz="UTC").isoformat(),
            "derived source structural validation",
        ),
    )
    declared_start = meta.get("coverage_start_session")
    if (
        declared_start not in set(cal.session)
        or meta.get("coverage_end_session") != cal.session.iloc[-1]
    ):
        raise DataContractError(
            "quality calendar prefix does not match explicit source coverage boundaries"
        )
    grid = canonical_half_hour_grid(cal.loc[cal.session.ge(declared_start)].reset_index(drop=True))
    if (
        not bundle.coverage[["session", "bar_start", "bar_end"]]
        .reset_index(drop=True)
        .equals(grid[["session", "bar_start", "bar_end"]])
    ):
        raise DataContractError(
            "quality coverage must retain every scheduled30m slot including absent source intervals"
        )
    calculated, blackouts = classify_observation_coverage(grid, bars)
    if not calculated[list(COVERAGE_BINDING_COLUMNS)].equals(
        bundle.coverage[list(COVERAGE_BINDING_COLUMNS)].reset_index(drop=True)
    ):
        raise DataContractError(
            "quality coverage flags disagree with source presence/precise known halt"
        )
    calculated_blackout_hashes = [
        _row_hash(row, BLACKOUT_BINDING_COLUMNS) for _, row in blackouts.iterrows()
    ]
    actual_blackout_hashes = [
        _row_hash(row, BLACKOUT_BINDING_COLUMNS) for _, row in bundle.blackouts.iterrows()
    ]
    if calculated_blackout_hashes != actual_blackout_hashes:
        raise DataContractError(
            "quality blackout table disagrees with missing observations/known halt"
        )
    matched = bars.merge(
        calculated[["bar_start", "status", "signal_eligible", "fill_eligible"]],
        on="bar_start",
        suffixes=("", "_expected"),
        validate="one_to_one",
    )
    for field in ("status", "signal_eligible", "fill_eligible"):
        if not matched[field].equals(matched[field + "_expected"]):
            raise DataContractError(
                "observed book eligibility flags differ from frozen coverage flags"
            )
    bind = bundle.unit_proof["data_bindings"]
    for key, frame, columns in (
        ("daily_session_hashes", bundle.daily, DAILY_BINDING_COLUMNS),
        ("bar_session_hashes", bars, BAR_BINDING_COLUMNS),
        ("calendar_session_hashes", cal, CALENDAR_BINDING_COLUMNS),
        ("coverage_session_hashes", bundle.coverage, COVERAGE_BINDING_COLUMNS),
    ):
        actual = _session_hashes(frame, columns)
        if any(bind.get(key, {}).get(session) != value for session, value in actual.items()):
            raise DataContractError(
                f"numeric prefix changed from independently audited source-unit proof: {key}"
            )
    if any(
        _row_hash(row, BLACKOUT_BINDING_COLUMNS) not in bind.get("blackout_row_hashes", [])
        for _, row in bundle.blackouts.iterrows()
    ):
        raise DataContractError("blackout prefix changed from master unit proof")
    required = str(
        meta.get("required_study_start_session", bundle.unit_proof.get("bar_start_session"))
    )
    active = calculated.loc[calculated.session.ge(required)]
    return {
        "rows_observed": len(bars),
        "rows_scheduled": len(calculated),
        # Strict engine validates the entire supplied source prefix, not just
        # scored/development dates. Keep the narrower count diagnostic-only.
        "unknown_missing_count": int(calculated.status.eq("unknown_missing").sum()),
        "unknown_missing_gate_scope": "all_supplied_scheduled_coverage",
        "required_window_unknown_missing_count": int(active.status.eq("unknown_missing").sum()),
        "known_halt_overlap_count": int(active.status.eq("known_halt_overlap").sum()),
        "signal_eligible_count": int(active.signal_eligible.sum()),
        "fill_eligible_count": int(active.fill_eligible.sum()),
        "prefix_end_session": cal.session.iloc[-1],
        "sampling_interval_minutes": 30,
    }


def _gap_authorization(profile: str, authorization: dict[str, Any] | None) -> bool:
    if profile not in ("strict_tradable_v1", "observed_only_blackout_v1"):
        raise DataContractError("unknown quality execution profile")
    if profile == "strict_tradable_v1":
        if authorization is not None:
            raise DataContractError(
                "strict profile cannot silently apply unknown-gap authorization"
            )
        return False
    if (
        not isinstance(authorization, dict)
        or authorization.get("explicit_user_approval") is not True
        or authorization.get("quality_profile") != profile
        or not str(authorization.get("instruction", "")).strip()
        or not authorization.get("approved_at")
    ):
        raise DataContractError(
            "observed-only blackouts require explicit recorded human gap-policy approval"
        )
    utc(authorization["approved_at"], "gap policy approved_at")
    return True


def _validate_reconciled_cash(
    frame: pd.DataFrame, *, as_of: Any, first_bar_start: Any
) -> pd.DataFrame:
    if frame.empty or "series_id" not in frame or not frame.series_id.eq("DTB3").all():
        raise DataContractError(
            "reconciled cash observations require nonempty DTB3 series_id for every row"
        )
    return validate_cash_observations(frame, as_of=as_of, first_bar_start=first_bar_start)


def write_reconciled_packet(
    output_dir: str | Path,
    bundle: ReconciledBundle,
    *,
    cash_observations: pd.DataFrame | None = None,
    cash_source_file: str | Path | None = None,
    quality_profile: str = "strict_tradable_v1",
    gap_authorization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save an immutable classified packet, even when strict readiness is false.

    This is not a permission to backtest missing intervals. Strict profile fails
    closed; observed-only policy requires separately recorded explicit approval.
    """
    target = Path(output_dir)
    if target.exists() or target.is_symlink():
        raise FileExistsError("immutable reconciled input packet already exists")
    validation = _validate_bundle(bundle)
    approved = _gap_authorization(quality_profile, gap_authorization)
    cash = None
    if cash_observations is not None:
        end_at = pd.Timestamp(
            bundle.calendar.session.iloc[-1] + " 17:00", tz="America/New_York"
        ).tz_convert("UTC")
        cash = _validate_reconciled_cash(
            cash_observations, as_of=end_at, first_bar_start=bundle.bars.bar_start.iloc[0]
        )
    elif cash_source_file is not None:
        raise DataContractError("cash source supplied without timestamped cash observations")
    cash_meta = None
    if cash_source_file is not None:
        path = Path(cash_source_file)
        if path.is_symlink() or not path.is_file():
            raise DataContractError("invalid cash source provenance file")
        cash_meta = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "provider": "FRED",
            "series_id": "DTB3",
            "role": "non_price_cash_benchmark",
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="." + target.name + ".staging-", dir=target.parent))
    try:
        frames = {
            "daily.csv": bundle.daily,
            "bars.csv": bundle.bars,
            "calendar.csv": bundle.calendar,
            "coverage.csv": bundle.coverage,
            "blackouts.csv": bundle.blackouts,
        }
        if cash is not None:
            frames["DTB3_available.csv"] = cash
        for name, frame in frames.items():
            frame.to_csv(stage / name, index=False)
        (stage / "unit_proof.json").write_text(
            json.dumps(bundle.unit_proof, indent=2, sort_keys=True) + "\n"
        )
        names = [*frames, "unit_proof.json"]
        manifest = {
            "schema": "qqq_alpha_30m_reconciled_packet_v1",
            "provider": "AlphaVantage",
            "symbol": "QQQ",
            "source_metadata": bundle.source_metadata,
            "source_files": bundle.source_files,
            "sampling_interval_minutes": 30,
            "observation_interval_minutes": 30,
            "daily_source_metadata": {
                "provider": "AlphaVantage",
                "symbol": "QQQ",
                "price_basis": "raw_as_traded",
                "price_field": "provider_daily_close",
                "verification": "native Alpha JSON/cache every-field match",
            },
            "price_basis": DERIVED_PRICE_BASIS,
            "native_raw_capture": False,
            "quote_role_profile": "distinct_daily_feature_and_rth_book_quotes",
            "quality_profile": quality_profile,
            "gap_authorization": gap_authorization,
            "historical_ready": validation["unknown_missing_count"] == 0 or approved,
            "performance_generated": False,
            "classified_packet_not_a_simulation": True,
            "quality_model": QUALITY_MODEL,
            "quality_model_sha256": _json_hash(QUALITY_MODEL),
            "implementation_hashes": _implementation_hashes(),
            "unit_proof_sha256": sha256_file(stage / "unit_proof.json"),
            "calendar_provenance": bundle.calendar.attrs.get("calendar_provenance", {}),
            "audit": validation,
            "cash_source": cash_meta,
            "cash_observations": None
            if cash is None
            else {
                "filename": "DTB3_available.csv",
                "rows": len(cash),
                "tie_policy": "latest_observation_date_at_same_available_at",
                "availability_column": "available_at",
                "rate_column": "annual_rate",
            },
            "files": {
                name: {"sha256": sha256_file(stage / name), "bytes": (stage / name).stat().st_size}
                for name in names
            },
        }
        manifest = json.loads(json.dumps(manifest))
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if target.exists():
            raise FileExistsError("immutable reconciled packet already exists")
        os.rename(stage, target)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def read_reconciled_packet(
    root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Verify bytes, source custody, tested unit mapping, clocks and numeric prefix.

    Loading a strict-not-ready classified packet is allowed for inspection; its
    performance gate remains false. Only source BYTES are re-hashed—future quote
    arrays in original master sources are not parsed into a stage evaluation.
    """
    root = Path(root)
    if not root.is_dir() or root.is_symlink() or (root / "manifest.json").is_symlink():
        raise DataContractError("reconciled packet root/manifest must be nonsymlink files")
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest.get("schema") != "qqq_alpha_30m_reconciled_packet_v1"
        or manifest.get("provider") != "AlphaVantage"
        or manifest.get("symbol") != "QQQ"
        or manifest.get("price_basis") != DERIVED_PRICE_BASIS
        or manifest.get("native_raw_capture") is not False
        or manifest.get("sampling_interval_minutes") != 30
        or manifest.get("observation_interval_minutes") != 30
    ):
        raise DataContractError("unsupported quality packet provider/unit/mode contract")
    daily_meta = manifest.get("daily_source_metadata", {})
    if (
        daily_meta.get("provider") != "AlphaVantage"
        or daily_meta.get("symbol") != "QQQ"
        or daily_meta.get("price_basis") != "raw_as_traded"
        or daily_meta.get("price_field") != "provider_daily_close"
        or not daily_meta.get("verification")
        or manifest.get("quote_role_profile") != "distinct_daily_feature_and_rth_book_quotes"
    ):
        raise DataContractError(
            "independent daily source attestation/quote roles changed from Alpha contract"
        )
    if manifest.get("quality_model") != QUALITY_MODEL or manifest.get(
        "quality_model_sha256"
    ) != _json_hash(QUALITY_MODEL):
        raise DataContractError("quality packet model/17NY/source-role/counter contract changed")
    if manifest.get("implementation_hashes") != _implementation_hashes():
        raise DataContractError(
            "quality/counter implementation changed from frozen packet; explicit re-freeze needed"
        )
    required = {
        "daily.csv",
        "bars.csv",
        "calendar.csv",
        "coverage.csv",
        "blackouts.csv",
        "unit_proof.json",
    }
    allowed = required | {"DTB3_available.csv"}
    files = manifest.get("files", {})
    if not required.issubset(files) or set(files) - allowed:
        raise DataContractError("quality packet missing files or contains unsafe/unexpected paths")
    for name, record in files.items():
        path = root / name
        if path.is_symlink() or not path.is_file() or sha256_file(path) != record["sha256"]:
            raise DataContractError(f"quality packet file hash mismatch: {name}")
    if manifest.get("unit_proof_sha256") != files["unit_proof.json"]["sha256"]:
        raise DataContractError("unit proof file is not bound to manifest")
    frames = {
        name: pd.read_csv(root / name, float_precision="round_trip", dtype={"session": str})
        for name in required
        if name.endswith(".csv")
    }
    d, b, c, cov, bl = (
        frames[name]
        for name in ("daily.csv", "bars.csv", "calendar.csv", "coverage.csv", "blackouts.csv")
    )
    for frame, cols in (
        (d, ("daily_price_available_at",)),
        (b, ("bar_start", "bar_end")),
        (c, ("market_open", "market_close")),
        (cov, ("bar_start", "bar_end")),
        (bl, ("start", "end")),
    ):
        for col in cols:
            frame[col] = pd.to_datetime(frame[col], utc=True)
    if "source_url" in bl:
        bl["source_url"] = bl.source_url.fillna("")
    c.attrs["calendar_provenance"] = manifest.get("calendar_provenance", {})
    proof = json.loads((root / "unit_proof.json").read_text())
    bundle = ReconciledBundle(
        d, b, c, cov, bl, proof, manifest["source_metadata"], manifest["source_files"]
    )
    validation = _validate_bundle(bundle)
    if validation != manifest.get("audit"):
        raise DataContractError("reconciled packet validation differs from saved audit")
    approved = _gap_authorization(manifest["quality_profile"], manifest.get("gap_authorization"))
    if manifest.get("historical_ready") != (validation["unknown_missing_count"] == 0 or approved):
        raise DataContractError("quality readiness cannot bypass frozen missing-observation policy")
    if "DTB3_available.csv" in files:
        end_at = pd.Timestamp(c.session.iloc[-1] + " 17:00", tz="America/New_York").tz_convert(
            "UTC"
        )
        cash = _validate_reconciled_cash(
            pd.read_csv(root / "DTB3_available.csv", float_precision="round_trip"),
            as_of=end_at,
            first_bar_start=b.bar_start.iloc[0],
        )
        if manifest.get("cash_observations", {}).get("rows") != len(cash):
            raise DataContractError("cash observations metadata mismatch")
    if manifest.get("cash_source"):
        cs = manifest["cash_source"]
        if sha256_file(cs["path"]) != cs["sha256"]:
            raise DataContractError("cash source custody changed")
    d.attrs["source_metadata"] = manifest["daily_source_metadata"]
    b.attrs["source_metadata"] = manifest["source_metadata"]
    b.attrs["unit_proof"] = proof
    return d, b, c, cov, bl, manifest
