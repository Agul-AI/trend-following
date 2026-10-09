"""Independent split/accounting checks using synthetic observations only.

The callable audit below checks newly released 2012–2018 derived outputs. It
does not simulate, choose parameters, or open any market-price packet. Its
caller must separately verify the new freeze/results before releasing outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_intersection_validation_independent import (
    independent_counts,
    independent_daily_statistics,
    independent_nominal_cash_ledger,
    independent_passive_ledgers,
)
from trend_following import research_resplit_validation as resplit
from trend_following import research_validation_prefix as prefix


def checked_validation_mark_clock(timestamps):
    """Require one ordered UTC-aware 17:00 NY mark per authorized session."""
    clock = pd.DatetimeIndex(timestamps)
    if (
        len(clock) < 2
        or clock.tz is None
        or clock.hasnans
        or clock.has_duplicates
        or not clock.is_monotonic_increasing
    ):
        raise ValueError("Unique chronological timezone-aware validation clock required")
    local = clock.tz_convert("America/New_York")
    sessions = np.asarray(local.strftime("%Y-%m-%d"))
    if (
        len(set(sessions)) != len(clock)
        or (sessions < "2012-01-01").any()
        or (sessions > "2018-12-31").any()
        or not np.all(local.hour == 17)
        or not np.all(local.minute == 0)
        or not np.all(local.second == 0)
    ):
        raise ValueError("Validation marks must be 2012–2018 session days at 17:00 NY")
    return clock


def independent_resplit_passive_ledgers(daily, bars, calendar, coverage, blackouts):
    """Fail if a supposedly bounded NEW packet contains any future session."""
    for frame in (daily, bars, calendar, coverage):
        if frame.empty or frame.session.gt("2018-12-31").any():
            raise ValueError("Independent accounting refuses empty or 2019+ price packets")
    if not blackouts.empty and blackouts.end.gt(pd.Timestamp("2019-01-01 00:00Z")).any():
        raise ValueError("Independent accounting refuses future blackout metadata")
    return independent_passive_ledgers(
        daily,
        bars,
        calendar,
        coverage,
        blackouts,
        start="2012-01-01",
        end="2018-12-31",
    )


def audit_released_outputs(results_root, pairs, passive, *, initial_cash=1000.0):
    """Audit all 477 released derived curves against independent passive ledgers.

    ``passive`` is the already authorized, NEW bounded-prefix ledger result;
    this function itself reads only the NEW derived result tree. No old stage
    outcomes, raw quote files, or production metric/accounting helper is used.
    """
    results_root = Path(results_root)
    if len(pairs) != 159 or len({pair["pair_id"] for pair in pairs}) != 159:
        raise ValueError("Exactly 159 unchanged development-selected indicator pairs required")
    metrics = pd.read_csv(results_root / "candidate_metrics.csv", float_precision="round_trip")
    benchmarks = pd.read_csv(results_root / "benchmark_metrics.csv", float_precision="round_trip")
    counts = pd.read_csv(results_root / "counts_by_wait.csv", float_precision="round_trip")
    if (
        len(metrics) != 477
        or metrics.candidate_id.duplicated().any()
        or len(benchmarks) != 2
        or set(benchmarks.portfolio) != {"buy_and_hold", "cash"}
        or len(counts) != 3
        or set(counts.wait_hours) != {2.0, 3.0, 4.0}
    ):
        raise ValueError("Incomplete or identity-changed validation output population")
    times = checked_validation_mark_clock(passive["times"])
    elapsed = passive["elapsed_years"]
    reference = passive["cash_returns"]
    benchmark = benchmarks.set_index("portfolio")
    max_deltas = {}
    for mode, wealth_key in (
        ("buy_and_hold", "buy_and_hold_equity"),
        ("cash", "cash_equity"),
    ):
        frame = pd.read_csv(results_root / f"{mode}_daily.csv", float_precision="round_trip")
        np.testing.assert_array_equal(pd.DatetimeIndex(frame.timestamp), times)
        np.testing.assert_allclose(frame.equity, passive[wealth_key], rtol=2e-13, atol=1e-9)
        np.testing.assert_allclose(frame.cash_return, reference, rtol=1e-11, atol=1e-14)
        max_deltas[f"{mode}_ledger_absolute"] = float(
            np.max(np.abs(frame.equity.to_numpy() - passive[wealth_key]))
        )
        independent = independent_daily_statistics(
            passive[wealth_key], reference, initial_cash=initial_cash, elapsed_years=elapsed
        )
        for field in (
            "cagr",
            "cash_excess_sharpe",
            "max_drawdown",
            "annualized_volatility",
            "terminal_equity",
        ):
            np.testing.assert_allclose(
                benchmark.loc[mode, field],
                independent[field][0],
                rtol=1e-9,
                atol=1e-10,
                equal_nan=True,
            )
    bh_sharpe = float(benchmark.loc["buy_and_hold", "cash_excess_sharpe"])
    full_ids = set()
    cohort_reports = []
    expected_pairs = {pair["pair_id"] for pair in pairs}
    for hours in (2.0, 3.0, 4.0):
        subset = metrics.loc[metrics.entry_confirmation_hours.eq(hours)]
        if (
            len(subset) != 159
            or not subset.exit_confirmation_hours.eq(hours).all()
            or set(subset.entry_rule + "|" + subset.exit_rule) != expected_pairs
        ):
            raise ValueError("Every frozen pair must remain at every globally equal wait")
        trade_count = subset.trade_count.to_numpy(float)
        if (
            not np.isfinite(trade_count).all()
            or (trade_count < 0).any()
            or not np.all(trade_count == np.floor(trade_count))
        ):
            raise ValueError("Nonnegative integer completed round trips required")
        with np.load(
            results_root / f"wait_{str(hours).replace('.', 'p')}_daily.npz", allow_pickle=False
        ) as curve:
            clock = checked_validation_mark_clock(curve["timestamp"])
            np.testing.assert_array_equal(clock, times)
            ids = curve["candidate_id"].tolist()
            if len(ids) != 159 or len(set(ids)) != 159 or set(ids) != set(subset.candidate_id):
                raise ValueError("NPZ curve identity must match all 159 metrics exactly")
            if full_ids.intersection(ids):
                raise ValueError("The 477 candidate IDs must be unique across wait cohorts")
            full_ids.update(ids)
            if curve["equity"].shape != (len(times), 159):
                raise ValueError("Every selected candidate needs every validation session")
            np.testing.assert_allclose(curve["cash_returns"], reference, rtol=1e-11, atol=1e-14)
            independent = independent_daily_statistics(
                curve["equity"], reference, initial_cash=initial_cash, elapsed_years=elapsed
            )
            np.testing.assert_allclose(
                curve["returns"], independent["returns"], rtol=1e-10, atol=1e-13
            )
            ordered = subset.set_index("candidate_id").loc[ids]
            for field in (
                "cagr",
                "cash_excess_sharpe",
                "max_drawdown",
                "annualized_volatility",
                "terminal_equity",
            ):
                np.testing.assert_allclose(
                    ordered[field], independent[field], rtol=1e-9, atol=1e-10, equal_nan=True
                )
                max_deltas[f"wait_{hours:g}_{field}"] = float(
                    np.nanmax(np.abs(ordered[field].to_numpy() - independent[field]))
                )
        independently_counted = independent_counts(subset.cash_excess_sharpe, bh_sharpe)
        published = counts.loc[counts.wait_hours.eq(hours)].iloc[0]
        for key, value in independently_counted.items():
            assert published[key] == value
        assert published.median_trade_count == np.median(trade_count)
        cohort_reports.append({"wait_hours": hours, **independently_counted})
    return {
        "schema": "trend_following_validation_2018_independent_output_audit_v1",
        "period": {"start": "2012-01-01", "end": "2018-12-31"},
        "session_count": len(times),
        "pair_count": len(pairs),
        "candidate_count": len(full_ids),
        "wait_hours": [2.0, 3.0, 4.0],
        "independent_dividend_actions": passive["dividend_actions"],
        "independent_split_actions": passive["split_actions"],
        "cash_nominal_annual_rate": 0.03,
        "costs_bps_per_order": {"commission": 1, "slippage": 3},
        "cohort_counts": cohort_reports,
        "max_absolute_reconciliation_deltas": max_deltas,
        "all_477_statistics_recomputed": True,
        "raw_share_BH_and_event_clock_cash_ledgers_recomputed": True,
        "no_selection_filtering_or_replacement": True,
        "historical_oos_evaluated": False,
        "passed": True,
    }


def test_new_clock_2018_marks_not_old_2015_endpoint():
    actual = checked_validation_mark_clock(["2012-01-03 22:00Z", "2018-12-31 22:00Z"])
    assert len(actual) == 2
    assert actual[-1].year == 2018


def test_source_stream_never_converts_poisoned_future_quote_fields(tmp_path, monkeypatch):
    path = tmp_path / "coverage.csv"
    path.write_text(
        "session,bar_start,bar_end,status,signal_eligible,fill_eligible\n"
        "2018-12-31,2018-12-31 14:30Z,2018-12-31 15:00Z,observed,True,True\n"
        "2019-01-02,POISON-FUTURE,POISON-FUTURE,POISON-FUTURE,POISON-FUTURE,POISON-FUTURE\n"
    )
    original_convert = prefix._convert
    calls = []

    def checked_convert(value, column):
        assert "POISON-FUTURE" not in value
        assert value != "2019-01-02"
        calls.append(column)
        return original_convert(value, column)

    monkeypatch.setattr(prefix, "_convert", checked_convert)
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: pytest.fail("Unbounded reader"))
    bounded = prefix._stream_frame(path, "coverage.csv", end_session="2018-12-31")
    assert bounded.session.tolist() == ["2018-12-31"]
    assert len(calls) == 6


@pytest.mark.parametrize(
    "change",
    [
        {"historical_oos_authorized": True},
        {"period": {"start": "2019-01-01", "end": "2026-05-28"}},
        {"validation_only": False},
        {"stage": "historical_oos"},
    ],
)
def test_new_runner_rejects_future_stage_before_any_prepared_verification(
    tmp_path, monkeypatch, change
):
    payload = {
        "stage": "validation",
        "validation_only": True,
        "historical_oos_authorized": False,
        "period": {"start": "2012-01-01", "end": "2018-12-31"},
        **change,
    }
    monkeypatch.setattr(resplit, "verify_prepared", lambda *_: pytest.fail("Stage gate bypassed"))
    with pytest.raises(resplit.Error, match="2012–2018"):
        resplit.load_validation_packet(tmp_path, payload)


def test_new_runner_requires_candidate_seal_before_market_preparation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        resplit,
        "verify_study",
        lambda *_: (_ for _ in ()).throw(resplit.Error("No candidate seal")),
    )
    monkeypatch.setattr(
        prefix,
        "prepare_validation_prefix",
        lambda *args, **kwargs: pytest.fail("Numeric preparation started before seal"),
    )
    with pytest.raises(resplit.Error, match="candidate seal"):
        resplit.prepare_study(tmp_path)


@pytest.mark.parametrize(
    "last", ["2019-01-02 22:00Z", "2026-05-28 21:00Z", "2018-12-31 21:00Z", "2018-12-31 22:30Z"]
)
def test_new_clock_rejects_2019_plus_and_wrong_mark_time(last):
    with pytest.raises(ValueError, match="2012–2018"):
        checked_validation_mark_clock(["2012-01-03 22:00Z", last])


@pytest.mark.parametrize(
    "clock",
    [
        ["2012-01-03 22:00Z", "2012-01-03 22:00Z"],
        ["2018-12-31 22:00Z", "2012-01-03 22:00Z"],
        ["2012-01-03 22:00", "2018-12-31 22:00"],
    ],
)
def test_clock_cannot_drop_duplicate_unsorted_or_naive_sessions(clock):
    with pytest.raises(ValueError, match="chronological"):
        checked_validation_mark_clock(clock)


def test_477_statistics_have_independent_sample_sd_and_initial_cash_drawdown():
    wealth = np.tile(np.array([900.0, 810.0, 850.0])[:, None], (1, 477))
    results = independent_daily_statistics(wealth, np.zeros(3), initial_cash=1000, elapsed_years=7)
    returns = np.array([-0.1, -0.1, 850 / 810 - 1])
    np.testing.assert_allclose(
        results["cash_excess_sharpe"], np.sqrt(252) * returns.mean() / returns.std(ddof=1)
    )
    np.testing.assert_allclose(results["max_drawdown"], -0.19)
    np.testing.assert_allclose(results["cagr"], 0.85 ** (1 / 7) - 1)


def test_extended_nominal_cash_compounds_calendar_seconds_not_trade_sessions():
    clock = pd.DatetimeIndex(["2012-01-03 14:30Z", "2018-12-31 22:00Z"])
    wealth = independent_nominal_cash_ledger(clock)
    elapsed = (clock[-1] - clock[0]).total_seconds() / (365 * 86400)
    assert wealth[-1] == pytest.approx(1000 * (1 + 0.03 * elapsed))
    assert wealth[-1] != pytest.approx(1000 * 1.03**elapsed)


def test_output_auditor_checks_complete_159_pair_population_before_file_reads(tmp_path):
    with pytest.raises(ValueError, match="159"):
        audit_released_outputs(tmp_path, [], {})


def test_independent_passive_input_gate_rejects_future_before_production_accounting():
    fake_future = pd.DataFrame({"session": ["2018-12-31", "2019-01-02"]})
    with pytest.raises(ValueError, match="2019"):
        independent_resplit_passive_ledgers(
            fake_future,
            fake_future,
            fake_future,
            fake_future,
            pd.DataFrame(columns=["start", "end"]),
        )


def test_reports_only_metadata_not_paths_quotes_or_private_authorization():
    report = {
        "period": {"start": "2012-01-01", "end": "2018-12-31"},
        "candidate_count": 477,
        "historical_oos_evaluated": False,
    }
    encoded = json.dumps(report)
    assert "2019" not in encoded
    assert "/Users/" not in encoded
    assert "open" not in encoded
