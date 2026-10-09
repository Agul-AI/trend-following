"""Synthetic fixed-cash diagnostic contract tests; never open market packets."""

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from trend_following import research_constant_cash as fixed
from trend_following import research_quality_engine_v5 as engine
from trend_following.research_rates import rate_accrual


def authorization(tmp_path):
    path = tmp_path / "authorization.json"
    path.write_text(
        json.dumps(
            {
                "schema": "trend_following_development_equal_wait_constant_cash_authorization_v1",
                "authorized_stages": ["development_equal_wait_diagnostic"],
                "explicit_user_approval": True,
                "scope": "diagnostic_not_production_freeze",
                "known_global_wait_value_pending": True,
                "authorized_by": "human_user",
                "user_requests": [
                    "for each choice of time, how many out of 289 configs, beat BH' sharpe ratio after cost.",
                    "you shall set 3% cash interest rate for all",
                ],
            }
        )
    )
    return path


def synthetic_metrics():
    return pd.DataFrame(
        {
            "candidate_id": c.candidate_id,
            "entry_confirmation_hours": hours,
            "exit_confirmation_hours": hours,
            "cash_excess_sharpe": 0.5,
        }
        for hours in fixed.WAIT_HOURS
        for c in fixed.candidates_for_wait(hours)
    )


def synthetic_packet():
    history = pd.bdate_range("1999-11-01", periods=300)
    daily = [
        {
            "session": day.strftime("%Y-%m-%d"),
            "close": 100,
            "dividend_amount": 0,
            "split_coefficient": 1,
        }
        for day in history
    ]
    bars, coverage, calendar = [], [], []
    for session in ("2001-01-05", "2001-01-08", "2001-01-09"):
        opening = pd.Timestamp(f"{session} 09:30", tz="America/New_York").tz_convert("UTC")
        calendar.append(
            {
                "session": session,
                "market_open": opening,
                "market_close": opening + pd.Timedelta(hours=6.5),
            }
        )
        daily.append(
            {"session": session, "close": 99, "dividend_amount": 0, "split_coefficient": 1}
        )
        for number in range(13):
            start = opening + pd.Timedelta(minutes=30 * number)
            end = start + pd.Timedelta(minutes=30)
            bars.append(
                {
                    "session": session,
                    "bar_start": start,
                    "bar_end": end,
                    "open": 99,
                    "high": 99,
                    "low": 99,
                    "close": 99,
                    "volume": 1,
                    "duration_minutes": 30,
                    "is_full_hour": False,
                }
            )
            coverage.append(
                {
                    "session": session,
                    "bar_start": start,
                    "bar_end": end,
                    "status": "observed",
                    "signal_eligible": True,
                    "fill_eligible": True,
                }
            )
    return pd.DataFrame(daily), pd.DataFrame(bars), pd.DataFrame(calendar), pd.DataFrame(coverage)


def test_exact_common_wait_membership_and_hits():
    assert fixed.WAIT_HOURS == tuple(n / 2 for n in range(13))
    ids = set()
    for hours in fixed.WAIT_HOURS:
        candidates = fixed.candidates_for_wait(hours)
        assert len(candidates) == 289
        assert len({(c.entry_rule, c.exit_rule) for c in candidates}) == 289
        assert all(
            c.entry_confirmation_hours == c.exit_confirmation_hours == hours for c in candidates
        )
        assert all(
            c.entry_required_checks == c.exit_required_checks == 1 + int(2 * hours)
            for c in candidates
        )
        ids.update(c.candidate_id for c in candidates)
    assert len(ids) == 3757
    assert fixed.BATCH_SIZE <= 256


@pytest.mark.parametrize("invalid", [False, True, -0.5, 0.25, 6.5, np.nan, np.inf])
def test_invalid_diagnostic_wait_rejected(invalid):
    with pytest.raises(fixed.Error):
        fixed.candidates_for_wait(invalid)


def test_constant_cash_is_a_model_not_dtb3():
    first = pd.Timestamp("2001-01-02 14:30Z")
    last = pd.Timestamp("2011-12-30 22:00Z")
    cash = fixed.constant_cash_observations(first, last)
    assert cash.annual_rate.eq(0.03).all()
    assert "series_id" not in cash
    assert cash.rate_model.str.contains("assumption").all()
    assert cash.known_from.le(first).all()
    assert cash.available_at.le(last).all()
    assert fixed.cash_policy()["day_count"] == "ACT/365"
    assert fixed.cash_policy()["risk_free_reference"].startswith("same_assumed_cash")


@pytest.mark.parametrize(
    "first,last", [("2001-01-02", "2001-01-03"), ("2001-01-02 14:30Z", "2001-01-01 14:30Z")]
)
def test_invalid_cash_endpoints_rejected(first, last):
    with pytest.raises(fixed.Error):
        fixed.constant_cash_observations(first, last)


def test_elapsed_calendar_rate_includes_overnight_weekend_and_daily_releases():
    clock = pd.DatetimeIndex(
        ["2001-01-05 21:00Z", "2001-01-08 14:30Z", "2001-01-08 15:00Z", "2001-01-08 22:00Z"]
    )
    cash = fixed.constant_cash_observations(clock[0], clock[-1])
    result = rate_accrual(clock, cash, interval_start=clock[0])
    expected = np.r_[0, 0.03 * np.diff(clock.asi8) / 1e9 / (365 * 86400)]
    np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-18)
    assert result.iloc[1] > 0.03 * 2 / 365
    assert np.prod(1 + result) > 1 + result.sum()


def test_assumed_rate_does_not_need_future_values():
    clock = pd.date_range("2001-01-02 14:30Z", periods=5, freq="30min")
    cash = fixed.constant_cash_observations(clock[0], clock[-1])
    altered = cash.copy()
    future = pd.DataFrame(
        {"available_at": [pd.Timestamp("2001-01-03 00:00Z")], "annual_rate": [999]}
    )
    altered = pd.concat([altered, future], ignore_index=True)
    pd.testing.assert_series_equal(rate_accrual(clock, cash), rate_accrual(clock, altered))


def test_whole_cash_effective_growth_is_above_nominal_three_percent():
    clock = pd.date_range("2001-01-01 00:00Z", "2002-01-01 00:00Z", freq="D")
    cash = fixed.constant_cash_observations(clock[0], clock[-1])
    growth = np.prod(1 + rate_accrual(clock, cash)) - 1
    assert 0.03 < growth < 0.03046
    assert "slightly_above_3pct" in fixed.cash_policy()["effective_annual_cash_growth"]


def test_counts_preserve_full_denominator_undefined_and_tolerance():
    metrics = synthetic_metrics()
    metrics.loc[0, "cash_excess_sharpe"] = 0.7
    metrics.loc[1, "cash_excess_sharpe"] = np.nan
    metrics.loc[2, "cash_excess_sharpe"] = np.inf
    metrics.loc[3, "cash_excess_sharpe"] = 0.5 + 5e-11
    metrics.loc[4, "cash_excess_sharpe"] = 0.5 - 5e-11
    metrics.loc[5, "cash_excess_sharpe"] = 0.3
    counts = fixed.counts_against_bh(metrics, 0.5).set_index("wait_hours")
    assert len(counts) == 13
    row = counts.loc[0]
    assert row.configuration_count == 289
    assert row.finite_Sharpe_count == 287
    assert row.undefined_Sharpe_count == 2
    assert row.beat_BH_Sharpe_count == 1
    assert row.strict_gt_BH_Sharpe_count == 2
    assert row.tie_BH_Sharpe_count == 285
    assert row.below_BH_Sharpe_count == 1
    assert row.beat_BH_Sharpe_pct == pytest.approx(100 / 289)


@pytest.mark.parametrize(
    "corruption", ["missing", "duplicate", "unequal_wait", "extra", "wrong_id"]
)
def test_counts_reject_partial_wrong_or_unpermitted_grid(corruption):
    metrics = synthetic_metrics()
    if corruption == "missing":
        metrics = metrics.iloc[1:]
    elif corruption == "duplicate":
        metrics.loc[1, "candidate_id"] = metrics.loc[0, "candidate_id"]
    elif corruption == "unequal_wait":
        metrics.loc[0, "exit_confirmation_hours"] = 1
    elif corruption == "extra":
        metrics = pd.concat([metrics, metrics.iloc[[0]]], ignore_index=True)
    else:
        metrics.loc[0, "candidate_id"] = "made_up"
    with pytest.raises(fixed.Error):
        fixed.counts_against_bh(metrics, 0.5)


@pytest.mark.parametrize("score", [np.nan, np.inf, -np.inf])
def test_undefined_bh_blocks_counts(score):
    with pytest.raises(fixed.Error, match="undefined BH"):
        fixed.counts_against_bh(synthetic_metrics(), score)


def test_exact_human_authorization_before_any_packet_lookup(tmp_path, monkeypatch):
    receipt = authorization(tmp_path)
    value = json.loads(receipt.read_text())
    value["authorized_stages"].append("validation")
    receipt.write_text(json.dumps(value))
    monkeypatch.setattr(
        fixed,
        "_read_metadata",
        lambda *_: pytest.fail("No packet lookup before authorization gate"),
    )
    with pytest.raises(fixed.Error, match="authorization"):
        fixed._binding_payload(tmp_path, tmp_path, receipt)


@pytest.mark.parametrize(
    "key,value",
    [
        ("explicit_user_approval", False),
        ("scope", "production"),
        ("known_global_wait_value_pending", False),
        ("authorized_by", "assistant"),
        ("user_requests", []),
    ],
)
def test_authorization_does_not_infer_later_stage_or_production_permission(tmp_path, key, value):
    receipt = authorization(tmp_path)
    edited = json.loads(receipt.read_text())
    edited[key] = value
    receipt.write_text(json.dumps(edited))
    with pytest.raises(fixed.Error):
        fixed._authorization(receipt)


def test_seal_precedes_all_market_array_loading_and_failed_attempt_is_retained(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        fixed, "verify_study", lambda *_: (_ for _ in ()).throw(fixed.Error("missing seal"))
    )
    monkeypatch.setattr(
        fixed, "load_development_packet", lambda *_: pytest.fail("must not read any numeric packet")
    )
    with pytest.raises(fixed.Error, match="missing seal"):
        fixed.evaluate_study(tmp_path)
    assert (tmp_path / "attempts.jsonl").is_file()
    assert json.loads((tmp_path / "attempts.jsonl").read_text())["status"] == "failed"
    assert len(list((tmp_path / "attempts").glob("*/failure.json"))) == 1
    assert not (tmp_path / ".stage.lock").exists()


def test_completed_results_are_not_overwritten_or_rerun(tmp_path, monkeypatch):
    (tmp_path / "results").mkdir()
    marker = tmp_path / "results/marker"
    marker.write_text("sealed old results")
    monkeypatch.setattr(fixed, "verify_study", lambda *_: {})
    monkeypatch.setattr(
        fixed, "load_development_packet", lambda *_: pytest.fail("No duplicate market run")
    )
    with pytest.raises(fixed.Error, match="already exist"):
        fixed.evaluate_study(tmp_path)
    assert marker.read_text() == "sealed old results"


def test_no_broad_or_master_reader_in_development_implementation():
    # AST/call inspection is supplemented by numeric-loading pre-gate tests.
    source = inspect.getsource(fixed.load_development_packet)
    assert "prepared/development" in source
    assert "read_reconciled_packet(" not in source
    assert "_load_stage(" not in source
    assert "_verify_prepared(" not in source
    assert "2011-12-31" in source
    metadata = inspect.getsource(fixed._read_metadata)
    assert 'prepared["prefixes"][STAGE]' in metadata
    assert "for stage in" not in metadata
    assert "DTB3_available.csv" not in metadata


def test_only_fixed_new_source_and_existing_frozen_engine_sources_bound():
    assert fixed.SOURCE_FILES == (
        *fixed.protocol.SOURCE_FILES,
        "src/trend_following/research_constant_cash.py",
    )
    assert fixed.STAGE == "development"
    assert fixed.COMMISSION_BPS == 1
    assert fixed.SLIPPAGE_BPS == 3
    assert fixed.cash_policy()["source"] == "user_assumption_not_DTB3_or_observed_market_yield"


def test_synthetic_all_cash_engine_benchmarks_share_reference_and_terminal_clock():
    daily, bars, calendar, coverage = synthetic_packet()
    cash = fixed.constant_cash_observations(
        calendar.market_open.iloc[0], calendar.market_close.iloc[-1] + pd.Timedelta(hours=1)
    )
    benchmark = engine.simulate_quality_benchmarks(
        daily,
        bars,
        start="2001-01-05",
        end="2001-01-09",
        cash_observations=cash,
        calendar=calendar,
        coverage=coverage,
        commission_bps=1,
        slippage_bps=3,
    )
    assert benchmark["cash"].metrics["order_count"] == 0
    assert benchmark["buy_and_hold"].metrics["order_count"] == 2
    assert benchmark["cash"].daily_equity.index.equals(benchmark["buy_and_hold"].daily_equity.index)
    pd.testing.assert_series_equal(
        benchmark["cash"].daily_cash_returns, benchmark["buy_and_hold"].daily_cash_returns
    )
    assert benchmark["cash"].metrics["terminal_equity"] > 1000
    assert (
        benchmark["cash"].metrics["cash_terminal_equity"]
        == benchmark["buy_and_hold"].metrics["cash_terminal_equity"]
    )


def test_private_curve_export_contains_only_derived_series(tmp_path):
    daily, bars, calendar, coverage = synthetic_packet()
    cash = fixed.constant_cash_observations(
        calendar.market_open.iloc[0], calendar.market_close.iloc[-1] + pd.Timedelta(hours=1)
    )
    result = engine.simulate_quality_benchmarks(
        daily,
        bars,
        start="2001-01-05",
        end="2001-01-09",
        cash_observations=cash,
        calendar=calendar,
        coverage=coverage,
    )["cash"]
    path = tmp_path / "daily.csv"
    fixed._write_curve(path, result)
    frame = pd.read_csv(path)
    assert list(frame) == ["timestamp", "return", "equity", "cash_return", "exposure"]
    assert not set(frame).intersection({"close", "open", "reference_price", "trade_price"})


def synthetic_result_seal(tmp_path, monkeypatch):
    payload = {
        "period": {"start": "2001-01-01", "end": "2011-12-31"},
        "cash": fixed.cash_policy(),
        "grid_sha256": "test-grid",
        "development_manifest_sha256": "test-development",
        "quality_policy": {"post_gap_diagnostics": False},
    }
    monkeypatch.setattr(fixed, "verify_study", lambda *_: payload)
    root = tmp_path / "results"
    root.mkdir()
    names = [
        "benchmark_metrics.csv",
        "buy_and_hold_daily.csv",
        "cash_daily.csv",
        "candidate_metrics.csv",
        "counts_by_wait.csv",
        *[f"wait_{str(hours).replace('.', 'p')}_daily.npz" for hours in fixed.WAIT_HOURS],
    ]
    for name in names:
        (root / name).write_text("synthetic")
    result = {
        "schema": "trend_following_constant_cash_equal_wait_results_v1",
        "freeze_sha256": fixed.protocol.canonical_hash(payload),
        "waits_evaluated": 13,
        "configurations_per_wait": 289,
        "configurations_evaluated": 3757,
        "stage": "development",
        "period": payload["period"],
        "cash_policy_sha256": fixed.protocol.canonical_hash(payload["cash"]),
        "grid_sha256": payload["grid_sha256"],
        "commission_bps_per_order": 1,
        "slippage_bps_per_order": 3,
        "development_manifest_sha256": payload["development_manifest_sha256"],
        "no_winner_selected": True,
        "no_later_market_files_loaded": True,
        "started_at": "2026-10-09T02:00:00Z",
        "finished_at": "2026-10-09T02:01:00Z",
        "BH_cash_excess_Sharpe": 0.5,
        "files": {name: fixed.protocol.sha256_file(root / name) for name in names},
    }
    fixed.protocol._seal(root / "seal.json", result)
    return root, result


def test_verify_result_hash_tamper_is_rejected(tmp_path, monkeypatch):
    root, _ = synthetic_result_seal(tmp_path, monkeypatch)
    (root / "counts_by_wait.csv").write_text("changed")
    with pytest.raises(fixed.Error, match="bytes changed"):
        fixed.verify_results(tmp_path)


def test_verify_untracked_result_file_rejected(tmp_path, monkeypatch):
    root, _ = synthetic_result_seal(tmp_path, monkeypatch)
    (root / "extra.csv").write_text("untracked")
    with pytest.raises(fixed.Error, match="untracked"):
        fixed.verify_results(tmp_path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("stage", "validation"),
        ("configurations_evaluated", 289),
        ("commission_bps_per_order", 0),
        ("cash_policy_sha256", "other"),
        ("no_winner_selected", False),
    ],
)
def test_verify_wrong_result_contract_rejected(tmp_path, monkeypatch, key, value):
    root, result = synthetic_result_seal(tmp_path, monkeypatch)
    result[key] = value
    (root / "seal.json").unlink()
    fixed.protocol._seal(root / "seal.json", result)
    with pytest.raises(fixed.Error, match="contract"):
        fixed.verify_results(tmp_path)


def test_verify_frozen_assumptions_tamper_rejected(tmp_path, monkeypatch):
    payload = {
        "source_study": str(tmp_path.parent / "source"),
        "config_path": str(tmp_path / "config"),
        "authorization": {"path": str(tmp_path.parent / "auth")},
        "cash": {"annual_rate": 0.03},
    }
    fixed.protocol._seal(tmp_path / "freeze.json", payload)
    monkeypatch.setattr(
        fixed, "_binding_payload", lambda *_: {**payload, "cash": {"annual_rate": 0.04}}
    )
    with pytest.raises(fixed.Error, match="assumptions changed"):
        fixed.verify_study(tmp_path)


def test_numeric_reader_refuses_unsafe_file_before_reading_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pd,
        "read_csv",
        lambda *_args, **_kwargs: pytest.fail("No numeric read before hash/path gates"),
    )
    with pytest.raises(fixed.Error, match="file binding"):
        fixed.load_development_packet(
            {
                "source_study": str(tmp_path),
                "development_files": {"../../validation.csv": {"sha256": "anything"}},
            }
        )


def test_no_auth_cli_fails_without_market_loading():
    with pytest.raises(SystemExit) as error:
        fixed.main(["seal"])
    assert error.value.code == 2


@pytest.mark.parametrize("relative", ["same", "ancestor", "descendant"])
def test_sidecar_overlap_rejected_before_original_attempt_write(tmp_path, monkeypatch, relative):
    source = tmp_path / "original"
    source.mkdir()
    sidecar = {"same": source, "ancestor": tmp_path, "descendant": source / "new"}[relative]
    auth = authorization(tmp_path)
    monkeypatch.setattr(
        fixed.protocol,
        "_attempt",
        lambda *_: pytest.fail("Must reject before writing original logs"),
    )
    with pytest.raises(fixed.Error, match="separate tree"):
        fixed.seal_study(sidecar, source_study=source, authorization=auth)
    assert not (source / "attempts").exists()


def test_authorization_must_not_reside_in_original_or_sidecar(tmp_path):
    source = tmp_path / "source"
    sidecar = tmp_path / "diagnostic"
    for unsafe in (source / "auth.json", sidecar / "auth.json"):
        with pytest.raises(fixed.Error, match="authorization"):
            fixed._assert_separate(sidecar, source, unsafe)


def test_safe_separate_roots_accepted(tmp_path):
    fixed._assert_separate(
        tmp_path / "diagnostic", tmp_path / "source", tmp_path / "authorization.json"
    )
