"""Synthetic shortlist governance and batching; no market files or performance."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from trend_following import research_daily_hourly_v5 as core
from trend_following import research_quality_engine_v5 as engine
from trend_following import research_shortlist as shortlist


def synthetic_prices():
    daily = [
        dict(
            session=day.strftime("%Y-%m-%d"),
            close=100.0,
            dividend_amount=0.0,
            split_coefficient=1.0,
        )
        for day in pd.bdate_range("1999-11-01", periods=300)
    ]
    bars, calendar, coverage = [], [], []
    for session, price in (("2002-01-02", 110.0), ("2002-01-03", 113.0), ("2002-01-04", 95.0)):
        opening = pd.Timestamp(session + " 09:30", tz="America/New_York").tz_convert("UTC")
        calendar.append(
            dict(
                session=session, market_open=opening, market_close=opening + pd.Timedelta(hours=6.5)
            )
        )
        daily.append(dict(session=session, close=price, dividend_amount=0.0, split_coefficient=1.0))
        for index in range(13):
            start = opening + pd.Timedelta(minutes=30 * index)
            end = start + pd.Timedelta(minutes=30)
            coverage.append(
                dict(
                    session=session,
                    bar_start=start,
                    bar_end=end,
                    status="observed",
                    signal_eligible=True,
                    fill_eligible=True,
                )
            )
            bars.append(
                dict(
                    session=session,
                    bar_start=start,
                    bar_end=end,
                    open=price,
                    close=price,
                    high=price,
                    low=price,
                    volume=100.0,
                    duration_minutes=30.0,
                    is_full_hour=False,
                    is_completed=True,
                )
            )
    calendar = pd.DataFrame(calendar)
    cash = pd.DataFrame(
        {
            "available_at": [pd.Timestamp("2001-12-31T00:00Z")],
            "annual_rate": [0.02],
            "series_id": ["DTB3"],
        }
    )
    return pd.DataFrame(daily), pd.DataFrame(bars), calendar, pd.DataFrame(coverage), cash


@pytest.fixture
def sealed_synthetic(tmp_path, monkeypatch):
    original = tmp_path / "parent"
    (original / "stages/development").mkdir(parents=True)
    rules = (
        core.IndicatorRule("SMA", period=10),
        core.IndicatorRule("EMA", period=10),
        core.IndicatorRule("MACD", fast=6, slow=13, signal=5),
    )
    candidates = [core.Candidate(entry, exit_rule, 0, 0) for entry in rules for exit_rule in rules]
    pd.DataFrame(
        [
            {
                "candidate_id": candidate.candidate_id,
                "cash_excess_sharpe": 1.0,
                "commission_bps": 1.0,
                "slippage_bps": 3.0,
            }
            for candidate in candidates
        ]
    ).to_csv(original / "stages/development/metrics.csv", index=False)
    config_path = tmp_path / "SYNTHETIC.yaml"
    config_path.write_text("SYNTHETIC TEST CONFIG ONLY")
    segments = {
        stage: {"start": "2002-01-02", "end": "2002-01-04"}
        for stage in ("validation", "historical-oos")
    }
    config = {
        "selection": {"tie_tolerance": 1e-10},
        "segments": segments,
        "accounting": {"initial_cash": 1000.0},
    }
    registration = {
        "grid_sha256": "SYNTHETIC_GRID",
        "candidate_count": len(candidates),
        "source_hashes": {"SYNTHETIC_FROZEN_SOURCE": "unchanged"},
    }
    context = (
        config,
        registration,
        {"SYNTHETIC_DATA": "unchanged"},
        {"SYNTHETIC_DEVELOPMENT": "sealed"},
        candidates,
        {candidate.candidate_id: candidate for candidate in candidates},
    )
    monkeypatch.setattr(shortlist, "_parent_context", lambda original, config_path: context)
    monkeypatch.setattr(shortlist, "_source_hashes", lambda: {"SYNTHETIC_NEW_SOURCE": "unchanged"})
    authorization = tmp_path / "SYNTHETIC_auth.json"
    authorization.write_text(
        json.dumps(
            {
                "schema": "trend_following_shortlist_authorization_v1",
                "project": "Trend Following Study",
                "authorized_by": "human_user",
                "explicit_user_approval": True,
                "user_request": "SYNTHETIC TEST AUTHORIZATION, NOT HUMAN CONSENT",
                "authorized_stages": ["validation", "historical-oos"],
            }
        )
    )
    study = tmp_path / "shortlist"
    args = dict(parent_study=original, config_path=config_path, study_dir=study)
    result = shortlist.seal_shortlist(**args, authorization_receipt=authorization, per_family=1)
    return args, authorization, context, result


def replace_seal(path, mutate):
    payload = shortlist.parent._read_seal(path)
    mutate(payload)
    path.unlink()
    shortlist.parent._seal(path, payload)


def configure_synthetic_execution(monkeypatch):
    daily, bars, calendar, coverage, cash = synthetic_prices()
    policy = engine.QualityPolicy("observed_only_blackout_v1")
    monkeypatch.setattr(
        shortlist.parent,
        "_load_stage",
        lambda *args: (daily, bars, calendar, coverage, None, cash, {}),
    )
    monkeypatch.setattr(shortlist.parent, "_quality_policy", lambda config: policy)
    monkeypatch.setattr(
        shortlist.parent,
        "_engine_kwargs",
        lambda config, stage, cash, calendar, coverage, blackouts, slippage_bps: {
            **config["segments"][stage],
            "cash_observations": cash,
            "calendar": calendar,
            "coverage": coverage,
            "blackouts": blackouts,
            "quality_policy": policy,
            "commission_bps": 1.0,
            "slippage_bps": slippage_bps,
            "initial_cash": 1000.0,
        },
    )


def test_seal_opens_no_later_numerical_stage_or_performance(sealed_synthetic, monkeypatch):
    args, _, _, result = sealed_synthetic
    monkeypatch.setattr(
        shortlist.parent, "_load_stage", lambda *args: pytest.fail("No market prefix permitted")
    )
    monkeypatch.setattr(
        engine, "_run", lambda *args, **kwargs: pytest.fail("No performance permitted")
    )
    assert result["candidate_count"] == 9 and result["performance_run"] is False
    verified = shortlist.verify_shortlist(**args)
    assert len(verified["candidates"]) == 9
    assert verified["registration"]["historical_label"] == shortlist.LABEL


def test_same_directory_and_parent_nested_collision_are_rejected(tmp_path):
    for target in (tmp_path, tmp_path / "nested", tmp_path.parent):
        with pytest.raises(ValueError, match="separate study directory"):
            shortlist._paths(tmp_path, tmp_path / "config", target)


def test_seal_never_overwrites_existing_run(sealed_synthetic):
    args, receipt, _, _ = sealed_synthetic
    before = (args["study_dir"] / "registration.json").read_bytes()
    with pytest.raises(ValueError, match="existing shortlist"):
        shortlist.seal_shortlist(**args, authorization_receipt=receipt, per_family=1)
    assert (args["study_dir"] / "registration.json").read_bytes() == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("authorized_by", "agent"),
        ("explicit_user_approval", False),
        ("authorized_stages", ["validation"]),
        ("user_request", ""),
    ],
)
def test_bad_authorization_rejected_before_stage(sealed_synthetic, monkeypatch, field, value):
    args, receipt, _, _ = sealed_synthetic
    payload = json.loads(receipt.read_text())
    payload[field] = value
    receipt.write_text(json.dumps(payload))
    monkeypatch.setattr(
        shortlist.parent,
        "_load_stage",
        lambda *args: pytest.fail("Authorization must precede prefix read"),
    )
    with pytest.raises(ValueError, match="human authorization"):
        shortlist.run_shortlist_stage("validation", **args)
    assert list((args["study_dir"] / "attempts").glob("*/failure.json"))


def test_valid_receipt_text_change_is_still_hash_bound(sealed_synthetic):
    args, receipt, _, _ = sealed_synthetic
    payload = json.loads(receipt.read_text())
    payload["user_request"] += " ALTERED"
    receipt.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="receipt changed"):
        shortlist.verify_shortlist(**args)


@pytest.mark.parametrize("tamper", ("source", "original_source", "config", "set", "rule"))
def test_tamper_before_any_later_prefix_read(sealed_synthetic, monkeypatch, tamper):
    args, _, context, _ = sealed_synthetic
    if tamper == "source":
        monkeypatch.setattr(
            shortlist, "_source_hashes", lambda: {"SYNTHETIC_NEW_SOURCE": "changed"}
        )
    elif tamper == "original_source":
        context[1]["source_hashes"]["SYNTHETIC_FROZEN_SOURCE"] = "changed"
    elif tamper == "config":
        args["config_path"].write_text("ALTERED")
    elif tamper == "set":
        replace_seal(
            args["study_dir"] / "candidate_set.json", lambda payload: payload["candidates"].pop()
        )
    else:
        replace_seal(
            args["study_dir"] / "registration.json",
            lambda payload: payload["selection"].update(later_stage_selection=True),
        )
    monkeypatch.setattr(
        shortlist.parent,
        "_load_stage",
        lambda *args: pytest.fail("Integrity gate must precede prefix read"),
    )
    with pytest.raises(ValueError):
        shortlist.run_shortlist_stage("validation", **args)
    assert list((args["study_dir"] / "attempts").glob("*/failure.json"))


def test_historical_requires_completed_validation_first(sealed_synthetic, monkeypatch):
    args, _, _, _ = sealed_synthetic
    monkeypatch.setattr(
        shortlist.parent,
        "_load_stage",
        lambda *args: pytest.fail("No historical prefix before validation seal"),
    )
    with pytest.raises(ValueError, match="Required prior sealed artifact missing"):
        shortlist.run_shortlist_stage("historical-oos", **args)


def test_complete_fixed_set_retained_four_costs_and_no_raw_quotes(sealed_synthetic, monkeypatch):
    args, _, context, _ = sealed_synthetic
    configure_synthetic_execution(monkeypatch)
    validation = shortlist.run_shortlist_stage("validation", **args)
    assert validation["candidate_count"] == 9
    assert validation["slippage_bps_scenarios"] == [3.0]
    historical = shortlist.run_shortlist_stage("historical-oos", **args)
    assert historical["candidate_ids"] == validation["candidate_ids"]
    assert historical["slippage_bps_scenarios"] == [1.0, 3.0, 10.0, 25.0]
    root = args["study_dir"] / "stages/historical-oos"
    metrics = pd.read_csv(root / "metrics.csv")
    curves = pd.read_csv(root / "daily.csv.gz")
    assert len(metrics) == 36 and not metrics[["candidate_id", "slippage_bps"]].duplicated().any()
    for _, group in metrics.groupby("slippage_bps"):
        assert set(group.candidate_id) == {candidate.candidate_id for candidate in context[4]}
    assert tuple(curves.columns) == shortlist.DAILY_COLUMNS
    assert not curves[["candidate_id", "timestamp", "slippage_bps"]].duplicated().any()
    assert not ({"open", "close", "reference_price", "execution_price", "quantity"} & set(curves))
    assert set(path.name for path in root.iterdir()) == {
        "metrics.csv",
        "daily.csv.gz",
        "benchmarks.csv",
        "benchmark_daily.csv.gz",
        "stage_seal.json",
    }
    controls = pd.read_csv(root / "benchmark_daily.csv.gz")
    assert set(controls.candidate_id) == {"cash", "buy_and_hold"}
    assert not controls[["candidate_id", "timestamp", "slippage_bps"]].duplicated().any()
    assert (
        shortlist.verify_stage(
            args["study_dir"], "historical-oos", shortlist.verify_shortlist(**args)["registration"]
        )
        == historical
    )
    monkeypatch.setattr(
        shortlist.parent,
        "_load_stage",
        lambda *args: pytest.fail("Sealed stage cannot rerun performance"),
    )
    assert shortlist.run_shortlist_stage("historical-oos", **args) == historical


def test_stage_artifact_tamper_blocks_historical_data_read(sealed_synthetic, monkeypatch):
    args, _, _, _ = sealed_synthetic
    configure_synthetic_execution(monkeypatch)
    shortlist.run_shortlist_stage("validation", **args)
    metrics = args["study_dir"] / "stages/validation/metrics.csv"
    metrics.write_text(metrics.read_text() + "\nALTERED")
    monkeypatch.setattr(
        shortlist.parent,
        "_load_stage",
        lambda *args: pytest.fail("Tampered validation cannot open historical prefix"),
    )
    with pytest.raises(ValueError, match="artifacts changed"):
        shortlist.run_shortlist_stage("historical-oos", **args)


def test_vectorized_derived_batch_matches_public_grid_and_single_curves():
    daily, bars, calendar, coverage, cash = synthetic_prices()
    rules = (core.IndicatorRule("SMA", period=10), core.IndicatorRule("EMA", period=10))
    candidates = [
        core.Candidate(rules[0], rules[1], 0, 0),
        core.Candidate(rules[1], rules[0], 1, 0.5),
    ]
    kwargs = dict(
        start=calendar.session.iloc[0],
        end=calendar.session.iloc[-1],
        cash_observations=cash,
        calendar=calendar,
        coverage=coverage,
        quality_policy=engine.QualityPolicy("observed_only_blackout_v1"),
        commission_bps=1.0,
        slippage_bps=3.0,
        initial_cash=1000.0,
    )
    cache = engine.build_quality_support_cache(
        daily, bars, calendar, coverage=coverage, quality_policy=kwargs["quality_policy"]
    )
    metrics, curves = shortlist.batch_derived(
        cache,
        candidates,
        **{
            key: kwargs[key]
            for key in (
                "start",
                "end",
                "cash_observations",
                "commission_bps",
                "slippage_bps",
                "initial_cash",
            )
        },
    )
    grid = engine.evaluate_quality_grid(
        daily, bars, candidates=candidates, support_cache=cache, **kwargs
    )
    pd.testing.assert_frame_equal(metrics.drop(columns="return_type"), grid)
    for candidate in candidates:
        result = engine.simulate_quality_candidate(
            daily, bars, candidate, support_cache=cache, **kwargs
        )
        curve = curves[curves.candidate_id == candidate.candidate_id]
        assert np.array_equal(curve.equity.to_numpy(), result.daily_equity.to_numpy())
        assert np.array_equal(curve["return"].to_numpy(), result.daily_returns.to_numpy())
        assert np.array_equal(curve.exposure.to_numpy(), result.daily_exposure.to_numpy())


@pytest.mark.parametrize("extra", (["--per-family", "1"], ["--authorization-receipt", "file"]))
def test_cli_forbids_late_selection_overrides(tmp_path, extra):
    with pytest.raises(SystemExit):
        shortlist.main(["validation", "--study-dir", str(tmp_path / "study"), *extra])


def test_cli_requires_direct_authorization_on_seal(tmp_path):
    with pytest.raises(SystemExit):
        shortlist.main(["seal", "--study-dir", str(tmp_path / "study")])
