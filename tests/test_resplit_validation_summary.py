"""Synthetic-only publisher checks; no market runs or private outcome access."""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest
from scripts import summarize_resplit_validation as helper


def fabricated_prepared():
    return {"manifest_sha256": "synthetic_manifest", "prepared_at": "2026-10-08T17:01Z"}


@pytest.fixture(autouse=True)
def synthetic_prepared_gate(monkeypatch):
    monkeypatch.setattr(helper.study, "verify_prepared", lambda _: fabricated_prepared())
    monkeypatch.setattr(
        helper.study.prefix,
        "verify_validation_prefix",
        lambda *_a, **_k: {
            "audit": {
                "validation_sessions": 1760,
                "first_validation_session": "2012-01-03",
                "last_validation_session": "2018-12-31",
            }
        },
    )


@pytest.fixture
def fabricated():
    rules = helper.core.indicator_rules()
    pairs = [(entry, exit) for entry in rules for exit in rules][:159]
    candidates = [
        helper.core.Candidate(entry, exit, hours, hours)
        for hours in helper.WAITS
        for entry, exit in pairs
    ]
    payload = {
        "cash": helper.cash_model.cash_policy(),
        "wait_hours": list(helper.WAITS),
        "commission_bps_per_order": 1,
        "slippage_bps_per_order": 3,
        "candidates": [
            {"candidate_id": candidate.candidate_id, **candidate.to_dict()}
            for candidate in candidates
        ],
        "pairs": [
            {
                "pair_id": f"{en.rule_id}|{ex.rule_id}",
                "entry_rule": en.to_dict(),
                "exit_rule": ex.to_dict(),
            }
            for en, ex in pairs
        ],
        "selection": {
            "BH_cash_excess_Sharpe": 0.05,
            "uses_validation_or_OOS_outcomes": False,
            "wait_hours": list(helper.WAITS),
            "shared_pair_count": 159,
            "full_indicator_pairs_per_wait": 289,
            "tolerance": 1e-10,
        },
        "stage": "validation",
        "period": {"start": "2012-01-01", "end": "2018-12-31"},
        "configuration_count_per_wait": 159,
        "configuration_count": 477,
        "historical_oos_authorized": False,
        "schema": "trend_following_resplit_intersection_validation_v1",
        "validation_only": True,
        "historical_oos_period": {"start": "2019-01-01", "end": "2026-05-28"},
        "candidate_set_sealed_at": "2026-10-08T17:00Z",
        "prior_validation_or_OOS_outcomes_used_for_selection": False,
        "no_validation_selection_or_replacement": True,
        "initial_cash": 1000,
        "source_hashes": {"src/example.py": "synthetic"},
        "authorization": {"path": "/private/secret_receipt.json", "consent": "never publish"},
    }
    payload["pairs"] = sorted(payload["pairs"], key=lambda pair: pair["pair_id"])
    days = pd.DatetimeIndex(helper.SESSION_DATES)
    clock = (days + pd.Timedelta(hours=17)).tz_localize("America/New_York").tz_convert("UTC")
    dt = np.diff(np.r_[helper.START.value, clock.asi8]) / 1e9 / (365 * 86400)
    cash_returns = np.expm1(0.03 * dt * 0.999999)
    cash_wealth = 1000 * np.cumprod(1 + cash_returns)
    cash_returns = cash_wealth / np.r_[1000, cash_wealth[:-1]] - 1
    bh_returns = 0.0002 + 0.003 * np.sin(np.arange(len(clock)))
    bh_wealth = 1000 * np.cumprod(1 + bh_returns)
    bh_returns = bh_wealth / np.r_[1000, bh_wealth[:-1]] - 1
    excess = bh_returns - cash_returns
    benchmark = np.sqrt(252) * excess.mean() / excess.std(ddof=1)
    constants = {
        "commission_bps": 1,
        "slippage_bps": 3,
        "sampling_minutes": 30,
        "scored_sessions": helper.SESSIONS,
        "start_session": "2012-01-03",
        "end_session": "2018-12-31",
        "cash_terminal_equity": cash_wealth[-1],
    }
    benchmarks, curves = [], {}
    for policy, returns, equity, exposure, sharpe in (
        ("buy_and_hold", bh_returns, bh_wealth, 1, benchmark),
        ("cash", cash_returns, cash_wealth, 0, np.nan),
    ):
        wealth = np.r_[1000, equity]
        benchmarks.append(
            {
                **constants,
                "candidate_id": "benchmark_" + policy,
                "portfolio": policy,
                "entry_rule": "sma_10",
                "cash_excess_sharpe": sharpe,
                "cash_excess_Sharpe": sharpe,
                "terminal_equity": equity[-1],
                "cagr": (equity[-1] / 1000) ** (1 / helper.YEARS) - 1,
                "max_drawdown": (wealth / np.maximum.accumulate(wealth) - 1).min(),
            }
        )
        curves[policy] = pd.DataFrame(
            {
                "timestamp": clock,
                "return": returns,
                "equity": equity,
                "cash_return": cash_returns,
                "exposure": exposure,
            }
        )
    metrics = []
    for index, candidate in enumerate(candidates):
        score = benchmark + (index % 159 - 79) * 0.01
        if index % 159 == 0:
            score = np.nan
        if index % 159 == 1:
            score = np.inf
        if index % 159 == 2:
            score = benchmark + 5e-11
        metrics.append(
            {
                **constants,
                "candidate_id": candidate.candidate_id,
                **{key: getter(candidate) for key, getter in helper.shared.IDENTITY.items()},
                "cash_excess_sharpe": score,
                "cash_excess_Sharpe": score,
                "cagr": 0.05,
                "max_drawdown": -0.2,
                "terminal_equity": 1200,
                "trade_count": index % 159,
                "order_count": 2 * (index % 159),
                "post_gap_confirmed_signal_count": 0,
                "has_post_gap_confirmed_signal": False,
            }
        )
    validation = pd.DataFrame(metrics)
    development = validation.copy()
    development["annualized_volatility"] = 0.2
    development["exposure_fraction"] = 0.5
    development["annualized_turnover"] = 4.0
    development["average_holding_hours"] = 100.0
    development["cash_excess_sharpe"] = np.linspace(0.1, 0.8, len(development))
    development = development[list(helper.DEV_EXPORT_COLUMNS)]
    payload["development_selection_rows"] = development.to_dict("records")
    result = {
        "stage": "validation",
        "no_winner_selected": True,
        "configuration_count": 477,
        "configuration_count_per_wait": 159,
        "waits_evaluated": list(helper.WAITS),
        "no_validation_filtering_or_replacement": True,
        "no_2019plus_numeric_prices_or_prior_validation_outcomes_loaded": True,
        "schema": "trend_following_resplit_intersection_validation_results_v1",
        "period": {"start": "2012-01-01", "end": "2018-12-31"},
        "prepared_at": "2026-10-08T17:01Z",
        "validation_manifest_sha256": "synthetic_manifest",
        "market_outcomes_started_at": "2026-10-08T17:02Z",
        "finished_at": "2026-10-08T17:03Z",
        "historical_oos_authorized": False,
        "freeze_sha256": helper.protocol.canonical_hash(payload),
        "private_notes": "/private/sensitive",
    }
    result["prepared_sha256"] = helper.protocol.canonical_hash(fabricated_prepared())
    return payload, result, validation, development, pd.DataFrame(benchmarks), curves


def checked(fabricated):
    payload, result, validation, development, benchmarks, curves = fabricated
    table = helper.frozen_candidates(payload)
    return (
        payload,
        result,
        helper.validate_metrics(validation, table),
        helper.validate_development_selection(development, table, payload),
        helper.validate_metrics(benchmarks, table, benchmark=True),
        curves,
    )


def test_frozen_same_159_pairs_at_three_equal_waits(fabricated):
    payload = fabricated[0]
    assert len(helper.frozen_candidates(payload)) == 477
    for mutation in ("missing", "duplicate", "unequal_wait", "pair_table", "rule"):
        bad = copy.deepcopy(payload)
        if mutation == "missing":
            bad["candidates"].pop()
        elif mutation == "duplicate":
            bad["candidates"][-1] = bad["candidates"][0]
        elif mutation == "unequal_wait":
            bad["candidates"][0]["exit_confirmation_hours"] = 3
        elif mutation == "pair_table":
            bad["pairs"].pop()
        else:
            bad["candidates"][0]["entry_rule"]["period"] = 999
        with pytest.raises(ValueError):
            helper.frozen_candidates(bad)


def test_missing_either_seal_prevents_numeric_reader_and_output(tmp_path, monkeypatch, fabricated):
    payload = fabricated[0]
    for missing in ("study", "results"):
        calls = []

        def first(*_, calls=calls, missing=missing):
            calls.append("study")
            if missing == "study":
                raise ValueError("missing study seal")
            return payload

        def second(*_, calls=calls):
            calls.append("results")
            raise ValueError("missing result seal")

        monkeypatch.setattr(helper.study, "verify_study", first)
        monkeypatch.setattr(helper.study, "verify_results", second)
        monkeypatch.setattr(helper.shared, "_read", lambda *_: pytest.fail("numeric gate bypass"))
        with pytest.raises(ValueError, match="seal"):
            helper.publish(tmp_path / "private", tmp_path / "public")
        assert calls == (["study"] if missing == "study" else ["study", "results"])
        assert not (tmp_path / "public").exists()


@pytest.mark.parametrize(
    "mutation", ["cash", "rate", "cost", "stage", "winner", "freeze", "source_path"]
)
def test_invalid_contract_blocks_all_numeric_reads(tmp_path, monkeypatch, fabricated, mutation):
    payload, result = copy.deepcopy(fabricated[:2])
    if mutation == "cash":
        payload["cash"]["source"] = "DTB3"
    elif mutation == "rate":
        payload["cash"]["annual_rate"] = 0.02
    elif mutation == "cost":
        payload["commission_bps_per_order"] = 2
    elif mutation == "stage":
        result["stage"] = "historical-oos"
    elif mutation == "winner":
        result["no_winner_selected"] = False
    elif mutation == "freeze":
        result["freeze_sha256"] = "wrong"
    else:
        payload["source_hashes"] = {"/private/source.py": "wrong"}
    if mutation != "freeze":
        result["freeze_sha256"] = helper.protocol.canonical_hash(payload)
    monkeypatch.setattr(helper.study, "verify_study", lambda _: payload)
    monkeypatch.setattr(helper.study, "verify_results", lambda _: result)
    monkeypatch.setattr(helper.shared, "_read", lambda *_: pytest.fail("numeric gate bypass"))
    with pytest.raises(ValueError):
        helper.publish(tmp_path / "private", tmp_path / "public")
    assert not (tmp_path / "public").exists()


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "cost", "wait", "period", "raw_price", "private_path"]
)
def test_full_validation_477_and_derived_privacy_required(fabricated, mutation):
    payload, _, frame, *_ = fabricated
    table = helper.frozen_candidates(payload)
    bad = frame.copy()
    if mutation == "missing":
        bad = bad.iloc[:-1]
    elif mutation == "duplicate":
        bad = pd.concat([bad, bad.iloc[[0]]])
    elif mutation == "cost":
        bad.loc[0, "slippage_bps"] = 25
    elif mutation == "wait":
        bad.loc[0, "exit_confirmation_hours"] = 4
    elif mutation == "period":
        bad.loc[0, "end_session"] = "2026-05-28"
    elif mutation == "raw_price":
        bad["execution_price"] = 99
    else:
        bad.loc[0, "quality_profile"] = "/private/cache/file"
    with pytest.raises(ValueError):
        helper.validate_metrics(bad, table)


def test_undefined_sharpes_retained_denominator_and_all_pair_trade_medians(fabricated):
    _, _, validation, _, benchmarks, _ = checked(fabricated)
    bh = benchmarks.loc[benchmarks.portfolio.eq("buy_and_hold"), "cash_excess_sharpe"].iloc[0]
    counts = helper.counts_by_wait(validation, bh)
    assert counts.configuration_count.eq(159).all()
    assert counts.undefined_Sharpe_count.eq(2).all()
    assert counts.finite_Sharpe_count.eq(157).all()
    assert counts.tie_BH_Sharpe_count.eq(2).all()
    assert counts.median_completed_round_trips.eq(79).all()
    assert counts.median_one_way_orders.eq(158).all()
    assert np.allclose(counts.beat_BH_Sharpe_pct, counts.beat_BH_Sharpe_count * 100 / 159)
    assert len(validation) == 477


@pytest.mark.parametrize("mutation", ["negative", "fraction", "unpaired"])
def test_trade_count_reconciliation_required(fabricated, mutation):
    validation = checked(fabricated)[2]
    if mutation == "negative":
        validation.loc[0, "trade_count"] = -1
    elif mutation == "fraction":
        validation["trade_count"] = validation.trade_count.astype(float)
        validation.loc[0, "trade_count"] = 0.5
    else:
        validation.loc[0, "order_count"] = 3
    with pytest.raises(ValueError):
        helper.counts_by_wait(validation, 0.5)


def test_development_selection_must_beat_bh_all_three_times_without_val_rank(fabricated):
    _, _, validation, development, *_ = checked(fabricated)
    wide = helper.wide_pairs(development, validation, 0.05)
    assert len(wide) == 159
    assert list(wide.pair_id) == sorted(wide.pair_id)
    assert wide.selected_using_development_only.all()
    assert "validation_cash_excess_sharpe_2h" in wide
    bad = development.copy()
    bad.loc[0, "cash_excess_sharpe"] = 0.05 + 5e-11
    with pytest.raises(ValueError, match="Every selected pair"):
        helper.wide_pairs(bad, validation, 0.05)


@pytest.mark.parametrize("mutation", ["missing", "clock", "cash_rate", "quote", "equity", "metric"])
def test_fresh_benchmark_clock_cash_and_metrics_reconcile(fabricated, mutation):
    *_, benchmarks, curves = checked(fabricated)
    bh, cash = curves["buy_and_hold"].copy(), curves["cash"].copy()
    if mutation == "missing":
        bh = bh.iloc[:-1]
    elif mutation == "clock":
        bh.loc[0, "timestamp"] += pd.Timedelta(minutes=30)
    elif mutation == "cash_rate":
        cash.loc[0, "return"] = 0.02
    elif mutation == "quote":
        cash["close"] = 101
    elif mutation == "equity":
        bh.loc[0, "equity"] = 2000
    else:
        benchmarks = benchmarks.copy()
        benchmarks.loc[0, "cagr"] += 0.1
    with pytest.raises(ValueError):
        helper.validate_curves(bh, cash, benchmarks, 1000)


def test_small_derived_benchmark_curves_pass_without_raw_arrays(fabricated):
    *_, benchmarks, curves = checked(fabricated)
    result = helper.validate_curves(curves["buy_and_hold"], curves["cash"], benchmarks, 1000)
    assert 0.03 < result["cash"]["cagr"] < 0.030455
    assert np.isnan(result["cash"]["cash_excess_sharpe"])


def test_publication_exports_every_pair_but_no_consent_paths_quotes_or_npz(
    tmp_path, monkeypatch, fabricated
):
    payload, result, validation, development, benchmarks, curves = fabricated
    monkeypatch.setattr(helper.study, "verify_study", lambda _: payload)
    monkeypatch.setattr(helper.study, "verify_results", lambda _: result)
    files = {
        "candidate_metrics.csv": validation,
        "development_selection_metrics.csv": development,
        "benchmark_metrics.csv": benchmarks,
        "counts_by_wait.csv": helper.study.counts_against_bh(
            validation,
            benchmarks.loc[benchmarks.portfolio.eq("buy_and_hold"), "cash_excess_sharpe"].iloc[0],
            payload["pairs"],
        ),
        **{policy + "_daily.csv": frame for policy, frame in curves.items()},
    }
    reads = []

    def read(path):
        reads.append(path.name)
        assert path.name in files
        return files[path.name].copy()

    monkeypatch.setattr(helper.shared, "_read", read)
    output = tmp_path / "public"
    summary = helper.publish(tmp_path / "private", output)
    assert len(pd.read_csv(output / "validation_candidate_metrics.csv")) == 477
    assert len(pd.read_csv(output / "pair_comparison.csv")) == 159
    assert len(pd.read_csv(output / "counts_by_wait.csv")) == 3
    assert set(reads) == set(files)
    assert not any(path.suffix == ".npz" for path in output.iterdir())
    text = (output / "summary.json").read_text()
    assert "/private" not in text and "never publish" not in text
    assert not summary["winner_selected_after_validation"]
    assert not summary["historical_oos_authorized_or_opened_in_this_follow_up"]
    assert not summary["validation_is_untouched_confirmation"]
    public_benchmarks = pd.read_csv(output / "benchmark_metrics.csv")
    assert "entry_rule" not in public_benchmarks
    assert "candidate_id" not in public_benchmarks
    assert isinstance(json.loads(text), dict)
    with pytest.raises(FileExistsError):
        helper.publish(tmp_path / "private", output)


@pytest.mark.parametrize("mutation", ["count", "median", "wait", "partial"])
def test_private_sealed_counts_must_match_all_477_outcomes(fabricated, mutation):
    payload, _, validation, _, benchmarks, _ = checked(fabricated)
    bh = benchmarks.loc[benchmarks.portfolio.eq("buy_and_hold"), "cash_excess_sharpe"].iloc[0]
    counts = helper.study.counts_against_bh(validation, bh, payload["pairs"])
    if mutation == "count":
        counts.loc[0, "configuration_count"] = 239
    elif mutation == "median":
        counts.loc[0, "median_trade_count"] += 1
    elif mutation == "wait":
        counts.loc[0, "wait_hours"] = 6
    else:
        counts = counts.iloc[:-1]
    with pytest.raises(ValueError, match="complete 477"):
        helper.reconcile_private_counts(counts, validation, bh, payload["pairs"])


def test_prepared_seal_is_required_before_any_derived_numeric_reads(
    tmp_path, monkeypatch, fabricated
):
    monkeypatch.setattr(helper.study, "verify_study", lambda _: fabricated[0])
    monkeypatch.setattr(
        helper.study,
        "verify_prepared",
        lambda _: (_ for _ in ()).throw(ValueError("missing prepared seal")),
    )
    monkeypatch.setattr(helper.shared, "_read", lambda _: pytest.fail("prepared gate bypass"))
    with pytest.raises(ValueError, match="prepared seal"):
        helper.publish(tmp_path / "private", tmp_path / "public")
    assert not (tmp_path / "public").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "old_period",
        "OOS_authorized",
        "old_schema",
        "future_schema",
        "missing_prepared",
        "reversed_time",
    ],
)
def test_resplit_or_future_OOS_contract_cannot_be_relabeled(
    tmp_path, monkeypatch, fabricated, mutation
):
    payload, result = copy.deepcopy(fabricated[:2])
    if mutation == "old_period":
        payload["period"]["end"] = "2015-12-31"
    elif mutation == "OOS_authorized":
        payload["historical_oos_authorized"] = True
    elif mutation == "old_schema":
        result["schema"] = "trend_following_constant_cash_intersection_validation_results_v1"
    elif mutation == "future_schema":
        payload["historical_oos_period"]["start"] = "2016-01-01"
    elif mutation == "missing_prepared":
        result["prepared_sha256"] = "unbound"
    else:
        result["market_outcomes_started_at"] = "2026-10-08T16:59Z"
    result["freeze_sha256"] = helper.protocol.canonical_hash(payload)
    monkeypatch.setattr(helper.study, "verify_study", lambda _: payload)
    monkeypatch.setattr(helper.study, "verify_results", lambda _: result)
    monkeypatch.setattr(helper.shared, "_read", lambda _: pytest.fail("contract gate bypass"))
    with pytest.raises(ValueError):
        helper.publish(tmp_path / "private", tmp_path / "public")
    assert not (tmp_path / "public").exists()
