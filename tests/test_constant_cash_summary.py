"""Synthetic publication tests; no study outcomes, price packets or market runs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scripts import summarize_constant_cash as helper


@pytest.fixture
def fabricated():
    days = pd.bdate_range("2001-01-02", "2011-12-30")
    # Synthetic weekday clock, intentionally not a provider/exchange calendar.
    days = days[np.r_[np.arange(helper.SESSIONS - 1), len(days) - 1]]
    clock = (days + pd.Timedelta(hours=17)).tz_localize("America/New_York").tz_convert("UTC")
    dt = np.diff(np.r_[helper.START.value, clock.asi8]) / 1e9 / (365 * 86400)
    cash_returns = np.expm1(0.03 * dt * 0.999999)
    cash_wealth = 1000 * np.cumprod(1 + cash_returns)
    # Recover returns as the frozen engine does, from portfolio wealth.
    cash_returns = cash_wealth / np.r_[1000, cash_wealth[:-1]] - 1
    bh_returns = 0.0002 + 0.003 * np.sin(np.arange(len(clock)))
    bh_wealth = 1000 * np.cumprod(1 + bh_returns)
    bh_returns = bh_wealth / np.r_[1000, bh_wealth[:-1]] - 1
    excess = bh_returns - cash_returns
    benchmark = np.sqrt(252) * excess.mean() / excess.std(ddof=1)
    base = {
        "commission_bps": 1,
        "slippage_bps": 3,
        "sampling_minutes": 30,
        "scored_sessions": helper.SESSIONS,
        "start_session": "2001-01-02",
        "end_session": "2011-12-30",
        "cash_terminal_equity": cash_wealth[-1],
    }
    benchmarks, curves = [], {}
    for name, returns, wealth, exposure, sharpe in (
        ("buy_and_hold", bh_returns, bh_wealth, 1, benchmark),
        ("cash", cash_returns, cash_wealth, 0, np.nan),
    ):
        all_wealth = np.r_[1000, wealth]
        benchmarks.append(
            {
                **base,
                "candidate_id": "benchmark_" + name,
                "portfolio": name,
                "entry_rule": "sma_10",
                "cash_excess_sharpe": sharpe,
                "cash_excess_Sharpe": sharpe,
                "terminal_equity": wealth[-1],
                "cagr": (wealth[-1] / 1000) ** (1 / helper.YEARS) - 1,
                "max_drawdown": (all_wealth / np.maximum.accumulate(all_wealth) - 1).min(),
            }
        )
        curves[name] = pd.DataFrame(
            {
                "timestamp": clock,
                "return": returns,
                "equity": wealth,
                "cash_return": cash_returns,
                "exposure": exposure,
            }
        )
    rows = []
    for hours in helper.study.WAIT_HOURS:
        scores = np.linspace(benchmark - 0.5, benchmark + 0.5, 289)
        scores[:6] = [
            np.nan,
            np.inf,
            benchmark + 5e-11,
            benchmark - 5e-11,
            benchmark,
            benchmark + 0.4,
        ]
        for candidate, score in zip(helper.study.candidates_for_wait(hours), scores, strict=True):
            rows.append(
                {
                    **base,
                    "candidate_id": candidate.candidate_id,
                    **{key: getter(candidate) for key, getter in helper.IDENTITY.items()},
                    "cash_excess_sharpe": score,
                    "cash_excess_Sharpe": score,
                    "cagr": 0.04,
                    "max_drawdown": -0.2,
                    "terminal_equity": 1500,
                }
            )
    metrics = pd.DataFrame(rows)
    return metrics, pd.DataFrame(benchmarks), curves, helper.independent_counts(metrics, benchmark)


def payload():
    value = {
        "cash": helper.study.cash_policy(),
        "period": {"start": "2001-01-01", "end": "2011-12-31"},
        "wait_hours": list(helper.study.WAIT_HOURS),
        "candidate_count_per_wait": 289,
        "candidate_count": 3757,
        "commission_bps_per_order": 1,
        "slippage_bps_per_order": 3,
        "source_hashes": {"src/example.py": "synthetic"},
        "initial_cash": 1000,
        "development_manifest_sha256": "synthetic_development_only",
        "authorization": {"path": "/private/secret.json", "consent": "not_public"},
    }
    return value


def release(value, benchmark):
    return {
        "freeze_sha256": helper.protocol.canonical_hash(value),
        "cash_policy_sha256": helper.protocol.canonical_hash(value["cash"]),
        "stage": "development",
        "no_winner_selected": True,
        "no_later_market_files_loaded": True,
        "BH_cash_excess_Sharpe": benchmark,
        "files": {
            name: "synthetic"
            for name in (
                "candidate_metrics.csv",
                "benchmark_metrics.csv",
                "counts_by_wait.csv",
                "buy_and_hold_daily.csv",
                "cash_daily.csv",
            )
        },
    }


def test_missing_either_seal_prevents_any_numeric_reader_or_output(tmp_path, monkeypatch):
    value = payload()
    for missing in ("study", "results"):
        calls = []

        def verify_study(*_, calls=calls, missing=missing):
            calls.append("study")
            if missing == "study":
                raise ValueError("missing study seal")
            return value

        def verify_results(*_, calls=calls):
            calls.append("results")
            raise ValueError("missing results seal")

        monkeypatch.setattr(helper.study, "verify_study", verify_study)
        monkeypatch.setattr(helper.study, "verify_results", verify_results)
        monkeypatch.setattr(pd, "read_csv", lambda *_a, **_kw: pytest.fail("numeric gate bypass"))
        with pytest.raises(ValueError, match="seal"):
            helper.publish(tmp_path / "private", tmp_path / "public")
        assert calls == (["study"] if missing == "study" else ["study", "results"])
        assert not (tmp_path / "public").exists()


@pytest.mark.parametrize("mutation", ["rate", "DTB3", "period", "cost", "grid", "later"])
def test_wrong_release_assumption_refuses_numeric_access(tmp_path, monkeypatch, mutation):
    value = payload()
    sealed = release(value, 0.5)
    if mutation == "rate":
        value["cash"]["annual_rate"] = 0.02
    elif mutation == "DTB3":
        value["cash"]["source"] = "DTB3"
    elif mutation == "period":
        value["period"]["end"] = "2026-05-28"
    elif mutation == "cost":
        value["slippage_bps_per_order"] = 1
    elif mutation == "grid":
        value["candidate_count"] = 56644
    else:
        sealed["no_later_market_files_loaded"] = False
    monkeypatch.setattr(helper.study, "verify_study", lambda _: value)
    monkeypatch.setattr(helper.study, "verify_results", lambda _: sealed)
    monkeypatch.setattr(helper, "_read", lambda *_: pytest.fail("numeric gate bypass"))
    with pytest.raises(ValueError, match="Release must bind"):
        helper.publish(tmp_path / "private", tmp_path / "public")
    assert not (tmp_path / "public").exists()


def test_complete_canonical_grid_and_costs_are_required(fabricated):
    metrics = fabricated[0]
    assert len(helper.validate_metrics(metrics)) == 13 * 289
    for bad in (metrics.iloc[:-1], pd.concat([metrics, metrics.iloc[[0]]])):
        with pytest.raises(ValueError, match="complete|Unique"):
            helper.validate_metrics(bad)
    for field, wrong in (
        ("exit_confirmation_hours", 6.5),
        ("commission_bps", 2),
        ("end_session", "2015-12-31"),
        ("entry_rule", "ema_999"),
    ):
        bad = metrics.copy()
        bad.loc[0, field] = wrong
        with pytest.raises(ValueError, match="contract differs|canonical"):
            helper.validate_metrics(bad)


@pytest.mark.parametrize("raw", ["close", "open", "execution_price", "quantity", "authorization"])
def test_derived_schema_rejects_quotes_trades_and_private_consent(fabricated, raw):
    with pytest.raises(ValueError, match="whitelisted"):
        helper.validate_metrics(fabricated[0].assign(**{raw: 123}))
    bad = fabricated[0].copy()
    bad.loc[0, "entry_rule"] = "/private/credential"
    with pytest.raises(ValueError, match="private paths"):
        helper.validate_metrics(bad)


def test_counts_keep_all_289_and_undefined_scores_separate(fabricated):
    metrics, _, _, counts = fabricated
    independent = helper.independent_counts(metrics, counts.BH_cash_excess_Sharpe.iloc[0])
    helper.validate_counts(counts, independent)
    assert counts.configuration_count.eq(289).all()
    assert counts.undefined_Sharpe_count.eq(2).all()
    assert counts.strict_gt_BH_Sharpe_count.gt(counts.beat_BH_Sharpe_count).all()
    assert (
        (
            counts.undefined_Sharpe_count
            + counts.beat_BH_Sharpe_count
            + counts.tie_BH_Sharpe_count
            + counts.below_BH_Sharpe_count
        )
        .eq(289)
        .all()
    )
    for column, wrong in (
        ("beat_BH_Sharpe_count", 0),
        ("cash_nominal_annual_rate", 0.015),
        ("BH_cash_excess_Sharpe", 0.99),
        ("commission_bps", 2),
    ):
        bad = counts.copy()
        bad.loc[0, column] = wrong
        with pytest.raises(ValueError, match="independently recomputed"):
            helper.validate_counts(bad, independent)


def test_cash_curve_reference_and_actual_growth_not_3pct_apy(fabricated):
    _, benchmarks, curves, _ = fabricated
    statistics = helper.validate_curves(curves["buy_and_hold"], curves["cash"], benchmarks, 1000)
    assert 0.03 < statistics["cash"]["cagr"] < np.expm1(0.03)
    assert np.isnan(statistics["cash"]["cash_excess_sharpe"])
    bad = curves["cash"].copy()
    bad["return"] = 0.01 / 252  # Old variable/other-rate curves cannot be relabeled.
    with pytest.raises(ValueError, match="3% ACT/365"):
        helper.validate_curves(curves["buy_and_hold"], bad, benchmarks, 1000)
    bad = curves["buy_and_hold"].copy()
    bad["cash_return"] = 0
    with pytest.raises(ValueError, match="new cash reference"):
        helper.validate_curves(bad, curves["cash"], benchmarks, 1000)


def test_duplicate_headers_rejected_before_pandas_mangling(tmp_path):
    path = tmp_path / "duplicate.csv"
    path.write_text("cash_excess_sharpe,cash_excess_sharpe\n0.4,0.5\n")
    with pytest.raises(ValueError, match="Duplicate CSV"):
        helper._read(path)


def test_publication_has_only_derived_aggregate_files_no_private_paths(
    tmp_path, monkeypatch, fabricated
):
    metrics, benchmarks, curves, counts = fabricated
    private = tmp_path / "private"
    (private / "results").mkdir(parents=True)
    for name, frame in (
        ("candidate_metrics.csv", metrics),
        ("benchmark_metrics.csv", benchmarks),
        ("counts_by_wait.csv", counts),
        ("buy_and_hold_daily.csv", curves["buy_and_hold"]),
        ("cash_daily.csv", curves["cash"]),
    ):
        frame.to_csv(private / "results" / name, index=False)
    value = payload()
    sealed = release(value, counts.BH_cash_excess_Sharpe.iloc[0])
    monkeypatch.setattr(helper.study, "verify_study", lambda _: value)
    monkeypatch.setattr(helper.study, "verify_results", lambda _: sealed)
    output = tmp_path / "public"
    summary = helper.publish(private, output)
    assert set(path.name for path in output.iterdir()) == {
        "candidate_metrics.csv",
        "benchmark_metrics.csv",
        "counts_by_wait.csv",
        "benchmark_daily_curves.csv",
        "summary.json",
        "artifact_hashes.json",
    }
    public_benchmarks = pd.read_csv(output / "benchmark_metrics.csv")
    assert "entry_rule" not in public_benchmarks and "candidate_id" not in public_benchmarks
    assert set(public_benchmarks.portfolio_label) == {"Cash", "QQQ buy-and-hold"}
    assert summary["new_global_wait_or_strategy_selected"] is False
    assert summary["later_stage_numerical_inputs_opened"] is False
    source = (output / "summary.json").read_text()
    assert "/private" not in source and "authorization" not in source and "consent" not in source
    assert json.loads(source)["cash_policy"]["annual_rate"] == 0.03
    with pytest.raises(FileExistsError):
        helper.publish(private, output)


def test_wrong_saved_count_produces_no_partial_public_folder(tmp_path, monkeypatch, fabricated):
    metrics, benchmarks, curves, counts = fabricated
    private = tmp_path / "private"
    (private / "results").mkdir(parents=True)
    counts = counts.copy()
    counts.loc[0, "beat_BH_Sharpe_count"] += 1
    for name, frame in (
        ("candidate_metrics.csv", metrics),
        ("benchmark_metrics.csv", benchmarks),
        ("counts_by_wait.csv", counts),
        ("buy_and_hold_daily.csv", curves["buy_and_hold"]),
        ("cash_daily.csv", curves["cash"]),
    ):
        frame.to_csv(private / "results" / name, index=False)
    value = payload()
    sealed = release(value, counts.BH_cash_excess_Sharpe.iloc[0])
    monkeypatch.setattr(helper.study, "verify_study", lambda _: value)
    monkeypatch.setattr(helper.study, "verify_results", lambda _: sealed)
    with pytest.raises(ValueError, match="independently recomputed"):
        helper.publish(private, tmp_path / "public")
    assert not (tmp_path / "public").exists()
