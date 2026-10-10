"""Synthetic integration through real old/new accounting and actual archive schemas.

Fixtures contain invented observations only. Existing market packets, validation
outcomes and OOS prices are never opened.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from test_hybrid_allocation_protocol import membership
from test_multi_scale_engine import fixture, observations
from trend_following import research_hybrid_allocation_engine as engine
from trend_following import research_hybrid_allocation_protocol as p
from trend_following import research_mean_reversion_engine as mr_engine
from trend_following import research_portfolio_engine as portfolio
from trend_following import research_quality_engine_v5 as quality


def invented_case():
    daily, bars, calendar, coverage = fixture(
        [
            {"closes": [130.0] * 13, "daily_close": 130.0},
            {"closes": [90.0] * 13, "daily_close": 90.0},
            {"closes": [130.0] * 13, "daily_close": 130.0},
            {"closes": [90.0] * 13, "daily_close": 90.0},
        ],
        history=(100.0,) * 295,
    )
    cache = engine.build_cache(
        daily,
        bars,
        calendar,
        coverage=coverage,
        quality_policy=quality.QualityPolicy("observed_only_blackout_v1"),
    )
    cash = observations(cache)
    rows = membership()
    from trend_following import research_daily_hourly_v5 as tf
    from trend_following import research_mean_reversion as mr

    tf_members = [tf.Candidate.from_dict(r["parameters"]) for r in rows[:12]]
    mr_members = [mr.Candidate.from_dict(r["parameters"]) for r in rows[12:]]
    return (
        cache,
        cash,
        rows,
        tf_members,
        mr_members,
        (daily, bars, calendar, coverage, None, cash, {}),
    )


def sealed_schema_controls(root, cache, cash, rows, tf, mrs, stage="development"):
    directory = root / stage
    directory.mkdir(parents=True)
    metric_rows, benchmark_rows = [], []
    start, end = cache.calendar.session.iloc[0], cache.calendar.session.iloc[-1]
    for slip in p.SLIPPAGES:
        kwargs = dict(
            start=start,
            end=end,
            cash_observations=cash,
            commission_bps=1.0,
            slippage_bps=slip,
            initial_cash=1000.0,
            record=True,
        )
        tp = quality._run(cache.tf, tf, **kwargs)
        mp = mr_engine.run(cache.mr, mrs, **kwargs)
        paths = portfolio.merge_sleeves(
            portfolio.replay_recorded(cache.tf, tp, start=start, end=end, cash_observations=cash),
            portfolio.replay_recorded(cache.mr, mp, start=start, end=end, cash_observations=cash),
        )
        p.preserved._save_sleeves(directory / f"constituent_s{slip:g}_daily.npz", paths)
        pd.DataFrame([*tp[0], *mp[0]]).to_csv(
            directory / f"constituent_metrics_s{slip:g}.csv", index=False
        )
        pd.DataFrame([*tp[6], *mp[6]]).to_csv(directory / f"records_s{slip:g}.csv", index=False)
        allocations = (
            portfolio.allocation_grid()
            if stage == "development"
            else p.preserved.eligible_validation_allocations(
                {"allocation": portfolio.Allocation("portfolio_tf_000", 0).to_dict()}
            )
        )
        combined = [portfolio.evaluate_portfolio(paths, a) for a in allocations]
        p.preserved._save_portfolios(
            directory / f"allocation_s{slip:g}_daily.npz", combined, paths.cash_returns
        )
        metric_rows.extend({**r.metrics, "scenario_slippage_bps": slip} for r in combined)
        for mode in ("cash", "buy_and_hold"):
            parts = mr_engine.run(cache.mr, mrs[:1], mode=mode, **kwargs)
            pd.DataFrame(
                {
                    "timestamp": parts[1],
                    "equity": parts[2][:, 0],
                    "return": parts[3][:, 0],
                    "cash_return": parts[4],
                    "exposure": parts[5][:, 0],
                }
            ).to_csv(directory / f"{mode}_s{slip:g}_daily.csv", index=False)
            benchmark_rows.append(
                {**parts[0][0], "portfolio_id": mode, "scenario_slippage_bps": slip}
            )
    pd.DataFrame(metric_rows).to_csv(directory / "metrics.csv", index=False)
    pd.DataFrame(benchmark_rows).to_csv(directory / "benchmark_metrics.csv", index=False)
    return directory


def test_actual_event_sleeve_schema_reconstructs_missing_independent_70_90_without_rerun(tmp_path):
    cache, cash, rows, tf, mrs, _packet = invented_case()
    directory = sealed_schema_controls(tmp_path, cache, cash, rows, tf, mrs, stage="validation")
    frozen = {"membership": rows, "preserved_portfolio": str(tmp_path)}
    with np.load(directory / "allocation_s3_daily.npz", allow_pickle=False) as z:
        assert "portfolio_tf_070" not in list(z["portfolio_id"])
        assert "portfolio_tf_090" not in list(z["portfolio_id"])
    restored = p._load_preserved_sleeves(directory, 3.0, frozen)
    metrics, paths = p._read_existing_controls(
        frozen, "validation", 3.0, restored.times, restored.cash_returns
    )
    assert list(paths) == p.control_ids()[:-1]
    by_id = {r["portfolio_id"]: r for r in metrics}
    assert by_id["portfolio_tf_050"]["historically_saved_allocation"] is True
    assert by_id["portfolio_tf_070"]["historically_saved_allocation"] is False
    assert by_id["portfolio_tf_090"]["historically_saved_allocation"] is False
    for weight in (0.5, 0.7, 0.9):
        pid = f"portfolio_tf_{int(weight * 100):03d}"
        independent = portfolio.evaluate_portfolio(restored, portfolio.Allocation(pid, weight))
        np.testing.assert_array_equal(paths[pid]["equity"], independent.daily_equity)
    # The NumPy fill timestamp scalars are np.str_, which must be normalized.
    assert restored.fill_groups
    assert all(isinstance(group.timestamp, pd.Timestamp) for group in restored.fill_groups)


def test_baseline_preflight_compares_all_44_original_paths_and_tf_shadow_fills(tmp_path):
    cache, cash, rows, tf, mrs, _packet = invented_case()
    sealed_schema_controls(tmp_path, cache, cash, rows, tf, mrs)
    shadows = engine.build_shadow_paths(
        cache, tf, mrs, start="2001-01-02", end="2001-01-05", record=True
    )
    frozen = {"membership": rows, "preserved_portfolio": str(tmp_path)}
    result = p._baseline_parity(cache, shadows, tf, mrs, frozen, "development", cash)
    assert result["baseline_all12_TF_all32_MR_paths_match"] is True
    assert result["TF_shadow_open_states_match"] is True


def test_closed_dependency_observer_captures_metadata_results_hashes_and_restores(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(p, "ROOT", tmp_path)
    source = tmp_path / "multi_scale"
    old = tmp_path / "portfolio"
    for folder in (source, old):
        (folder / "development").mkdir(parents=True)
        (folder / "validation").mkdir()
        for stage in ("development", "validation"):
            p.custody._seal(folder / stage / "seal.json", {"stage": stage})
    p.custody._seal(source / "freeze.json", {"preserved_portfolio": str(old)})
    p.custody._seal(source / "locked_selections.json", {"selections": []})
    p.custody._seal(old / "freeze.json", {"membership": membership()})
    p.custody._seal(old / "locked_allocation.json", {"allocation": {}})
    metadata = tmp_path / "input_metadata.json"
    metadata.write_text("{}")
    derived = tmp_path / "derived.csv"
    derived.write_text("metric\n1\n")

    def verified(_source, _stage):
        p.custody.sha256_file(metadata)
        metadata.read_text()
        pd.read_csv(derived)
        return {"stage": "validation"}

    monkeypatch.setattr(p.multi, "verify_results", verified)
    import sys

    before = sys.getprofile()
    _inherited, _old, _lock, _seals, files = p._capture_verified_closure(source)
    assert sys.getprofile() is before
    assert metadata in files and derived in files
    assert source / "freeze.json" in files and old / "validation/seal.json" in files
    binding = p._file_bindings(files)
    p._verify_files(binding)
    derived.write_text("metric\n2\n")
    with pytest.raises(p.Error, match="changed"):
        p._verify_files(binding)


def test_bootstrap_reference_routes_match_locked_allocation_without_reselection():
    values = np.arange(4) * 0.001
    ids = [d["portfolio_id"] for d in p.principal_definitions()]
    principal = {pid: {"returns": values + j * 0.001} for j, pid in enumerate(ids)}
    controls = {pid: {"returns": values - j * 0.001} for j, pid in enumerate(p.control_ids())}
    lock = {
        "selections": [
            {"architecture": "A", "portfolio_id": "hybrid_core_070", "trend_allocation": 0.7},
            {"architecture": "B", "portfolio_id": "hybrid_blend_090", "trend_allocation": 0.9},
        ]
    }
    references = p._validation_comparisons(lock, principal, controls)
    assert [r["reference_id"] for r in references[:3]] == [
        p.PURE_TREND_ID,
        "buy_and_hold",
        "portfolio_tf_070",
    ]
    assert [r["reference_id"] for r in references[3:6]] == [
        p.PURE_TREND_ID,
        "buy_and_hold",
        "hybrid_core_090",
    ]
    assert [r["reference_id"] for r in references[6:]] == [
        p.PURE_TREND_ID,
        "buy_and_hold",
        p.GATED_CONTROL_ID,
    ]


def test_complete_synthetic_development_then_locked_validation_accounting(monkeypatch, tmp_path):
    cache, cash, rows, tf, mrs, packet = invented_case()
    old = tmp_path / "old_portfolio"
    sealed_schema_controls(old, cache, cash, rows, tf, mrs, "development")
    sealed_schema_controls(old, cache, cash, rows, tf, mrs, "validation")
    pairs = p._pair_definitions(rows)
    frozen = {
        "membership": rows,
        "membership_sha256": p.custody.canonical_hash(rows),
        "pairs": pairs,
        "pairs_sha256": p.custody.canonical_hash(pairs),
        "preserved_portfolio": str(old),
        "frozen_at": "2026-10-10T01:00:00+00:00",
    }
    monkeypatch.setattr(p, "verify_study", lambda _: frozen)
    monkeypatch.setattr(p, "load_packet", lambda *_a, **_k: packet)
    monkeypatch.setattr(p.custody, "_quality_policy", lambda _: cache.policy)
    # Both phases use this same tiny invented fixture to exercise sealing; no
    # historical validation dataset or old outcome is loaded.
    monkeypatch.setattr(
        p,
        "PERIODS",
        {
            "development": {"start": "2001-01-02", "end": "2001-01-05"},
            "validation": {"start": "2001-01-02", "end": "2001-01-05"},
        },
    )
    sidecar = tmp_path / "hybrid_allocation_research_synthetic"
    dev = p.evaluate_development(sidecar)
    assert dev["pair_count"] == 384 and dev["principal_count"] == 8
    lock = p.verify_locked_selections(sidecar)
    assert len(lock["selections"]) == 2 and lock["fixed_regime_rule"] == p.REGIME_RULE
    val = p.evaluate_validation(sidecar)
    assert val["selection_lock_sha256"] == p.custody.canonical_hash(lock)
    for stage in ("development", "validation"):
        assert p.verify_results(sidecar, stage)["original_baseline_parity_verified"] is True
        assert len(pd.read_csv(sidecar / stage / "metrics.csv")) == 32
        assert len(pd.read_csv(sidecar / stage / "controls_metrics.csv")) == 32
        for slip in p.SLIPPAGES:
            with np.load(sidecar / stage / f"account_s{slip:g}_daily.npz", allow_pickle=False) as z:
                assert z["equity"].shape[1] == 1932
            with np.load(sidecar / stage / f"pair_s{slip:g}_daily.npz", allow_pickle=False) as z:
                assert z["equity"].shape[1] == 2688
                assert len(set(z["pair_id"])) == 384
    assert (sidecar / "validation/bootstrap.csv").exists()
    with pytest.raises(p.Error, match="sealed"):
        p.evaluate_development(sidecar)
