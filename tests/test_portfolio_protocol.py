"""Synthetic freezes, membership and no-OOS gates for the additive portfolio.

No market packets or actual prior/new numerical outcomes are read by these
checks. Candidate definitions use public mathematical grids only.
"""

from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from trend_following import research_daily_hourly_v5 as trend
from trend_following import research_mean_reversion as mr
from trend_following import research_portfolio_protocol as study

TREND_PAIRS = (
    ("SMA", 100, "SMA", 200),
    ("MACD", (12, 26, 9), "SMA", 200),
    ("MACD", (24, 52, 18), "SMA", 200),
    ("SMA", 20, "SMA", 200),
    ("SMA", 50, "SMA", 200),
    ("EMA", 100, "SMA", 200),
    ("EMA", 10, "SMA", 200),
    ("EMA", 20, "SMA", 200),
    ("EMA", 50, "SMA", 200),
    ("SMA", 10, "SMA", 200),
    ("EMA", 100, "EMA", 200),
    ("SMA", 50, "EMA", 200),
)


def rule(family, value):
    return (
        trend.IndicatorRule(family, fast=value[0], slow=value[1], signal=value[2])
        if family == "MACD"
        else trend.IndicatorRule(family, period=value)
    )


def synthetic_trend():
    pairs = (*TREND_PAIRS, ("SMA", 250, "EMA", 250))
    candidates = [
        trend.Candidate(rule(ef, ep), rule(xf, xp), hours, hours)
        for ef, ep, xf, xp in pairs
        for hours in (2.0, 3.0, 4.0)
    ]
    selected = {c.candidate_id for c in candidates if c.entry_confirmation_hours == 3}
    selected.remove(candidates[-2].candidate_id)
    rows = []
    for candidate in candidates:
        score = 0.4 if candidate.candidate_id in selected else -0.3
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "entry_confirmation_hours": candidate.entry_confirmation_hours,
                "exit_confirmation_hours": candidate.exit_confirmation_hours,
                "cash_excess_sharpe": score,
                "BH_cash_excess_sharpe": 0.1,
                "Sharpe_difference_vs_BH": score - 0.1,
            }
        )
    return pd.DataFrame(rows), candidates


def synthetic_mr_selection():
    counts = {"SMA_DISTANCE": 6, "EMA_DISTANCE": 4, "ZSCORE": 5, "RSI": 17}
    selected = []
    for family, count in counts.items():
        selected.extend([c for c in mr.candidate_grid() if c.family == family][:count])
    selected.sort(key=lambda c: c.candidate_id)
    rows = [{"candidate_id": c.candidate_id, **c.to_dict()} for c in selected]
    return {
        "candidates": rows,
        "candidate_set_sha256": study.custody.canonical_hash(rows),
    }


def synthetic_membership():
    metrics, candidates = synthetic_trend()
    selected = study.select_trend_members(metrics, candidates)
    mean_reversion = study.select_mean_reversion_members(synthetic_mr_selection())
    return [
        {"candidate_id": c.candidate_id, "style": style, "parameters": c.to_dict()}
        for style, group in (("trend", selected), ("mean_reversion", mean_reversion))
        for c in group
    ]


def write_config(path, value=None):
    path.write_text(yaml.safe_dump(study.approved_config() if value is None else value))
    return path


def write_authorization(path, **changes):
    payload = {
        "schema": "qqq_strategy_portfolio_authorization",
        "authorized_by": "human_user",
        "explicit_user_approval": True,
        "authorized_stages": ["development", "validation"],
        "approval_excerpt": study.APPROVAL_EXCERPT,
        "configuration_count": 44,
        "trend_count": 12,
        "mean_reversion_count": 32,
        "shared_confirmation_hours": 3.0,
        "development_period": dict(study.PERIODS["development"]),
        "validation_period": dict(study.PERIODS["validation"]),
        "historical_oos_authorized": False,
        "no_constituent_retuning_or_replacement": True,
        **changes,
    }
    path.write_text(json.dumps(payload))
    return path


def test_approved_config_has_twelve_fixed_sleeve_allocations_and_no_rebalancing(tmp_path):
    config = study.load_config(write_config(tmp_path / "portfolio.yaml"))
    assert config == study.approved_config()
    assert config["portfolio"]["allocation_count"] == 12
    assert config["portfolio"]["trend_allocations"] == [i / 10 for i in range(11)]
    assert config["portfolio"]["additional_allocation"] == "equal44"
    assert config["portfolio"]["construction"] == "fixed_initial_capital_sleeves_no_rebalancing"
    assert config["portfolio"]["order_netting"] is False
    assert config["membership"]["total_count"] == 44
    assert config["bootstrap"] == {
        "replicates": 2000,
        "block_lengths": [5, 20, 60],
        "seed": 20261009,
        "paired": True,
        "confidence_level": 0.95,
    }
    assert config["protocol"]["historical_oos_authorized"] is False
    assert config["protocol"]["hindsight_conditioned_membership"] is True


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("membership", "total_count", 20),
        ("membership", "required_checks", 6),
        ("membership", "shared_confirmation_hours", 4.0),
        ("portfolio", "allow_pure_style", 1),
        ("portfolio", "construction", "constant_daily_weights"),
        ("portfolio", "order_netting", True),
        ("accounting", "cash_nominal_annual_rate", 0.04),
        ("accounting", "commission_bps", True),
        ("accounting", "taxes", 0),
        ("accounting", "matched_terminal_liquidation", 1),
        ("selection", "standard_deviation_ddof", 0),
        ("selection", "minimum_trade_filter", True),
        ("selection", "baseline_only", 1),
        ("validation", "no_reselection", False),
        ("bootstrap", "replicates", 1000),
        ("bootstrap", "paired", 1),
        ("protocol", "historical_oos_authorized", 0),
        ("protocol", "historical_oos_authorized", True),
        ("protocol", "hindsight_conditioned_membership", False),
    ],
)
def test_config_refuses_policy_changes_or_numeric_boolean_substitution(
    tmp_path, section, key, value
):
    config = copy.deepcopy(study.approved_config())
    config[section][key] = value
    with pytest.raises(study.Error, match="contract"):
        study.load_config(write_config(tmp_path / "changed.yaml", config))


def test_config_refuses_extra_allocation_or_extended_period(tmp_path):
    config = copy.deepcopy(study.approved_config())
    config["portfolio"]["trend_allocations"].append(0.55)
    with pytest.raises(study.Error):
        study.load_config(write_config(tmp_path / "allocation.yaml", config))
    config = copy.deepcopy(study.approved_config())
    config["periods"]["validation"]["end"] = "2019-01-01"
    with pytest.raises(study.Error):
        study.load_config(write_config(tmp_path / "period.yaml", config))


def test_explicit_authorization_closes_2019_and_preserves_both_segments(tmp_path):
    value = study.authorization(write_authorization(tmp_path / "authorization.json"))
    assert value["historical_oos_authorized"] is False
    assert value["authorized_stages"] == ["development", "validation"]
    assert value["no_constituent_retuning_or_replacement"] is True


@pytest.mark.parametrize(
    "changes",
    [
        {"authorized_by": "agent"},
        {"explicit_user_approval": 1},
        {"explicit_user_approval": False},
        {"authorized_stages": ["development"]},
        {"authorized_stages": ["development", "validation", "historical_oos"]},
        {"approval_excerpt": "run anything"},
        {"configuration_count": 20},
        {"trend_count": 11},
        {"mean_reversion_count": 31},
        {"shared_confirmation_hours": 2},
        {"historical_oos_authorized": 0},
        {"historical_oos_authorized": True},
        {"no_constituent_retuning_or_replacement": 1},
        {"development_period": {"start": "2002-01-01", "end": "2011-12-31"}},
        {"validation_period": {"start": "2012-01-01", "end": "2019-01-01"}},
    ],
)
def test_authorization_rejects_wrong_stage_period_count_wait_or_boolean(tmp_path, changes):
    with pytest.raises(study.Error):
        study.authorization(write_authorization(tmp_path / "authorization.json", **changes))


def test_trend_membership_exact_known_twelve_at_three_hours_and_shuffle_invariant():
    metrics, canonical = synthetic_trend()
    before = study.select_trend_members(metrics, canonical)
    after = study.select_trend_members(metrics.sample(frac=1, random_state=44), canonical)
    expected = {
        trend.Candidate(rule(ef, ep), rule(xf, xp), 3, 3).candidate_id
        for ef, ep, xf, xp in TREND_PAIRS
    }
    assert {c.candidate_id for c in before} == expected
    assert [c.candidate_id for c in before] == sorted(expected)
    assert [c.candidate_id for c in before] == [c.candidate_id for c in after]
    assert all(c.entry_required_checks == c.exit_required_checks == 7 for c in before)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "unknown",
        "changed_wait",
        "wrong_BH_delta",
        "undefined",
        "only11",
        "13",
    ],
)
def test_trend_membership_refuses_modified_or_nonfinite_selection(change):
    metrics, canonical = synthetic_trend()
    selected_index = metrics.index[
        (metrics.cash_excess_sharpe > 0) & metrics.entry_confirmation_hours.eq(3)
    ][0]
    if change == "missing":
        metrics = metrics.iloc[:-1]
    elif change == "duplicate":
        metrics.loc[1, "candidate_id"] = metrics.loc[0, "candidate_id"]
    elif change == "unknown":
        metrics.loc[0, "candidate_id"] = "unapproved"
    elif change == "changed_wait":
        metrics.loc[selected_index, "entry_confirmation_hours"] = 2.0
    elif change == "wrong_BH_delta":
        metrics.loc[selected_index, "BH_cash_excess_sharpe"] = 0.2
    elif change == "undefined":
        metrics.loc[selected_index, "cash_excess_sharpe"] = np.nan
    elif change == "only11":
        metrics.loc[selected_index, ["cash_excess_sharpe", "Sharpe_difference_vs_BH"]] = [
            -0.3,
            -0.4,
        ]
    else:
        index = metrics.index[metrics.candidate_id.eq(canonical[-2].candidate_id)][0]
        metrics.loc[index, ["cash_excess_sharpe", "Sharpe_difference_vs_BH"]] = [0.4, 0.3]
    with pytest.raises(study.Error):
        study.select_trend_members(metrics, canonical)


def test_mean_reversion_membership_has_thirty_two_not_twenty_finalists():
    selected = study.select_mean_reversion_members(synthetic_mr_selection())
    assert len(selected) == 32
    assert pd.Series([c.family for c in selected]).value_counts().to_dict() == {
        "RSI": 17,
        "SMA_DISTANCE": 6,
        "ZSCORE": 5,
        "EMA_DISTANCE": 4,
    }
    assert all(c.entry_required_checks == c.exit_required_checks == 7 for c in selected)


@pytest.mark.parametrize("change", ["20", "duplicate", "parameters", "hash"])
def test_mr_membership_refuses_shortlist_or_modified_definitions(change):
    selection = synthetic_mr_selection()
    if change == "20":
        selection["candidates"] = selection["candidates"][:20]
    elif change == "duplicate":
        selection["candidates"][1] = copy.deepcopy(selection["candidates"][0])
    elif change == "parameters":
        selection["candidates"][0]["confirmation_hours"] = 2.0
    else:
        selection["candidate_set_sha256"] = "tampered"
    if change != "hash":
        selection["candidate_set_sha256"] = study.custody.canonical_hash(selection["candidates"])
    with pytest.raises(study.Error):
        study.select_mean_reversion_members(selection)


def test_combined_membership_has_exact_canonical_style_order():
    rows = synthetic_membership()
    assert study.validate_membership(rows) == rows
    assert len(rows) == 44
    assert [r["style"] for r in rows] == ["trend"] * 12 + ["mean_reversion"] * 32


@pytest.mark.parametrize(
    "change",
    ["missing", "duplicate", "unknown_id", "style", "order", "extra_field", "wait", "rebound"],
)
def test_combined_membership_refuses_changed_count_order_definitions(change):
    rows = copy.deepcopy(synthetic_membership())
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[1] = copy.deepcopy(rows[0])
    elif change == "unknown_id":
        rows[0]["candidate_id"] = "replacement"
    elif change == "style":
        rows[0]["style"] = "mean_reversion"
    elif change == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif change == "extra_field":
        rows[0]["weight"] = 0.5
    elif change == "wait":
        rows[0]["parameters"]["exit_confirmation_hours"] = 4.0
    else:
        rows[12]["parameters"]["rebound_required"] = True
    with pytest.raises(study.Error):
        study.validate_membership(rows)


def test_private_sidecar_requires_new_top_level_tree_and_external_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "ROOT", tmp_path)
    protected = tmp_path / ".cache/preserved"
    valid = tmp_path / ".cache/portfolio_research_synthetic"
    study._separate(valid, (protected,), tmp_path / "authorization.json")
    for path in (
        protected,
        protected / "nested",
        tmp_path / ".cache",
        tmp_path / ".cache/portfolio_research_synthetic/nested",
        tmp_path / "portfolio_research_synthetic",
    ):
        with pytest.raises(study.Error):
            study._separate(path, (protected,))
    for authorization in (valid / "authorization.json", protected / "authorization.json"):
        with pytest.raises(study.Error):
            study._separate(valid, (protected,), authorization)


def test_safe_path_refuses_symlink_and_symlink_ancestor(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "file.json").write_text("{}")
    link = tmp_path / "link"
    link.symlink_to(actual, target_is_directory=True)
    with pytest.raises(study.Error):
        study.safe_path(link / "file.json")
    with pytest.raises(study.Error):
        study.safe_path(link)


def test_no_public_oos_route_or_broad_stage_loader():
    source = inspect.getsource(study)
    assert "custody._load_stage(" not in source
    assert "custody._verify_prepared(" not in source
    for command in ("historical-oos", "historical_oos", "oos"):
        with pytest.raises(SystemExit):
            study.main([command])


def synthetic_allocation_metrics():
    from trend_following.research_portfolio_engine import allocation_grid

    return pd.DataFrame(
        {
            "portfolio_id": allocation.portfolio_id,
            "trend_weight": allocation.trend_weight,
            "mean_reversion_weight": 1 - allocation.trend_weight,
            "scenario_slippage_bps": slip,
            "cash_excess_sharpe": 0.4 if allocation.trend_weight == 0.6 else 0.2,
        }
        for allocation in allocation_grid()
        for slip in study.SLIPPAGES
    )


def test_selection_uses_baseline_only_and_includes_pure_styles():
    metrics = synthetic_allocation_metrics()
    metrics.loc[metrics.scenario_slippage_bps.ne(3), "cash_excess_sharpe"] = 100
    assert study.select_allocation_metrics(metrics).portfolio_id == "portfolio_tf_060"
    metrics.loc[
        metrics.portfolio_id.eq("portfolio_tf_100") & metrics.scenario_slippage_bps.eq(3),
        "cash_excess_sharpe",
    ] = 0.8
    assert study.select_allocation_metrics(metrics).portfolio_id == "portfolio_tf_100"
    metrics.loc[
        metrics.portfolio_id.eq("portfolio_tf_000") & metrics.scenario_slippage_bps.eq(3),
        "cash_excess_sharpe",
    ] = 0.9
    assert study.select_allocation_metrics(metrics).portfolio_id == "portfolio_tf_000"


def test_selection_ties_choose_near_half_then_stable_id():
    metrics = synthetic_allocation_metrics()
    metrics["cash_excess_sharpe"] = 0.2
    mask = metrics.scenario_slippage_bps.eq(3)
    metrics.loc[mask & metrics.portfolio_id.eq("portfolio_tf_040"), "cash_excess_sharpe"] = 1
    metrics.loc[mask & metrics.portfolio_id.eq("portfolio_tf_060"), "cash_excess_sharpe"] = (
        1 + study.TOLERANCE / 2
    )
    # Floating representability of |.4-.5| and |.6-.5| is identical.
    assert study.select_allocation_metrics(metrics).portfolio_id == "portfolio_tf_040"
    metrics.loc[mask & metrics.portfolio_id.eq("portfolio_tf_050"), "cash_excess_sharpe"] = 1
    assert study.select_allocation_metrics(metrics).portfolio_id == "portfolio_tf_050"
    metrics.loc[mask & metrics.portfolio_id.eq("portfolio_tf_060"), "cash_excess_sharpe"] = (
        1 + 2 * study.TOLERANCE
    )
    assert study.select_allocation_metrics(metrics).portfolio_id == "portfolio_tf_060"


@pytest.mark.parametrize("change", ["missing", "duplicate", "unknown", "no_finite", "no_baseline"])
def test_selection_rejects_missing_extra_modified_or_undefined_baseline_grid(change):
    metrics = synthetic_allocation_metrics()
    if change == "missing":
        metrics = metrics.loc[
            ~(metrics.portfolio_id.eq("portfolio_tf_000") & metrics.scenario_slippage_bps.eq(3))
        ]
    elif change == "duplicate":
        metrics = pd.concat(
            [metrics, metrics.loc[metrics.scenario_slippage_bps.eq(3)].iloc[:1]], ignore_index=True
        )
    elif change == "unknown":
        metrics.loc[metrics.portfolio_id.eq("portfolio_tf_000"), "portfolio_id"] = "unapproved"
    elif change == "no_finite":
        metrics.loc[metrics.scenario_slippage_bps.eq(3), "cash_excess_sharpe"] = np.nan
    else:
        metrics = metrics.loc[metrics.scenario_slippage_bps.ne(3)]
    with pytest.raises(study.Error):
        study.select_allocation_metrics(metrics)


def test_validation_permits_only_lock_and_four_predefined_controls_deduplicated():
    from trend_following.research_portfolio_engine import allocation_grid

    grid = {a.portfolio_id: a for a in allocation_grid()}
    expected_controls = {
        "portfolio_tf_100",
        "portfolio_tf_000",
        "portfolio_tf_050",
        "portfolio_equal_44",
    }
    for pid, allocation in grid.items():
        actual = study.eligible_validation_allocations({"allocation": allocation.to_dict()})
        assert actual[0].portfolio_id == pid
        assert {a.portfolio_id for a in actual} == expected_controls | {pid}
        assert len(actual) == (4 if pid in expected_controls else 5)
    for changed in (
        {"portfolio_id": "unapproved", "trend_weight": 0.55, "mean_reversion_weight": 0.45},
        {"portfolio_id": "portfolio_tf_050", "trend_weight": 0.6, "mean_reversion_weight": 0.4},
        {"portfolio_id": "portfolio_tf_050", "trend_weight": 0.5},
    ):
        with pytest.raises(study.Error):
            study.eligible_validation_allocations(changed)


@pytest.fixture
def synthetic_freeze_context(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "ROOT", tmp_path)
    sources = {
        k: str(tmp_path / ".cache" / k)
        for k in (
            "trend_development",
            "trend_validation",
            "mean_reversion_development",
            "mean_reversion_validation",
        )
    }
    config = write_config(tmp_path / "config.yaml")
    auth = write_authorization(tmp_path / "authorization.json")
    members = synthetic_membership()
    payload = {
        "schema": "qqq_strategy_portfolio_freeze",
        "config_path": str(config),
        "authorization": {"path": str(auth), "sha256": study.custody.sha256_file(auth)},
        "source_paths": sources,
        "membership": members,
        "membership_sha256": study.custody.canonical_hash(members),
        "historical_oos_authorized": False,
        "hindsight_conditioned_membership": True,
    }
    monkeypatch.setattr(study, "_bindings", lambda *_: copy.deepcopy(payload))
    sidecar = tmp_path / ".cache/portfolio_research_synthetic"
    kwargs = {
        "authorization": auth,
        "config_path": config,
        **{k: Path(v) for k, v in sources.items()},
    }
    return sidecar, kwargs, payload


def test_freeze_seals_before_any_price_reader_and_is_idempotent(
    synthetic_freeze_context, monkeypatch
):
    sidecar, kwargs, payload = synthetic_freeze_context
    monkeypatch.setattr(
        study, "load_packet", lambda *_: pytest.fail("No market packet reader during freeze")
    )
    first = study.freeze_study(sidecar, **kwargs)
    assert first["membership"] == payload["membership"]
    assert first["historical_oos_authorized"] is False
    assert study.freeze_study(sidecar, **kwargs) == first
    assert study.verify_study(sidecar) == first
    assert (sidecar / "freeze.json").is_file()


def test_freeze_refuses_overwrite_or_changed_source_membership(synthetic_freeze_context):
    sidecar, kwargs, payload = synthetic_freeze_context
    first = study.freeze_study(sidecar, **kwargs)
    payload["membership"][0]["parameters"]["exit_confirmation_hours"] = 4
    with pytest.raises(study.Error, match="frozen differently"):
        study.freeze_study(sidecar, **kwargs)
    with pytest.raises(study.Error, match="changed"):
        study.verify_study(sidecar)
    assert study.custody._read_seal(sidecar / "freeze.json") == first


@pytest.mark.parametrize("stage", ["development", "validation"])
def test_missing_freeze_blocks_packet_access(tmp_path, monkeypatch, stage):
    calls = []
    monkeypatch.setattr(
        study, "verify_study", lambda *_: (_ for _ in ()).throw(study.Error("freeze missing"))
    )
    monkeypatch.setattr(study, "load_packet", lambda *_: calls.append("prices"))
    with pytest.raises(study.Error, match="freeze missing"):
        study._evaluate_stage(tmp_path / "study", stage)
    assert calls == []


def test_missing_development_lock_blocks_validation_price_reader(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(study, "verify_study", lambda *_: {"frozen_at": "2001-01-01T00:00Z"})
    monkeypatch.setattr(
        study,
        "verify_locked_allocation",
        lambda *_: (_ for _ in ()).throw(study.Error("lock missing")),
    )
    monkeypatch.setattr(study, "load_packet", lambda *_: calls.append("prices"))
    with pytest.raises(study.Error, match="lock missing"):
        study.evaluate_validation(tmp_path / "study")
    assert calls == []


@pytest.mark.parametrize("stage", ["oos", "historical_oos", "historical-oos", "2019-2026"])
def test_unapproved_stage_stops_before_prices_or_existing_result_verifier(
    tmp_path, monkeypatch, stage
):
    calls = []
    monkeypatch.setattr(study, "verify_study", lambda *_: calls.append("freeze"))
    monkeypatch.setattr(
        study.fixed, "load_development_packet", lambda *_: calls.append("development_prices")
    )
    monkeypatch.setattr(
        study.prefix, "read_validation_prefix", lambda *_a, **_k: calls.append("validation_prices")
    )
    for fn, args in (
        (study.load_packet, ({}, stage)),
        (study._evaluate_stage, (tmp_path / "study", stage)),
        (study.verify_results, (tmp_path / "study", stage)),
    ):
        with pytest.raises(study.Error):
            fn(*args)
    assert calls == []


def test_lock_is_derived_from_development_baseline_and_cannot_be_replaced(tmp_path, monkeypatch):
    from trend_following.research_portfolio_engine import Allocation

    sidecar = tmp_path / "study"
    (sidecar / "development").mkdir(parents=True)
    synthetic_allocation_metrics().to_csv(sidecar / "development/metrics.csv", index=False)
    seal = {"finished_at": "2001-01-01T00:00Z", "stage": "development"}
    calls = []
    monkeypatch.setattr(study, "verify_results", lambda _p, stage: calls.append(stage) or seal)
    with pytest.raises(study.Error, match="baseline winner"):
        study.seal_locked_allocation(sidecar, Allocation("portfolio_tf_050", 0.5))
    value = study.seal_locked_allocation(sidecar)
    assert value["allocation"]["portfolio_id"] == "portfolio_tf_060"
    assert value["validation_reselection"] is False
    assert value["historical_oos_authorized"] is False
    assert study.verify_locked_allocation(sidecar) == value
    assert set(calls) == {"development"}


@pytest.mark.parametrize(
    "change",
    ["allocation", "baseline_cost", "OOS", "reselection", "development_seal", "chronology"],
)
def test_changed_lock_cannot_unlock_validation(tmp_path, monkeypatch, change):
    from trend_following.research_portfolio_engine import Allocation

    sidecar = tmp_path / "study"
    (sidecar / "development").mkdir(parents=True)
    synthetic_allocation_metrics().to_csv(sidecar / "development/metrics.csv", index=False)
    development_seal = {"finished_at": "2001-01-01T00:00Z", "stage": "development"}
    monkeypatch.setattr(study, "verify_results", lambda *_: development_seal)
    lock = study.seal_locked_allocation(sidecar)
    if change == "allocation":
        lock["allocation"] = Allocation("portfolio_tf_050", 0.5).to_dict()
    elif change == "baseline_cost":
        lock["baseline_slippage_bps"] = 1
    elif change == "OOS":
        lock["historical_oos_authorized"] = True
    elif change == "reselection":
        lock["validation_reselection"] = True
    elif change == "development_seal":
        lock["development_result_sha256"] = "tampered"
    else:
        lock["sealed_at"] = "2000-12-31T00:00Z"
    monkeypatch.setattr(study.custody, "_read_seal", lambda *_: lock)
    with pytest.raises(study.Error):
        study.verify_locked_allocation(sidecar)


@pytest.fixture
def derived_directory(tmp_path):
    """Three fabricated session marks; no source simulation or price files."""
    from dataclasses import replace

    from test_portfolio_engine import synthetic_paths
    from trend_following.research_portfolio_engine import allocation_grid, evaluate_portfolio

    directory = tmp_path / "derived"
    directory.mkdir()
    membership = synthetic_membership()
    base = synthetic_paths()
    base = replace(
        base,
        candidate_ids=tuple(r["candidate_id"] for r in membership),
        metrics=tuple(
            {**r, "candidate_id": member["candidate_id"]}
            for r, member in zip(base.metrics, membership, strict=True)
        ),
    )
    allocations = list(allocation_grid())
    rows = []
    bench = []
    for slip in study.SLIPPAGES:
        paths = replace(base, metrics=tuple({**r, "slippage_bps": slip} for r in base.metrics))
        study._save_sleeves(directory / f"constituent_s{slip:g}_daily.npz", paths)
        results = [evaluate_portfolio(paths, allocation) for allocation in allocations]
        study._save_portfolios(
            directory / f"allocation_s{slip:g}_daily.npz", results, paths.cash_returns
        )
        for result in results:
            rows.append({**result.metrics, "scenario_slippage_bps": slip})
        for mode, equity in (
            ("cash", np.full(3, 1000.0)),
            ("buy_and_hold", np.array([1010.0, 990.0, 1020.0])),
        ):
            ret = equity / np.r_[1000.0, equity[:-1]] - 1
            cash = paths.cash_returns
            sd = np.std(ret - cash, ddof=1)
            sharpe = np.sqrt(252) * np.mean(ret - cash) / sd if sd > 0 else np.nan
            bench.append(
                {
                    "portfolio_id": mode,
                    "scenario_slippage_bps": slip,
                    "terminal_equity": equity[-1],
                    "cash_excess_sharpe": sharpe,
                }
            )
            pd.DataFrame(
                {"timestamp": paths.times, "equity": equity, "return": ret, "cash_return": cash}
            ).to_csv(directory / f"{mode}_s{slip:g}_daily.csv", index=False)
    pd.DataFrame(rows).to_csv(directory / "metrics.csv", index=False)
    pd.DataFrame(bench).to_csv(directory / "benchmark_metrics.csv", index=False)
    return directory, {"membership": membership}, allocations


def test_derived_review_accepts_all_twelve_allocations_four_costs_and_eight_passives(
    derived_directory,
):
    directory, frozen, allocations = derived_directory
    result = study.review_derived_results(directory, frozen, "development", allocations)
    assert result["portfolio_scenarios_checked"] == 48
    assert result["benchmark_scenarios_checked"] == 8
    assert result["exact_clock_cash_and_scaled_sleeve_wealth"] is True
    assert result["historical_oos_authorized"] is False


@pytest.mark.parametrize(
    "change",
    [
        "wealth",
        "implicit_rebalanced_returns",
        "cash_clock",
        "2019",
        "constituent_ids",
        "duplicate_metric",
        "changed_weights",
        "commission",
        "passive_cash",
    ],
)
def test_derived_review_rejects_bad_aggregation_clock_membership_accounting(
    derived_directory, change
):
    directory, frozen, allocations = derived_directory
    file = directory / "allocation_s3_daily.npz"
    with np.load(file, allow_pickle=False) as stored:
        arrays = {k: stored[k].copy() for k in stored.files}
    metrics = pd.read_csv(directory / "metrics.csv", float_precision="round_trip")
    if change == "wealth":
        arrays["equity"][1, 0] += 1
    elif change == "implicit_rebalanced_returns":
        with np.load(directory / "constituent_s3_daily.npz", allow_pickle=False) as sleeves:
            j = list(arrays["portfolio_id"]).index("portfolio_tf_050")
            arrays["returns"][:, j] = np.sum(sleeves["returns"] * allocations[j].weights(), axis=1)
    elif change == "cash_clock":
        arrays["cash_returns"][1] = 0.001
    elif change == "2019":
        arrays["timestamp"][-1] = "2019-01-02T22:00:00+00:00"
    elif change == "constituent_ids":
        path = directory / "constituent_s3_daily.npz"
        with np.load(path, allow_pickle=False) as saved:
            data = {k: saved[k].copy() for k in saved.files}
        data["candidate_id"] = data["candidate_id"][::-1]
        np.savez_compressed(path, **data)
    elif change == "duplicate_metric":
        metrics = pd.concat([metrics, metrics.iloc[:1]], ignore_index=True)
    elif change == "changed_weights":
        metrics.loc[
            metrics.portfolio_id.eq("portfolio_tf_050") & metrics.scenario_slippage_bps.eq(3),
            "trend_weight",
        ] = 0.6
    elif change == "commission":
        metrics.loc[metrics.scenario_slippage_bps.eq(3), "commission_bps"] = 2
    else:
        path = directory / "cash_s3_daily.csv"
        frame = pd.read_csv(path, float_precision="round_trip")
        frame.loc[1, "cash_return"] = 0.001
        frame.to_csv(path, index=False)
    np.savez_compressed(file, **arrays)
    metrics.to_csv(directory / "metrics.csv", index=False)
    with pytest.raises(study.Error):
        study.review_derived_results(directory, frozen, "development", allocations)
