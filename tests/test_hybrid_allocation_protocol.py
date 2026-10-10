"""Synthetic custody, ranking and physically closed-stage hybrid checks."""

from __future__ import annotations

import copy
import inspect
import json
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from trend_following import research_daily_hourly_v5 as tf
from trend_following import research_hybrid_allocation_protocol as p
from trend_following import research_mean_reversion as mr


def membership():
    trends = [
        tf.Candidate(tf.IndicatorRule(f, period=n), tf.IndicatorRule("SMA", period=200), 3, 3)
        for f in ("SMA", "EMA")
        for n in (10, 20, 50, 100, 200, 250)
    ]
    reversions = mr.candidate_grid()[:32]
    return [
        {"style": style, "candidate_id": c.candidate_id, "parameters": c.to_dict()}
        for style, group in (("trend", trends), ("mean_reversion", reversions))
        for c in sorted(group, key=lambda c: c.candidate_id)
    ]


def config_file(path, value=None):
    path.write_text(yaml.safe_dump(p.approved_config() if value is None else value))
    return path


def auth_value():
    return {
        "schema": "qqq_hybrid_allocation_authorization",
        "authorized_by": "human_user",
        "explicit_user_approval": True,
        "approval_excerpt": p.APPROVAL_EXCERPT,
        "authorized_stages": ["development", "validation"],
        "trend_count": 12,
        "mean_reversion_count": 32,
        "pair_count": 384,
        "principal_count": 8,
        "shared_confirmation_hours": 3.0,
        "required_checks": 7,
        "development_period": p.PERIODS["development"],
        "validation_period": p.PERIODS["validation"],
        "initial_trend_allocations": [0.5, 0.7, 0.9, 1.0],
        "architectures": ["A", "B", "C"],
        "regime_rule": p.REGIME_RULE,
        "historical_oos_authorized": False,
        "no_constituent_retuning_or_replacement": True,
        "reuse_sealed_controls": True,
    }


def score_table(scores=None):
    if scores is None:
        scores = dict(
            zip(
                [d["portfolio_id"] for d in p.principal_definitions()],
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                strict=True,
            )
        )
    return pd.DataFrame(
        [
            {
                **d,
                "scenario_slippage_bps": s,
                "cash_excess_sharpe": scores[d["portfolio_id"]] if s == 3 else 99,
            }
            for d in p.principal_definitions()
            for s in p.SLIPPAGES
        ]
    )


def test_exact_principal_grid_and_candidate_pairs():
    config = p.approved_config()
    assert config["membership"]["pair_count"] == 384
    assert config["core"]["funded_trend_accounts"] == 12
    assert config["core"]["funded_tactical_accounts"] == 384
    assert config["portfolio"]["pair_result_count_per_cost"] == 2688
    assert len({d["portfolio_id"] for d in p.principal_definitions()}) == 8
    pairs = p._pair_definitions(membership())
    assert len(pairs) == len({x["pair_id"] for x in pairs}) == 384
    assert {(x["tf_index"], x["mr_index"]) for x in pairs} == {
        (i, j) for i in range(12) for j in range(32)
    }
    assert all(
        set(x) == {"pair_id", "tf_candidate_id", "mr_candidate_id", "tf_index", "mr_index"}
        for x in pairs
    )


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("membership", "pair_count", 32),
        ("membership", "pair_selection", True),
        ("core", "funded_trend_accounts", 384),
        ("core", "rebalance", True),
        ("core", "MR_gate", "SMA200"),
        ("core", "TF_exit_forces_MR_at_same_safe_open", False),
        ("blend", "sizing", "pre_cost_equity"),
        ("blend", "maintenance_rebalance", True),
        ("regime", "lower", 0.2),
        ("regime", "upper", 0.4),
        ("regime", "period_days", 10),
        ("regime", "availability", "intraday_preview"),
        ("regime", "initial_mode", "MR"),
        ("regime", "threshold_selection", True),
        ("execution", "required_checks", 6),
        ("accounting", "cash_nominal_annual_rate", 0.04),
        ("accounting", "commission_bps", True),
        ("accounting", "taxes", 1),
        ("selection", "filters", True),
        ("selection", "finite_only", 1),
        ("protocol", "historical_oos_authorized", True),
        ("protocol", "historical_oos_authorized", 0),
    ],
)
def test_config_refuses_rule_cost_or_boolean_substitution(tmp_path, section, key, value):
    config = p.approved_config()
    config[section][key] = value
    with pytest.raises(p.Error, match="contract"):
        p.load_config(config_file(tmp_path / "changed.yaml", config))


def test_config_and_authorization_accept_exact_human_contract(tmp_path):
    cfg = config_file(tmp_path / "config.yaml")
    assert p.load_config(cfg) == p.approved_config()
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps(auth_value()))
    assert p.authorization(auth) == auth_value()


@pytest.mark.parametrize(
    "key,value",
    [
        ("authorized_by", "agent"),
        ("explicit_user_approval", 1),
        ("pair_count", 90),
        ("principal_count", 9),
        ("shared_confirmation_hours", 2),
        ("initial_trend_allocations", [0.5, 0.7]),
        ("historical_oos_authorized", True),
        ("reuse_sealed_controls", False),
        ("regime_rule", {"period_days": 10}),
        ("authorized_stages", ["development"]),
        ("no_constituent_retuning_or_replacement", False),
    ],
)
def test_authorization_rejects_changed_or_missing_fields(tmp_path, key, value):
    auth = auth_value()
    auth[key] = value
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(auth))
    with pytest.raises(p.Error, match="human"):
        p.authorization(path)


def test_each_architecture_can_lock_same_pure_trend_without_duplicate_error():
    selected = p.select_architecture_winners(score_table())
    assert [r["architecture"] for r in selected] == ["A", "B"]
    assert [r["portfolio_id"] for r in selected] == [p.PURE_TREND_ID] * 2
    assert [r["trend_allocation"] for r in selected] == [1.0, 1.0]


def test_architecture_selection_uses_baseline_only_and_ties_stable_ids():
    scores = {d["portfolio_id"]: 0.1 for d in p.principal_definitions()}
    scores["hybrid_core_050"] = 0.9
    scores["hybrid_core_070"] = 0.9 + 5e-11
    scores["hybrid_blend_090"] = 0.8
    winners = p.select_architecture_winners(score_table(scores))
    assert [x["portfolio_id"] for x in winners] == ["hybrid_core_050", "hybrid_blend_090"]
    scores["hybrid_core_070"] = 0.9 + 2e-10
    assert (
        p.select_architecture_winners(score_table(scores))[0]["portfolio_id"] == "hybrid_core_070"
    )


def test_no_return_trade_or_BH_filters_and_no_C_selection():
    frame = score_table()
    frame["trade_count"] = 0
    frame["cagr"] = -0.9
    frame["max_drawdown"] = -0.99
    frame["beats_BH_Sharpe"] = False
    frame.loc[frame.portfolio_id.eq(p.REGIME_ID), "cash_excess_sharpe"] = 1000
    winners = p.select_architecture_winners(frame)
    assert all(x["portfolio_id"] != p.REGIME_ID for x in winners)


@pytest.mark.parametrize(
    "change", ["drop", "duplicate", "new_id", "weight", "arch", "none_A", "none_B"]
)
def test_selection_refuses_incomplete_grid_and_undefined_required_architecture(change):
    frame = score_table()
    base = frame.scenario_slippage_bps.eq(3)
    if change == "drop":
        frame = frame.drop(frame.index[base][0])
    elif change == "duplicate":
        frame = pd.concat([frame, frame.loc[base].iloc[:1]], ignore_index=True)
    elif change == "new_id":
        frame.loc[base & frame.portfolio_id.eq("hybrid_core_050"), "portfolio_id"] = "invented"
    elif change == "weight":
        frame.loc[base & frame.portfolio_id.eq("hybrid_core_050"), "trend_allocation"] = 0.51
    elif change == "arch":
        frame.loc[base & frame.portfolio_id.eq("hybrid_core_050"), "architecture"] = "B"
    else:
        arch = "A" if change == "none_A" else "B"
        frame.loc[base & frame.architecture.isin([arch, "pure_trend"]), "cash_excess_sharpe"] = (
            np.nan
        )
    with pytest.raises(p.Error):
        p.select_architecture_winners(frame)


def test_binding_content_changes_rejected_even_same_size(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "ROOT", tmp_path)
    file = tmp_path / "original_seal.json"
    file.write_text("original")
    bindings = p._file_bindings([file])
    p._verify_files(bindings)
    file.write_text("modified")
    with pytest.raises(p.Error, match="changed"):
        p._verify_files(bindings)


def test_binding_rejects_duplicates_symlinks_and_extra_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "ROOT", tmp_path)
    file = tmp_path / "source.py"
    file.write_text("source")
    bindings = p._file_bindings([file])
    with pytest.raises(p.Error):
        p._verify_files(bindings * 2)
    extra = copy.deepcopy(bindings)
    extra[0]["skip_hash"] = True
    with pytest.raises(p.Error):
        p._verify_files(extra)
    link = tmp_path / "link.py"
    link.symlink_to(file)
    with pytest.raises(p.Error):
        p._file_bindings([link])


def test_protected_hashes_reject_any_old_source_or_metadata_exemption(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "ROOT", tmp_path)
    old = tmp_path / "old.py"
    old.write_text("old")
    snapshot = tmp_path / "protected.json"
    snapshot.write_text(json.dumps({"old.py": p.custody.sha256_file(old)}))
    assert len(p._protected(snapshot)) == 1
    old.write_text("new")
    with pytest.raises(p.Error):
        p._protected(snapshot)
    snapshot.write_text(json.dumps({"README.md": "a" * 64}))
    with pytest.raises(p.Error):
        p._protected(snapshot)


def test_profiler_restored_after_verifier_failure_and_no_function_replacement(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(p, "ROOT", tmp_path)
    metadata = tmp_path / "metadata.json"
    metadata.write_text("{}")
    original = p.multi.verify_results

    def fake(_source, _stage):
        p.custody.sha256_file(metadata)
        metadata.read_text()
        raise p.Error("failed verification")

    monkeypatch.setattr(p.multi, "verify_results", fake)
    prior = sys.getprofile()

    def marker(*_):
        return None

    sys.setprofile(marker)
    try:
        with pytest.raises(p.Error, match="failed verification"):
            p._capture_verified_closure(tmp_path / "preserved")
        assert sys.getprofile() is marker
        assert p.multi.verify_results is fake
    finally:
        sys.setprofile(prior)
    assert original is not fake


def test_observer_blocks_OOS_csv_before_reader_opens_it(monkeypatch, tmp_path):
    monkeypatch.setattr(p, "ROOT", tmp_path)
    folder = tmp_path / "prepared/historical-oos"
    folder.mkdir(parents=True)
    later = folder / "bars.csv"
    later.write_text("would be later prices")
    called = []

    def fake(_source, _stage):
        p.custody.sha256_file(later)
        called.append("verification returned")

    monkeypatch.setattr(p.multi, "verify_results", fake)
    before = sys.getprofile()
    with pytest.raises(p.Error, match="later numeric"):
        p._capture_verified_closure(tmp_path)
    assert sys.getprofile() is before and not called


@pytest.mark.parametrize("stage", ["oos", "historical-oos", "2019", "live"])
def test_forbidden_stage_rejected_before_any_verifier_or_numeric_reader(monkeypatch, stage):
    monkeypatch.setattr(p, "verify_study", lambda _: pytest.fail("unexpected verification"))
    for operation in (
        lambda: p.load_packet({}, stage),
        lambda: p.verify_results(stage=stage),
        lambda: p._evaluate_stage(p.DEFAULT_SIDECAR, stage),
    ):
        with pytest.raises(p.Error):
            operation()


def test_new_validation_locks_required_before_any_new_price_read(monkeypatch):
    frozen = {"validation_packet": "bounded", "validation_metadata": {"manifest_sha256": "hash"}}
    monkeypatch.setattr(p, "verify_study", lambda _: frozen)

    def stop(_):
        raise p.Error("new locks missing")

    monkeypatch.setattr(p, "verify_locked_selections", stop)
    monkeypatch.setattr(
        p.prefix, "read_validation_prefix", lambda *_a, **_k: pytest.fail("premature price read")
    )
    with pytest.raises(p.Error, match="locks missing"):
        p.load_packet(frozen, "validation")


def test_caller_cannot_change_frozen_packet(monkeypatch):
    monkeypatch.setattr(p, "verify_study", lambda _: {"development_adapter": {"actual": "bound"}})
    monkeypatch.setattr(
        p.fixed, "load_development_packet", lambda *_: pytest.fail("unverified reader")
    )
    with pytest.raises(p.Error, match="Caller"):
        p.load_packet({"development_adapter": {}}, "development")


def generated_packet(session="2011-12-30"):
    opening = pd.Timestamp(session + " 09:30", tz="America/New_York").tz_convert("UTC")
    return (
        pd.DataFrame(
            {
                "session": [session],
                "daily_price_available_at": [opening + pd.Timedelta(hours=7, minutes=30)],
            }
        ),
        pd.DataFrame(
            {
                "session": [session],
                "bar_start": [opening],
                "bar_end": [opening + pd.Timedelta(minutes=30)],
            }
        ),
        pd.DataFrame(
            {
                "session": [session],
                "market_open": [opening],
                "market_close": [opening + pd.Timedelta(hours=6, minutes=30)],
            }
        ),
        pd.DataFrame(
            {
                "session": [session],
                "bar_start": [opening],
                "bar_end": [opening + pd.Timedelta(minutes=30)],
            }
        ),
        pd.DataFrame(),
        pd.DataFrame({"annual_rate": [0.03], "available_at": [opening - pd.Timedelta(days=1)]}),
        {},
    )


def test_bounded_numeric_session_and_timestamp_guards():
    p._assert_packet_bounds(generated_packet(), "development")
    with pytest.raises(p.Error):
        p._assert_packet_bounds(generated_packet("2012-01-03"), "development")
    with pytest.raises(p.Error):
        p._assert_packet_bounds(generated_packet("2019-01-02"), "validation")
    packet = generated_packet()
    packet[1].loc[0, "bar_end"] = pd.Timestamp("2012-01-02T15:00:00Z")
    with pytest.raises(p.Error, match="timestamp"):
        p._assert_packet_bounds(packet, "development")


def test_fixed_cash_rate_is_assumed_not_future_market_evidence():
    packet = generated_packet()
    packet[5].loc[0, "annual_rate"] = 0.04
    with pytest.raises(p.Error, match="3%"):
        p._assert_packet_bounds(packet, "development")


def test_locks_are_development_only_and_fixed_C_rule_sealed(monkeypatch, tmp_path):
    monkeypatch.setattr(
        p, "verify_results", lambda *_: {"finished_at": "2026-10-10T01:00:00+00:00"}
    )
    directory = tmp_path / "development"
    directory.mkdir()
    score_table().to_csv(directory / "metrics.csv", index=False)
    lock = p.seal_locked_selections(tmp_path)
    assert lock["fixed_regime_rule"] == p.REGIME_RULE
    assert [x["portfolio_id"] for x in lock["selections"]] == [p.PURE_TREND_ID] * 2
    assert p.verify_locked_selections(tmp_path) == lock
    path = tmp_path / "locked_selections.json"
    path.unlink()
    changed = copy.deepcopy(lock)
    changed["fixed_regime_rule"]["lower"] = 0.1
    p.custody._seal(path, changed)
    with pytest.raises(p.Error, match="locks"):
        p.verify_locked_selections(tmp_path)


def test_bootstrap_keeps_semantic_aliases_when_selected_paths_identical():
    x = np.linspace(-0.005, 0.008, 70)
    y = np.sin(np.arange(70)) * 0.008 + 0.001
    comparisons = [
        {
            "architecture": "A",
            "subject_id": "pure",
            "reference_id": "TF",
            "subject_returns": x,
            "reference_returns": x,
        },
        {
            "architecture": "B",
            "subject_id": "pure",
            "reference_id": "matchingA",
            "subject_returns": x.copy(),
            "reference_returns": x.copy(),
        },
        {
            "architecture": "C",
            "subject_id": "C",
            "reference_id": "BH",
            "subject_returns": y,
            "reference_returns": x,
        },
    ]
    out = p.bootstrap_comparisons(
        comparisons, np.full(70, 0.0001), replicates=25, blocks=(5, 20), seed=3
    )
    assert len(out) == 4
    first = out.loc[out.architecture.eq("A")]
    assert (first.observed_sharpe_difference == 0).all()
    assert (first.ci_lower == 0).all() and (first.ci_upper == 0).all()
    assert all(len(json.loads(value)) == 2 for value in first.comparison_aliases)
    pd.testing.assert_frame_equal(
        out,
        p.bootstrap_comparisons(
            comparisons, np.full(70, 0.0001), replicates=25, blocks=(5, 20), seed=3
        ),
    )


def test_cli_check_and_OOS_route_closed(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(p, "verify_study", lambda *_: pytest.fail("unfrozen verification"))
    assert (
        p.main(
            [
                "check",
                "--config",
                str(config_file(tmp_path / "config.yaml")),
                "--sidecar",
                str(tmp_path / "unfrozen"),
            ]
        )
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    assert out["pair_count"] == 384 and out["market_evaluation_performed"] is False
    with pytest.raises(SystemExit) as exc:
        p.main(["oos"])
    assert exc.value.code == 2


def test_new_interfaces_never_modify_or_monkeypatch_frozen_engines():
    source = inspect.getsource(p)
    assert "setattr(" not in source
    assert "sys.setprofile(previous)" in source
    assert "read_validation_prefix" in source
    assert "original_baseline_parity_verified" in source
    assert "with_constant_gate" not in source
    assert p.REGIME_RULE["availability"] == "finalized_daily_17_NY"


def test_full_new_freeze_verifier_rejects_bound_old_metadata_changes(monkeypatch, tmp_path):
    monkeypatch.setattr(p, "ROOT", tmp_path)
    source = tmp_path / ".cache/multi_scale_research_synthetic"
    old_root = tmp_path / ".cache/portfolio_research_synthetic"
    source.mkdir(parents=True)
    old_root.mkdir()
    data = tmp_path / ".cache/data"
    dev = data / "prepared/development"
    val = tmp_path / ".cache/validation_prefix"
    dev.mkdir(parents=True)
    val.mkdir()
    for directory, names in (
        (dev, (*p.fixed.PACKET_FILES, "manifest.json")),
        (val, (*p.prefix.PACKET_FILES, "manifest.json", "freeze.json")),
    ):
        for name in names:
            (directory / name).write_text("opaque already-verified fixture")
    cfgdir = tmp_path / "configs"
    cfgdir.mkdir()
    cfg = config_file(cfgdir / "qqq_hybrid_allocation_research.yaml")
    new_source = tmp_path / "new_source.py"
    new_source.write_text("new synthetic source")
    old_source = tmp_path / "old_source.py"
    old_source.write_text("preserved source")
    protected = tmp_path / "protected.json"
    protected.write_text(json.dumps({"old_source.py": p.custody.sha256_file(old_source)}))
    auth = tmp_path / "authorization.json"
    auth.write_text(json.dumps(auth_value()))
    rows = membership()
    old = {
        "membership": rows,
        "development_adapter": {"source_study": str(data), "config_path": str(cfg)},
        "validation_packet": str(val),
        "validation_metadata": {"manifest_sha256": "verified fixture"},
    }
    p.custody._seal(old_root / "freeze.json", old)
    inherited = {"preserved_portfolio": str(old_root), "membership": rows}
    seals = {
        "development": {"verified": True},
        "validation": {"verified": True},
        "locks": {"verified": True},
    }
    monkeypatch.setattr(
        p, "SOURCE_FILES", ("new_source.py", "configs/qqq_hybrid_allocation_research.yaml")
    )
    monkeypatch.setattr(
        p,
        "_capture_verified_closure",
        lambda _: (
            inherited,
            old,
            {"allocation": {"portfolio_id": p.PURE_TREND_ID}},
            seals,
            {old_root / "freeze.json"},
        ),
    )
    sidecar = tmp_path / ".cache/hybrid_allocation_research_synthetic"
    frozen = p.freeze_study(
        sidecar, authorization=auth, config_path=cfg, multi_scale=source, protected=protected
    )
    assert len(frozen["pairs"]) == 384
    assert frozen["protected_snapshot"]["count"] == 1
    assert p.verify_study(sidecar) == frozen
    assert (
        p.freeze_study(
            sidecar, authorization=auth, config_path=cfg, multi_scale=source, protected=protected
        )
        == frozen
    )
    changed = copy.deepcopy(old)
    changed["validation_metadata"]["manifest_sha256"] = "changed fixture"
    (old_root / "freeze.json").unlink()
    p.custody._seal(old_root / "freeze.json", changed)
    with pytest.raises(p.Error, match="changed"):
        p.verify_study(sidecar)
