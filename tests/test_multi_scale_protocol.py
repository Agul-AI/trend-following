"""Synthetic multi-scale custody, membership, selection and no-OOS tests.

No historical packets or existing strategy outcomes are opened here.
"""

from __future__ import annotations

import copy
import inspect
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from trend_following import research_daily_hourly_v5 as trend
from trend_following import research_mean_reversion as mr
from trend_following import research_multi_scale_engine as engine
from trend_following import research_multi_scale_protocol as study


def members():
    tf = [
        trend.Candidate(
            trend.IndicatorRule(f, period=n), trend.IndicatorRule("SMA", period=200), 3, 3
        )
        for f in ("SMA", "EMA")
        for n in (10, 20, 50, 100, 200, 250)
    ]
    tf.sort(key=lambda c: c.candidate_id)
    mrs = sorted(mr.candidate_grid()[:32], key=lambda c: c.candidate_id)
    return [
        {"candidate_id": c.candidate_id, "style": style, "parameters": c.to_dict()}
        for style, group in (("trend", tf), ("mean_reversion", mrs))
        for c in group
    ]


def definitions():
    return [
        {"candidate_id": c.candidate_id, "basket_id": c.basket_id, "parameters": c.to_dict()}
        for c in engine.candidate_grid(
            [mr.Candidate.from_dict(r["parameters"]) for r in members()[12:]]
        )
    ]


def metrics(scores=(0.3, 0.4, 0.5, 0.6)):
    return pd.DataFrame(
        [
            {
                **b,
                "scenario_slippage_bps": s,
                "cash_excess_sharpe": scores[i] if s == 3 else 100 - i,
            }
            for i, b in enumerate(study.baskets())
            for s in study.SLIPPAGES
        ]
    )


def auth_payload():
    return {
        "schema": "qqq_multi_scale_authorization",
        "authorized_by": "human_user",
        "explicit_user_approval": True,
        "authorized_stages": ["development", "validation"],
        "approval_excerpt": study.APPROVAL_EXCERPT,
        "configuration_count": 44,
        "trend_count": 12,
        "mean_reversion_count": 32,
        "filtered_rule_count": 128,
        "filtered_basket_count": 4,
        "shared_confirmation_hours": 3.0,
        "development_period": study.PERIODS["development"],
        "validation_period": study.PERIODS["validation"],
        "historical_oos_authorized": False,
        "no_constituent_retuning_or_replacement": True,
        "reuse_sealed_independent_portfolios": True,
    }


def write_config(path, value=None):
    path.write_text(yaml.safe_dump(value or study.approved_config()))
    return path


def test_configuration_pins_exact_four_baskets_and_daily_clock(tmp_path):
    c = study.load_config(write_config(tmp_path / "config.yaml"))
    assert c == study.approved_config()
    assert c["gate"]["periods_days"] == [100, 200]
    assert c["gate"]["availability"] == "finalized_daily_17_NY"
    assert c["membership"]["filtered_rule_count"] == 128
    assert c["construction"]["portfolio_rebalancing"] is False
    assert c["selection"]["one_filter_per_exit_policy"] is True
    assert c["protocol"]["historical_oos_authorized"] is False


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("membership", "mean_reversion_count", 20),
        ("membership", "required_checks", 6),
        ("membership", "shared_confirmation_hours", 4),
        ("gate", "intraday_preview", True),
        ("gate", "availability", "intraday_preview"),
        ("gate", "periods_days", [100, 250]),
        ("gate", "equality", "ON"),
        ("construction", "initial_cash", 10000),
        ("construction", "order_netting", True),
        ("construction", "portfolio_rebalancing", True),
        ("execution", "separate_recovery_and_trend_break_streaks", False),
        ("execution", "UNKNOWN_holds_cancels_resets", False),
        ("accounting", "cash_nominal_annual_rate", 0.04),
        ("accounting", "taxes", True),
        ("accounting", "commission_bps", True),
        ("selection", "tie_order", "SMA200"),
        ("selection", "standard_deviation_ddof", 0),
        ("selection", "finite_scores_only", 1),
        ("selection", "trade_return_drawdown_BH_filters", True),
        ("validation", "no_reselection", False),
        ("bootstrap", "replicates", 1000),
        ("bootstrap", "seed", 1),
        ("protocol", "historical_oos_authorized", True),
        ("protocol", "historical_oos_authorized", 0),
    ],
)
def test_exact_config_rejects_policy_or_boolean_substitution(tmp_path, section, key, value):
    c = study.approved_config()
    c[section][key] = value
    with pytest.raises(study.Error, match="contract"):
        study.load_config(write_config(tmp_path / "changed.yaml", c))


def test_authorization_exact_and_human(tmp_path):
    path = tmp_path / "auth.json"
    path.write_text(json.dumps(auth_payload()))
    assert study.authorization(path) == auth_payload()


@pytest.mark.parametrize(
    "field,value",
    [
        ("authorized_by", "agent"),
        ("explicit_user_approval", 1),
        ("authorized_stages", ["development"]),
        ("filtered_rule_count", 90),
        ("filtered_basket_count", 2),
        ("mean_reversion_count", 20),
        ("shared_confirmation_hours", 2),
        ("historical_oos_authorized", 0),
        ("historical_oos_authorized", True),
        ("reuse_sealed_independent_portfolios", False),
        ("no_constituent_retuning_or_replacement", False),
        ("approval_excerpt", "generic"),
    ],
)
def test_authorization_rejects_missing_or_changed_contract(tmp_path, field, value):
    p = auth_payload()
    p[field] = value
    path = tmp_path / "auth.json"
    path.write_text(json.dumps(p))
    with pytest.raises(study.Error, match="human"):
        study.authorization(path)


def test_membership_and_weights_are_four_unmodified_32_sleeve_groups():
    frozen = {"gated_candidates": definitions()}
    assert len(frozen["gated_candidates"]) == 128
    for b in study.baskets():
        w = study.basket_weights(frozen, b["basket_id"])
        assert len(w) == 128 and np.count_nonzero(w) == 32
        assert np.all(w[w > 0] == 1 / 32)
        assert w.sum() == 1
    with pytest.raises(study.Error):
        study.basket_weights(frozen, "unfrozen")


def test_weights_reject_changed_membership_count():
    d = definitions()
    d.pop()
    with pytest.raises(study.Error):
        study.basket_weights({"gated_candidates": d}, study.baskets()[-1]["basket_id"])


def test_selection_uses_only_baseline_one_winner_per_policy():
    winners = study.select_policy_winners(metrics())
    assert [x["gate_period"] for x in winners] == [200, 200]
    assert [x["exit_policy"] for x in winners] == list(study.POLICIES)


def test_selection_ties_within_tolerance_use_stable_id():
    winners = study.select_policy_winners(metrics((0.4, 0.4 + 5e-11, 0.6, 0.6)))
    assert [x["gate_period"] for x in winners] == [100, 100]
    winners = study.select_policy_winners(metrics((0.4, 0.4 + 2e-10, 0.6, 0.6 + 2e-10)))
    assert [x["gate_period"] for x in winners] == [200, 200]


def test_selection_retains_poor_or_low_trade_scores():
    frame = metrics((-10, -9, -8, -7))
    frame["trade_count"] = 0
    frame["cagr"] = -0.5
    frame["max_drawdown"] = -0.9
    frame["beats_BH_Sharpe"] = False
    assert [x["gate_period"] for x in study.select_policy_winners(frame)] == [200, 200]


@pytest.mark.parametrize(
    "scores", [(np.nan, np.nan, 0.2, 0.3), (0.1, 0.2, np.nan, np.inf), (-np.inf, np.nan, 0.3, 0.4)]
)
def test_selection_stops_when_either_policy_has_no_finite_score(scores):
    with pytest.raises(study.Error, match="finite"):
        study.select_policy_winners(metrics(scores))


@pytest.mark.parametrize("mutation", ["drop", "duplicate", "id", "period", "policy"])
def test_selection_rejects_incomplete_or_mutated_grid(mutation):
    m = metrics()
    if mutation == "drop":
        m = m.drop(m.index[m.scenario_slippage_bps.eq(3)][0])
    elif mutation == "duplicate":
        m = pd.concat([m, m.loc[m.scenario_slippage_bps.eq(3)].iloc[:1]], ignore_index=True)
    elif mutation == "id":
        m.loc[m.scenario_slippage_bps.eq(3) & m.gate_period.eq(100), "basket_id"] = "new"
    elif mutation == "period":
        m.loc[m.scenario_slippage_bps.eq(3), "gate_period"] = 250
    else:
        m.loc[m.scenario_slippage_bps.eq(3), "exit_policy"] = "pooled_exit"
    with pytest.raises(study.Error):
        study.select_policy_winners(m)


@pytest.mark.parametrize("stage", ["historical-oos", "oos", "2019", "live"])
def test_unauthorized_route_rejected_before_verifiers_or_readers(monkeypatch, stage):
    monkeypatch.setattr(study, "verify_study", lambda *_: pytest.fail("verification reached"))
    with pytest.raises(study.Error):
        study.load_packet({}, stage)
    with pytest.raises(study.Error):
        study.verify_results(stage=stage)
    with pytest.raises(study.Error):
        study._evaluate_stage(study.DEFAULT_SIDECAR, stage)


def test_validation_numeric_read_follows_both_policy_locks(monkeypatch):
    frozen = {
        "development_adapter": {},
        "validation_packet": "packet",
        "validation_metadata": {"manifest_sha256": "hash"},
    }
    events = []
    monkeypatch.setattr(study, "verify_study", lambda _: frozen)

    def stop(_):
        events.append("lock")
        raise study.Error("both policy locks missing")

    monkeypatch.setattr(study, "verify_locked_selections", stop)
    monkeypatch.setattr(
        study.prefix,
        "read_validation_prefix",
        lambda *_a, **_k: pytest.fail("unsealed validation numeric read"),
    )
    with pytest.raises(study.Error, match="locks"):
        study.load_packet(frozen, "validation")
    assert events == ["lock"]


def test_caller_cannot_change_verified_packet_contract(monkeypatch):
    monkeypatch.setattr(
        study, "verify_study", lambda _: {"development_adapter": {"source": "real"}}
    )
    monkeypatch.setattr(
        study.fixed, "load_development_packet", lambda _: pytest.fail("tampered numeric read")
    )
    with pytest.raises(study.Error, match="Caller"):
        study.load_packet({"development_adapter": {"source": "other"}}, "development")


def test_development_uses_only_bounded_existing_adapter(monkeypatch):
    frozen = {"development_adapter": {"source": "bounded"}}
    monkeypatch.setattr(study, "verify_study", lambda _: frozen)
    monkeypatch.setattr(
        study,
        "verify_locked_selections",
        lambda _: pytest.fail("validation lock during development"),
    )
    monkeypatch.setattr(study.fixed, "load_development_packet", lambda x: ("bounded", x))
    assert study.load_packet(frozen, "development") == ("bounded", frozen["development_adapter"])


def test_stage_artifacts_are_closed_and_include_gate_and_full_records():
    dev = study._required_files("development")
    assert "gate_history.csv" in dev
    assert "preflight.json" in dev
    assert "bootstrap.csv" not in dev
    assert study._required_files("validation") == dev | {"bootstrap.csv"}
    for slip in study.SLIPPAGES:
        assert f"records_s{slip:g}.csv" in dev
        assert f"gate_only_records_s{slip:g}.csv" in dev
        assert f"post_gap_s{slip:g}.csv" in dev


def test_cli_check_is_non_performance_and_has_no_oos(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(study, "verify_study", lambda _: pytest.fail("unexpected freeze"))
    cfg = write_config(tmp_path / "config.yaml")
    assert study.main(["check", "--config", str(cfg), "--sidecar", str(tmp_path / "unfrozen")]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["market_evaluation_performed"] is False
    assert payload["filtered_rule_count"] == 128
    with pytest.raises(SystemExit) as exc:
        study.main(["oos"])
    assert exc.value.code == 2


def test_freeze_requires_explicit_authorization_argument(tmp_path):
    with pytest.raises(SystemExit) as exc:
        study.main(["freeze", "--config", str(write_config(tmp_path / "config.yaml"))])
    assert exc.value.code == 2


def test_lock_is_derived_from_baseline_only_and_refuses_changed_selection(monkeypatch, tmp_path):
    dev = {"finished_at": "2026-10-09T12:00:00+00:00"}
    monkeypatch.setattr(study, "verify_results", lambda *_: dev)
    folder = tmp_path / "development"
    folder.mkdir()
    metrics().to_csv(folder / "metrics.csv", index=False)
    lock = study.seal_locked_selections(tmp_path)
    assert [s["gate_period"] for s in lock["selections"]] == [200, 200]
    assert study.verify_locked_selections(tmp_path) == lock
    path = tmp_path / "locked_selections.json"
    payload = study.custody._read_seal(path)
    payload["selections"][0] = study.baskets()[0]
    path.unlink()
    study.custody._seal(path, payload)
    with pytest.raises(study.Error, match="development"):
        study.verify_locked_selections(tmp_path)


def test_freeze_sidecar_never_overlaps_preserved_tree(monkeypatch, tmp_path):
    monkeypatch.setattr(study, "ROOT", tmp_path)
    cache = tmp_path / ".cache"
    source = cache / "portfolio_research_synthetic"
    study._separate(
        cache / "multi_scale_research_synthetic", source, cache / "implementation/auth.json"
    )
    for sidecar in (
        source,
        source / "multi_scale_research_x",
        tmp_path / "reports/multi_scale_research_x",
    ):
        with pytest.raises(study.Error):
            study._separate(sidecar, source)
    with pytest.raises(study.Error):
        study._separate(cache / "multi_scale_research_x", source, source / "auth.json")


def test_pair_bootstrap_keeps_aliases_and_pairing():
    rng = np.random.default_rng(31)
    cash = np.full(60, 0.0001)
    mr = rng.normal(0.0002, 0.002, 60)
    bh = rng.normal(0.0004, 0.005, 60)
    a = mr * 0.9 + cash * 0.1
    out = study.paired_bootstrap(
        {"selected": a},
        {"locked_independent": mr, "ungated_mean_reversion": mr.copy(), "buy_and_hold": bh},
        cash,
        replicates=40,
        block_lengths=(5, 20),
        seed=17,
    )
    assert len(out) == 4
    assert set(out.reference_id) == {"locked_independent", "buy_and_hold"}
    aliased = out.loc[out.reference_id.eq("locked_independent")]
    assert all(
        json.loads(x) == ["locked_independent", "ungated_mean_reversion"]
        for x in aliased.reference_aliases
    )
    assert (out.finite_replicates == 40).all()
    for row in out.itertuples(index=False):
        ref = mr if row.reference_id == "locked_independent" else bh
        expected = study.portfolio.cash_excess_sharpe(a, cash) - study.portfolio.cash_excess_sharpe(
            ref, cash
        )
        assert row.observed_sharpe_difference == pytest.approx(expected)
    pd.testing.assert_frame_equal(
        out,
        study.paired_bootstrap(
            {"selected": a},
            {"locked_independent": mr, "ungated_mean_reversion": mr.copy(), "buy_and_hold": bh},
            cash,
            replicates=40,
            block_lengths=(5, 20),
            seed=17,
        ),
    )


def test_pair_bootstrap_same_selected_and_reference_has_zero_interval():
    values = np.linspace(-0.002, 0.003, 40)
    out = study.paired_bootstrap(
        {"selected": values},
        {"reference": values.copy()},
        np.full(40, 0.0001),
        replicates=20,
        block_lengths=(5,),
    )
    assert out.observed_sharpe_difference.iloc[0] == 0
    assert out.ci_lower.iloc[0] == 0 and out.ci_upper.iloc[0] == 0


@pytest.mark.parametrize("case", ["empty", "shape", "nan", "block", "replicates"])
def test_bootstrap_refuses_invalid_input(case):
    x = np.linspace(-0.002, 0.003, 30)
    kwargs = {"replicates": 10, "block_lengths": (5,)}
    selected = {"x": x}
    refs = {"r": x + 0.001}
    cash = np.zeros(30)
    if case == "empty":
        selected = {}
    elif case == "shape":
        cash = np.zeros(29)
    elif case == "nan":
        refs["r"][0] = np.nan
    elif case == "block":
        kwargs["block_lengths"] = (0,)
    else:
        kwargs["replicates"] = 0
    with pytest.raises(study.Error):
        study.paired_bootstrap(selected, refs, cash, **kwargs)


def test_no_full_history_readers_and_no_mutation_of_preserved_sources():
    source = inspect.getsource(study)
    assert "read_validation_prefix" in source
    assert "read_full" not in source
    assert "load_development_packet" in source
    assert "setattr(" not in source
    assert "cache.gate_override is not None" in source
    assert "with_constant_gate(cache, 1)" in source
    assert "with_constant_gate(cache, 0)" in source


def test_full_synthetic_stage_seals_development_then_validation_without_market_reads(
    monkeypatch, tmp_path
):
    # The fixture is generated mathematical OHLC, not any stored market packet.
    from test_multi_scale_engine import fixture, observations
    from trend_following import research_quality_engine_v5 as quality

    specs = [
        {"closes": [100] * 13, "daily_close": 101},
        {"closes": [90] * 13, "daily_close": 105},
        {"closes": [110] * 13, "daily_close": 110},
    ]
    daily, bars, calendar, coverage = fixture(specs)
    cache = engine.build_cache(
        daily,
        bars,
        calendar,
        coverage=coverage,
        quality_policy=quality.QualityPolicy(
            "observed_only_blackout_v1", post_gap_diagnostics=True
        ),
    )
    cash = observations(cache)
    membership = members()
    frozen = {
        "membership": membership,
        "membership_sha256": study.custody.canonical_hash(membership),
        "gated_candidates": definitions(),
        "gated_candidates_sha256": study.custody.canonical_hash(definitions()),
        "preserved_portfolio": str(tmp_path / "old"),
        "independent_locked_allocation": {"portfolio_id": "portfolio_tf_000"},
        "frozen_at": "2026-10-09T01:00:00+00:00",
    }
    monkeypatch.setattr(
        study,
        "PERIODS",
        {
            "development": {"start": "2001-01-02", "end": "2001-01-04"},
            "validation": {"start": "2001-01-02", "end": "2001-01-04"},
        },
    )
    monkeypatch.setattr(study, "verify_study", lambda _: frozen)
    monkeypatch.setattr(
        study, "load_packet", lambda *_a, **_k: (daily, bars, calendar, coverage, None, cash, {})
    )
    monkeypatch.setattr(study.custody, "_quality_policy", lambda _: cache.policy)
    monkeypatch.setattr(study, "_compare_preserved_sleeves", lambda *_a, **_k: None)

    def controls(_f, _stage, slip, times, cash_returns):
        # Exact-clock synthetic cash controls stand in for previously sealed A.
        wealth = 1000 * np.cumprod(1 + cash_returns)
        paths = {
            pid: {
                "equity": wealth.copy(),
                "returns": cash_returns.copy(),
                "exposure": np.zeros(len(times)),
            }
            for pid in study.expected_control_ids(_f, _stage)[:-2]
        }
        rows = pd.DataFrame(
            [
                {
                    "portfolio_id": pid,
                    "scenario_slippage_bps": slip,
                    "cash_excess_sharpe": np.nan,
                    "terminal_equity": wealth[-1],
                    "cagr": 0.03,
                    "max_drawdown": 0,
                    "exposure_fraction": 0,
                    "stage": _stage,
                }
                for pid in paths
            ]
        )
        return rows, paths

    monkeypatch.setattr(study, "_read_old_controls", controls)
    sidecar = tmp_path / "multi_scale_research_synthetic"
    dev = study.evaluate_development(sidecar)
    assert dev["constituent_count"] == 128
    assert dev["always_ON_all128_allcost_parity_verified"] is True
    assert dev["always_OFF_all128_cash_reconciles"] is True
    assert study.verify_results(sidecar, "development") == dev
    lock = study.verify_locked_selections(sidecar)
    assert len(lock["selections"]) == 2
    val = study.evaluate_validation(sidecar)
    assert val["selection_lock_sha256"] == study.custody.canonical_hash(lock)
    assert study.verify_results(sidecar, "validation") == val
    assert (
        len(pd.read_csv(sidecar / "validation/bootstrap.csv")) == 6
    )  # All three cash refs deduplicate.
    parity = json.loads((sidecar / "validation/preflight.json").read_text())
    assert len(parity) == 4 and all(r["filtered_rule_count"] == 128 for r in parity)
    for stage in ("development", "validation"):
        assert len(pd.read_csv(sidecar / stage / "metrics.csv")) == 16
        with np.load(sidecar / stage / "constituent_s3_daily.npz", allow_pickle=False) as data:
            assert data["equity"].shape[1] == 128
        assert (sidecar / stage / "gate_history.csv").exists()
    with pytest.raises(study.Error, match="sealed"):
        study.evaluate_development(sidecar)


def test_freeze_metadata_binds_exact_old_membership_sources_and_preserved_seals(
    monkeypatch, tmp_path
):
    old = {
        "membership": members(),
        "development_adapter": {"source": "bounded development"},
        "validation_packet": "bounded validation",
        "validation_metadata": {"manifest_sha256": "original"},
    }
    monkeypatch.setattr(study.preserved, "verify_study", lambda _: copy.deepcopy(old))
    monkeypatch.setattr(
        study.preserved, "verify_results", lambda _p, stage: {"stage": stage, "unchanged": True}
    )
    monkeypatch.setattr(
        study.preserved,
        "verify_locked_allocation",
        lambda _: {
            "allocation": {
                "portfolio_id": "portfolio_tf_000",
                "trend_weight": 0.0,
                "mean_reversion_weight": 1.0,
            }
        },
    )
    cfg = write_config(tmp_path / "config.yaml")
    auth = tmp_path / "authorization.json"
    auth.write_text(json.dumps(auth_payload()))
    value = study._bindings(cfg, auth, tmp_path / "preserved")
    assert value["membership"] == members()
    assert len(value["gated_candidates"]) == 128
    assert value["baskets"] == study.baskets()
    assert set(value["preserved_result_hashes"]) == {"development", "validation"}
    assert value["validation_metadata"]["manifest_sha256"] == "original"
    assert value["development_adapter"] == old["development_adapter"]
    assert value["historical_oos_authorized"] is False
    assert "src/trend_following/research_multi_scale_report.py" in value["source_hashes"]
    assert "src/trend_following/research_mean_reversion_engine.py" in value["source_hashes"]
    assert value["config_sha256"] == study.custody.sha256_file(cfg)


def test_freeze_is_idempotent_and_refuses_runtime_binding_changes(monkeypatch, tmp_path):
    monkeypatch.setattr(study, "ROOT", tmp_path)
    source = tmp_path / ".cache/portfolio_research_preserved"
    sidecar = tmp_path / ".cache/multi_scale_research_synthetic"
    cfg = tmp_path / "config.yaml"
    auth = tmp_path / "authorization.json"
    cfg.write_text("synthetic")
    auth.write_text("synthetic")
    payload = {"binding": "unchanged"}
    monkeypatch.setattr(study, "_bindings", lambda *_: copy.deepcopy(payload))
    a = study.freeze_study(sidecar, authorization=auth, config_path=cfg, preserved_portfolio=source)
    b = study.freeze_study(sidecar, authorization=auth, config_path=cfg, preserved_portfolio=source)
    assert a == b
    # verify uses explicit fields; freeze itself still refuses a changed binding.
    payload["binding"] = "changed"
    with pytest.raises(study.Error, match="overwrite"):
        study.freeze_study(sidecar, authorization=auth, config_path=cfg, preserved_portfolio=source)


def preserved_sleeve_archive(tmp_path):
    """Synthetic arrays using the actual old portfolio's complete NPZ field set."""
    rows = members()
    ids = [r["candidate_id"] for r in rows]
    times = pd.DatetimeIndex(pd.to_datetime(["2001-01-02T22:00:00Z", "2001-01-03T22:00:00Z"]))
    events = pd.DatetimeIndex(
        pd.to_datetime(
            [
                "2001-01-02T14:30:00Z",
                "2001-01-02T22:00:00Z",
                "2001-01-03T14:30:00Z",
                "2001-01-03T22:00:00Z",
            ]
        )
    )
    equity = np.vstack([1000 + np.arange(44), 1010 + np.arange(44)])
    returns = equity / np.vstack([np.full(44, 1000), equity[:-1]]) - 1
    longs = np.array([[False] * 44, [bool(j % 2) for j in range(44)], [True] * 44, [False] * 44])
    data = {
        "candidate_id": np.array(ids, dtype=str),
        "timestamp": np.array([t.isoformat() for t in times], dtype=str),
        "equity": equity,
        "returns": returns,
        "cash_returns": np.array([0.0001, 0.0002]),
        "gross_terminal_equity": equity[-1] + 1,
        "event_timestamp": np.array([t.isoformat() for t in events], dtype=str),
        "event_equity": np.vstack([np.full(44, 1000), equity[0], equity[0], equity[1]]),
        "event_invested_notional": np.zeros((4, 44)),
        "event_long": longs,
        "fill_timestamp": np.array([], dtype=str),
        "fill_phase": np.array([], dtype=str),
        "fill_reference_price": np.array([], dtype=float),
        "fill_equity_before": np.empty((0, 44)),
        "segment_start": events[0].isoformat(),
        "segment_end": events[-1].isoformat(),
        "initial_cash": 1000.0,
    }
    candidates = [mr.Candidate.from_dict(r["parameters"]) for r in rows[12:]]
    parts = (
        [],
        times,
        equity[:, 12:],
        returns[:, 12:],
        data["cash_returns"],
        longs[[1, 3]][:, 12:].astype(float),
        [],
        [],
        None,
    )
    path = tmp_path / "constituent_s3_daily.npz"
    np.savez_compressed(path, **data)
    return path, data, candidates, parts


def test_preserved_event_archive_without_exposure_reconciles_every_mr_column(tmp_path):
    path, data, candidates, parts = preserved_sleeve_archive(tmp_path)
    assert "exposure" not in data
    study._compare_preserved_sleeves(parts, candidates, path)
    assert set(data) == {
        "candidate_id",
        "timestamp",
        "equity",
        "returns",
        "cash_returns",
        "gross_terminal_equity",
        "event_timestamp",
        "event_equity",
        "event_invested_notional",
        "event_long",
        "fill_timestamp",
        "fill_phase",
        "fill_reference_price",
        "fill_equity_before",
        "segment_start",
        "segment_end",
        "initial_cash",
    }
    # A reversed candidate list must compare against corresponding archive columns,
    # not assume the thirty-two mean-reversion columns are already ordered.
    reordered = (
        parts[0],
        parts[1],
        parts[2][:, ::-1],
        parts[3][:, ::-1],
        parts[4],
        parts[5][:, ::-1],
        *parts[6:],
    )
    study._compare_preserved_sleeves(reordered, list(reversed(candidates)), path)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_event",
        "duplicate_event",
        "nonbinary",
        "shape",
        "returns",
        "cash",
        "position",
        "ids",
        "missing_field",
    ],
)
def test_preserved_event_archive_rejects_missing_or_changed_exact_observations(tmp_path, mutation):
    path, data, candidates, parts = preserved_sleeve_archive(tmp_path)
    original = copy.deepcopy(parts)
    if mutation == "missing_event":
        data["event_timestamp"][1] = "2001-01-02T21:59:59+00:00"
    elif mutation == "duplicate_event":
        data["event_timestamp"][1] = data["event_timestamp"][0]
    elif mutation == "nonbinary":
        data["event_long"] = data["event_long"].astype(float)
        data["event_long"][1, 12] = 0.5
    elif mutation == "shape":
        data["equity"] = data["equity"][:, :43]
    elif mutation == "returns":
        data["returns"][0, 12] += 0.1
    elif mutation == "cash":
        data["cash_returns"][0] += 0.1
    elif mutation == "position":
        data["event_long"][1, 12] = not data["event_long"][1, 12]
    elif mutation == "ids":
        data["candidate_id"][12] = data["candidate_id"][13]
    else:
        del data["event_long"]
    np.savez_compressed(path, **data)
    with pytest.raises(study.Error):
        study._compare_preserved_sleeves(original, candidates, path)


@pytest.mark.parametrize("stage", ["development", "validation"])
def test_independent_and_passive_read_only_adapters_accept_actual_archive_schemas(
    monkeypatch, tmp_path, stage
):
    path, data, _candidates, _parts = preserved_sleeve_archive(tmp_path)
    directory = tmp_path / stage
    directory.mkdir()
    lock = {"allocation": study.portfolio.Allocation("portfolio_tf_000", 0).to_dict()}
    allocations = (
        study.portfolio.allocation_grid()
        if stage == "development"
        else study.preserved.eligible_validation_allocations(lock)
    )
    ids = [a.portfolio_id for a in allocations]
    times = pd.DatetimeIndex(pd.to_datetime(data["timestamp"], utc=True))
    wealth = np.column_stack([data["equity"] @ a.weights() for a in allocations])
    returns = wealth / np.vstack([np.full(len(ids), 1000), wealth[:-1]]) - 1
    np.savez_compressed(
        directory / "allocation_s3_daily.npz",
        portfolio_id=np.array(ids, dtype=str),
        timestamp=data["timestamp"],
        equity=wealth,
        returns=returns,
        cash_returns=data["cash_returns"],
        exposure=np.zeros_like(wealth),
    )
    rows = [
        {
            "portfolio_id": cid,
            "scenario_slippage_bps": 3.0,
            "terminal_equity": wealth[-1, j],
            "cash_excess_sharpe": 0.2,
            "cagr": 0.1,
            "max_drawdown": -0.1,
        }
        for j, cid in enumerate(ids)
    ]
    pd.DataFrame(rows).to_csv(directory / "metrics.csv", index=False)
    constituent = pd.DataFrame(
        {
            "candidate_id": data["candidate_id"],
            "trade_count": np.arange(1, 45),
            "average_holding_hours": np.arange(1, 45) * 10.0,
        }
    )
    constituent.to_csv(directory / "constituent_metrics_s3.csv", index=False)
    bench = []
    for mode in ("buy_and_hold", "cash"):
        frame = pd.DataFrame(
            {
                "timestamp": data["timestamp"],
                "equity": np.array([1000.1, 1000.3]),
                "return": np.array([0.0001, 0.0002]),
                "cash_return": data["cash_returns"],
                "exposure": np.zeros(2),
            }
        )
        frame.to_csv(directory / f"{mode}_s3_daily.csv", index=False)
        bench.append(
            {
                "portfolio_id": mode,
                "scenario_slippage_bps": 3.0,
                "terminal_equity": 1000.3,
                "cash_excess_sharpe": np.nan,
                "cagr": 0.03,
                "max_drawdown": 0.0,
            }
        )
    pd.DataFrame(bench).to_csv(directory / "benchmark_metrics.csv", index=False)
    monkeypatch.setattr(study.preserved, "verify_results", lambda *_: {"derived_synthetic": True})
    monkeypatch.setattr(study.preserved, "verify_locked_allocation", lambda *_: lock)
    frozen = {"preserved_portfolio": str(tmp_path), "membership": members()}
    frame, paths = study._read_old_controls(frozen, stage, 3, times, data["cash_returns"])
    assert list(paths) == ids + ["buy_and_hold", "cash"]
    assert len(frame) == len(ids) + 2
    for a in allocations:
        active = constituent.loc[a.weights() > 0]
        expected = (
            active.trade_count * active.average_holding_hours
        ).sum() / active.trade_count.sum()
        assert frame.loc[
            frame.portfolio_id.eq(a.portfolio_id), "average_gross_sleeve_holding_hours"
        ].iloc[0] == pytest.approx(expected)
    assert (
        frame.loc[
            frame.portfolio_id.isin(["buy_and_hold", "cash"]), "average_gross_sleeve_holding_hours"
        ]
        .isna()
        .all()
    )
    with pytest.raises(study.Error, match="clock"):
        study._read_old_controls(
            frozen, stage, 3, times + pd.Timedelta(seconds=1), data["cash_returns"]
        )
