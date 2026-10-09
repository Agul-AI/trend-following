"""Synthetic gates for 32 unchanged mean-reversion development cases.

All scores, metadata and packet adapters below are fabricated. These tests do
not open real validation prices, prior strategy outcomes, or historical OOS.
"""

import copy
import inspect
import json

import numpy as np
import pandas as pd
import pytest

from trend_following import research_mean_reversion as features
from trend_following import research_mean_reversion_validation as study


def synthetic_development():
    limits = {"SMA_DISTANCE": 6, "EMA_DISTANCE": 4, "ZSCORE": 5, "RSI": 17}
    used = dict.fromkeys(limits, 0)
    rows = []
    for index, candidate in enumerate(features.candidate_grid()):
        used[candidate.family] += 1
        score = 0.4 if used[candidate.family] <= limits[candidate.family] else -0.5
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "family": candidate.family,
                "period": candidate.rule.period,
                "indicator_units": "daily_sessions",
                "entry_threshold": candidate.entry_threshold,
                "exit_threshold": candidate.exit_threshold,
                "confirmation_hours": 3.0,
                "entry_confirmation_hours": 3.0,
                "exit_confirmation_hours": 3.0,
                "entry_required_checks": 7,
                "exit_required_checks": 7,
                "scenario_slippage_bps": 3.0,
                "commission_bps": 1.0,
                "slippage_bps": 3.0,
                "cash_excess_sharpe": score,
                "BH_cash_excess_sharpe": 0.0,
                "Sharpe_difference_vs_BH": score,
                "beats_BH_Sharpe": score > study.TOLERANCE,
                "cagr": 0.04,
                "max_drawdown": -0.2,
                "trade_count": index,
                "trades_per_year": index / 11,
            }
        )
    benchmark = pd.DataFrame(
        [
            {"portfolio": "buy_and_hold", "scenario_slippage_bps": 3.0, "cash_excess_sharpe": 0.0},
            {"portfolio": "buy_and_hold", "scenario_slippage_bps": 1.0, "cash_excess_sharpe": 9.0},
            {"portfolio": "cash", "scenario_slippage_bps": 3.0, "cash_excess_sharpe": np.nan},
        ]
    )
    return pd.DataFrame(rows), benchmark


def synthetic_selection():
    metrics, benchmark = synthetic_development()
    candidates, selected, _ = study.select_development_cases(metrics, benchmark)
    definitions = [{"candidate_id": c.candidate_id, **c.to_dict()} for c in candidates]
    return {
        "stage": "validation",
        "period": dict(study.PERIOD),
        "validation_only": True,
        "historical_oos_authorized": False,
        "candidates": definitions,
        "candidate_set_sha256": study.protocol.canonical_hash(definitions),
        "configuration_count": 32,
        "shared_confirmation_hours": 3.0,
        "required_checks": 7,
        "development_selection_rows": selected.to_dict("records"),
    }


def authorization(tmp_path, **changes):
    value = {
        "schema": "mean_reversion_validation_authorization",
        "authorized_by": "human_user",
        "explicit_user_approval": True,
        "authorized_stages": ["validation"],
        "user_request": study.USER_REQUEST,
        "configuration_count": 32,
        "shared_confirmation_hours": 3.0,
        "validation_period": dict(study.PERIOD),
        "historical_oos_authorized": False,
        "no_retuning_or_replacement": True,
        **changes,
    }
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(value))
    return path


def metadata():
    return {
        "schema": "trend_following_validation_2018_prefix_v1",
        "end_session": "2018-12-31",
        "feature_history_start": "1999-11-01",
        "stage": "validation",
        "historical_oos_authorized": False,
        "files": {name: {"sha256": "synthetic-" + name} for name in study.prefix.PACKET_FILES},
        "audit": {"OOS_numeric_rows_parsed": 0},
        "prepared_at": "2001-01-01T00:00:00Z",
    }


def prepared_metadata(manifest):
    return {
        "schema": "trend_following_resplit_validation_prepared_v1",
        "period": dict(study.PERIOD),
        "feature_history_start": study.prefix.START_SESSION,
        "historical_oos_authorized": False,
        "manifest_sha256": study.protocol.canonical_hash(manifest),
        "prepared_at": manifest["prepared_at"],
    }


def test_development_selection_uses_all_32_not_top20_and_never_trade_filters():
    metrics, benchmark = synthetic_development()
    candidates, selected, bh = study.select_development_cases(metrics, benchmark)
    assert bh == 0.0
    assert len(candidates) == len(selected) == 32
    assert selected.groupby("family").size().to_dict() == {
        "SMA_DISTANCE": 6,
        "EMA_DISTANCE": 4,
        "ZSCORE": 5,
        "RSI": 17,
    }
    assert selected.candidate_id.tolist() == sorted(selected.candidate_id)
    assert selected.trade_count.min() == 0
    assert all(c.confirmation_hours == 3 and c.required_checks == 7 for c in candidates)
    assert all(c.entry_required_checks == c.exit_required_checks for c in candidates)


def test_development_selection_preserves_membership_when_rows_are_shuffled():
    metrics, benchmark = synthetic_development()
    before = study.select_development_cases(metrics, benchmark)
    after = study.select_development_cases(metrics.sample(frac=1, random_state=47), benchmark)
    assert [c.candidate_id for c in before[0]] == [c.candidate_id for c in after[0]]
    pd.testing.assert_frame_equal(before[1], after[1])


def test_selection_requires_finite_score_and_existing_tolerance():
    metrics, benchmark = synthetic_development()
    yes = metrics.index[metrics.beats_BH_Sharpe].tolist()
    no = metrics.index[~metrics.beats_BH_Sharpe].tolist()
    metrics.loc[yes[0], "cash_excess_sharpe"] = 2 * study.TOLERANCE
    for index, score in zip(
        no[:6], [np.nan, np.inf, -np.inf, 0, study.TOLERANCE / 2, study.TOLERANCE], strict=True
    ):
        metrics.loc[index, "cash_excess_sharpe"] = score
    metrics["Sharpe_difference_vs_BH"] = metrics.cash_excess_sharpe
    metrics["beats_BH_Sharpe"] = np.isfinite(metrics.cash_excess_sharpe) & (
        metrics.cash_excess_sharpe > study.TOLERANCE
    )
    candidates, selected, _ = study.select_development_cases(metrics, benchmark)
    assert len(candidates) == 32
    assert metrics.loc[yes[0], "candidate_id"] in selected.candidate_id.tolist()
    assert not set(metrics.loc[no[:6], "candidate_id"]) & set(selected.candidate_id)


@pytest.mark.parametrize("change", ["missing", "duplicate", "unknown", "not32"])
def test_selection_rejects_missing_replacement_duplicate_or_changed_count(change):
    metrics, benchmark = synthetic_development()
    if change == "missing":
        metrics = metrics.iloc[:-1]
    elif change == "duplicate":
        metrics.loc[1, "candidate_id"] = metrics.loc[0, "candidate_id"]
    elif change == "unknown":
        metrics.loc[0, "candidate_id"] = "mean-reversion-unapproved"
    else:
        metrics.loc[0, ["cash_excess_sharpe", "Sharpe_difference_vs_BH"]] = -0.5
        metrics.loc[0, "beats_BH_Sharpe"] = False
    with pytest.raises(study.Error):
        study.select_development_cases(metrics, benchmark)


@pytest.mark.parametrize(
    "field,value",
    [
        ("family", "RSI"),
        ("period", 200),
        ("entry_threshold", -0.99),
        ("exit_threshold", 0.99),
        ("entry_confirmation_hours", 2),
        ("exit_confirmation_hours", 4),
        ("entry_required_checks", 6),
        ("exit_required_checks", 8),
        ("scenario_slippage_bps", 1),
        ("commission_bps", 0),
        ("slippage_bps", 25),
        ("BH_cash_excess_sharpe", 1),
        ("Sharpe_difference_vs_BH", 4),
        ("beats_BH_Sharpe", False),
    ],
)
def test_selection_refuses_changed_candidate_policy_or_baseline_comparison(field, value):
    metrics, benchmark = synthetic_development()
    metrics.loc[0, field] = value
    with pytest.raises(study.Error):
        study.select_development_cases(metrics, benchmark)


@pytest.mark.parametrize("change", ["missing", "duplicate", "undefined", "wrong_cost"])
def test_selection_requires_single_finite_baseline_buy_and_hold(change):
    metrics, benchmark = synthetic_development()
    if change == "missing":
        benchmark = benchmark.iloc[1:]
    elif change == "duplicate":
        benchmark = pd.concat([benchmark, benchmark.iloc[:1]])
    elif change == "undefined":
        benchmark.loc[0, "cash_excess_sharpe"] = np.nan
    else:
        benchmark.loc[0, "scenario_slippage_bps"] = 25
    with pytest.raises(study.Error):
        study.select_development_cases(metrics, benchmark)


def test_only_32_canonical_daily_definitions_reconstruct_from_seal():
    selection = synthetic_selection()
    candidates = study.candidates_from_selection(selection)
    assert len(candidates) == 32
    assert len({c.candidate_id for c in candidates}) == 32
    assert {c.confirmation_hours for c in candidates} == {3.0}
    assert {c.required_checks for c in candidates} == {7}


@pytest.mark.parametrize("change", ["missing", "duplicate", "order", "threshold", "wait", "hash"])
def test_reconstruction_refuses_altered_sealed_membership_parameters_or_wait(change):
    selection = synthetic_selection()
    if change == "missing":
        selection["candidates"].pop()
    elif change == "duplicate":
        selection["candidates"][-1] = copy.deepcopy(selection["candidates"][0])
    elif change == "order":
        selection["candidates"].reverse()
    elif change == "threshold":
        selection["candidates"][0]["entry_threshold"] = -0.99
    elif change == "wait":
        selection["candidates"][0]["confirmation_hours"] = 4.0
    else:
        selection["candidate_set_sha256"] = "not-the-actual-candidates"
    if change != "hash":
        selection["candidate_set_sha256"] = study.protocol.canonical_hash(selection["candidates"])
    with pytest.raises(study.Error):
        study.candidates_from_selection(selection)


def test_authorization_allows_only_explicit_unchanged_validation(tmp_path):
    approved = study._authorization(authorization(tmp_path))
    assert approved["historical_oos_authorized"] is False
    assert approved["validation_period"] == study.PERIOD
    assert approved["configuration_count"] == 32


@pytest.mark.parametrize(
    "changes",
    [
        {"explicit_user_approval": 1},
        {"explicit_user_approval": False},
        {"authorized_by": "assistant"},
        {"authorized_stages": ["validation", "historical_oos"]},
        {"historical_oos_authorized": True},
        {"historical_oos_authorized": 0},
        {"configuration_count": 20},
        {"shared_confirmation_hours": 4},
        {"validation_period": {"start": "2012-01-01", "end": "2015-12-31"}},
        {"validation_period": {"start": "2012-01-01", "end": "2019-01-01"}},
        {"no_retuning_or_replacement": False},
        {"no_retuning_or_replacement": 1},
        {"user_request": "run all stages"},
    ],
)
def test_authorization_rejects_other_membership_wait_period_stage_and_boolean(tmp_path, changes):
    with pytest.raises(study.Error):
        study._authorization(authorization(tmp_path, **changes))


def test_preselection_metadata_reads_only_manifest_not_numeric_packet(tmp_path, monkeypatch):
    calls = []
    manifest = metadata()
    monkeypatch.setattr(study.prefix, "_json_file", lambda p: calls.append(p) or manifest)
    monkeypatch.setattr(
        study.protocol,
        "_read_seal",
        lambda p: calls.append(p) or prepared_metadata(manifest),
    )
    monkeypatch.setattr(
        study.prefix,
        "read_validation_prefix",
        lambda *a, **k: pytest.fail("No numerical prefix before selection"),
    )
    binding = study._metadata_binding(tmp_path)
    assert calls == [tmp_path / "manifest.json", tmp_path / "freeze.json"]
    assert binding["manifest_sha256"] == study.protocol.canonical_hash(manifest)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "unbounded-packet"),
        ("end_session", "2026-05-28"),
        ("feature_history_start", "2001-01-01"),
        ("stage", "historical-oos"),
        ("historical_oos_authorized", True),
        ("historical_oos_authorized", 0),
        ("files", {"daily.csv": {"sha256": "incomplete"}}),
        ("audit", {"OOS_numeric_rows_parsed": 1}),
    ],
)
def test_metadata_refuses_unbounded_wrong_stage_or_incomplete_packet(
    tmp_path, monkeypatch, field, value
):
    manifest = metadata()
    manifest[field] = value
    monkeypatch.setattr(study.prefix, "_json_file", lambda p: manifest)
    with pytest.raises(study.Error):
        study._metadata_binding(tmp_path)


def test_freeze_cannot_touch_price_packet_before_selection_verifies(tmp_path, monkeypatch):
    monkeypatch.setattr(
        study, "verify_selection", lambda *_: (_ for _ in ()).throw(study.Error("selection absent"))
    )
    monkeypatch.setattr(
        study,
        "_prepare_bounded_packet",
        lambda *a: pytest.fail("No prices before intact selection"),
    )
    with pytest.raises(study.Error, match="selection absent"):
        study.freeze_study(tmp_path / "new")


def test_evaluation_refuses_price_access_without_intact_runtime_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "verify_selection", lambda *_: synthetic_selection())
    monkeypatch.setattr(
        study, "verify_study", lambda *_: (_ for _ in ()).throw(study.Error("freeze absent"))
    )
    monkeypatch.setattr(
        study, "load_validation_packet", lambda *a: pytest.fail("No prices before runtime freeze")
    )
    with pytest.raises(study.Error, match="freeze absent"):
        study.evaluate_study(tmp_path / "new")


@pytest.mark.parametrize(
    "changes",
    [
        {"stage": "historical_oos"},
        {"period": {"start": "2019-01-01", "end": "2026-05-28"}},
        {"period": {"start": "2012-01-01", "end": "2019-01-01"}},
        {"historical_oos_authorized": True},
        {"historical_oos_authorized": 0},
        {"validation_only": False},
        {"validation_only": 1},
    ],
)
def test_reader_blocks_other_stage_or_endpoint_before_any_packet_access(
    tmp_path, monkeypatch, changes
):
    selection = {**synthetic_selection(), **changes}
    monkeypatch.setattr(
        study, "verify_study", lambda *_: pytest.fail("Wrong stage before seal read")
    )
    monkeypatch.setattr(
        study.prefix, "read_validation_prefix", lambda *a, **k: pytest.fail("No later price read")
    )
    with pytest.raises(study.Error, match="Only frozen"):
        study.load_validation_packet(tmp_path, selection)


def test_sources_do_not_use_old_validation_results_or_unbounded_preparer():
    source = inspect.getsource(study)
    for forbidden in (
        "research_resplit_validation",
        "prefix.prepare_validation_prefix(",
        "protocol._load_stage(",
        "protocol._verify_prepared(",
        "data.read_reconciled_packet(",
    ):
        assert forbidden not in source
    loader = inspect.getsource(study.load_validation_packet)
    assert "prefix.read_validation_prefix(" in loader
    assert "historical-oos" not in inspect.getsource(study.main)


def test_chronology_requires_aware_past_ordered_instants():
    assert len(study._chronology("2001-01-01T00:00:00Z", "2002-01-01T00:00:00Z")) == 2
    for values in (
        ("2001-01-01",),
        ("2002-01-01T00:00:00Z", "2001-01-01T00:00:00Z"),
        ("2099-01-01T00:00:00Z",),
    ):
        with pytest.raises(study.Error):
            study._chronology(*values)


def test_common_cash_and_cost_scenarios_preserve_development_policy():
    assert study.PERIOD == {"start": "2012-01-01", "end": "2018-12-31"}
    assert study.HISTORICAL_OOS_PERIOD == {"start": "2019-01-01", "end": "2026-05-28"}
    assert study.SLIPPAGES == (1.0, 3.0, 10.0, 25.0)
    assert study.fixed.cash_policy()["annual_rate"] == 0.03
    assert study.COUNT == 32
    assert set(study.prefix.PACKET_FILES) == {
        "daily.csv",
        "bars.csv",
        "calendar.csv",
        "coverage.csv",
        "blackouts.csv",
        "unit_proof.json",
    }


@pytest.fixture
def selection_context(tmp_path, monkeypatch):
    """Immutable real seals; fake scores/provenance and no price adapters."""
    monkeypatch.setattr(study, "ROOT", tmp_path)
    dev, packet, source = (tmp_path / ".cache" / name for name in ("dev", "packet", "source"))
    for path in (dev, packet, source):
        path.mkdir(parents=True)
    approval = authorization(tmp_path)
    implementation = tmp_path / "synthetic_runtime.py"
    implementation.write_text("# synthetic implementation\n")
    sidecar = tmp_path / ".cache/mean_reversion_validation_synthetic"

    def bindings(dev_arg, packet_arg, authorization_arg):
        assert (dev_arg, packet_arg, authorization_arg) == (dev, packet, approval)
        study._authorization(authorization_arg)
        return {
            **synthetic_selection(),
            "development_sidecar": str(dev),
            "bounded_packet": str(packet),
            "source_study": str(source),
            "authorization": {
                "path": str(approval),
                "sha256": study.protocol.sha256_file(approval),
            },
            "development_finished_at": "2001-01-01T00:00:00Z",
            "source_hashes": {implementation.name: study.protocol.sha256_file(implementation)},
            "runtime": {"mode": "fabricated"},
        }

    monkeypatch.setattr(study, "_selection_bindings", bindings)
    monkeypatch.setattr(
        study.prefix,
        "read_validation_prefix",
        lambda *a, **k: pytest.fail("No numeric validation packet in selection tests"),
    )
    return {
        "sidecar": sidecar,
        "authorization": approval,
        "development_sidecar": dev,
        "bounded_packet": packet,
        "implementation": implementation,
        "source": source,
    }


def selection_kwargs(context):
    return {k: context[k] for k in ("authorization", "development_sidecar", "bounded_packet")}


def test_selection_seal_is_idempotent_and_preserves_original_bytes(selection_context):
    context = selection_context
    sealed = study.seal_selection(context["sidecar"], **selection_kwargs(context))
    before = (context["sidecar"] / "selection.json").read_bytes()
    assert study.verify_selection(context["sidecar"]) == sealed
    assert study.seal_selection(context["sidecar"], **selection_kwargs(context)) == sealed
    assert (context["sidecar"] / "selection.json").read_bytes() == before
    assert len(sealed["candidates"]) == 32


def test_selection_seal_detects_changed_sources_without_replacing_it(selection_context):
    context = selection_context
    study.seal_selection(context["sidecar"], **selection_kwargs(context))
    before = (context["sidecar"] / "selection.json").read_bytes()
    context["implementation"].write_text("# changed synthetic implementation\n")
    with pytest.raises(study.Error, match="changed"):
        study.verify_selection(context["sidecar"])
    with pytest.raises(study.Error, match="sealed differently"):
        study.seal_selection(context["sidecar"], **selection_kwargs(context))
    assert (context["sidecar"] / "selection.json").read_bytes() == before


def test_selection_seal_binds_exact_authorization_bytes(selection_context):
    context = selection_context
    study.seal_selection(context["sidecar"], **selection_kwargs(context))
    approval = context["authorization"]
    approval.write_text(approval.read_text() + "\n")
    with pytest.raises(study.Error, match="changed"):
        study.verify_selection(context["sidecar"])


def test_selection_refuses_protected_tree_or_authorization_inside_new_tree(selection_context):
    context = selection_context
    preserved = (context["development_sidecar"], context["bounded_packet"], context["source"])
    for target in preserved:
        with pytest.raises(study.Error):
            study._separate(target, preserved)
        with pytest.raises(study.Error):
            study._separate(target / "mean_reversion_validation_child", preserved)
    with pytest.raises(study.Error):
        study._separate(context["sidecar"], preserved, context["sidecar"] / "authorization.json")


def test_freeze_refuses_unsealed_prepared_tree_and_never_overwrites_prices(tmp_path, monkeypatch):
    sidecar = tmp_path / "new"
    (sidecar / "prepared").mkdir(parents=True)
    sentinel = sidecar / "prepared/do_not_replace.txt"
    sentinel.write_text("original")
    monkeypatch.setattr(study, "verify_selection", lambda *_: synthetic_selection())
    monkeypatch.setattr(
        study, "_prepare_bounded_packet", lambda *a: pytest.fail("Never overwrite existing prices")
    )
    with pytest.raises(study.Error, match="Unsealed"):
        study.freeze_study(sidecar)
    assert sentinel.read_text() == "original"


def test_evaluation_refuses_existing_results_before_market_access(tmp_path, monkeypatch):
    sidecar = tmp_path / "new"
    (sidecar / "results").mkdir(parents=True)
    sentinel = sidecar / "results/preserve.txt"
    sentinel.write_text("sealed earlier")
    monkeypatch.setattr(study, "verify_selection", lambda *_: synthetic_selection())
    monkeypatch.setattr(study, "verify_study", lambda *_: {})
    monkeypatch.setattr(
        study,
        "load_validation_packet",
        lambda *a: pytest.fail("Cannot run again or replace results"),
    )
    with pytest.raises(study.Error, match="never overwrite"):
        study.evaluate_study(sidecar)
    assert sentinel.read_text() == "sealed earlier"


def test_valid_loader_uses_only_new_physical_prefix_and_copied_period_config(tmp_path, monkeypatch):
    source = tmp_path / "source"
    selection = {
        **synthetic_selection(),
        "source_config_path": str(tmp_path / "immutable.yaml"),
        "source_study": str(source),
        "bounded_packet_metadata": {"manifest": {"gap_authorization": {}}},
    }
    original = {"segments": {"validation": {"start": "2012-01-01", "end": "2015-12-31"}}}
    before = copy.deepcopy(original)
    daily = pd.DataFrame({"daily_price_available_at": [pd.Timestamp("2018-12-31T22:00:00Z")]})
    calendar = pd.DataFrame({"market_open": [pd.Timestamp("1999-11-01T14:30:00Z")]})
    empty = pd.DataFrame()
    monkeypatch.setattr(
        study,
        "verify_study",
        lambda *_: {
            "manifest_sha256": "synthetic-prefix",
            "selection_sha256": study.protocol.canonical_hash(selection),
        },
    )
    calls = []

    def read_prefix(path, *, expected_manifest_sha256):
        calls.append((path, expected_manifest_sha256))
        return daily, empty, calendar, empty, empty, {}

    monkeypatch.setattr(study.prefix, "read_validation_prefix", read_prefix)
    monkeypatch.setattr(study.protocol, "load_config", lambda *_: original)
    monkeypatch.setattr(study.protocol, "_read_seal", lambda p: {"gap_authorization": {}})
    periods = []
    monkeypatch.setattr(
        study.protocol,
        "_check_prefix",
        lambda *args: periods.append(args[-1]["segments"]["validation"]),
    )
    monkeypatch.setattr(study.protocol, "_quality_gate", lambda *a: None)
    result = study.load_validation_packet(tmp_path, selection)
    assert calls == [(tmp_path / "prepared", "synthetic-prefix")]
    assert original == before
    assert result[-1]["segments"]["validation"] == study.PERIOD
    assert result[-1]["segments"]["historical-oos"] == study.HISTORICAL_OOS_PERIOD
    assert periods == [study.PERIOD]


def test_empty_validation_BH_beater_group_has_undefined_medians_not_fake_success():
    metrics, _ = synthetic_development()
    selected = metrics.loc[metrics.beats_BH_Sharpe]
    summary = study.development.family_summary(selected, 1.0)
    total = summary.loc[summary.family.eq("ALL")].iloc[0]
    assert total.configurations == 32
    assert total.beat_BH_Sharpe_count == 0
    assert np.isnan(total.BH_beaters_median_cash_excess_sharpe)
    assert np.isnan(total.BH_beaters_median_trade_count)
    assert np.isfinite(total.all_median_cash_excess_sharpe)


def test_changed_preserved_packet_seal_cannot_bless_metadata(tmp_path, monkeypatch):
    manifest = metadata()
    prepared = prepared_metadata(manifest)
    prepared["manifest_sha256"] = "wrong-original-binding"
    monkeypatch.setattr(study.prefix, "_json_file", lambda p: manifest)
    monkeypatch.setattr(study.protocol, "_read_seal", lambda p: prepared)
    with pytest.raises(study.Error, match="custody"):
        study._metadata_binding(tmp_path)


@pytest.mark.parametrize(
    "field,value", [("source_config_path", "/unfrozen-config.yaml"), ("cash", {"annual_rate": 0.0})]
)
def test_loader_rejects_forged_caller_contract_before_reading_prices(
    tmp_path, monkeypatch, field, value
):
    selection = synthetic_selection()
    frozen_hash = study.protocol.canonical_hash(selection)
    selection[field] = value
    monkeypatch.setattr(
        study,
        "verify_study",
        lambda *_: {"manifest_sha256": "synthetic", "selection_sha256": frozen_hash},
    )
    monkeypatch.setattr(
        study.prefix,
        "read_validation_prefix",
        lambda *a, **k: pytest.fail("Forged caller contract must fail before any price access"),
    )
    with pytest.raises(study.Error, match="Caller selection"):
        study.load_validation_packet(tmp_path, selection)


@pytest.fixture(scope="module")
def synthetic_paths_master(tmp_path_factory):
    """Invented wealth on 1760 exchange dates; no historical prices or outcomes."""
    from trend_following.research_hourly_data_v5 import build_xnys_calendar

    private = tmp_path_factory.mktemp("invented_mr_validation_wealth")
    selection = synthetic_selection()
    selection["bounded_packet_metadata"] = {
        "manifest": {
            "audit": {
                "validation_sessions": 1760,
                "first_validation_session": "2012-01-03",
                "last_validation_session": "2018-12-31",
            }
        }
    }
    candidates = study.candidates_from_selection(selection)
    calendar = build_xnys_calendar("2012-01-01", "2018-12-31")
    times = pd.DatetimeIndex(
        [
            pd.Timestamp(f"{s} 17:00", tz="America/New_York").tz_convert("UTC")
            for s in calendar.session
        ]
    )
    assert len(times) == 1760
    first = pd.Timestamp("2012-01-03 09:30", tz="America/New_York").tz_convert("UTC")
    years = (times[-1] - first).total_seconds() / (365 * 86400)
    t = np.arange(len(times), dtype=float)
    oscillation = 0.002 * np.sin(t * 0.23) + 0.001 * np.cos(t * 0.037)
    cash_wealth = 1000 * np.cumprod(np.ones(len(times)) * 1.00008)
    cash_returns = cash_wealth / np.r_[1000.0, cash_wealth[:-1]] - 1

    def stats(equity, returns):
        excess = returns - cash_returns
        sd = excess.std(ddof=1)
        wealth = np.r_[1000.0, equity]
        return {
            "cash_excess_sharpe": float(np.sqrt(252) * excess.mean() / sd) if sd > 0 else np.nan,
            "terminal_equity": equity[-1],
            "cagr": (equity[-1] / 1000) ** (1 / years) - 1,
            "max_drawdown": (wealth / np.maximum.accumulate(wealth) - 1).min(),
            "annualized_volatility": returns.std(ddof=1) * np.sqrt(252),
            "total_return": equity[-1] / 1000 - 1,
        }

    all_rows, benchmarks = [], []
    for slip in study.SLIPPAGES:
        columns = np.arange(32)[None, :]
        invented = oscillation[:, None] * (1 + columns / 50) + (columns + 1) * 3e-6 - slip * 1e-7
        equity = 1000 * np.cumprod(1 + invented, axis=0)
        returns = equity / np.vstack([np.full(32, 1000.0), equity[:-1]]) - 1
        np.savez_compressed(
            private / f"slippage_{slip:g}_daily.npz",
            timestamp=np.array([x.isoformat() for x in times]),
            candidate_id=np.array([c.candidate_id for c in candidates]),
            equity=equity,
            returns=returns,
            cash_returns=cash_returns,
            exposure=np.zeros_like(equity),
        )
        bh = None
        for portfolio in ("cash", "buy_and_hold"):
            wealth = (
                cash_wealth
                if portfolio == "cash"
                else 1000 * np.cumprod(1 + oscillation * 1.2 + 0.00004 - slip * 1e-7)
            )
            passive_returns = wealth / np.r_[1000.0, wealth[:-1]] - 1
            passive_stats = stats(wealth, passive_returns)
            pd.DataFrame(
                {
                    "timestamp": times,
                    "equity": wealth,
                    "return": passive_returns,
                    "cash_return": cash_returns,
                    "exposure": np.zeros(len(times)),
                }
            ).to_csv(private / f"{portfolio}_{slip:g}_daily.csv", index=False)
            benchmarks.append(
                {
                    "portfolio": portfolio,
                    "scenario_slippage_bps": slip,
                    "trade_count": 0 if portfolio == "cash" else 1,
                    "order_count": 0 if portfolio == "cash" else 2,
                    "trades_per_year": 0 if portfolio == "cash" else 1 / years,
                    **passive_stats,
                }
            )
            if portfolio == "buy_and_hold":
                bh = passive_stats["cash_excess_sharpe"]
        rows = []
        for index, candidate in enumerate(candidates):
            candidate_stats = stats(equity[:, index], returns[:, index])
            score = candidate_stats["cash_excess_sharpe"]
            trades = 10 + index % 7
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "family": candidate.family,
                    "period": candidate.rule.period,
                    "indicator_units": "daily_sessions",
                    "entry_threshold": candidate.entry_threshold,
                    "exit_threshold": candidate.exit_threshold,
                    "confirmation_hours": 3.0,
                    "entry_confirmation_hours": 3.0,
                    "exit_confirmation_hours": 3.0,
                    "entry_required_checks": 7,
                    "exit_required_checks": 7,
                    "commission_bps": 1.0,
                    "slippage_bps": slip,
                    "sampling_minutes": 30,
                    "scenario_slippage_bps": slip,
                    "start_session": "2012-01-03",
                    "end_session": "2018-12-31",
                    "scored_sessions": 1760,
                    "trade_count": trades,
                    "order_count": 2 * trades,
                    "trades_per_year": trades / years,
                    "total_turnover": 2.0 * trades,
                    "annualized_turnover": 2.0 * trades / years,
                    "gross_terminal_equity": equity[-1, index] + 5.0,
                    "cost_drag_return": 0.005,
                    "cash_terminal_equity": cash_wealth[-1],
                    "BH_cash_excess_sharpe": bh,
                    "Sharpe_difference_vs_BH": score - bh,
                    "beats_BH_Sharpe": np.isfinite(score) and score - bh > study.TOLERANCE,
                    "post_gap_confirmed_buy_count": 0,
                    "post_gap_confirmed_sell_count": 0,
                    "post_gap_confirmed_signal_count": 0,
                    "has_post_gap_confirmed_signal": False,
                    **candidate_stats,
                }
            )
        frame = pd.DataFrame(rows)
        study.development.family_summary(frame, bh).to_csv(
            private / f"family_summary_{slip:g}.csv", index=False
        )
        all_rows.extend(rows)
    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(private / "candidate_metrics_all_costs.csv", index=False)
    baseline = metrics.loc[metrics.scenario_slippage_bps.eq(3)].reset_index(drop=True)
    baseline.to_csv(private / "candidate_metrics.csv", index=False)
    pd.DataFrame(benchmarks).to_csv(private / "benchmark_metrics.csv", index=False)
    pd.DataFrame(selection["development_selection_rows"]).to_csv(
        private / "development_selection_metrics.csv", index=False
    )
    flag_fields = [
        "candidate_id",
        "post_gap_confirmed_buy_count",
        "post_gap_confirmed_sell_count",
        "post_gap_confirmed_signal_count",
        "has_post_gap_confirmed_signal",
    ]
    baseline[flag_fields].to_csv(private / "post_gap_candidate_flags.csv", index=False)
    pd.DataFrame(
        columns=["candidate_id", "side", "decision_at", "gap_start", "gap_end", "window_end"]
    ).to_csv(private / "post_gap_signal_events.csv", index=False)
    (private / "post_gap_diagnostic_metadata.json").write_text(
        json.dumps(
            {
                "descriptive_only": True,
                "used_for_filtering_ranking_or_selection": False,
                "truncated": False,
                "stored_event_rows": 0,
                "total_event_rows": 0,
            }
        )
    )
    return private, selection


@pytest.fixture
def synthetic_paths(synthetic_paths_master, tmp_path):
    import shutil

    master, selection = synthetic_paths_master
    private = tmp_path / "invented_wealth"
    shutil.copytree(master, private)
    return private, copy.deepcopy(selection)


def rewrite_paths(path, transform):
    with np.load(path, allow_pickle=False) as arrays:
        values = {key: arrays[key].copy() for key in arrays.files}
    transform(values)
    np.savez_compressed(path, **values)


def test_derived_audit_accepts_all128_invented_paths_and_eight_passive_curves(
    synthetic_paths_master,
):
    private, selection = synthetic_paths_master
    metrics, baseline, benchmark = study.review_derived_results(private, selection)
    assert len(metrics) == 128
    assert len(baseline) == 32
    assert len(benchmark) == 8
    assert set(metrics.candidate_id) == {
        c.candidate_id for c in study.candidates_from_selection(selection)
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("cash_excess_sharpe", 9.0),
        ("cagr", 9.0),
        ("cost_drag_return", 9.0),
        ("annualized_turnover", 9.0),
        ("order_count", 21),
        ("BH_cash_excess_sharpe", 9.0),
        ("beats_BH_Sharpe", True),
        ("entry_confirmation_hours", 4.0),
    ],
)
def test_derived_audit_rejects_relabelled_scores_costs_trades_and_comparisons(
    synthetic_paths, field, value
):
    private, selection = synthetic_paths
    path = private / "candidate_metrics_all_costs.csv"
    frame = pd.read_csv(path, float_precision="round_trip")
    frame.loc[0, field] = value
    frame.to_csv(path, index=False)
    with pytest.raises(study.Error):
        study.review_derived_results(private, selection)


@pytest.mark.parametrize("change", ["returns", "wealth", "clock", "exposure", "candidate"])
def test_derived_audit_rejects_changed_wealth_clock_identity_and_unreconciled_returns(
    synthetic_paths, change
):
    private, selection = synthetic_paths

    def mutate(values):
        if change == "returns":
            values["returns"][10, 0] += 0.01
        elif change == "wealth":
            values["equity"][10, 0] = 0.0
        elif change == "clock":
            values["timestamp"][0] = "2012-01-03T21:00:00+00:00"
        elif change == "exposure":
            values["exposure"][10, 0] = 1.1
        else:
            values["candidate_id"][0] = values["candidate_id"][1]

    rewrite_paths(private / "slippage_1_daily.npz", mutate)
    with pytest.raises(study.Error):
        study.review_derived_results(private, selection)


def test_derived_audit_rejects_inflated_family_medians(synthetic_paths):
    private, selection = synthetic_paths
    path = private / "family_summary_1.csv"
    frame = pd.read_csv(path, float_precision="round_trip")
    frame.loc[0, "all_median_cash_excess_sharpe"] = 9.0
    frame.to_csv(path, index=False)
    with pytest.raises(AssertionError):
        study.review_derived_results(private, selection)


def test_derived_audit_rejects_replaced_development_selection_rows(synthetic_paths):
    private, selection = synthetic_paths
    path = private / "development_selection_metrics.csv"
    frame = pd.read_csv(path, float_precision="round_trip")
    frame.loc[0, "cash_excess_sharpe"] = 9.0
    frame.to_csv(path, index=False)
    with pytest.raises(AssertionError):
        study.review_derived_results(private, selection)


def test_derived_audit_keeps_zero_gap_flags_descriptive_and_untruncated(synthetic_paths):
    private, selection = synthetic_paths
    path = private / "post_gap_diagnostic_metadata.json"
    value = json.loads(path.read_text())
    value["used_for_filtering_ranking_or_selection"] = True
    path.write_text(json.dumps(value))
    with pytest.raises(study.Error, match="selection-affecting"):
        study.review_derived_results(private, selection)


def test_derived_audit_rejects_holiday_replacing_session_even_with_common_clock(synthetic_paths):
    private, selection = synthetic_paths
    replacement = "2012-01-16T22:00:00+00:00"  # MLK holiday, between adjacent valid sessions.
    for slip in study.SLIPPAGES:

        def mutate(values):
            index = np.flatnonzero(np.char.startswith(values["timestamp"], "2012-01-17"))[0]
            values["timestamp"][index] = replacement

        rewrite_paths(private / f"slippage_{slip:g}_daily.npz", mutate)
        for portfolio in ("cash", "buy_and_hold"):
            path = private / f"{portfolio}_{slip:g}_daily.csv"
            frame = pd.read_csv(path, float_precision="round_trip")
            frame.loc[frame.timestamp.str.startswith("2012-01-17"), "timestamp"] = replacement
            frame.to_csv(path, index=False)
    with pytest.raises(study.Error, match="clock"):
        study.review_derived_results(private, selection)
