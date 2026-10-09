"""Synthetic development-only mean-reversion seals, ranking and publication gates.

All provenance/packet adapters are monkeypatched to generated metadata or
sentinels. These tests never read market prices or previously observed outcomes.
"""

import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from trend_following import research_mean_reversion as features
from trend_following import research_mean_reversion_protocol as mr
from trend_following import research_quality_engine_v5 as quality


def write_config(path, value=None):
    value = copy.deepcopy(mr.approved_config()) if value is None else value
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    return path


def write_authorization(path, config=None):
    config = mr.approved_config() if config is None else config
    path.write_text(
        json.dumps(
            {
                "authorized_by": "human_user",
                "explicit_user_approval": True,
                "authorized_stages": ["development"],
                "validation_authorized": False,
                "historical_oos_authorized": False,
                "approved_config_sha256": mr.protocol.canonical_hash(config),
                "user_request_excerpt": mr.APPROVAL_EXCERPT,
            }
        )
    )
    return path


def synthetic_metrics():
    return pd.DataFrame(
        {
            "candidate_id": c.candidate_id,
            "family": c.family,
            "period": c.rule.period,
            "cash_excess_sharpe": -0.5,
            "cagr": -0.01,
            "max_drawdown": -0.30,
            "trade_count": 100,
            "trades_per_year": 10.0,
            "beats_BH_Sharpe": False,
            "post_gap_confirmed_signal_count": 0,
        }
        for c in features.candidate_grid()
    )


@pytest.fixture
def sealed_context(tmp_path, monkeypatch):
    """Real immutable seal writes with wholly synthetic bound provenance."""
    monkeypatch.setattr(mr, "ROOT", tmp_path)
    source = tmp_path / ".cache/original_synthetic_study"
    source.mkdir(parents=True)
    source_config = tmp_path / "source_config.yaml"
    source_config.write_text("synthetic: true\n")
    config = write_config(tmp_path / "mean_reversion.yaml")
    authorization = write_authorization(tmp_path / "authorization.json")
    impl = tmp_path / "synthetic_source.py"
    impl.write_text("# synthetic source binding\n")
    monkeypatch.setattr(mr, "SOURCE_FILES", (impl.name,))
    files = {n: {"sha256": "synthetic-" + n} for n in mr.fixed.PACKET_FILES}
    prepared = {"prefixes": {"development": {"files": files}}}
    calls = []

    def metadata(study, path):
        assert study == source and path == source_config
        calls.append((study, path))
        return {}, {"registration": "synthetic"}, prepared, {"stage": "development"}

    monkeypatch.setattr(mr.fixed, "_read_metadata", metadata)
    monkeypatch.setattr(mr.protocol, "load_config", lambda _: {"synthetic_source": True})
    monkeypatch.setattr(
        mr.protocol,
        "_quality_policy",
        lambda _: quality.QualityPolicy("observed_only_blackout_v1", post_gap_diagnostics=True),
    )
    monkeypatch.setattr(mr.protocol, "_runtime", lambda: {"runtime": "synthetic"})
    # Broad/future access must remain unreachable even during binding verification.
    for name in ("_verify_prepared", "_load_stage"):
        monkeypatch.setattr(
            mr.protocol,
            name,
            lambda *a, **k: pytest.fail("Forbidden broad/later-stage packet reader invoked"),
        )
    return dict(
        sidecar=tmp_path / ".cache/mean_reversion_synthetic",
        authorization=authorization,
        config_path=config,
        source_study=source,
        source_config=source_config,
        impl=impl,
        calls=calls,
    )


def seal_kwargs(context):
    return {
        k: context[k] for k in ("authorization", "config_path", "source_study", "source_config")
    }


def test_approved_config_matches_public_file_and_ninety_complete_family_memberships(tmp_path):
    actual = mr.load_config(write_config(tmp_path / "approved.yaml"))
    assert actual == mr.approved_config()
    assert actual["signals"]["confirmation_hours"] == 3
    assert actual["signals"]["required_consecutive_checks"] == 7
    counts = Counter(c.family for c in features.candidate_grid())
    assert counts == {"SMA_DISTANCE": 18, "EMA_DISTANCE": 18, "ZSCORE": 27, "RSI": 27}
    assert len({c.candidate_id for c in features.candidate_grid()}) == 90
    assert not actual["protocol"]["validation_authorized"]
    assert not actual["protocol"]["historical_oos_authorized"]


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("grid", "same_family_and_period_for_entry_exit", 1),
        ("signals", "extra_risk_exits", 0),
        ("signals", "zscore_ddof", True),
        ("signals", "confirmation_hours", False),
        ("accounting", "taxes", 0),
        ("accounting", "commission_bps", True),
        ("accounting", "cash_nominal_annual_rate", 0.04),
        ("selection", "finite_scores_only", 1),
        ("selection", "minimum_trade_filter", True),
        ("protocol", "validation_authorized", 0),
        ("protocol", "historical_oos_authorized", True),
    ],
)
def test_config_rejects_numeric_boolean_substitution_and_unapproved_changes(
    tmp_path, section, key, value
):
    config = copy.deepcopy(mr.approved_config())
    config[section][key] = value
    with pytest.raises(mr.Error, match="contract"):
        mr.load_config(write_config(tmp_path / "changed.yaml", config))


def test_config_rejects_extra_fields_and_changed_daily_lookback(tmp_path):
    config = copy.deepcopy(mr.approved_config())
    config["grid"]["ma_periods"].append(200)
    with pytest.raises(mr.Error):
        mr.load_config(write_config(tmp_path / "grid.yaml", config))
    config = copy.deepcopy(mr.approved_config())
    config["silent_fallback"] = True
    with pytest.raises(mr.Error):
        mr.load_config(write_config(tmp_path / "extra.yaml", config))


@pytest.mark.parametrize(
    "key,value",
    [
        ("explicit_user_approval", 1),
        ("authorized_by", "assistant"),
        ("authorized_stages", ["development", "validation"]),
        ("validation_authorized", 0),
        ("historical_oos_authorized", True),
        ("approved_config_sha256", "not-the-config"),
        ("user_request_excerpt", "run anything"),
    ],
)
def test_authorization_invalid_before_any_provenance_read(sealed_context, monkeypatch, key, value):
    context = sealed_context
    path = context["authorization"]
    receipt = json.loads(path.read_text())
    receipt[key] = value
    path.write_text(json.dumps(receipt))
    monkeypatch.setattr(
        mr.fixed,
        "_read_metadata",
        lambda *a, **k: pytest.fail("No provenance read before exact human-authorization gate"),
    )
    with pytest.raises(mr.Error, match="authorization"):
        mr._bindings(
            context["config_path"], context["source_study"], context["source_config"], path
        )


def test_bindings_only_generated_development_metadata_and_explicit_shared_policy(sealed_context):
    c = sealed_context
    payload = mr._bindings(
        c["config_path"], c["source_study"], c["source_config"], c["authorization"]
    )
    assert c["calls"] == [(c["source_study"], c["source_config"])]
    assert set(payload["development_files"]) == set(mr.fixed.PACKET_FILES)
    assert len(payload["candidates"]) == 90
    assert payload["period"] == {"start": "2001-01-01", "end": "2011-12-31"}
    assert payload["confirmation_hours"] == 3
    assert payload["cash"]["annual_rate"] == 0.03
    assert payload["validation_authorized"] is payload["historical_oos_authorized"] is False
    assert payload["no_prior_strategy_outcomes_used"] is True


def test_freeze_is_idempotent_and_cannot_replace_changed_source_binding(sealed_context):
    c = sealed_context
    first = mr.seal_study(c["sidecar"], **seal_kwargs(c))
    before = (c["sidecar"] / "freeze.json").read_bytes()
    assert mr.seal_study(c["sidecar"], **seal_kwargs(c)) == first
    assert mr.verify_study(c["sidecar"]) == first
    assert (c["sidecar"] / "freeze.json").read_bytes() == before
    c["impl"].write_text("# changed synthetic source binding\n")
    with pytest.raises(mr.Error, match="changed"):
        mr.verify_study(c["sidecar"])
    with pytest.raises(mr.Error, match="sealed differently"):
        mr.seal_study(c["sidecar"], **seal_kwargs(c))
    assert (c["sidecar"] / "freeze.json").read_bytes() == before


def test_freeze_detects_changed_config_bytes_even_if_yaml_meaning_is_same(sealed_context):
    c = sealed_context
    mr.seal_study(c["sidecar"], **seal_kwargs(c))
    with c["config_path"].open("a") as stream:
        stream.write("# semantically harmless but different frozen bytes\n")
    with pytest.raises(mr.Error, match="changed"):
        mr.verify_study(c["sidecar"])


def test_freeze_detects_authorization_bytes_or_development_metadata_binding_changes(
    sealed_context, monkeypatch
):
    c = sealed_context
    mr.seal_study(c["sidecar"], **seal_kwargs(c))
    original = c["authorization"].read_text()
    c["authorization"].write_text(original + "\n")
    with pytest.raises(mr.Error, match="changed"):
        mr.verify_study(c["sidecar"])
    c["authorization"].write_text(original)
    monkeypatch.setattr(
        mr.fixed,
        "_read_metadata",
        lambda *a, **k: (
            {},
            {"registration": "changed"},
            {
                "prefixes": {
                    "development": {
                        "files": {n: {"sha256": "synthetic-" + n} for n in mr.fixed.PACKET_FILES}
                    }
                }
            },
            {"stage": "development"},
        ),
    )
    with pytest.raises(mr.Error, match="changed"):
        mr.verify_study(c["sidecar"])


def test_safe_paths_reject_symlink_leaves_and_ancestors(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "data").write_text("synthetic")
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(mr.Error, match="Symlink"):
        mr.safe_path(link)
    with pytest.raises(mr.Error, match="Symlink"):
        mr.safe_path(link / "data")


@pytest.mark.parametrize(
    "kind",
    [
        "nested_source",
        "parent_source",
        "authorization_source",
        "authorization_sidecar",
        "wrong_name",
    ],
)
def test_separate_sidecar_never_nests_source_or_authorization(sealed_context, kind):
    c = sealed_context
    sidecar, source, authorization = c["sidecar"], c["source_study"], c["authorization"]
    if kind == "nested_source":
        sidecar = source / "mean_reversion_child"
    elif kind == "parent_source":
        source = sidecar / "original_child"
    elif kind == "authorization_source":
        authorization = source / "approval.json"
    elif kind == "authorization_sidecar":
        authorization = sidecar / "approval.json"
    else:
        sidecar = sidecar.with_name("not_a_mean_reversion_study")
    with pytest.raises(mr.Error):
        mr._separate(sidecar, source, authorization)


def test_loader_forwards_only_development_binding_keys(monkeypatch):
    payload = {
        "source_study": Path("synthetic-development"),
        "source_config_path": Path("synthetic-config"),
        "development_manifest_sha256": "synthetic-manifest",
        "development_files": {"daily.csv": "synthetic"},
        "historical_oos_path": "MUST_NOT_BE_FORWARDED",
        "prior_strategy_results": "MUST_NOT_BE_READ",
    }
    received = []
    sentinel = object()

    def loader(adapter):
        received.append(adapter)
        return sentinel

    monkeypatch.setattr(mr.fixed, "load_development_packet", loader)
    assert mr.load_development(payload) is sentinel
    assert received == [
        {
            "source_study": payload["source_study"],
            "config_path": payload["source_config_path"],
            "development_manifest_sha256": payload["development_manifest_sha256"],
            "development_files": payload["development_files"],
        }
    ]


def test_finite_top5_per_family_ignores_BH_and_trade_frequency_and_gap_flags():
    metrics = synthetic_metrics()
    metrics["cash_excess_sharpe"] = np.nan
    for family in mr.FAMILIES:
        indices = metrics.index[metrics.family.eq(family)][:7]
        metrics.loc[indices, "cash_excess_sharpe"] = [-0.1, -0.2, -0.3, -0.4, -0.5, -np.inf, np.inf]
        metrics.loc[indices, "trade_count"] = [0, 10**9, 0, 0, 1, 5, 5]
        metrics.loc[indices, "post_gap_confirmed_signal_count"] = 10000
    finalists = mr.select_finalists(metrics)
    assert len(finalists) == 20
    assert Counter(finalists.family) == {family: 5 for family in mr.FAMILIES}
    assert np.isfinite(finalists.cash_excess_sharpe).all()
    assert finalists.cash_excess_sharpe.lt(0).all()
    assert not finalists.beats_BH_Sharpe.any()
    assert {0, 10**9}.issubset(set(finalists.trade_count))
    for _, family in finalists.groupby("family"):
        assert family.family_rank.tolist() == [1, 2, 3, 4, 5]


def test_selection_tolerance_is_max_relative_not_transitive_and_stable_under_shuffle():
    metrics = synthetic_metrics()
    metrics["cash_excess_sharpe"] = np.nan
    ids = sorted(metrics.loc[metrics.family.eq("SMA_DISTANCE"), "candidate_id"])[:3]
    scores = [1 - 1.6 * mr.TOLERANCE, 1 - 0.8 * mr.TOLERANCE, 1.0]
    for identity, score in zip(ids, scores, strict=True):
        metrics.loc[metrics.candidate_id.eq(identity), "cash_excess_sharpe"] = score
    # The lexically smallest ID is outside the first max-relative tie window.
    expected = [ids[1], ids[2], ids[0]]
    assert mr.select_finalists(metrics).candidate_id.tolist() == expected
    assert (
        mr.select_finalists(metrics.sample(frac=1, random_state=5)).candidate_id.tolist()
        == expected
    )


def test_empty_or_all_undefined_scores_leave_empty_finalists_without_BH_fallback():
    metrics = synthetic_metrics()
    metrics["cash_excess_sharpe"] = np.nan
    for frame in (metrics, metrics.iloc[:0]):
        finalists = mr.select_finalists(frame)
        assert finalists.empty
        assert "family_rank" in finalists


def test_family_summary_keeps_all90_and_conditions_winner_medians_explicitly():
    metrics = synthetic_metrics()
    indices = metrics.index[metrics.family.eq("SMA_DISTANCE")][:4]
    metrics.loc[indices, "cash_excess_sharpe"] = [0.2, 0.3, np.nan, np.inf]
    metrics.loc[indices[:2], "cagr"] = [0.1, 0.2]
    metrics.loc[indices[:2], "trade_count"] = [10, 30]
    summary = mr.family_summary(metrics, 0.0).set_index("family")
    sma = summary.loc["SMA_DISTANCE"]
    assert sma.configurations == 18
    assert sma.finite_Sharpe_count == 16
    assert sma.undefined_Sharpe_count == 2
    assert sma.beat_BH_Sharpe_count == 2
    assert sma.BH_beaters_median_cash_excess_sharpe == pytest.approx(0.25)
    assert sma.BH_beaters_median_cagr == pytest.approx(0.15)
    assert sma.BH_beaters_median_trade_count == 20
    assert sma.all_median_trade_count == 100
    assert summary.loc["ALL", "configurations"] == 90
    assert summary.loc["ALL", "beat_BH_Sharpe_count"] == 2
    assert np.isnan(summary.loc["RSI", "BH_beaters_median_trade_count"])
    pd.testing.assert_frame_equal(metrics, metrics.copy())


def test_beat_BH_uses_finite_strict_tolerance_not_any_positive_difference():
    metrics = synthetic_metrics()
    metrics.loc[0, "cash_excess_sharpe"] = 0.5 + 0.5 * mr.TOLERANCE
    metrics.loc[1, "cash_excess_sharpe"] = 0.5 + 2 * mr.TOLERANCE
    row = mr.family_summary(metrics, 0.5).set_index("family").loc["ALL"]
    assert row.beat_BH_Sharpe_count == 1


def test_existing_results_cannot_rerun_before_loading_packet(tmp_path, monkeypatch):
    sidecar = tmp_path / "mean_reversion_synthetic"
    (sidecar / "results").mkdir(parents=True)
    monkeypatch.setattr(mr, "verify_study", lambda _: {"synthetic": True})
    monkeypatch.setattr(
        mr, "load_development", lambda *_: pytest.fail("Existing results must not reread data")
    )
    with pytest.raises(mr.Error, match="cannot be overwritten"):
        mr.evaluate_study(sidecar)


@pytest.mark.parametrize("command", ["validation", "historical_oos", "oos", "monitor"])
def test_cli_exposes_no_later_stage_or_monitor_route(command, monkeypatch):
    monkeypatch.setattr(
        mr, "evaluate_study", lambda *_: pytest.fail("Forbidden CLI command reached evaluation")
    )
    with pytest.raises(SystemExit) as error:
        mr.main([command])
    assert error.value.code == 2


def test_check_command_reports_closed_later_stages_without_loading_prices(
    tmp_path, monkeypatch, capsys
):
    config = write_config(tmp_path / "config.yaml")
    monkeypatch.setattr(mr.fixed, "_read_metadata", lambda *_: (None,) * 4)
    monkeypatch.setattr(
        mr, "load_development", lambda *_: pytest.fail("Check must not load numeric market data")
    )
    assert (
        mr.main(
            [
                "check",
                "--config",
                str(config),
                "--source-study",
                str(tmp_path / "source"),
                "--source-config",
                str(tmp_path / "source_config"),
            ]
        )
        == 0
    )
    value = json.loads(capsys.readouterr().out)
    assert value["configurations"] == 90
    assert value["shared_wait_hours"] == 3
    assert value["validation_authorized"] is value["historical_oos_authorized"] is False


def test_report_verifies_result_integrity_before_any_csv_read_or_output_creation(
    tmp_path, monkeypatch
):
    sidecar, output = tmp_path / "private", tmp_path / "new-public"

    def verification(_):
        raise mr.Error("Synthetic integrity failure")

    monkeypatch.setattr(mr, "verify_results", verification)
    monkeypatch.setattr(
        mr.pd, "read_csv", lambda *a, **k: pytest.fail("Never read unsigned result values")
    )
    with pytest.raises(mr.Error, match="integrity"):
        mr.report(sidecar, output)
    assert not output.exists()


def test_public_projection_omits_raw_quotes_paths_credentials_and_private_receipts():
    frame = synthetic_metrics().iloc[:1].copy()
    for name, value in (
        ("open", 123),
        ("close", 456),
        ("reference_price", 789),
        ("source_study", "/Users/private/.cache/data"),
        ("authorization", "private.json"),
        ("credential", "secret@example.com"),
    ):
        frame[name] = value
    public = mr._public_frame(frame, mr.PUBLIC_METRICS)
    assert set(public.columns).issubset(set(mr.PUBLIC_METRICS))
    assert not set(
        ("open", "close", "reference_price", "source_study", "authorization", "credential")
    ) & set(public)
    assert len(public) == len(frame)


@pytest.mark.parametrize(
    "private_text", ["/Users/private/path", ".cache/prices.csv", "private@example.com"]
)
def test_private_text_inside_allowed_public_column_is_rejected(private_text):
    frame = synthetic_metrics().iloc[:1].copy()
    frame["candidate_id"] = private_text
    with pytest.raises(mr.Error, match="Private text"):
        mr._public_frame(frame, mr.PUBLIC_METRICS)


def test_public_json_clean_maps_undefined_to_null_not_infinity_or_nan():
    cleaned = mr._clean(
        {
            "undefined": np.nan,
            "infinite": np.inf,
            "nested": [np.float64(1.0), -np.inf],
            "count": np.int64(90),
        }
    )
    assert cleaned == {"undefined": None, "infinite": None, "nested": [1.0, None], "count": 90}
    assert "NaN" not in json.dumps(cleaned, allow_nan=False)


@pytest.fixture(scope="module")
def synthetic_derived_master(tmp_path_factory):
    """Fabricated daily equity paths on exchange times; not market prices."""
    from trend_following.research_hourly_data_v5 import build_xnys_calendar

    private = tmp_path_factory.mktemp("fabricated_mean_reversion_paths")
    calendar = build_xnys_calendar("2001-01-01", "2011-12-31")
    times = pd.DatetimeIndex(
        [
            pd.Timestamp(f"{s} 17:00", tz="America/New_York").tz_convert("UTC")
            for s in calendar.session
        ]
    )
    assert len(times) == 2767
    first = pd.Timestamp("2001-01-02 09:30", tz="America/New_York").tz_convert("UTC")
    years = (times[-1] - first).total_seconds() / (365 * 86400)
    candidates = features.candidate_grid()
    t = np.arange(len(times), dtype=float)
    base = 0.001 * np.sin(t * 0.23) + 0.0008 * np.cos(t * 0.037)
    cash_returns = np.zeros(len(times))
    all_metrics, benchmark_metrics = [], []

    def statistics(equity, returns):
        excess = returns - cash_returns
        sd = excess.std(ddof=1)
        wealth = np.r_[1000.0, equity]
        return dict(
            cash_excess_sharpe=float(np.sqrt(252) * excess.mean() / sd) if sd > 0 else np.nan,
            terminal_equity=equity[-1],
            cagr=(equity[-1] / 1000) ** (1 / years) - 1,
            max_drawdown=(wealth / np.maximum.accumulate(wealth) - 1).min(),
            annualized_volatility=returns.std(ddof=1) * np.sqrt(252),
        )

    for slip in mr.SLIPPAGES:
        column = np.arange(90)[None, :]
        invented_returns = base[:, None] * (1 + column / 200) + (column + 1) * 1e-6 - slip * 1e-7
        equity = 1000 * np.cumprod(1 + invented_returns, axis=0)
        returns = equity / np.vstack([np.full(90, 1000.0), equity[:-1]]) - 1
        np.savez_compressed(
            private / f"slippage_{slip:g}_daily.npz",
            timestamp=np.asarray([x.isoformat() for x in times]),
            candidate_id=np.asarray([c.candidate_id for c in candidates]),
            equity=equity,
            returns=returns,
            cash_returns=cash_returns,
            exposure=np.zeros_like(equity),
        )
        reference = {}
        for portfolio in ("cash", "buy_and_hold"):
            fabricated = (
                np.zeros(len(times)) if portfolio == "cash" else base * 1.3 + 0.00002 - slip * 1e-7
            )
            wealth = 1000 * np.cumprod(1 + fabricated)
            passive_returns = wealth / np.r_[1000, wealth[:-1]] - 1
            stats = statistics(wealth, passive_returns)
            reference[portfolio] = stats
            pd.DataFrame(
                {
                    "timestamp": times,
                    "equity": wealth,
                    "return": passive_returns,
                    "cash_return": cash_returns,
                    "exposure": np.zeros(len(times)),
                }
            ).to_csv(private / f"{portfolio}_{slip:g}_daily.csv", index=False)
            benchmark_metrics.append(
                dict(
                    portfolio=portfolio,
                    scenario_slippage_bps=slip,
                    commission_bps=1.0,
                    slippage_bps=slip,
                    trade_count=0 if portfolio == "cash" else 1,
                    order_count=0 if portfolio == "cash" else 2,
                    start_session="2001-01-02",
                    end_session="2011-12-30",
                    scored_sessions=2767,
                    trades_per_year=0 if portfolio == "cash" else 1 / years,
                    **stats,
                )
            )
        bh = reference["buy_and_hold"]["cash_excess_sharpe"]
        for i, candidate in enumerate(candidates):
            stats = statistics(equity[:, i], returns[:, i])
            trades = 10 + i % 7
            difference = stats["cash_excess_sharpe"] - bh
            all_metrics.append(
                dict(
                    candidate_id=candidate.candidate_id,
                    family=candidate.family,
                    period=candidate.rule.period,
                    entry_threshold=candidate.entry_threshold,
                    exit_threshold=candidate.exit_threshold,
                    entry_confirmation_hours=3.0,
                    exit_confirmation_hours=3.0,
                    entry_required_checks=7,
                    exit_required_checks=7,
                    commission_bps=1.0,
                    slippage_bps=slip,
                    sampling_minutes=30,
                    start_session="2001-01-02",
                    end_session="2011-12-30",
                    scored_sessions=2767,
                    trade_count=trades,
                    order_count=2 * trades,
                    trades_per_year=trades / years,
                    scenario_slippage_bps=slip,
                    BH_cash_excess_sharpe=bh,
                    Sharpe_difference_vs_BH=difference,
                    beats_BH_Sharpe=np.isfinite(difference) and difference > mr.TOLERANCE,
                    **stats,
                )
            )
    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(private / "candidate_metrics_all_costs.csv", index=False)
    baseline = metrics.loc[metrics.scenario_slippage_bps.eq(3.0)].copy()
    baseline.to_csv(private / "candidate_metrics.csv", index=False)
    pd.DataFrame(benchmark_metrics).to_csv(private / "benchmark_metrics.csv", index=False)
    # Synthetic scores differ by much more than1e-10; independently sort each
    # family rather than using the production finalist selector to manufacture expectations.
    selected = []
    for family in mr.FAMILIES:
        best = (
            baseline.loc[baseline.family.eq(family)]
            .sort_values(["cash_excess_sharpe", "candidate_id"], ascending=[False, True])
            .head(5)
            .copy()
        )
        best["family_rank"] = np.arange(1, len(best) + 1)
        selected.append(best)
    pd.concat(selected, ignore_index=True).to_csv(
        private / "development_finalists.csv", index=False
    )
    return private


@pytest.fixture
def synthetic_derived(synthetic_derived_master, tmp_path):
    import shutil

    private = tmp_path / "fabricated_derived"
    shutil.copytree(synthetic_derived_master, private)
    return private


def rewrite_array(path, transform):
    with np.load(path, allow_pickle=False) as arrays:
        values = {key: arrays[key].copy() for key in arrays.files}
    transform(values)
    np.savez_compressed(path, **values)


def test_derived_review_accepts_all360_fabricated_paths_and_twenty_finalists(
    synthetic_derived_master,
):
    metrics, baseline, benchmark, finalists = mr.review_derived_results(synthetic_derived_master)
    assert len(metrics) == 360 and len(baseline) == 90 and len(benchmark) == 8
    assert len(finalists) == 20
    assert Counter(finalists.family) == {family: 5 for family in mr.FAMILIES}


def test_derived_review_rejects_duplicate_passive_benchmark(synthetic_derived):
    path = synthetic_derived / "benchmark_metrics.csv"
    frame = pd.read_csv(path, float_precision="round_trip")
    frame.iloc[-1] = frame.iloc[0]
    frame.to_csv(path, index=False)
    with pytest.raises(mr.Error, match="eight matched"):
        mr.review_derived_results(synthetic_derived)


@pytest.mark.parametrize(
    "field,value",
    [
        ("family", "RSI"),
        ("entry_threshold", -0.07),
        ("entry_confirmation_hours", 2.0),
        ("exit_required_checks", 6),
        ("period", 200),
    ],
)
def test_derived_review_rejects_candidate_identity_relabel(synthetic_derived, field, value):
    path = synthetic_derived / "candidate_metrics_all_costs.csv"
    frame = pd.read_csv(path, float_precision="round_trip")
    frame.loc[0, field] = value
    frame.to_csv(path, index=False)
    with pytest.raises(mr.Error, match="identity, period or accounting"):
        mr.review_derived_results(synthetic_derived)


def test_derived_review_rejects_returns_not_reconciled_to_equity(synthetic_derived):
    def change(values):
        values["returns"][10, 0] += 0.01

    rewrite_array(synthetic_derived / "slippage_1_daily.npz", change)
    with pytest.raises(mr.Error, match="Daily returns"):
        mr.review_derived_results(synthetic_derived)


def test_derived_review_rejects_unsupported_sharpe_relabel(synthetic_derived):
    path = synthetic_derived / "candidate_metrics_all_costs.csv"
    frame = pd.read_csv(path, float_precision="round_trip")
    frame.loc[0, "cash_excess_sharpe"] += 0.1
    frame.to_csv(path, index=False)
    with pytest.raises(mr.Error, match="Derived cash_excess_sharpe"):
        mr.review_derived_results(synthetic_derived)


@pytest.mark.parametrize("timestamp", ["2000-12-31T22:00:00+00:00", "2001-01-02T21:00:00+00:00"])
def test_derived_review_rejects_premature_or_non17NY_valuation_clock(synthetic_derived, timestamp):
    def change(values):
        values["timestamp"][0] = timestamp

    rewrite_array(synthetic_derived / "slippage_1_daily.npz", change)
    with pytest.raises(mr.Error, match="clock/wealth"):
        mr.review_derived_results(synthetic_derived)


def test_derived_review_rejects_saved_finalist_membership_mutation(synthetic_derived):
    path = synthetic_derived / "development_finalists.csv"
    frame = pd.read_csv(path, float_precision="round_trip")
    frame.loc[0, "candidate_id"] = frame.loc[1, "candidate_id"]
    frame.to_csv(path, index=False)
    with pytest.raises(AssertionError):
        mr.review_derived_results(synthetic_derived)


def test_derived_review_rejects_cost_matched_BH_comparison_mutation(synthetic_derived):
    path = synthetic_derived / "candidate_metrics_all_costs.csv"
    frame = pd.read_csv(path, float_precision="round_trip")
    frame.loc[0, "BH_cash_excess_sharpe"] += 0.01
    frame.to_csv(path, index=False)
    with pytest.raises(mr.Error, match="BH comparison"):
        mr.review_derived_results(synthetic_derived)


def test_derived_review_rejects_annualized_trade_count_or_roundtrip_order_mismatch(
    synthetic_derived,
):
    path = synthetic_derived / "candidate_metrics_all_costs.csv"
    frame = pd.read_csv(path, float_precision="round_trip")
    frame.loc[0, "trades_per_year"] += 0.1
    frame.to_csv(path, index=False)
    with pytest.raises(mr.Error, match="Annualized trade count"):
        mr.review_derived_results(synthetic_derived)
    frame.loc[0, "trades_per_year"] -= 0.1
    frame.loc[0, "order_count"] += 1
    frame.to_csv(path, index=False)
    with pytest.raises(mr.Error, match="Round trips"):
        mr.review_derived_results(synthetic_derived)
