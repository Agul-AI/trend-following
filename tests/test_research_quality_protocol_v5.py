"""Quality governance uses SYNTHETIC packets/mocked metrics, never market outcomes."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from trend_following import research_daily_hourly_v5 as core
from trend_following import research_quality_engine_v5 as quality
from trend_following import research_quality_protocol_v5 as protocol

ACTIVE_CONFIG = protocol.WORKSPACE / "configs/qqq_daily_halfhour_v5_gap_flags_v2.yaml"


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    path = tmp_path / "quality.yaml"
    # Exercise strict safety gates as a synthetic variation of the current study,
    # without retaining or reading a superseded study configuration.
    config = yaml.safe_load(ACTIVE_CONFIG.read_text())
    config["quality_policy"].update(profile="strict_tradable_v1", gap_authorization_receipt=None)
    config.pop("diagnostics", None)
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(protocol, "_source_hashes", lambda: {"SYNTHETIC_CODE": "a" * 64})
    monkeypatch.setattr(protocol, "_runtime", lambda: {"python": "TEST", "packages": {}})
    return path


def set_profile(config_file, tmp_path):
    receipt = {
        "schema": "qqq_quality_profile_authorization_v1",
        "study_id": protocol.STUDY_ID,
        "quality_profile": "observed_only_blackout_v1",
        "explicit_user_approval": True,
        "authorized_by": "human_user",
        "acknowledges_unrepaired_source_gaps": True,
        "approved_at": "2026-01-01T00:00:00Z",
        "instruction": "SYNTHETIC TEST RECEIPT, NOT AN ACTUAL USER APPROVAL",
    }
    path = tmp_path / "TEST-auth.json"
    path.write_text(json.dumps(receipt))
    config = yaml.safe_load(config_file.read_text())
    config["quality_policy"].update(
        profile="observed_only_blackout_v1", gap_authorization_receipt=str(path)
    )
    config_file.write_text(yaml.safe_dump(config))
    return path


def test_strict_synthetic_profile_roles_and_grid(config_file):
    config = protocol.load_config(config_file)
    assert config["quality_policy"]["profile"] == "strict_tradable_v1"
    assert protocol._authorization(config) is None
    assert config["quality_policy"]["valuation_hour_ny"] == 17
    assert config["quote_roles"]["daily_quote_is_NOCP_guaranteed"] is False
    assert config["quote_roles"]["intraday_is_native_raw_capture"] is False
    assert config["data_contract"]["daily_intraday_closing_identity_required"] is False
    assert len(protocol._candidate_table(config)[0]) == 56644


def test_missing_packet_registers_no_performance(config_file, tmp_path, monkeypatch):
    monkeypatch.setattr(
        quality,
        "evaluate_quality_grid",
        lambda *a, **k: pytest.fail("No performance at missing-data gate"),
    )
    result = protocol.prepare_quality_study(
        config_file,
        study_dir=tmp_path / "study",
        input_packet=tmp_path / "absent",
        cash_file=tmp_path / "absent.csv",
    )
    assert result["status"] == "data_not_ready" and result["performance_run"] is False
    assert result["registered_candidates"] == 56644
    assert result["native_raw_capture"] is False


def test_profile_alone_is_not_human_permission(config_file):
    config = yaml.safe_load(config_file.read_text())
    config["quality_policy"]["profile"] = "observed_only_blackout_v1"
    config_file.write_text(yaml.safe_dump(config))
    with pytest.raises(protocol.ProtocolError, match="HUMAN"):
        protocol._authorization(protocol.load_config(config_file))


@pytest.mark.parametrize(
    "key,value",
    [
        ("explicit_user_approval", False),
        ("authorized_by", "agent"),
        ("acknowledges_unrepaired_source_gaps", False),
        ("quality_profile", "strict_tradable_v1"),
    ],
)
def test_bad_permission_receipts_rejected(config_file, tmp_path, key, value):
    path = set_profile(config_file, tmp_path)
    receipt = json.loads(path.read_text())
    receipt[key] = value
    path.write_text(json.dumps(receipt))
    with pytest.raises(protocol.ProtocolError, match="explicit human consent"):
        protocol._authorization(protocol.load_config(config_file))


def test_all_supplied_unknowns_block_even_before_scored_start(config_file):
    coverage = pd.DataFrame(
        {
            "session": ["2000-01-03"],
            "bar_start": [pd.Timestamp("2000-01-03T14:30Z")],
            "bar_end": [pd.Timestamp("2000-01-03T15:00Z")],
            "status": ["unknown_missing"],
            "signal_eligible": [False],
            "fill_eligible": [False],
        }
    )
    with pytest.raises(protocol.ProtocolError, match="rejects unknown"):
        protocol._quality_gate(protocol.load_config(config_file), coverage, None)
    coverage["status"] = "known_halt_overlap"
    assert (
        protocol._quality_gate(protocol.load_config(config_file), coverage, None)[
            "unknown_missing_count"
        ]
        == 0
    )


class SyntheticData:
    def ReconciledBundle(self, *args):
        keys = (
            "daily",
            "bars",
            "calendar",
            "coverage",
            "blackouts",
            "unit_proof",
            "source_metadata",
            "source_files",
        )
        return SimpleNamespace(**dict(zip(keys, args, strict=True)))

    def write_reconciled_packet(
        self,
        root,
        bundle,
        *,
        cash_observations=None,
        cash_source_file=None,
        quality_profile="strict_tradable_v1",
        gap_authorization=None,
    ):
        root = Path(root)
        root.mkdir(parents=True)
        frames = {
            "daily.csv": bundle.daily,
            "bars.csv": bundle.bars,
            "calendar.csv": bundle.calendar,
            "coverage.csv": bundle.coverage,
            "blackouts.csv": bundle.blackouts,
        }
        if cash_observations is not None:
            frames["DTB3_available.csv"] = cash_observations
        for name, frame in frames.items():
            frame.to_csv(root / name, index=False)
        (root / "unit_proof.json").write_text(json.dumps(bundle.unit_proof))
        files = {
            name: {"sha256": protocol.sha256_file(root / name)}
            for name in [*frames, "unit_proof.json"]
        }
        manifest = {
            "schema": "qqq_alpha_30m_reconciled_packet_v1",
            "provider": "AlphaVantage",
            "symbol": "QQQ",
            "sampling_interval_minutes": 30,
            "observation_interval_minutes": 30,
            "source_metadata": bundle.source_metadata,
            "source_files": bundle.source_files,
            "daily_source_metadata": {
                "provider": "AlphaVantage",
                "symbol": "QQQ",
                "price_basis": "raw_as_traded",
                "price_field": "provider_daily_close",
                "verification": "SYNTHETIC_TEST_ONLY",
            },
            "native_raw_capture": False,
            "price_basis": bundle.source_metadata["price_origin"],
            "quote_role_profile": "distinct_daily_feature_and_rth_book_quotes",
            "quality_profile": quality_profile,
            "gap_authorization": gap_authorization,
            "historical_ready": not bundle.coverage.status.eq("unknown_missing").any()
            or gap_authorization is not None,
            "audit": {"sampling_interval_minutes": 30},
            "files": files,
            "unit_proof_sha256": files["unit_proof.json"]["sha256"],
        }
        (root / "manifest.json").write_text(json.dumps(manifest))
        return manifest

    def read_reconciled_packet(self, root):
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        for name, record in manifest["files"].items():
            assert protocol.sha256_file(root / name) == record["sha256"]
        frames = [
            pd.read_csv(root / name)
            for name in ("daily.csv", "bars.csv", "calendar.csv", "coverage.csv", "blackouts.csv")
        ]
        for frame, fields in zip(
            frames,
            [
                ("daily_price_available_at",),
                ("bar_start", "bar_end"),
                ("market_open", "market_close"),
                ("bar_start", "bar_end"),
                ("start", "end"),
            ],
            strict=True,
        ):
            for field in fields:
                frame[field] = pd.to_datetime(frame[field], utc=True)
        return (*frames, manifest)


@pytest.fixture
def packet_fixture(config_file, tmp_path, monkeypatch):
    sessions = [
        "1999-11-01",
        "2000-01-03",
        "2001-01-02",
        "2011-12-30",
        "2012-01-03",
        "2015-12-31",
        "2016-01-04",
        "2026-05-28",
    ]
    daily = pd.DataFrame(
        {
            "session": sessions,
            "open": 100.0,
            "high": 102.0,
            "low": 99.0,
            "close": 101.0,
            "volume": 1,
            "dividend_amount": 0.0,
            "split_coefficient": 1.0,
            "daily_price_available_at": pd.to_datetime([f"{d}T22:00Z" for d in sessions]),
        }
    )
    cal = pd.DataFrame(
        {
            "session": sessions,
            "market_open": pd.to_datetime([f"{d}T14:30Z" for d in sessions]),
            "market_close": pd.to_datetime([f"{d}T21:00Z" for d in sessions]),
        }
    )
    bars = pd.DataFrame(
        {
            "session": sessions[1:],
            "bar_start": pd.to_datetime([f"{d}T14:30Z" for d in sessions[1:]]),
            "bar_end": pd.to_datetime([f"{d}T15:00Z" for d in sessions[1:]]),
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 1,
            "duration_minutes": 30,
        }
    )
    coverage = bars[["session", "bar_start", "bar_end"]].copy()
    coverage["status"], coverage["signal_eligible"], coverage["fill_eligible"] = (
        "observed",
        True,
        True,
    )
    blackouts = pd.DataFrame(columns=["start", "end", "kind", "reason", "source_url"])
    raw = tmp_path / "SYNTHETIC-SOURCE.txt"
    raw.write_text("SYNTHETIC TEST PACKET; NEVER MARKET DATA")
    proof = {
        "unit_mapping_proof_passed": True,
        "quote_relationship_reconciled_by_explicit_roles": True,
        "native_raw_capture": False,
        "independent_price_anchor_proof": {
            "minimum_required_anchor_days_per_month": 10,
            "minimum_anchor_days_per_month": 10,
            "unanchored_months": [],
            "daily_close_not_used_as_an_anchor": True,
        },
    }
    meta = {
        "provider": "AlphaVantage",
        "symbol": "QQQ",
        "price_origin": protocol.load_config(config_file)["data_contract"]["price_basis"],
        "coverage_start_session": "2000-01-03",
        "coverage_end_session": "2026-05-28",
        "required_study_start_session": "2001-01-02",
    }
    adapter = SyntheticData()
    bundle = adapter.ReconciledBundle(
        daily,
        bars,
        cal,
        coverage,
        blackouts,
        proof,
        meta,
        [{"path": str(raw), "sha256": protocol.sha256_file(raw)}],
    )
    master = tmp_path / "master"
    adapter.write_reconciled_packet(master, bundle)
    cash_path = tmp_path / "cash.csv"
    pd.DataFrame(
        {
            "available_at": [f"{d}T14:00Z" for d in sessions],
            "observation_date": sessions,
            "annual_rate": 0.03,
            "series_id": "DTB3",
        }
    ).to_csv(cash_path, index=False)
    monkeypatch.setattr(protocol, "_data", lambda: adapter)
    monkeypatch.setattr(
        protocol,
        "_check_prefix",
        lambda d, b, c, rates, stage, config: [
            pytest.fail("Future numeric price")
            for frame in (d, b, c)
            if frame.session.gt(protocol.SPLITS[stage]["end"]).any()
        ],
    )
    calls = []

    def evaluate(d, b, *, candidates, coverage, blackouts, quality_policy, **kwargs):
        assert d.session.max() <= kwargs["end"]
        assert coverage.session.max() <= kwargs["end"]
        assert quality_policy.valuation_hour_ny == 17
        calls.append((kwargs["start"], len(candidates)))
        return pd.DataFrame(
            {"candidate_id": [c.candidate_id for c in candidates], "cash_excess_sharpe": 1.0}
        )

    fake_engine = SimpleNamespace(
        QualityPolicy=quality.QualityPolicy,
        generate_candidates=core.generate_candidates,
        evaluate_quality_grid=evaluate,
    )
    monkeypatch.setattr(protocol, "_engine", lambda: fake_engine)
    return SimpleNamespace(
        config=config_file,
        study=tmp_path / "study",
        master=master,
        cash=cash_path,
        adapter=adapter,
        bundle=bundle,
        calls=calls,
    )


def test_strict_unknown_preparation_has_proof_but_no_engine_call(packet_fixture):
    f = packet_fixture
    f.bundle.coverage.loc[
        f.bundle.coverage.index[-1], ["status", "signal_eligible", "fill_eligible"]
    ] = ["unknown_missing", False, False]
    master = f.master.parent / "unknown-master"
    f.adapter.write_reconciled_packet(master, f.bundle)
    result = protocol.prepare_quality_study(
        f.config, study_dir=f.study, input_packet=master, cash_file=f.cash
    )
    assert result["status"] == "data_not_ready" and result["unknown_missing_count"] == 1
    assert result["source_unit_proof_verified"] is True
    assert not f.calls and not (f.study / "prepared").exists()


def test_known_halt_allows_strict_preparation_without_closing_identity(packet_fixture):
    f = packet_fixture
    f.bundle.coverage.loc[
        f.bundle.coverage.index[-1], ["status", "signal_eligible", "fill_eligible"]
    ] = ["known_halt_overlap", False, False]
    master = f.master.parent / "halt-master"
    f.adapter.write_reconciled_packet(master, f.bundle)
    result = protocol.prepare_quality_study(
        f.config, study_dir=f.study, input_packet=master, cash_file=f.cash
    )
    assert result["status"] == "data_ready"
    assert result["known_halt_overlap_count"] == 1
    assert (f.bundle.daily.close != 100).all()  # daily 101 vs RTH 100 is NOT a failed identity
    assert not f.calls


def test_mocked_grid_then_nine_and_prerequisite_before_oos_parse(packet_fixture, monkeypatch):
    f = packet_fixture
    protocol.prepare_quality_study(
        f.config, study_dir=f.study, input_packet=f.master, cash_file=f.cash
    )
    protocol.run_quality_development(f.config, study_dir=f.study)
    protocol.run_quality_validation(f.config, study_dir=f.study)
    assert [count for _, count in f.calls] == [56644, 9]
    winner = f.study / "stages/validation/winner.json"
    winner.write_text(winner.read_text() + " ")
    monkeypatch.setattr(
        protocol,
        "_load_stage",
        lambda *args: pytest.fail("OOS prices parsed before winner integrity"),
    )
    with pytest.raises(protocol.ProtocolError, match="artifact changed"):
        protocol.run_quality_historical_oos(f.config, study_dir=f.study)


def test_changed_receipt_or_policy_cannot_reuse_existing_registration(packet_fixture, tmp_path):
    f = packet_fixture
    path = set_profile(f.config, tmp_path)
    protocol.prepare_quality_study(
        f.config, study_dir=f.study, input_packet=f.master, cash_file=f.cash
    )
    receipt = json.loads(path.read_text())
    receipt["instruction"] += " CHANGED"
    path.write_text(json.dumps(receipt))
    with pytest.raises(protocol.ProtocolError, match="changed"):
        protocol.run_quality_development(f.config, study_dir=f.study)
    assert not f.calls


def test_quality_oos_only_sealed_winner_four_costs_matched_17_endpoints(
    packet_fixture, monkeypatch
):
    f = packet_fixture
    protocol.prepare_quality_study(
        f.config, study_dir=f.study, input_packet=f.master, cash_file=f.cash
    )
    protocol.run_quality_development(f.config, study_dir=f.study)
    validation = protocol.run_quality_validation(f.config, study_dir=f.study)
    fake = protocol._engine()
    checks = []

    def result(policy):
        index = pd.DatetimeIndex(["2016-01-04T22:00Z", "2026-05-28T21:00Z"])
        return SimpleNamespace(
            metrics={"candidate_id": policy},
            daily_returns=pd.Series(0.0, index=index),
            daily_equity=pd.Series(1000.0, index=index),
            daily_cash_returns=pd.Series(0.0, index=index),
            daily_exposure=pd.Series(0.0, index=index),
            trades=pd.DataFrame(columns=["side"]),
            diagnostics=pd.DataFrame(columns=["state"]),
        )

    def simulate(d, b, candidate, **kwargs):
        assert d.session.max() <= "2026-05-28"
        assert kwargs["quality_policy"].valuation_hour_ny == 17
        assert kwargs["coverage"].session.max() <= "2026-05-28"
        checks.append((candidate.candidate_id, kwargs["slippage_bps"]))
        return result(candidate.candidate_id)

    fake.build_quality_support_cache = lambda *args, **kwargs: "SYNTHETIC_NO_PERFORMANCE_CACHE"
    fake.simulate_quality_candidate = simulate
    fake.simulate_quality_benchmarks = lambda *args, **kwargs: {
        k: result(k) for k in ("cash", "buy_and_hold")
    }
    monkeypatch.setattr(protocol, "_engine", lambda: fake)
    out = protocol.run_quality_historical_oos(f.config, study_dir=f.study)
    assert {identifier for identifier, _ in checks} == {validation["winner_id"]}
    assert [cost for _, cost in checks] == [1, 3, 10, 25]
    assert out["candidate_count"] == 1 and out["daily_valuation_hour_ny"] == 17


def test_monitor_unknown_blackout_cancels_pending_without_price_invention(monkeypatch):
    calendar = pd.DataFrame(
        {
            "session": ["2026-10-08"],
            "market_open": [pd.Timestamp("2026-10-08T13:30Z")],
            "market_close": [pd.Timestamp("2026-10-08T20:00Z")],
        }
    )
    grid = pd.DataFrame(
        {
            "session": ["2026-10-08"] * 2,
            "bar_start": pd.to_datetime(["2026-10-08T13:30Z", "2026-10-08T14:00Z"]),
            "bar_end": pd.to_datetime(["2026-10-08T14:00Z", "2026-10-08T14:30Z"]),
        }
    )
    cov = grid.copy()
    cov["status"] = ["observed", "unknown_missing"]
    cov["signal_eligible"], cov["fill_eligible"] = [True, False], [True, False]
    blackouts = pd.DataFrame(
        {
            "start": [pd.Timestamp("2026-10-08T14:00Z")],
            "end": [pd.Timestamp("2026-10-08T14:30Z")],
            "kind": ["unknown_missing"],
            "reason": ["SYNTHETIC_TEST_NO_SOURCE_QUOTE"],
        }
    )
    candidate = core.Candidate(
        core.IndicatorRule("SMA", period=10), core.IndicatorRule("EMA", period=20), 0.0, 0.0
    )
    state = {
        "schema": protocol.PAPER_STATE_SCHEMA,
        "sampling_interval_minutes": 30,
        "counter_unit": protocol.COUNTER_UNIT,
        "quality_profile": "observed_only_blackout_v1",
        "candidate_id": candidate.candidate_id,
        "position": 0,
        "entry_count": 1,
        "exit_count": 0,
        "pending_target": 1,
        "last_bar_end": "2026-10-08T14:00:00Z",
    }
    monkeypatch.setattr(
        protocol,
        "_calendar_data",
        lambda: SimpleNamespace(
            build_xnys_calendar=lambda *args: calendar,
            canonical_interval_grid=lambda *args, **kwargs: grid,
        ),
    )
    monkeypatch.setattr(
        protocol, "_core", lambda: pytest.fail("No quote exists: no indicator preview is allowed")
    )
    bars = pd.DataFrame(
        {
            "session": ["2026-10-08"],
            "bar_start": [pd.Timestamp("2026-10-08T13:30Z")],
            "bar_end": [pd.Timestamp("2026-10-08T14:00Z")],
            "close": [100.0],
        }
    )
    approval = {"explicit_user_approval": True, "quality_profile": "observed_only_blackout_v1"}
    out = protocol.monitor_quality_policy(
        pd.DataFrame(),
        bars,
        calendar,
        cov,
        blackouts,
        candidate,
        quality_policy=quality.QualityPolicy("observed_only_blackout_v1"),
        as_of="2026-10-08T14:35Z",
        state=state,
        authorization=approval,
    )
    assert not out["actionable"] and out["next_state"]["pending_target"] is None
    assert out["next_state"]["entry_count"] == 0 and out["next_state"]["position"] == 0
    assert len(bars) == 1  # no fabricated missing-bar row


@pytest.mark.parametrize("count", [float("nan"), float("inf"), 10.5, True, "10", 9])
def test_unit_anchor_policy_rejects_nan_or_noninteger_counts(count):
    proof = {
        "unit_mapping_proof_passed": True,
        "quote_relationship_reconciled_by_explicit_roles": True,
        "native_raw_capture": False,
        "independent_price_anchor_proof": {
            "minimum_required_anchor_days_per_month": count,
            "minimum_anchor_days_per_month": 10,
            "unanchored_months": [],
            "daily_close_not_used_as_an_anchor": True,
        },
    }
    with pytest.raises(protocol.ProtocolError, match="finite integers"):
        protocol._check_unit_proof(proof)


def test_undefined_family_excluded_actual_eight_sealed_and_validated(packet_fixture):
    f = packet_fixture
    fake = protocol._engine()
    original = fake.evaluate_quality_grid

    def evaluate(d, b, *, candidates, **kwargs):
        metrics = original(d, b, candidates=candidates, **kwargs)
        undefined = {c.candidate_id for c in candidates if c.family_cell == "MACD/MACD"}
        metrics.loc[metrics.candidate_id.isin(undefined), "cash_excess_sharpe"] = float("nan")
        return metrics

    fake.evaluate_quality_grid = evaluate
    protocol.prepare_quality_study(
        f.config, study_dir=f.study, input_packet=f.master, cash_file=f.cash
    )
    dev = protocol.run_quality_development(f.config, study_dir=f.study)
    val = protocol.run_quality_validation(f.config, study_dir=f.study)
    assert dev["finalist_count"] == val["candidate_count"] == 8
    assert [count for _, count in f.calls] == [56644, 8]


def test_all_undefined_families_stop_without_invented_finalist(packet_fixture):
    f = packet_fixture
    fake = protocol._engine()
    fake.evaluate_quality_grid = lambda d, b, *, candidates, **kwargs: pd.DataFrame(
        {"candidate_id": [c.candidate_id for c in candidates], "cash_excess_sharpe": float("nan")}
    )
    protocol.prepare_quality_study(
        f.config, study_dir=f.study, input_packet=f.master, cash_file=f.cash
    )
    with pytest.raises(protocol.ProtocolError, match="All development family scores are undefined"):
        protocol.run_quality_development(f.config, study_dir=f.study)
    assert not (f.study / "stages/development").exists()
    assert list((f.study / "attempts").glob("*/results/metrics.csv"))


def diagnostic_fixture(tmp_path):
    config = yaml.safe_load(ACTIVE_CONFIG.read_text())
    candidates = [
        core.Candidate(
            core.IndicatorRule("SMA", period=10), core.IndicatorRule("EMA", period=20), 0.0, 0.0
        ),
        core.Candidate(
            core.IndicatorRule("EMA", period=20), core.IndicatorRule("EMA", period=20), 0.0, 0.0
        ),
    ]
    metrics = pd.DataFrame(
        {
            "candidate_id": [c.candidate_id for c in candidates],
            "cash_excess_sharpe": [2.0, 1.0],
            "post_gap_confirmed_buy_count": [1, 0],
            "post_gap_confirmed_sell_count": [0, 0],
            "post_gap_confirmed_signal_count": [1, 0],
            "has_post_gap_confirmed_signal": [True, False],
        }
    )
    # Two overlapping windows associate the SAME confirmed decision with both
    # gaps, while the candidate count remains one distinct confirmed decision.
    metrics.attrs["post_gap_signal_events"] = pd.DataFrame(
        {
            "candidate_id": [candidates[0].candidate_id] * 2,
            "side": ["buy"] * 2,
            "decision_at": pd.to_datetime(["2004-12-01T15:00Z"] * 2),
            "gap_start": pd.to_datetime(["2004-11-30T14:30Z", "2004-11-30T15:00Z"]),
            "gap_end": pd.to_datetime(["2004-11-30T15:00Z", "2004-11-30T15:30Z"]),
            "window_end": pd.to_datetime(["2004-12-01T21:00Z"] * 2),
        }
    )
    metrics.attrs["post_gap_signal_event_metadata"] = {
        "enabled": True,
        "event_limit": 2500000,
        "total_event_rows": 2,
        "stored_event_rows": 2,
        "truncated": False,
        "window_count": 2,
        "windows": [],
    }
    output = tmp_path / "diagnostics"
    output.mkdir()
    return config, candidates, metrics, output


def test_all_candidate_flags_events_and_overlap_dedup_persist_without_reranking(tmp_path):
    config, candidates, metrics, output = diagnostic_fixture(tmp_path)
    out = protocol._persist_post_gap_diagnostics(output, metrics, candidates, config, "development")
    assert out["event_rows"] == 2 and out["candidate_flag_rows"] == 2
    flags = pd.read_csv(output / "post_gap_candidate_flags.csv")
    assert len(flags) == len(candidates) and flags.post_gap_confirmed_signal_count.sum() == 1
    winner = protocol._finite_winner(metrics, {c.candidate_id: c for c in candidates})
    assert winner == candidates[0]  # Highest Sharpe remains selected even when flagged.
    saved = json.loads((output / "post_gap_diagnostic_metadata.json").read_text())
    assert saved["used_for_filtering_ranking_or_selection"] is False
    assert saved["gap_prices_repaired"] is False
    selection = protocol._selection_view(metrics)
    assert not selection.attrs
    assert len(selection) == len(metrics)
    assert selection.candidate_id.equals(metrics.candidate_id)
    assert metrics.attrs["post_gap_signal_events"].shape[0] == 2
    assert (
        protocol._finite_winner(selection, {c.candidate_id: c for c in candidates}) == candidates[0]
    )


def test_truncated_diagnostic_table_refuses_seal_and_retains_partial_artifacts(tmp_path):
    config, candidates, metrics, output = diagnostic_fixture(tmp_path)
    metrics.attrs["post_gap_signal_event_metadata"]["truncated"] = True
    with pytest.raises(protocol.ProtocolError, match="truncated.*refusing to seal"):
        protocol._persist_post_gap_diagnostics(output, metrics, candidates, config, "development")
    assert (output / "post_gap_signal_events.csv").is_file()
    assert (output / "post_gap_candidate_flags.csv").is_file()
    assert (output / "post_gap_diagnostic_metadata.json").is_file()


def test_missing_all_candidate_flags_or_false_selection_usage_rejected(tmp_path):
    config, candidates, metrics, output = diagnostic_fixture(tmp_path)
    bad = metrics.iloc[:1].copy()
    with pytest.raises(protocol.ProtocolError, match="Every permitted candidate"):
        protocol._persist_post_gap_diagnostics(output, bad, candidates, config, "development")
    config["diagnostics"]["affects_selection"] = True
    with pytest.raises(protocol.ProtocolError, match="cannot filter or rank"):
        protocol._diagnostics_settings(config)


def test_current_gap_config_routes_private_dir_without_changing_strict_test_fixture(config_file):
    _, output = protocol._context(ACTIVE_CONFIG, None)
    assert output.name == "qqq_daily_halfhour_v5_alpha_gap_flags_v2"
    assert protocol.load_config(config_file)["quality_policy"]["profile"] == "strict_tradable_v1"
    assert protocol._diagnostics_settings(protocol.load_config(config_file))["enabled"] is False


def test_current_gap_config_paths_and_frozen_research_contract():
    config = protocol.load_config(ACTIVE_CONFIG)
    assert config["paths"]["private_output"] == ".cache/qqq_daily_halfhour_v5_alpha_gap_flags_v2"
    assert (
        config["paths"]["input_packet"]
        == "data/processed/research_daily_halfhour_v5_alpha_gap_flags_v2/classified_alpha_input"
    )
    assert (
        config["quality_policy"]["gap_authorization_receipt"]
        == ".cache/qqq_daily_halfhour_v5_alpha_gap_flags_v2/gap_policy_authorization.json"
    )
    assert config["quality_policy"]["profile"] == "observed_only_blackout_v1"
    assert config["grid"]["candidate_count"] == 56644
    assert config["grid"]["entry_confirmation_hours"] == protocol.CONFIRMATION_HOURS
    assert config["grid"]["exit_confirmation_hours"] == protocol.CONFIRMATION_HOURS
    diagnostics = protocol._diagnostics_settings(config)
    assert diagnostics["enabled"] is True
    assert diagnostics["descriptive_only"] is True
    assert diagnostics["affects_selection"] is False


def test_gap_flags_cli_uses_v2_config_and_private_dir_by_default(monkeypatch):
    import importlib.util

    script = protocol.WORKSPACE / "scripts/run_qqq_gap_flags_v5.py"
    monkeypatch.syspath_prepend(str(script.parent))
    spec = importlib.util.spec_from_file_location("gap_flags_cli_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    monkeypatch.setattr(module, "quality_main", lambda args: calls.append(args) or 0)
    assert module.main(["development"]) == 0  # Captured argv only; NEVER an evaluation.
    args = calls[0]
    assert Path(args[args.index("--config") + 1]).name == "qqq_daily_halfhour_v5_gap_flags_v2.yaml"
    assert (
        Path(args[args.index("--study-dir") + 1]).name == "qqq_daily_halfhour_v5_alpha_gap_flags_v2"
    )
