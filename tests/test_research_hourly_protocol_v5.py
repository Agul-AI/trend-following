"""Governance/monitor tests use fabricated data and mocked P&L ONLY."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from trend_following import research_daily_hourly_v5 as engine
from trend_following import research_hourly_data_v5 as data
from trend_following import research_hourly_protocol_v5 as protocol


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(protocol.DEFAULT_CONFIG.read_text())
    return path


@pytest.fixture
def fixed_sources(monkeypatch):
    # Other agents may edit their implementation concurrently. These fixtures
    # exercise the hash protocol, not hash a moving local implementation.
    monkeypatch.setattr(protocol, "_source_hashes", lambda: {"fabricated_code": "a" * 64})
    monkeypatch.setattr(protocol, "_runtime", lambda: {"python": "fixture", "packages": {}})


def test_grid_and_registered_semantics(config_file):
    config = protocol.load_config(config_file)
    candidates, table = protocol._candidate_table(config)
    assert len(candidates) == len(table) == 17 * 17 * 14 * 14 == 56644
    assert {c.entry_rule.family for c in candidates} == {"SMA", "EMA", "MACD"}
    assert set(config["grid"]["ma_periods"]) == {10, 20, 50, 100, 200, 250}
    assert config["study_id"] == "qqq_daily_halfhour_v5_alpha_only"
    assert config["segments"]["historical-oos"]["end"] == "2026-05-28"
    assert protocol.DEFAULT_STUDY_DIR.name == "qqq_daily_halfhour_v5_alpha_only"
    assert config["grid"]["entry_confirmation_hours"] == [n / 2 for n in range(14)]
    assert {c.entry_required_checks for c in candidates} == set(range(1, 15))
    assert {c.exit_required_checks for c in candidates} == set(range(1, 15))


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("grid", "ma_periods", [10, 20, 50, 100, 200, 250, 300]),
        ("signals", "daily_state_update", "every_hour"),
        ("accounting", "baseline_slippage_bps", 1.0),
        ("selection", "tie_break", "best_cagr"),
        ("data_contract", "price_provider", "Yahoo Finance"),
        ("data_contract", "no_other_price_provider", False),
        ("data_contract", "sampling_interval_minutes", 60),
        ("segments", "historical-oos", {"start": "2016-01-01", "end": "2026-06-01"}),
    ],
)
def test_changed_frozen_semantics_rejected(config_file, section, key, value):
    config = yaml.safe_load(config_file.read_text())
    config[section][key] = value
    config_file.write_text(yaml.safe_dump(config))
    with pytest.raises(protocol.ProtocolError, match="Frozen"):
        protocol.load_config(config_file)


def test_missing_data_registration_no_performance_and_failures_retained(
    config_file, tmp_path, fixed_sources, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("No real/fabricated performance may run at data-not-ready gate")

    monkeypatch.setattr(engine, "evaluate_grid", forbidden)
    study = tmp_path / "study"
    receipt = protocol.prepare_study(
        config_file,
        study_dir=study,
        input_packet=tmp_path / "absent",
        cash_file=tmp_path / "missing.csv",
    )
    assert receipt["status"] == "data_not_ready"
    assert receipt["registered_candidates"] == 56644
    assert receipt["performance_run"] is False
    assert receipt["price_provider"] == "Alpha Vantage"
    assert receipt["historical_cutoff"] == "2026-05-28"
    assert protocol._read_seal(study / "registration.json")["candidate_count"] == 56644
    with pytest.raises(protocol.ProtocolError, match="prior sealed artifact"):
        protocol.run_development(config_file, study_dir=study)
    attempts = [json.loads(line) for line in (study / "attempts.jsonl").read_text().splitlines()]
    assert [row["status"] for row in attempts] == ["data_not_ready", "failed"]
    assert len(list((study / "attempts").glob("*/failure.json"))) == 1


def test_hash_change_blocks_before_engine(config_file, tmp_path, fixed_sources, monkeypatch):
    study = tmp_path / "study"
    protocol.prepare_study(
        config_file,
        study_dir=study,
        input_packet=tmp_path / "absent",
        cash_file=tmp_path / "missing.csv",
    )
    monkeypatch.setattr(protocol, "_source_hashes", lambda: {"fabricated_code": "b" * 64})
    monkeypatch.setattr(
        protocol,
        "_load_stage",
        lambda *args: pytest.fail("Changed code must not read price frames"),
    )
    with pytest.raises(protocol.ProtocolError, match="changed"):
        protocol.run_development(config_file, study_dir=study)


def test_symlink_config_not_hidden_by_resolve(config_file, tmp_path):
    alias = tmp_path / "config-alias.yaml"
    alias.symlink_to(config_file)
    with pytest.raises(protocol.ProtocolError, match="symlink"):
        protocol.prepare_study(alias, study_dir=tmp_path / "study")


def test_selection_is_finite_primary_score_then_id_only():
    candidates = engine.generate_candidates()
    metrics = pd.DataFrame(
        {
            "candidate_id": [c.candidate_id for c in candidates],
            "cash_excess_sharpe": np.ones(len(candidates)),
            "cagr": np.arange(len(candidates))[::-1],
        }
    )
    # A near-best within the fixed tolerance must still lose to a smaller ID,
    # irrespective of enormous secondary CAGR values.
    finalists = protocol.select_family_finalists(metrics, candidates)
    for finalist in finalists:
        cell_ids = [c.candidate_id for c in candidates if c.family_cell == finalist.family_cell]
        assert finalist.candidate_id == min(cell_ids)
    assert len({c.family_cell for c in finalists}) == 9
    subset = metrics.loc[metrics.candidate_id.isin([c.candidate_id for c in finalists])].copy()
    lexical = min(subset.candidate_id)
    subset.loc[subset.candidate_id.ne(lexical), "cash_excess_sharpe"] += 0.5e-10
    winner = protocol._finite_winner(subset, {c.candidate_id: c for c in finalists})
    assert winner.candidate_id == lexical
    metrics.loc[
        metrics.candidate_id.str.contains("entry_macd")
        & metrics.candidate_id.str.contains("exit_macd"),
        "cash_excess_sharpe",
    ] = np.nan
    with pytest.raises(protocol.ProtocolError, match="MACD/MACD"):
        protocol.select_family_finalists(metrics, candidates)


def test_cash_duplicate_availability_preserved_and_dated_tie_order(tmp_path):
    path = tmp_path / "cash.csv"
    frame = pd.DataFrame(
        {
            "available_at": ["2001-01-03T00:00:00Z"] * 2,
            "annual_rate": [0.03, 0.04],
            "observation_date": ["2001-01-01", "2001-01-02"],
            "series_id": ["DTB3"] * 2,
        }
    )
    frame.to_csv(path, index=False)
    assert len(protocol._read_cash(path)) == 2
    frame.iloc[::-1].to_csv(path, index=False)
    with pytest.raises(protocol.ProtocolError, match="tie ordering"):
        protocol._read_cash(path)


class FabricatedPacketAdapter:
    """Tiny CSV packets, intentionally mocked calendar coverage for stage tests."""

    SourceMetadata = data.SourceMetadata

    def prepare_input_packet(
        self,
        root,
        daily,
        bars,
        calendar,
        metadata,
        *,
        as_of,
        cash_observations=None,
        cash_source_file=None,
        bar_start_session=None,
        **kwargs,
    ):
        root = Path(root)
        root.mkdir(parents=True)
        frames = {"daily.csv": daily, "bars.csv": bars, "calendar.csv": calendar}
        if cash_observations is not None:
            frames["DTB3_available.csv"] = cash_observations
        for name, frame in frames.items():
            frame.to_csv(root / name, index=False)
        paths = [*metadata.source_files, *([str(cash_source_file)] if cash_source_file else [])]
        manifest = {
            "schema": "fabricated_packet_only",
            "source_metadata": asdict(metadata),
            "daily_source_metadata": kwargs.get(
                "daily_source_metadata", daily.attrs.get("source_metadata")
            ),
            "as_of": as_of,
            "audit": {
                "bar_start_session": bar_start_session,
                "sampling_interval_minutes": kwargs["sampling_interval_minutes"],
                "observation_interval_minutes": 30,
            },
            "source_files": [{"path": str(p), "sha256": protocol.sha256_file(p)} for p in paths],
            "files": {name: {"sha256": protocol.sha256_file(root / name)} for name in frames},
        }
        manifest = json.loads(json.dumps(manifest))
        (root / "manifest.json").write_text(json.dumps(manifest))
        return manifest

    def read_input_packet(self, root):
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        for name, info in manifest["files"].items():
            assert protocol.sha256_file(root / name) == info["sha256"]
        daily = pd.read_csv(root / "daily.csv")
        daily.attrs["source_metadata"] = manifest["daily_source_metadata"]
        bars = pd.read_csv(root / "bars.csv")
        calendar = pd.read_csv(root / "calendar.csv")
        for frame, names in (
            (bars, ("bar_start", "bar_end")),
            (calendar, ("market_open", "market_close")),
        ):
            for name in names:
                frame[name] = pd.to_datetime(frame[name], utc=True)
        return daily, bars, calendar, manifest


@pytest.fixture
def prepared_fixture(config_file, tmp_path, fixed_sources, monkeypatch):
    raw = tmp_path / "fabricated-raw.txt"
    raw.write_text("FABRICATED TEST DATA, NOT MARKET DATA")
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
            "close": [100.0] * len(sessions),
            "dividend_amount": 0.0,
            "split_coefficient": 1.0,
        }
    )
    daily.attrs["source_metadata"] = {
        "provider": "Alpha Vantage",
        "symbol": "QQQ",
        "price_basis": "raw_as_traded",
        "verification": "TEST_FIXTURE_SYNTHETIC_NOT_MARKET_DATA",
    }
    calendar = pd.DataFrame(
        {
            "session": sessions,
            "market_open": pd.to_datetime([f"{d}T14:30:00Z" for d in sessions]),
            "market_close": pd.to_datetime([f"{d}T21:00:00Z" for d in sessions]),
            "is_early_close": False,
        }
    )
    bars = pd.DataFrame(
        {
            "session": sessions[2:],
            "bar_start": pd.to_datetime([f"{d}T14:30:00Z" for d in sessions[2:]]),
            "bar_end": pd.to_datetime([f"{d}T15:00:00Z" for d in sessions[2:]]),
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 1.0,
            "is_full_hour": False,
            "duration_minutes": 30.0,
        }
    )
    cash_path = tmp_path / "cash.csv"
    pd.DataFrame(
        {
            "available_at": [f"{d}T22:00:00Z" for d in sessions],
            "observation_date": sessions,
            "annual_rate": 0.03,
            "series_id": "DTB3",
        }
    ).to_csv(cash_path, index=False)
    adapter = FabricatedPacketAdapter()
    metadata = data.SourceMetadata(
        "Alpha Vantage",
        "QQQ",
        30,
        "start",
        "UTC",
        "raw_as_traded",
        False,
        "2026-05-29T00:00:00Z",
        "TEST_FIXTURE_SYNTHETIC_NOT_MARKET_DATA",
        (str(raw),),
    )
    master = tmp_path / "master"
    adapter.prepare_input_packet(
        master,
        daily,
        bars,
        calendar,
        metadata,
        as_of="2026-05-29T00:00:00Z",
        bar_start_session="2001-01-02",
        sampling_interval_minutes=30,
    )
    monkeypatch.setattr(protocol, "_data", lambda: adapter)

    def prefix_assert(d, b, c, rates, stage, config):
        end = protocol.SPLITS[stage]["end"]
        for frame in (d, b, c):
            assert frame.session.le(end).all()
        assert d.session.iloc[0] == "1999-11-01"
        assert rates.available_at.le(protocol._session_end_utc(end)).all()

    monkeypatch.setattr(protocol, "_check_prefix", prefix_assert)
    calls = []

    def evaluate(d, b, *, candidates, **kwargs):
        assert d.session.max() <= kwargs["end"]
        assert b.session.max() <= kwargs["end"]
        assert kwargs["slippage_bps"] == 3
        assert kwargs["commission_bps"] == 1
        calls.append(
            ("evaluate", kwargs["start"], kwargs["end"], tuple(c.candidate_id for c in candidates))
        )
        return pd.DataFrame(
            {"candidate_id": [c.candidate_id for c in candidates], "cash_excess_sharpe": 1.0}
        )

    def result(d, candidate_id, start, end):
        index = pd.Index(d.loc[d.session.between(start, end), "session"], name="session")
        return SimpleNamespace(
            metrics={"candidate_id": candidate_id, "cash_excess_sharpe": 1.0},
            daily_returns=pd.Series(0.0, index=index),
            daily_equity=pd.Series(1000.0, index=index),
            daily_cash_returns=pd.Series(0.0, index=index),
            daily_exposure=pd.Series(0.0, index=index),
            trades=pd.DataFrame(columns=["side"]),
            diagnostics=pd.DataFrame(columns=["state"]),
        )

    def simulate(d, b, candidate, **kwargs):
        assert d.session.max() <= kwargs["end"]
        calls.append(("winner", candidate.candidate_id, kwargs["slippage_bps"]))
        return result(d, candidate.candidate_id, kwargs["start"], kwargs["end"])

    def benchmarks(d, b, **kwargs):
        calls.append(("benchmarks", kwargs["slippage_bps"]))
        return {
            name: result(d, name, kwargs["start"], kwargs["end"])
            for name in ("cash", "buy_and_hold")
        }

    fake_engine = SimpleNamespace(
        generate_candidates=engine.generate_candidates,
        evaluate_grid=evaluate,
        build_support_cache=lambda *args: "FABRICATED_NO_FEATURE_COMPUTATION",
        simulate_candidate=simulate,
        simulate_benchmarks=benchmarks,
    )
    monkeypatch.setattr(protocol, "_engine", lambda: fake_engine)
    study = tmp_path / "study"
    receipt = protocol.prepare_study(
        config_file, study_dir=study, input_packet=master, cash_file=cash_path
    )
    assert receipt["status"] == "data_ready"
    assert not calls
    return SimpleNamespace(
        study=study,
        config=config_file,
        calls=calls,
        engine=fake_engine,
        master=master,
        cash_path=cash_path,
        adapter=adapter,
    )


def test_staged_pipeline_only_full_grid_then_nine_then_winner_costs(prepared_fixture):
    fixture = prepared_fixture
    development = protocol.run_development(fixture.config, study_dir=fixture.study)
    validation = protocol.run_validation(fixture.config, study_dir=fixture.study)
    historical = protocol.run_historical_oos(fixture.config, study_dir=fixture.study)
    assert development["candidate_count"] == 56644
    assert validation["candidate_count"] == 9
    assert historical["candidate_count"] == 1
    searches = [call for call in fixture.calls if call[0] == "evaluate"]
    assert [len(call[3]) for call in searches] == [56644, 9]
    winner_runs = [call for call in fixture.calls if call[0] == "winner"]
    assert {call[1] for call in winner_runs} == {validation["winner_id"]}
    assert [call[2] for call in winner_runs] == [1, 3, 10, 25]
    assert len([call for call in fixture.calls if call[0] == "benchmarks"]) == 4
    assert historical["historical_label"] == protocol.HISTORICAL_LABEL
    for stage in protocol.HISTORICAL_STAGES:
        daily = pd.read_csv(fixture.study / "prepared" / stage / "daily.csv")
        assert daily.session.max() <= protocol.SPLITS[stage]["end"]
        assert daily.session.min() == "1999-11-01"


def test_validation_does_not_parse_validation_before_finalist_gate(prepared_fixture, monkeypatch):
    fixture = prepared_fixture
    monkeypatch.setattr(
        protocol,
        "_load_stage",
        lambda *args: pytest.fail("Validation frames released before finalist seal"),
    )
    with pytest.raises(protocol.ProtocolError, match="prior sealed artifact"):
        protocol.run_validation(fixture.config, study_dir=fixture.study)
    assert not fixture.calls


def test_historical_does_not_parse_oos_before_winner_gate(prepared_fixture, monkeypatch):
    fixture = prepared_fixture
    protocol.run_development(fixture.config, study_dir=fixture.study)
    monkeypatch.setattr(
        protocol, "_load_stage", lambda *args: pytest.fail("OOS frames released before winner seal")
    )
    with pytest.raises(protocol.ProtocolError, match="prior sealed artifact"):
        protocol.run_historical_oos(fixture.config, study_dir=fixture.study)
    assert len(fixture.calls) == 1


def test_tampered_finalist_rejected_before_validation_read(prepared_fixture, monkeypatch):
    fixture = prepared_fixture
    protocol.run_development(fixture.config, study_dir=fixture.study)
    path = fixture.study / "stages/development/finalists.json"
    path.write_text(path.read_text() + " ")
    monkeypatch.setattr(
        protocol,
        "_load_stage",
        lambda *args: pytest.fail("Tampered finalist must precede validation data release"),
    )
    with pytest.raises(protocol.ProtocolError, match="artifact changed"):
        protocol.run_validation(fixture.config, study_dir=fixture.study)


def test_tampered_winner_rejected_before_oos_read(prepared_fixture, monkeypatch):
    fixture = prepared_fixture
    protocol.run_development(fixture.config, study_dir=fixture.study)
    protocol.run_validation(fixture.config, study_dir=fixture.study)
    path = fixture.study / "stages/validation/winner.json"
    path.write_text(path.read_text() + " ")
    monkeypatch.setattr(
        protocol,
        "_load_stage",
        lambda *args: pytest.fail("Tampered winner must precede OOS data release"),
    )
    with pytest.raises(protocol.ProtocolError, match="artifact changed"):
        protocol.run_historical_oos(fixture.config, study_dir=fixture.study)


def test_prefix_change_and_failed_metric_attempts_preserved(prepared_fixture):
    fixture = prepared_fixture
    fixture.engine.evaluate_grid = lambda d, b, *, candidates, **kwargs: pd.DataFrame(
        {
            "candidate_id": [c.candidate_id for c in candidates],
            "cash_excess_sharpe": np.nan,
        }
    )
    with pytest.raises(protocol.ProtocolError, match="No valid finite finalist"):
        protocol.run_development(fixture.config, study_dir=fixture.study)
    assert list((fixture.study / "attempts").glob("*/results/metrics.csv"))
    assert list((fixture.study / "attempts").glob("*/failure.json"))
    path = fixture.study / "prepared/development/bars.csv"
    path.write_text(path.read_text() + " ")
    with pytest.raises(protocol.ProtocolError, match="prefix data changed"):
        protocol.run_development(fixture.config, study_dir=fixture.study)
    assert len(list((fixture.study / "attempts").glob("*/failure.json"))) == 2


def test_completed_stage_no_reperformance(prepared_fixture):
    fixture = prepared_fixture
    first = protocol.run_development(fixture.config, study_dir=fixture.study)
    second = protocol.run_development(fixture.config, study_dir=fixture.study)
    assert first == second
    assert len(fixture.calls) == 1


def test_yahoo_master_rejected_as_source_before_numeric_read_or_performance(prepared_fixture):
    fixture = prepared_fixture
    manifest_path = fixture.master / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_metadata"]["provider"] = "Yahoo Finance"
    manifest_path.write_text(json.dumps(manifest))
    fixture.adapter.read_input_packet = lambda *args: pytest.fail(
        "Yahoo receipt must be rejected before price parsing"
    )
    fixture.engine.evaluate_grid = lambda *args, **kwargs: pytest.fail(
        "Wrong source must never reach numerical performance"
    )
    with pytest.raises(protocol.ProtocolError, match="exactly Alpha Vantage"):
        protocol.prepare_study(
            fixture.config,
            study_dir=fixture.study.parent / "rejected-yahoo",
            input_packet=fixture.master,
            cash_file=fixture.cash_path,
        )
    assert not fixture.calls


def test_missing_daily_attestation_cannot_default_to_alpha(prepared_fixture):
    fixture = prepared_fixture
    manifest_path = fixture.master / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("daily_source_metadata")
    manifest_path.write_text(json.dumps(manifest))
    fixture.adapter.read_input_packet = lambda *args: pytest.fail(
        "Unknown daily provider must be rejected before price parsing"
    )
    with pytest.raises(protocol.ProtocolError, match="independently attested"):
        protocol.prepare_study(
            fixture.config,
            study_dir=fixture.study.parent / "rejected-unknown-daily",
            input_packet=fixture.master,
            cash_file=fixture.cash_path,
        )
    assert not fixture.calls


def test_yahoo_stage_metadata_rejected_before_grid_performance(prepared_fixture):
    fixture = prepared_fixture
    manifest_path = fixture.study / "prepared/development/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_metadata"]["provider"] = "Yahoo Finance"
    manifest_path.write_text(json.dumps(manifest))
    fixture.engine.evaluate_grid = lambda *args, **kwargs: pytest.fail(
        "Yahoo stage must be rejected before engine evaluation"
    )
    with pytest.raises(protocol.ProtocolError, match="exactly Alpha Vantage"):
        protocol.run_development(fixture.config, study_dir=fixture.study)
    assert not fixture.calls


def test_daily_provider_constraint_does_not_exclude_fred_cash():
    manifest = {
        "source_metadata": {"provider": "Alpha Vantage", "symbol": "QQQ"},
        "daily_source_metadata": {
            "provider": "Alpha Vantage",
            "symbol": "QQQ",
            "price_basis": "raw_as_traded",
            "verification": "TEST_FIXTURE_SYNTHETIC_NOT_MARKET_DATA",
        },
        "cash_source_metadata": {"provider": "FRED", "series_id": "DTB3"},
    }
    protocol._require_alpha_price_source(manifest)
    manifest["daily_source_metadata"] = {"provider": "Yahoo Finance", "symbol": "QQQ"}
    with pytest.raises(protocol.ProtocolError, match="Daily price source"):
        protocol._require_alpha_price_source(manifest)


def test_master_prices_beyond_common_alpha_cutoff_rejected(prepared_fixture):
    fixture = prepared_fixture
    path = fixture.master / "daily.csv"
    frame = pd.read_csv(path)
    frame.loc[frame.session.eq("2026-05-28"), "session"] = "2026-06-01"
    frame.to_csv(path, index=False)
    manifest_path = fixture.master / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["daily.csv"]["sha256"] = protocol.sha256_file(path)
    manifest_path.write_text(json.dumps(manifest))
    fixture.engine.evaluate_grid = lambda *args, **kwargs: pytest.fail(
        "A beyond-cutoff packet must not run performance"
    )
    with pytest.raises(protocol.ProtocolError, match="must not extend beyond.*2026-05-28"):
        protocol.prepare_study(
            fixture.config,
            study_dir=fixture.study.parent / "rejected-extension",
            input_packet=fixture.master,
            cash_file=fixture.cash_path,
        )
    assert not fixture.calls


def test_old_registration_is_rejected_not_overwritten(config_file, tmp_path, fixed_sources):
    config = protocol.load_config(config_file)
    legacy = protocol._registration_payload(config, config_file)
    legacy["study_id"] = "qqq_daily_hourly_v5"
    legacy["configuration"]["study_id"] = "qqq_daily_hourly_v5"
    old = tmp_path / "legacy-study"
    protocol._seal(old / "registration.json", legacy)
    before = protocol.sha256_file(old / "registration.json")
    with pytest.raises(protocol.ProtocolError, match="new preregistration"):
        protocol.prepare_study(
            config_file,
            study_dir=old,
            input_packet=tmp_path / "absent",
            cash_file=tmp_path / "missing.csv",
        )
    assert protocol.sha256_file(old / "registration.json") == before


@pytest.fixture
def live_fixture(monkeypatch):
    calendar = data.build_xnys_calendar("2026-10-07", "2026-10-08")
    grid = data.canonical_interval_grid(calendar, interval_minutes=30)
    history_dates = list(pd.bdate_range("1999-11-01", periods=300).strftime("%Y-%m-%d")) + [
        "2026-10-07"
    ]
    daily = pd.DataFrame(
        {"session": history_dates, "close": 100.0, "dividend_amount": 0.0, "split_coefficient": 1.0}
    )
    daily.attrs["source_metadata"] = {
        "provider": "Alpha Vantage",
        "symbol": "QQQ",
        "price_basis": "raw_as_traded",
        "verification": "TEST_FIXTURE_SYNTHETIC_NOT_MARKET_DATA",
    }
    candidate = engine.Candidate(
        engine.IndicatorRule("SMA", period=10), engine.IndicatorRule("EMA", period=20), 0.5, 0.5
    )
    metadata = data.SourceMetadata(
        "Alpha Vantage",
        "QQQ",
        30,
        "start",
        "UTC",
        "raw_as_traded",
        False,
        "2026-10-08T14:00:00Z",
        "TEST_FIXTURE_SYNTHETIC_NOT_MARKET_DATA",
        (),
        {
            "session": "2026-10-08",
            "dividend_amount": 0.0,
            "split_coefficient": 1.0,
            "available_at": "2026-10-08T13:30:00Z",
            "verification": "fabricated",
        },
    )

    def validate(d, bars, c, m, *, as_of, **kwargs):
        if bars.bar_end.gt(as_of).any():
            raise data.DataContractError("unfinished bar")

    monkeypatch.setattr(
        protocol,
        "_data",
        lambda: SimpleNamespace(
            validate_monitor_inputs=validate,
            canonical_interval_grid=lambda c, *, interval_minutes: grid,
        ),
    )

    def preview(history, close, *, rules, **kwargs):
        values = pd.Series({rule.rule_id: close - 100.0 for rule in rules})
        values.attrs["prior_session_count"] = len(history)
        return values

    monkeypatch.setattr(
        protocol,
        "_engine",
        lambda: SimpleNamespace(
            preview_daily_margins=preview,
            WARMUP_SESSIONS=295,
        ),
    )

    def bars_until(now, price=101.0):
        bars = grid.loc[grid.bar_end.le(pd.Timestamp(now))].copy()
        for name in ("open", "high", "low", "close"):
            bars[name] = price
        bars["volume"] = 1.0
        return bars.reset_index(drop=True)

    return SimpleNamespace(
        daily=daily,
        calendar=calendar,
        candidate=candidate,
        metadata=metadata,
        bars_until=bars_until,
        previous_close="2026-10-07T20:00:00Z",
    )


def monitor(fixture, now, *, state=None, bars=None):
    if state is not None:
        state = {
            "schema": protocol.PAPER_STATE_SCHEMA,
            "sampling_interval_minutes": 30,
            "counter_unit": protocol.COUNTER_UNIT,
            "candidate_id": fixture.candidate.candidate_id,
            **state,
        }
    return protocol.monitor_live_policy(
        fixture.daily,
        fixture.bars_until(now) if bars is None else bars,
        fixture.calendar,
        fixture.candidate,
        fixture.metadata,
        as_of=now,
        state=state,
        bar_start_session="2026-10-07",
    )


def test_live_condition_only_no_assumed_holding(live_fixture):
    result = monitor(live_fixture, "2026-10-08T14:35:00Z")
    assert result["status"] == "needs_state"
    assert result["entry_support"] and result["exit_support"]
    assert not result["actionable"] and result["action"] is None


def test_live_cross_session_carry_and_running_next_hour_allowed(live_fixture):
    state = {
        "position": 0,
        "entry_count": 1,
        "exit_count": 0,
        "pending_target": None,
        "last_bar_end": live_fixture.previous_close,
    }
    before = dict(state)
    result = monitor(live_fixture, "2026-10-08T14:05:00Z", state=state)
    assert result["action"] == "buy" and result["actionable"]
    assert result["next_state"]["pending_target"] == 1
    assert result["next_state"]["entry_count"] == 2
    assert state == before  # read-only paper input, no file/order writes


@pytest.mark.parametrize("hours,checks", [(0.0, 1), (0.5, 2), (1.0, 3), (6.5, 14)])
def test_explicit_duration_waits_after_first_hit_and_serialization(hours, checks):
    candidate = engine.Candidate(
        engine.IndicatorRule("SMA", period=10),
        engine.IndicatorRule("EMA", period=20),
        np.float64(hours),
        6.5,
    )
    assert candidate.entry_required_checks == checks
    assert candidate.exit_required_checks == 14
    assert candidate.candidate_id.startswith("v5_30m__")
    serialized = json.loads(json.dumps(candidate.to_dict()))
    assert engine.Candidate.from_dict(serialized) == candidate
    assert protocol.canonical_hash(candidate.to_dict()) == protocol.canonical_hash(serialized)
    legacy = {
        "entry_rule": candidate.entry_rule.to_dict(),
        "exit_rule": candidate.exit_rule.to_dict(),
        "entry_confirmation": 2,
        "exit_confirmation": 6,
    }
    with pytest.raises(ValueError):
        engine.Candidate.from_dict(legacy)


@pytest.mark.parametrize(
    "hours,when,checks", [(0.0, "14:05", 1), (0.5, "14:35", 2), (1.0, "15:05", 3)]
)
def test_first_hit_wait_decisions_at_10_10_30_11(live_fixture, hours, when, checks):
    live_fixture.candidate = engine.Candidate(
        live_fixture.candidate.entry_rule, live_fixture.candidate.exit_rule, hours, 6.5
    )
    state = {
        "position": 0,
        "entry_count": 0,
        "exit_count": 0,
        "pending_target": None,
        "last_bar_end": live_fixture.previous_close,
    }
    as_of = f"2026-10-08T{when}:00Z"
    result = monitor(live_fixture, as_of, state=state)
    assert result["action"] == "buy" and result["actionable"]
    assert result["next_state"]["position"] == 0  # queued after close, not a prior open fill
    assert result["next_state"]["entry_count"] == checks
    assert result["decision_at"] == (pd.Timestamp(as_of) - pd.Timedelta(minutes=5)).isoformat()
    before = monitor(live_fixture, "2026-10-08T13:45:00Z", state=state)
    assert not before["actionable"] and before["action"] is None


def test_six_and_half_hours_needs_fourteen_checks_across_sessions(live_fixture, monkeypatch):
    live_fixture.candidate = engine.Candidate(
        live_fixture.candidate.entry_rule, live_fixture.candidate.exit_rule, 6.5, 0.0
    )
    state = {
        "position": 0,
        "entry_count": 0,
        "exit_count": 0,
        "pending_target": None,
        "last_bar_end": live_fixture.previous_close,
    }
    first_day = monitor(live_fixture, "2026-10-08T20:01:00Z", state=state)
    assert first_day["next_state"]["entry_count"] == 13
    assert first_day["next_state"]["pending_target"] is None
    assert not first_day["actionable"]
    calendar = data.build_xnys_calendar("2026-10-07", "2026-10-09")
    grid = data.canonical_interval_grid(calendar, interval_minutes=30)
    bars = grid.loc[grid.bar_end.le(pd.Timestamp("2026-10-09T14:05:00Z"))].copy()
    for name in ("open", "high", "low", "close"):
        bars[name] = 101.0
    bars["volume"] = 1.0
    daily = pd.concat(
        [
            live_fixture.daily,
            pd.DataFrame(
                {
                    "session": ["2026-10-08"],
                    "close": [100.0],
                    "dividend_amount": [0.0],
                    "split_coefficient": [1.0],
                }
            ),
        ],
        ignore_index=True,
    )
    daily.attrs["source_metadata"] = live_fixture.daily.attrs["source_metadata"]
    metadata = replace(
        live_fixture.metadata,
        observed_at="2026-10-09T14:00:00Z",
        current_session_actions={
            **live_fixture.metadata.current_session_actions,
            "session": "2026-10-09",
            "available_at": "2026-10-09T13:30:00Z",
        },
    )
    monkeypatch.setattr(
        protocol,
        "_data",
        lambda: SimpleNamespace(
            validate_monitor_inputs=lambda *args, **kwargs: None,
            canonical_interval_grid=lambda c, *, interval_minutes: grid,
        ),
    )
    next_day = protocol.monitor_live_policy(
        daily,
        bars,
        calendar,
        live_fixture.candidate,
        metadata,
        as_of="2026-10-09T14:05:00Z",
        state=first_day["next_state"],
    )
    assert next_day["action"] == "buy"
    assert next_day["next_state"]["entry_count"] == 14
    assert next_day["decision_at"] == "2026-10-09T14:00:00+00:00"


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "qqq_hourly_v5_paper_state_v1"),
        ("counter_unit", "hours"),
        ("sampling_interval_minutes", 60),
    ],
)
def test_legacy_paper_state_schema_or_counter_units_rejected(live_fixture, field, value):
    state = {
        "schema": protocol.PAPER_STATE_SCHEMA,
        "sampling_interval_minutes": 30,
        "counter_unit": protocol.COUNTER_UNIT,
        "candidate_id": live_fixture.candidate.candidate_id,
        "position": 0,
        "entry_count": 0,
        "exit_count": 0,
        "pending_target": None,
        "last_bar_end": live_fixture.previous_close,
        field: value,
    }
    result = monitor(live_fixture, "2026-10-08T14:05:00Z", state=state)
    assert not result["actionable"] and "Legacy/ambiguous" in result["detail"]


def test_legacy_60_minute_packet_rejected_before_price_reader(prepared_fixture):
    fixture = prepared_fixture
    manifest_path = fixture.master / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["audit"]["sampling_interval_minutes"] = 60
    manifest_path.write_text(json.dumps(manifest))
    fixture.adapter.read_input_packet = lambda *args: pytest.fail(
        "Legacy clock must be rejected before numerical parsing"
    )
    with pytest.raises(protocol.ProtocolError, match="legacy 60-minute packets"):
        protocol.prepare_study(
            fixture.config,
            study_dir=fixture.study.parent / "reject-hourly",
            input_packet=fixture.master,
            cash_file=fixture.cash_path,
        )
    assert not fixture.calls


def test_live_prior_pending_fills_next_open_then_resets(live_fixture):
    state = {
        "position": 0,
        "entry_count": 1,
        "exit_count": 0,
        "pending_target": None,
        "last_bar_end": live_fixture.previous_close,
    }
    result = monitor(live_fixture, "2026-10-08T14:35:00Z", state=state)
    assert result["next_state"]["position"] == 1
    assert result["next_state"]["entry_count"] == 0
    assert result["next_state"]["pending_target"] is None
    assert not result["actionable"]


def test_live_final_half_hour_is_eligible_and_pending_fills_next_open(live_fixture):
    state = {
        "position": 0,
        "entry_count": 1,
        "exit_count": 0,
        "pending_target": None,
        "last_bar_end": "2026-10-08T19:30:00Z",
    }
    bars = live_fixture.bars_until("2026-10-08T20:01:00Z", price=101.0)
    result = monitor(live_fixture, "2026-10-08T20:01:00Z", state=state, bars=bars)
    assert result["next_state"]["entry_count"] == 2
    assert result["actionable"] and result["action"] == "buy"
    assert result["decision_at"] == "2026-10-08T20:00:00+00:00"
    assert result["check_half_hour"] is True
    assert not bars.is_full_hour.any()
    bars["close"] = 99.0
    result = monitor(live_fixture, "2026-10-08T20:01:00Z", state=state, bars=bars)
    assert result["next_state"]["entry_count"] == 0
    assert not result["actionable"]
    bars["close"] = 101.0
    state.update(entry_count=2, pending_target=1)
    result = monitor(live_fixture, "2026-10-08T20:01:00Z", state=state, bars=bars)
    assert result["next_state"]["position"] == 1
    assert result["next_state"]["entry_count"] == 0


def test_live_stale_unfinished_and_state_gaps_fail_closed(live_fixture):
    stale = live_fixture.bars_until("2026-10-08T14:35:00Z").iloc[:-1]
    result = monitor(live_fixture, "2026-10-08T14:35:00Z", bars=stale)
    assert not result["actionable"] and "stale" in result["suppressed_reason"]
    future = live_fixture.bars_until("2026-10-08T15:35:00Z")
    result = monitor(live_fixture, "2026-10-08T14:35:00Z", bars=future)
    assert not result["actionable"] and "unfinished" in result["detail"]
    state = {
        "position": 0,
        "entry_count": 0,
        "exit_count": 0,
        "pending_target": None,
        "last_bar_end": "2026-10-07T19:15:00Z",
    }
    result = monitor(live_fixture, "2026-10-08T14:35:00Z", state=state)
    assert not result["actionable"] and "canonical grid" in result["detail"]


def test_live_explicit_state_must_be_bound_to_winner(live_fixture):
    state = {
        "position": 0,
        "entry_count": 1,
        "exit_count": 0,
        "pending_target": None,
        "last_bar_end": live_fixture.previous_close,
    }
    result = protocol.monitor_live_policy(
        live_fixture.daily,
        live_fixture.bars_until("2026-10-08T14:35:00Z"),
        live_fixture.calendar,
        live_fixture.candidate,
        live_fixture.metadata,
        as_of="2026-10-08T14:35:00Z",
        state=state,
    )
    assert not result["actionable"] and "candidate_id" in result["detail"]
    result = monitor(
        live_fixture, "2026-10-08T14:35:00Z", state={"candidate_id": "wrong_winner", **state}
    )
    assert not result["actionable"] and "different winner" in result["detail"]


def test_live_yahoo_source_rejected_without_preview_or_action(live_fixture, monkeypatch):
    metadata = replace(live_fixture.metadata, provider="Yahoo Finance")
    monkeypatch.setattr(
        protocol, "_engine", lambda: pytest.fail("Yahoo source must not reach indicator preview")
    )
    result = protocol.monitor_live_policy(
        live_fixture.daily,
        live_fixture.bars_until("2026-10-08T14:35:00Z"),
        live_fixture.calendar,
        live_fixture.candidate,
        metadata,
        as_of="2026-10-08T14:35:00Z",
    )
    assert not result["actionable"] and result["action"] is None
    assert "exactly Alpha Vantage" in result["detail"]


def test_live_daily_provenance_is_independently_required(live_fixture, monkeypatch):
    live_fixture.daily.attrs.clear()
    monkeypatch.setattr(
        protocol,
        "_engine",
        lambda: pytest.fail("Missing daily provenance must not reach indicator preview"),
    )
    result = protocol.monitor_live_policy(
        live_fixture.daily,
        live_fixture.bars_until("2026-10-08T14:35:00Z"),
        live_fixture.calendar,
        live_fixture.candidate,
        live_fixture.metadata,
        as_of="2026-10-08T14:35:00Z",
    )
    assert not result["actionable"] and "independently attested" in result["detail"]


def test_monitor_winner_live_rejects_yahoo_packet_before_reader(prepared_fixture):
    fixture = prepared_fixture
    protocol.run_development(fixture.config, study_dir=fixture.study)
    protocol.run_validation(fixture.config, study_dir=fixture.study)
    yahoo = fixture.study.parent / "yahoo-live"
    yahoo.mkdir()
    (yahoo / "manifest.json").write_text(
        json.dumps(
            {
                "source_metadata": {
                    "provider": "Yahoo Finance",
                    "symbol": "QQQ",
                }
            }
        )
    )
    # FabricatedPacketAdapter deliberately has no read_monitor_packet. The
    # provider receipt must reject before even selecting that numerical reader.
    result = protocol.monitor_winner(
        fixture.config,
        study_dir=fixture.study,
        input_packet=yahoo,
        as_of="2026-10-08T14:35:00Z",
        live=True,
    )
    assert not result["actionable"] and result["action"] is None
    assert "exactly Alpha Vantage" in result["detail"]
