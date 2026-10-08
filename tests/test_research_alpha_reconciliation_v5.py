"""Fabricated Alpha-style prices only; no strategy/performance artifacts."""

import json

import numpy as np
import pandas as pd
import pytest

from trend_following.research_alpha_reconciliation_v5 import (
    DERIVED_PRICE_BASIS,
    JSON_FIELDS,
    ROOT_APPROVED_ROLE_POLICY,
    QuoteRolePolicy,
    classify_observation_coverage,
    factor_action_proof,
    reconcile_alpha_cache,
    reconcile_alpha_frames,
    reconstruct_alpha_raw_observations,
    verify_native_daily_json,
)
from trend_following.research_hourly_data_v5 import (
    DataContractError,
    build_xnys_calendar,
    canonical_half_hour_grid,
)


def fabricated_sources(*, close_role_difference=False):
    calendar = build_xnys_calendar("2024-01-02", "2024-01-31")
    rows = []
    for session in calendar.itertuples():
        for start in pd.date_range(
            session.market_open, session.market_close, freq="30min", inclusive="left"
        ):
            price = round(100 + len(rows) * 0.01, 4)
            rows.append(
                dict(
                    date=start.tz_convert("America/New_York").tz_localize(None),
                    open=price,
                    high=price + 0.02,
                    low=price - 0.01,
                    close=price + 0.01,
                    volume=100.0,
                )
            )
    thirty = pd.DataFrame(rows)
    fifteen_rows = []
    for row in thirty.itertuples():
        for minutes in (0, 15):
            middle = round((row.open + row.close) / 2, 4)
            fifteen_rows.append(
                dict(
                    date=row.date + pd.Timedelta(minutes=minutes),
                    open=row.open if minutes == 0 else middle,
                    high=row.high,
                    low=row.low,
                    close=middle if minutes == 0 else row.close,
                    volume=row.volume / 2,
                )
            )
    fifteen = pd.DataFrame(fifteen_rows)
    daily = (
        thirty.assign(date=thirty.date.dt.normalize())
        .groupby("date")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
    )
    multiplier = np.where(daily.date.lt("2024-01-17"), 2.0, 1.0)
    daily[["open", "high", "low", "close"]] = daily[["open", "high", "low", "close"]].mul(
        multiplier, axis=0
    )
    daily["adj_close"] = daily.close / multiplier
    daily["dividend_amount"] = 0.0
    daily["split_coefficient"] = np.where(daily.date.eq("2024-01-17"), 2.0, 1.0)
    if close_role_difference:
        daily["close"] += 1.0
        daily["high"] = daily[["high", "close"]].max(axis=1)
        daily["adj_close"] = daily.close / multiplier
    return daily, thirty, fifteen, native_json(daily)


def native_json(daily):
    return {
        "Meta Data": {
            "1. Information": "Daily Time Series with Splits and Dividend Events",
            "2. Symbol": "QQQ",
            "3. Last Refreshed": daily.date.iloc[-1].strftime("%Y-%m-%d"),
        },
        "Time Series (Daily)": {
            row.date.strftime("%Y-%m-%d"): {
                json_key: str(getattr(row, column)) for json_key, column in JSON_FIELDS.items()
            }
            for row in daily.itertuples()
        },
    }


def test_explicit_roles_fix_false_identity_without_replacing_book_prices_or_tolerance():
    daily, thirty, fifteen, native = fabricated_sources(close_role_difference=True)
    strict = reconcile_alpha_frames(daily, thirty, fifteen, native, bar_start_session="2024-01-02")
    assert strict.audit["old_identity_diagnostic"]["strict_identity_gate_still_required"]
    assert strict.audit["old_identity_diagnostic"]["required_mismatches"] > 0
    fixed = reconcile_alpha_frames(
        daily,
        thirty,
        fifteen,
        native,
        policy=ROOT_APPROVED_ROLE_POLICY,
        bar_start_session="2024-01-02",
    )
    assert fixed.audit["quote_relationship_reconciled_by_explicit_roles"]
    assert not fixed.audit["old_identity_diagnostic"]["strict_identity_gate_still_required"]
    assert (
        fixed.audit["old_identity_diagnostic"]["rtol"]
        == strict.audit["old_identity_diagnostic"]["rtol"]
        == 0.001
    )
    assert fixed.audit["old_identity_diagnostic"]["atol"] == 0.02
    assert fixed.audit["native_raw_capture"] is False
    assert fixed.audit["price_basis"] == DERIVED_PRICE_BASIS
    assert fixed.audit["physical_split_dividend_model_changed"] is False
    assert fixed.audit["performance_generated"] is False
    factors = np.where(thirty.date.dt.normalize().lt("2024-01-17"), 2.0, 1.0)
    assert np.allclose(fixed.reconstructed_30min.close, thirty.close * factors)
    final = fixed.reconstructed_30min.groupby("session").close.last()
    assert np.allclose(fixed.daily.close.to_numpy() - final.to_numpy(), 1.0)
    assert set(
        [
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "dividend_amount",
            "split_coefficient",
            "daily_price_available_at",
        ]
    ).issubset(fixed.daily)
    assert (
        fixed.daily.daily_price_available_at.dt.tz_convert("America/New_York").dt.hour.eq(17).all()
    )


def test_distinct_roles_require_explicit_policy_approval():
    daily, thirty, fifteen, native = fabricated_sources()
    with pytest.raises(DataContractError, match="approval_reason"):
        reconcile_alpha_frames(
            daily,
            thirty,
            fifteen,
            native,
            policy=QuoteRolePolicy(profile="distinct_daily_feature_and_rth_book_quotes"),
            bar_start_session="2024-01-02",
        )
    with pytest.raises(DataContractError, match="at least10"):
        QuoteRolePolicy(minimum_anchor_days_per_month=1).validate()


def test_native_daily_json_cache_mismatch_and_wrong_security_rejected():
    daily, _, _, native = fabricated_sources()
    assert verify_native_daily_json(daily, native)["verified_sessions"] == len(daily)
    bad = json.loads(json.dumps(native))
    bad["Time Series (Daily)"]["2024-01-02"]["4. close"] = "999"
    with pytest.raises(DataContractError, match="value mismatch"):
        verify_native_daily_json(daily, bad)
    bad = json.loads(json.dumps(native))
    bad["Meta Data"]["2. Symbol"] = "OTHER"
    with pytest.raises(DataContractError, match="QQQ daily"):
        verify_native_daily_json(daily, bad)


def test_factor_changes_must_be_reported_actions_and_terminal_anchor_must_match():
    daily, _, _, _ = fabricated_sources()
    factor, proof = factor_action_proof(daily)
    assert factor.iloc[0] == 2 and factor.iloc[-1] == 1
    assert proof["split_only_transitions_exact"] == 1
    changed = daily.copy()
    changed.loc[1, "adj_close"] *= 0.99
    with pytest.raises(DataContractError, match="without Alpha corporate actions"):
        factor_action_proof(changed)
    changed = daily.copy()
    changed.loc[len(changed) - 1, "adj_close"] *= 0.9
    with pytest.raises(DataContractError, match="terminal adjustment anchor"):
        factor_action_proof(changed)


def test_intraday_resolution_and_monthly_factor_vintage_checks_are_hard_gates():
    daily, thirty, fifteen, native = fabricated_sources()
    changed = fifteen.copy()
    changed.loc[1, "close"] += 0.001
    changed.loc[1, "high"] = max(changed.loc[1, "high"], changed.loc[1, "close"])
    with pytest.raises(DataContractError, match="quote windows disagree"):
        reconcile_alpha_frames(daily, thirty, changed, native, bar_start_session="2024-01-02")
    t, f = thirty.copy(), fifteen.copy()
    for frame in (t, f):
        frame[["open", "high", "low", "close"]] *= 1.10
    with pytest.raises(DataContractError, match="independent daily price"):
        reconcile_alpha_frames(daily, t, f, native, bar_start_session="2024-01-02")


def test_missing_quotes_remain_missing_without_fill_or_price_fabrication():
    daily, thirty, fifteen, native = fabricated_sources()
    absent = thirty.date.eq(pd.Timestamp("2024-01-10 09:30"))
    missing = thirty.loc[~absent].reset_index(drop=True)
    result = reconcile_alpha_frames(
        daily,
        missing,
        fifteen,
        native,
        policy=ROOT_APPROVED_ROLE_POLICY,
        bar_start_session="2024-01-02",
    )
    assert len(result.reconstructed_30min) == len(thirty) - 1
    assert result.observation_coverage.status.eq("unknown_missing").sum() == 1
    row = result.observation_coverage[
        result.observation_coverage.status.eq("unknown_missing")
    ].iloc[0]
    assert not row.signal_eligible and not row.fill_eligible
    assert result.audit["coverage"]["unresolved_nonhalt_missing"] == 1
    assert not result.audit["market_backtest_ready"]
    assert result.audit["coverage"]["holes_or_halt_prices_fabricated"] is False


def test_known_halt_precise_blackout_keeps_valid12_open_but_no_overlap_confirmations():
    grid = canonical_half_hour_grid(build_xnys_calendar("2013-08-22", "2013-08-22"))
    coverage, blackouts = classify_observation_coverage(grid, grid.copy())
    local = coverage.bar_start.dt.tz_convert("America/New_York").dt.strftime("%H:%M")
    noon = coverage[local.eq("12:00")].iloc[0]
    assert noon.status == "known_halt_overlap" and noon.fill_eligible and not noon.signal_eligible
    three = coverage[local.eq("15:00")].iloc[0]
    assert not three.fill_eligible and not three.signal_eligible
    resumed = coverage[local.eq("15:30")].iloc[0]
    assert resumed.fill_eligible and resumed.signal_eligible
    assert blackouts.start.iloc[0] == pd.Timestamp("2013-08-22T16:23Z")
    assert blackouts.end.iloc[0] == pd.Timestamp("2013-08-22T19:25Z")


def test_cache_wrapper_no_network_no_other_provider_no_quote_exports_or_source_mutation(
    tmp_path, monkeypatch
):
    import requests

    daily, thirty, fifteen, native = fabricated_sources(close_role_difference=True)
    root = tmp_path / "source"
    paths = {}
    for name, frame in [("daily", daily), ("30min", thirty), ("15min", fifteen)]:
        folder = "alpha_vantage_daily_adjusted" if name == "daily" else "alpha_vantage_" + name
        path = root / "data/raw" / folder / "QQQ.parquet"
        path.parent.mkdir(parents=True)
        frame.to_parquet(path, index=False)
        paths[name] = path
    paths["native_daily_json"] = root / "data/raw/valuation/alpha_vantage_daily_adjusted_QQQ.json"
    paths["native_daily_json"].parent.mkdir(parents=True)
    paths["native_daily_json"].write_text(json.dumps(native))
    original = {path: path.read_bytes() for path in paths.values()}

    def no_network(*args, **kwargs):
        raise AssertionError("no network/data-provider calls allowed")

    monkeypatch.setattr(requests, "get", no_network)
    output = tmp_path / "private_metadata_audit"
    result = reconcile_alpha_cache(
        root, policy=ROOT_APPROVED_ROLE_POLICY, bar_start_session="2024-01-02", output_dir=output
    )
    assert not result.audit["performance_generated"]
    assert all(record["provider"] == "AlphaVantage" for record in result.audit["source_files"])
    assert all("yfinance" not in record["path"] for record in result.audit["source_files"])
    assert {path: path.read_bytes() for path in paths.values()} == original
    for path in output.glob("*.csv"):
        columns = set(pd.read_csv(path).columns)
        assert not {"open", "high", "low", "close", "volume"}.intersection(columns)
    observed, audit = reconstruct_alpha_raw_observations(
        paths["daily"],
        paths["30min"],
        paths["15min"],
        paths["native_daily_json"],
        bar_start_session="2024-01-02",
    )
    assert len(observed) == len(thirty) and audit["native_raw_capture"] is False
    with pytest.raises(FileExistsError, match="immutable"):
        reconcile_alpha_cache(root, bar_start_session="2024-01-02", output_dir=output)


def make_source_root(tmp_path, *, one_missing=False):
    daily, thirty, fifteen, native = fabricated_sources(close_role_difference=True)
    if one_missing:
        thirty = thirty.loc[thirty.date.ne(pd.Timestamp("2024-01-10 09:30"))].reset_index(drop=True)
    root = tmp_path / "alpha_source"
    for name, frame in (("daily_adjusted", daily), ("30min", thirty), ("15min", fifteen)):
        path = root / "data/raw" / ("alpha_vantage_" + name) / "QQQ.parquet"
        path.parent.mkdir(parents=True)
        frame.to_parquet(path, index=False)
    jp = root / "data/raw/valuation/alpha_vantage_daily_adjusted_QQQ.json"
    jp.parent.mkdir(parents=True)
    jp.write_text(json.dumps(native))
    return root


def test_quality_packet_hashes_raw_source_proof_and_does_not_waive_unknown_gap(tmp_path):
    from trend_following.research_alpha_reconciliation_v5 import (
        build_reconciled_alpha_bundle,
        read_reconciled_packet,
        write_reconciled_packet,
    )

    root = make_source_root(tmp_path, one_missing=True)
    bundle = build_reconciled_alpha_bundle(
        root, bar_start_session="2024-01-02", cutoff_session="2024-01-31"
    )
    packet = tmp_path / "strict_classified_packet"
    cash = pd.DataFrame(
        {"available_at": ["2023-12-29T23:00Z"], "annual_rate": [0.04], "series_id": ["DTB3"]}
    )
    manifest = write_reconciled_packet(packet, bundle, cash_observations=cash)
    assert not manifest["historical_ready"]
    assert manifest["audit"]["unknown_missing_count"] == 1
    assert manifest["performance_generated"] is False
    assert set(manifest["files"]) == {
        "daily.csv",
        "bars.csv",
        "calendar.csv",
        "coverage.csv",
        "blackouts.csv",
        "unit_proof.json",
        "DTB3_available.csv",
    }
    assert manifest["quality_model"]["daily_valuation_clock_NY"] == "17:00"
    assert manifest["quality_model"]["source_price_origin"] == DERIVED_PRICE_BASIS
    d, b, c, cov, bl, m = read_reconciled_packet(packet)
    assert m == manifest
    assert len(b) == len(bundle.bars) and len(cov) == len(bundle.coverage)
    assert cov.status.eq("unknown_missing").sum() == 1
    assert set(["open", "high", "low", "close", "daily_price_available_at"]).issubset(d)
    with pytest.raises(FileExistsError, match="immutable"):
        write_reconciled_packet(packet, bundle)
    with (packet / "bars.csv").open("a") as stream:
        stream.write("tampering\n")
    with pytest.raises(DataContractError, match="hash mismatch"):
        read_reconciled_packet(packet)


def test_observed_only_packet_requires_explicit_user_authorization(tmp_path):
    from trend_following.research_alpha_reconciliation_v5 import (
        build_reconciled_alpha_bundle,
        read_reconciled_packet,
        write_reconciled_packet,
    )

    root = make_source_root(tmp_path, one_missing=True)
    bundle = build_reconciled_alpha_bundle(
        root, bar_start_session="2024-01-02", cutoff_session="2024-01-31"
    )
    with pytest.raises(DataContractError, match="human gap-policy approval"):
        write_reconciled_packet(
            tmp_path / "unapproved", bundle, quality_profile="observed_only_blackout_v1"
        )
    assert not (tmp_path / "unapproved").exists()
    auth = {
        "quality_profile": "observed_only_blackout_v1",
        "explicit_user_approval": True,
        "approved_at": "2024-02-01T15:00Z",
        "instruction": "Explicit fabricated-test approval: no fills/signals, cancel pending, reset counts, hold positions through unknown blackouts",
    }
    manifest = write_reconciled_packet(
        tmp_path / "approved",
        bundle,
        quality_profile="observed_only_blackout_v1",
        gap_authorization=auth,
    )
    assert manifest["historical_ready"] and manifest["audit"]["unknown_missing_count"] == 1
    assert read_reconciled_packet(tmp_path / "approved")[-1]["gap_authorization"] == auth


def test_packet_prefix_preserves_full_master_proof_without_exposing_future_numeric_rows(tmp_path):
    from trend_following.research_alpha_reconciliation_v5 import (
        ReconciledBundle,
        build_reconciled_alpha_bundle,
        read_reconciled_packet,
        write_reconciled_packet,
    )

    root = make_source_root(tmp_path)
    master = build_reconciled_alpha_bundle(
        root, bar_start_session="2024-01-02", cutoff_session="2024-01-31"
    )
    end = "2024-01-19"
    meta = {**master.source_metadata, "coverage_end_session": end}
    prefix = ReconciledBundle(
        master.daily.loc[master.daily.session.le(end)].copy(),
        master.bars.loc[master.bars.session.le(end)].copy(),
        master.calendar.loc[master.calendar.session.le(end)].copy(),
        master.coverage.loc[master.coverage.session.le(end)].copy(),
        master.blackouts.copy(),
        master.unit_proof,
        meta,
        master.source_files,
    )
    manifest = write_reconciled_packet(tmp_path / "prefix", prefix)
    d, b, c, cov, _, m = read_reconciled_packet(tmp_path / "prefix")
    assert d.session.max() == b.session.max() == c.session.max() == cov.session.max() == end
    assert m["unit_proof_sha256"] == manifest["unit_proof_sha256"]
    assert len(prefix.unit_proof["data_bindings"]["daily_session_hashes"]) > len(d)
    changed = prefix.bars.copy()
    changed.loc[0, "close"] += 0.000001
    broken = ReconciledBundle(
        prefix.daily,
        changed,
        prefix.calendar,
        prefix.coverage,
        prefix.blackouts,
        prefix.unit_proof,
        prefix.source_metadata,
        prefix.source_files,
    )
    with pytest.raises(DataContractError, match="numeric prefix changed"):
        write_reconciled_packet(tmp_path / "altered", broken)


def test_strict_readiness_blocks_predevelopment_missing_observation_even_when_scored_window_clean(
    tmp_path,
):
    from trend_following.research_alpha_reconciliation_v5 import (
        build_reconciled_alpha_bundle,
        read_reconciled_packet,
        write_reconciled_packet,
    )

    source = make_source_root(tmp_path, one_missing=True)  # Jan10 gap precedes declaredJan22 start.
    bundle = build_reconciled_alpha_bundle(
        source, bar_start_session="2024-01-22", cutoff_session="2024-01-31"
    )
    output = tmp_path / "predevelopment_gap_strict_packet"
    manifest = write_reconciled_packet(output, bundle)
    assert manifest["audit"]["unknown_missing_count"] == 1
    assert manifest["audit"]["required_window_unknown_missing_count"] == 0
    assert manifest["audit"]["unknown_missing_gate_scope"] == "all_supplied_scheduled_coverage"
    assert not manifest["historical_ready"]
    loaded = read_reconciled_packet(output)[-1]
    assert not loaded["historical_ready"]
    assert loaded["audit"]["unknown_missing_count"] == 1
    assert loaded["audit"]["required_window_unknown_missing_count"] == 0
    from trend_following.research_quality_engine_v5 import (
        QualityPolicy,
        build_quality_support_cache,
    )

    with pytest.raises(ValueError, match="unknown"):
        build_quality_support_cache(
            bundle.daily,
            bundle.bars,
            bundle.calendar,
            coverage=bundle.coverage,
            blackouts=bundle.blackouts,
            quality_policy=QualityPolicy(profile="strict_tradable_v1"),
        )


@pytest.mark.parametrize(
    "bad", [np.nan, np.inf, -np.inf, True, np.bool_(True), 10.0, 10.5, "10", None, 9]
)
def test_minimum_anchor_days_are_finite_non_bool_integers(bad):
    with pytest.raises(DataContractError, match="at least10"):
        QuoteRolePolicy(minimum_anchor_days_per_month=bad).validate()


@pytest.fixture
def valid_unit_proof():
    daily, thirty, fifteen, native = fabricated_sources(close_role_difference=True)
    return reconcile_alpha_frames(
        daily,
        thirty,
        fifteen,
        native,
        policy=ROOT_APPROVED_ROLE_POLICY,
        bar_start_session="2024-01-02",
    ).audit


@pytest.mark.parametrize(
    "section,field,bad",
    [
        ("native_daily_json_proof", "verified_sessions", np.nan),
        ("native_daily_json_proof", "verified_sessions", True),
        ("native_daily_json_proof", "verified_fields", None),
        ("daily_action_factor_proof", "action_transitions", np.inf),
        ("daily_action_factor_proof", "nonaction_transitions", False),
        ("daily_action_factor_proof", "cash_convention_max_relative_error", np.nan),
        ("same_provider_15_30_alignment", "windows", True),
        ("same_provider_15_30_alignment", "all_ohlc_matches", 1.5),
        ("independent_price_anchor_proof", "minimum_required_anchor_days_per_month", np.nan),
        ("independent_price_anchor_proof", "minimum_anchor_days_per_month", np.nan),
        ("independent_price_anchor_proof", "minimum_anchor_days_per_month", np.inf),
        ("independent_price_anchor_proof", "minimum_anchor_days_per_month", 10.0),
        ("independent_price_anchor_proof", "months", True),
        ("independent_price_anchor_proof", "anchored_sessions", 9999),
        ("old_identity_diagnostic", "required_mismatches", 9999),
        ("coverage", "known_halt_consistent_missing", np.nan),
    ],
)
def test_malformed_proof_counts_and_diagnostics_rejected(valid_unit_proof, section, field, bad):
    import copy

    from trend_following.research_alpha_reconciliation_v5 import _validate_unit_proof

    proof = copy.deepcopy(valid_unit_proof)
    proof[section][field] = bad
    with pytest.raises(DataContractError):
        _validate_unit_proof(proof)


def test_proof_count_consistency_and_boolean_zero_are_not_accepted(valid_unit_proof):
    import copy

    from trend_following.research_alpha_reconciliation_v5 import _validate_unit_proof

    _validate_unit_proof(valid_unit_proof)
    p = copy.deepcopy(valid_unit_proof)
    p["native_daily_json_proof"]["mismatch_counts"]["close"] = False
    with pytest.raises(DataContractError, match="non-bool integer"):
        _validate_unit_proof(p)
    p = copy.deepcopy(valid_unit_proof)
    p["daily_action_factor_proof"]["nonaction_transitions"] -= 1
    with pytest.raises(DataContractError, match="every native daily transition"):
        _validate_unit_proof(p)
    p = copy.deepcopy(valid_unit_proof)
    p["coverage"]["required_missing_expected_timestamps"] = 1
    with pytest.raises(DataContractError, match="missing counts are inconsistent"):
        _validate_unit_proof(p)


def test_reconciled_cash_series_must_be_dtb3_in_writer_and_reader(tmp_path):
    from trend_following.research_alpha_reconciliation_v5 import (
        build_reconciled_alpha_bundle,
        read_reconciled_packet,
        write_reconciled_packet,
    )
    from trend_following.research_hourly_data_v5 import sha256_file

    source = make_source_root(tmp_path)
    bundle = build_reconciled_alpha_bundle(
        source, bar_start_session="2024-01-02", cutoff_session="2024-01-31"
    )
    cash = pd.DataFrame(
        {"available_at": ["2023-12-29T23:00Z"], "annual_rate": [0.04], "series_id": ["NOT_DTB3"]}
    )
    with pytest.raises(DataContractError, match="DTB3 series_id"):
        write_reconciled_packet(tmp_path / "wrong_series", bundle, cash_observations=cash)
    assert not (tmp_path / "wrong_series").exists()
    with pytest.raises(DataContractError, match="DTB3 series_id"):
        write_reconciled_packet(
            tmp_path / "missing_series", bundle, cash_observations=cash.drop(columns="series_id")
        )
    cash["series_id"] = "DTB3"
    packet = tmp_path / "valid_cash"
    write_reconciled_packet(packet, bundle, cash_observations=cash)
    wrong = pd.read_csv(packet / "DTB3_available.csv")
    wrong["series_id"] = "NOT_DTB3"
    wrong.to_csv(packet / "DTB3_available.csv", index=False)
    manifest = json.loads((packet / "manifest.json").read_text())
    manifest["files"]["DTB3_available.csv"]["sha256"] = sha256_file(packet / "DTB3_available.csv")
    manifest["files"]["DTB3_available.csv"]["bytes"] = (
        (packet / "DTB3_available.csv").stat().st_size
    )
    (packet / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(DataContractError, match="DTB3 series_id"):
        read_reconciled_packet(packet)


def test_session_hashing_preserves_exact_hashes_and_ignores_large_attrs_without_mutation():
    from trend_following.research_alpha_reconciliation_v5 import (
        DAILY_BINDING_COLUMNS,
        _json_hash,
        _row_hash,
        _session_hashes,
    )

    daily, _, _, _ = fabricated_sources()
    daily["session"] = daily.date.dt.strftime("%Y-%m-%d")
    daily["daily_price_available_at"] = pd.DatetimeIndex(
        [
            pd.Timestamp(session + " 17:00", tz="America/New_York").tz_convert("UTC")
            for session in daily.session
        ]
    )
    # The original exact row/day hashing algorithm on metadata-free values is
    # the authority; only metadata propagation is optimized, never encoding.
    expected = {
        str(session): _json_hash(
            [_row_hash(row, DAILY_BINDING_COLUMNS) for _, row in part.iterrows()]
        )
        for session, part in daily.groupby("session", sort=False)
    }
    large_proof = {
        "data_bindings": {
            name: {"session_" + str(i): format(i, "064x") for i in range(6683)}
            for name in (
                "daily_session_hashes",
                "bar_session_hashes",
                "calendar_session_hashes",
                "coverage_session_hashes",
            )
        }
    }
    daily.attrs["unit_proof"] = large_proof
    daily.attrs["source_metadata"] = {
        "provider": "explicit_fabricated_fixture",
        "native_raw_capture": False,
    }
    attrs_object = daily.attrs
    before_values = daily.to_numpy(copy=True)
    before_columns = daily.columns.copy()
    before_dtypes = daily.dtypes.copy()
    actual = _session_hashes(daily, DAILY_BINDING_COLUMNS)
    assert actual == expected
    assert daily.attrs is attrs_object
    assert daily.attrs["unit_proof"] is large_proof
    assert daily.attrs["source_metadata"] == {
        "provider": "explicit_fabricated_fixture",
        "native_raw_capture": False,
    }
    assert daily.columns.equals(before_columns) and daily.dtypes.equals(before_dtypes)
    assert np.array_equal(daily.to_numpy(copy=False), before_values)


def test_session_hashing_never_deepcopies_metadata_into_group_or_row_series():
    from trend_following.research_alpha_reconciliation_v5 import (
        DAILY_BINDING_COLUMNS,
        _session_hashes,
    )

    class DeepcopyBomb:
        def __deepcopy__(self, memo):
            raise AssertionError("Metadata was copied while hashing numeric rows")

    daily, _, _, _ = fabricated_sources()
    daily["session"] = daily.date.dt.strftime("%Y-%m-%d")
    daily["daily_price_available_at"] = pd.DatetimeIndex(
        [
            pd.Timestamp(session + " 17:00", tz="America/New_York").tz_convert("UTC")
            for session in daily.session
        ]
    )
    expected = _session_hashes(daily, DAILY_BINDING_COLUMNS)
    marker = DeepcopyBomb()
    daily.attrs["unit_proof"] = {"deep_nested_marker": marker}
    assert _session_hashes(daily, DAILY_BINDING_COLUMNS) == expected
    assert daily.attrs["unit_proof"]["deep_nested_marker"] is marker
