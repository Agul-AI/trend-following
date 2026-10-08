from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from trend_following.research_hourly_data_v5 import (
    BAR_COLUMNS,
    DataContractError,
    SourceMetadata,
    audit_source_coverage,
    build_xnys_calendar,
    canonical_hour_grid,
    download_alpha_vantage_raw_month,
    normalize_daily_actions,
    normalize_provider_present_day,
    normalize_raw_subbars,
    prepare_input_packet,
    read_input_packet,
    validate_calendar,
    validate_cash_observations,
    validate_research_inputs,
)

AS_OF = pd.Timestamp("2024-12-01 22:00:00Z")


def metadata(**kwargs):
    result = SourceMetadata(
        "fabricated_test_fixture",
        "QQQ",
        30,
        "start",
        "America/New_York",
        "raw_as_traded",
        False,
        AS_OF.isoformat(),
        "fabricated known raw start-labelled fixture",
    )
    return replace(result, **kwargs)


def raw_fixture(calendar):
    rows = []
    for session in calendar.itertuples():
        for start in pd.date_range(
            session.market_open, session.market_close, freq="30min", inclusive="left"
        ):
            value = 100 + len(rows) / 10
            rows.append(
                dict(
                    date=start.tz_convert("America/New_York").tz_localize(None),
                    open=value,
                    high=value + 0.2,
                    low=value - 0.1,
                    close=value + 0.1,
                    volume=100,
                )
            )
    return pd.DataFrame(rows)


def prepared_fixture(start="2024-11-27", end="2024-11-29"):
    calendar = build_xnys_calendar(start, end)
    raw = raw_fixture(calendar)
    bars = normalize_raw_subbars(raw, calendar, metadata(), as_of=AS_OF)
    daily = pd.DataFrame(
        {
            "session": calendar.session,
            "close": bars.groupby("session").close.last().to_numpy(),
            "dividend_amount": 0.0,
            "split_coefficient": 1.0,
        }
    )
    return daily, bars, calendar


def test_xnys_grid_regular_early_close_dst_and_special_closure():
    calendar = build_xnys_calendar("2024-11-27", "2024-11-29")
    assert list(calendar.session) == ["2024-11-27", "2024-11-29"]
    grid = canonical_hour_grid(calendar)
    normal = grid[grid.session.eq("2024-11-27")]
    early = grid[grid.session.eq("2024-11-29")]
    assert len(normal) == 7 and normal.is_full_hour.sum() == 6
    assert list(normal.duration_minutes) == [60.0] * 6 + [30.0]
    assert normal.bar_start.iloc[0] == pd.Timestamp("2024-11-27 14:30Z")
    assert normal.bar_end.iloc[-1] == pd.Timestamp("2024-11-27 21:00Z")
    assert len(early) == 4 and early.is_full_hour.sum() == 3
    assert early.bar_end.iloc[-1] == pd.Timestamp("2024-11-29 18:00Z")
    summer = canonical_hour_grid(build_xnys_calendar("2024-07-02", "2024-07-02"))
    assert summer.bar_start.iloc[0] == pd.Timestamp("2024-07-02 13:30Z")
    special = build_xnys_calendar("2001-09-10", "2001-09-18")
    assert list(special.session) == ["2001-09-10", "2001-09-17", "2001-09-18"]


def test_calendar_cannot_remove_an_entire_session():
    calendar = build_xnys_calendar("2024-01-02", "2024-01-04")
    with pytest.raises(DataContractError, match="differs from XNYS"):
        canonical_hour_grid(calendar.drop(index=1))


def test_calendar_incorrect_early_close_rejected():
    calendar = build_xnys_calendar("2024-11-29", "2024-11-29")
    calendar.loc[0, "market_close"] = pd.Timestamp("2024-11-29 21:00Z")
    calendar.loc[0, "is_early_close"] = False
    with pytest.raises(DataContractError, match="differs from XNYS"):
        validate_calendar(calendar)


def test_raw_30min_aggregation_open_high_low_close_volume():
    calendar = build_xnys_calendar("2024-11-27", "2024-11-27")
    raw = raw_fixture(calendar)
    bars = normalize_raw_subbars(raw, calendar, metadata(), as_of=AS_OF)
    assert tuple(bars.columns) == BAR_COLUMNS
    first = bars.iloc[0]
    assert first.open == raw.open.iloc[0]
    assert first.high == raw.high.iloc[:2].max()
    assert first.low == raw.low.iloc[:2].min()
    assert first.close == raw.close.iloc[1]
    assert first.volume == 200
    assert bars.iloc[-1].volume == 100
    assert not bars.iloc[-1].is_full_hour
    assert bars.attrs["source_metadata"] == asdict(metadata())


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"adjusted": True}, "RAW"),
        ({"price_basis": "unknown"}, "RAW"),
        ({"timestamp_label": "end"}, "start-labelled"),
        ({"timestamp_label": "unverified"}, "start-labelled"),
        ({"verification": ""}, "start-labelled"),
        ({"interval_minutes": 60}, "must not be relabelled"),
    ],
)
def test_unverified_or_adjusted_source_is_never_relabelled(changes, match):
    calendar = build_xnys_calendar("2024-11-27", "2024-11-27")
    with pytest.raises(DataContractError, match=match):
        normalize_raw_subbars(raw_fixture(calendar), calendar, metadata(**changes), as_of=AS_OF)


def test_gap_report_tracks_wholly_missing_session_and_no_padding():
    calendar = build_xnys_calendar("2024-01-02", "2024-01-04")
    raw = raw_fixture(calendar)
    raw = raw[raw.date.dt.strftime("%Y-%m-%d").ne("2024-01-03")].reset_index(drop=True)
    report = audit_source_coverage(raw, calendar, metadata(adjusted=True, price_basis="adjusted"))
    assert len(report) == 13
    assert set(report.session) == {"2024-01-03"}
    assert not report.attrs["coverage"]["basis_verified"]
    with pytest.raises(DataContractError, match="unrepaired"):
        normalize_raw_subbars(raw, calendar, metadata(), as_of=AS_OF)


def test_missing_subbar_extra_end_label_and_duplicate_rejected():
    calendar = build_xnys_calendar("2024-11-27", "2024-11-27")
    raw = raw_fixture(calendar)
    with pytest.raises(DataContractError, match="unrepaired"):
        normalize_raw_subbars(raw.drop(index=1), calendar, metadata(), as_of=AS_OF)
    endlabel = raw.copy()
    endlabel.date += pd.Timedelta(minutes=30)
    with pytest.raises(DataContractError, match="unrepaired"):
        normalize_raw_subbars(endlabel, calendar, metadata(), as_of=AS_OF)
    duplicated = pd.concat([raw.iloc[:1], raw], ignore_index=True)
    with pytest.raises(DataContractError, match="unique"):
        normalize_raw_subbars(duplicated, calendar, metadata(), as_of=AS_OF)
    with pytest.raises(DataContractError, match="ordered"):
        normalize_raw_subbars(raw.iloc[::-1], calendar, metadata(), as_of=AS_OF)


@pytest.mark.parametrize(
    "column,value", [("close", np.inf), ("volume", -1), ("open", 0), ("high", 1), ("low", 200)]
)
def test_nonfinite_nonpositive_and_inconsistent_ohlc_rejected(column, value):
    calendar = build_xnys_calendar("2024-11-27", "2024-11-27")
    raw = raw_fixture(calendar)
    raw.loc[0, column] = value
    with pytest.raises(DataContractError):
        normalize_raw_subbars(raw, calendar, metadata(), as_of=AS_OF)


def test_as_of_does_not_accept_moving_or_unfinished_source():
    calendar = build_xnys_calendar("2024-11-27", "2024-11-27")
    before = pd.Timestamp("2024-11-27 20:45Z")
    with pytest.raises(DataContractError, match="unfinished"):
        normalize_raw_subbars(
            raw_fixture(calendar), calendar, metadata(observed_at=before.isoformat()), as_of=before
        )
    with pytest.raises(DataContractError, match="observed_at exceeds"):
        normalize_raw_subbars(raw_fixture(calendar), calendar, metadata(), as_of=before)


def test_daily_action_schema_coverage_and_raw_close_not_adjusted():
    daily, bars, calendar = prepared_fixture()
    source = daily.rename(columns={"session": "date"}).copy()
    source["adj_close"] = 1.0
    assert np.array_equal(normalize_daily_actions(source, calendar).close, daily.close)
    with pytest.raises(DataContractError, match="missing columns"):
        normalize_daily_actions(source.drop(columns="dividend_amount"), calendar)
    with pytest.raises(DataContractError, match="coverage missing"):
        normalize_daily_actions(source.iloc[:1], calendar)
    for column, value in [("dividend_amount", -0.1), ("split_coefficient", 0)]:
        broken = source.copy()
        broken.loc[0, column] = value
        with pytest.raises(DataContractError, match="daily raw close/actions"):
            normalize_daily_actions(broken, calendar)
    assert validate_research_inputs(daily, bars, calendar, as_of=AS_OF)["complete"]


def test_canonical_bar_and_daily_close_validation():
    daily, bars, calendar = prepared_fixture()
    with pytest.raises(DataContractError, match="grid mismatch"):
        validate_research_inputs(daily, bars.drop(index=0), calendar, as_of=AS_OF)
    changed = bars.copy()
    changed.loc[6, "is_full_hour"] = True
    with pytest.raises(DataContractError, match="grid mismatch"):
        validate_research_inputs(daily, changed, calendar, as_of=AS_OF)
    changed = daily.copy()
    changed.loc[0, "close"] *= 2
    with pytest.raises(DataContractError, match="RAW close mismatch"):
        validate_research_inputs(changed, bars, calendar, as_of=AS_OF)
    with pytest.raises(DataContractError, match="unfinished"):
        validate_research_inputs(daily, bars, calendar, as_of=pd.Timestamp("2024-11-29 17:45Z"))


def test_explicit_bar_start_allows_daily_warmup_but_never_hides_leading_gaps():
    daily, bars, calendar = prepared_fixture("2024-01-02", "2024-01-04")
    bars = bars[bars.session.ge("2024-01-03")].reset_index(drop=True)
    with pytest.raises(DataContractError, match="grid mismatch"):
        validate_research_inputs(daily, bars, calendar, as_of=AS_OF)
    audit = validate_research_inputs(
        daily, bars, calendar, as_of=AS_OF, bar_start_session="2024-01-03"
    )
    assert audit["bar_start_session"] == "2024-01-03"
    with pytest.raises(DataContractError, match="declared bar_start_session"):
        validate_research_inputs(daily, bars, calendar, as_of=AS_OF, bar_start_session="2024-01-01")


def test_packet_roundtrip_hashes_provenance_and_no_overwrite(tmp_path):
    daily, bars, calendar = prepared_fixture()
    source = tmp_path / "original_source.csv"
    source.write_text("raw test source remains immutable\n")
    meta = metadata(source_files=(str(source),))
    bars.attrs["source_metadata"] = asdict(meta)
    cash = pd.DataFrame(
        {
            "available_at": ["2024-01-02T23:00:00Z"],
            "annual_rate": [0.04],
            "observation_date": ["2024-01-02"],
            "series_id": ["DTB3"],
        }
    )
    packet = tmp_path / "packet"
    manifest = prepare_input_packet(
        packet,
        daily,
        bars,
        calendar,
        meta,
        as_of=AS_OF,
        cash_observations=cash,
        cash_source_file=source,
    )
    assert len(manifest["files"]) == 4
    assert manifest["cash_observations"]["filename"] == "DTB3_available.csv"
    assert manifest["calendar_provenance"]["name"] == "XNYS"
    d2, b2, c2, receipt = read_input_packet(packet)
    assert d2.session.equals(daily.session)
    assert b2.attrs["source_metadata"] == asdict(meta)
    assert c2.attrs["calendar_provenance"]["name"] == "XNYS"
    assert receipt == manifest
    with pytest.raises(FileExistsError, match="immutable"):
        prepare_input_packet(packet, daily, bars, calendar, meta, as_of=AS_OF)
    with (packet / "bars.csv").open("a") as stream:
        stream.write("tampering\n")
    with pytest.raises(DataContractError, match="hash mismatch"):
        read_input_packet(packet)


def test_prepare_requires_matching_verified_normalization_provenance(tmp_path):
    daily, bars, calendar = prepared_fixture()
    bars.attrs.clear()
    with pytest.raises(DataContractError, match="normalization provenance"):
        prepare_input_packet(tmp_path / "packet", daily, bars, calendar, metadata(), as_of=AS_OF)
    assert not (tmp_path / "packet").exists()


def test_cash_timestamp_schema_and_future_unavailable_rates_rejected():
    with pytest.raises(DataContractError, match="missing columns"):
        validate_cash_observations(pd.DataFrame({"annual_rate": [0.04]}), as_of=AS_OF)
    cash = pd.DataFrame({"available_at": ["2025-01-01T23:00Z"], "annual_rate": [0.04]})
    with pytest.raises(DataContractError, match="future rates"):
        validate_cash_observations(cash, as_of=AS_OF)
    cash.available_at = "2024-01-02T23:00Z"
    with pytest.raises(DataContractError, match="first bar"):
        validate_cash_observations(cash, as_of=AS_OF, first_bar_start="2024-01-02T14:30Z")


def test_present_day_monitor_is_separate_raw_completed_and_partial_aware():
    daily, bars, calendar = prepared_fixture("2024-11-29", "2024-11-29")
    now = pd.Timestamp("2024-11-29 18:01Z")
    meta = metadata(
        provider="Yahoo_verified_fixture", interval_minutes=60, observed_at=now.isoformat()
    )
    provider = bars[["bar_start", "open", "high", "low", "close", "volume"]]
    monitor = normalize_provider_present_day(provider, calendar, meta, as_of=now)
    assert monitor.is_full_hour.sum() == 3 and len(monitor) == 4
    assert monitor.attrs["role"] == "monitor_only_not_historical_research_splice"
    with pytest.raises(DataContractError, match="unfinished monitor"):
        normalize_provider_present_day(
            provider,
            calendar,
            replace(meta, observed_at="2024-11-29T17:45Z"),
            as_of="2024-11-29T17:45Z",
        )
    with pytest.raises(DataContractError, match="historical"):
        normalize_provider_present_day(provider, calendar, meta, as_of=AS_OF)
    with pytest.raises(DataContractError, match="RAW"):
        normalize_provider_present_day(
            provider, calendar, replace(meta, price_basis="split_adjusted"), as_of=now
        )


def test_download_denial_is_immutable_and_never_leaks_credentials(monkeypatch, tmp_path):
    import requests

    key = "private_test_credential"
    calls = []

    class Response:
        status_code = 200
        text = '{"Information": "premium endpoint ' + key + '"}'

    def get(url, **kwargs):
        calls.append(kwargs)
        return Response()

    monkeypatch.setattr(requests, "get", get)
    with pytest.raises(DataContractError, match="denied/raw data unavailable") as error:
        download_alpha_vantage_raw_month("2004-11", tmp_path, api_key=key)
    assert key not in str(error.value)
    cache = tmp_path / "QQQ" / "2004-11"
    assert key not in (cache / "provider_error.json").read_text()
    assert key not in (cache / "metadata.json").read_text()
    assert calls[0]["params"]["adjusted"] == "false"
    assert calls[0]["params"]["extended_hours"] == "false"
    with pytest.raises(FileExistsError, match="immutable"):
        download_alpha_vantage_raw_month("2004-11", tmp_path, api_key=key)
    assert len(calls) == 1


def test_documented_announced_special_open_is_frozen_not_inferred_from_prices():
    calendar = build_xnys_calendar("2002-09-11", "2002-09-11")
    assert calendar.market_open.iloc[0] == pd.Timestamp("2002-09-11 15:00Z")
    assert calendar.market_close.iloc[0] == pd.Timestamp("2002-09-11 20:00Z")
    grid = canonical_hour_grid(calendar)
    assert len(grid) == 5 and grid.is_full_hour.all()
    record = calendar.attrs["calendar_provenance"]["scheduled_open_overrides"][0]
    assert record["source_url"].startswith("https://www.nasdaqtrader.com/")
    assert "not_actual_first_trade" in record["basis"]
    altered = calendar.copy()
    altered.loc[0, "market_open"] = pd.Timestamp("2002-09-11 13:30Z")
    with pytest.raises(DataContractError):
        validate_calendar(altered)


def test_legacy_hour_monitor_receipt_excludes_today_daily_and_uncompleted_hour(tmp_path):
    from trend_following.research_hourly_data_v5 import (
        normalize_monitor_subbars,
        prepare_monitor_packet,
        read_monitor_packet,
        validate_monitor_inputs,
    )

    calendar = build_xnys_calendar("2024-11-27", "2024-11-29")
    now = pd.Timestamp("2024-11-29 17:15Z")
    actions = {
        "session": "2024-11-29",
        "dividend_amount": 0.0,
        "split_coefficient": 1.0,
        "available_at": "2024-11-29T14:25Z",
        "verification": "known fabricated fixture before open",
    }
    meta = metadata(observed_at=now.isoformat(), current_session_actions=actions)
    raw = raw_fixture(calendar)
    completed_raw = raw[
        raw.date.dt.tz_localize("America/New_York")
        .dt.tz_convert("UTC")
        .add(pd.Timedelta(minutes=30))
        .le(now)
    ].reset_index(drop=True)
    bars = normalize_monitor_subbars(
        completed_raw, calendar, meta, as_of=now, sampling_interval_minutes=60
    )
    assert len(bars) == 9
    assert bars.attrs["completed_subbars_within_unfinished_hour_excluded"] == 1
    daily = pd.DataFrame(
        {
            "session": ["2024-11-27"],
            "close": [bars[bars.session.eq("2024-11-27")].close.iloc[-1]],
            "dividend_amount": [0.0],
            "split_coefficient": [1.0],
        }
    )
    report = validate_monitor_inputs(
        daily, bars, calendar, meta, as_of=now, sampling_interval_minutes=60
    )
    assert report["completed_daily_sessions"] == 1
    with pytest.raises(DataContractError, match="coverage missing"):
        validate_research_inputs(daily, bars, calendar, as_of=now, sampling_interval_minutes=60)
    packet = tmp_path / "monitor_packet"
    manifest = prepare_monitor_packet(
        packet, daily, bars, calendar, meta, as_of=now, sampling_interval_minutes=60
    )
    assert manifest["schema"] == "qqq_hourly_v5_monitor_packet_v1"
    assert len(read_monitor_packet(packet)[1]) == 9
    with pytest.raises(DataContractError, match="unsupported packet"):
        read_input_packet(packet)
    with pytest.raises(DataContractError, match="separately verified"):
        validate_monitor_inputs(
            daily,
            bars,
            calendar,
            replace(meta, current_session_actions=None),
            as_of=now,
            sampling_interval_minutes=60,
        )
    late_actions = {**actions, "available_at": "2024-11-29T15:00Z"}
    with pytest.raises(DataContractError, match="known by the session open"):
        validate_monitor_inputs(
            daily,
            bars,
            calendar,
            replace(meta, current_session_actions=late_actions),
            as_of=now,
            sampling_interval_minutes=60,
        )
    future_raw = pd.concat([completed_raw, raw.iloc[len(completed_raw) : len(completed_raw) + 1]])
    with pytest.raises(DataContractError, match="unfinished"):
        normalize_monitor_subbars(
            future_raw, calendar, meta, as_of=now, sampling_interval_minutes=60
        )


def test_cash_observations_preserve_holiday_availability_ties_with_explicit_policy():
    cash = pd.DataFrame(
        {
            "available_at": ["2000-01-04T23:00Z"] * 2,
            "observation_date": ["1999-12-30", "1999-12-31"],
            "annual_rate": [0.052, 0.053],
        }
    )
    result = validate_cash_observations(cash, as_of=AS_OF)
    assert len(result) == 2  # No source rows silently discarded.
    assert result.attrs["availability_tie_policy"] == "latest_observation_date_at_same_available_at"
    with pytest.raises(DataContractError, match="requires observation_date"):
        validate_cash_observations(cash.drop(columns="observation_date"), as_of=AS_OF)
    with pytest.raises(DataContractError, match="unique and ordered"):
        validate_cash_observations(cash.iloc[::-1], as_of=AS_OF)


def test_frozen_calendar_dependency_change_requires_explicit_reaudit():
    calendar = build_xnys_calendar("2024-11-27", "2024-11-29")
    calendar.attrs["calendar_provenance"] = {
        **calendar.attrs["calendar_provenance"],
        "package_version": "0.0.unverified",
    }
    with pytest.raises(DataContractError, match="dependency version changed"):
        validate_calendar(calendar)


def test_completed_monitor_partial_tail_retained_for_fill_mark_not_confirmation():
    from trend_following.research_hourly_data_v5 import normalize_monitor_subbars

    calendar = build_xnys_calendar("2024-11-27", "2024-11-29")
    now = pd.Timestamp("2024-11-29T18:01Z")
    meta = metadata(observed_at=now.isoformat())
    bars = normalize_monitor_subbars(raw_fixture(calendar), calendar, meta, as_of=now)
    tail = bars.iloc[-1]
    assert tail.bar_start == pd.Timestamp("2024-11-29T17:30Z")
    assert tail.bar_end == pd.Timestamp("2024-11-29T18:00Z")
    assert tail.duration_minutes == 30 and not tail.is_full_hour
    assert tail.volume == 100


def test_alpha_only_audit_stops_at_primary_cutoff_and_never_reads_other_provider_or_downloads(
    tmp_path, monkeypatch
):
    import requests

    from trend_following.research_hourly_data_v5 import audit_alpha_only_cache

    root = tmp_path / "source"
    whole_calendar = build_xnys_calendar("2026-05-26", "2026-06-01")
    whole = raw_fixture(whole_calendar)
    whole["session"] = whole.date.dt.strftime("%Y-%m-%d")
    daily = (
        whole.groupby("session")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
        .rename(columns={"session": "date"})
    )
    daily["date"] = pd.to_datetime(daily.date)
    daily[["open", "high", "low", "close"]] *= 2
    daily["adj_close"] = daily.close / 2
    daily["dividend_amount"] = 0.0
    daily["split_coefficient"] = 1.0
    primary = whole[whole.session.le("2026-05-28")].drop(columns="session").reset_index(drop=True)
    secondary_source = primary[primary.date.dt.strftime("%Y-%m-%d").le("2026-05-27")]
    fifteen_rows = []
    for row in secondary_source.itertuples():
        for minute in (0, 15):
            fifteen_rows.append(
                {
                    "date": row.date + pd.Timedelta(minutes=minute),
                    "open": row.open if minute == 0 else (row.open + row.close) / 2,
                    "high": row.high,
                    "low": row.low,
                    "close": (row.open + row.close) / 2 if minute == 0 else row.close,
                    "volume": row.volume / 2,
                }
            )
    frames = {
        "alpha_vantage_daily_adjusted": daily,
        "alpha_vantage_30min": primary,
        "alpha_vantage_15min": pd.DataFrame(fifteen_rows),
    }
    paths = []
    for directory, frame in frames.items():
        path = root / "data/raw" / directory / "QQQ.parquet"
        path.parent.mkdir(parents=True)
        frame.to_parquet(path, index=False)
        paths.append(path)
    original = {path: path.read_bytes() for path in paths}
    read_paths = []
    original_reader = pd.read_parquet

    def only_alpha_prices(path, *args, **kwargs):
        path = str(path)
        assert "yfinance" not in path and "60min" not in path
        assert "alpha_vantage_" in path
        read_paths.append(path)
        return original_reader(path, *args, **kwargs)

    def no_network(*args, **kwargs):
        raise AssertionError("Alpha-only audit must use existing files without downloads")

    monkeypatch.setattr(pd, "read_parquet", only_alpha_prices)
    monkeypatch.setattr(requests, "get", no_network)
    report, gaps, calendar = audit_alpha_only_cache(root, bar_start_session="2026-05-26")
    assert report["cutoff_session"] == "2026-05-28"
    assert report["terminal_session_complete"]
    assert calendar.session.iloc[-1] == "2026-05-28"
    assert report["secondary_15min_alignment"]["source_end_session"] == "2026-05-27"
    assert gaps.empty
    assert not report["historical_ready"] and not report["prepared_raw_packet_generated"]
    assert report["daily"]["source_rows_after_cutoff_excluded_from_analysis"] == 2
    assert all(source["provider"] == "AlphaVantage" for source in report["source_files"])
    assert (
        report["reconstruction_candidate"]["required_study_session_quality"]["fields"]["close"][
            "mismatch_count"
        ]
        == 0
    )
    assert len(read_paths) == 3
    assert {path: path.read_bytes() for path in paths} == original


def test_daily_source_attestation_independent_preserved_and_never_inferred_alpha(tmp_path):
    daily, bars, calendar = prepared_fixture()
    original = prepare_input_packet(
        tmp_path / "unknown_daily", daily, bars, calendar, metadata(), as_of=AS_OF
    )
    assert "daily_source_metadata" not in original
    unknown = read_input_packet(tmp_path / "unknown_daily")[0]
    assert "source_metadata" not in unknown.attrs
    declaration = {
        "provider": "explicit_fabricated_daily_not_Alpha",
        "symbol": "QQQ",
        "price_basis": "raw_as_traded",
        "verification": "fabricated daily-source proof",
        "observed_at": AS_OF.isoformat(),
    }
    declared = prepare_input_packet(
        tmp_path / "declared_daily",
        daily,
        bars,
        calendar,
        metadata(),
        as_of=AS_OF,
        daily_source_metadata=declaration,
    )
    assert declared["daily_source_metadata"] == declaration
    loaded_daily, loaded_bars, loaded_cal, _ = read_input_packet(tmp_path / "declared_daily")
    assert loaded_daily.attrs["source_metadata"] == declaration
    preserved = prepare_input_packet(
        tmp_path / "prefix_provenance",
        loaded_daily.copy(),
        loaded_bars.copy(),
        loaded_cal.copy(),
        metadata(),
        as_of=AS_OF,
    )
    assert preserved["daily_source_metadata"]["provider"] == "explicit_fabricated_daily_not_Alpha"
    with pytest.raises(DataContractError, match="source symbols"):
        prepare_input_packet(
            tmp_path / "mismatched_symbol",
            daily,
            bars,
            calendar,
            metadata(),
            as_of=AS_OF,
            daily_source_metadata={**declaration, "symbol": "OTHER"},
        )
    with pytest.raises(DataContractError, match="raw as-traded"):
        prepare_input_packet(
            tmp_path / "adjusted_daily",
            daily,
            bars,
            calendar,
            metadata(),
            as_of=AS_OF,
            daily_source_metadata={**declaration, "price_basis": "adjusted"},
        )


def test_half_hour_grid_every_regular_and_early_close_bar_eligible():
    from trend_following.research_hourly_data_v5 import (
        canonical_half_hour_grid,
        canonical_interval_grid,
    )

    calendar = build_xnys_calendar("2024-11-27", "2024-11-29")
    grid = canonical_half_hour_grid(calendar)
    assert len(grid[grid.session.eq("2024-11-27")]) == 13
    assert len(grid[grid.session.eq("2024-11-29")]) == 7
    assert grid.duration_minutes.eq(30).all()
    assert not grid.is_full_hour.any()  # Legacy clock-hour flag isn't observation eligibility.
    assert grid[grid.session.eq("2024-11-27")].bar_end.iloc[0] == pd.Timestamp("2024-11-27T15:00Z")
    assert grid[grid.session.eq("2024-11-27")].bar_start.iloc[-1] == pd.Timestamp(
        "2024-11-27T20:30Z"
    )
    assert grid[grid.session.eq("2024-11-27")].bar_end.iloc[-1] == pd.Timestamp("2024-11-27T21:00Z")
    with pytest.raises(DataContractError, match="explicitly 30 or 60"):
        canonical_interval_grid(calendar, 45)


def test_half_hour_raw_quotes_unaggregated_and_packet_mode_explicit(tmp_path):
    import json

    calendar = build_xnys_calendar("2024-11-27", "2024-11-29")
    raw = raw_fixture(calendar)
    bars = normalize_raw_subbars(
        raw, calendar, metadata(), as_of=AS_OF, sampling_interval_minutes=30
    )
    assert len(bars) == len(raw) == 20
    for column in ("open", "high", "low", "close", "volume"):
        assert np.array_equal(bars[column], raw[column])
    daily = pd.DataFrame(
        {
            "session": calendar.session,
            "close": bars.groupby("session").close.last().to_numpy(),
            "dividend_amount": 0.0,
            "split_coefficient": 1.0,
        }
    )
    with pytest.raises(DataContractError, match="grid mismatch"):
        validate_research_inputs(daily, bars, calendar, as_of=AS_OF, sampling_interval_minutes=60)
    audit = validate_research_inputs(
        daily, bars, calendar, as_of=AS_OF, sampling_interval_minutes=30
    )
    assert audit["sampling_interval_minutes"] == audit["observation_interval_minutes"] == 30
    assert audit["full_observations"] == 20 and audit["partial_marking_bars"] == 0
    assert audit["full_hours"] == 0
    packet = tmp_path / "half_hour_packet"
    manifest = prepare_input_packet(
        packet, daily, bars, calendar, metadata(), as_of=AS_OF, sampling_interval_minutes=30
    )
    assert (
        manifest["sampling_interval_minutes"]
        == manifest["audit"]["sampling_interval_minutes"]
        == 30
    )
    loaded = read_input_packet(packet)
    assert len(loaded[1]) == 20
    assert loaded[1].attrs["sampling_interval_minutes"] == 30
    changed = json.loads((packet / "manifest.json").read_text())
    changed["sampling_interval_minutes"] = 60
    (packet / "manifest.json").write_text(json.dumps(changed))
    with pytest.raises(DataContractError, match="top-level interval"):
        read_input_packet(packet)


def test_hour_mode_packet_remains_generic_readable_and_cannot_be_declared_half_hour(tmp_path):
    import json

    daily, bars, calendar = prepared_fixture()
    with pytest.raises(DataContractError, match="grid mismatch"):
        prepare_input_packet(
            tmp_path / "false_halfhour",
            daily,
            bars,
            calendar,
            metadata(),
            as_of=AS_OF,
            sampling_interval_minutes=30,
        )
    packet = tmp_path / "legacy_hour_mode"
    manifest = prepare_input_packet(
        packet, daily, bars, calendar, metadata(), as_of=AS_OF, sampling_interval_minutes=60
    )
    assert len(read_input_packet(packet)[1]) == 11
    # Historical generic packet-schema compatibility is explicit 60m default,
    # never inferred from row spacing; new-study guards separately require30.
    for key in ("sampling_interval_minutes", "observation_interval_minutes"):
        manifest.pop(key)
        manifest["audit"].pop(key)
    manifest["audit"].pop("full_observations")
    (packet / "manifest.json").write_text(json.dumps(manifest))
    assert read_input_packet(packet)[1].attrs["sampling_interval_minutes"] == 60


def test_first_completed_half_hour_monitor_is_usable_even_with_unfinished_clock_hour(tmp_path):
    from trend_following.research_hourly_data_v5 import (
        normalize_monitor_subbars,
        prepare_monitor_packet,
        read_monitor_packet,
        validate_monitor_inputs,
    )

    calendar = build_xnys_calendar("2024-11-27", "2024-11-29")
    now = pd.Timestamp("2024-11-29T15:00Z")  # Exactly10:00 NY: first30m is complete.
    actions = {
        "session": "2024-11-29",
        "dividend_amount": 0.0,
        "split_coefficient": 1.0,
        "available_at": "2024-11-29T14:25Z",
        "verification": "fabricated verified before-open actions",
    }
    meta = metadata(observed_at=now.isoformat(), current_session_actions=actions)
    raw = raw_fixture(calendar)
    ends = (
        raw.date.dt.tz_localize("America/New_York")
        .dt.tz_convert("UTC")
        .add(pd.Timedelta(minutes=30))
    )
    raw = raw.loc[ends.le(now)].reset_index(drop=True)
    bars = normalize_monitor_subbars(raw, calendar, meta, as_of=now)
    assert len(bars) == 14
    assert bars.bar_end.iloc[-1] == now
    assert bars.attrs["completed_subbars_within_unfinished_hour_excluded"] == 0
    assert bars.duration_minutes.eq(30).all()
    daily = pd.DataFrame(
        {
            "session": ["2024-11-27"],
            "close": [bars[bars.session.eq("2024-11-27")].close.iloc[-1]],
            "dividend_amount": [0.0],
            "split_coefficient": [1.0],
        }
    )
    audit = validate_monitor_inputs(daily, bars, calendar, meta, as_of=now)
    assert audit["full_observations"] == 14 and audit["sampling_interval_minutes"] == 30
    packet = tmp_path / "first_halfhour_monitor"
    manifest = prepare_monitor_packet(packet, daily, bars, calendar, meta, as_of=now)
    assert manifest["observation_interval_minutes"] == 30
    assert (
        manifest["monitor_normalization"]["completed_partial_bar_role"]
        == "every_completed_30min_bar_is_confirmation_eligible"
    )
    assert len(read_monitor_packet(packet)[1]) == 14
    with pytest.raises(DataContractError, match="grid mismatch"):
        validate_monitor_inputs(
            daily, bars, calendar, meta, as_of=now, sampling_interval_minutes=60
        )
