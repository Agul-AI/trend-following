"""Synthetic bounded-reader tests: never import or evaluate actual market data."""

import json

import pandas as pd
import pytest

from trend_following import research_alpha_reconciliation_v5 as data
from trend_following import research_constant_cash as fixed
from trend_following import research_intersection_validation as intersection
from trend_following import research_quality_protocol_v5 as protocol
from trend_following import research_validation_prefix as prefix


def _daily(session="2018-12-31", price=100.0):
    return {
        "date": session,
        "open": price,
        "high": price,
        "low": price,
        "close": price,
        "adj_close": price,
        "volume": 1000.0,
        "dividend_amount": 0.0,
        "split_coefficient": 1.0,
        "session": session,
        "daily_price_available_at": pd.Timestamp(
            session + " 17:00", tz="America/New_York"
        ).tz_convert("UTC"),
    }


def _csv(tmp_path, name, rows):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_2019_poison_numbers_are_never_converted(tmp_path, monkeypatch):
    rows = [_daily("2018-12-31"), _daily("2019-01-02", "DO_NOT_PARSE_THIS_NUMBER")]
    path = _csv(tmp_path, "daily.csv", rows)
    convert = prefix._convert
    calls = []

    def tripwire(value, column):
        assert value != "DO_NOT_PARSE_THIS_NUMBER"
        calls.append(column)
        return convert(value, column)

    monkeypatch.setattr(prefix, "_convert", tripwire)
    monkeypatch.setattr(pd, "read_csv", lambda *a, **kw: pytest.fail("Full CSV loading forbidden"))
    result = prefix._stream_frame(path, "daily.csv", end_session=prefix.END_SESSION)
    assert result.session.tolist() == ["2018-12-31"]
    assert len(calls) == len(result.columns)
    assert result.close.tolist() == [100.0]


def test_first_future_marker_stops_before_later_invalid_marker(tmp_path):
    path = _csv(
        tmp_path,
        "daily.csv",
        [_daily(), _daily("2019-01-02", "poison"), {**_daily(), "session": "invalid"}],
    )
    result = prefix._stream_frame(path, "daily.csv", end_session=prefix.END_SESSION)
    assert len(result) == 1


@pytest.mark.parametrize("price", ["not_a_number", float("nan"), float("inf"), float("-inf")])
def test_included_nonfinite_or_invalid_price_is_rejected(tmp_path, price):
    path = _csv(tmp_path, "daily.csv", [_daily(price=price)])
    with pytest.raises(prefix.Error, match="numeric field"):
        prefix._stream_frame(path, "daily.csv", end_session=prefix.END_SESSION)


@pytest.mark.parametrize(
    "sessions", [["2018-12-31", "2018-12-28"], ["2018-12-31", "2018-12-31"], ["1999-10-29"]]
)
def test_prefix_rejects_unsorted_duplicate_or_pre_initialization_dates(tmp_path, sessions):
    path = _csv(tmp_path, "daily.csv", [_daily(session) for session in sessions])
    with pytest.raises(prefix.Error, match="Nonchronological"):
        prefix._stream_frame(path, "daily.csv", end_session=prefix.END_SESSION)


def test_wrong_or_duplicated_header_rejected(tmp_path):
    path = _csv(tmp_path, "daily.csv", [{**_daily(), "future_trade_return": 1.0}])
    with pytest.raises(prefix.Error, match="columns"):
        prefix._stream_frame(path, "daily.csv", end_session=prefix.END_SESSION)
    path.write_text("session,session\n2018-12-31,2018-12-31\n")
    with pytest.raises(prefix.Error, match="columns"):
        prefix._stream_frame(path, "daily.csv", end_session=prefix.END_SESSION)


def test_blackout_future_row_ignored_and_cross_endpoint_rejected(tmp_path):
    fields = {
        "start": "2018-12-31 14:30:00+00:00",
        "end": "2018-12-31 15:00:00+00:00",
        "kind": "unknown_missing",
        "reason": "synthetic gap",
        "source_url": "",
    }
    future = {**fields, "start": "2019-01-02 14:30:00+00:00", "end": "POISON_TIMESTAMP"}
    path = _csv(tmp_path, "blackouts.csv", [fields, future])
    assert len(prefix._stream_frame(path, "blackouts.csv", end_session=prefix.END_SESSION)) == 1
    path = _csv(tmp_path, "blackouts.csv", [{**fields, "end": "2019-01-02 15:00:00+00:00"}])
    with pytest.raises(prefix.Error, match="crosses"):
        prefix._stream_frame(path, "blackouts.csv", end_session=prefix.END_SESSION)


def _synthetic_frames():
    session = "2018-12-31"
    opening = pd.Timestamp(session + " 09:30", tz="America/New_York").tz_convert("UTC")
    closing = pd.Timestamp(session + " 16:00", tz="America/New_York").tz_convert("UTC")
    calendar = pd.DataFrame(
        [
            {
                "session": session,
                "market_open": opening,
                "market_close": closing,
                "is_early_close": False,
            }
        ]
    )
    bars = pd.DataFrame(
        [
            {
                "session": session,
                "bar_start": at,
                "bar_end": at + pd.Timedelta(minutes=30),
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 1000.0,
                "volume_reported": 1000.0,
                "duration_minutes": 30.0,
                "is_full_hour": False,
                "status": "observed",
                "signal_eligible": True,
                "fill_eligible": True,
            }
            for at in pd.date_range(opening, closing, freq="30min", inclusive="left")
        ]
    )
    coverage = bars[list(data.COVERAGE_BINDING_COLUMNS)].copy()
    blackouts = pd.DataFrame(columns=data.BLACKOUT_BINDING_COLUMNS)
    return {
        "daily.csv": pd.DataFrame([_daily()]),
        "bars.csv": bars,
        "calendar.csv": calendar,
        "coverage.csv": coverage,
        "blackouts.csv": blackouts,
    }


def _proof(frames):
    return {
        "data_bindings": {
            **{
                key: data._session_hashes(frames[name], columns)
                for name, (key, columns) in prefix.BINDINGS.items()
            },
            "blackout_row_hashes": [],
        }
    }


def test_every_original_included_session_and_row_is_required():
    frames = _synthetic_frames()
    proof = _proof(frames)
    for value in proof["data_bindings"].values():
        if isinstance(value, dict):
            value["2019-01-02"] = "a" * 64
    bindings = prefix._verify_rows(frames, proof, end_session=prefix.END_SESSION)
    assert all("2019-01-02" not in binding for binding in bindings.values())
    changed = {name: frame.copy() for name, frame in frames.items()}
    changed["daily.csv"].loc[0, "close"] = 101.0
    with pytest.raises(prefix.Error, match="changed"):
        prefix._verify_rows(changed, proof, end_session=prefix.END_SESSION)
    omitted = {name: frame.copy() for name, frame in frames.items()}
    omitted["bars.csv"] = omitted["bars.csv"].iloc[:-1]
    with pytest.raises(prefix.Error, match="changed"):
        prefix._verify_rows(omitted, proof, end_session=prefix.END_SESSION)


def test_structural_early_close_and_complete_grid_preserved(monkeypatch):
    frames = _synthetic_frames()
    monkeypatch.setattr(prefix, "START_SESSION", prefix.END_SESSION)
    audit = prefix._validate_frames(frames, {}, prefix.END_SESSION)
    assert audit["validation_sessions"] == 1
    assert audit["OOS_numeric_rows_parsed"] == 0
    frames["calendar.csv"].loc[0, "market_close"] -= pd.Timedelta(hours=3)
    frames["calendar.csv"].loc[0, "is_early_close"] = True
    with pytest.raises(data.DataContractError):
        prefix._validate_frames(frames, {}, prefix.END_SESSION)


def test_omitted_scheduled_slot_rejected_even_if_quote_rows_exist(monkeypatch):
    frames = _synthetic_frames()
    monkeypatch.setattr(prefix, "START_SESSION", prefix.END_SESSION)
    frames["coverage.csv"] = frames["coverage.csv"].iloc[:-1]
    with pytest.raises(prefix.Error, match="omits scheduled"):
        prefix._validate_frames(frames, {}, prefix.END_SESSION)


def test_unsealed_selection_blocks_before_any_metadata_or_price_read(tmp_path, monkeypatch):
    source = tmp_path / "original"
    source.mkdir()
    monkeypatch.setattr(
        prefix, "preseal_metadata", lambda *a: pytest.fail("Unsealed candidate read metadata")
    )
    monkeypatch.setattr(
        prefix, "_stream_frame", lambda *a, **k: pytest.fail("Unsealed candidate read prices")
    )
    with pytest.raises(prefix.Error, match="sealed artifact missing"):
        prefix.prepare_validation_prefix(
            source,
            tmp_path / "new-prefix",
            selection_seal=tmp_path / "absent.json",
            expected_selection_sha256="a" * 64,
        )


@pytest.mark.parametrize("location", ["existing", "inside", "ancestor"])
def test_output_refuses_overwrite_or_protected_tree(tmp_path, location):
    source = tmp_path / "original"
    source.mkdir()
    outputs = {"existing": source, "inside": source / "new-prefix", "ancestor": tmp_path}
    with pytest.raises(prefix.Error, match="new and outside"):
        prefix.prepare_validation_prefix(
            source,
            outputs[location],
            selection_seal=tmp_path / "absent.json",
            expected_selection_sha256="a" * 64,
        )


def test_verify_hashes_only_new_prefix_without_parsing_any_csv(tmp_path, monkeypatch):
    root = tmp_path / "prepared"
    root.mkdir()
    for name in prefix.PACKET_FILES:
        (root / name).write_text("not numerical; verify bytes only\n")
    manifest = {
        "schema": "trend_following_validation_2018_prefix_v1",
        "end_session": prefix.END_SESSION,
        "stage": "validation",
        "historical_oos_authorized": False,
        "candidate_set_sealed_at": "2026-10-08T12:00:00+00:00",
        "prepared_at": "2026-10-08T12:00:01+00:00",
        "files": {
            name: {
                "sha256": protocol.sha256_file(root / name),
                "bytes": (root / name).stat().st_size,
            }
            for name in prefix.PACKET_FILES
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(
        prefix, "_stream_frame", lambda *a, **k: pytest.fail("Verifier parsed a CSV")
    )
    assert (
        prefix.verify_validation_prefix(
            root, expected_manifest_sha256=protocol.canonical_hash(manifest)
        )
        == manifest
    )
    (root / "bars.csv").write_text("changed\n")
    with pytest.raises(prefix.Error, match="bytes changed"):
        prefix.verify_validation_prefix(
            root, expected_manifest_sha256=protocol.canonical_hash(manifest)
        )


def _selection(tmp_path, **changes):
    pairs = sorted(
        [
            {
                "pair_id": intersection._pair_id(candidate),
                "entry_rule": candidate.entry_rule.to_dict(),
                "exit_rule": candidate.exit_rule.to_dict(),
            }
            for candidate in fixed.candidates_for_wait(2.0)
        ],
        key=lambda pair: pair["pair_id"],
    )[:159]
    candidates = [
        {"candidate_id": candidate.candidate_id, **candidate.to_dict()}
        for hours in (2.0, 3.0, 4.0)
        for candidate in intersection.candidates_from_pairs(pairs, hours)
    ]
    source = tmp_path / "original"
    source.mkdir(exist_ok=True)
    value = {
        "schema": "trend_following_resplit_intersection_validation_v1",
        "stage": "validation",
        "period": {"start": "2012-01-01", "end": "2018-12-31"},
        "historical_oos_period": {"start": "2019-01-01", "end": "2026-05-28"},
        "historical_oos_authorized": False,
        "validation_only": True,
        "configuration_count_per_wait": 159,
        "configuration_count": 477,
        "wait_hours": [2.0, 3.0, 4.0],
        "source_study": str(source),
        "config_path": "unused.yaml",
        "pairs": pairs,
        "pairs_sha256": protocol.canonical_hash(pairs),
        "candidates": candidates,
        "candidate_set_sha256": protocol.canonical_hash(candidates),
        "candidate_set_sealed_at": "2026-10-08T12:00:00+00:00",
        **changes,
    }
    path = tmp_path / "selection.json"
    protocol._seal(path, value)
    return source, path, value


def test_selection_requires_exact_sealed_159_by_three_canonical_grid(tmp_path):
    source, path, value = _selection(tmp_path)
    assert prefix._selection_guard(path, protocol.canonical_hash(value), source) == value
    changed = {**value, "candidates": [dict(row) for row in value["candidates"]]}
    changed["candidates"][0]["exit_confirmation_hours"] = 3.0
    changed["candidate_set_sha256"] = protocol.canonical_hash(changed["candidates"])
    corrupt_path = tmp_path / "altered.json"
    protocol._seal(corrupt_path, changed)
    with pytest.raises(prefix.Error, match="canonical pairs"):
        prefix._selection_guard(corrupt_path, protocol.canonical_hash(changed), source)


@pytest.mark.parametrize(
    "changes",
    [
        {"historical_oos_authorized": True},
        {"historical_oos_authorized": 0},
        {"configuration_count": 476},
        {"wait_hours": [2.0, 3.0, 6.0]},
        {"period": {"start": "2012-01-01", "end": "2026-05-28"}},
    ],
)
def test_selection_rejects_wrong_scope_counts_waits_or_endpoint(tmp_path, changes):
    source, path, value = _selection(tmp_path, **changes)
    with pytest.raises(prefix.Error, match="BEFORE prices"):
        prefix._selection_guard(path, protocol.canonical_hash(value), source)
