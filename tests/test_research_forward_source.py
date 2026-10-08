from __future__ import annotations

import pandas as pd
import pytest

from trend_following.research_forward_source import latest_closed_bar, source_row


def test_completed_bar_clocks_and_half_day():
    calendar = pd.DataFrame(
        {
            "market_open": ["2026-10-05T13:30Z", "2026-11-27T14:30Z"],
            "market_close": ["2026-10-05T20:00Z", "2026-11-27T18:00Z"],
        }
    )
    holdout = pd.Timestamp("2026-10-05T13:30Z")
    with pytest.raises(ValueError, match="no completed"):
        latest_closed_bar(calendar, pd.Timestamp("2026-10-05T14:29Z"), holdout)
    start, end = latest_closed_bar(calendar, pd.Timestamp("2026-10-05T14:35Z"), holdout)
    assert start == pd.Timestamp("2026-10-05T13:30Z")
    assert end == pd.Timestamp("2026-10-05T14:30Z")
    start, end = latest_closed_bar(calendar, pd.Timestamp("2026-11-27T18:05Z"), holdout)
    assert start == pd.Timestamp("2026-11-27T17:30Z")
    assert end == pd.Timestamp("2026-11-27T18:00Z")
    with pytest.raises(ValueError, match="stale"):
        latest_closed_bar(calendar, pd.Timestamp("2026-11-27T18:16Z"), holdout)


def test_source_rejects_missing_quotes_actions_and_price_errors():
    start = pd.Timestamp("2026-10-05T13:30Z")
    frame = pd.DataFrame(
        {
            "Open": [100],
            "High": [102],
            "Low": [99],
            "Close": [101],
            "Volume": [10],
            "Dividends": [0],
            "Stock Splits": [0],
        },
        index=[start],
    )
    assert source_row(frame, start)["close"] == 101
    with pytest.raises(ValueError, match="missing"):
        source_row(frame, start + pd.Timedelta(hours=1))
    frame["Dividends"] = 0.5
    with pytest.raises(ValueError, match="corporate"):
        source_row(frame, start)
    frame["Dividends"] = 0
    frame["Low"] = 103
    with pytest.raises(ValueError, match="OHLC"):
        source_row(frame, start)


def test_collector_writes_source_immutably_and_rejects_late_retrieval(tmp_path, monkeypatch):
    import json

    import yaml

    from trend_following.research_forward_source import collect_snapshot
    from trend_following.research_protocol import sha256_file

    frozen = tmp_path / "freeze"
    (frozen / "snapshot").mkdir(parents=True)
    calendar = frozen / "snapshot" / "calendar.csv"
    pd.DataFrame(
        {"market_open": ["2026-10-05T13:30Z"], "market_close": ["2026-10-05T20:00Z"]}
    ).to_csv(calendar, index=False)
    protocol = frozen / "protocol.yaml"
    protocol.write_text(
        yaml.safe_dump(
            {
                "bars_per_day": 7,
                "provider": "Yahoo Finance via yfinance",
                "exchange_calendar": {"path": "calendar.csv", "sha256": sha256_file(calendar)},
            }
        )
    )
    manifest = frozen / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "stage": "final_code",
                "prospective_eligible": True,
                "protocol_id": "qqq_forward_paper_v1",
                "protocol_sha256": sha256_file(protocol),
                "genuine_holdout_start_utc": "2026-10-05T13:30Z",
                "data_files": [
                    {
                        "path": "calendar.csv",
                        "snapshot_path": "snapshot/calendar.csv",
                        "sha256": sha256_file(calendar),
                    }
                ],
                "code_files": [],
            }
        )
    )
    monkeypatch.setattr(
        "trend_following.research_forward_source.verify_frozen_inputs", lambda *a: None
    )
    start = pd.Timestamp("2026-10-05T13:30Z")
    frame = pd.DataFrame(
        {
            "Open": [100],
            "High": [102],
            "Low": [99],
            "Close": [101],
            "Volume": [10],
            "Dividends": [0],
            "Stock Splits": [0],
        },
        index=[start],
    )
    source = tmp_path / "fresh.json"
    result = collect_snapshot(
        tmp_path,
        manifest,
        source,
        loader=lambda symbol: frame,
        clock=lambda: pd.Timestamp("2026-10-05T14:32Z"),
    )
    payload = json.loads(source.read_text())
    assert set(payload["bars"]) == {"QQQ", "TQQQ"}
    assert payload["bars"]["QQQ"]["close"] == 101
    assert result["sha256"] == sha256_file(source)
    with pytest.raises(FileExistsError):
        collect_snapshot(
            tmp_path,
            manifest,
            source,
            loader=lambda symbol: frame,
            clock=lambda: pd.Timestamp("2026-10-05T14:32Z"),
        )
    late = tmp_path / "late.json"
    ticks = iter([pd.Timestamp("2026-10-05T14:32Z"), pd.Timestamp("2026-10-05T14:46Z")])
    with pytest.raises(ValueError, match="retrieval"):
        collect_snapshot(
            tmp_path, manifest, late, loader=lambda symbol: frame, clock=lambda: next(ticks)
        )
    assert not late.exists()


def test_collector_rejects_reordered_calendar_and_boolean_quotes():
    holdout = pd.Timestamp("2026-10-05T13:30Z")
    rows = pd.DataFrame(
        {
            "market_open": ["2026-10-06T13:30Z", "2026-10-05T13:30Z"],
            "market_close": ["2026-10-06T20:00Z", "2026-10-05T20:00Z"],
        }
    )
    with pytest.raises(ValueError, match="calendar"):
        latest_closed_bar(rows, pd.Timestamp("2026-10-06T14:32Z"), holdout)
    row = pd.DataFrame(
        {
            "Open": [True],
            "High": [102],
            "Low": [0.5],
            "Close": [101],
            "Volume": [10],
            "Dividends": [0],
            "Stock Splits": [0],
        },
        index=[holdout],
    )
    with pytest.raises(ValueError, match="boolean"):
        source_row(row, holdout)
    row["Open"] = 100
    row["Capital Gains"] = 0.5
    with pytest.raises(ValueError, match="capital-gain"):
        source_row(row, holdout)
    row["Capital Gains"] = 0
    assert source_row(row, holdout)["capital_gains"] == 0
