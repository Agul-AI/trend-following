"""Artificial fixture checks; no market downloads or performance claims."""

import json

import numpy as np
import pandas as pd
import pytest

from trend_following.research_oos_accounting import normalize_oos_actions
from trend_following.research_oos_data import (
    coverage_report,
    expected_bars,
    file_hash,
    merge_packages,
    normalize_yahoo_frames,
    read_input_package,
    verify_package_derivation,
    write_input_package,
)


def fixture_inputs():
    days = ["2024-10-01", "2024-10-02", "2024-10-03"]
    cal = pd.DataFrame(
        {
            "session": days,
            "market_open": [
                pd.Timestamp(x + " 09:30", tz="America/New_York").tz_convert("UTC") for x in days
            ],
            "market_close": [
                pd.Timestamp(x + " 16:00", tz="America/New_York").tz_convert("UTC") for x in days
            ],
            "is_early_close": False,
        }
    )
    grid = expected_bars(cal, days[0], days[-1])
    hourly, daily = {}, {}
    for asset in ["QQQ", "TQQQ"]:
        hourly[asset] = pd.DataFrame(
            {
                "Close": np.arange(len(grid)) + 100.0,
                "Dividends": 0.0,
                "Stock Splits": 0.0,
                "Capital Gains": 0.0,
            },
            index=pd.DatetimeIndex(grid.bar_start),
        )
        daily[asset] = pd.DataFrame(
            {
                "Close": [106.0, 113.0, 120.0],
                "Dividends": 0.0,
                "Stock Splits": 0.0,
                "Capital Gains": 0.0,
            },
            index=pd.DatetimeIndex([pd.Timestamp(x, tz="America/New_York") for x in days]),
        )
    reference = pd.DataFrame(columns=["asset", "ex_date", "pay_date", "source_url"])
    return cal, hourly, daily, reference


def production_package(tmp_path, name="input", *, change=0.0, missing=False):
    cal, hourly, daily, reference = fixture_inputs()
    if missing:
        for asset in hourly:
            hourly[asset] = hourly[asset].drop(hourly[asset].index[3])
    for asset in hourly:
        hourly[asset]["Close"] += change
    observed = pd.Timestamp("2024-10-03 21:00Z") + pd.Timedelta(seconds=change)
    source = tmp_path / (name + "_raw")
    source.mkdir()
    files, acquisition = {}, {}
    for asset in hourly:
        for key, frame in [("hourly", hourly[asset]), ("daily", daily[asset])]:
            frame.to_parquet(source / f"{asset}_{key}.parquet")
            files[f"{asset}_{key}.parquet"] = source / f"{asset}_{key}.parquet"
        info = {
            "observed_at_utc": observed.isoformat(),
            "provider": "artificial fixture",
            "yfinance_version": "test",
            "request": {
                "period": "2y",
                "intervals": ["60m", "1d"],
                "auto_adjust": False,
                "back_adjust": False,
                "actions": True,
                "prepost": False,
                "repair": False,
            },
        }
        path = source / f"{asset}_metadata.json"
        path.write_text(json.dumps(info))
        acquisition[asset] = info
        files[path.name] = path
    ref = source / "distribution_calendar.csv"
    reference.to_csv(ref, index=False)
    files[ref.name] = ref
    packaged = observed + pd.Timedelta(seconds=5)
    prices, actions, basis = normalize_yahoo_frames(
        hourly, daily, cal, reference, observed_at=packaged
    )
    target = tmp_path / name
    write_input_package(
        target,
        prices,
        actions,
        cal,
        observed_at=packaged,
        provenance={
            **basis,
            "adapter": "qqq_oos_v2_Yahoo_AV_full_session_v1",
            "source_acquisition": acquisition,
            "fallback_supplied": False,
        },
        raw_files=files,
    )
    return target, cal


def test_complete_production_source_replay(tmp_path):
    package, cal = production_package(tmp_path)
    assert verify_package_derivation(package, cal)["verified"]
    prices, _, meta = read_input_package(package)
    assert coverage_report(prices, cal, as_of=pd.Timestamp(meta["observed_at_utc"]))["complete"]


def test_missing_prices_not_padded(tmp_path):
    package, _ = production_package(tmp_path, missing=True)
    _, _, meta = read_input_package(package)
    assert meta["quality"]["missing_count"] == 1
    assert not meta["quality"]["complete"]


@pytest.mark.parametrize("field", ["Stock Splits", "Dividends", "Capital Gains"])
def test_missing_daily_action_row_refuses(field):
    cal, hourly, daily, ref = fixture_inputs()
    daily["QQQ"] = daily["QQQ"].drop(daily["QQQ"].index[1])
    with pytest.raises(ValueError, match="daily action source"):
        normalize_yahoo_frames(
            hourly, daily, cal, ref, observed_at=pd.Timestamp("2024-10-03 21:00Z")
        )


def test_hourly_nonzero_action_disagreement_refuses():
    cal, hourly, daily, ref = fixture_inputs()
    hourly["QQQ"].iloc[7, hourly["QQQ"].columns.get_loc("Dividends")] = 1.0
    with pytest.raises(ValueError, match="hourly action disagrees"):
        normalize_yahoo_frames(
            hourly, daily, cal, ref, observed_at=pd.Timestamp("2024-10-03 21:00Z")
        )


def test_split_restores_raw_prices_exactly_once():
    cal, hourly, daily, ref = fixture_inputs()
    for asset in hourly:
        hourly[asset].loc[hourly[asset].index[:7], "Close"] = 50.0
        daily[asset].iloc[1, daily[asset].columns.get_loc("Stock Splits")] = 2.0
        hourly[asset].iloc[7, hourly[asset].columns.get_loc("Stock Splits")] = 2.0
    prices, actions, _ = normalize_yahoo_frames(
        hourly, daily, cal, ref, observed_at=pd.Timestamp("2024-10-03 21:00Z")
    )
    assert prices.iloc[0].tolist() == [100.0, 100.0]
    assert prices.iloc[7].tolist() == [107.0, 107.0]
    assert len(normalize_oos_actions(actions)) == 2


def test_distribution_ex_date_is_receivable_pay_cash_1600():
    cal, hourly, daily, ref = fixture_inputs()
    daily["QQQ"].iloc[1, daily["QQQ"].columns.get_loc("Dividends")] = 0.5
    hourly["QQQ"].iloc[7, hourly["QQQ"].columns.get_loc("Dividends")] = 0.5
    ref = pd.DataFrame(
        [
            {
                "asset": "QQQ",
                "ex_date": "2024-10-02",
                "pay_date": "2024-10-03",
                "source_url": "https://www.invesco.com/",
            }
        ]
    )
    _, actions, _ = normalize_yahoo_frames(
        hourly, daily, cal, ref, observed_at=pd.Timestamp("2024-10-03 21:00Z")
    )
    action = actions.iloc[0]
    assert action.effective_at == pd.Timestamp("2024-10-02 13:30Z")
    assert action.pay_at == pd.Timestamp("2024-10-03 20:00Z")
    assert action.character == "unknown"


def test_missing_verified_payment_date_refuses():
    cal, hourly, daily, ref = fixture_inputs()
    daily["QQQ"].iloc[1, daily["QQQ"].columns.get_loc("Dividends")] = 0.5
    with pytest.raises(ValueError, match="missing verified payment"):
        normalize_yahoo_frames(
            hourly, daily, cal, ref, observed_at=pd.Timestamp("2024-10-03 21:00Z")
        )


def test_full_session_fallback_replaces_both_symbols():
    cal, hourly, daily, ref = fixture_inputs()
    hourly["QQQ"] = hourly["QQQ"].drop(hourly["QQQ"].index[3])
    subs, factors, proof = {}, {}, {}
    for asset in hourly:
        starts = pd.date_range(
            cal.market_open.iloc[0], cal.market_close.iloc[0], freq="30min", inclusive="left"
        )
        subs[asset] = pd.DataFrame(
            {
                "date": starts.tz_convert("America/New_York").tz_localize(None),
                "close": np.arange(len(starts)) + 50.0,
            }
        )
        factors[asset] = pd.DataFrame(
            {"date": ["2024-10-01"], "close": [100.0], "adj_close": [50.0]}
        )
        proof[asset] = subs[asset].copy()
        proof[asset]["date"] += pd.Timedelta(minutes=15)
    prices, _, provenance = normalize_yahoo_frames(
        hourly,
        daily,
        cal,
        ref,
        observed_at=pd.Timestamp("2024-10-03 21:00Z"),
        fallback_30min=subs,
        fallback_daily=factors,
        fallback_15min=proof,
    )
    assert prices.iloc[:7].QQQ.tolist() == [102.0, 106.0, 110.0, 114.0, 118.0, 122.0, 124.0]
    assert prices.iloc[:7].TQQQ.equals(prices.iloc[:7].QQQ.rename("TQQQ"))
    assert len(provenance["full_session_fallbacks"]) == 2
    subs["QQQ"] = subs["QQQ"].drop(2)
    with pytest.raises(ValueError, match="missing complete observed subbars"):
        normalize_yahoo_frames(
            hourly,
            daily,
            cal,
            ref,
            observed_at=pd.Timestamp("2024-10-03 21:00Z"),
            fallback_30min=subs,
            fallback_daily=factors,
            fallback_15min=proof,
        )


@pytest.mark.parametrize("mutation", ["change", "extra", "symlink"])
def test_package_inventory_integrity(tmp_path, mutation):
    package, _ = production_package(tmp_path)
    if mutation == "change":
        (package / "prices.csv").write_text("changed")
    if mutation == "extra":
        (package / "unlisted.txt").write_text("extra")
    if mutation == "symlink":
        (package / "link").symlink_to(package / "prices.csv")
    with pytest.raises(ValueError):
        read_input_package(package)


def test_rehashed_derived_prices_do_not_pass_source_replay(tmp_path):
    package, cal = production_package(tmp_path)
    path = package / "prices.csv"
    d = pd.read_csv(path)
    d["QQQ"] += 1.0
    d.to_csv(path, index=False)
    m = json.loads((package / "metadata.json").read_text())
    for r in m["files"]:
        if r["path"] == "prices.csv":
            r.update(sha256=file_hash(path), bytes=path.stat().st_size)
    (package / "metadata.json").write_text(json.dumps(m))
    with pytest.raises((AssertionError, ValueError)):
        verify_package_derivation(package, cal)


def test_first_vintage_price_revisions_reported_not_replaced(tmp_path):
    a, _ = production_package(tmp_path, "a")
    b, _ = production_package(tmp_path, "b", change=1.0)
    p, _, info = merge_packages([a, b])
    assert p.iloc[0].QQQ == 100.0
    assert len(info["overlap_revisions"]) == 21
    with pytest.raises(ValueError, match="retrieval vintage"):
        merge_packages([b, a])


def _inject_action(package, *, kind="distribution"):
    frame = pd.read_csv(package / "actions.csv")
    action = {
        "action_id": "QQQ:2024-10-02:distribution",
        "asset": "QQQ",
        "kind": kind,
        "effective_at": "2024-10-02T13:30:00Z",
        "available_at": "2024-10-02T13:30:00Z",
        "split_ratio": 1.0,
        "amount_per_share": 0.5,
        "pay_at": "2024-10-03T20:00:00Z",
        "character": "unknown",
        "availability_basis": "assumed_public_by_ex_date",
        "source_observed_at": "2024-10-03T21:00:00Z",
    }
    frame = pd.DataFrame([action]) if frame.empty else pd.concat([frame, pd.DataFrame([action])])
    path = package / "actions.csv"
    frame.to_csv(path, index=False)
    m = json.loads((package / "metadata.json").read_text())
    for r in m["files"]:
        if r["path"] == "actions.csv":
            r.update(sha256=file_hash(path), bytes=path.stat().st_size)
    (package / "metadata.json").write_text(json.dumps(m))


def test_late_added_action_cannot_rewrite_frozen_history(tmp_path):
    a, _ = production_package(tmp_path, "a")
    b, _ = production_package(tmp_path, "b", change=1.0)
    _inject_action(b)
    with pytest.raises(ValueError, match="late-added"):
        merge_packages([a, b])


def test_withdrawn_action_cannot_rewrite_first_vintage(tmp_path):
    a, _ = production_package(tmp_path, "a")
    b, _ = production_package(tmp_path, "b", change=1.0)
    _inject_action(a)
    with pytest.raises(ValueError, match="withdrawn"):
        merge_packages([a, b])


def test_generic_normalized_package_is_not_production(tmp_path):
    p, cal = production_package(tmp_path)
    m = json.loads((p / "metadata.json").read_text())
    m["provenance"]["adapter"] = "unknown"
    (p / "metadata.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="reviewed source adapter"):
        verify_package_derivation(p, cal)
