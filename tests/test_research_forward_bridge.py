"""Fictional, clock-controlled fixtures only; no production forward observations."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from trend_following import research_forward as journal
from trend_following import research_forward_bridge as bridge
from trend_following.research_protocol import sha256_file


@pytest.fixture
def env(tmp_path, monkeypatch):
    clock = [pd.Timestamp("2026-10-02T21:01:00Z")]
    monkeypatch.setattr(journal, "_utc_now", lambda: clock[0])
    prototype = Path(__file__).resolve().parents[1] / "configs/qqq_forward_paper_v1.yaml"
    contract = yaml.safe_load(prototype.read_text())
    freeze = tmp_path / "freeze"
    freeze.mkdir()
    (freeze / "snapshot").mkdir()
    calendar = tmp_path / "calendar.csv"
    calendar.write_text(
        "market_open,market_close\n2026-10-02T13:30:00Z,2026-10-02T20:00:00Z\n"
        "2026-10-05T13:30:00Z,2026-10-05T20:00:00Z\n"
        "2026-10-06T13:30:00Z,2026-10-06T17:00:00Z\n"
        "2026-10-07T13:30:00Z,2026-10-07T20:00:00Z\n"
    )
    dates = pd.date_range(end="2026-10-02T20:00:00Z", periods=1900, freq="h")
    qqq = 100 + np.arange(1900) ** 2 * 0.0001
    files = {"exchange_calendar": calendar}
    for symbol, value in [("QQQ", qqq), ("TQQQ", np.full(1900, 50.0))]:
        path = tmp_path / f"{symbol}.parquet"
        pd.DataFrame(
            {"date": dates - pd.Timedelta(hours=1), "bar_end": dates, "close": value}
        ).to_parquet(path)
        files[symbol.lower() + "_warmup"] = path
    metadata = tmp_path / "metadata.json"
    metadata.write_text('{"fixture":"fictional"}')
    files["warmup_metadata"] = metadata
    data = []
    for key, path in files.items():
        contract[key] = {"path": path.name, "sha256": sha256_file(path)}
        snapshot = freeze / "snapshot" / path.name
        shutil.copyfile(path, snapshot)
        data.append({**contract[key], "snapshot_path": "snapshot/" + path.name})
    pp = freeze / "protocol.yaml"
    pp.write_text(yaml.safe_dump(contract))
    (freeze / "dirty.patch").write_text("")
    manifest = {
        "stage": "final_code",
        "prospective_eligible": True,
        "protocol_id": contract["protocol_id"],
        "frozen_at_utc": "2026-10-02T21:00:00Z",
        "genuine_holdout_start_utc": "2026-10-05T13:30:00Z",
        "code_files": [],
        "data_files": data,
        "protocol_sha256": sha256_file(pp),
        "parameters_sha256": journal._digest(contract),
        "dirty_tree_diff": "dirty.patch",
        "dirty_tree_diff_sha256": sha256_file(freeze / "dirty.patch"),
        "exchange_calendar": {"path": str(calendar), "sha256": sha256_file(calendar)},
    }
    mp = freeze / "manifest.json"
    mp.write_text(json.dumps(manifest))
    ledger = tmp_path / "paper.jsonl"
    common = (tmp_path, pp, mp, ledger)
    bridge.initialize_paper_bridge(*common)
    return {
        "root": tmp_path,
        "contract": pp,
        "manifest": mp,
        "ledger": ledger,
        "common": common,
        "clock": clock,
        "config": contract,
    }


def source(env, n, *, qqq=None, risk=None, mutate=None):
    starts = list(pd.date_range("2026-10-05T13:30:00Z", periods=7, freq="h"))
    starts += list(pd.date_range("2026-10-06T13:30:00Z", periods=4, freq="h"))
    starts += list(pd.date_range("2026-10-07T13:30:00Z", periods=7, freq="h"))
    start = starts[n]
    end = min(
        start + pd.Timedelta(hours=1),
        pd.Timestamp(
            "2026-10-05T20:00:00Z"
            if n < 7
            else ("2026-10-06T17:00:00Z" if n < 11 else "2026-10-07T20:00:00Z")
        ),
    )
    env["clock"][0] = end + pd.Timedelta(seconds=30)
    value = {
        "schema_version": 1,
        "provider": "Yahoo Finance via yfinance",
        "interval": "60m",
        "retrieved_at": (end + pd.Timedelta(seconds=20)).isoformat(),
        "bars": {},
    }
    for symbol, price in [
        ("QQQ", qqq if qqq is not None else 100 + (1900 + n) ** 2 * 0.0001),
        ("TQQQ", risk if risk is not None else 50 + n),
    ]:
        value["bars"][symbol] = {
            "bar_start": start.isoformat(),
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 1000,
            "dividends": 0,
            "stock_splits": 0,
        }
    if mutate:
        mutate(value)
    path = env["root"] / f"source_{n}.json"
    path.write_text(json.dumps(value))
    return path


def record(env, n, *, acknowledge=True, **kwargs):
    return bridge.record_completed_bar(
        *env["common"], source(env, n, **kwargs), acknowledge_fill=acknowledge
    )


def verify(env):
    return bridge.verify_paper_bridge(*env["common"])


def rewrite_chain(path, mutate):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(records)
    previous = journal.ZERO_HASH
    for i, row in enumerate(records):
        row["sequence"] = i
        row["previous_hash"] = previous
        if "event" in row:
            row["event"]["expected_previous_hash"] = previous
        row.pop("record_hash", None)
        row["record_hash"] = journal._digest(row)
        previous = row["record_hash"]
    path.write_bytes(b"".join(journal._canonical(r) + b"\n" for r in records))


def test_header_and_warmup_are_flat_no_inherited_trade(env):
    initial = verify(env)
    assert initial["bars"] == initial["events"] == 0
    assert initial["state"]["equity"] == 1000
    assert initial["state"]["positions"] == {}
    assert initial["engine_state"]["entry_price"] is None
    assert initial["engine_state"]["trade_id"] == 0
    first = record(env, 0)
    assert [e["event_type"] for e in first["events"]] == [
        "equity_mark",
        "signal_available",
        "hypothetical_order",
    ]
    assert first["state"]["equity"] == 1000
    assert first["state"]["positions"] == {}
    assert verify(env)["engine_state"]["entry_price"] is None


def test_delayed_fill_uses_future_quote_not_future_sized_order_and_costs(env):
    first = record(env, 0, risk=50)
    order = first["events"][-1]
    assert "requested_quantities" not in order
    assert order["allocation_order"]["target_weight"] == 1
    second = record(env, 1, risk=100)
    fill = next(e for e in second["events"] if e["event_type"] == "fill")
    assert fill["fills"][0]["quantity"] == pytest.approx(1000 / 1.0006 / 100)
    assert second["state"]["equity"] == pytest.approx(1000 / 1.0006)
    assert second["state"]["cash"] == pytest.approx(0)
    assert fill["costs"]["transaction"] == pytest.approx((1000 / 1.0006) / 10000)
    assert fill["costs"]["slippage"] == pytest.approx((1000 / 1.0006) * 5 / 10000)
    assert pd.Timestamp(order["observation_timestamp"]) < pd.Timestamp(order["decision_timestamp"])
    assert pd.Timestamp(order["order_timestamp"]) < pd.Timestamp(fill["fill_timestamp"])
    assert "return_timestamp" not in fill
    third = record(env, 2, risk=110)
    assert third["state"]["equity"] == pytest.approx((1000 / 1.0006) * 1.1)
    assert verify(env)["engine_state"]["entry_price"] == pytest.approx(100 * 1.0006)
    assert verify(env)["bars"] == 3


def test_missed_fill_explicit_and_never_imports_fictional_trade(env):
    record(env, 0)
    result = record(env, 1, acknowledge=False)
    assert result["halted"]
    assert [e["event_type"] for e in result["events"]] == ["equity_mark", "missed_fill"]
    state = verify(env)
    assert state["state"]["positions"] == {}
    assert state["engine_state"]["entry_price"] is None
    assert state["engine_state"]["trade_id"] == 0
    with pytest.raises(ValueError, match="halted"):
        record(env, 2)


def test_missing_next_bar_and_duplicate_fail_without_appending(env):
    record(env, 0)
    original = env["ledger"].read_bytes()
    with pytest.raises(ValueError, match="expected next"):
        record(env, 2)
    assert env["ledger"].read_bytes() == original
    with pytest.raises(ValueError, match="expected next"):
        record(env, 0)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("dividends", 0.01, "Corporate action"),
        ("stock_splits", 2, "Corporate action"),
        ("close", 0, "OHLCV"),
        ("high", float("nan"), "nonfinite"),
    ],
)
def test_unsupported_actions_and_invalid_quotes_fail_closed(env, field, value, match):
    before = env["ledger"].read_bytes()
    with pytest.raises(ValueError, match=match):
        record(env, 0, mutate=lambda x: x["bars"]["TQQQ"].update({field: value}))
    assert env["ledger"].read_bytes() == before


def test_stale_incomplete_or_unavailable_feed_fails(env):
    p = source(env, 0)
    env["clock"][0] += pd.Timedelta(minutes=16)
    with pytest.raises(ValueError, match="stale"):
        bridge.record_completed_bar(*env["common"], p)
    with pytest.raises(ValueError, match="Incomplete"):
        record(env, 0, mutate=lambda x: x.update(retrieved_at="2026-10-05T14:29:59Z"))
    with pytest.raises(ValueError, match="provider"):
        record(env, 0, mutate=lambda x: x.update(provider="unsupported"))


def test_source_mutation_detected_on_replay(env):
    record(env, 0)
    p = env["root"] / "source_0.json"
    p.write_text(p.read_text() + " ")
    with pytest.raises(ValueError, match="snapshot changed"):
        verify(env)


def test_recomputed_policy_rejects_forged_checkpoint_even_with_rehashed_chain(env):
    record(env, 0)
    rewrite_chain(
        env["ledger"],
        lambda rows: rows[-1]["event"]["policy_state"].update(engine_state_sha256="a" * 64),
    )
    with pytest.raises(ValueError, match="replayed"):
        verify(env)


def test_recomputed_policy_rejects_arbitrary_target_even_with_valid_hashes(env):
    record(env, 0)
    rewrite_chain(
        env["ledger"], lambda rows: rows[-1]["event"]["allocation_order"].update(target_weight=0.5)
    )
    with pytest.raises(ValueError, match="replayed"):
        verify(env)


def test_partial_batch_never_silently_repaired(env):
    record(env, 0)
    raw = env["ledger"].read_bytes().splitlines(keepends=True)
    env["ledger"].write_bytes(b"".join(raw[:-1]))
    with pytest.raises(ValueError, match="Partial"):
        verify(env)


def test_new_data_cannot_change_prior_event_or_signal(env):
    record(env, 0)
    old = env["ledger"].read_bytes()
    record(env, 1, qqq=2000, risk=150)
    assert env["ledger"].read_bytes().startswith(old)
    assert verify(env)["bars"] == 2


def test_seven_normal_bars_last_half_hour_and_early_close_no_padding(env):
    for n in range(18):
        result = record(env, n)
        if n in (6, 10, 17):
            proof = result["events"][-1]["policy_state"]
            assert pd.Timestamp(proof["bar_end"]) - pd.Timestamp(
                proof["bar_start"]
            ) == pd.Timedelta(minutes=30)
    assert verify(env)["bars"] == 18
    assert verify(env)["next_bar_start"] is None
    records = [json.loads(x) for x in env["ledger"].read_text().splitlines()][1:]
    dates = [
        pd.Timestamp(r["event"]["fill_timestamp"]).tz_convert("America/New_York").date()
        for r in records
        if r["event"]["event_type"] == "fill"
    ]
    assert len(dates) == len(set(dates))


def test_warmup_uses_immutable_archive_not_mutating_original(env):
    (env["root"] / "QQQ.parquet").write_bytes(b"mutable download replaced")
    record(env, 0)
    assert verify(env)["bars"] == 1


def test_contract_cannot_change_without_new_freeze(env):
    env["contract"].write_text(env["contract"].read_text() + "\n#changed\n")
    with pytest.raises(ValueError, match="protocol snapshot changed"):
        verify(env)


def test_phase_availability_proof_is_recomputed(env):
    record(env, 0)

    def change(rows):
        for row in rows[1:]:
            row["event"]["policy_state"]["decision_timestamp"] = "2026-10-05T14:30:00Z"

    rewrite_chain(env["ledger"], change)
    with pytest.raises(ValueError, match="signal availability"):
        verify(env)


def test_session_defer_cancellation_and_next_session_exit_follow_actual_fills(env):
    record(env, 0, risk=50)
    record(env, 1, risk=50)
    trim = record(env, 2, risk=210)
    trim_order = next(e for e in trim["events"] if e["event_type"] == "hypothetical_order")
    assert trim_order["allocation_order"]["target_weight"] == 0.75
    assert not any(e["event_type"] == "fill" for e in trim["events"])
    stop = record(env, 3, risk=120)
    assert not stop["halted"]
    cancelled = next(e for e in stop["events"] if e["event_type"] == "missed_fill")
    assert cancelled["order_id"] == trim_order["event_id"]
    assert cancelled["reason"] == "policy_replaced_before_session_eligible_fill"
    exit_order = next(e for e in stop["events"] if e["event_type"] == "hypothetical_order")
    assert exit_order["allocation_order"]["target_weight"] == 0
    for n in (4, 5, 6):
        assert not any(e["event_type"] == "fill" for e in record(env, n, risk=120)["events"])
    exited = record(env, 7, risk=100)
    fill = next(e for e in exited["events"] if e["event_type"] == "fill")
    assert fill["order_id"] == exit_order["event_id"]
    assert fill["fills"][0]["quantity"] < 0
    assert exited["state"]["positions"]["TQQQ"] == 0
    assert exited["state"]["cash"] == pytest.approx(1000 / 1.0006 * 2 * (1 - 0.0006))
    assert verify(env)["engine_state"]["entry_price"] is None


def test_batch_prevalidation_failure_leaves_no_partial_append(env, monkeypatch):
    original = env["ledger"].read_bytes()
    validate = journal._validate_event

    def refuse_order(event, *args):
        if event["event_type"] == "hypothetical_order":
            raise ValueError("test-only rejected order")
        return validate(event, *args)

    monkeypatch.setattr(journal, "_validate_event", refuse_order)
    with pytest.raises(ValueError, match="rejected order"):
        record(env, 0)
    assert env["ledger"].read_bytes() == original


def test_arbitrary_general_recorder_event_is_not_policy_evidence(env):
    result = record(env, 0)
    event = copy.deepcopy(result["events"][1])
    event["policy_state"] = {"held_trade": "user asserted"}
    rewrite_chain(
        env["ledger"], lambda rows: rows[1]["event"].update(policy_state=event["policy_state"])
    )
    with pytest.raises(ValueError, match="arbitrary policy"):
        verify(env)


@pytest.mark.parametrize("value", [0.1, float("nan")])
def test_capital_gain_distribution_cannot_be_silently_dropped(env, value):
    with pytest.raises(ValueError, match="capital gain"):
        record(env, 0, mutate=lambda x: x["bars"]["TQQQ"].update(capital_gains=value))


def test_trim_then_restore_sizes_from_actual_costed_cash_and_drifted_shares(env, monkeypatch):
    # Isolate qtrim/accounting from the independent volatility overlay in this unit fixture.
    compute_features = bridge.causal_features

    def low_vol_features(*args, **kwargs):
        result = compute_features(*args, **kwargs)
        result["vol_percentile"] = 0.0
        return result

    monkeypatch.setattr(bridge, "causal_features", low_vol_features)
    record(env, 0, risk=50)
    entry = record(env, 1, risk=50)
    old_qty = entry["state"]["positions"]["TQQQ"]
    signal = record(env, 2, qqq=800, risk=110)
    target = next(e for e in signal["events"] if e["event_type"] == "hypothetical_order")
    assert target["allocation_order"]["target_weight"] == 0.5
    for n in (3, 4, 5, 6):
        record(env, n, qqq=800, risk=110)
    trimmed = record(env, 7, qqq=800, risk=120)
    sell = next(e for e in trimmed["events"] if e["event_type"] == "fill")
    expected_notional = -0.5 * old_qty * 120
    assert sell["fills"][0]["quantity"] == pytest.approx(expected_notional / 120)
    assert trimmed["state"]["cash"] == pytest.approx(-expected_notional * 0.9994)
    restored_signal = record(env, 8, qqq=450, risk=120)
    order = next(e for e in restored_signal["events"] if e["event_type"] == "hypothetical_order")
    assert order["allocation_order"]["target_weight"] == 1
    for n in (9, 10):
        record(env, n, qqq=450, risk=120)
    cash = trimmed["state"]["cash"]
    restored = record(env, 11, qqq=450, risk=130)
    buy = next(e for e in restored["events"] if e["event_type"] == "fill")
    assert buy["fills"][0]["quantity"] == pytest.approx(cash / 1.0006 / 130)
    assert restored["state"]["cash"] == pytest.approx(0)
    assert verify(env)["bars"] == 12


@pytest.mark.parametrize("field", ["quantity", "transaction", "cash"])
def test_replay_rejects_rehashed_quantity_cost_or_cash_tampering(env, field):
    record(env, 0)
    record(env, 1)

    def tamper(rows):
        event = next(r["event"] for r in rows[1:] if r["event"]["event_type"] == "fill")
        if field == "quantity":
            event["fills"][0]["quantity"] *= 0.9
        elif field == "transaction":
            event["costs"]["transaction"] = 0
        else:
            event["state"]["cash"] += 1

    rewrite_chain(env["ledger"], tamper)
    with pytest.raises(ValueError, match="replayed"):
        verify(env)


def test_even_newly_hash_bound_contract_cannot_change_sizing_convention(env):
    contract = yaml.safe_load(env["contract"].read_text())
    contract["execution"]["sizing_convention"] = "post_cost_target_weight"
    env["contract"].write_text(yaml.safe_dump(contract))
    manifest = json.loads(env["manifest"].read_text())
    manifest["protocol_sha256"] = sha256_file(env["contract"])
    manifest["parameters_sha256"] = journal._digest(contract)
    env["manifest"].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="sizing_convention"):
        verify(env)


def test_initial_header_persists_canonical_flat_checkpoint_and_freeze_binding(env):
    header = json.loads(env["ledger"].read_text())
    proof = header["policy_state"]
    assert proof["contract_sha256"] == sha256_file(env["contract"])
    assert proof["manifest_sha256"] == header["manifest_sha256"] == sha256_file(env["manifest"])
    assert proof["engine_state_sha256"] == journal._digest(proof["engine_state"])
    assert proof["financial_state_sha256"] == journal._digest(header["state"])
    assert proof["engine_state"]["weight"] == 0
    assert proof["engine_state"]["pending"] is None
    assert proof["engine_state"]["entry_price"] is None
    assert proof["engine_state"]["bar"] == -1
    assert proof["bars"] == 0 and verify(env)["events"] == 0


@pytest.mark.parametrize("change", ["trade", "empty", "binding"])
def test_initial_checkpoint_is_recomputed_not_trusted_even_when_rehashed(env, change):
    def mutate(rows):
        proof = rows[0]["policy_state"]
        if change == "empty":
            rows[0]["policy_state"] = {}
        elif change == "binding":
            proof["contract_sha256"] = "f" * 64
        else:
            proof["engine_state"]["entry_price"] = 50
            proof["engine_state"]["trade_id"] = 1
            proof["engine_state_sha256"] = journal._digest(proof["engine_state"])

    rewrite_chain(env["ledger"], mutate)
    with pytest.raises(ValueError, match="Initial bridge checkpoint"):
        verify(env)
