"""Prospective journal gates use a fixed clock; no fabricated production events."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from trend_following import research_forward as forward


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    clock = [pd.Timestamp("2026-10-02T13:10:00Z")]
    monkeypatch.setattr(forward, "_utc_now", lambda: clock[0])
    code = tmp_path / "code.py"
    code.write_text("x = 1\n")
    snapshot = tmp_path / "freeze"
    snapshot.mkdir()
    protocol = {"protocol_id": "test_forward", "strategy": {"timezone": "America/New_York"}}
    pp = snapshot / "protocol.yaml"
    pp.write_text(yaml.safe_dump(protocol))
    diff = snapshot / "dirty.patch"
    diff.write_text("")
    calendar = tmp_path / "sessions.csv"
    calendar.write_text("market_open\n2026-10-02T14:00:00Z\n2026-10-05T14:00:00Z\n")
    manifest = {
        "stage": "final_code",
        "prospective_eligible": True,
        "protocol_id": "test_forward",
        "frozen_at_utc": "2026-10-02T13:00:00Z",
        "genuine_holdout_start_utc": "2026-10-02T14:00:00Z",
        "code_files": [{"path": "code.py", "sha256": sha(code)}],
        "data_files": [],
        "protocol_sha256": sha(pp),
        "parameters_sha256": hashlib.sha256(
            json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "dirty_tree_diff": "dirty.patch",
        "dirty_tree_diff_sha256": sha(diff),
        "exchange_calendar": {"path": str(calendar), "sha256": sha(calendar)},
    }
    mp = snapshot / "manifest.json"
    mp.write_text(json.dumps(manifest))
    source = tmp_path / "new_source_snapshot.json"
    source.write_text('{"QQQ": 100.0}')
    ledger = tmp_path / "forward.jsonl"
    header = forward.initialize_forward_ledger(tmp_path, mp, ledger, initial_cash=1000.0)
    return {
        "root": tmp_path,
        "manifest": mp,
        "ledger": ledger,
        "header": header,
        "clock": clock,
        "source": source,
        "code": code,
    }


def event(env, kind="signal_available", event_id="s1", time="2026-10-02T14:00:00Z"):
    head = forward.verify_forward_ledger(env["root"], env["manifest"], env["ledger"])
    return {
        "event_type": kind,
        "event_id": event_id,
        "protocol_id": "test_forward",
        "paper_only": True,
        "expected_previous_hash": head["head_hash"],
        "event_timestamp": time,
        "price_timestamp": time,
        "observation_timestamp": time,
        "policy_state": {"held_trade": None},
        "source": {
            "provider": "test_fixture_not_market_data",
            "path": str(env["source"]),
            "sha256": sha(env["source"]),
            "retrieved_at": time,
        },
        "state": {"cash": 1000.0, "positions": {}, "prices": {"QQQ": 100.0}, "equity": 1000.0},
        "costs": {"transaction": 0.0, "slippage": 0.0, "financing": 0.0, "tax": 0.0},
    }


def append(env, value):
    return forward.append_forward_event(env["root"], env["manifest"], env["ledger"], value)


def test_pending_header_does_not_claim_performance_and_never_overwrites(setup):
    env = setup
    head = forward.verify_forward_ledger(env["root"], env["manifest"], env["ledger"])
    assert head["events"] == 0
    assert head["state"]["cash"] == 1000
    assert "no_performance_claim" in env["header"]["status"]
    with pytest.raises(ValueError, match="already initialized"):
        forward.initialize_forward_ledger(env["root"], env["manifest"], env["ledger"])


def test_prospective_signal_order_fill_and_mark_reconcile_account(setup):
    env = setup
    env["clock"][0] = pd.Timestamp("2026-10-02T14:00:00Z")
    append(env, event(env))
    env["clock"][0] += pd.Timedelta(seconds=1)
    order = event(env, "hypothetical_order", "o1", "2026-10-02T14:00:01Z")
    order.update(
        signal_id="s1",
        requested_quantities={"QQQ": 5.0},
        observation_timestamp="2026-10-02T14:00:00Z",
        decision_timestamp="2026-10-02T14:00:00.5Z",
        order_timestamp=order["event_timestamp"],
    )
    append(env, order)
    env["clock"][0] = pd.Timestamp("2026-10-02T15:00:00Z")
    fill = event(env, "fill", "f1", "2026-10-02T15:00:00Z")
    fill.update(
        order_id="o1",
        observation_timestamp=order["observation_timestamp"],
        decision_timestamp=order["decision_timestamp"],
        order_timestamp=order["order_timestamp"],
        fill_timestamp=fill["event_timestamp"],
        fills=[{"symbol": "QQQ", "quantity": 5.0, "price": 100.0}],
    )
    fill["costs"]["transaction"] = 1.0
    fill["state"] = {
        "cash": 499.0,
        "positions": {"QQQ": 5.0},
        "prices": {"QQQ": 100.0},
        "equity": 999.0,
    }
    append(env, fill)
    env["clock"][0] = pd.Timestamp("2026-10-02T16:00:00Z")
    mark = event(env, "equity_mark", "m1", "2026-10-02T16:00:00Z")
    mark["return_timestamp"] = mark["event_timestamp"]
    mark["state"] = {
        "cash": 499.0,
        "positions": {"QQQ": 5.0},
        "prices": {"QQQ": 110.0},
        "equity": 1049.0,
    }
    append(env, mark)
    final = forward.verify_forward_ledger(env["root"], env["manifest"], env["ledger"])
    assert final["events"] == 4 and final["state"]["equity"] == 1049.0


@pytest.mark.parametrize(
    "when,match",
    [
        ("2026-10-02T13:59:59Z", "holdout"),
        ("2026-10-02T14:00:01Z", "future"),
    ],
)
def test_preholdout_or_future_timestamps_rejected(setup, when, match):
    env = setup
    env["clock"][0] = pd.Timestamp("2026-10-02T14:00:00Z")
    with pytest.raises(ValueError, match=match):
        append(env, event(env, time=when))


def test_old_bars_and_retrospective_flag_rejected(setup):
    env = setup
    env["clock"][0] = pd.Timestamp("2026-10-02T15:00:00Z")
    with pytest.raises(ValueError, match="Old/backfilled"):
        append(env, event(env))
    current = event(env, time="2026-10-02T15:00:00Z")
    current["retrospective"] = True
    with pytest.raises(ValueError, match="prospective"):
        append(env, current)


def test_every_append_rechecks_frozen_code_and_source_hash(setup):
    env = setup
    env["clock"][0] = pd.Timestamp("2026-10-02T14:00:00Z")
    e = event(env)
    e["source"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="source snapshot"):
        append(env, e)
    e = event(env)
    env["code"].write_text("x = 2\n")
    with pytest.raises(ValueError, match="Frozen input changed"):
        append(env, e)


def test_state_reconciliation_and_cash_continuity_fail_closed(setup):
    env = setup
    env["clock"][0] = pd.Timestamp("2026-10-02T14:00:00Z")
    e = event(env)
    e["state"]["equity"] = 1001.0
    with pytest.raises(ValueError, match="reconcile"):
        append(env, e)
    e["state"]["cash"] = 1001.0
    with pytest.raises(ValueError, match="Cash continuity"):
        append(env, e)


def test_lineage_requires_recorded_signal_and_order(setup):
    env = setup
    env["clock"][0] = pd.Timestamp("2026-10-02T14:00:00Z")
    order = event(env, "hypothetical_order")
    order.update(
        signal_id="not_recorded",
        decision_timestamp=order["event_timestamp"],
        order_timestamp=order["event_timestamp"],
    )
    with pytest.raises(ValueError, match="signal_id"):
        append(env, order)
    fill = event(env, "fill")
    fill["order_id"] = "not_recorded"
    with pytest.raises(ValueError, match="order_id"):
        append(env, fill)


def test_stale_head_and_duplicate_ids_rejected(setup):
    env = setup
    env["clock"][0] = pd.Timestamp("2026-10-02T14:00:00Z")
    e = event(env)
    stale = copy.deepcopy(e)
    stale["event_id"] = "s2"
    append(env, e)
    with pytest.raises(ValueError, match="Stale"):
        append(env, stale)
    with pytest.raises(ValueError, match="Duplicate"):
        append(env, e)


def test_tampered_or_torn_ledger_detected_without_rewriting(setup):
    env = setup
    original = env["ledger"].read_bytes()
    env["ledger"].write_bytes(original.replace(b"1000.0", b"1001.0"))
    with pytest.raises(ValueError, match="hash mismatch"):
        forward.verify_forward_ledger(env["root"], env["manifest"], env["ledger"])
    env["ledger"].write_bytes(original[:-1])
    with pytest.raises(ValueError, match="Truncated"):
        forward.verify_forward_ledger(env["root"], env["manifest"], env["ledger"])
    assert env["ledger"].read_bytes() == original[:-1]


def test_protocol_only_manifest_cannot_initialize_forward_recording(setup):
    env = setup
    manifest = json.loads(env["manifest"].read_text())
    manifest["stage"] = "protocol_only"
    env["manifest"].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="final_code"):
        forward.initialize_forward_ledger(
            env["root"], env["manifest"], env["root"] / "another.jsonl"
        )


def test_changed_manifest_rejected_even_if_still_self_consistent(setup):
    env = setup
    manifest = json.loads(env["manifest"].read_text())
    manifest["extra_metadata"] = "new"
    env["manifest"].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Manifest changed"):
        forward.verify_forward_ledger(env["root"], env["manifest"], env["ledger"])


def test_optional_initial_policy_proof_preserves_generic_default_and_binds_object(setup):
    env = setup
    assert env["header"]["policy_state"] == {}
    path = env["root"] / "another_pending.jsonl"
    proof = {"model": {"trade_id": 0}, "status": "pending"}
    header = forward.initialize_forward_ledger(
        env["root"], env["manifest"], path, initial_cash=1000, policy_state=proof
    )
    proof["model"]["trade_id"] = 99
    assert header["policy_state"]["model"]["trade_id"] == 0
    verified = forward.verify_forward_ledger(env["root"], env["manifest"], path)
    assert verified["events"] == 0
    assert verified["policy_state"] == {"model": {"trade_id": 0}, "status": "pending"}


def test_optional_initial_policy_proof_rejects_non_object_before_creating_header(setup):
    env = setup
    path = env["root"] / "invalid_pending.jsonl"
    with pytest.raises(ValueError, match="policy_state"):
        forward.initialize_forward_ledger(env["root"], env["manifest"], path, policy_state=[])
    assert not path.exists()
