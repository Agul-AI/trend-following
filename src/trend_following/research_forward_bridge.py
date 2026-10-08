"""Manually advance a fixed, frozen B5 PAPER policy from immutable live evidence.

No network, scheduler, broker, historical fill backfill, or free-form policy state.
Every verification replays the model from feature-only warmup and flat holdings.
The seven-observation Yahoo cadence is a NEW experiment, not the six-bar backtest.
Missing bars, unsupported corporate actions and partial writes stop this pilot;
an unacknowledged eligible paper fill is logged as missed and stops the contract.
Hash/source timestamps remain local attestations, not independent notarization.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from trend_following import research_forward as journal
from trend_following.research_execution import (
    ExecutionPolicyEngine,
    causal_features,
    normalize_spec,
)
from trend_following.research_protocol import sha256_file

BRIDGE_VERSION = "manual_fixed_b5_v1"
PHASES = ("observation_timestamp", "decision_timestamp", "order_timestamp")
ZERO_COSTS = {"transaction": 0.0, "slippage": 0.0, "financing": 0.0, "tax": 0.0}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (pd.Timestamp, dt.date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _engine_state(engine: ExecutionPolicyEngine) -> dict[str, Any]:
    # Model's gross diagnostic is not the costed account; account is journal.state.
    return _jsonable({k: v for k, v in vars(engine).items() if k != "events"})


def _frozen_file(root: Path, manifest_path: Path, manifest: dict, record: dict) -> Path:
    path = Path(record["path"])
    absolute = path if path.is_absolute() else root / path
    matched = [
        r
        for r in manifest["data_files"]
        if (Path(r["path"]) if Path(r["path"]).is_absolute() else root / r["path"]).resolve()
        == absolute.resolve()
    ]
    if len(matched) != 1 or matched[0]["sha256"] != record["sha256"]:
        raise ValueError("Contract input is not hash-bound to the final manifest")
    item = matched[0]
    # Final-code snapshots are preferred; old test/legacy manifests may bind originals.
    bound = manifest_path.parent / item["snapshot_path"] if item.get("snapshot_path") else absolute
    if sha256_file(bound) != record["sha256"]:
        raise ValueError("Frozen bridge data hash changed")
    return bound


def _context(root: Path, contract_path: Path, manifest_path: Path) -> dict:
    root, contract_path, manifest_path = map(Path, (root, contract_path, manifest_path))
    manifest = journal._manifest(root, manifest_path, journal._utc_now())
    if sha256_file(contract_path) != manifest["protocol_sha256"]:
        raise ValueError("Bridge contract must be the exact frozen protocol")
    contract = yaml.safe_load(contract_path.read_text())
    required = {
        "forward_policy": "fixed_b5",
        "refit_schedule": "never",
        "bars_per_day": 7,
        "signal_symbol": "QQQ",
        "trade_symbol": "TQQQ",
        "cash_mode": "zero_cash_interest",
        "execution_version": "fill_referenced_v1",
        "initial_cash": 1000.0,
        "initial_units": 0.0,
        "fee_bps": 1.0,
        "slippage_bps": 5.0,
        "adaptive_candidate_selection": False,
    }
    if any(contract.get(k) != v for k, v in required.items()):
        raise ValueError("Unsupported manual forward contract")
    if contract.get("execution", {}).get("sizing_convention") != "mark_notional_then_cash_budget":
        raise ValueError("Unsupported forward sizing_convention")
    if contract.get("contract_id") != manifest["protocol_id"]:
        raise ValueError("Contract identity mismatch")
    spec = normalize_spec(contract["policy_spec"])
    if spec["kind"] != "b5" or spec["delay_bars"] != 1 or spec["max_changes_per_session"] != 1:
        raise ValueError("Bridge supports fixed B5, one-bar delay and one fill/session only")
    semantics = contract["bar_semantics"]
    if (
        semantics["timestamp"] != "start"
        or semantics["interval_minutes"] != 60
        or semantics["timezone"] != "America/New_York"
        or semantics["last_bar_minutes"] != 30
        or semantics["availability_lag_seconds"] != 0
        or semantics["maximum_recording_age_seconds"] != 900
        or semantics.get("include_extended_hours", False)
    ):
        raise ValueError("Unsupported bar semantics")
    calpath = _frozen_file(root, manifest_path, manifest, contract["exchange_calendar"])
    calendar = pd.read_csv(calpath)
    opens = pd.to_datetime(calendar["market_open"], utc=True)
    closes = pd.to_datetime(calendar["market_close"], utc=True)
    if (
        opens.isna().any()
        or closes.isna().any()
        or not opens.is_monotonic_increasing
        or not opens.is_unique
        or (closes <= opens).any()
    ):
        raise ValueError("Invalid frozen exchange sessions")
    schedule = []
    previous_close = None
    for op, cl in zip(opens, closes, strict=True):
        local_open, local_close = (
            op.tz_convert("America/New_York"),
            cl.tz_convert("America/New_York"),
        )
        if (
            local_open.strftime("%H:%M:%S") != "09:30:00"
            or local_close.strftime("%H:%M:%S") not in {"13:00:00", "16:00:00"}
            or local_open.date() != local_close.date()
            or (previous_close is not None and op <= previous_close)
        ):
            raise ValueError("Unsupported or overlapping frozen exchange sessions")
        previous_close = cl
        for start in pd.date_range(op, cl, freq="60min", inclusive="left"):
            schedule.append((start, min(start + pd.Timedelta(hours=1), cl)))
    holdout = journal._timestamp(manifest["genuine_holdout_start_utc"], "holdout")
    forward_schedule = [(s, e) for s, e in schedule if s >= holdout]
    if not forward_schedule or forward_schedule[0][0] != holdout:
        raise ValueError("Calendar does not cover the holdout")
    warmups = []
    for key in ("qqq_warmup", "tqqq_warmup"):
        frame = pd.read_parquet(_frozen_file(root, manifest_path, manifest, contract[key]))
        idx = pd.DatetimeIndex(pd.to_datetime(frame["bar_end"], utc=True))
        values = pd.Series(frame["close"].to_numpy(dtype=float), index=idx)
        if (
            not idx.is_unique
            or not idx.is_monotonic_increasing
            or (idx >= holdout).any()
            or not np.isfinite(values).all()
            or (values <= 0).any()
        ):
            raise ValueError("Invalid feature-only warmup")
        warmups.append(values)
    if not warmups[0].index.equals(warmups[1].index):
        raise ValueError("QQQ/TQQQ warmup timestamps differ")
    prior_closes = closes[closes < holdout]
    if len(prior_closes) == 0 or warmups[0].index[-1] != prior_closes.iloc[-1]:
        raise ValueError("Warmup must reach the last scheduled close before holdout")
    # Metadata must be frozen too; never silently accept a different warmup acquisition.
    _frozen_file(root, manifest_path, manifest, contract["warmup_metadata"])
    return {
        "root": root,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "contract": contract,
        "contract_hash": sha256_file(contract_path),
        "spec": spec,
        "warmup": warmups[0],
        "schedule": forward_schedule,
    }


def _source(ctx: dict, path: Path, sequence: int, *, now: pd.Timestamp | None) -> dict:
    path = Path(path).resolve()
    source_bytes = path.read_bytes()
    source = json.loads(source_bytes)
    contract = ctx["contract"]
    if (
        source.get("schema_version") != 1
        or source.get("provider") != contract["provider"]
        or source.get("interval") != "60m"
        or set(source.get("bars", {})) != {"QQQ", "TQQQ"}
    ):
        raise ValueError("Unsupported source provider/schema/interval/symbols")
    if sequence >= len(ctx["schedule"]):
        raise ValueError("Frozen calendar exhausted; reviewed new version required")
    start, end = ctx["schedule"][sequence]
    observed = journal._timestamp(source["retrieved_at"], "retrieved_at")
    for symbol, bar in source["bars"].items():
        if journal._timestamp(bar["bar_start"], "bar_start") != start:
            raise ValueError(
                "Missing, duplicate or out-of-session bar: expected next scheduled start"
            )
        for key in ("open", "high", "low", "close", "volume", "dividends", "stock_splits"):
            if key not in bar or isinstance(bar[key], bool) or not math.isfinite(float(bar[key])):
                raise ValueError(f"Missing/nonfinite raw {symbol} {key}")
        if "capital_gains" in bar and (
            isinstance(bar["capital_gains"], bool)
            or not math.isfinite(float(bar["capital_gains"]))
            or float(bar["capital_gains"]) != 0
        ):
            raise ValueError("Corporate action capital gain unsupported by bridge V1")
        if float(bar["dividends"]) != 0 or float(bar["stock_splits"]) != 0:
            raise ValueError("Corporate action unsupported by bridge V1: stop; do not backfill")
        op, hi, lo, cl = (float(bar[k]) for k in ("open", "high", "low", "close"))
        if (
            min(op, hi, lo, cl) <= 0
            or not lo <= min(op, cl) <= max(op, cl) <= hi
            or bar["volume"] < 0
        ):
            raise ValueError("Invalid raw OHLCV")
    if observed < end or (
        now is not None and (observed > now or (now - end).total_seconds() > 900)
    ):
        raise ValueError("Incomplete, future, stale or backfilled bar rejected")
    return {
        "snapshot": source,
        "path": str(path),
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
        "start": start,
        "end": end,
        "observed": observed,
        "qqq": float(source["bars"]["QQQ"]["close"]),
        "risk": float(source["bars"]["TQQQ"]["close"]),
    }


def _initial(ctx: dict) -> dict:
    return {
        "engine": ExecutionPolicyEngine(
            ctx["spec"], bars_per_day=7, slippage_bps=5, transaction_cost_bps=1
        ),
        "account": {"cash": 1000.0, "positions": {}, "prices": {}, "equity": 1000.0},
        "prices": ctx["warmup"].copy(),
        "orders": {},
        "bars": 0,
        "halted": False,
    }


def _initial_proof(ctx: dict, runtime: dict) -> dict:
    """Acyclic checkpoint: the header hash binds proof, manifest and account."""
    checkpoint = _engine_state(runtime["engine"])
    return {
        "bridge_version": BRIDGE_VERSION,
        "protocol_id": ctx["manifest"]["protocol_id"],
        "contract_sha256": ctx["contract_hash"],
        "manifest_sha256": sha256_file(ctx["manifest_path"]),
        "frozen_at_utc": ctx["manifest"]["frozen_at_utc"],
        "genuine_holdout_start_utc": ctx["manifest"]["genuine_holdout_start_utc"],
        "status": "pending_no_forward_observations",
        "bars": 0,
        "engine_state": checkpoint,
        "engine_state_sha256": journal._digest(checkpoint),
        "engine_state_timing": "initial_flat_features_only_warmup_no_policy_steps",
        "financial_state_sha256": journal._digest(runtime["account"]),
        "warmup_last_bar_end": ctx["warmup"].index[-1].isoformat(),
    }


def _plan(
    ctx: dict,
    runtime: dict,
    source: dict,
    *,
    acknowledge_fill: bool,
    decision_time: pd.Timestamp,
    order_time: pd.Timestamp,
) -> list[dict]:
    if runtime["halted"]:
        raise ValueError("This contract halted after a missed fill; never silently restart")
    if not source["observed"] <= decision_time <= order_time:
        raise ValueError("Decision/order cannot precede signal availability")
    engine, n = runtime["engine"], runtime["bars"]
    end, quote = source["end"], source["risk"]
    prices = pd.concat([runtime["prices"], pd.Series([source["qqq"]], index=[end])])
    features = causal_features(prices, ctx["spec"], bars_per_day=7).iloc[-1]
    if not np.isfinite(features.drop(labels=["entry", "exit"]).astype(float)).all():
        raise ValueError("Incomplete warmup indicators; no discretion to invent signals")
    account = copy.deepcopy(runtime["account"])
    account["prices"] = {"TQQQ": quote}
    account["equity"] = account["cash"] + account["positions"].get("TQQQ", 0) * quote
    cid = ctx["contract"]["contract_id"]
    prefix = f"{cid}:bar:{n}"
    events = []

    def add(kind: str, timestamp: pd.Timestamp, **fields) -> dict:
        event = {
            "event_type": kind,
            "event_id": f"{prefix}:{len(events)}:{kind}",
            "protocol_id": cid,
            "paper_only": True,
            "event_timestamp": timestamp.isoformat(),
            "price_timestamp": end.isoformat(),
            "observation_timestamp": end.isoformat(),
            "source": {
                "provider": ctx["contract"]["provider"],
                "path": source["path"],
                "sha256": source["sha256"],
                "retrieved_at": source["observed"].isoformat(),
            },
            "state": copy.deepcopy(account),
            "costs": dict(ZERO_COSTS),
            **fields,
        }
        events.append(event)
        return event

    add("equity_mark", end, return_timestamp=end.isoformat())
    session = end.tz_convert("America/New_York").date()
    due = (
        engine.pending is not None
        and engine.bar + 1 >= engine.pending["due_bar"]
        and (session != engine.last_session or engine.session_fills < 1)
    )
    prior_pending = copy.deepcopy(engine.pending)
    if due and not acknowledge_fill:
        order = runtime["orders"][prior_pending["order_id"]]
        add(
            "missed_fill",
            end,
            order_id=order["event_id"],
            reason="manual_fill_not_acknowledged_contract_halted",
            **{k: order[k] for k in PHASES},
        )
        runtime["halted"] = True
        decision = None
    else:
        old_count = len(engine.events)
        decision = engine.step(end, source["qqq"], quote, features, session=session)
        decision["observation_timestamp"] = source["observed"]
        decision["decision_timestamp"] = decision_time
        new = engine.events[old_count:]
        for event in new:
            if event["event"] != "filled":
                continue
            order = runtime["orders"][event["order_id"]]
            if end <= journal._timestamp(order["order_timestamp"], "prior order"):
                raise ValueError("Prior order was not available before this future close")
            qty = account["positions"].get("TQQQ", 0.0)
            delta = (
                float(order["allocation_order"]["target_weight"]) * account["equity"] - qty * quote
            )
            notional = min(delta, account["cash"] / 1.0006) if delta > 0 else delta
            if abs(notional) <= 1e-10:
                raise ValueError("No-op fill unsupported: stop for explicit reviewed resolution")
            costs = {
                **ZERO_COSTS,
                "transaction": abs(notional) / 10000,
                "slippage": abs(notional) * 5 / 10000,
            }
            account["positions"]["TQQQ"] = max(0.0, qty + notional / quote)
            account["cash"] = max(0.0, account["cash"] - notional - sum(costs.values()))
            account["equity"] = account["cash"] + account["positions"]["TQQQ"] * quote
            add(
                "fill",
                end,
                order_id=order["event_id"],
                fill_timestamp=end.isoformat(),
                fills=[{"symbol": "TQQQ", "quantity": notional / quote, "price": quote}],
                costs=costs,
                **{k: order[k] for k in PHASES},
            )
        signal = add(
            "signal_available",
            source["observed"],
            observation_timestamp=source["observed"].isoformat(),
        )
        for event in new:
            if event["event"] == "cancelled_replaced":
                order = runtime["orders"][event["order_id"]]
                add(
                    "missed_fill",
                    decision_time,
                    order_id=order["event_id"],
                    reason="policy_replaced_before_session_eligible_fill",
                    **{k: order[k] for k in PHASES},
                )
            elif event["event"] == "submitted":
                order = add(
                    "hypothetical_order",
                    order_time,
                    signal_id=signal["event_id"],
                    observation_timestamp=source["observed"].isoformat(),
                    decision_timestamp=decision_time.isoformat(),
                    order_timestamp=order_time.isoformat(),
                    allocation_order={
                        "asset": "TQQQ",
                        "target_weight": float(event["target_weight"]),
                        "fee_bps": 1.0,
                        "slippage_bps": 5.0,
                        "sizing_convention": "mark_notional_then_cash_budget",
                    },
                )
                runtime["orders"][event["order_id"]] = order
                # Replace research-only logical nanosecond phases with measured source/decision times.
                for key in PHASES:
                    engine.pending[key] = journal._timestamp(order[key], key)
    runtime.update(account=account, prices=prices, bars=n + 1)
    checkpoint = _engine_state(engine)
    proof = {
        "bridge_version": BRIDGE_VERSION,
        "contract_sha256": ctx["contract_hash"],
        "bar_sequence": n,
        "bar_start": source["start"].isoformat(),
        "bar_end": end.isoformat(),
        "acknowledge_fill": acknowledge_fill,
        "decision_timestamp": decision_time.isoformat(),
        "order_timestamp": order_time.isoformat(),
        "halted": runtime["halted"],
        "engine_state": checkpoint,
        "engine_state_sha256": journal._digest(checkpoint),
        "engine_state_timing": "post_batch_checkpoint_not_each_event_asof_state",
        "decision": _jsonable(decision),
        "features": _jsonable(features.to_dict()),
        "batch_size": len(events),
    }
    for i, event in enumerate(events):
        event["policy_state"] = {**proof, "batch_index": i}
    return events


def _replay(ctx: dict, records: list[dict]) -> dict:
    journal._bound_header(records, ctx["manifest"], ctx["manifest_path"])
    runtime = _initial(ctx)
    if records[0]["state"] != runtime["account"]:
        raise ValueError("Bridge requires the frozen flat initial cash account")
    if journal._canonical(records[0].get("policy_state")) != journal._canonical(
        _initial_proof(ctx, runtime)
    ):
        raise ValueError("Initial bridge checkpoint differs from recomputed frozen policy")
    offset = 1
    while offset < len(records):
        first = records[offset]["event"]
        proof = first.get("policy_state", {})
        if proof.get("bridge_version") != BRIDGE_VERSION or proof.get("batch_index") != 0:
            raise ValueError("Not a generated bridge batch; arbitrary policy state is not evidence")
        size = proof.get("batch_size", 0)
        if not isinstance(size, int) or size < 2 or offset + size > len(records):
            raise ValueError("Partial bridge batch: halt; never rewrite or backfill")
        recorded_at = journal._timestamp(records[offset]["recorded_at_utc"], "recorded_at")
        if (
            journal._timestamp(proof["order_timestamp"], "order") > recorded_at
            or recorded_at > journal._utc_now()
        ):
            raise ValueError("Checkpoint phases cannot postdate recording or the real clock")
        source = _source(ctx, Path(first["source"]["path"]), runtime["bars"], now=None)
        if source["sha256"] != first["source"]["sha256"]:
            raise ValueError("Forward source snapshot changed")
        planned = _plan(
            ctx,
            runtime,
            source,
            acknowledge_fill=proof["acknowledge_fill"],
            decision_time=journal._timestamp(proof["decision_timestamp"], "decision"),
            order_time=journal._timestamp(proof["order_timestamp"], "order"),
        )
        if len(planned) != size:
            raise ValueError("Recorded batch differs from recomputed policy")
        for j, expected in enumerate(planned):
            record = records[offset + j]
            actual = {k: v for k, v in record["event"].items() if k != "expected_previous_hash"}
            if journal._canonical(expected) != journal._canonical(actual):
                raise ValueError("Recorded event differs from replayed policy/account/source")
            # Validate original recording freshness/chronology and financial continuity too.
            journal._validate_event(
                record["event"],
                records[: offset + j],
                ctx["manifest"],
                journal._timestamp(record["recorded_at_utc"], "recorded_at"),
            )
        offset += size
    return runtime


def initialize_paper_bridge(
    root: Path, contract_path: Path, manifest_path: Path, ledger_path: Path
) -> dict:
    """Create only the pending header. No signals, orders or fabricated performance."""
    ctx = _context(root, contract_path, manifest_path)
    proof = _initial_proof(ctx, _initial(ctx))
    return journal.initialize_forward_ledger(
        root, manifest_path, ledger_path, initial_cash=1000, policy_state=proof
    )


def verify_paper_bridge(
    root: Path, contract_path: Path, manifest_path: Path, ledger_path: Path
) -> dict:
    """Replay all source-bound decisions and actual paper fills; do not trust checkpoints."""
    ctx = _context(root, contract_path, manifest_path)
    with journal._locked(Path(ledger_path)) as stream:
        records = journal._read(stream)
        runtime = _replay(ctx, records)
    n = runtime["bars"]
    return {
        "bridge_version": BRIDGE_VERSION,
        "paper_only": True,
        "bars": n,
        "events": len(records) - 1,
        "head_hash": records[-1]["record_hash"],
        "halted": runtime["halted"],
        "state": runtime["account"],
        "next_bar_start": ctx["schedule"][n][0].isoformat() if n < len(ctx["schedule"]) else None,
        "engine_state": _engine_state(runtime["engine"]),
    }


def record_completed_bar(
    root: Path,
    contract_path: Path,
    manifest_path: Path,
    ledger_path: Path,
    source_path: Path,
    *,
    acknowledge_fill: bool = False,
) -> dict:
    """Generate and append policy-validated events from ONE fresh completed bar.

    ``acknowledge_fill`` accepts the PREVIOUSLY declared next-close paper-fill
    assumption, not a broker trade and not discretion to alter target or quantity.
    Omitted acknowledgement when a fill is due records missed_fill and halts V1.
    A complete batch is validated under one file lock before any writes. A crash
    during fsync can leave an incomplete batch; subsequent replay then stops.
    """
    if not isinstance(acknowledge_fill, bool):
        raise ValueError("acknowledge_fill must be boolean")
    ctx = _context(root, contract_path, manifest_path)
    with journal._locked(Path(ledger_path)) as stream:
        records = journal._read(stream)
        runtime = _replay(ctx, records)
        source = _source(ctx, Path(source_path), runtime["bars"], now=journal._utc_now())
        decision_time, order_time = journal._utc_now(), journal._utc_now()
        events = _plan(
            ctx,
            runtime,
            source,
            acknowledge_fill=acknowledge_fill,
            decision_time=decision_time,
            order_time=order_time,
        )
        # Prevalidate the complete generated batch, then append without releasing the lock.
        trial = list(records)
        payloads = []
        for event in events:
            event["expected_previous_hash"] = trial[-1]["record_hash"]
            now = journal._utc_now()
            clean = journal._validate_event(event, trial, ctx["manifest"], now)
            payload = {
                "record_type": "event",
                "sequence": len(trial),
                "previous_hash": trial[-1]["record_hash"],
                "protocol_id": ctx["manifest"]["protocol_id"],
                "recorded_at_utc": now.isoformat(),
                "event": clean,
            }
            trial.append({**payload, "record_hash": journal._digest(payload)})
            payloads.append(payload)
        for payload in payloads:
            journal._write(stream, payload)
    return {
        "bar_sequence": runtime["bars"] - 1,
        "paper_only": True,
        "halted": runtime["halted"],
        "events": events,
        "state": runtime["account"],
        "head_hash": trial[-1]["record_hash"],
    }
