"""Append-only, fail-closed prospective PAPER observations; never broker orders.

The hash chain detects broken/reordered records and binds records to a frozen
manifest. It is not an externally notarized log: somebody who can rewrite the
entire file and recompute its hashes can forge local history. Provider timestamps
and price truth remain source attestations, not independently verified facts.
New source snapshots must be separate from immutable historical frozen inputs.

Target-allocation orders are sized only at a later observed fill quote; they do
not embed a future-quote share calculation in the preceding order. Explicit
corporate_action events bind split/distribution fields and the post-action raw
quote to a hashed JSON snapshot, use prior-holder share entitlement, and record
distribution cash separately from interest/trades. Cash is hypothetically
credited on the first observed ex-date bar, not the actual pay date. The split
reference mark is mechanical; any difference in the source post-action quote
is genuine marked price movement, not extra split P&L. These account checks do
not repair indicator or strategy-policy continuity: the B5 bridge may and should
refuse actions until that separate path is validated. No investor tax compliance
is claimed by this primary paper-account recorder.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from trend_following.research_protocol import (
    next_complete_session,
    sha256_file,
    verify_frozen_inputs,
)

EVENT_TYPES = {
    "signal_available",
    "hypothetical_order",
    "fill",
    "missed_fill",
    "equity_mark",
    "corporate_action",
}
ZERO_HASH = "0" * 64
MAX_EVENT_AGE_SECONDS = 900
COST_KEYS = {"transaction", "slippage", "financing", "tax"}


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _timestamp(value: Any, label: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if pd.isna(ts) or ts.tzinfo is None:
        raise ValueError(f"{label} must be a timezone-aware timestamp")
    return ts.tz_convert("UTC")


def _amount(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, not boolean")
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0):
        raise ValueError(f"Invalid {label}")
    return number


def _allocation_order(value: Any) -> dict[str, Any]:
    """Validate a prior target instruction, without inventing its future fill quote."""
    required = {"asset", "target_weight", "fee_bps", "slippage_bps", "sizing_convention"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(
            "allocation_order requires asset, target_weight, costs and sizing_convention"
        )
    if not isinstance(value["asset"], str) or not value["asset"]:
        raise ValueError("allocation_order asset is required")
    weight = _amount(value["target_weight"], "target_weight", nonnegative=True)
    fee = _amount(value["fee_bps"], "fee_bps", nonnegative=True)
    slip = _amount(value["slippage_bps"], "slippage_bps", nonnegative=True)
    if weight > 1 or fee + slip >= 10_000:
        raise ValueError("Invalid allocation weight or combined execution costs")
    if value["sizing_convention"] != "mark_notional_then_cash_budget":
        raise ValueError("Unsupported allocation sizing convention")
    return value


def _manifest(root: Path, manifest_path: Path, now: pd.Timestamp) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("stage") != "final_code" or manifest.get("prospective_eligible") is not True:
        raise ValueError("A final_code, prospective-eligible manifest is required")
    frozen = _timestamp(manifest["frozen_at_utc"], "frozen_at_utc")
    holdout = _timestamp(manifest["genuine_holdout_start_utc"], "genuine_holdout_start_utc")
    if frozen > now or holdout <= frozen:
        raise ValueError("Invalid final freeze/holdout chronology")
    verify_frozen_inputs(root, manifest_path)
    protocol_path = manifest_path.parent / "protocol.yaml"
    diff_path = manifest_path.parent / str(manifest["dirty_tree_diff"])
    if sha256_file(protocol_path) != manifest["protocol_sha256"]:
        raise ValueError("Frozen protocol snapshot changed")
    if sha256_file(diff_path) != manifest["dirty_tree_diff_sha256"]:
        raise ValueError("Frozen dirty-tree diff changed")
    protocol = yaml.safe_load(protocol_path.read_text())
    if _digest(protocol) != manifest["parameters_sha256"]:
        raise ValueError("Frozen parameter snapshot mismatch")
    calendar = manifest["exchange_calendar"]
    if not calendar or sha256_file(Path(calendar["path"])) != calendar["sha256"]:
        raise ValueError("Frozen exchange calendar changed or missing")
    calendar_frame = pd.read_csv(calendar["path"])
    opens = pd.DatetimeIndex(pd.to_datetime(calendar_frame["market_open"], utc=True))
    if next_complete_session(frozen, opens).tz_convert("UTC") != holdout:
        raise ValueError("Holdout is not the first complete session after the final freeze")
    timezone = protocol.get("strategy", {}).get("timezone", "America/New_York")
    for record in manifest.get("known_local_bar_endpoints", []):
        end = pd.Timestamp(record["index_end"])
        if end.tzinfo is None:
            end = end.tz_localize(timezone)
        if end.tz_convert("UTC") >= holdout:
            raise ValueError("Holdout overlaps known local historical bars")
    return manifest


@contextmanager
def _locked(path: Path, *, create: bool = False):
    flags = os.O_RDWR | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    if create:
        flags |= os.O_CREAT
    fd = os.open(path, flags, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        with os.fdopen(fd, "r+b", closefd=False) as stream:
            yield stream
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _read(stream) -> list[dict[str, Any]]:
    stream.seek(0)
    data = stream.read()
    if not data:
        return []
    if not data.endswith(b"\n"):
        raise ValueError(
            "Truncated ledger tail: repair outside this recorder; no automatic rewrite"
        )
    records = []
    previous = ZERO_HASH
    for number, line in enumerate(data.splitlines()):
        record = json.loads(line)
        expected_hash = record.get("record_hash")
        payload = {k: v for k, v in record.items() if k != "record_hash"}
        if record.get("sequence") != number or record.get("previous_hash") != previous:
            raise ValueError("Ledger continuity/hash chain broken")
        if _digest(payload) != expected_hash:
            raise ValueError("Ledger record hash mismatch")
        if number == 0 and record.get("record_type") != "header":
            raise ValueError("Ledger header missing")
        if number > 0 and record.get("record_type") != "event":
            raise ValueError("Unexpected non-event after header")
        records.append(record)
        previous = expected_hash
    return records


def _write(stream, payload: dict[str, Any]) -> dict[str, Any]:
    record = {**payload, "record_hash": _digest(payload)}
    data = _canonical(record) + b"\n"
    # O_APPEND + exclusive lock serializes writers. A torn crash-tail is detected
    # and rejected on next read, never silently truncated/repaired.
    stream.seek(0, os.SEEK_END)
    stream.write(data)
    stream.flush()
    os.fsync(stream.fileno())
    return record


def _bound_header(
    records: list[dict[str, Any]], manifest: dict[str, Any], manifest_path: Path
) -> None:
    if not records:
        raise ValueError("Initialize a forward ledger before appending")
    header = records[0]
    if header["manifest_sha256"] != sha256_file(manifest_path):
        raise ValueError("Manifest changed or this ledger belongs to another freeze")
    if header["protocol_id"] != manifest["protocol_id"]:
        raise ValueError("Protocol ID mismatch")


def initialize_forward_ledger(
    root: Path,
    manifest_path: Path,
    ledger_path: Path,
    *,
    initial_cash: float = 1.0,
    policy_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write only a pending paper header; never invent observations/performance.

    Optional policy proof is hash-bound verbatim; a strategy-specific bridge must
    recompute and validate it. Generic recorder callers retain the empty default.
    """
    root, manifest_path, ledger_path = Path(root), Path(manifest_path), Path(ledger_path)
    now = _utc_now()
    manifest = _manifest(root, manifest_path, now)
    cash = _amount(initial_cash, "initial_cash", nonnegative=True)
    if policy_state is not None and not isinstance(policy_state, dict):
        raise ValueError("Initial policy_state must be a JSON object")
    initial_policy_state = json.loads(_canonical({} if policy_state is None else policy_state))
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with _locked(ledger_path, create=True) as stream:
        if _read(stream):
            raise ValueError("Forward ledger already initialized; overwrite is prohibited")
        return _write(
            stream,
            {
                "record_type": "header",
                "schema_version": 1,
                "sequence": 0,
                "previous_hash": ZERO_HASH,
                "protocol_id": manifest["protocol_id"],
                "manifest_sha256": sha256_file(manifest_path),
                "created_at_utc": now.isoformat(),
                "frozen_at_utc": manifest["frozen_at_utc"],
                "genuine_holdout_start_utc": manifest["genuine_holdout_start_utc"],
                "status": "pending_prospective_observations_no_performance_claim",
                "paper_only": True,
                "maximum_event_age_seconds": MAX_EVENT_AGE_SECONDS,
                "state": {"cash": cash, "positions": {}, "prices": {}, "equity": cash},
                "policy_state": initial_policy_state,
            },
        )


def _validate_event(
    event: dict[str, Any],
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
    now: pd.Timestamp,
) -> dict[str, Any]:
    kind = event.get("event_type")
    if kind not in EVENT_TYPES:
        raise ValueError("Unsupported forward event type")
    if event.get("protocol_id") != manifest["protocol_id"]:
        raise ValueError("Event protocol_id differs from the final freeze")
    if event.get("paper_only") is not True or event.get("retrospective", False):
        raise ValueError("Only genuinely prospective PAPER events are accepted")
    if not isinstance(event.get("event_id"), str) or not event["event_id"]:
        raise ValueError("event_id is required")
    if any(r.get("event", {}).get("event_id") == event["event_id"] for r in records[1:]):
        raise ValueError("Duplicate event_id")
    if event.get("expected_previous_hash") != records[-1]["record_hash"]:
        raise ValueError("Stale/missing expected_previous_hash; reread the current ledger head")
    if not isinstance(event.get("policy_state"), dict):
        raise ValueError("Explicit policy_state snapshot required")
    holdout = _timestamp(manifest["genuine_holdout_start_utc"], "holdout")
    event_time = _timestamp(event["event_timestamp"], "event_timestamp")
    price_time = _timestamp(event["price_timestamp"], "price_timestamp")
    times = {"event_timestamp": event_time, "price_timestamp": price_time}
    for field in (
        "observation_timestamp",
        "decision_timestamp",
        "order_timestamp",
        "fill_timestamp",
        "return_timestamp",
    ):
        if event.get(field) is not None:
            times[field] = _timestamp(event[field], field)
    if "observation_timestamp" not in times:
        raise ValueError("observation_timestamp is required")
    if any(t < holdout or t > now for t in times.values()):
        raise ValueError(
            "Forward timestamps must be after the genuine holdout start and not in the future"
        )
    if (now - event_time).total_seconds() > MAX_EVENT_AGE_SECONDS or (
        now - price_time
    ).total_seconds() > MAX_EVENT_AGE_SECONDS:
        raise ValueError("Old/backfilled event or price bar rejected")
    if price_time > event_time:
        raise ValueError("Observed prices cannot postdate the event")
    if len(records) > 1 and event_time < _timestamp(
        records[-1]["event"]["event_timestamp"], "previous event"
    ):
        raise ValueError("Event time must not move backward")
    phases = [
        times[k]
        for k in (
            "observation_timestamp",
            "decision_timestamp",
            "order_timestamp",
            "fill_timestamp",
            "return_timestamp",
        )
        if k in times
    ]
    if any(a > b for a, b in zip(phases, phases[1:], strict=False)):
        raise ValueError("Observation/decision/order/fill/return chronology is inconsistent")
    if phases[-1] > event_time:
        raise ValueError("Event contains a later phase that has not happened yet")
    previous_events = {r["event"]["event_id"]: r["event"] for r in records[1:]}
    if kind in {"signal_available", "hypothetical_order", "missed_fill"} and (
        "fill_timestamp" in times or "return_timestamp" in times
    ):
        raise ValueError("Unfilled events cannot claim fills or subsequent returns")
    if (
        "fill_timestamp" in times
        and "return_timestamp" in times
        and times["return_timestamp"] <= times["fill_timestamp"]
    ):
        raise ValueError("Earned return must follow, not coincide with, its fill")
    if kind == "fill" and "return_timestamp" in times:
        raise ValueError("Record a subsequent return as a later equity_mark, not the fill")
    if kind == "signal_available":
        if times["observation_timestamp"] != event_time:
            raise ValueError("signal_available timestamp must equal its observation timestamp")
    if kind == "hypothetical_order":
        signal = previous_events.get(event.get("signal_id"))
        if signal is None or signal["event_type"] != "signal_available":
            raise ValueError("Order requires a previously recorded signal_id")
        if not {"decision_timestamp", "order_timestamp"}.issubset(times):
            raise ValueError("Orders need decision and order timestamps")
        if times["observation_timestamp"] != _timestamp(
            signal["observation_timestamp"], "signal observation"
        ):
            raise ValueError("Order observation differs from recorded signal")
        if event_time != times["order_timestamp"]:
            raise ValueError("Order event timestamp must equal order timestamp")
        if "allocation_order" in event:
            if "requested_quantities" in event:
                raise ValueError(
                    "Order must use allocation_order OR requested_quantities, not both"
                )
            _allocation_order(event["allocation_order"])
        else:
            requested = event.get("requested_quantities")
            if not isinstance(requested, dict) or not requested:
                raise ValueError(
                    "Hypothetical order requires explicit requested_quantities or allocation_order"
                )
            if any(_amount(qty, "requested quantity") == 0 for qty in requested.values()):
                raise ValueError("Requested quantities must be finite and nonzero")
    if kind in {"fill", "missed_fill"}:
        order_id = event.get("order_id")
        order = previous_events.get(order_id)
        if order is None or order["event_type"] != "hypothetical_order":
            raise ValueError(
                "Fill/missed fill requires a previously recorded hypothetical order_id"
            )
        if any(
            e.get("order_id") == order_id and e["event_type"] in {"fill", "missed_fill"}
            for e in previous_events.values()
        ):
            raise ValueError(
                "Order was already resolved; implicit partial/duplicate fills prohibited"
            )
        for field in ("observation_timestamp", "decision_timestamp", "order_timestamp"):
            if field not in times or times[field] != _timestamp(order[field], field):
                raise ValueError("Fill must preserve its recorded order phase timestamps")
        if event_time <= times["order_timestamp"] or price_time <= times["observation_timestamp"]:
            raise ValueError("Same-bar/retroactive fill rejected")
        if kind == "fill":
            if times.get("fill_timestamp") != event_time:
                raise ValueError("Fill event requires matching fill timestamp")
            fill_date = event_time.tz_convert("America/New_York").date()
            if any(
                e["event_type"] == "fill"
                and _timestamp(e["event_timestamp"], "fill").tz_convert("America/New_York").date()
                == fill_date
                for e in previous_events.values()
            ):
                raise ValueError("One discretionary fill per actual exchange session")
        elif "fill_timestamp" in times:
            raise ValueError("A missed fill cannot claim a fill timestamp")
    if kind == "equity_mark" and times.get("return_timestamp") != event_time:
        raise ValueError("Equity marks require matching return_timestamp")
    if kind == "corporate_action" and any(
        field in times
        for field in ("decision_timestamp", "order_timestamp", "fill_timestamp", "return_timestamp")
    ):
        raise ValueError(
            "A corporate action is not a decision, order, fill, or earned-return event"
        )
    source = event.get("source")
    if not isinstance(source, dict) or not source.get("provider"):
        raise ValueError("Source provider/path/hash/retrieval evidence required")
    source_path = Path(source["path"])
    if not source_path.is_absolute() or sha256_file(source_path) != source["sha256"]:
        raise ValueError("Observed source snapshot hash mismatch or nonabsolute path")
    retrieved = _timestamp(source["retrieved_at"], "source retrieval")
    if not price_time <= retrieved <= now:
        raise ValueError("Source retrieval must follow the observed bar and not be in the future")
    state = event.get("state")
    if not isinstance(state, dict):
        raise ValueError("Explicit account state required")
    cash = _amount(state["cash"], "cash", nonnegative=True)
    equity = _amount(state["equity"], "equity", nonnegative=True)
    positions = {
        str(k): _amount(v, "position", nonnegative=True) for k, v in state["positions"].items()
    }
    prices = {str(k): _amount(v, "price", nonnegative=True) for k, v in state["prices"].items()}
    if any(value <= 0 for value in prices.values()) or not set(positions).issubset(prices):
        raise ValueError("All held positions require positive observed prices")
    marked = cash + sum(qty * prices[symbol] for symbol, qty in positions.items())
    if not math.isclose(marked, equity, rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError("Cash plus marked positions does not reconcile to equity")
    costs = event.get("costs")
    if not isinstance(costs, dict) or set(costs) != COST_KEYS:
        raise ValueError("Explicit transaction/slippage/financing/tax cash costs required")
    total_cost = sum(_amount(v, k, nonnegative=True) for k, v in costs.items())
    interest = _amount(event.get("cash_interest", 0), "cash_interest")
    previous_state = (
        records[-1].get("state") if len(records) == 1 else records[-1]["event"]["state"]
    )
    expected_positions = dict(previous_state["positions"])
    cash_delta = interest - total_cost
    fills = event.get("fills", [])
    if kind == "fill" and not fills:
        raise ValueError("Fill requires explicit signed share quantities and prices")
    if kind != "fill" and fills:
        raise ValueError("Non-fill event cannot change positions")
    if kind == "corporate_action":
        if total_cost or interest:
            raise ValueError("Corporate actions cannot disguise costs, taxes or cash interest")
        action = event.get("corporate_action")
        source_fields = {
            "action_id",
            "symbol",
            "effective_timestamp",
            "split_ratio",
            "distribution_per_share",
            "distribution_unit",
            "cash_timing",
        }
        account_fields = {
            "entitled_pre_split_quantity",
            "post_split_reference_price",
            "cash_credit",
        }
        if not isinstance(action, dict) or set(action) != source_fields | account_fields:
            raise ValueError(
                "Explicit source-bound corporate_action and entitlement fields required"
            )
        if not isinstance(action["action_id"], str) or not action["action_id"]:
            raise ValueError("corporate_action action_id is required")
        symbol = action["symbol"]
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("corporate_action symbol is required")
        effective = _timestamp(
            action["effective_timestamp"], "corporate action effective timestamp"
        )
        if (
            effective != event_time
            or price_time != event_time
            or times["observation_timestamp"] != event_time
        ):
            raise ValueError(
                "Corporate action must be recorded at its observed effective bar, not backdated"
            )
        ratio = _amount(action["split_ratio"], "split_ratio", nonnegative=True)
        per_share = _amount(
            action["distribution_per_share"], "distribution_per_share", nonnegative=True
        )
        if ratio <= 0 or (ratio == 1 and per_share == 0):
            raise ValueError("Corporate action needs a positive nontrivial split or distribution")
        if (
            action["distribution_unit"] != "post_split_share"
            or action["cash_timing"] != "first_observed_ex_date_bar_hypothetical"
        ):
            raise ValueError(
                "Explicit post-split units and hypothetical ex-date cash timing required"
            )
        # A hash alone does not bind declared action numbers. Read the frozen
        # observed snapshot and match the action payload AND current raw quote.
        snapshot = json.loads(source_path.read_text())
        if not isinstance(snapshot, dict) or not isinstance(
            snapshot.get("corporate_actions"), list
        ):
            raise ValueError("Corporate action source must contain corporate_actions JSON rows")
        matching = [
            row
            for row in snapshot["corporate_actions"]
            if isinstance(row, dict) and row.get("action_id") == action["action_id"]
        ]
        if len(matching) != 1 or any(matching[0].get(k) != action[k] for k in source_fields):
            raise ValueError("Declared corporate action differs from its source snapshot")
        if _timestamp(snapshot.get("price_timestamp"), "source price timestamp") != price_time:
            raise ValueError("Corporate action source quote timestamp mismatch")
        quote = _amount(
            snapshot.get("prices", {}).get(symbol), "source post-action quote", nonnegative=True
        )
        if (
            symbol not in prices
            or quote <= 0
            or not math.isclose(prices[symbol], quote, rel_tol=1e-12, abs_tol=1e-10)
        ):
            raise ValueError("Corporate action must mark the source-bound post-action quote")
        session = effective.tz_convert("America/New_York").date()
        for prior in previous_events.values():
            prior_action = prior.get("corporate_action", {})
            if prior.get("event_type") == "corporate_action" and (
                prior_action.get("action_id") == action["action_id"]
                or (
                    prior_action.get("symbol") == symbol
                    and _timestamp(prior_action["effective_timestamp"], "prior corporate action")
                    .tz_convert("America/New_York")
                    .date()
                    == session
                )
            ):
                raise ValueError("Duplicate corporate action for source ID or symbol/session")
            if prior.get("event_type") == "fill" and any(
                f.get("symbol") == symbol for f in prior.get("fills", [])
            ):
                if (
                    _timestamp(prior["event_timestamp"], "prior fill")
                    .tz_convert("America/New_York")
                    .date()
                    == session
                ):
                    raise ValueError(
                        "Process corporate actions before same-session fills to preserve prior-holder entitlement"
                    )
            if prior.get("event_type") == "hypothetical_order":
                assets = set(prior.get("requested_quantities", {}))
                if "allocation_order" in prior:
                    assets.add(prior["allocation_order"]["asset"])
                resolved = any(
                    p.get("order_id") == prior["event_id"]
                    and p["event_type"] in {"fill", "missed_fill"}
                    for p in previous_events.values()
                )
                if symbol in assets and not resolved:
                    raise ValueError(
                        "Resolve pending pre-action orders explicitly before a corporate action"
                    )
        prior_quantity = float(previous_state["positions"].get(symbol, 0.0))
        entitled = _amount(
            action["entitled_pre_split_quantity"], "entitled quantity", nonnegative=True
        )
        if not math.isclose(prior_quantity, entitled, rel_tol=1e-12, abs_tol=1e-10):
            raise ValueError("Distribution entitlement must equal prior-holder share inventory")
        old_quote = previous_state["prices"].get(symbol)
        if old_quote is not None:
            reference = _amount(
                action["post_split_reference_price"], "post-split reference", nonnegative=True
            )
            if not math.isclose(reference, float(old_quote) / ratio, rel_tol=1e-12, abs_tol=1e-10):
                raise ValueError(
                    "Mechanical split reference must equal previous quote divided by ratio"
                )
        elif prior_quantity or action["post_split_reference_price"] is not None:
            raise ValueError("Missing prior split-reference quote")
        expected_positions[symbol] = prior_quantity * ratio
        credit = _amount(action["cash_credit"], "distribution cash credit", nonnegative=True)
        if not math.isclose(
            credit, prior_quantity * ratio * per_share, rel_tol=1e-12, abs_tol=1e-10
        ):
            raise ValueError(
                "Distribution cash credit does not reconcile to prior-held post-split shares"
            )
        cash_delta += credit
    if kind == "fill":
        actual_quantities: dict[str, float] = {}
        for fill in fills:
            symbol = str(fill["symbol"])
            actual_quantities[symbol] = actual_quantities.get(symbol, 0.0) + _amount(
                fill["quantity"], "filled quantity"
            )
        original_order = previous_events[event["order_id"]]
        if "allocation_order" in original_order:
            allocation = _allocation_order(original_order["allocation_order"])
            symbol = allocation["asset"]
            if set(actual_quantities) != {symbol} or len(fills) != 1 or symbol not in prices:
                raise ValueError(
                    "Allocation fill requires exactly one fill at its current marked quote"
                )
            if any(v > 1e-10 for k, v in previous_state["positions"].items() if k != symbol):
                raise ValueError("Allocation sizing currently supports one risk asset only")
            quote = prices[symbol]
            if not math.isclose(
                _amount(fills[0]["price"], "fill price"), quote, rel_tol=1e-12, abs_tol=1e-10
            ):
                raise ValueError("Allocation fill quote must match its current raw mark")
            old_quantity = float(previous_state["positions"].get(symbol, 0.0))
            old_cash = float(previous_state["cash"])
            wealth = old_cash + old_quantity * quote
            delta = float(allocation["target_weight"]) * wealth - old_quantity * quote
            cost_rate = (float(allocation["fee_bps"]) + float(allocation["slippage_bps"])) / 10_000
            notional = min(delta, old_cash / (1 + cost_rate)) if delta > 0 else delta
            if abs(notional) <= 1e-10:
                raise ValueError("No-op allocation cannot be represented as a fill")
            requested = {symbol: notional / quote}
            expected_fee = abs(notional) * float(allocation["fee_bps"]) / 10_000
            expected_slip = abs(notional) * float(allocation["slippage_bps"]) / 10_000
            if (
                not math.isclose(
                    float(costs["transaction"]), expected_fee, rel_tol=1e-10, abs_tol=1e-8
                )
                or not math.isclose(
                    float(costs["slippage"]), expected_slip, rel_tol=1e-10, abs_tol=1e-8
                )
                or float(costs["financing"])
                or float(costs["tax"])
                or interest
            ):
                raise ValueError("Allocation fill costs must match its recorded fee/slippage terms")
        else:
            requested = original_order["requested_quantities"]
        if set(actual_quantities) != set(requested) or any(
            not math.isclose(
                actual_quantities[symbol], float(requested[symbol]), rel_tol=1e-10, abs_tol=1e-10
            )
            for symbol in requested
        ):
            raise ValueError(
                "Fill quantities differ from the recorded hypothetical order; partial fills unsupported"
            )
    for fill in fills:
        symbol = str(fill["symbol"])
        qty, price = (
            _amount(fill["quantity"], "filled quantity"),
            _amount(fill["price"], "fill price"),
        )
        if qty == 0 or price <= 0:
            raise ValueError("Nonzero fill quantity and positive fill price required")
        expected_positions[symbol] = expected_positions.get(symbol, 0) + qty
        cash_delta -= qty * price
    for symbol in set(expected_positions) | set(positions):
        if not math.isclose(
            expected_positions.get(symbol, 0),
            positions.get(symbol, 0),
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise ValueError("Position continuity broken")
    if not math.isclose(
        float(previous_state["cash"]) + cash_delta, cash, rel_tol=1e-10, abs_tol=1e-8
    ):
        raise ValueError("Cash continuity broken; unexplained deposits/withdrawals prohibited")
    if kind in {"signal_available", "hypothetical_order", "missed_fill"} and (
        total_cost or interest
    ):
        raise ValueError("Signal/order/missed-fill cannot create accounting cash flows")
    # Preserve the submitted provenance and policy snapshot verbatim; canonical
    # serialization also rejects NaN/Infinity anywhere in the submitted object.
    _canonical(event)
    return event


def append_forward_event(
    root: Path, manifest_path: Path, ledger_path: Path, event: dict[str, Any]
) -> dict[str, Any]:
    """Validate frozen inputs, current timestamps, lineage and financial state."""
    root, manifest_path, ledger_path = Path(root), Path(manifest_path), Path(ledger_path)
    with _locked(ledger_path) as stream:
        now = _utc_now()
        manifest = _manifest(root, manifest_path, now)
        records = _read(stream)
        _bound_header(records, manifest, manifest_path)
        clean = _validate_event(event, records, manifest, now)
        return _write(
            stream,
            {
                "record_type": "event",
                "sequence": len(records),
                "previous_hash": records[-1]["record_hash"],
                "protocol_id": manifest["protocol_id"],
                "recorded_at_utc": now.isoformat(),
                "event": clean,
            },
        )


def verify_forward_ledger(root: Path, manifest_path: Path, ledger_path: Path) -> dict[str, Any]:
    """Read-only integrity check and head needed for optimistic append locking."""
    root, manifest_path, ledger_path = Path(root), Path(manifest_path), Path(ledger_path)
    with _locked(ledger_path) as stream:
        manifest = _manifest(root, manifest_path, _utc_now())
        records = _read(stream)
        _bound_header(records, manifest, manifest_path)
        last = records[-1]
        return {
            "protocol_id": manifest["protocol_id"],
            "records": len(records),
            "events": len(records) - 1,
            "head_hash": last["record_hash"],
            "state": last["state"] if len(records) == 1 else last["event"]["state"],
            "policy_state": last["policy_state"]
            if len(records) == 1
            else last["event"]["policy_state"],
            "genuine_holdout_start_utc": manifest["genuine_holdout_start_utc"],
            "paper_only": True,
        }
