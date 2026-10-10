"""Separate causal TF-gated tactical, partial-blend and ER20-routing experiment.

All inputs are supplied bounded in-memory packets. Frozen feature, event-clock
and cash helpers are reused read-only; no original engine is modified. Shadow
states incur no financial fees. Normalized actually funded account ledgers are
streamed separately and combined only with frozen initial capital weights.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from trend_following import research_daily_hourly_v5 as core
from trend_following import research_mean_reversion as mr
from trend_following import research_mean_reversion_engine as mr_engine
from trend_following import research_portfolio_engine as portfolio
from trend_following import research_quality_engine_v5 as quality
from trend_following.research_hybrid_allocation_ledger import TargetBook
from trend_following.research_rates import rate_accrual

ALPHAS = (0.5, 0.7, 0.9)
TF_MODE, MR_MODE = 0, 1
NAT = np.iinfo(np.int64).min
PRINCIPAL_IDS = tuple(
    [f"hybrid_core_{round(a * 100):03d}" for a in ALPHAS]
    + [f"hybrid_blend_{round(a * 100):03d}" for a in ALPHAS]
    + ["hybrid_regime_er20_025_035", "portfolio_tf_100"]
)
PORTFOLIO_IDS = (*PRINCIPAL_IDS, "hybrid_gated_mr")
SHADOW_TRADE_COLUMNS = (
    "strategy",
    "candidate_id",
    "base_candidate_id",
    "pair_id",
    "trend_candidate_id",
    "side",
    "timestamp",
    "phase",
    "decision_at",
    "decision_bar",
    "execution_bar",
    "cause",
    "state",
    "post_gap_flag",
    "post_gap_windows",
)


@dataclass(frozen=True)
class Pair:
    tf: core.Candidate
    mr: mr.Candidate
    tf_index: int
    mr_index: int

    def __post_init__(self):
        if not isinstance(self.tf, core.Candidate) or not isinstance(self.mr, mr.Candidate):
            raise ValueError("Original TF and MR candidate types required")
        if (
            self.tf.entry_confirmation_hours != 3
            or self.tf.exit_confirmation_hours != 3
            or self.mr.confirmation_hours != 3
        ):
            raise ValueError("All constituent waits remain exactly three trading hours")
        if not all(
            isinstance(i, (int, np.integer)) and not isinstance(i, (bool, np.bool_)) and i >= 0
            for i in (self.tf_index, self.mr_index)
        ):
            raise ValueError("Nonnegative canonical constituent indices required")

    @property
    def pair_id(self):
        return f"hybrid_pair__{self.tf.candidate_id}__{self.mr.candidate_id}"

    @property
    def candidate_id(self):
        return self.pair_id

    def to_dict(self):
        return dict(
            tf=self.tf.to_dict(),
            mr=self.mr.to_dict(),
            tf_index=int(self.tf_index),
            mr_index=int(self.mr_index),
        )

    @classmethod
    def from_dict(cls, value):
        if set(value) != {"tf", "mr", "tf_index", "mr_index"}:
            raise ValueError("Explicit original constituents and canonical indices required")
        return cls(
            core.Candidate.from_dict(value["tf"]),
            mr.Candidate.from_dict(value["mr"]),
            value["tf_index"],
            value["mr_index"],
        )


def pair_grid(tf_candidates, mr_candidates, *, require_full=True):
    if not isinstance(require_full, bool):
        raise ValueError("Explicit full-grid/test-grid boolean required")
    trends, reversions = tuple(tf_candidates), tuple(mr_candidates)
    if (
        not trends
        or not reversions
        or not all(isinstance(x, core.Candidate) for x in trends)
        or not all(isinstance(x, mr.Candidate) for x in reversions)
        or len({x.candidate_id for x in trends}) != len(trends)
        or len({x.candidate_id for x in reversions}) != len(reversions)
    ):
        raise ValueError("Unique unchanged nonempty TF/MR universes required")
    if require_full and (len(trends) != 12 or len(reversions) != 32):
        raise ValueError("The frozen study requires exactly 12 TF and 32 MR members")
    trends = tuple(sorted(trends, key=lambda c: c.candidate_id))
    reversions = tuple(sorted(reversions, key=lambda c: c.candidate_id))
    return tuple(
        Pair(tf, reversal, i, j)
        for i, tf in enumerate(trends)
        for j, reversal in enumerate(reversions)
    )


def efficiency_ratio(values, period=20):
    if not core._positive_integer(period) or period != 20:
        raise ValueError("The ER period is frozen at 20 finalized daily changes")
    series = pd.Series(values, copy=True, dtype=float)
    out = np.full(len(series), np.nan)
    array = series.to_numpy()
    for i in range(20, len(array)):
        window = array[i - 20 : i + 1]
        if np.isfinite(window).all() and (window > 0).all():
            denominator = np.abs(np.diff(window)).sum()
            out[i] = 0 if denominator == 0 else abs(window[-1] - window[0]) / denominator
    return pd.Series(out, series.index, name="ER20")


@dataclass(frozen=True)
class HybridCache:
    tf: quality.QualitySupportCache
    mr: mr.MeanReversionCache
    finalized_er: pd.Series
    er_fingerprint: str = ""
    test_gate_override: bool | None = None

    def __getattr__(self, name):
        return getattr(self.tf, name)


def _er_fingerprint(cache):
    digest = hashlib.sha256()
    digest.update(cache.tf.feature_fingerprint.encode())
    digest.update(cache.mr.feature_fingerprint.encode())
    digest.update(json.dumps([list(cache.finalized_er.index), cache.test_gate_override]).encode())
    digest.update(np.ascontiguousarray(cache.finalized_er.to_numpy(float)).tobytes())
    return digest.hexdigest()


def build_cache(
    daily, bars, calendar, *, coverage=None, blackouts=None, quality_policy=quality.DEFAULT_POLICY
):
    kwargs = dict(coverage=coverage, blackouts=blackouts, quality_policy=quality_policy)
    tf = quality.build_quality_support_cache(daily, bars, calendar, **kwargs)
    reversal = mr_engine.build_cache(daily, bars, calendar, **kwargs)
    if tf.input_fingerprint != reversal.input_fingerprint or not np.array_equal(
        tf.finalized_total_return_index, reversal.finalized_total_return_index
    ):
        raise ValueError("Constituent data clocks and causal signal indices must agree exactly")
    cache = HybridCache(tf, reversal, efficiency_ratio(tf.finalized_total_return_index))
    return replace(cache, er_fingerprint=_er_fingerprint(cache))


def _validate_cache(cache):
    if not isinstance(cache, HybridCache):
        raise ValueError("Separate HybridCache required")
    quality._validate_runtime_cache(
        cache.tf,
        cache.daily,
        cache.bars,
        cache.calendar,
        cache.coverage,
        cache.blackouts,
        cache.policy,
    )
    mr_engine._validate_cache(cache.mr)
    if (
        cache.tf.input_fingerprint != cache.mr.input_fingerprint
        or not cache.finalized_er.index.equals(cache.tf.finalized_total_return_index.index)
        or not cache.er_fingerprint
        or cache.er_fingerprint != _er_fingerprint(cache)
    ):
        raise ValueError("Hybrid source or finalized ER identity changed")
    finite = cache.finalized_er.dropna().to_numpy()
    if not np.isfinite(finite).all() or ((finite < 0) | (finite > 1 + 1e-12)).any():
        raise ValueError("ER20 must be UNKNOWN or a finite value in [0,1]")


def with_test_tf_gate(cache, state):
    """Synthetic parity override only; production protocols must reject its marker."""
    _validate_cache(cache)
    if not isinstance(state, bool):
        raise ValueError("Explicit boolean test gate required")
    result = replace(cache, test_gate_override=state, er_fingerprint="")
    return replace(result, er_fingerprint=_er_fingerprint(result))


@dataclass(frozen=True)
class ShadowPaths:
    pairs: tuple[Pair, ...]
    tf_ids: tuple[str, ...]
    mr_ids: tuple[str, ...]
    clock: pd.DatetimeIndex
    daily_times: pd.DatetimeIndex
    open_times: pd.DatetimeIndex
    open_bar_indices: np.ndarray
    tf_open_state: np.ndarray
    mr_open_state: np.ndarray
    mode_open_state: np.ndarray
    er_open_valid: np.ndarray
    tf_open_decision_at: np.ndarray
    mr_open_decision_at: np.ndarray
    tf_open_decision_bar: np.ndarray
    mr_open_decision_bar: np.ndarray
    mode_open_decision_at: np.ndarray
    mode_open_decision_bar: np.ndarray
    mr_open_cause: np.ndarray
    mode_event_state: np.ndarray
    er_event_valid: np.ndarray
    shadow_trades: pd.DataFrame
    diagnostics: pd.DataFrame
    trace: dict
    pair_diagnostics: tuple[dict, ...]
    post_gap_by_bar: dict
    start: str
    end: str
    input_fingerprint: str
    feature_fingerprint: str
    test_only: bool = False
    state_fingerprint: str = ""
    post_gap_windows: tuple = ()
    tf_diagnostics: tuple = ()

    @property
    def pair_ids(self):
        return tuple(p.pair_id for p in self.pairs)


def build_shadow_paths(
    cache, tf_candidates, mr_candidates, *, start, end, record=True, require_full=True
):
    """Generate cost-independent filled shadow states and prior-close intents."""
    _validate_cache(cache)
    if not isinstance(record, bool):
        raise ValueError("Explicit boolean recording policy required")
    pairs = pair_grid(tf_candidates, mr_candidates, require_full=require_full)
    trends = tuple({p.tf_index: p.tf for p in pairs}.values())
    reversions = tuple({p.mr_index: p.mr for p in pairs}.values())
    nt, n = len(trends), len(pairs)
    ti = np.asarray([p.tf_index for p in pairs])
    tf_lookup = {rule: i for i, rule in enumerate(cache.tf.rules)}
    entry_indices = np.asarray([tf_lookup[c.entry_rule] for c in trends])
    exit_indices = np.asarray([tf_lookup[c.exit_rule] for c in trends])
    mr_lookup = {rule: i for i, rule in enumerate(cache.mr.rules)}
    mr_indices = np.asarray([mr_lookup[p.mr.rule] for p in pairs])
    entries = np.asarray([p.mr.entry_threshold for p in pairs])
    exits = np.asarray([p.mr.exit_threshold for p in pairs])
    calendar, events, clock, terminal = quality._segment_events(cache.tf, start, end)
    tf_long = np.zeros(nt, dtype=bool)
    if cache.test_gate_override is not None:
        tf_long[:] = cache.test_gate_override
    mr_long = np.zeros(n, dtype=bool)
    tf_entry, tf_exit, mr_entry, mr_exit = (np.zeros(k, dtype=int) for k in (nt, nt, n, n))
    tf_pending, mr_pending = np.zeros(nt, dtype=np.int8), np.zeros(n, dtype=np.int8)
    tf_at, mr_at = np.full(nt, NAT, dtype=np.int64), np.full(n, NAT, dtype=np.int64)
    tf_bar, mr_bar = np.full(nt, -1, dtype=int), np.full(n, -1, dtype=int)
    mr_cause = np.zeros(n, dtype=np.int8)
    mode, mode_count, pending_mode, mode_at, mode_bar = TF_MODE, 0, -1, NAT, -1
    available = pd.DatetimeIndex(
        [quality._valuation_clock(s, cache.policy) for s in cache.daily.session]
    )
    initial = available.searchsorted(clock[0], side="right") - 1
    er = float(cache.finalized_er.iloc[initial]) if initial >= 0 else np.nan
    daily_positions = dict(zip(cache.daily.session, range(len(cache.daily)), strict=True))
    records, diagnostics = [], []
    opens, open_bars, tf_states, mr_states, modes, er_valid = [], [], [], [], [], []
    (
        tf_decisions,
        mr_decisions,
        tf_decision_bars,
        mr_decision_bars,
        mode_decisions,
        mode_decision_bars,
        mr_causes,
    ) = [], [], [], [], [], [], []
    trace = {
        name: []
        for name in (
            "bar_index",
            "timestamp_ns",
            "tf_entry_count",
            "tf_exit_count",
            "tf_long",
            "tf_pending",
            "mr_entry_count",
            "mr_exit_count",
            "mr_long",
            "mr_pending",
            "mode",
            "mode_count",
            "pending_mode",
            "er",
        )
    }
    blocked = np.zeros(n, dtype=int)
    forced = np.zeros(n, dtype=int)
    entry_cancels = np.zeros(n, dtype=int)
    blackout_cancels = np.zeros(n, dtype=int)
    terminal_cancels = np.zeros(n, dtype=int)
    tf_blackout_cancels = np.zeros(nt, dtype=int)
    tf_terminal_cancels = np.zeros(nt, dtype=int)
    mode_switches, mode_cancellations = 0, 0
    clock_modes, clock_er = [], []
    post = (
        quality._post_gap_context(cache.tf, calendar)
        if cache.policy.post_gap_diagnostics
        else {"by_bar": {}}
    )
    events_by_time = {}
    for event in events:
        events_by_time.setdefault(event[0], []).append(event)

    def wanted_mode():
        if not np.isfinite(er):
            return -1
        if mode == TF_MODE:
            return MR_MODE if er <= 0.25 else TF_MODE
        return TF_MODE if er >= 0.35 else MR_MODE

    def shadow_record(kind, j, side, timestamp, bar_index, at, decision_bar, cause, phase="open"):
        if not record:
            return
        p = pairs[j] if kind == "GMR" else None
        records.append(
            dict(
                strategy=kind,
                candidate_id=f"hybrid_gmr__{p.pair_id}" if p else trends[j].candidate_id,
                base_candidate_id=p.mr.candidate_id if p else trends[j].candidate_id,
                pair_id=p.pair_id if p else None,
                trend_candidate_id=p.tf.candidate_id if p else trends[j].candidate_id,
                side=side,
                timestamp=timestamp,
                phase=phase,
                decision_at=None if at == NAT else pd.Timestamp(int(at), tz="UTC"),
                decision_bar=int(decision_bar),
                execution_bar=int(bar_index),
                cause=cause,
                state=side == "buy",
                post_gap_flag=int(decision_bar) in post["by_bar"],
                post_gap_windows=json.dumps(
                    [
                        [x.isoformat() for x in post["windows"][g]]
                        for g in post["by_bar"].get(int(decision_bar), ())
                    ]
                )
                if "windows" in post
                else "[]",
            )
        )

    for timestamp in clock:
        for _, _, kind, item in events_by_time[timestamp]:
            if kind == "blackout_start":
                blackout_cancels += mr_pending != 0
                tf_blackout_cancels += tf_pending != 0
                if record:
                    diagnostics.append(
                        dict(
                            timestamp=timestamp,
                            event="blackout_cancel_reset",
                            pending_tf=int((tf_pending != 0).sum()),
                            pending_mr=int((mr_pending != 0).sum()),
                            pending_mode=int(pending_mode >= 0),
                        )
                    )
                tf_pending[:], mr_pending[:], mr_cause[:] = 0, 0, 0
                tf_entry[:], tf_exit[:], mr_entry[:], mr_exit[:] = 0, 0, 0, 0
                tf_at[:], mr_at[:] = NAT, NAT
                tf_bar[:], mr_bar[:] = -1, -1
                mode_cancellations += pending_mode >= 0
                pending_mode, mode_count, mode_at, mode_bar = -1, 0, NAT, -1
            elif kind == "bar_open" and cache.fill_eligible[item]:
                tf_decision = np.full(nt, NAT, dtype=np.int64)
                mr_decision = np.full(n, NAT, dtype=np.int64)
                tf_db, mr_db = np.full(nt, -1, dtype=int), np.full(n, -1, dtype=int)
                causes = np.zeros(n, dtype=np.int8)
                changed_tf = tf_pending != 0
                projected_tf = tf_long.copy()
                if cache.test_gate_override is None:
                    projected_tf[tf_pending == 1] = True
                    projected_tf[tf_pending == -1] = False
                exiting_tf = tf_long & ~projected_tf
                tf_decision[changed_tf], tf_db[changed_tf] = tf_at[changed_tf], tf_bar[changed_tf]
                for j in np.flatnonzero(changed_tf):
                    shadow_record(
                        "TF",
                        j,
                        "buy" if projected_tf[j] else "sell",
                        timestamp,
                        item,
                        tf_at[j],
                        tf_bar[j],
                        "confirmed_entry" if projected_tf[j] else "confirmed_exit",
                    )
                force = mr_long & ~projected_tf[ti]
                cancel_buy = (mr_pending == 1) & ~projected_tf[ti]
                entry_cancels += cancel_buy
                mr_pending[cancel_buy] = 0
                sells = mr_long & ((mr_pending == -1) | force)
                buys = ~mr_long & (mr_pending == 1) & projected_tf[ti]
                changed_mr = sells | buys
                causes[sells] = mr_cause[sells]
                causes[force] |= 2
                forced += force
                mr_decision[changed_mr], mr_db[changed_mr] = mr_at[changed_mr], mr_bar[changed_mr]
                mr_decision[force], mr_db[force] = tf_at[ti[force]], tf_bar[ti[force]]
                for j in np.flatnonzero(changed_mr):
                    cause = (
                        "confirmed_entry"
                        if buys[j]
                        else "recovery+tf_exit"
                        if causes[j] == 3
                        else "tf_forced_exit"
                        if causes[j] & 2
                        else "confirmed_recovery"
                    )
                    shadow_record(
                        "GMR",
                        j,
                        "buy" if buys[j] else "sell",
                        timestamp,
                        item,
                        mr_decision[j],
                        mr_db[j],
                        cause,
                    )
                tf_long[:] = projected_tf
                mr_long[sells], mr_long[buys] = False, True
                reset_mr = changed_mr | ~tf_long[ti] | exiting_tf[ti]
                mr_entry[reset_mr], mr_exit[reset_mr] = 0, 0
                mr_pending[changed_mr | ~tf_long[ti]], mr_cause[changed_mr | ~tf_long[ti]] = 0, 0
                mr_at[changed_mr | ~tf_long[ti]], mr_bar[changed_mr | ~tf_long[ti]] = NAT, -1
                tf_entry[changed_tf], tf_exit[changed_tf] = 0, 0
                tf_pending[changed_tf], tf_at[changed_tf], tf_bar[changed_tf] = 0, NAT, -1
                applied_mode_at, applied_mode_bar = NAT, -1
                if pending_mode >= 0:
                    if wanted_mode() == pending_mode and np.isfinite(er):
                        mode = pending_mode
                        mode_switches += 1
                        applied_mode_at, applied_mode_bar = mode_at, mode_bar
                        if record:
                            diagnostics.append(
                                dict(
                                    timestamp=timestamp,
                                    event="regime_switch_filled",
                                    mode=mode,
                                    er=er,
                                    decision_at=pd.Timestamp(int(mode_at), tz="UTC"),
                                    decision_bar=mode_bar,
                                    execution_bar=item,
                                )
                            )
                    else:
                        mode_cancellations += 1
                    pending_mode, mode_count, mode_at, mode_bar = -1, 0, NAT, -1
                opens.append(timestamp)
                open_bars.append(item)
                tf_states.append(tf_long.copy())
                mr_states.append(mr_long.copy())
                modes.append(mode)
                er_valid.append(np.isfinite(er))
                tf_decisions.append(tf_decision)
                mr_decisions.append(mr_decision)
                tf_decision_bars.append(tf_db)
                mr_decision_bars.append(mr_db)
                mode_decisions.append(applied_mode_at)
                mode_decision_bars.append(applied_mode_bar)
                mr_causes.append(causes)
            elif kind == "bar_close":
                if cache.signal_eligible[item]:
                    values = cache.tf.margins[item]
                    tf_entry_ok = (values[entry_indices] > 0) & (values[exit_indices] > 0)
                    tf_exit_ok = np.isfinite(values[exit_indices]) & ~(values[exit_indices] > 0)
                    if cache.test_gate_override is None:
                        tf_entry[:] = np.where(~tf_long & tf_entry_ok, tf_entry + 1, 0)
                        tf_exit[:] = np.where(tf_long & tf_exit_ok, tf_exit + 1, 0)
                        tb, ts = ~tf_long & (tf_entry >= 7), tf_long & (tf_exit >= 7)
                        tf_pending[tb], tf_pending[ts] = 1, -1
                        tf_at[tb | ts], tf_bar[tb | ts] = timestamp.value, item
                    mr_values = cache.mr.margins[item, mr_indices]
                    valid = np.isfinite(mr_values)
                    oversold = valid & (mr_values <= entries)
                    recovery = valid & (mr_values >= exits)
                    permitted = tf_long[ti] & (tf_pending[ti] != -1)
                    blocked += ~mr_long & oversold & ~permitted
                    mr_entry[:] = np.where(~mr_long & permitted & oversold, mr_entry + 1, 0)
                    mr_exit[:] = np.where(mr_long & valid & recovery, mr_exit + 1, 0)
                    cancel_buys = (mr_pending == 1) & ~permitted
                    entry_cancels += cancel_buys
                    mr_pending[cancel_buys], mr_cause[cancel_buys] = 0, 0
                    mb, ms = ~mr_long & (mr_entry >= 7), mr_long & (mr_exit >= 7)
                    forced_pending = mr_long & (tf_pending[ti] == -1)
                    mr_pending[mb], mr_pending[ms | forced_pending] = 1, -1
                    mr_cause[ms] |= 1
                    mr_cause[forced_pending] |= 2
                    mr_at[mb | ms | forced_pending], mr_bar[mb | ms | forced_pending] = (
                        timestamp.value,
                        item,
                    )
                    wanted = wanted_mode()
                    mode_count = mode_count + 1 if wanted >= 0 and wanted != mode else 0
                    if mode_count >= 7:
                        pending_mode, mode_at, mode_bar = wanted, timestamp.value, item
                if record:
                    row = {
                        "bar_index": item,
                        "timestamp_ns": timestamp.value,
                        "tf_entry_count": tf_entry.copy(),
                        "tf_exit_count": tf_exit.copy(),
                        "tf_long": tf_long.copy(),
                        "tf_pending": tf_pending.copy(),
                        "mr_entry_count": mr_entry.copy(),
                        "mr_exit_count": mr_exit.copy(),
                        "mr_long": mr_long.copy(),
                        "mr_pending": mr_pending.copy(),
                        "mode": mode,
                        "mode_count": mode_count,
                        "pending_mode": pending_mode,
                        "er": er,
                    }
                    for name, value in row.items():
                        trace[name].append(value)
                if item == terminal:
                    terminal_cancels += mr_pending != 0
                    tf_terminal_cancels += tf_pending != 0
                    for j in np.flatnonzero(tf_long):
                        shadow_record(
                            "TF",
                            j,
                            "sell",
                            timestamp,
                            item,
                            NAT,
                            -1,
                            "terminal_liquidation",
                            "terminal",
                        )
                    for j in np.flatnonzero(mr_long):
                        shadow_record(
                            "GMR",
                            j,
                            "sell",
                            timestamp,
                            item,
                            NAT,
                            -1,
                            "terminal_liquidation",
                            "terminal",
                        )
                    tf_long[:], mr_long[:] = False, False
                    tf_pending[:], mr_pending[:], mr_cause[:] = 0, 0, 0
                    pending_mode, mode_count = -1, 0
            elif kind == "valuation":
                er = float(cache.finalized_er.iloc[daily_positions[item]])
                wanted = wanted_mode()
                if not np.isfinite(er) or wanted == mode:
                    mode_count = 0
                if pending_mode >= 0 and wanted != pending_mode:
                    mode_cancellations += 1
                    if record:
                        diagnostics.append(
                            dict(
                                timestamp=timestamp,
                                event="regime_daily_cancelled",
                                pending_mode=pending_mode,
                                er=er,
                            )
                        )
                    pending_mode, mode_at, mode_bar = -1, NAT, -1
        clock_modes.append(mode)
        clock_er.append(np.isfinite(er))
    pair_diag = tuple(
        dict(
            pair_id=p.pair_id,
            blocked_oversold_checks=int(blocked[j]),
            forced_tactical_exit_count=int(forced[j]),
            entry_cancellation_count=int(entry_cancels[j]),
            cancelled_blackout_order_count=int(blackout_cancels[j]),
            cancelled_terminal_order_count=int(terminal_cancels[j]),
            regime_switch_count=mode_switches,
            regime_switch_cancellation_count=mode_cancellations,
        )
        for j, p in enumerate(pairs)
    )
    day_times = pd.DatetimeIndex(
        [quality._valuation_clock(s, cache.policy) for s in calendar.session]
    )
    result = ShadowPaths(
        pairs,
        tuple(c.candidate_id for c in trends),
        tuple(c.candidate_id for c in reversions),
        clock,
        day_times,
        pd.DatetimeIndex(opens),
        np.asarray(open_bars),
        np.asarray(tf_states, dtype=bool).reshape(-1, nt),
        np.asarray(mr_states, dtype=bool).reshape(-1, n),
        np.asarray(modes, dtype=np.int8),
        np.asarray(er_valid, dtype=bool),
        np.asarray(tf_decisions, dtype=np.int64).reshape(-1, nt),
        np.asarray(mr_decisions, dtype=np.int64).reshape(-1, n),
        np.asarray(tf_decision_bars).reshape(-1, nt),
        np.asarray(mr_decision_bars).reshape(-1, n),
        np.asarray(mode_decisions, dtype=np.int64),
        np.asarray(mode_decision_bars),
        np.asarray(mr_causes, dtype=np.int8).reshape(-1, n),
        np.asarray(clock_modes, dtype=np.int8),
        np.asarray(clock_er, dtype=bool),
        pd.DataFrame(records, columns=SHADOW_TRADE_COLUMNS),
        pd.DataFrame(diagnostics) if diagnostics else pd.DataFrame(columns=["timestamp", "event"]),
        {
            key: np.asarray(values, dtype=np.int16 if key.endswith("_count") else None)
            for key, values in trace.items()
        },
        pair_diag,
        post["by_bar"],
        str(start),
        str(end),
        cache.input_fingerprint,
        cache.er_fingerprint,
        not require_full or cache.test_gate_override is not None,
    )
    tf_diag = tuple(
        dict(
            candidate_id=c.candidate_id,
            cancelled_blackout_order_count=int(tf_blackout_cancels[j]),
            cancelled_terminal_order_count=int(tf_terminal_cancels[j]),
        )
        for j, c in enumerate(trends)
    )
    result = replace(result, post_gap_windows=post.get("windows", ()), tf_diagnostics=tf_diag)
    return replace(result, state_fingerprint=_shadow_fingerprint(result))


def _shadow_fingerprint(paths):
    digest = hashlib.sha256()
    digest.update(json.dumps([p.to_dict() for p in paths.pairs], sort_keys=True).encode())
    digest.update(
        json.dumps([paths.start, paths.end, paths.feature_fingerprint, paths.test_only]).encode()
    )
    for field in ("clock", "daily_times", "open_times"):
        digest.update(np.ascontiguousarray(getattr(paths, field).asi8).tobytes())
    for field in (
        "open_bar_indices",
        "tf_open_state",
        "mr_open_state",
        "mode_open_state",
        "er_open_valid",
        "tf_open_decision_at",
        "mr_open_decision_at",
        "tf_open_decision_bar",
        "mr_open_decision_bar",
        "mode_open_decision_at",
        "mode_open_decision_bar",
        "mr_open_cause",
        "mode_event_state",
        "er_event_valid",
    ):
        value = np.asarray(getattr(paths, field))
        digest.update(json.dumps([field, value.shape, str(value.dtype)]).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    for field, value in paths.trace.items():
        digest.update(field.encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    for frame in (paths.shadow_trades, paths.diagnostics):
        digest.update(json.dumps(list(frame.columns)).encode())
        digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    digest.update(json.dumps(paths.pair_diagnostics, sort_keys=True).encode())
    digest.update(json.dumps(paths.tf_diagnostics, sort_keys=True).encode())
    digest.update(
        json.dumps([[x.isoformat() for x in window] for window in paths.post_gap_windows]).encode()
    )
    return digest.hexdigest()


@dataclass(frozen=True)
class RunResult:
    portfolio_ids: tuple[str, ...]
    metrics: tuple[dict, ...]
    daily_times: pd.DatetimeIndex
    daily_equity: np.ndarray
    daily_returns: np.ndarray
    cash_returns: np.ndarray
    daily_exposure: np.ndarray
    event_times: pd.DatetimeIndex
    event_equity: np.ndarray
    event_exposure: np.ndarray
    account_ids: tuple[str, ...]
    account_groups: tuple[str, ...]
    account_equity: np.ndarray
    account_returns: np.ndarray
    account_exposure: np.ndarray
    account_metrics: tuple[dict, ...]
    account_gross_terminal: np.ndarray
    account_gross_equity: np.ndarray
    trade_rows: pd.DataFrame
    command_rows: pd.DataFrame
    pair_metrics: pd.DataFrame
    annual_returns: pd.DataFrame
    diagnostics: pd.DataFrame


def _series_stats(equity, cash_returns, years, initial_cash):
    equity = np.asarray(equity, dtype=float)
    if equity.ndim == 1:
        equity = equity[:, None]
    returns = equity / np.vstack([np.full(equity.shape[1], initial_cash), equity[:-1]]) - 1
    excess = returns - cash_returns[:, None]
    sd = np.std(excess, axis=0, ddof=1) if len(returns) > 1 else np.full(equity.shape[1], np.nan)
    score = np.full(len(sd), np.nan)
    valid = np.isfinite(sd) & (sd > 0)
    score[valid] = np.sqrt(252) * np.mean(excess[:, valid], axis=0) / sd[valid]
    wealth = np.vstack([np.full(equity.shape[1], initial_cash), equity])
    drawdown = np.min(wealth / np.maximum.accumulate(wealth, axis=0) - 1, axis=0)
    volatility = (
        np.std(returns, axis=0, ddof=1) * np.sqrt(252)
        if len(returns) > 1
        else np.full(equity.shape[1], np.nan)
    )
    return returns, tuple(
        dict(
            cash_excess_sharpe=float(score[j]),
            cagr=float((equity[-1, j] / initial_cash) ** (1 / years) - 1),
            max_drawdown=float(drawdown[j]),
            annualized_volatility=float(volatility[j]),
            terminal_equity=float(equity[-1, j]),
            total_return=float(equity[-1, j] / initial_cash - 1),
        )
        for j in range(equity.shape[1])
    )


def _account_layout(shadows):
    nt, n = len(shadows.tf_ids), len(shadows.pairs)
    ids = [*shadows.tf_ids, *(f"hybrid_gmr__{p.pair_id}" for p in shadows.pairs)]
    groups = ["TF"] * nt + ["GMR"] * n
    offsets = [0, nt]
    for alpha in ALPHAS:
        label = f"B{round(alpha * 100):03d}"
        offsets.append(len(ids))
        ids.extend(f"hybrid_blend_{round(alpha * 100):03d}__{p.pair_id}" for p in shadows.pairs)
        groups.extend([label] * n)
    offsets.append(len(ids))
    ids.extend(f"hybrid_regime_er20_025_035__{p.pair_id}" for p in shadows.pairs)
    groups.extend(["C"] * n)
    return tuple(ids), tuple(groups), tuple(offsets)


def _affine_mean(values):
    return values[0] + np.mean(values - values[0])


def _portfolio_values(values, nt, n, offsets):
    tf_value, mr_value = _affine_mean(values[:nt]), _affine_mean(values[nt : nt + n])
    blended = [_affine_mean(values[offset : offset + n]) for offset in offsets[2:5]]
    router = _affine_mean(values[offsets[5] : offsets[5] + n])
    return np.asarray(
        [
            *(tf_value + (1 - a) * (mr_value - tf_value) for a in ALPHAS),
            *blended,
            router,
            tf_value,
            mr_value,
        ]
    )


def _weights(nt, n, offsets):
    weights = np.zeros((9, nt + 5 * n))
    for j, a in enumerate(ALPHAS):
        weights[j, :nt] = a / nt
        weights[j, nt : nt + n] = (1 - a) / n
        weights[j + 3, offsets[j + 2] : offsets[j + 2] + n] = 1 / n
    weights[6, offsets[5] : offsets[5] + n] = 1 / n
    weights[7, :nt] = 1 / nt
    weights[8, nt : nt + n] = 1 / n
    return weights


def pair_daily_equity(result, shadows, portfolio_id):
    """Reconstruct exact development-defined paired curves without new accounting."""
    nt, n = len(shadows.tf_ids), len(shadows.pairs)
    ti = np.asarray([p.tf_index for p in shadows.pairs])
    _, _, offsets = _account_layout(shadows)
    if portfolio_id.startswith("hybrid_core_"):
        a = int(portfolio_id.rsplit("_", 1)[1]) / 100
        if a not in ALPHAS:
            raise ValueError("Unknown fixed-capital allocation")
        tf_eq, mr_eq = result.account_equity[:, ti], result.account_equity[:, nt : nt + n]
        return tf_eq + (1 - a) * (mr_eq - tf_eq)
    if portfolio_id.startswith("hybrid_blend_"):
        a = int(portfolio_id.rsplit("_", 1)[1]) / 100
        if a not in ALPHAS:
            raise ValueError("Unknown blend allocation")
        offset = offsets[2 + ALPHAS.index(a)]
    elif portfolio_id == "hybrid_regime_er20_025_035":
        offset = offsets[5]
    elif portfolio_id == "hybrid_gated_mr":
        offset = nt
    elif portfolio_id == "portfolio_tf_100":
        return result.account_equity[:, ti]
    else:
        raise ValueError("Unknown current-study portfolio")
    return result.account_equity[:, offset : offset + n]


def run(
    cache,
    shadows,
    *,
    cash_observations,
    commission_bps=1,
    slippage_bps=3,
    initial_cash=1000,
    record=True,
):
    """Stream actually funded accounts and paired baskets on one sealed event clock.

    No account-by-event matrices are retained. Daily account paths, nine principal
    event curves, actual order/target records and compact shadow traces provide
    independent replay without charging virtual model activity.
    """
    _validate_cache(cache)
    if (
        not isinstance(shadows, ShadowPaths)
        or shadows.input_fingerprint != cache.input_fingerprint
        or shadows.feature_fingerprint != cache.er_fingerprint
        or not shadows.state_fingerprint
        or shadows.state_fingerprint != _shadow_fingerprint(shadows)
        or initial_cash != 1000
        or not isinstance(record, bool)
    ):
        raise ValueError("Intact matching shadow paths and $1000 normalization required")
    calendar, events, clock, terminal = quality._segment_events(
        cache.tf, shadows.start, shadows.end
    )
    if not clock.equals(shadows.clock):
        raise ValueError("Shadow/financial clocks changed")
    nt, n = len(shadows.tf_ids), len(shadows.pairs)
    ti = np.asarray([p.tf_index for p in shadows.pairs])
    ids, groups, offsets = _account_layout(shadows)
    weights = _weights(nt, n, offsets)
    book = TargetBook(
        ids,
        groups,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        initial_cash=initial_cash,
    )
    gross = TargetBook(ids, groups, commission_bps=0, slippage_bps=0, initial_cash=initial_cash)
    accrual = rate_accrual(clock, cash_observations, interval_start=clock[0]).to_numpy()
    if not np.isfinite(accrual).all() or (accrual <= -1).any():
        raise ValueError("Invalid causal cash clock")
    open_lookup = {int(bar): k for k, bar in enumerate(shadows.open_bar_indices)}
    daily = {row.session: row for row in cache.daily.itertuples(index=False)}
    events_by_time = {}
    for event in events:
        events_by_time.setdefault(event[0], []).append(event)
    day_eq, day_gross, day_exp, day_portfolio, day_pexp, day_cash = [], [], [], [], [], []
    event_eq, event_exp = [], []
    exposure_seconds = np.zeros(len(ids))
    pair_a_exposure = np.zeros((3, n))
    pair_a_turnover = np.zeros((3, n))
    principal_turnover, principal_orders, principal_fees, principal_slips = (
        np.zeros(9),
        np.zeros(9, dtype=int),
        np.zeros(9),
        np.zeros(9),
    )
    principal_timestamps = [set() for _ in range(9)]
    post_gap_orders = np.zeros(9, dtype=int)
    portfolio_episodes = np.zeros(9, dtype=int)
    portfolio_holds = np.zeros(9)
    portfolio_entered = np.full(9, np.nan)
    previous_portfolio_long = np.zeros(9, dtype=bool)
    c_blocked = np.zeros(n, dtype=int)
    previous_account_exposure = np.zeros(len(ids))
    previous_a_exposure = np.zeros((3, n))
    previous_mode = TF_MODE
    signal_component_seconds = np.zeros((3, 2))
    previous_signal_components = np.zeros((3, 2))
    mode_seconds, unknown_seconds = np.zeros(2), 0.0
    diagnostics = []
    cash_reference, mark = initial_cash, None

    def execute_group(
        targets, mask, price, timestamp, *, phase, reasons, decision_at, decision_bar, execution_bar
    ):
        trade_start, command_start = len(book.trade_rows), len(book.command_rows)
        before, notional, fees, slips, actual = book.execute(
            targets,
            mask,
            price,
            timestamp,
            phase=phase,
            reasons=reasons,
            decision_at=decision_at,
            decision_bar=decision_bar,
            execution_bar=execution_bar,
            record=record,
        )
        gross.execute(targets, mask, price, timestamp, phase=phase, record=False)
        if record:
            for row in (*book.trade_rows[trade_start:], *book.command_rows[command_start:]):
                gaps = shadows.post_gap_by_bar.get(int(row["decision_bar"]), ())
                row["post_gap_flag"] = bool(gaps)
                row["post_gap_windows"] = json.dumps(
                    [[x.isoformat() for x in shadows.post_gap_windows[g]] for g in gaps]
                )
        pre_portfolio = _portfolio_values(before, nt, n, offsets)
        principal_turnover[:] += weights @ notional / pre_portfolio
        principal_orders[:] += (weights > 0) @ actual.astype(int)
        principal_fees[:] += weights @ fees
        principal_slips[:] += weights @ slips
        for j in range(9):
            if np.any(actual & (weights[j] > 0)):
                principal_timestamps[j].add(timestamp)
                flagged = (
                    actual
                    & (weights[j] > 0)
                    & np.isin(decision_bar, tuple(shadows.post_gap_by_bar))
                )
                post_gap_orders[j] += int(flagged.sum())
        for j, a in enumerate(ALPHAS):
            denominator = before[ti] + (1 - a) * (before[nt : nt + n] - before[ti])
            pair_a_turnover[j] += (a * notional[ti] + (1 - a) * notional[nt : nt + n]) / denominator

    for k, timestamp in enumerate(clock):
        if k:
            elapsed = (timestamp - clock[k - 1]).total_seconds()
            exposure_seconds += previous_account_exposure * elapsed
            pair_a_exposure += previous_a_exposure * elapsed
            signal_component_seconds += previous_signal_components * elapsed
            mode_seconds[previous_mode] += elapsed
            unknown_seconds += (not shadows.er_event_valid[k - 1]) * elapsed
        book.accrue(accrual[k])
        gross.accrue(accrual[k])
        cash_reference *= 1 + accrual[k]
        for _, _, kind, item in events_by_time[timestamp]:
            if kind == "actions":
                action = daily[item]
                if mark is not None:
                    mark /= action.split_coefficient
                book.actions(action.split_coefficient, action.dividend_amount)
                gross.actions(action.split_coefficient, action.dividend_amount)
            elif kind == "bar_open":
                row = cache.bar_rows[item]
                mark = float(row.open)
                if item in open_lookup:
                    op = open_lookup[item]
                    book.drip(mark)
                    gross.drip(mark)
                    tf_state, mr_state = shadows.tf_open_state[op], shadows.mr_open_state[op]
                    targets = np.zeros(len(ids))
                    targets[:nt], targets[nt : nt + n] = tf_state, mr_state
                    for j, a in enumerate(ALPHAS):
                        targets[offsets[j + 2] : offsets[j + 2] + n] = (
                            a * tf_state[ti] + (1 - a) * mr_state
                        )
                    incoming = tf_state[ti] if shadows.mode_open_state[op] == TF_MODE else mr_state
                    targets[offsets[5] : offsets[5] + n] = incoming
                    if not shadows.er_open_valid[op]:
                        risk_add = incoming & (book.targets[offsets[5] : offsets[5] + n] == 0)
                        c_blocked += risk_add
                        targets[offsets[5] : offsets[5] + n][risk_add] = 0
                        if record and risk_add.any():
                            diagnostics.append(
                                dict(
                                    timestamp=timestamp,
                                    event="regime_unknown_blocked_risk_add",
                                    blocked_pair_count=int(risk_add.sum()),
                                    execution_bar=item,
                                )
                            )
                    # Only discrete confirmed-target changes command rebalancing.
                    mask = targets != book.targets
                    reasons = np.full(len(ids), "confirmed_state_target", dtype=object)
                    at, db = (
                        np.full(len(ids), NAT, dtype=np.int64),
                        np.full(len(ids), -1, dtype=int),
                    )
                    at[:nt], db[:nt] = (
                        shadows.tf_open_decision_at[op],
                        shadows.tf_open_decision_bar[op],
                    )
                    at[nt : nt + n], db[nt : nt + n] = (
                        shadows.mr_open_decision_at[op],
                        shadows.mr_open_decision_bar[op],
                    )
                    causes = shadows.mr_open_cause[op]
                    reasons[nt : nt + n][causes == 1] = "confirmed_recovery"
                    reasons[nt : nt + n][causes == 2] = "tf_forced_exit"
                    reasons[nt : nt + n][causes == 3] = "recovery+tf_exit"
                    for offset in offsets[2:]:
                        tf_at = shadows.tf_open_decision_at[op, ti]
                        mr_at = shadows.mr_open_decision_at[op]
                        choose_tf = tf_at >= mr_at
                        at[offset : offset + n] = np.where(choose_tf, tf_at, mr_at)
                        db[offset : offset + n] = np.where(
                            choose_tf,
                            shadows.tf_open_decision_bar[op, ti],
                            shadows.mr_open_decision_bar[op],
                        )
                        forced = ~tf_state[ti] & (book.targets[offset : offset + n] > 0)
                        reasons[offset : offset + n][forced] = "tf_exit_priority"
                    if shadows.mode_open_decision_at[op] != NAT:
                        at[offsets[5] : offsets[5] + n] = shadows.mode_open_decision_at[op]
                        db[offsets[5] : offsets[5] + n] = shadows.mode_open_decision_bar[op]
                        current_reasons = reasons[offsets[5] : offsets[5] + n]
                        reasons[offsets[5] : offsets[5] + n] = np.where(
                            current_reasons == "tf_exit_priority",
                            "regime_handoff+tf_exit_priority",
                            "regime_handoff",
                        )
                    reconciliation = mask[offsets[5] : offsets[5] + n] & (
                        at[offsets[5] : offsets[5] + n] == NAT
                    )
                    reasons[offsets[5] : offsets[5] + n][reconciliation] = "known_er_reconciliation"
                    if reconciliation.any():
                        available = pd.DatetimeIndex(
                            [quality._valuation_clock(s, cache.policy) for s in cache.daily.session]
                        )
                        er_index = available.searchsorted(timestamp, side="right") - 1
                        at[offsets[5] : offsets[5] + n][reconciliation] = (
                            available[er_index].value if er_index >= 0 else NAT
                        )
                    execute_group(
                        targets,
                        mask,
                        mark,
                        timestamp,
                        phase="open",
                        reasons=reasons,
                        decision_at=at,
                        decision_bar=db,
                        execution_bar=item,
                    )
            elif kind == "bar_close":
                row = cache.bar_rows[item]
                mark = float(row.close)
                if item == terminal:
                    targets = np.zeros(len(ids))
                    mask = book.shares > 0
                    execute_group(
                        targets,
                        mask,
                        mark,
                        timestamp,
                        phase="terminal",
                        reasons=np.full(len(ids), "terminal_liquidation", dtype=object),
                        decision_at=np.full(len(ids), NAT, dtype=np.int64),
                        decision_bar=np.full(len(ids), -1, dtype=int),
                        execution_bar=item,
                    )
            elif kind == "valuation":
                mark = float(daily[item].close)
                values, gross_values = book.wealth(mark), gross.wealth(mark)
                account_exp = book.shares * mark / values
                port_values = _portfolio_values(values, nt, n, offsets)
                port_invested = _portfolio_values(book.shares * mark, nt, n, offsets)
                day_eq.append(values.copy())
                day_gross.append(gross_values.copy())
                day_exp.append(account_exp.copy())
                day_portfolio.append(port_values.copy())
                day_pexp.append(port_invested / port_values)
                day_cash.append(cash_reference)
        price = 1 if mark is None else mark
        values = book.wealth(price)
        invested = book.shares * price
        previous_account_exposure = invested / values
        for j, a in enumerate(ALPHAS):
            pair_wealth = values[ti] + (1 - a) * (values[nt : nt + n] - values[ti])
            previous_a_exposure[j] = (
                a * invested[ti] + (1 - a) * invested[nt : nt + n]
            ) / pair_wealth
        port_values = _portfolio_values(values, nt, n, offsets)
        port_invested = _portfolio_values(invested, nt, n, offsets)
        event_eq.append(port_values)
        event_exp.append(port_invested / port_values)
        current_long = port_invested > 1e-11
        entered = ~previous_portfolio_long & current_long
        closed = previous_portfolio_long & ~current_long
        portfolio_entered[entered] = timestamp.value / 1e9
        portfolio_episodes += closed
        portfolio_holds[closed] += (timestamp.value / 1e9 - portfolio_entered[closed]) / 3600
        previous_portfolio_long = current_long
        previous_mode = int(shadows.mode_event_state[k])
        for j, a in enumerate(ALPHAS):
            previous_signal_components[j] = (
                a * np.mean(book.targets[:nt]),
                (1 - a) * np.mean(book.targets[nt : nt + n]),
            )
    if (book.shares != 0).any() or (gross.shares != 0).any():
        raise ValueError("Terminal liquidation failed")
    seconds = (clock[-1] - clock[0]).total_seconds()
    years = seconds / (365 * 86400)
    equity, gross_eq, account_exposure = (
        np.asarray(day_eq),
        np.asarray(day_gross),
        np.asarray(day_exp),
    )
    cash_values = np.asarray(day_cash)
    cash_returns = cash_values / np.r_[initial_cash, cash_values[:-1]] - 1
    account_returns, stats = _series_stats(equity, cash_returns, years, initial_cash)
    account_metrics = []
    for j, row in enumerate(stats):
        account_metrics.append(
            {
                **row,
                "candidate_id": ids[j],
                "account_id": ids[j],
                "account_group": groups[j],
                "commission_bps": float(commission_bps),
                "slippage_bps": float(slippage_bps),
                "exposure_fraction": float(exposure_seconds[j] / seconds),
                "order_count": int(book.orders[j]),
                "orders_per_year": float(book.orders[j] / years),
                "trade_count": int(book.trips[j]),
                "flat_to_flat_episode_count": int(book.trips[j]),
                "average_holding_hours": float(book.holds[j] / book.trips[j])
                if book.trips[j]
                else 0.0,
                "total_commission": float(book.commission[j]),
                "total_slippage": float(book.slippage[j]),
                "total_turnover": float(book.turnover[j]),
                "annualized_turnover": float(book.turnover[j] / years),
                "gross_terminal_equity": float(gross_eq[-1, j]),
                "cost_drag_return": float((gross_eq[-1, j] - equity[-1, j]) / initial_cash),
            }
        )
    principal_equity = np.asarray(day_portfolio)
    principal_exposure = np.asarray(day_pexp)
    principal_returns, principal_stats = _series_stats(
        principal_equity, cash_returns, years, initial_cash
    )
    event_equity, event_exposure = np.asarray(event_eq), np.asarray(event_exp)
    elapsed = np.diff(clock.asi8) / 1e9
    principal_exposure_time = event_exposure[:-1].T @ elapsed / seconds
    gross_portfolio = _portfolio_values(gross_eq[-1], nt, n, offsets)
    medians = []
    pair_rows = []
    for j, a in enumerate(ALPHAS):
        paired_eq = equity[:, ti] + (1 - a) * (equity[:, nt : nt + n] - equity[:, ti])
        _, paired_stats = _series_stats(paired_eq, cash_returns, years, initial_cash)
        for i, (p, row) in enumerate(zip(shadows.pairs, paired_stats, strict=True)):
            tfj, mrj = p.tf_index, nt + i
            paired_gross = gross_eq[-1, tfj] + (1 - a) * (gross_eq[-1, mrj] - gross_eq[-1, tfj])
            pair_rows.append(
                {
                    **row,
                    **shadows.pair_diagnostics[i],
                    "portfolio_id": PRINCIPAL_IDS[j],
                    "architecture": "A",
                    "pair_id": p.pair_id,
                    "trend_candidate_id": p.tf.candidate_id,
                    "mean_reversion_candidate_id": p.mr.candidate_id,
                    "core_weight": a,
                    "exposure_fraction": float(pair_a_exposure[j, i] / seconds),
                    "total_turnover": float(pair_a_turnover[j, i]),
                    "annualized_turnover": float(pair_a_turnover[j, i] / years),
                    "order_count": int(book.orders[tfj] + book.orders[mrj]),
                    "orders_per_year": float((book.orders[tfj] + book.orders[mrj]) / years),
                    "flat_to_flat_episode_count": int(book.trips[tfj]),
                    "average_holding_hours": float(book.holds[tfj] / book.trips[tfj])
                    if book.trips[tfj]
                    else 0,
                    "tactical_average_holding_hours": float(book.holds[mrj] / book.trips[mrj])
                    if book.trips[mrj]
                    else 0,
                    "total_commission": float(
                        a * book.commission[tfj] + (1 - a) * book.commission[mrj]
                    ),
                    "total_slippage": float(a * book.slippage[tfj] + (1 - a) * book.slippage[mrj]),
                    "gross_terminal_equity": float(paired_gross),
                    "cost_drag_return": float((paired_gross - paired_eq[-1, i]) / initial_cash),
                    "commission_bps": float(commission_bps),
                    "slippage_bps": float(slippage_bps),
                }
            )
        medians.append(float(np.median(book.orders[ti] + book.orders[nt : nt + n]) / years))
    for j, offset in enumerate(offsets[2:]):
        pid = PRINCIPAL_IDS[j + 3]
        for i, p in enumerate(shadows.pairs):
            row = account_metrics[offset + i]
            pair_rows.append(
                {
                    **row,
                    **shadows.pair_diagnostics[i],
                    "portfolio_id": pid,
                    "architecture": "B" if j < 3 else "C",
                    "pair_id": p.pair_id,
                    "trend_candidate_id": p.tf.candidate_id,
                    "mean_reversion_candidate_id": p.mr.candidate_id,
                    "core_weight": ALPHAS[j] if j < 3 else np.nan,
                    "er_unknown_blocked_risk_add_count": int(c_blocked[i]) if j == 3 else 0,
                }
            )
        medians.append(float(np.median(book.orders[offset : offset + n]) / years))
    medians.extend(
        [
            float(np.median(book.orders[:nt]) / years),
            float(np.median(book.orders[nt : nt + n]) / years),
        ]
    )
    metrics = []
    for j, (pid, row) in enumerate(zip(PORTFOLIO_IDS, principal_stats, strict=True)):
        active = weights[j] > 0
        trips, hours = book.trips[active].sum(), book.holds[active].sum()
        core_weight = (
            ALPHAS[j]
            if j < 3
            else ALPHAS[j - 3]
            if j < 6
            else 1
            if j == 7
            else 0
            if j == 8
            else np.nan
        )
        end_core = (
            float((weights[j, :nt] @ equity[-1, :nt]) / principal_equity[-1, j])
            if j < 3
            else 1.0
            if j == 7
            else 0.0
            if j == 8
            else np.nan
        )
        diag_total = {
            key: sum(d[key] for d in shadows.pair_diagnostics)
            for key in (
                "blocked_oversold_checks",
                "forced_tactical_exit_count",
                "entry_cancellation_count",
                "cancelled_blackout_order_count",
                "cancelled_terminal_order_count",
            )
        }
        if j in {0, 1, 2, 3, 4, 5, 6, 7}:
            for key in ("cancelled_blackout_order_count", "cancelled_terminal_order_count"):
                tf_count = sum(d[key] for d in shadows.tf_diagnostics)
                diag_total[key] = tf_count if j == 7 else diag_total[key] + tf_count
        if j == 7:
            diag_total = {
                key: value if key.startswith("cancelled_") else 0
                for key, value in diag_total.items()
            }
        metrics.append(
            {
                **row,
                "portfolio_id": pid,
                "architecture": "A"
                if j < 3
                else "B"
                if j < 6
                else "C"
                if j == 6
                else "pure_TF"
                if j == 7
                else "pure_gated_MR",
                "core_weight": core_weight,
                "initial_trend_weight": core_weight if j < 3 or j in {7, 8} else np.nan,
                "ending_trend_weight": end_core,
                "initial_mean_reversion_weight": 1 - core_weight
                if j < 3 or j in {7, 8}
                else np.nan,
                "ending_mean_reversion_weight": 1 - end_core if np.isfinite(end_core) else np.nan,
                "blend_coefficient": core_weight if 3 <= j < 6 else np.nan,
                "ending_tf_signal_contribution": ALPHAS[j - 3]
                * float(shadows.tf_open_state[-1].mean())
                if 3 <= j < 6
                else np.nan,
                "ending_mr_signal_contribution": (1 - ALPHAS[j - 3])
                * float(shadows.mr_open_state[-1].mean())
                if 3 <= j < 6
                else np.nan,
                "mean_tf_signal_contribution": float(signal_component_seconds[j - 3, 0] / seconds)
                if 3 <= j < 6
                else np.nan,
                "mean_mr_signal_contribution": float(signal_component_seconds[j - 3, 1] / seconds)
                if 3 <= j < 6
                else np.nan,
                "signal_contribution_definition": "confirmed_target_components_not_capital; ending_at_last_safe_open_before_terminal_liquidation"
                if 3 <= j < 6
                else "not_applicable",
                "ending_regime_mode": ("TF" if shadows.mode_event_state[-1] == TF_MODE else "MR")
                if j == 6
                else "not_applicable",
                "tactical_average_holding_hours": float(
                    book.holds[nt : nt + n].sum() / book.trips[nt : nt + n].sum()
                )
                if j != 7 and book.trips[nt : nt + n].sum()
                else 0.0,
                "tactical_holding_duration_definition": "funded_GMR_sleeve_episodes"
                if j < 3 or j == 8
                else "warm_GMR_target_component_episode_diagnostic_not_separate_funded_capital"
                if j != 7
                else "not_applicable",
                "style_weight_definition": "funded_self_financing_sleeve_capital"
                if j < 3 or j in {7, 8}
                else "signal_target_attribution_not_funded_sleeve_capital",
                "exposure_fraction": float(principal_exposure_time[j]),
                "exposure_definition": "calendar_time_weighted_event_left_QQQ_notional_over_total_wealth",
                "gross_sleeve_order_count": int(principal_orders[j]),
                "gross_sleeve_orders_per_year": float(principal_orders[j] / years),
                "distinct_execution_timestamp_count": len(principal_timestamps[j]),
                "distinct_execution_timestamps_per_year": len(principal_timestamps[j]) / years,
                "total_turnover": float(principal_turnover[j]),
                "annualized_turnover": float(principal_turnover[j] / years),
                "total_commission": float(principal_fees[j]),
                "total_slippage": float(principal_slips[j]),
                "gross_terminal_equity": float(gross_portfolio[j]),
                "cost_drag_return": float(
                    (gross_portfolio[j] - principal_equity[-1, j]) / initial_cash
                ),
                "positive_weight_sleeve_count": int(active.sum()),
                "gross_sleeve_completed_flat_to_flat_episodes": int(trips),
                "average_gross_sleeve_holding_hours": float(hours / trips) if trips else 0,
                "portfolio_flat_to_flat_episode_count": int(portfolio_episodes[j]),
                "portfolio_average_flat_to_flat_holding_hours": float(
                    portfolio_holds[j] / portfolio_episodes[j]
                )
                if portfolio_episodes[j]
                else 0,
                "median_pair_orders_per_year": medians[j],
                "scored_sessions": len(shadows.daily_times),
                "elapsed_calendar_years": years,
                "start_session": calendar.session.iloc[0],
                "end_session": calendar.session.iloc[-1],
                "commission_bps": float(commission_bps),
                "slippage_bps": float(slippage_bps),
                "regime_tf_occupancy_fraction": float(mode_seconds[TF_MODE] / seconds)
                if j == 6
                else np.nan,
                "regime_mr_occupancy_fraction": float(mode_seconds[MR_MODE] / seconds)
                if j == 6
                else np.nan,
                "er_unknown_time_fraction": float(unknown_seconds / seconds) if j == 6 else np.nan,
                "regime_switch_count": shadows.pair_diagnostics[0]["regime_switch_count"]
                if j == 6
                else 0,
                "regime_switch_cancellation_count": shadows.pair_diagnostics[0][
                    "regime_switch_cancellation_count"
                ]
                if j == 6
                else 0,
                "er_unknown_blocked_risk_add_count": int(c_blocked.sum()) if j == 6 else 0,
                "actual_post_gap_order_count": int(post_gap_orders[j]),
                **diag_total,
                "cancellation_count_definition": "funded_sleeve_pending_intents"
                if j < 3 or j in {7, 8}
                else "shared_TF_and_warm_GMR_model_intents_not_cancelled_broker_orders",
            }
        )
    annual = []
    for j, pid in enumerate(PORTFOLIO_IDS):
        frame = portfolio.annual_return_table(
            pd.Series(principal_equity[:, j], shadows.daily_times)
        )
        frame["portfolio_id"] = pid
        annual.append(frame)
    diagnostic_frames = [
        frame for frame in (shadows.diagnostics, pd.DataFrame(diagnostics)) if not frame.empty
    ]
    full_diagnostics = (
        pd.concat(diagnostic_frames, ignore_index=True)
        if diagnostic_frames
        else pd.DataFrame(columns=["timestamp", "event"])
    )
    return RunResult(
        PORTFOLIO_IDS,
        tuple(metrics),
        shadows.daily_times,
        principal_equity,
        principal_returns,
        cash_returns,
        principal_exposure,
        clock,
        event_equity,
        event_exposure,
        ids,
        groups,
        equity,
        account_returns,
        account_exposure,
        tuple(account_metrics),
        gross_eq[-1].copy(),
        gross_eq,
        pd.DataFrame(book.trade_rows, columns=TargetBook.TRADE_COLUMNS),
        pd.DataFrame(book.command_rows, columns=TargetBook.COMMAND_COLUMNS),
        pd.DataFrame(pair_rows),
        pd.concat(annual, ignore_index=True),
        full_diagnostics,
    )
