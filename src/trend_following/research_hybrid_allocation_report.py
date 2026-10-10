"""Safe repository-native publication of sealed QQQ hybrid allocation results.

Only derived, physically bounded development/validation outcomes are read. Raw
quotes, fills, private packet bindings, authorizations and full freezes are never
published. Numeric derived arrays cannot be opened before stage-clock checks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trend_following import research_quality_protocol_v5 as seals
from trend_following.research_multi_scale_report import SAFE_METRICS as PREVIOUS_METRICS
from trend_following.research_portfolio_report import _clean, _privacy, _safe

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = ROOT / "reports/hybrid_allocation_research/20261010"
STAGES = {"development": (2001, 2011), "validation": (2012, 2018)}
SLIPPAGES = (1, 3, 10, 25)
REGIME_ID = "hybrid_regime_er20_025_035"
TF_ID = "portfolio_tf_100"
PRINCIPAL_IDS = tuple(
    [f"hybrid_core_{a:03d}" for a in (50, 70, 90)]
    + [f"hybrid_blend_{a:03d}" for a in (50, 70, 90)]
    + [REGIME_ID, TF_ID]
)
SAFE_METRICS = PREVIOUS_METRICS | {
    "architecture",
    "core_weight",
    "blend_coefficient",
    "signal_contribution_definition",
    "tactical_holding_definition",
    "trend_allocation",
    "flat_to_flat_episode_count",
    "tactical_average_holding_hours",
    "gross_sleeve_completed_flat_to_flat_episodes",
    "portfolio_flat_to_flat_episode_count",
    "portfolio_average_flat_to_flat_holding_hours",
    "regime_tf_occupancy_fraction",
    "regime_mr_occupancy_fraction",
    "regime_switch_cancellation_count",
    "er_unknown_time_fraction",
    "er_unknown_blocked_risk_add_count",
    "actual_post_gap_order_count",
    "ending_regime_mode",
    "mean_tf_signal_contribution",
    "mean_mr_signal_contribution",
    "ending_tf_signal_contribution",
    "ending_mr_signal_contribution",
    "alpha",
    "trend_initial_fraction",
    "tactical_initial_fraction",
    "pair_id",
    "tf_candidate_id",
    "mr_candidate_id",
    "trend_candidate_id",
    "mean_reversion_candidate_id",
    "model",
    "rule_id",
    "diagnostic_role",
    "pair_result_kind",
    "pair_path_role",
    "historically_saved_allocation",
    "control_origin",
    "funded_account_count",
    "actual_account_count",
    "actual_order_count",
    "actual_orders_per_year",
    "median_pair_orders_per_year",
    "median_pair_order_count",
    "median_pair_flat_to_flat_holding_hours",
    "median_pair_tactical_holding_hours",
    "average_flat_to_flat_holding_hours",
    "flat_to_flat_round_trips",
    "flat_to_flat_round_trips_per_year",
    "average_tactical_holding_hours",
    "tactical_round_trips",
    "tactical_round_trips_per_year",
    "initial_core_weight",
    "ending_core_weight",
    "initial_tactical_weight",
    "ending_tactical_weight",
    "style_weight_definition",
    "style_weights_are_funded_capital",
    "style_attribution_definition",
    "ending_trend_target_contribution",
    "ending_mr_target_contribution",
    "mean_trend_target_contribution",
    "mean_mr_target_contribution",
    "regime_tf_occupancy",
    "regime_mr_occupancy",
    "regime_unknown_occupancy",
    "regime_tf_time_fraction",
    "regime_mr_time_fraction",
    "regime_unknown_time_fraction",
    "regime_switch_count",
    "regime_switches_per_year",
    "mode_switch_count",
    "blocked_oversold_check_count",
    "forced_tf_exit_count",
    "forced_tactical_exit_count",
    "forced_exit_count",
    "tf_exit_cancelled_mr_buy_count",
    "regime_cancelled_switch_count",
    "cancelled_order_count",
    "cancellation_count_definition",
    "suppressed_risk_add_count",
    "partial_order_count",
    "post_gap_confirmed_signal_count",
    "post_gap_confirmed_buy_count",
    "post_gap_confirmed_sell_count",
    "has_post_gap_confirmed_signal",
    "initial_capital",
    "pair_count",
    "core_account_count",
    "tactical_account_count",
    "actual_sleeve_order_count",
    "actual_sleeve_orders_per_year",
    "distinct_execution_timestamp_count",
    "distinct_execution_timestamps_per_year",
}
ANNUAL_COLUMNS = {"portfolio_id", "scenario_slippage_bps", "stage", "year", "annual_return"}
BOOTSTRAP_COLUMNS = {
    "subject_id",
    "comparison_aliases",
    "architecture",
    "portfolio_id",
    "selected_id",
    "reference_id",
    "reference_aliases",
    "block_length",
    "block_length_sessions",
    "replicates",
    "finite_replicates",
    "seed",
    "observed_sharpe_difference",
    "ci_lower",
    "ci_upper",
    "confidence_level",
    "lower_95_percentile",
    "upper_95_percentile",
}


def _number(value, percent=False):
    if value is None or not np.isfinite(value):
        return "—"
    # A no-loss cash account is not a negative-zero drawdown.
    if value == 0:
        value = 0.0
    return f"{value:.2%}" if percent else f"{value:.4f}"


def _read(path):
    return pd.read_csv(_safe(path), float_precision="round_trip")


def _public(frame):
    result = frame.loc[:, [c for c in frame if c in SAFE_METRICS]].copy()
    if not {"cash_excess_sharpe", "cagr", "max_drawdown"}.issubset(result):
        raise seals.ProtocolError("Required derived hybrid metrics are absent")
    if not any(c in result for c in ("portfolio_id", "pair_id", "candidate_id")):
        raise seals.ProtocolError("Public outcomes require explicit identities")
    _privacy(result.to_csv(index=False))
    return result


def _get(row, *names):
    return next((row[n] for n in names if n in row), np.nan)


def _label(portfolio_id):
    if portfolio_id.startswith("hybrid_core_"):
        weight = int(portfolio_id.rsplit("_", 1)[1])
        return f"Core/tactical {weight}/{100 - weight}"
    if portfolio_id.startswith("hybrid_blend_"):
        weight = int(portfolio_id.rsplit("_", 1)[1])
        return f"Target blend {weight}/{100 - weight}"
    return {
        REGIME_ID: "ER20 regime switching",
        TF_ID: "Pure trend ensemble",
        "hybrid_gated_mr": "Pure TF-gated mean reversion",
        "portfolio_tf_000": "Ungated mean-reversion ensemble",
        "portfolio_tf_050": "Independent sleeves 50/50",
        "portfolio_tf_070": "Independent sleeves 70/30",
        "portfolio_tf_090": "Independent sleeves 90/10",
        "buy_and_hold": "QQQ buy-and-hold",
        "cash": "Cash",
    }.get(portfolio_id, portfolio_id)


def _table(frame, *, costs=False):
    if "portfolio_id" not in frame:
        raise seals.ProtocolError("Tables require portfolio identities")
    lines = [
        "| Portfolio |"
        + (" Slippage + 1-bp commission |" if costs else "")
        + " Cash-excess Sharpe | CAGR | Daily drawdown loss | Exposure | Actual sleeve orders/year | Notional turnover/year |",
        "|---|" + ("---:|" if costs else "") + "---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        label = _label(str(row.portfolio_id))
        if "|" in label or "\n" in label:
            raise seals.ProtocolError("Unsafe public table label")
        cost = f" {_get(row, 'scenario_slippage_bps', 'slippage_bps'):g} + 1 bp |" if costs else ""
        lines.append(
            f"| {label} |{cost} {_number(row.cash_excess_sharpe)} | {_number(row.cagr, True)} | "
            f"{_number(-row.max_drawdown, True)} | {_number(_get(row, 'exposure_fraction'), True)} | "
            f"{_number(_get(row, 'actual_sleeve_orders_per_year', 'actual_orders_per_year', 'gross_sleeve_orders_per_year', 'orders_per_year'))} | "
            f"{_number(_get(row, 'annualized_turnover'))} |"
        )
    return lines


def _locks(lock):
    selections = lock.get("selections")
    if not isinstance(selections, list) or len(selections) != 2:
        raise seals.ProtocolError("Two frozen architecture selections are required")
    if {r.get("architecture") for r in selections} != {"A", "B"}:
        raise seals.ProtocolError("Locks must contain one A and one B selection")
    eligible = {"A": {TF_ID, *PRINCIPAL_IDS[:3]}, "B": {TF_ID, *PRINCIPAL_IDS[3:6]}}
    if any(r.get("portfolio_id") not in eligible[r["architecture"]] for r in selections):
        raise seals.ProtocolError("A lock references a non-predefined portfolio")
    # Both searches may legitimately choose the SAME pure-TF fallback.
    return sorted(selections, key=lambda row: row["architecture"])


def _derived(path, stage):
    """Guard the clock before reading any numeric portfolio arrays."""
    if stage not in STAGES:
        raise seals.ProtocolError("No historical OOS derived-data route")
    with np.load(_safe(path), allow_pickle=False) as source:
        needed = {"portfolio_id", "timestamp", "equity", "returns", "cash_returns", "exposure"}
        if not needed.issubset(source.files):
            raise seals.ProtocolError("Derived portfolio arrays omit required fields")
        ids = tuple(map(str, source["portfolio_id"]))
        times = pd.DatetimeIndex(pd.to_datetime(source["timestamp"], utc=True))
        if (
            not ids
            or len(set(ids)) != len(ids)
            or not len(times)
            or times.hasnans
            or times.has_duplicates
            or not times.is_monotonic_increasing
        ):
            raise seals.ProtocolError("Derived clock/identity is invalid")
        ny = times.tz_convert("America/New_York")
        first, last = STAGES[stage]
        if (
            (ny.year < first).any()
            or (ny.year > last).any()
            or (ny.hour != 17).any()
            or (ny.minute != 0).any()
            or (ny.second != 0).any()
            or (ny.microsecond != 0).any()
            or (ny.nanosecond != 0).any()
        ):
            raise seals.ProtocolError("Derived rows cross the authorized stage or 17:00 NY clock")
        data = {k: source[k].copy() for k in ("equity", "returns", "cash_returns", "exposure")}
    for field in ("equity", "returns", "exposure"):
        if data[field].shape != (len(times), len(ids)) or not np.isfinite(data[field]).all():
            raise seals.ProtocolError("Derived arrays do not match the exact clock")
    if data["cash_returns"].shape != (len(times),) or not np.isfinite(data["cash_returns"]).all():
        raise seals.ProtocolError("Derived cash reference is invalid")
    if (data["equity"] <= 0).any() or (
        (data["exposure"] < -1e-10) | (data["exposure"] > 1 + 1e-10)
    ).any():
        raise seals.ProtocolError("Derived wealth/exposure is invalid")
    return (
        times,
        {
            cid: {name: data[name][:, j] for name in ("equity", "returns", "exposure")}
            for j, cid in enumerate(ids)
        },
        data["cash_returns"],
    )


def _merged_paths(directory, stage, slip=3):
    times, paths, cash = _derived(directory / f"principal_s{slip}_daily.npz", stage)
    other_times, controls, other_cash = _derived(directory / f"controls_s{slip}_daily.npz", stage)
    if not times.equals(other_times) or not np.array_equal(cash, other_cash):
        raise seals.ProtocolError(
            "Hybrid/control clocks or cash references differ; no forward fill"
        )
    for cid in set(paths) & set(controls):
        if any(
            not np.allclose(paths[cid][k], controls[cid][k], atol=1e-10, rtol=1e-10)
            for k in paths[cid]
        ):
            raise seals.ProtocolError("Shared pure-trend/control identity has unequal paths")
    return times, controls | paths


def _correlations(frame):
    allowed = set(PRINCIPAL_IDS) | {
        "hybrid_gated_mr",
        "portfolio_tf_000",
        "portfolio_tf_050",
        "portfolio_tf_070",
        "portfolio_tf_090",
        "buy_and_hold",
        "cash",
    }
    if "portfolio_id" not in frame or not set(frame.columns).issubset(allowed | {"portfolio_id"}):
        raise seals.ProtocolError(
            "Only the declared portfolio cash-excess correlation matrix may be published"
        )
    labels = list(frame.portfolio_id)
    if len(set(labels)) != len(labels) or set(labels) != set(frame.columns) - {"portfolio_id"}:
        raise seals.ProtocolError(
            "Correlation row/column identities must match; undefined cases cannot be dropped"
        )
    result = frame.copy()
    _privacy(result.to_csv(index=False))
    return result


def _pair_membership(frozen):
    """Expose every pair's two constituents; never silently export an ID alone."""
    definitions = frozen.get("pairs")
    members = frozen["membership"]
    trends = [r["candidate_id"] for r in members if r["style"] == "trend"]
    reversions = [r["candidate_id"] for r in members if r["style"] == "mean_reversion"]
    if not isinstance(definitions, list) or len(definitions) != 384:
        raise seals.ProtocolError("All 384 explicit pair membership definitions are required")
    rows = []
    for r in definitions:
        tf = r.get("tf_candidate_id", r.get("trend_candidate_id"))
        mr = r.get("mr_candidate_id", r.get("mean_reversion_candidate_id"))
        if tf not in trends or mr not in reversions or not isinstance(r.get("pair_id"), str):
            raise seals.ProtocolError(
                "Paired membership must identify unchanged trend and MR constituents"
            )
        if ("trend_candidate_id" in r and r["trend_candidate_id"] != tf) or (
            "mean_reversion_candidate_id" in r and r["mean_reversion_candidate_id"] != mr
        ):
            raise seals.ProtocolError("Contradictory pair constituent aliases")
        rows.append(
            {
                "pair_id": r["pair_id"],
                "trend_candidate_id": tf,
                "mean_reversion_candidate_id": mr,
                "trend_index": trends.index(tf),
                "mean_reversion_index": reversions.index(mr),
            }
        )
    result = pd.DataFrame(rows)
    if result.pair_id.duplicated().any() or set(
        zip(result.trend_candidate_id, result.mean_reversion_candidate_id, strict=True)
    ) != {(tf, mr) for tf in trends for mr in reversions}:
        raise seals.ProtocolError(
            "Paired membership must be the complete unchanged Cartesian product"
        )
    _privacy(result.to_csv(index=False))
    return result


def _save(fig, output, stem):
    fig.savefig(output / f"{stem}.png", dpi=160, metadata={"Title": stem})
    fig.savefig(output / f"{stem}.svg", metadata={"Date": None})
    path = output / f"{stem}.svg"
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def selection_chart(metrics, selected_ids, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    needed = {"portfolio_id", "scenario_slippage_bps", "cash_excess_sharpe"}
    if not needed.issubset(metrics) or len(metrics) != 32:
        raise seals.ProtocolError("Selection figure requires eight portfolios at four costs")
    expected = {(pid, slip) for pid in PRINCIPAL_IDS for slip in SLIPPAGES}
    if (
        metrics.duplicated(["portfolio_id", "scenario_slippage_bps"]).any()
        or set(zip(metrics.portfolio_id, metrics.scenario_slippage_bps, strict=True)) != expected
    ):
        raise seals.ProtocolError("Selection figure identities/costs differ from the frozen grid")
    if not set(selected_ids).issubset(PRINCIPAL_IDS):
        raise seals.ProtocolError("Selection figure uses an unsealed ID")
    colors = {1: "#899ba7", 3: "#295c83", 10: "#b58c41", 25: "#915963"}
    markers = {1: "o", 3: "s", 10: "^", 25: "D"}
    with plt.rc_context({"font.size": 11, "svg.hashsalt": "qqq-hybrid-study"}):
        fig, ax = plt.subplots(figsize=(13, 6.8))
        fig.subplots_adjust(left=0.09, right=0.98, top=0.73, bottom=0.20)
        fig.suptitle("Hybrid development allocation sensitivity — 2001–2011", y=0.98, fontsize=16)
        for slip in SLIPPAGES:
            values = (
                metrics.loc[metrics.scenario_slippage_bps.eq(slip)]
                .set_index("portfolio_id")
                .loc[list(PRINCIPAL_IDS)]
            )
            ax.scatter(
                np.arange(8) + (SLIPPAGES.index(slip) - 1.5) * 0.13,
                values.cash_excess_sharpe,
                color=colors[slip],
                marker=markers[slip],
                s=42,
                label=f"1-bp commission + {slip}-bp slippage"
                + (" (selection)" if slip == 3 else ""),
            )
        baseline = metrics.loc[metrics.scenario_slippage_bps.eq(3)].set_index("portfolio_id")
        for cid in set(selected_ids):
            ax.scatter(
                [PRINCIPAL_IDS.index(cid) - 0.065],
                [baseline.loc[cid, "cash_excess_sharpe"]],
                s=140,
                facecolors="none",
                edgecolors="#242a30",
                linewidths=1.5,
            )
        ax.axhline(0, color="#71777c", linewidth=0.8)
        ax.set(ylabel="Daily cash-excess Sharpe (252 sessions)", xlim=(-0.5, 7.5))
        labels = [
            _label(cid)
            .replace("Core/tactical ", "Core\n")
            .replace("Target blend ", "Blend\n")
            .replace("ER20 regime switching", "ER20\nregime")
            .replace("Pure trend ensemble", "Pure\ntrend")
            for cid in PRINCIPAL_IDS
        ]
        ax.set_xticks(range(8), labels)
        ax.grid(axis="y", color="#e1e4e6", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        fig.legend(
            *ax.get_legend_handles_labels(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.91),
            ncol=2,
            fontsize=10,
        )
        fig.canvas.draw()
        _save(fig, output, "development_allocation_sharpe")
        plt.close(fig)


def wealth_exposure_chart(times, paths, selected_ids, stage, output):
    if stage not in STAGES:
        raise seals.ProtocolError("No OOS figure route")
    ids = tuple(dict.fromkeys(selected_ids))
    if not ids or len(ids) > 6 or not set(ids).issubset(paths):
        raise seals.ProtocolError("Figure needs at most six predefined, available comparison paths")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    colors = ("#295c83", "#b58c41", "#915963", "#444c52", "#444c52", "#829270")
    styles = ("-", "--", "-.", "-", ":", "--")
    digests = {}
    with plt.rc_context(
        {"font.size": 11, "svg.hashsalt": "qqq-hybrid-study", "text.parse_math": False}
    ):
        fig, (top, bottom) = plt.subplots(2, 1, figsize=(13, 9.5), sharex=True)
        fig.subplots_adjust(left=0.11, right=0.97, top=0.76, bottom=0.09, hspace=0.18)
        period = "2001–2011" if stage == "development" else "2012–2018"
        title = fig.suptitle(
            f"{stage.title()} hybrid allocation comparison — {period}", y=0.98, fontsize=16
        )
        subtitle = fig.text(
            0.5,
            0.932,
            "Baseline: 1-bp commission + 3-bp slippage · $1,000 initial capital · 3% nominal ACT/365 cash",
            ha="center",
            fontsize=11,
        )
        for j, cid in enumerate(ids):
            top.plot(
                times,
                paths[cid]["equity"],
                color=colors[j],
                linestyle=styles[j],
                label=_label(cid),
                linewidth=1.5,
            )
            bottom.plot(
                times, paths[cid]["exposure"], color=colors[j], linestyle=styles[j], linewidth=1
            )
            digests[cid] = {}
            for key in ("equity", "exposure"):
                digest = hashlib.sha256(
                    np.ascontiguousarray(times.asi8).tobytes()
                    + np.ascontiguousarray(paths[cid][key]).tobytes()
                ).hexdigest()
                digests[cid][key] = digest
        top.set_ylabel("Portfolio wealth (USD)")
        bottom.set_ylabel("Session-mark QQQ notional / wealth")
        bottom.set_ylim(-0.02, 1.02)
        bottom.yaxis.set_major_formatter(PercentFormatter(1))
        anchors = (
            [times[0]]
            + [
                pd.Timestamp(f"{y}-01-01", tz="UTC")
                for y in range(times[0].year + 2, times[-1].year, 2)
            ]
            + [times[-1]]
        )
        bottom.set_xticks(anchors)
        bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        bottom.set_xlim(times[0], times[-1])
        bottom.get_xticklabels()[0].set_ha("left")
        bottom.get_xticklabels()[-1].set_ha("right")
        legend = fig.legend(
            *top.get_legend_handles_labels(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.894),
            ncol=2,
            fontsize=10,
            frameon=False,
        )
        for ax in (top, bottom):
            ax.grid(axis="y", color="#e1e4e6", linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [x.get_window_extent(renderer) for x in (title, subtitle, legend)]
        if any(
            a.overlaps(b) for j, a in enumerate(boxes) for b in boxes[j + 1 :]
        ) or legend.get_window_extent(renderer).overlaps(top.get_window_extent(renderer)):
            raise seals.ProtocolError("Hybrid figure header/legend overlaps the plotted evidence")
        _save(fig, output, f"{stage}_wealth_exposure")
        plt.close(fig)
    return {
        "stage": stage,
        "portfolio_ids": list(ids),
        "observed_marks": len(times),
        "plotted_data_sha256": digests,
        "header_geometry_nonoverlap": True,
    }


def publish_report(sidecar, output=DEFAULT_REPORT):
    from trend_following import research_hybrid_allocation_protocol as protocol

    sidecar, output = map(_safe, (sidecar, output))
    frozen = protocol.verify_study(sidecar)
    stage_seals = {stage: protocol.verify_results(sidecar, stage) for stage in STAGES}
    lock = protocol.verify_locked_selections(sidecar)
    selections = _locks(lock)
    if output.exists() or not output.is_relative_to(ROOT / "reports/hybrid_allocation_research"):
        raise seals.ProtocolError("A fresh separate hybrid report destination is required")
    frames = {}
    paths = {}
    expected_pairs = frozen.get("pair_result_count_per_cost", 2688)
    if expected_pairs != 2688:
        raise seals.ProtocolError(
            "Exactly 384 pairs across three A, three B and one C outcomes are required"
        )
    for stage in STAGES:
        folder = sidecar / stage
        paths[stage] = _merged_paths(folder, stage)
        metrics = _public(_read(folder / "metrics.csv"))
        controls = _public(_read(folder / "controls_metrics.csv"))
        if len(metrics) != 32 or set(metrics.portfolio_id) != set(PRINCIPAL_IDS):
            raise seals.ProtocolError(
                "All eight principal portfolios at four costs must remain published"
            )
        frames[f"{stage}_metrics.csv"] = metrics
        frames[f"{stage}_controls.csv"] = controls
        pair_rows = []
        for slip in SLIPPAGES:
            table = _public(_read(folder / f"pair_metrics_s{slip}.csv"))
            if len(table) != expected_pairs:
                raise seals.ProtocolError("All predefined pair outcomes must remain visible")
            table["scenario_slippage_bps"] = slip
            pair_rows.append(table)
        frames[f"{stage}_pair_metrics.csv"] = pd.concat(pair_rows, ignore_index=True)
        annual = _read(folder / "annual_returns.csv")
        annual = annual.loc[:, [c for c in annual if c in ANNUAL_COLUMNS]]
        if not {"portfolio_id", "year", "annual_return"}.issubset(annual):
            raise seals.ProtocolError("Annual outcome table is incomplete")
        frames[f"{stage}_annual_returns.csv"] = annual
        correlations = _correlations(_read(folder / "correlations.csv"))
        frames[f"{stage}_correlations.csv"] = correlations
    bootstrap = _read(sidecar / "validation/bootstrap.csv")
    frames["validation_bootstrap.csv"] = bootstrap.loc[
        :, [c for c in bootstrap if c in BOOTSTRAP_COLUMNS]
    ]
    members = frozen["membership"]
    if len(members) != 44 or sum(r["style"] == "trend" for r in members) != 12:
        raise seals.ProtocolError("Exact unchanged twelve-trend/thirty-two-MR membership required")
    for frame in frames.values():
        _privacy(frame.to_csv(index=False))
    paired_membership = _pair_membership(frozen)
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in frames.items():
        frame.to_csv(output / name, index=False)
    pd.DataFrame(
        [
            {
                "style": r["style"],
                "candidate_id": r["candidate_id"],
                "parameters": json.dumps(r["parameters"], sort_keys=True),
            }
            for r in members
        ]
    ).to_csv(output / "constituents.csv", index=False)
    paired_membership.to_csv(output / "pairs.csv", index=False)
    selected_ids = tuple(r["portfolio_id"] for r in selections)
    selection_chart(frames["development_metrics.csv"], selected_ids, output)
    figure_reviews = []
    for stage, (clock, values) in paths.items():
        ids = list(
            dict.fromkeys([*selected_ids, REGIME_ID, TF_ID, "buy_and_hold", "hybrid_gated_mr"])
        )
        figure_reviews.append(wealth_exposure_chart(clock, values, ids, stage, output))
    summary = {
        "study_name": "Trend Following Study — Trend Core, Tactical Mean Reversion and Adaptive Allocation",
        "constituent_count": 44,
        "trend_count": 12,
        "mean_reversion_count": 32,
        "pair_count": 384,
        "principal_count": 8,
        "pair_result_count_per_cost": expected_pairs,
        "initial_cash": 1000,
        "development_period": "2001–2011",
        "retrospective_validation_period": "2012–2018",
        "selections": [
            {
                k: r[k]
                for k in (
                    "architecture",
                    "portfolio_id",
                    "alpha",
                    "trend_allocation",
                    "core_weight",
                    "blend_coefficient",
                    "signal_contribution_definition",
                    "tactical_holding_definition",
                )
                if k in r
            }
            for r in selections
        ],
        "regime_rule": {
            "period_days": 20,
            "MR_support_ER_le": 0.25,
            "TF_support_ER_ge": 0.35,
            "confirmation_checks": 7,
            "initial_mode": "TF",
            "daily_availability": "17:00 America/New_York",
        },
        "global_confirmation_hours": 3,
        "consecutive_completed_halfhour_checks": 7,
        "funded_style_wealth_weight_architectures": ["A"],
        "B_C_funded_style_wealth_weights": "N/A",
        "cash_nominal_annual_rate": 0.03,
        "cash_day_count": "ACT/365",
        "commission_bps": 1,
        "slippage_scenarios_bps": list(SLIPPAGES),
        "selection_slippage_bps": 3,
        "hindsight_conditioned_membership": True,
        "validation_reselection": False,
        "historical_oos_authorized": False,
        "2019_onward_numeric_rows_opened": False,
        "taxes_financing_separate_expenses": False,
        "cross_pair_order_netting": False,
        "source_sha256": frozen.get("source_hashes", {}),
        "freeze_sha256": seals.canonical_hash(frozen),
        "locked_selections_sha256": seals.canonical_hash(lock),
        "result_sha256": {s: seals.canonical_hash(v) for s, v in stage_seals.items()},
        "bootstrap_scope": "fixed historical allocations and routing; not corrected for earlier candidate selection bias",
    }
    seals._write_json(output / "summary.json", _clean(summary))
    seals._write_json(output / "figure_audit.json", figure_reviews)
    dev = frames["development_metrics.csv"]
    val = frames["validation_metrics.csv"]
    control = frames["validation_controls.csv"]
    locked_text = "; ".join(
        f"{r['architecture']}: **{_label(r['portfolio_id'])}**" for r in selections
    )
    fallback = [r["architecture"] for r in selections if r["portfolio_id"] == TF_ID]
    mr50 = [
        r
        for r in members
        if r["style"] == "mean_reversion" and r["parameters"]["rule"]["period"] == 50
    ]
    horizon = [
        "| MR family | Daily sessions | Oversold entry | Recovery exit |",
        "|---|---:|---:|---:|",
    ]
    for r in mr50:
        p = r["parameters"]
        family = p["rule"]["family"]
        entry = (
            f"{p['entry_threshold']:.0%}"
            if family.endswith("DISTANCE")
            else f"{p['entry_threshold']:g} SD"
        )
        recovery = (
            f"{p['exit_threshold']:.0%}"
            if family.endswith("DISTANCE")
            else f"{p['exit_threshold']:g} SD"
        )
        horizon.append(f"| {family} | 50 | {entry} | {recovery} |")
    lines = [
        "# Trend Following Study — Trend Core, Tactical Mean Reversion and Adaptive Allocation",
        "",
        "Compare **A: fixed-capital core/tactical sleeves**, **B: actual partial-position target blending**, and **C: confirmed ER20 regime switching**. "
        "All 12 unchanged trend rules × all 32 unchanged mean-reversion rules form **384 equal-initial-capital pairs**; no pair is selected or removed.",
        "",
        "**Retrospective and hindsight-conditioned:** the twelve trend rules were chosen using 2012–2018 validation performance. "
        "New development selection does not restore independent confirmation. **2019 onward remains closed.**",
        "",
        "## Development selections — 2001–2011",
        "",
        f"Frozen baseline selections: {locked_text}. The ER20 regime route is fixed, not performance-selected. "
        "A and B independently maximize daily cash-excess Sharpe after 1-bp commission + 3-bps slippage; undefined scores are excluded, "
        "ties within 1e-10 use stable portfolio ID. No pair, trade-frequency, return, drawdown or beat-BH filter applies.",
        "",
        (
            f"**Pure trend won architecture(s) {', '.join(fallback)}: mixing did not improve the selected development objective.**"
            if fallback
            else "Both selected allocations are retained unchanged at every validation cost."
        ),
        "",
        *_table(dev.loc[dev.scenario_slippage_bps.eq(3)]),
        "",
        "![Development allocation sensitivity](development_allocation_sharpe.png)",
        "",
        "Rings mark the A/B baseline selections, deduplicating a shared pure-trend fallback. Other cost scenarios never reselect weights or rules.",
        "",
        "## Retrospective validation — 2012–2018",
        "",
        "All eight principal portfolios are predefined: three A weights (50/70/90% core), three B weights, fixed C, and shared pure-trend fallback. "
        "Nonselected allocations remain controls—not replacement winners.",
        "",
        *_table(val.loc[val.scenario_slippage_bps.eq(3)]),
        "",
        "### Cost-matched controls",
        "",
        *_table(control.loc[control.scenario_slippage_bps.eq(3)]),
        "",
        "The independent 50/70/90% controls are **ungated fixed-capital sleeves**, not A's TF-state-gated tactical allocation. "
        "Pure trend is shared with the fallback; gated MR is a separate control, not a second invested trend core.",
        "",
        "![Validation wealth and exposure](validation_wealth_exposure.png)",
        "",
        "### Locked A/B and fixed C under hypothetical execution stress",
        "",
        *_table(val.loc[val.portfolio_id.isin(set(selected_ids) | {REGIME_ID})], costs=True),
        "",
        "[Paired bootstrap intervals](validation_bootstrap.csv) use 2,000 paired circular moving-block resamples, "
        "blocks of 5/20/60 sessions, seed 20261009, and 95% percentile intervals. Locked A compares with TF, BH and matching-weight independent sleeves; "
        "locked B compares with TF, BH and matching-weight A; fixed C compares with TF, BH and pure TF-gated MR. "
        "Identical subject-and-reference return-path comparisons are deduplicated while architecture/subject/reference aliases remain visible. These intervals do **not** correct historical member-selection bias.",
        "",
        "## Architecture and causal execution definitions",
        "",
        "- **A:** fund twelve unchanged TF core accounts at a/12 and 384 stateful tactical accounts at (1−a)/384. "
        "Each tactical MR may enter and re-enter only while its paired TF filled state is long. A confirmed TF exit cancels tactical buys and "
        "forces tactical sales at the same next safe open, without another MR wait. A recovery exit never sells the core. Cash and gains stay in their funded sleeve; weights drift.",
        "- **B:** fund 384 pair accounts equally. Target QQQ fraction is a×sTF+(1−a)×sMR,gated: zero with TF cash, a with TF long/MR cash, one with both long. "
        "Trade only when confirmed state changes alter the target—not to maintain weights after price drift. Desired fractions apply to post-cost equity at the raw execution open. "
        "Actual partial fills incur commission/slippage once; simultaneous target changes produce one pair order. This is not a weighted average of independently compounded curves.",
        "- **C:** daily ER20 is |S(t)−S(t−20)| divided by the sum of twenty absolute daily changes, using 21 consecutive finalized causal signal-index values. "
        "The ratio follows the general efficiency-ratio form documented in [StockCharts ChartSchool](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/kaufmans-adaptive-moving-average-kama); its example uses ten periods. This study fixes twenty daily sessions and 0.25/0.35 routing thresholds as research choices, not recommended or validated StockCharts settings. A zero denominator is valid ER=0 by this study's convention; incomplete/nonfinite required observations are UNKNOWN. ER is available at assumed 17:00 New York, never an intraday preview. "
        "Start each segment in TF mode; seven completed half-hour checks with ER≤0.25 confirm MR mode, or ER≥0.35 confirm TF mode. Neutral observations reset the switch streak but retain the mode.",
        "- Both models run warm within TF permission. Routing handoffs adopt the incoming model's already confirmed filled/projected state; no extra fresh-entry wait, "
        "counter reset or fictional sell/rebuy when the target agrees. Warm routing never activates MR history from a closed TF permission window. "
        "An invalidated daily switch trigger cancels a pending mode change without counting a new check. UNKNOWN ER affects only C: retain mode, cancel/reset switches, "
        "suppress risk increases but allow valid risk reductions; reconcile the current confirmed target at the next safe open when ER becomes valid.",
        "- Atomic open ordering is cash accrual/actions/DRIP, snapshot prior-close intents, project TF first, cancel MR buys/force tactical exit if TF exits, "
        "project MR/mode, then place each actual account order once. Signal decisions never use that execution open. New gated MR starts fresh after TF entry; "
        "its first eligible check is the newly filled bar's completed close. Closing half-hours, early closes and normal overnight streak carry are retained.",
        "- Missing/unsafe intervals hold the last filled position, cancel pending orders and reset confirmation streaks. Segment boundaries reset positions/orders/counters; "
        "prior data initialize daily features only. Post-gap flags through the next trading session's close are descriptive and never rank or filter pairs.",
        "- Cancellation counts are pending-intent diagnostics, not claims of real broker order cancellations. A and pure controls count funded sleeve intents: twelve TF cores only once, plus funded tactical intents where applicable. B/C count the shared TF and warm gated-MR model intents, including inactive routing models—not canceled actual broker orders. Executed sleeve orders and regime-switch cancellations are reported separately.",
        "",
        "## Accounting, metrics and interpretation limits",
        "",
        "- Start each segment at $1,000; fractional QQQ/cash only, no leverage/shorts/taxes/financing/separate fund expense. "
        "Cash accrues 3% nominal ACT/365 on the observed event clock—not exactly 3% APY. Preserve split/dividend/DRIP and terminal-close liquidation with cash accrual to 17:00 conventions.",
        "- Aggregate actual funded account wealth, then recompute returns and Sharpe; no cross-account netting, cash redistribution or implicit portfolio-level rebalancing. "
        "A core orders count twelve funded accounts, not 384 repeated pair attributions. Pair outcome/holding diagnostics do not become additional funded orders.",
        "- Daily cash-excess Sharpe uses sqrt(252) and sample standard deviation; CAGR uses actual calendar years. Daily drawdown uses provider marks at assumed 17:00 NY, "
        "not intraday lows. Exposure is actual QQQ notional/wealth on the event-left calendar-time ledger; the charts show session-mark exposure. "
        "Gross actual sleeve orders, distinct execution timestamps, portfolio-notional turnover and flat-to-flat/tactical holding durations are separate measures—not summed net portfolio round trips.",
        "- A's initial/ending style weights refer to funded capital. B/C funded ending style-wealth weights are explicitly **N/A**; their fixed signal coefficients, exposure contributions, mode occupancy and ending mode are routing diagnostics, **not separately funded sleeve weights**. Terminal liquidation can make final QQQ exposure zero without making A's core/tactical account-wealth weights zero. "
        "ER thresholds are fixed research hypotheses, not reconstructed market-regime labels. Undefined cash-only correlations remain visible rather than being dropped.",
        "- Gross shadow accounts follow the same recorded target command times using their **own** zero-cost wealth/share quantities. Cost drag is the gross-versus-net terminal difference, "
        "not terminal fees subtracted once or net quantities reused. All-period 1/3/10/25-bp slippage plus 1-bp commission are hypothetical stress assumptions, not measured QQQ costs.",
        "- Alpha-only execution units are reconstructed as-traded prices; provider marks are not guaranteed native raw/point-in-time executable official closing prices. "
        "A higher Sharpe need not mean higher CAGR. Interpret growth, drawdown, exposure, trading and cost drag together; no fixed historical result establishes live effectiveness.",
        "",
        "## Retained indicator horizons",
        "",
        "Daily indicator periods remain **days**, separate from the three-hour/seven-check confirmation policy. All twelve trend exits use 200-day SMA/EMA; "
        "entries include short 10/20/50/100-day MAs and daily MACD sets. Twenty-nine MR members have shorter periods, while the three 50-day members below remain unchanged. "
        "This is not a strict long/short-horizon universe, and every strategy trades the same QQQ rather than different securities.",
        "",
        *horizon,
        "",
        "## Inspectable derived evidence",
        "",
        "[Principal development outcomes](development_metrics.csv) · [Principal validation outcomes](validation_metrics.csv) · "
        "[Development controls](development_controls.csv) · [Validation controls](validation_controls.csv) · "
        "[Development pair outcomes](development_pair_metrics.csv) · [Validation pair outcomes](validation_pair_metrics.csv) · "
        "[Exact unchanged membership](constituents.csv) · [All 384 paired definitions](pairs.csv) · [Protocol summary](summary.json) · [Figure provenance](figure_audit.json) · [Artifact hashes](artifact_hashes.json)",
        "",
        "Annual returns and cash-excess correlations are published by stage. Metric tables retain pair frequency/holding, style attribution, regime occupancy/switches, "
        "blocked/forced actions and cancellations. Source prices, raw fills, private authorizations/full freezes and runtime logs remain excluded from Git. "
        "Existing studies, monitors and resumes remain unchanged.",
        "",
        "![Development wealth and exposure](development_wealth_exposure.png)",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines))
    for path in output.iterdir():
        if path.suffix in (".csv", ".json", ".md", ".svg"):
            _privacy(path.read_text())
    seals._write_json(
        output / "artifact_hashes.json",
        {p.name: seals.sha256_file(p) for p in sorted(output.iterdir())},
    )
    return _clean(summary)
