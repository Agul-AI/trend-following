"""Safe derived publication for the separately sealed multiscale experiment.

This report consumes only verified development/validation result artifacts. It
never loads price packets, raw fill records, OOS history or full private freezes.
The requested repository-native figures are reproducible static PNG/SVG exports.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trend_following import research_quality_protocol_v5 as seals
from trend_following.research_portfolio_report import SAFE_METRICS as PORTFOLIO_METRICS
from trend_following.research_portfolio_report import _clean, _number, _privacy, _safe

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = ROOT / "reports/multi_scale_research/20261009"
SLIPPAGES = (1, 3, 10, 25)
SAFE_METRICS = PORTFOLIO_METRICS | {
    "basket_id",
    "candidate_id",
    "base_candidate_id",
    "gate_period",
    "exit_policy",
    "family",
    "period",
    "entry_threshold",
    "exit_threshold",
    "portfolio_type",
    "gate_lookback_days",
    "gate_units",
    "indicator_units",
    "base_period_days",
    "blocked_oversold_checks",
    "trend_break_exit_count",
    "recovery_exit_count",
    "both_exit_count",
    "entry_cancellation_count",
    "exit_cancellation_count",
    "pending_order_cancellations",
    "gate_unknown_checks",
    "gate_off_checks",
    "trade_count",
    "completed_round_trips",
    "trades_per_year",
    "average_holding_hours",
    "weighted_average_holding_hours",
    "gross_sleeve_average_holding_hours",
    "terminal_pending_order_cancellations",
    "blocked_entry_checks",
    "cancelled_entry_orders",
    "cancelled_exit_orders",
    "trend_break_exits",
    "recovery_exits",
    "both_cause_exits",
    "policy",
    "gate_policy",
    "gate_indicator_units",
    "gate_update_hour_ny",
    "unknown_gate_reset_count",
    "entry_confirmation_hours",
    "exit_confirmation_hours",
    "entry_required_checks",
    "exit_required_checks",
    "completed_round_trips_per_year",
    "gross_sleeve_completed_round_trips",
    "average_gross_sleeve_holding_hours",
    "holding_duration_definition",
    "control_type",
    "stage",
    "cancelled_blackout_order_count",
    "cancelled_terminal_order_count",
}


def _read(path):
    return pd.read_csv(_safe(path), float_precision="round_trip")


def _public(frame):
    """Allowlist only derived metrics; quotes, account shares and bindings stay private."""
    result = frame.loc[:, [c for c in frame if c in SAFE_METRICS]].copy()
    if not {"cash_excess_sharpe", "cagr", "max_drawdown"}.issubset(result):
        raise seals.ProtocolError("Required public multiscale metrics are absent")
    if not any(c in result for c in ("basket_id", "portfolio_id", "candidate_id", "portfolio")):
        raise seals.ProtocolError("Public metrics require an explicit strategy identity")
    _privacy(result.to_csv(index=False))
    return result


def _identity(frame):
    for name in ("portfolio_id", "basket_id", "portfolio", "candidate_id"):
        if name in frame:
            return name
    raise seals.ProtocolError("Derived rows have no explicit identity")


def _cost_column(frame):
    for name in ("scenario_slippage_bps", "slippage_bps"):
        if name in frame:
            return name
    raise seals.ProtocolError("Derived rows omit execution scenario")


def _metric(row, *names):
    return next((row[n] for n in names if n in row), np.nan)


def _table(frame, *, costs=False):
    name = _identity(frame)
    cost_name = _cost_column(frame)
    lines = [
        "| Portfolio |"
        + (" Slippage + 1-bp commission |" if costs else "")
        + " Sharpe | CAGR | Daily drawdown loss | Exposure | Gross sleeve orders/year | Notional turnover/year |",
        "|---|" + ("---:|" if costs else "") + "---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        label = str(row[name])
        if "|" in label or "\n" in label:
            raise seals.ProtocolError("Unsafe table identity")
        cost = f" {row[cost_name]:g} + 1 bp |" if costs else ""
        lines.append(
            f"| {label} |{cost} {_number(row.cash_excess_sharpe)} | "
            f"{_number(row.cagr, True)} | {_number(-row.max_drawdown, True)} | "
            f"{_number(_metric(row, 'exposure_fraction'), True)} | "
            f"{_number(_metric(row, 'gross_sleeve_orders_per_year', 'orders_per_year'))} | "
            f"{_number(_metric(row, 'annualized_turnover'))} |"
        )
    return lines


def _locked_ids(lock):
    """Lock projection accepts the protocol's policy-to-selection mapping only."""
    selected = lock.get("selections", lock.get("locked_selections"))
    if isinstance(selected, dict):
        selected = list(selected.values())
    if not isinstance(selected, list) or len(selected) != 2:
        raise seals.ProtocolError("Exactly two frozen exit-policy selections are required")
    result = [
        x if isinstance(x, str) else x.get("basket_id", x.get("portfolio_id")) for x in selected
    ]
    if len(set(result)) != 2 or any(not isinstance(x, str) for x in result):
        raise seals.ProtocolError("Frozen policy selections have invalid identities")
    return tuple(result)


def _paths(path):
    """Read only derived portfolio wealth; never raw prices or recorded execution arrays."""
    with np.load(_safe(path), allow_pickle=False) as source:
        data = {k: source[k].copy() for k in source.files}
    key = next((k for k in ("basket_id", "portfolio_id") if k in data), None)
    needed = {"timestamp", "equity", "returns", "cash_returns", "exposure"}
    if key is None or not needed.issubset(data):
        raise seals.ProtocolError("Derived portfolio paths omit required fields")
    ids = tuple(map(str, data[key]))
    times = pd.DatetimeIndex(pd.to_datetime(data["timestamp"], utc=True))
    n = len(times)
    if (
        not ids
        or len(set(ids)) != len(ids)
        or times.has_duplicates
        or not times.is_monotonic_increasing
    ):
        raise seals.ProtocolError("Derived portfolio identity/clock is invalid")
    for field in ("equity", "returns", "exposure"):
        if data[field].shape != (n, len(ids)) or not np.isfinite(data[field]).all():
            raise seals.ProtocolError("Derived portfolio arrays do not align")
    if data["cash_returns"].shape != (n,) or not np.isfinite(data["cash_returns"]).all():
        raise seals.ProtocolError("Derived cash reference does not align")
    if (data["equity"] <= 0).any() or (
        (data["exposure"] < -1e-10) | (data["exposure"] > 1 + 1e-10)
    ).any():
        raise seals.ProtocolError("Derived portfolio wealth/exposure is invalid")
    return (
        times,
        {
            cid: {field: data[field][:, j] for field in ("equity", "returns", "exposure")}
            for j, cid in enumerate(ids)
        },
        data["cash_returns"],
    )


def _merged_paths(directory, slip):
    times, paths, cash = _paths(directory / f"basket_s{slip}_daily.npz")
    other_times, controls, other_cash = _paths(directory / f"controls_s{slip}_daily.npz")
    if not times.equals(other_times) or not np.array_equal(cash, other_cash):
        raise seals.ProtocolError(
            "New baskets and reused controls have different clocks/cash references"
        )
    if set(paths) & set(controls):
        raise seals.ProtocolError("Basket/control path identities overlap")
    return times, paths | controls


def selection_chart(metrics, locked_ids, output):
    """Four discrete baskets, rather than an implied continuous filter frontier."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    identity, cost = _identity(metrics), _cost_column(metrics)
    ids = sorted(metrics[identity].unique())
    if len(ids) != 4 or len(metrics) != 16 or metrics.duplicated([identity, cost]).any():
        raise seals.ProtocolError("Selection chart requires four baskets at all four costs")
    if set(metrics[cost]) != set(SLIPPAGES) or not set(locked_ids).issubset(ids):
        raise seals.ProtocolError("Chart costs or locked basket IDs disagree")
    colors = {1: "#899ba7", 3: "#295c83", 10: "#b58c41", 25: "#915963"}
    markers = {1: "o", 3: "s", 10: "^", 25: "D"}
    with plt.rc_context({"font.size": 11, "svg.hashsalt": "qqq-multiscale-study"}):
        fig, ax = plt.subplots(figsize=(11.5, 5.6), layout="constrained")
        for slip in SLIPPAGES:
            rows = metrics.loc[metrics[cost].eq(slip)].set_index(identity).reindex(ids)
            if rows.cash_excess_sharpe.isna().all():
                continue
            ax.scatter(
                np.arange(4) + (SLIPPAGES.index(slip) - 1.5) * 0.12,
                rows.cash_excess_sharpe,
                color=colors[slip],
                marker=markers[slip],
                s=42,
                label=f"1-bp commission + {slip}-bp slippage"
                + (" (selection)" if slip == 3 else ""),
            )
        baseline = metrics.loc[metrics[cost].eq(3)].set_index(identity)
        for cid in locked_ids:
            ax.scatter(
                [ids.index(cid) - 0.06],
                [baseline.loc[cid, "cash_excess_sharpe"]],
                s=135,
                facecolors="none",
                edgecolors="#242a30",
                linewidths=1.5,
            )
        ax.axhline(0, color="#71777c", linewidth=0.8)
        ax.set(
            title="Filtered-basket development sensitivity — 2001–2011",
            ylabel="Daily cash-excess Sharpe (252-session annualization)",
        )
        labels = [cid.replace("multi_scale_", "").replace("_", "\n") for cid in ids]
        ax.set_xticks(range(4), labels)
        ax.grid(axis="y", color="#e1e4e6", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2, fontsize=9)
        _save_figure(fig, output, "development_filter_sharpe")
        plt.close(fig)


def _save_figure(fig, output, stem):
    fig.savefig(output / f"{stem}.png", dpi=160, metadata={"Title": stem})
    fig.savefig(output / f"{stem}.svg", metadata={"Date": None})
    # Matplotlib's SVG indentation can contain trailing spaces; whitespace-only
    # normalization leaves all XML/path tokens and chart data unchanged.
    path = output / f"{stem}.svg"
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def wealth_exposure_chart(times, paths, selected_ids, stage, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    ids = tuple(dict.fromkeys(selected_ids))
    if not ids or not set(ids).issubset(paths):
        raise seals.ProtocolError("Required figure comparison paths are absent")
    colors = ("#295c83", "#b58c41", "#915963", "#51585e", "#829270")
    styles = ("-", "--", "-.", ":", "-")
    with plt.rc_context({"font.size": 11, "svg.hashsalt": "qqq-multiscale-study"}):
        fig, axes = plt.subplots(2, 1, figsize=(11.5, 8), sharex=True, layout="constrained")
        for j, cid in enumerate(ids):
            label = cid.replace("multi_scale_", "").replace("portfolio_", "")
            axes[0].plot(
                times,
                paths[cid]["equity"],
                color=colors[j % 5],
                linestyle=styles[j % 5],
                label=label,
                linewidth=1.5,
            )
            axes[1].plot(
                times,
                paths[cid]["exposure"],
                color=colors[j % 5],
                linestyle=styles[j % 5],
                linewidth=1,
            )
        axes[0].set(
            title=f"{stage.title()} combined-construction comparison — baseline costs",
            ylabel="Terminal-liquidation-matched wealth ($; initial $1,000)",
        )
        axes[1].set(ylabel="Session-mark QQQ notional / wealth", ylim=(-0.02, 1.02))
        axes[1].yaxis.set_major_formatter(PercentFormatter(1))
        # Explicit first/last observed dates and sparse calendar anchors.
        anchors = (
            [times[0]]
            + [
                pd.Timestamp(f"{y}-01-01", tz="UTC")
                for y in range(times[0].year + 2, times[-1].year, 2)
            ]
            + [times[-1]]
        )
        axes[1].set_xticks(anchors)
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axes[1].set_xlim(times[0], times[-1])
        axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=2, fontsize=9)
        for ax in axes:
            ax.grid(axis="y", color="#e1e4e6", linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
        _save_figure(fig, output, f"{stage}_wealth_exposure")
        plt.close(fig)


def publish_report(sidecar, output=DEFAULT_REPORT):
    from trend_following import research_multi_scale_protocol as protocol

    sidecar, output = map(_safe, (sidecar, output))
    frozen = protocol.verify_study(sidecar)
    devseal = protocol.verify_results(sidecar, "development")
    valseal = protocol.verify_results(sidecar, "validation")
    lock = protocol.verify_locked_selections(sidecar)
    locked = _locked_ids(lock)
    if output.exists() or not output.is_relative_to(ROOT / "reports/multi_scale_research"):
        raise seals.ProtocolError("A new separate multiscale report destination is required")
    frames = {}
    chart_data = {}
    for stage in ("development", "validation"):
        directory = sidecar / stage
        baskets = _public(_read(directory / "metrics.csv"))
        controls = _public(_read(directory / "controls_metrics.csv"))
        if len(baskets) != 16:
            raise seals.ProtocolError(
                "Every stage must publish all four predefined filtered baskets"
            )
        frames[f"{stage}_metrics.csv"] = baskets
        frames[f"{stage}_controls.csv"] = controls
        rows = []
        for slip in SLIPPAGES:
            frame = (
                _public(_read(directory / f"constituent_metrics_s{slip}_daily.csv"))
                if (directory / f"constituent_metrics_s{slip}_daily.csv").is_file()
                else _public(_read(directory / f"constituent_metrics_s{slip}.csv"))
            )
            if len(frame) != 128:
                raise seals.ProtocolError("Each cost scenario must retain all 128 modified rules")
            frame["scenario_slippage_bps"] = slip
            rows.append(frame)
        frames[f"{stage}_constituents.csv"] = pd.concat(rows, ignore_index=True)
        annual = _read(directory / "annual_returns.csv")
        annual = annual.loc[
            :,
            [
                c
                for c in annual
                if c in {"year", "annual_return", "portfolio_id", "scenario_slippage_bps", "stage"}
            ],
        ]
        required = {"year", "annual_return"}
        if not required.issubset(annual):
            raise seals.ProtocolError("Annual returns are absent")
        _privacy(annual.to_csv(index=False))
        frames[f"{stage}_annual_returns.csv"] = annual
        chart_data[stage] = _merged_paths(directory, 3)
    bootstrap = _read(sidecar / "validation/bootstrap.csv")
    bootstrap = bootstrap.loc[
        :,
        [
            c
            for c in bootstrap
            if c
            in {
                "basket_id",
                "reference_id",
                "reference_aliases",
                "block_length",
                "replicates",
                "finite_replicates",
                "seed",
                "observed_sharpe_difference",
                "ci_lower",
                "ci_upper",
                "confidence_level",
            }
        ],
    ]
    _privacy(bootstrap.to_csv(index=False))
    frames["validation_bootstrap.csv"] = bootstrap
    # Fail safely before writing a partial public artifact when charts are absent.
    for stage, (_times, paths) in chart_data.items():
        if not set(locked).issubset(paths) or "buy_and_hold" not in paths:
            raise seals.ProtocolError(f"{stage} comparison paths omit locked baskets or QQQ")
    members = frozen["membership"]
    if len(members) != 44:
        raise seals.ProtocolError("Report requires all 44 unchanged constituent definitions")
    mr50 = [
        r
        for r in members
        if r["style"] == "mean_reversion" and r["parameters"]["rule"]["period"] == 50
    ]
    if len(mr50) != 3:
        raise seals.ProtocolError("Frozen universe must retain the three known 50-day MR members")
    horizon_lines = [
        "| MR family | Daily lookback | Entry threshold | Recovery threshold |",
        "|---|---:|---:|---:|",
    ]
    for row in mr50:
        params = row["parameters"]
        horizon_lines.append(
            f"| {params['rule']['family']} | 50 | {params['entry_threshold']:g} | {params['exit_threshold']:g} |"
        )
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
    selection_chart(frames["development_metrics.csv"], locked, output)
    for stage, (times, paths) in chart_data.items():
        control = (
            "portfolio_tf_000"
            if "portfolio_tf_000" in paths
            else next(cid for cid in paths if "ungated" in cid)
        )
        figure_ids = [*locked, control, "buy_and_hold"]
        if "portfolio_tf_050" in paths:
            figure_ids.append("portfolio_tf_050")
        wealth_exposure_chart(times, paths, figure_ids, stage, output)
    summary = {
        "study_name": "Trend Following Study — Long-Term Trend + Short-Term Mean-Reversion Comparison",
        "member_count": 44,
        "trend_count": 12,
        "mean_reversion_count": 32,
        "modified_rule_count": 128,
        "filtered_basket_count": 4,
        "locked_basket_ids": locked,
        "development_period": "2001–2011",
        "retrospective_validation_period": "2012–2018",
        "selection_uses_development_only": True,
        "hindsight_conditioned_membership": True,
        "validation_winner_reselection": False,
        "historical_oos_authorized": False,
        "2019_onward_prices_opened": False,
        "gate_periods_daily_sessions": [100, 200],
        "gate_availability_assumption": "17:00 America/New_York",
        "confirmation_hours": 3,
        "required_checks": 7,
        "initial_capital": 1000,
        "cash_nominal_annual_rate": 0.03,
        "cash_day_count": "ACT/365",
        "commission_bps": 1,
        "slippage_scenarios_bps": list(SLIPPAGES),
        "selection_slippage_bps": 3,
        "initial_equal_weights_per_filtered_basket": 32,
        "portfolio_rebalancing": False,
        "order_netting": False,
        "idle_cash_redistributed": False,
        "horizon_caveat": "Existing universe is not a strict long/short horizon separation; retained daily parameters are unchanged.",
        "bootstrap_scope": "Fixed historical rules; paired intervals do not correct candidate selection bias.",
        "freeze_sha256": seals.canonical_hash(frozen),
        "development_result_sha256": seals.canonical_hash(devseal),
        "validation_result_sha256": seals.canonical_hash(valseal),
        "locked_selections_sha256": seals.canonical_hash(lock),
        "source_sha256": frozen.get("source_sha256", frozen.get("source_hashes", {})),
    }
    seals._write_json(output / "summary.json", _clean(summary))
    dev = frames["development_metrics.csv"]
    val = frames["validation_metrics.csv"]
    valcontrols = frames["validation_controls.csv"]
    lines = [
        "# Trend Following Study — Long-Term Trend + Short-Term Mean-Reversion Comparison",
        "",
        "**Two constructions:** independently invested core/tactical sleeves versus trend-filtered pullback purchases. "
        "Filtered baskets use 32 equal initial MR sleeves; a trend gate supplies permission, not an invested trend sleeve.",
        "",
        "**Retrospective, hindsight-conditioned research:** the 12 trend members were chosen using 2012–2018 performance. "
        "Development fitting does not restore independent validation. **2019 onward remains closed.**",
        "",
        "## Development selections — 2001–2011",
        "",
        f"The two frozen exit-policy selections are `{locked[0]}` and `{locked[1]}`. "
        "For each policy, choose SMA100 or SMA200 using baseline daily cash-excess Sharpe. "
        "Undefined scores are excluded; ties within 1e-10 use stable basket ID. No growth, drawdown or trade-frequency filter applies.",
        "",
        *_table(dev.loc[dev[_cost_column(dev)].eq(3)]),
        "",
        "![Development filter sensitivity](development_filter_sharpe.png)",
        "",
        "Rings mark the two baseline selections. Every hypothetical cost scenario evaluates unchanged rules; it does not select a new filter.",
        "",
        "The independently invested sleeve experiment is reused from its existing seals: all 12 development allocations, "
        "and only the deduplicated locked/pure-style/50-50/equal-44 validation controls. Its old selected allocation is never replaced.",
        "",
        "## Retrospective validation — 2012–2018",
        "",
        "All four filtered baskets were predefined. Two are locked policy selections; the others are controls, not replacement winners.",
        "",
        *_table(val.loc[val[_cost_column(val)].eq(3)]),
        "",
        "### Independent sleeves, gate-only controls, QQQ and cash",
        "",
        *_table(valcontrols.loc[valcontrols[_cost_column(valcontrols)].eq(3)]),
        "",
        "![Validation wealth and exposure](validation_wealth_exposure.png)",
        "",
        "The exposure chart uses daily provider marks; summary exposure is the elapsed-calendar-time integral on the full event clock. "
        "Neither exposure series is an average of fixed sleeve binary position flags.",
        "",
        "### Locked baskets under execution stresses",
        "",
        *_table(val.loc[val[_identity(val)].isin(locked)], costs=True),
        "",
        "See [paired bootstrap intervals](validation_bootstrap.csv): each locked filtered basket versus the old locked sleeve portfolio, "
        "ungated MR and QQQ; identical comparison paths are deduplicated with aliases retained. "
        "Two thousand paired circular moving-block resamples use 5/20/60-session blocks, seed 20261009 and 95% percentile intervals. "
        "Intervals describe fixed historical portfolios and **do not correct selection bias**.",
        "",
        "## Exact trading and accounting definitions",
        "",
        "- The SMA100/SMA200 gate uses finalized daily causal signal-index values only. ON means value > SMA including that final observation; "
        "equality is OFF. Missing/insufficient required history is UNKNOWN. Gate publication is assumed at 17:00 New York; "
        "intraday checks never finalize it. Normal closures do not make data stale.",
        "- Entry needs gate ON plus original oversold support for seven consecutive completed half-hour checks (3 trading hours). "
        "Recovery-only exits keep the original MR recovery condition. Trend-break exits use a separate seven-check OFF counter; "
        "alternating recovery and trend-break observations never form one mixed streak. Either completed exit streak can trigger a sell.",
        "- Fill at the next safe observed open. An OFF gate cancels unfilled entry; an ON gate cancels unfilled trend-break-only exit "
        "but preserves a confirmed recovery exit. UNKNOWN/stale gates and missing intervals hold the last filled position, cancel orders "
        "and reset streaks. Gate-driven entry/exit cancellations, blackout cancellations and terminal pending-order cancellations "
        "are reported separately: terminal cleanup is not a risk-management trade. Post-gap flags remain descriptive only. "
        "Gate-only controls use seven ON/OFF checks without an MR trigger.",
        "- Start each scored segment at $1,000 cash with positions/orders/counters reset. Equal capital remains in each original sleeve; "
        "weights drift. Aggregate wealth and recompute returns, not average Sharpes or fixed weighted daily returns. No rebalancing or netting savings.",
        "- Cash accrues 3% nominal ACT/365 on the frozen event clock; this is not exactly 3% APY. Preserve splits, dividends/DRIP "
        "and terminal liquidation. QQQ/cash only; no leverage, shorting, taxes, financing or separate fund expenses.",
        "- Sharpe uses daily cash-excess returns with sqrt(252) and sample SD. CAGR uses actual elapsed calendar years. "
        "Drawdown is daily at the assumed 17:00 provider mark, not intraday drawdown. Alpha-only execution prices are reconstructed "
        "as-traded values, not guaranteed native raw/point-in-time captures; provider marks need not be executable official closes.",
        "- Exposure is actual QQQ notional / portfolio wealth on the event-left state, time weighted over calendar seconds. "
        "Notional turnover uses scaled gross order notional over common wealth before a simultaneous fill group. "
        "Gross sleeve orders, distinct execution timestamps and standalone holding durations are separate diagnostics—not net portfolio round trips.",
        "- Cost drag compares zero-cost shadow wealth following the same recorded decisions to net wealth. Full-period "
        "1/3/10/25-bp slippage plus 1-bp commission are hypothetical research stresses, not reconstructed historical execution regimes.",
        "",
        "## Horizon and evidence caveats",
        "",
        "Membership retains every existing daily parameter. All twelve trend exits use 200-day SMA/EMA; their entries include daily SMA/EMA 10/20/50/100 and MACD sets. The MR universe has 29 shorter-period members and three 50-day members listed below. "
        "This is therefore a comparison of a **new slow trend gate** with unchanged MR timing, not proof of strict horizon diversification. "
        "All strategies trade the same QQQ, so independent sleeves diversify timing rather than securities.",
        "",
        *horizon_lines,
        "",
        "Inspect [all 44 definitions](constituents.csv), [development portfolios](development_metrics.csv), "
        "[validation portfolios](validation_metrics.csv), [development constituent outcomes](development_constituents.csv), "
        "[validation constituent outcomes](validation_constituents.csv), [development controls](development_controls.csv), "
        "[validation controls](validation_controls.csv), [protocol summary](summary.json) and [artifact hashes](artifact_hashes.json). "
        "Annual returns are exported by stage; blocked oversold checks, confirmed trend-break exits, cancellations and holding duration "
        "remain inspectable in metric tables. Private prices, execution records, full freezes and receipts are excluded.",
        "",
        "![Development wealth and exposure](development_wealth_exposure.png)",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines))
    for path in output.iterdir():
        if path.suffix in (".md", ".csv", ".json", ".svg"):
            _privacy(path.read_text())
    seals._write_json(
        output / "artifact_hashes.json",
        {p.name: seals.sha256_file(p) for p in sorted(output.iterdir())},
    )
    return _clean(summary)
