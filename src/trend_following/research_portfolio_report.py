"""Reviewed repository reports for the separately sealed QQQ sleeve study.

Only new derived development/validation artifacts are read. Raw execution
records, price packets, authorization, full private freezes and 2019+ history
never reach the public projection. The chart is a static research figure.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trend_following import research_quality_protocol_v5 as seals

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = ROOT / "reports/portfolio_research/20261009"
SAFE_METRICS = {
    "portfolio_id",
    "portfolio",
    "trend_weight",
    "mean_reversion_weight",
    "scenario_slippage_bps",
    "commission_bps",
    "slippage_bps",
    "initial_cash",
    "cash_excess_sharpe",
    "cagr",
    "max_drawdown",
    "annualized_volatility",
    "total_return",
    "terminal_equity",
    "gross_terminal_equity",
    "cash_terminal_equity",
    "cost_drag_return",
    "total_commission",
    "total_slippage",
    "exposure_fraction",
    "long_sleeve_exposure_fraction",
    "initial_weighted_long_duration_fraction",
    "initial_weighted_binary_long_time_fraction",
    "exposure_definition",
    "gross_sleeve_order_count",
    "gross_order_count",
    "order_count",
    "gross_sleeve_orders_per_year",
    "gross_orders_per_year",
    "orders_per_year",
    "distinct_execution_timestamps",
    "distinct_execution_count",
    "distinct_execution_timestamp_count",
    "distinct_execution_timestamps_per_year",
    "execution_timestamps_per_year",
    "total_turnover",
    "annualized_turnover",
    "traded_reference_notional",
    "reference_notional",
    "weighted_reference_notional",
    "years",
    "initial_trend_weight",
    "ending_trend_weight",
    "initial_mean_reversion_weight",
    "ending_mean_reversion_weight",
    "ending_trend_fraction",
    "ending_mean_reversion_fraction",
    "scored_sessions",
    "start_session",
    "end_session",
    "sampling_minutes",
    "confirmation_hours",
    "required_checks",
    "constituent_count",
    "positive_weight_constituent_count",
    "active_constituent_count",
    "positive_weight_sleeve_count",
    "elapsed_calendar_years",
    "BH_cash_excess_sharpe",
    "Sharpe_difference_vs_BH",
    "beats_BH_Sharpe",
}


def _safe(path):
    path = Path(path).expanduser().absolute()
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise seals.ProtocolError("Unsafe report path")
    return path.resolve()


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _public(frame):
    result = frame.loc[:, [c for c in frame if c in SAFE_METRICS]].copy()
    if "cash_excess_sharpe" not in result or "cagr" not in result:
        raise seals.ProtocolError("Required public portfolio metrics are absent")
    _privacy(result.to_csv(index=False))
    return result


def _privacy(text):
    if any(s in text for s in ("/Users/", "/home/", ".cache/", "PRIVATE KEY", "api_key=")):
        raise seals.ProtocolError("Private information in public report")


def _read(path):
    return pd.read_csv(_safe(path), float_precision="round_trip")


def _number(value, percent=False):
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.2%}" if percent else f"{value:.4f}"


def _get(row, *names):
    for name in names:
        if name in row:
            return row[name]
    return np.nan


def development_chart(metrics, benchmark, winner_id, output):
    """Plot all twelve frozen allocations under the four cost scenarios."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    required = {"portfolio_id", "trend_weight", "scenario_slippage_bps", "cash_excess_sharpe"}
    if not required.issubset(metrics):
        raise seals.ProtocolError("Chart inputs omit allocation identities or scores")
    if len(metrics) != 48 or metrics.duplicated(["portfolio_id", "scenario_slippage_bps"]).any():
        raise seals.ProtocolError("Development chart requires twelve allocations at four costs")
    colors = {1: "#b8c3cc", 3: "#295c83", 10: "#b58c41", 25: "#915963"}
    markers = {1: "o", 3: "s", 10: "^", 25: "D"}
    with plt.rc_context({"font.size": 11, "svg.hashsalt": "qqq-sleeve-allocation-study"}):
        fig, ax = plt.subplots(figsize=(10, 5.6), layout="constrained")
        for slip in (1, 3, 10, 25):
            part = metrics.loc[metrics.scenario_slippage_bps.eq(slip)].sort_values("trend_weight")
            if len(part) != 12:
                raise seals.ProtocolError("Missing frozen allocation/cost points")
            ax.plot(
                100 * part.trend_weight,
                part.cash_excess_sharpe,
                color=colors[slip],
                marker=markers[slip],
                markersize=4,
                linewidth=2 if slip == 3 else 1.2,
                label=f"1-bp commission + {slip}-bp slippage"
                + (" (selection)" if slip == 3 else ""),
            )
        baseline = metrics.loc[metrics.scenario_slippage_bps.eq(3)]
        chosen = baseline.loc[baseline.portfolio_id.eq(winner_id)]
        if len(chosen) != 1:
            raise seals.ProtocolError("Chart winner differs from the development lock")
        row = chosen.iloc[0]
        ax.scatter(
            [100 * row.trend_weight],
            [row.cash_excess_sharpe],
            s=110,
            facecolors="none",
            edgecolors="#242a30",
            linewidths=1.6,
            zorder=5,
            label="Locked baseline allocation",
        )
        passive = benchmark.loc[benchmark.scenario_slippage_bps.eq(3)]
        name = "portfolio" if "portfolio" in passive else "portfolio_id"
        bh = passive.loc[passive[name].eq("buy_and_hold")]
        if len(bh) == 1 and np.isfinite(bh.iloc[0].cash_excess_sharpe):
            ax.axhline(
                bh.iloc[0].cash_excess_sharpe,
                color="#50565c",
                linestyle="--",
                linewidth=1,
                label="QQQ buy-and-hold, baseline costs",
            )
        ax.axvline(100 * 12 / 44, color="#a0a5aa", linestyle=":", linewidth=1)
        ax.set(
            title="Development allocation sensitivity — 2001–2011",
            xlabel="Initial capital allocated to trend following (%)",
            ylabel="Daily cash-excess Sharpe (252-session annualization)",
        )
        ax.set_xlim(-2, 102)
        ax.set_xticks(range(0, 101, 10))
        ax.grid(axis="y", color="#e1e4e6", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9, loc="best")
        fig.savefig(
            output / "development_allocation_sharpe.png",
            dpi=160,
            metadata={"Title": "QQQ portfolio development allocation sensitivity"},
        )
        fig.savefig(output / "development_allocation_sharpe.svg", metadata={"Date": None})
        plt.close(fig)


def _table(frame, label_column="portfolio_id", *, show_cost=False):
    cost_header = " Slippage + 1-bp commission |" if show_cost else ""
    cost_rule = "---:|" if show_cost else ""
    lines = [
        f"|{cost_header} Portfolio | Initial trend | Sharpe | CAGR | Daily drawdown loss | Gross sleeve orders/year | Notional turnover/year |",
        f"|{cost_rule}---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        orders = _get(
            row, "gross_sleeve_orders_per_year", "gross_orders_per_year", "orders_per_year"
        )
        cost_cell = f" {row.scenario_slippage_bps:g} + 1 bp |" if show_cost else ""
        lines.append(
            f"|{cost_cell} {row[label_column]} | {_number(_get(row, 'trend_weight'), True)} | "
            f"{_number(row.cash_excess_sharpe)} | {_number(row.cagr, True)} | "
            f"{_number(-row.max_drawdown, True)} | {_number(orders)} | "
            f"{_number(_get(row, 'annualized_turnover'))} |"
        )
    return lines


def publish_report(sidecar, output=DEFAULT_REPORT):
    from trend_following import research_portfolio_protocol as protocol

    sidecar, output = map(_safe, (sidecar, output))
    frozen = protocol.verify_study(sidecar)
    devseal = protocol.verify_results(sidecar, "development")
    valseal = protocol.verify_results(sidecar, "validation")
    locked = seals._read_seal(sidecar / "locked_allocation.json")
    allocation = locked["allocation"]
    winner_id = allocation["portfolio_id"]
    if output.exists() or not output.is_relative_to(ROOT / "reports/portfolio_research"):
        raise seals.ProtocolError("A new separate portfolio report destination is required")
    dev = _read(sidecar / "development/metrics.csv")
    val = _read(sidecar / "validation/metrics.csv")
    devbench = _read(sidecar / "development/benchmark_metrics.csv")
    valbench = _read(sidecar / "validation/benchmark_metrics.csv")
    # Published columns never include private replay quotes or path bindings.
    frames = {
        "development_metrics.csv": _public(dev),
        "validation_metrics.csv": _public(val),
        "development_benchmarks.csv": _public(devbench),
        "validation_benchmarks.csv": _public(valbench),
    }
    baseline_dev = dev.loc[dev.scenario_slippage_bps.eq(3)].sort_values("trend_weight")
    baseline_val = val.loc[val.scenario_slippage_bps.eq(3)]
    winner = baseline_val.loc[baseline_val.portfolio_id.eq(winner_id)]
    if len(winner) != 1 or len(baseline_dev) != 12:
        raise seals.ProtocolError("Locked allocation or complete development grid absent")
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in frames.items():
        frame.to_csv(output / name, index=False)
    for stage in ("development", "validation"):
        for name in (
            "annual_returns.csv",
            "correlations_constituent.csv",
            "correlations_style.csv",
        ):
            path = sidecar / stage / name
            if not path.is_file():
                raise seals.ProtocolError(f"Required derived diagnostic absent: {stage}/{name}")
            text = _safe(path).read_text()
            _privacy(text)
            (output / f"{stage}_{name}").write_text(text)
    bootstrap = _safe(sidecar / "validation/bootstrap.csv").read_text()
    _privacy(bootstrap)
    (output / "validation_bootstrap.csv").write_text(bootstrap)
    members = frozen["membership"]
    member_rows = [
        {
            "style": r["style"],
            "candidate_id": r["candidate_id"],
            "parameters": json.dumps(r["parameters"], sort_keys=True),
        }
        for r in members
    ]
    pd.DataFrame(member_rows).to_csv(output / "constituents.csv", index=False)
    development_chart(dev, devbench, winner_id, output)
    summary = {
        "study_name": "QQQ Strategy Portfolio Study",
        "initial_capital": 1000,
        "constituent_count": 44,
        "trend_count": 12,
        "mean_reversion_count": 32,
        "development_allocation_count": 12,
        "locked_allocation": allocation,
        "allocation_selected_on": "2001–2011 baseline after-cost Sharpe",
        "development_period": {"start": "2001-01-01", "end": "2011-12-31"},
        "validation_period": {"start": "2012-01-01", "end": "2018-12-31"},
        "fixed_initial_capital_sleeves": True,
        "portfolio_rebalancing": False,
        "order_netting": False,
        "idle_cash_redistributed": False,
        "confirmation_hours": 3,
        "required_consecutive_checks": 7,
        "cash_nominal_annual_rate": 0.03,
        "cash_day_count": "ACT/365",
        "commission_bps": 1,
        "selection_slippage_bps": 3,
        "slippage_bps_scenarios": [1, 3, 10, 25],
        "taxes_financing_separate_expenses": False,
        "historical_oos_authorized": False,
        "2019_onward_prices_opened": False,
        "hindsight_conditioned_membership": True,
        "validation_winner_reselection": False,
        "bootstrap_scope": "fixed historical allocation, not corrected for constituent selection bias",
        "development_result_sha256": seals.canonical_hash(devseal),
        "validation_result_sha256": seals.canonical_hash(valseal),
        "freeze_sha256": seals.canonical_hash(frozen),
        "baseline_validation_locked_metrics": _public(winner).iloc[0].to_dict(),
    }
    seals._write_json(output / "summary.json", _clean(summary))
    lines = [
        "# QQQ Strategy Portfolio Study",
        "",
        "**Fixed initial capital sleeves; 12 trend + 32 mean-reversion strategies; no rebalancing or order netting.**",
        "",
        f"Development locked **{allocation['trend_weight']:.0%} trend / {allocation['mean_reversion_weight']:.0%} mean reversion** (`{winner_id}`). "
        "Within each style, initial capital is equal across candidates. The same allocation is used at every validation cost scenario.",
        "",
        "**This experiment is hindsight-conditioned:** the twelve trend constituents were identified using 2012–2018 validation performance. "
        "Neither development fitting nor this retrospective validation restores independent confirmation. **2019 onward remains closed.**",
        "",
        "## Development allocation search — 2001–2011",
        "",
        "Twelve predefined initial allocations: trend 0–100% in 10% steps, plus equal capital across all 44 candidates (27.27% trend). "
        "Selection maximizes daily cash-excess Sharpe after 1-bp commission + 3-bps slippage. Undefined scores are excluded; "
        "ties within 1e-10 favor the allocation closest to 50/50, then stable ID. No growth, drawdown or trade-count filter applies.",
        "",
        *_table(baseline_dev),
        "",
        "![Development allocation-versus-Sharpe](development_allocation_sharpe.png)",
        "",
        "The chart connects only the tested allocations, not a continuous optimization frontier. Its dotted vertical reference marks equal-44; "
        "the ring marks the one baseline-selected allocation. Other cost scenarios do not reselect it.",
        "",
        "## Retrospective validation — 2012–2018",
        "",
        "Only the locked allocation and predefined pure-style, 50/50 and equal-44 controls are evaluated, with duplicate IDs removed. "
        "Buy-and-hold and cash have matching start, dividend, cash-reference and terminal conventions.",
        "",
        *_table(baseline_val),
        "",
        "### Cost-matched QQQ and cash",
        "",
        *_table(valbench.loc[valbench.scenario_slippage_bps.eq(3)], "portfolio"),
        "",
        "### Locked allocation under hypothetical execution stresses",
        "",
        *_table(val.loc[val.portfolio_id.eq(winner_id)], show_cost=True),
        "",
        "See [paired bootstrap intervals](validation_bootstrap.csv) for baseline validation Sharpe differences versus QQQ and 50/50. "
        "Two thousand paired circular moving-block resamples use 5/20/60-session blocks and seed 20261009. "
        "The intervals describe this fixed historical portfolio; they do not adjust for prior candidate selection or constitute untouched evidence.",
        "",
        "## Accounting and metric definitions",
        "",
        "- Each sleeve independently holds QQQ or cash using its unchanged daily indicators, shared three-hour/seven-check confirmations and next-safe-bar execution. "
        "Start every segment at $1,000 total, reset positions/counters, and use prior prices for indicator history only. Fractional shares and proportional costs permit exact sleeve scaling.",
        "- Aggregate weighted **wealth**, then recompute daily returns; do not average constituent Sharpe or fixed-weight daily returns. "
        "Idle cash never moves between sleeves. Initial weights drift with relative sleeve performance. There are no extra portfolio-level trades.",
        "- Charge gross sleeve commission/slippage once; assume no netting savings. Fixed 3% nominal ACT/365 cash accrues over the frozen event clock, including closures; "
        "effective growth is not exactly 3% APY. Splits, dividends/DRIP, blackouts and terminal liquidation match the frozen engines. No taxes, financing, leverage, shorts or separate fund expenses.",
        "- Sharpe uses daily cash-excess returns, 252-session annualization and sample standard deviation. Drawdown uses daily provider marks at assumed 17:00 New York, not intraday lows. "
        "The Alpha-only execution quotes are reconstructed as-traded units, not guaranteed native raw/point-in-time captures.",
        "- Gross sleeve order counts and distinct execution timestamps are separate measures; summed standalone round trips are **not** net portfolio round trips. "
        "Notional turnover uses scaled actual gross order notional divided by common portfolio wealth before the simultaneous fill group. No portfolio netting is assumed.",
        "- Exposure is the elapsed-calendar-time average of actual QQQ notional divided by portfolio wealth, using the event-left state and latest available labelled quote between events. "
        "Carried marks serve valuation only, never fabricated signals or fills. It is not a fixed weighted average of standalone binary long-duration statistics; that separate diagnostic is also retained. Annual returns and terminal style weights are published with the other metrics.",
        "- Constituent/style correlations use daily cash-excess returns. Constant cash-only series retain undefined correlations rather than being removed. "
        "Cost scenarios are full-period research assumptions, not measured historical QQQ execution costs or retrospectively labeled market regimes.",
        "",
        "## Inspectable evidence",
        "",
        "[Development metrics](development_metrics.csv) · [Validation metrics](validation_metrics.csv) · "
        "[Development benchmarks](development_benchmarks.csv) · [Validation benchmarks](validation_benchmarks.csv) · "
        "[Exact constituent membership](constituents.csv) · [Protocol summary](summary.json) · [Artifact hashes](artifact_hashes.json)",
        "",
        "Annual return and correlation tables are published separately for each stage. Raw prices, execution records, private authorization, "
        "full freezes and replay arrays stay private. Existing frozen trend and mean-reversion sources/results, monitors and resumes remain unchanged.",
        "",
    ]
    if allocation["trend_weight"] in (0, 1):
        lines.insert(
            5,
            "**A pure style won development: mixing did not improve the selected development Sharpe objective.**",
        )
    (output / "README.md").write_text("\n".join(lines))
    for path in output.iterdir():
        if path.suffix in (".csv", ".json", ".md", ".svg"):
            _privacy(path.read_text())
    seals._write_json(
        output / "artifact_hashes.json",
        {path.name: seals.sha256_file(path) for path in sorted(output.iterdir())},
    )
    return _clean(summary)
