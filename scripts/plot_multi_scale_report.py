#!/usr/bin/env python3
"""Post-freeze display-only repair for the two multiscale wealth/exposure charts.

This helper is NOT a frozen performance engine. It makes no strategy, selection,
accounting or metric calculations. Only already-sealed derived basket/control
wealth and exposure arrays are read; source engines and result seals stay intact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIDECAR = ROOT / ".cache/multi_scale_research_20261009_checked"
DEFAULT_OUTPUT = ROOT / "reports/multi_scale_research/20261009"
BASE_CONTROLS = ("portfolio_tf_000", "buy_and_hold", "portfolio_tf_050")
STAGES = {"development": (2001, 2011), "validation": (2012, 2018)}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_hash(times, values):
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(times.asi8).tobytes())
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _derived(path, stage):
    """Open only the exact derived shape; never infer/fill a missing clock."""
    if stage not in STAGES:
        raise ValueError("Only development and retrospective validation are allowed")
    with np.load(path, allow_pickle=False) as source:
        expected = {"portfolio_id", "timestamp", "equity", "returns", "cash_returns", "exposure"}
        if set(source.files) != expected:
            raise ValueError("Only the exact sealed derived basket/control format is allowed")
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
            raise ValueError("Derived identities and clock must be unique, complete and sorted")
        ny = times.tz_convert("America/New_York")
        lower, upper = STAGES[stage]
        if (
            (ny.year < lower).any()
            or (ny.year > upper).any()
            or (ny.hour != 17).any()
            or (ny.minute != 0).any()
            or (ny.second != 0).any()
            or (ny.microsecond != 0).any()
            or (ny.nanosecond != 0).any()
        ):
            raise ValueError("Derived rows must remain within the authorized stage at 17:00 NY")
        equity, exposure = source["equity"].copy(), source["exposure"].copy()
    if (
        equity.shape != (len(times), len(ids))
        or exposure.shape != equity.shape
        or not np.isfinite(equity).all()
        or not np.isfinite(exposure).all()
        or (equity <= 0).any()
        or (exposure < -1e-10).any()
        or (exposure > 1 + 1e-10).any()
    ):
        raise ValueError("Derived wealth/exposure arrays must match the exact clock and identities")
    return times, {
        cid: {"equity": equity[:, i], "exposure": exposure[:, i]} for i, cid in enumerate(ids)
    }


def load_paths(directory, stage):
    if stage not in STAGES:
        raise ValueError("Only development and retrospective validation are allowed")
    files = {
        name: Path(directory) / name for name in ("basket_s3_daily.npz", "controls_s3_daily.npz")
    }
    manifest = json.loads((Path(directory) / "seal.json").read_text())
    payload = manifest["payload"]
    payload_digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if payload_digest != manifest.get("payload_sha256") or payload.get("stage") != stage:
        raise ValueError("Derived stage manifest integrity or identity changed")
    hashes = payload["files"]
    actual = {name: file_hash(path) for name, path in files.items()}
    if any(actual[name] != hashes.get(name) for name in files):
        raise ValueError("Derived source file differs from its sealed hash")
    times, baskets = _derived(files["basket_s3_daily.npz"], stage)
    controls_times, controls = _derived(files["controls_s3_daily.npz"], stage)
    if not times.equals(controls_times) or set(baskets) & set(controls):
        raise ValueError("Basket/control clocks must match exactly and identities cannot overlap")
    return times, baskets | controls, actual


def _label(cid):
    if cid == "portfolio_tf_000":
        return "Ungated mean reversion"
    if cid == "buy_and_hold":
        return "QQQ buy-and-hold"
    if cid == "portfolio_tf_050":
        return "50/50 independent-sleeve control"
    if cid.startswith("multi_scale_sma"):
        period, policy = cid.removeprefix("multi_scale_sma").split("_", 1)
        if period not in {"100", "200"} or policy not in {"entry_only", "recovery_or_trend_break"}:
            raise ValueError("Unknown locked filter identity")
        return f"Locked SMA{period}: " + (
            "recovery-only" if policy == "entry_only" else "recovery or break"
        )
    raise ValueError("Unknown predeclared figure identity")


def render_stage(times, paths, locked_ids, stage, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    ids = (*locked_ids, *BASE_CONTROLS)
    if (
        stage not in STAGES
        or len(locked_ids) != 2
        or len(set(ids)) != 5
        or not set(ids).issubset(paths)
    ):
        raise ValueError("Exactly two locked baskets and three predefined controls are required")
    if not locked_ids[0].endswith("_entry_only") or not locked_ids[1].endswith(
        "_recovery_or_trend_break"
    ):
        raise ValueError("Locked identities must retain their separate exit-policy order")
    labels = [_label(cid) for cid in ids]
    colors = ("#295c83", "#b58c41", "#915963", "#51585e", "#829270")
    styles = ("-", "--", "-.", ":", "-")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_digest = {}
    with plt.rc_context(
        {
            "font.size": 11,
            "svg.hashsalt": "qqq-multiscale-layout-review",
            "axes.labelcolor": "#242a30",
            "text.color": "#242a30",
        }
    ):
        fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.5), sharex=True)
        # Explicit fixed zones prevent a floating axis legend from touching the
        # title or plotted evidence. The same zones apply to both stage figures.
        fig.subplots_adjust(left=0.105, right=0.945, top=0.785, bottom=0.085, hspace=0.13)
        lower, upper = STAGES[stage]
        title = fig.suptitle(
            f"{stage.title()} combined-construction comparison — {lower}–{upper}",
            y=0.975,
            fontsize=14,
            fontweight="semibold",
        )
        subtitle = fig.text(
            0.5,
            0.941,
            "Baseline: 1-bp commission + 3-bp slippage · $1,000 initial capital · 3% nominal cash",
            ha="center",
            va="top",
            fontsize=10.5,
        )
        for j, cid in enumerate(ids):
            wealth_line = axes[0].plot(
                times,
                paths[cid]["equity"],
                color=colors[j],
                linestyle=styles[j],
                label=labels[j],
                linewidth=1.6,
            )[0]
            exposure_line = axes[1].plot(
                times, paths[cid]["exposure"], color=colors[j], linestyle=styles[j], linewidth=1
            )[0]
            source_digest[cid] = {}
            for field, line in (("equity", wealth_line), ("exposure", exposure_line)):
                actual = line.get_ydata(orig=True)
                if not np.array_equal(actual, paths[cid][field]) or not np.array_equal(
                    line.get_xdata(orig=True), times.to_numpy()
                ):
                    raise ValueError("Plotted coordinates changed the sealed source arrays")
                source_digest[cid][field] = _array_hash(times, paths[cid][field])
        legend = fig.legend(
            *axes[0].get_legend_handles_labels(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.915),
            ncol=2,
            fontsize=10,
            frameon=False,
            columnspacing=2.5,
            handlelength=3.0,
            labelspacing=0.55,
        )
        axes[0].set_ylabel("Portfolio wealth (USD)", parse_math=False)
        axes[1].set(ylabel="Session-mark QQQ notional / wealth", ylim=(-0.02, 1.02))
        axes[1].yaxis.set_major_formatter(PercentFormatter(1))
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
        tick_labels = axes[1].get_xticklabels()
        tick_labels[0].set_horizontalalignment("left")
        tick_labels[-1].set_horizontalalignment("right")
        for ax in axes:
            ax.grid(axis="y", color="#e1e4e6", linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        title_box, subtitle_box, legend_box, plot_box = (
            artist.get_window_extent(renderer) for artist in (title, subtitle, legend, axes[0])
        )
        if (
            title_box.overlaps(subtitle_box)
            or legend_box.overlaps(title_box)
            or legend_box.overlaps(subtitle_box)
            or legend_box.overlaps(plot_box)
        ):
            raise ValueError("Reserved title/subtitle/legend/plot zones still overlap")
        figure_box = fig.get_window_extent(renderer)
        visible_text = [
            title,
            subtitle,
            *axes[1].get_xticklabels(),
            *axes[0].get_yticklabels(),
            *axes[1].get_yticklabels(),
            axes[0].yaxis.label,
            axes[1].yaxis.label,
        ]
        if any(
            artist.get_window_extent(renderer).x0 < figure_box.x0
            or artist.get_window_extent(renderer).x1 > figure_box.x1
            or artist.get_window_extent(renderer).y0 < figure_box.y0
            or artist.get_window_extent(renderer).y1 > figure_box.y1
            for artist in visible_text
        ):
            raise ValueError("A chart title or axis label escapes the exported figure")
        stem = f"{stage}_wealth_exposure"
        fig.savefig(output / f"{stem}.png", dpi=160, metadata={"Title": stem})
        fig.savefig(output / f"{stem}.svg", metadata={"Date": None})
        plt.close(fig)
    svg = output / f"{stem}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    return dict(
        stage=stage,
        selected_series=list(ids),
        labels=labels,
        observed_marks=len(times),
        first_mark=times[0].isoformat(),
        last_mark=times[-1].isoformat(),
        plotted_arrays_exactly_equal=True,
        source_and_plotted_array_sha256=source_digest,
        non_overlapping_header_geometry_verified=True,
        title_and_axis_labels_within_export_bounds=True,
        endpoint_tick_alignment={
            "first": tick_labels[0].get_horizontalalignment(),
            "last": tick_labels[-1].get_horizontalalignment(),
        },
        wealth_axis_label=axes[0].get_ylabel(),
        wealth_axis_math_parsing=axes[0].yaxis.label.get_parse_math(),
        svg_trailing_whitespace_normalized=True,
    )


def render_review(sidecar, output, summary_path=None):
    output = Path(output)
    summary_path = output / "summary.json" if summary_path is None else Path(summary_path)
    summary = json.loads(summary_path.read_text())
    locked_ids = tuple(summary["locked_basket_ids"])
    review = dict(
        schema="qqq_multiscale_post_freeze_figure_layout_review",
        presentation_only=True,
        no_new_performance_or_selection=True,
        frozen_sources_edited=False,
        oos_2019_onward_accessed=False,
        renderer="scripts/plot_multi_scale_report.py",
        source_freeze_sha256=summary["freeze_sha256"],
        source_arrays_and_plotted_coordinates_unchanged=True,
        stages={},
    )
    for stage in STAGES:
        directory = Path(sidecar) / stage
        times, paths, hashes = load_paths(directory, stage)
        before = {name: file_hash(directory / name) for name in hashes}
        stage_review = render_stage(times, paths, locked_ids, stage, output)
        after = {name: file_hash(directory / name) for name in hashes}
        if before != after:
            raise ValueError("Rendering altered sealed derived inputs")
        stage_review["sealed_derived_file_sha256"] = hashes
        stage_review["sealed_derived_files_unchanged"] = True
        review["stages"][stage] = stage_review
    provenance = output / "figure_layout_audit.json"
    provenance.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
    return review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    review = render_review(args.sidecar, args.output, args.summary)
    print(json.dumps({"presentation_only": True, "figures": 2, "stages": list(review["stages"])}))


if __name__ == "__main__":
    main()
