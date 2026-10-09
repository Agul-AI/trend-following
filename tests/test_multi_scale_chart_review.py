"""Synthetic display-only chart repair checks: no performance or price packets."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from scripts import plot_multi_scale_report as chart

LOCKED = ("multi_scale_sma100_entry_only", "multi_scale_sma200_recovery_or_trend_break")


def packet(directory, stage, *, clock_offset=0, duplicate=False, extra=False):
    directory.mkdir(parents=True, exist_ok=True)
    times = pd.date_range(f"{chart.STAGES[stage][0]}-01-02 22:00", periods=3, freq="D", tz="UTC")
    file_hashes = {}
    for name, ids in (
        ("basket_s3_daily.npz", LOCKED),
        ("controls_s3_daily.npz", chart.BASE_CONTROLS),
    ):
        current_ids = ids if not duplicate else tuple(ids[0] for _ in ids)
        clock = times if name.startswith("basket") else times + pd.Timedelta(minutes=clock_offset)
        data = dict(
            portfolio_id=np.asarray(current_ids),
            timestamp=np.asarray(clock.astype(str), dtype=str),
            equity=np.column_stack(
                [np.array([1000.0, 1010.0 + j, 1020.0 + j]) for j in range(len(ids))]
            ),
            returns=np.zeros((3, len(ids))),
            cash_returns=np.zeros(3),
            exposure=np.column_stack([np.array([0.0, 0.2 + j / 10, 0.0]) for j in range(len(ids))]),
        )
        if extra:
            data["raw_price"] = np.ones(3)
        path = directory / name
        np.savez(path, **data)
        file_hashes[name] = chart.file_hash(path)
    payload = dict(stage=stage, files=file_hashes)
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (directory / "seal.json").write_text(json.dumps(dict(payload=payload, payload_sha256=digest)))
    return times


def test_full_layout_repair_preserves_exact_arrays_and_exports_safe_provenance(tmp_path):
    sidecar, output = tmp_path / "sidecar", tmp_path / "report"
    output.mkdir()
    summary = dict(locked_basket_ids=LOCKED, freeze_sha256="frozen-synthetic-source")
    (output / "summary.json").write_text(json.dumps(summary))
    before = {}
    for stage in chart.STAGES:
        packet(sidecar / stage, stage)
        before[stage] = {
            name: chart.file_hash(sidecar / stage / name)
            for name in ("basket_s3_daily.npz", "controls_s3_daily.npz")
        }
    review = chart.render_review(sidecar, output)
    assert review["presentation_only"] and review["source_arrays_and_plotted_coordinates_unchanged"]
    assert not review["oos_2019_onward_accessed"] and not review["frozen_sources_edited"]
    for stage in chart.STAGES:
        stage_review = review["stages"][stage]
        assert stage_review["plotted_arrays_exactly_equal"]
        assert stage_review["non_overlapping_header_geometry_verified"]
        assert stage_review["title_and_axis_labels_within_export_bounds"]
        assert stage_review["endpoint_tick_alignment"] == {"first": "left", "last": "right"}
        assert stage_review["wealth_axis_label"] == "Portfolio wealth (USD)"
        assert stage_review["wealth_axis_math_parsing"] is False
        assert stage_review["sealed_derived_file_sha256"] == before[stage]
        times, paths, _ = chart.load_paths(sidecar / stage, stage)
        for cid in (*LOCKED, *chart.BASE_CONTROLS):
            for field in ("equity", "exposure"):
                assert stage_review["source_and_plotted_array_sha256"][cid][
                    field
                ] == chart._array_hash(times, paths[cid][field])
        png = output / f"{stage}_wealth_exposure.png"
        assert png.read_bytes().startswith(b"\x89PNG")
        svg = (output / f"{stage}_wealth_exposure.svg").read_text()
        assert all(line == line.rstrip() for line in svg.splitlines())
        assert "Locked SMA100: recovery-only" in svg and "Portfolio wealth (USD)" in svg
    provenance = (output / "figure_layout_audit.json").read_text()
    assert str(tmp_path) not in provenance
    assert "raw_price" not in provenance


@pytest.mark.parametrize("mutation", ["hash", "clock", "duplicate", "extra", "manifest"])
def test_derived_reader_rejects_changed_sources_or_incompatible_clocks(tmp_path, mutation):
    directory = tmp_path / "development"
    packet(
        directory,
        "development",
        clock_offset=30 if mutation == "clock" else 0,
        duplicate=mutation == "duplicate",
        extra=mutation == "extra",
    )
    if mutation == "hash":
        path = directory / "basket_s3_daily.npz"
        path.write_bytes(path.read_bytes() + b"tampered")
    elif mutation == "manifest":
        manifest = json.loads((directory / "seal.json").read_text())
        manifest["payload"]["stage"] = "validation"
        (directory / "seal.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        chart.load_paths(directory, "development")


def test_mismatched_but_valid_seventeen_ny_session_clocks_are_rejected(tmp_path):
    directory = tmp_path / "development"
    packet(directory, "development", clock_offset=24 * 60)
    with pytest.raises(ValueError, match="clocks"):
        chart.load_paths(directory, "development")


def test_no_oos_route_or_historical_oos_rows(tmp_path):
    with pytest.raises(ValueError, match="development"):
        chart.load_paths(tmp_path, "historical_oos")
    directory = tmp_path / "development"
    packet(directory, "validation")
    with pytest.raises(ValueError):
        chart._derived(directory / "basket_s3_daily.npz", "development")


def test_renderer_rejects_missing_or_reordered_policy_comparisons(tmp_path):
    directory = tmp_path / "development"
    packet(directory, "development")
    times, paths, _ = chart.load_paths(directory, "development")
    with pytest.raises(ValueError, match="policy order"):
        chart.render_stage(times, paths, tuple(reversed(LOCKED)), "development", tmp_path / "out")
    del paths["buy_and_hold"]
    with pytest.raises(ValueError, match="three predefined"):
        chart.render_stage(times, paths, LOCKED, "development", tmp_path / "out")


def test_oos_numeric_arrays_are_not_read_when_clock_is_outside_authorized_stage(monkeypatch):
    class UnsafePacket:
        files = ("portfolio_id", "timestamp", "equity", "returns", "cash_returns", "exposure")

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def __getitem__(self, name):
            if name == "portfolio_id":
                return np.asarray(LOCKED)
            if name == "timestamp":
                return np.asarray(["2019-01-02 22:00:00+00:00"])
            raise AssertionError("Unauthorized numerical arrays must never be read")

    monkeypatch.setattr(chart.np, "load", lambda *args, **kwargs: UnsafePacket())
    with pytest.raises(ValueError, match="authorized stage"):
        chart._derived("not-opened", "validation")
