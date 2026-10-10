"""Synthetic publication, guard and chart tests; no market-price packets."""

import inspect
import sys
import types

import numpy as np
import pandas as pd
import pytest

import trend_following
from trend_following import research_hybrid_allocation_report as report
from trend_following import research_quality_protocol_v5 as seals


def metrics():
    return pd.DataFrame(
        [
            {
                "portfolio_id": pid,
                "scenario_slippage_bps": slip,
                "cash_excess_sharpe": 0.4 + j * 0.02 - slip * 0.003,
                "cagr": 0.09,
                "max_drawdown": -0.12,
                "exposure_fraction": 0.6,
                "actual_sleeve_orders_per_year": 150,
                "distinct_execution_timestamp_count": 25,
                "annualized_turnover": 5,
                "median_pair_orders_per_year": 3,
                "regime_tf_time_fraction": 0.7,
                "regime_mr_time_fraction": 0.3,
                "initial_core_weight": 0.7,
                "ending_core_weight": np.nan,
                "style_weights_are_funded_capital": False,
                "partial_order_count": 20,
                "reference_price": 123,
                "source_packet": "/Users/private/.cache/prices",
            }
            for j, pid in enumerate(report.PRINCIPAL_IDS)
            for slip in report.SLIPPAGES
        ]
    )


def test_safe_metric_projection_retains_actual_partial_diagnostics_not_quotes():
    result = report._public(metrics())
    assert {
        "partial_order_count",
        "median_pair_orders_per_year",
        "actual_sleeve_orders_per_year",
        "style_weights_are_funded_capital",
    }.issubset(result)
    assert "reference_price" not in result and "source_packet" not in result


@pytest.mark.parametrize("key", ["cash_excess_sharpe", "cagr", "max_drawdown", "portfolio_id"])
def test_required_metrics(key):
    with pytest.raises(seals.ProtocolError):
        report._public(metrics().drop(columns=key))


@pytest.mark.parametrize(
    "secret", ["/Users/secret", "/home/secret", ".cache/private", "PRIVATE KEY", "api_key=x"]
)
def test_public_identifiers_cannot_leak(secret):
    frame = metrics()
    frame.loc[0, "portfolio_id"] = secret
    with pytest.raises(seals.ProtocolError, match="Private"):
        report._public(frame)


def test_two_architectures_may_share_one_pure_trend_fallback():
    value = {
        "selections": [
            {"architecture": "B", "portfolio_id": report.TF_ID, "alpha": 1.0},
            {"architecture": "A", "portfolio_id": report.TF_ID, "alpha": 1.0},
        ]
    }
    assert [x["architecture"] for x in report._locks(value)] == ["A", "B"]
    assert report._locks(value)[0]["portfolio_id"] == report._locks(value)[1]["portfolio_id"]


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"selections": []},
        {"selections": [{"architecture": "A", "portfolio_id": report.TF_ID}] * 2},
        {
            "selections": [
                {"architecture": "A", "portfolio_id": "not_frozen"},
                {"architecture": "B", "portfolio_id": report.TF_ID},
            ]
        },
    ],
)
def test_locks_require_exact_frozen_architectures(bad):
    with pytest.raises(seals.ProtocolError):
        report._locks(bad)


def test_tables_do_not_call_gross_actual_orders_net_round_trips():
    frame = metrics()
    frame.loc[0, "max_drawdown"] = 0.0
    frame.loc[0, "cash_excess_sharpe"] = np.nan
    text = "\n".join(report._table(frame.iloc[:4], costs=True))
    assert "—" in text and "Actual sleeve orders/year" in text
    assert "-0.00%" not in text and "round trips" not in text
    for slip in report.SLIPPAGES:
        assert f" {slip} + 1 bp |" in text


def write_paths(path, ids, year=2001, count=8):
    clock = pd.date_range(
        f"{year}-01-02 17:00", periods=count, freq="D", tz="America/New_York"
    ).tz_convert("UTC")
    np.savez(
        path,
        portfolio_id=np.array(ids),
        timestamp=np.array([x.isoformat() for x in clock]),
        equity=np.full((count, len(ids)), 1000.0),
        returns=np.zeros((count, len(ids))),
        cash_returns=np.zeros(count),
        exposure=np.full((count, len(ids)), 0.5),
    )


def test_derived_paths_exact_clocks_shapes_and_fraction_bounds(tmp_path):
    path = tmp_path / "synthetic.npz"
    write_paths(path, report.PRINCIPAL_IDS)
    times, values, cash = report._derived(path, "development")
    assert len(times) == 8 and tuple(values) == report.PRINCIPAL_IDS and cash.shape == (8,)
    with np.load(path, allow_pickle=False) as f:
        data = {k: f[k].copy() for k in f.files}
    data["exposure"][:] = 1.1
    np.savez(path, **data)
    with pytest.raises(seals.ProtocolError, match="wealth/exposure"):
        report._derived(path, "development")


def test_wrong_stage_rejected_before_any_npz_open(monkeypatch, tmp_path):
    monkeypatch.setattr(
        report.np, "load", lambda *args, **kwargs: pytest.fail("Unexpected NPZ open")
    )
    with pytest.raises(seals.ProtocolError, match="OOS"):
        report._derived(tmp_path / "unused", "historical_oos")


def test_2019_clock_rejected_before_any_numeric_array_read(monkeypatch, tmp_path):
    class FakeSource:
        files = ("portfolio_id", "timestamp", "equity", "returns", "cash_returns", "exposure")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __getitem__(self, key):
            if key == "portfolio_id":
                return np.array([report.TF_ID])
            if key == "timestamp":
                return np.array(["2019-01-02T22:00:00Z"])
            pytest.fail(f"Numeric array {key} read before clock guard")

    monkeypatch.setattr(report.np, "load", lambda *args, **kwargs: FakeSource())
    with pytest.raises(seals.ProtocolError, match="authorized stage"):
        report._derived(tmp_path / "metadata", "validation")


def test_duplicate_shared_tf_control_paths_match_declared_arithmetic_tolerance(tmp_path):
    write_paths(tmp_path / "principal_s3_daily.npz", [report.TF_ID])
    write_paths(tmp_path / "controls_s3_daily.npz", [report.TF_ID, "cash"])
    assert len(report._merged_paths(tmp_path, "development")[1]) == 2
    p = tmp_path / "controls_s3_daily.npz"
    with np.load(p, allow_pickle=False) as f:
        data = {k: f[k].copy() for k in f.files}
    data["equity"][0, 0] += 1
    np.savez(p, **data)
    with pytest.raises(seals.ProtocolError, match="unequal"):
        report._merged_paths(tmp_path, "development")


def test_development_chart_contains_exact_eight_principals_all_costs(tmp_path):
    report.selection_chart(metrics(), ["hybrid_core_070", report.TF_ID], tmp_path)
    assert (tmp_path / "development_allocation_sharpe.png").stat().st_size > 1000
    text = (tmp_path / "development_allocation_sharpe.svg").read_text()
    assert "2001–2011" in text and "selection" in text
    assert all(line.rstrip() == line for line in text.splitlines())


@pytest.mark.parametrize("kind", ["missing", "duplicate", "cost", "winner"])
def test_chart_refuses_unfrozen_or_incomplete_inputs(tmp_path, kind):
    frame = metrics()
    locks = [report.TF_ID]
    if kind == "missing":
        frame = frame.iloc[:-1]
    elif kind == "duplicate":
        frame = pd.concat([frame.iloc[:-1], frame.iloc[:1]])
    elif kind == "cost":
        frame.loc[0, "scenario_slippage_bps"] = 2
    else:
        locks = ["not_frozen"]
    with pytest.raises(seals.ProtocolError):
        report.selection_chart(frame, locks, tmp_path)


def test_six_path_wealth_exposure_headers_have_reserved_geometry(tmp_path):
    times = pd.date_range(
        "2012-01-03 17:00", periods=40, freq="D", tz="America/New_York"
    ).tz_convert("UTC")
    ids = [
        "hybrid_core_070",
        "hybrid_blend_090",
        report.REGIME_ID,
        report.TF_ID,
        "buy_and_hold",
        "hybrid_gated_mr",
    ]
    values = {
        cid: {
            "equity": np.linspace(1000, 1200 + j * 80, len(times)),
            "exposure": np.full(len(times), 0.2 + j * 0.1),
        }
        for j, cid in enumerate(ids)
    }
    review = report.wealth_exposure_chart(times, values, ids, "validation", tmp_path)
    assert review["header_geometry_nonoverlap"] is True
    text = (tmp_path / "validation_wealth_exposure.svg").read_text()
    assert "$1,000 initial capital" in text and "Portfolio wealth (USD)" in text
    assert "2012-01" in text and "Session-mark QQQ notional / wealth" in text
    assert len(review["plotted_data_sha256"]) == 6


def test_report_source_has_no_market_reader_and_states_hindsight_limits():
    code = inspect.getsource(report)
    assert "load_packet(" not in code and "read_reconciled_packet(" not in code
    assert '"historical_oos_authorized": False' in code
    assert "not separately funded sleeve weights" in code
    assert "seven" in code and "17:00" in code


def test_complete_synthetic_publication_all_pairs_costs_and_cash_nan(tmp_path, monkeypatch):
    fake = types.ModuleType("trend_following.research_hybrid_allocation_protocol")
    membership = [
        {
            "style": "trend",
            "candidate_id": f"tf_{j}",
            "parameters": {"entry_rule": {"period": 10}, "exit_rule": {"period": 200}},
        }
        for j in range(12)
    ] + [
        {
            "style": "mean_reversion",
            "candidate_id": f"mr_{j}",
            "parameters": {
                "rule": {"family": "SMA_DISTANCE", "period": 50 if j < 3 else 10},
                "entry_threshold": -0.04,
                "exit_threshold": 0.01,
            },
        }
        for j in range(32)
    ]
    fake.verify_study = lambda _: {
        "membership": membership,
        "pair_result_count_per_cost": 2688,
        "pairs": [
            {"pair_id": f"pair_{j}_{k}", "tf_candidate_id": f"tf_{j}", "mr_candidate_id": f"mr_{k}"}
            for j in range(12)
            for k in range(32)
        ],
    }
    fake.verify_results = lambda _, stage: {"stage": stage}
    fake.verify_locked_selections = lambda _: {
        "selections": [
            {"architecture": "A", "portfolio_id": report.TF_ID, "alpha": 1.0},
            {"architecture": "B", "portfolio_id": report.TF_ID, "alpha": 1.0},
        ]
    }
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(trend_following, "research_hybrid_allocation_protocol", fake, raising=False)
    monkeypatch.setattr(report, "ROOT", tmp_path)
    sidecar = tmp_path / "private"
    for stage, year in (("development", 2001), ("validation", 2012)):
        folder = sidecar / stage
        folder.mkdir(parents=True)
        metrics().to_csv(folder / "metrics.csv", index=False)
        controls = pd.DataFrame(
            [
                {**metrics().iloc[0].to_dict(), "portfolio_id": cid, "scenario_slippage_bps": slip}
                for cid in (report.TF_ID, "cash", "buy_and_hold", "hybrid_gated_mr")
                for slip in report.SLIPPAGES
            ]
        )
        controls.loc[controls.portfolio_id.eq("cash"), "cash_excess_sharpe"] = np.nan
        controls.to_csv(folder / "controls_metrics.csv", index=False)
        for slip in report.SLIPPAGES:
            paired = pd.concat([metrics().iloc[:1]] * 2688, ignore_index=True)
            paired["pair_id"] = [f"pair_{j}" for j in range(2688)]
            paired.to_csv(folder / f"pair_metrics_s{slip}.csv", index=False)
        write_paths(folder / "principal_s3_daily.npz", report.PRINCIPAL_IDS, year)
        write_paths(
            folder / "controls_s3_daily.npz",
            [report.TF_ID, "cash", "buy_and_hold", "hybrid_gated_mr"],
            year,
        )
        pd.DataFrame(
            [
                {
                    "portfolio_id": report.TF_ID,
                    "year": year,
                    "annual_return": 0.1,
                    "reference_price": 99,
                }
            ]
        ).to_csv(folder / "annual_returns.csv", index=False)
        pd.DataFrame(
            {
                "portfolio_id": [report.TF_ID, "cash"],
                report.TF_ID: [1.0, np.nan],
                "cash": [np.nan, np.nan],
            }
        ).to_csv(folder / "correlations.csv", index=False)
    pd.DataFrame(
        [
            {
                "architecture": "A",
                "portfolio_id": report.TF_ID,
                "reference_id": "buy_and_hold",
                "block_length": 5,
                "ci_lower": -0.1,
                "ci_upper": 0.1,
                "reference_price": 99,
            }
        ]
    ).to_csv(sidecar / "validation/bootstrap.csv", index=False)
    output = tmp_path / "reports/hybrid_allocation_research/synthetic"
    summary = report.publish_report(sidecar, output)
    assert summary["pair_count"] == 384 and summary["principal_count"] == 8
    assert len(pd.read_csv(output / "development_pair_metrics.csv")) == 10752
    assert "reference_price" not in pd.read_csv(output / "validation_bootstrap.csv")
    assert "reference_price" not in pd.read_csv(output / "development_annual_returns.csv")
    correlation = pd.read_csv(output / "validation_correlations.csv")
    assert correlation.cash.isna().all()
    text = (output / "README.md").read_text()
    assert "Pure trend won architecture(s) A, B" in text
    assert "twelve funded accounts" in text and "2019 onward remains closed" in text


def test_shared_tf_roundoff_does_not_change_chosen_plot_binding(tmp_path):
    write_paths(tmp_path / "principal_s3_daily.npz", [report.TF_ID])
    write_paths(tmp_path / "controls_s3_daily.npz", [report.TF_ID, "cash"])
    path = tmp_path / "controls_s3_daily.npz"
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    data["equity"][0, 0] += 1e-11
    np.savez(path, **data)
    times, paths = report._merged_paths(tmp_path, "development")
    assert paths[report.TF_ID]["equity"][0] == 1000.0
    assert len(times) == 8


def test_bootstrap_schema_keeps_subject_and_all_architecture_aliases():
    assert {"subject_id", "comparison_aliases", "reference_id"}.issubset(report.BOOTSTRAP_COLUMNS)


def test_correlation_matrix_preserves_cash_and_rejects_quote_columns():
    frame = pd.DataFrame(
        {
            "portfolio_id": [report.TF_ID, "cash"],
            report.TF_ID: [1.0, np.nan],
            "cash": [np.nan, np.nan],
        }
    )
    result = report._correlations(frame)
    assert result.cash.isna().all() and len(result) == 2
    frame["reference_price"] = 123
    with pytest.raises(seals.ProtocolError, match="declared portfolio"):
        report._correlations(frame)


def test_correlation_matrix_cannot_drop_undefined_row_but_keep_its_column():
    frame = pd.DataFrame({"portfolio_id": [report.TF_ID], report.TF_ID: [1.0], "cash": [np.nan]})
    with pytest.raises(seals.ProtocolError, match="undefined cases"):
        report._correlations(frame)


def test_architecture_lock_cannot_select_other_architecture_or_fixed_regime():
    for wrong in ("hybrid_blend_070", report.REGIME_ID):
        with pytest.raises(seals.ProtocolError):
            report._locks(
                {
                    "selections": [
                        {"architecture": "A", "portfolio_id": wrong},
                        {"architecture": "B", "portfolio_id": report.TF_ID},
                    ]
                }
            )


def test_pair_membership_uses_actual_protocol_schema_and_both_constituent_ids():
    from trend_following import research_daily_hourly_v5 as tf
    from trend_following import research_hybrid_allocation_protocol as protocol
    from trend_following import research_mean_reversion as mr

    trends = [
        tf.Candidate(tf.IndicatorRule(f, period=n), tf.IndicatorRule("SMA", period=200), 3, 3)
        for f in ("SMA", "EMA")
        for n in (10, 20, 50, 100, 200, 250)
    ]
    reversions = mr.candidate_grid()[:32]
    membership = [
        {"style": style, "candidate_id": c.candidate_id, "parameters": c.to_dict()}
        for style, group in (("trend", trends), ("mean_reversion", reversions))
        for c in sorted(group, key=lambda c: c.candidate_id)
    ]
    pairs = protocol._pair_definitions(membership)
    result = report._pair_membership({"membership": membership, "pairs": pairs})
    assert len(result) == 384
    assert {"trend_candidate_id", "mean_reversion_candidate_id", "pair_id"}.issubset(result)
    assert (
        result.trend_candidate_id.nunique() == 12
        and result.mean_reversion_candidate_id.nunique() == 32
    )
    alternate = [
        {
            (
                "trend_candidate_id"
                if k == "tf_candidate_id"
                else "mean_reversion_candidate_id"
                if k == "mr_candidate_id"
                else k
            ): v
            for k, v in row.items()
        }
        for row in pairs
    ]
    pd.testing.assert_frame_equal(
        result, report._pair_membership({"membership": membership, "pairs": alternate})
    )


def test_pair_projection_cannot_silently_omit_a_constituent():
    membership = [{"style": "trend", "candidate_id": f"tf_{j}"} for j in range(12)] + [
        {"style": "mean_reversion", "candidate_id": f"mr_{j}"} for j in range(32)
    ]
    pairs = [
        {"pair_id": f"pair_{j}_{k}", "tf_candidate_id": f"tf_{j}", "mr_candidate_id": f"mr_{k}"}
        for j in range(12)
        for k in range(32)
    ]
    del pairs[0]["mr_candidate_id"]
    with pytest.raises(seals.ProtocolError, match="identify unchanged"):
        report._pair_membership({"membership": membership, "pairs": pairs})
