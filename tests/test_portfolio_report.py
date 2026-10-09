"""Synthetic projection and chart checks; never read market-price packets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trend_following import research_portfolio_engine as engine
from trend_following import research_portfolio_report as report
from trend_following import research_quality_protocol_v5 as seals


def synthetic_metrics():
    return pd.DataFrame(
        [
            {
                **allocation.to_dict(),
                "scenario_slippage_bps": slip,
                "cash_excess_sharpe": 0.4 - (allocation.trend_weight - 0.5) ** 2 - slip / 100,
                "cagr": 0.08,
                "max_drawdown": -0.15,
                "gross_sleeve_orders_per_year": 12,
                "annualized_turnover": 3,
                "distinct_execution_timestamp_count": 24,
                "initial_weighted_binary_long_time_fraction": 0.2,
                "positive_weight_sleeve_count": 44,
                "elapsed_calendar_years": 11,
                "exposure_definition": "calendar_time_weighted_event_left_QQQ_notional_over_total_wealth",
            }
            for slip in (1, 3, 10, 25)
            for allocation in engine.allocation_grid()
        ]
    )


def test_public_projection_excludes_quotes_private_paths_and_placeholder_parameters():
    frame = synthetic_metrics()
    frame["reference_price"] = 123
    frame["source_study"] = "/Users/private/.cache/prices"
    frame["candidate_id"] = "unused_passive_placeholder"
    frame["entry_threshold"] = -2
    result = report._public(frame)
    assert not {"reference_price", "source_study", "candidate_id", "entry_threshold"} & set(result)
    assert {
        "positive_weight_sleeve_count",
        "distinct_execution_timestamp_count",
        "initial_weighted_binary_long_time_fraction",
        "exposure_definition",
    }.issubset(result)


@pytest.mark.parametrize("required", ["cash_excess_sharpe", "cagr"])
def test_public_metrics_require_essential_fields(required):
    with pytest.raises(seals.ProtocolError, match="Required"):
        report._public(synthetic_metrics().drop(columns=required))


@pytest.mark.parametrize(
    "private", ["/Users/secret", "/home/secret", ".cache/prices", "PRIVATE KEY", "api_key=x"]
)
def test_public_identifier_cannot_leak_private_information(private):
    frame = synthetic_metrics()
    frame.loc[0, "portfolio_id"] = private
    with pytest.raises(seals.ProtocolError, match="Private information"):
        report._public(frame)


def test_report_path_rejects_symlink_ancestors(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(seals.ProtocolError, match="Unsafe"):
        report._safe(link / "report")


def test_clean_handles_undefined_and_numpy_without_inventing_scores():
    assert report._clean({"score": np.nan, "count": np.int64(3), "items": [np.inf]}) == {
        "score": None,
        "count": 3,
        "items": [None],
    }
    assert report._number(np.nan) == "—"


def test_development_chart_contains_all_tested_allocations(tmp_path):
    frame = synthetic_metrics()
    benchmark = pd.DataFrame(
        [{"portfolio": "buy_and_hold", "scenario_slippage_bps": 3, "cash_excess_sharpe": 0.1}]
    )
    report.development_chart(frame, benchmark, "portfolio_tf_050", tmp_path)
    assert (tmp_path / "development_allocation_sharpe.png").stat().st_size > 1000
    svg = (tmp_path / "development_allocation_sharpe.svg").read_text()
    assert "Initial capital allocated" in svg
    assert "252-session annualization" in svg
    assert "Locked baseline allocation" in svg
    assert "/Users/" not in svg


@pytest.mark.parametrize("change", ["missing", "duplicate", "identity", "winner"])
def test_chart_fails_on_incomplete_or_mismatched_inputs(tmp_path, change):
    frame = synthetic_metrics()
    winner = "portfolio_tf_050"
    if change == "missing":
        frame = frame.iloc[:-1]
    elif change == "duplicate":
        frame = pd.concat([frame.iloc[:-1], frame.iloc[:1]], ignore_index=True)
    elif change == "identity":
        frame = frame.drop(columns="trend_weight")
    else:
        winner = "not_frozen"
    with pytest.raises(seals.ProtocolError):
        report.development_chart(frame, pd.DataFrame(), winner, tmp_path)


def test_table_keeps_order_metric_distinct_from_round_trips():
    lines = report._table(synthetic_metrics().iloc[:1])
    assert "Gross sleeve orders/year" in lines[0]
    assert "round trips" not in "\n".join(lines)


def test_stress_table_labels_every_cost_scenario():
    frame = synthetic_metrics()
    lines = report._table(frame.loc[frame.portfolio_id.eq("portfolio_tf_050")], show_cost=True)
    assert "Slippage + 1-bp commission" in lines[0]
    for slip in (1, 3, 10, 25):
        assert f" {slip} + 1 bp |" in "\n".join(lines)


def test_report_source_has_no_price_reader_or_oos_route():
    import inspect

    code = inspect.getsource(report)
    assert "read_validation_prefix" not in code
    assert "read_reconciled_packet" not in code
    assert "load_packet(" not in code
    assert 'historical_oos_authorized": False' in code
