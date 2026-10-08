"""Pure comparison arithmetic; no market-data or other-study reads."""

import numpy as np
import pandas as pd
import pytest
from scripts import summarize_current_development as summary


@pytest.fixture
def example(monkeypatch):
    monkeypatch.setattr(summary, "COUNT", 6)
    return pd.DataFrame(
        {
            "candidate_id": list("abcdef"),
            "commission_bps": [1.0] * 6,
            "slippage_bps": [3.0] * 6,
            "cagr": [0.1, -0.1, 0.0, 0.2, 0.05, 0.01],
            "cash_excess_sharpe": [0.2, 0.3, 0.0, -0.1, 0.2, 0.1],
            "max_drawdown": [-0.2, -0.4, -0.3, -0.1, -0.35, -0.2],
            "has_post_gap_confirmed_signal": [True, False, False, True, False, False],
            "post_gap_confirmed_signal_count": [1, 0, 0, 2, 0, 0],
        }
    )


BH = {"cagr": 0.0, "cash_excess_sharpe": 0.0, "max_drawdown": -0.3}


def test_distinct_metric_comparisons_and_no_flag_filter(example):
    result = summary.compare_grid(example, BH)
    assert result["higher_net_cagr_than_qqq_bh"] == 4
    assert result["higher_cash_excess_sharpe_than_qqq_bh"] == 4
    assert result["shallower_daily_drawdown_than_qqq_bh"] == 3
    assert result["better_all_three_than_qqq_bh"] == 2
    assert result["post_gap_flagged_configurations"] == 2
    assert result["post_gap_confirmed_signals"] == 3
    assert result["primary_score_leader_id"] == "b"  # Not the highest-CAGR row.
    example["has_post_gap_confirmed_signal"] = ~example.has_post_gap_confirmed_signal
    assert summary.compare_grid(example, BH)["primary_score_leader_id"] == "b"


def test_primary_tie_uses_stable_id(example):
    example.loc[5, "cash_excess_sharpe"] = 0.30000000005
    assert summary.compare_grid(example, BH)["primary_score_leader_id"] == "b"


def test_wrong_costs_or_incomplete_grid_stop(example):
    with pytest.raises(ValueError, match="distinct"):
        summary.compare_grid(example.iloc[:-1], BH)
    example.loc[0, "slippage_bps"] = 10
    with pytest.raises(ValueError, match="baseline"):
        summary.compare_grid(example, BH)


def test_no_finite_score_is_not_an_invented_winner(example):
    example["cash_excess_sharpe"] = np.nan
    with pytest.raises(ValueError, match="No finite"):
        summary.compare_grid(example, BH)


def test_deterministic_compression_preserves_exact_bytes(tmp_path):
    source = tmp_path / "source.csv"
    source.write_bytes(b"candidate_id,cagr\na,0.1\n")
    first, second = tmp_path / "first.gz", tmp_path / "second.gz"
    summary.compress_copy(source, first)
    summary.compress_copy(source, second)
    assert first.read_bytes() == second.read_bytes()
    assert summary.clean_json({"missing": np.nan, "count": np.int64(6)}) == {
        "missing": None,
        "count": 6,
    }


def test_plain_benchmark_controls_have_no_dummy_indicator_labels():
    public = summary.benchmark_public_metrics(
        {"candidate_id": "benchmark_buy_and_hold", "entry_rule": "sma_10", "cagr": 0.1}
    )
    assert public == {"candidate_id": "benchmark_buy_and_hold", "cagr": 0.1}
