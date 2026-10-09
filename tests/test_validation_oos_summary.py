"""Publication-helper checks use SYNTHETIC metrics/curves; no evaluation APIs."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "stage_summary", ROOT / "scripts/summarize_validation_oos.py"
)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def candidates():
    core = helper.protocol._core()
    return [
        core.Candidate(
            core.IndicatorRule("EMA", period=100), core.IndicatorRule("EMA", period=100), 0.0, 0.0
        ),
        core.Candidate(
            core.IndicatorRule("EMA", period=200), core.IndicatorRule("EMA", period=100), 5.5, 2.5
        ),
    ]


def test_rank_respects_frozen_tolerance_and_winner_not_flag_filter():
    models = candidates()
    low, high = sorted(models, key=lambda c: c.candidate_id)
    metrics = pd.DataFrame(
        {
            "candidate_id": [high.candidate_id, low.candidate_id],
            "cash_excess_sharpe": [1 + 0.5e-10, 1.0],
            "has_post_gap_confirmed_signal": [False, True],
        }
    )
    ranked = helper.rank_validation(metrics, models, low.candidate_id)
    assert ranked.candidate_id.iloc[0] == low.candidate_id
    assert ranked.selected_validation_winner.iloc[0]
    assert ranked.has_post_gap_confirmed_signal.iloc[0]
    with pytest.raises(ValueError, match="sealed winner"):
        helper.rank_validation(metrics, models, high.candidate_id)


def test_missing_release_stops_before_any_numeric_csv_read(tmp_path, monkeypatch):
    monkeypatch.setattr(
        helper,
        "release_report",
        lambda *args: (_ for _ in ()).throw(ValueError("winner not sealed")),
    )
    monkeypatch.setattr(
        pd,
        "read_csv",
        lambda *args, **kwargs: pytest.fail("Numeric CSV opened before release gate"),
    )
    with pytest.raises(ValueError, match="winner not sealed"):
        helper.publish(
            tmp_path / "config", tmp_path / "study", "historical-oos", tmp_path / "output"
        )
    assert not (tmp_path / "output").exists()


def test_historical_stage_checked_only_after_validation_winner_integrity(tmp_path, monkeypatch):
    model = candidates()[0]
    p = helper.protocol
    calls = []
    monkeypatch.setattr(p, "load_config", lambda *args: {})
    monkeypatch.setattr(p, "_verify_registration", lambda *args: {})
    monkeypatch.setattr(p, "_verify_prepared", lambda *args: {})

    def stage(study, name, *args):
        calls.append(name)
        if name == "development":
            return {"finalist_count": 1, "finalists_sha256": p.canonical_hash({"finalists": 1})}
        if name == "validation":
            return {"winner_id": "wrong", "winner_sha256": "wrong"}
        pytest.fail("OOS numeric/stage release attempted without the validated winner")

    monkeypatch.setattr(p, "_verify_stage", stage)
    monkeypatch.setattr(
        p, "_candidate_table", lambda config: ([model], {model.candidate_id: model})
    )
    monkeypatch.setattr(
        p,
        "_load_candidates",
        lambda path, table, count: (
            [model],
            {"finalists": 1}
            if "finalists" in str(path)
            else {"sealed_before_historical_oos": True},
        ),
    )
    with pytest.raises(ValueError, match="frozen selection"):
        helper.release_report(tmp_path / "config", tmp_path / "study", "historical-oos")
    assert calls == ["development", "validation"]


def test_daily_curves_whitelist_rejects_raw_price_or_trade_fields(tmp_path):
    path = tmp_path / "daily.csv"
    pd.DataFrame({"timestamp": ["2016-01-04T22:00Z"], "equity": [1000.0], "close": [123.0]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="never quote/trade"):
        helper.read_public_daily(path)
    pd.DataFrame({"timestamp": ["2016-01-04T22:00Z"], "equity": [1000.0], "return": [0.0]}).to_csv(
        path, index=False
    )
    result = helper.read_public_daily(path)
    assert list(result) == ["timestamp", "equity", "return"]


def test_independent_control_daily_aliases_are_explicitly_normalized(tmp_path):
    path = tmp_path / "daily.csv"
    frame = pd.DataFrame(
        {
            "timestamp": ["2016-01-04T22:00Z"],
            "equity": [1100.0],
            "strategy_return": [0.1],
            "valuation_exposure": [1.0],
        }
    )
    frame.to_csv(path, index=False)
    result = helper.read_public_daily(path)
    assert list(result) == ["timestamp", "equity", "return", "exposure"]
    assert result["return"].iloc[0] == 0.1
    assert result.exposure.iloc[0] == 1.0
    for canonical in ("return", "exposure"):
        collision = frame.assign(**{canonical: [0.0]})
        collision.to_csv(path, index=False)
        with pytest.raises(ValueError, match="Ambiguous derived daily"):
            helper.read_public_daily(path)


def write_policies(root, identifier, slippage, mismatched=False):
    for policy in ("winner", "buy_and_hold", "cash"):
        folder = root / policy
        folder.mkdir(parents=True)
        pd.DataFrame(
            {
                "candidate_id": [identifier if policy == "winner" else f"benchmark_{policy}"],
                "commission_bps": [1],
                "slippage_bps": [slippage],
                "entry_rule": ["dummy"],
                "cash_excess_sharpe": [1],
                "cagr": [0.1],
                "max_drawdown": [-0.1],
                "terminal_equity": [1100],
            }
        ).to_csv(folder / "metrics.csv", index=False)
        stamp = "2016-01-05T22:00Z" if mismatched and policy == "cash" else "2016-01-04T22:00Z"
        pd.DataFrame(
            {
                "timestamp": [stamp],
                "equity": [1100],
                "return": [0.1],
                "cash_return": [0.01],
                "exposure": [1],
            }
        ).to_csv(folder / "daily.csv", index=False)


def test_four_scenario_report_preserves_single_winner_and_drops_benchmark_filters(tmp_path):
    identifier = candidates()[1].candidate_id
    for slip in helper.COSTS:
        root = tmp_path / f"cost_{slip}"
        write_policies(root, identifier, slip)
        metrics, curves = helper._curves(root, identifier, slip)
        assert set(metrics.policy) == {identifier, "buy_and_hold", "cash"}
        assert metrics.loc[metrics.policy.ne(identifier), "entry_rule"].isna().all()
        assert not {"open", "high", "low", "close", "quantity", "execution_price"}.intersection(
            curves
        )


def test_benchmark_endpoint_mismatch_rejected(tmp_path):
    identifier = candidates()[0].candidate_id
    write_policies(tmp_path / "curves", identifier, 3, mismatched=True)
    with pytest.raises(ValueError, match="matched daily valuation endpoints"):
        helper._curves(tmp_path / "curves", identifier, 3)
