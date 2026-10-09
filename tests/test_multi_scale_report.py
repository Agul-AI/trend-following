"""Synthetic report projection and visualization checks; never load market prices."""

import inspect

import numpy as np
import pandas as pd
import pytest

from trend_following import research_multi_scale_report as report
from trend_following import research_quality_protocol_v5 as seals

IDS = (
    "multi_scale_sma100_entry_only",
    "multi_scale_sma200_entry_only",
    "multi_scale_sma100_recovery_or_trend_break",
    "multi_scale_sma200_recovery_or_trend_break",
)


def metrics():
    return pd.DataFrame(
        [
            {
                "basket_id": cid,
                "scenario_slippage_bps": slip,
                "cash_excess_sharpe": 0.5 - j * 0.02 - slip * 0.001,
                "cagr": 0.07,
                "max_drawdown": -0.1,
                "exposure_fraction": 0.25,
                "gross_sleeve_orders_per_year": 30,
                "annualized_turnover": 8,
                "blocked_oversold_checks": 100,
                "trend_break_exit_count": 5,
                "entry_cancellation_count": 1,
                "average_holding_hours": 12,
                "source_packet": "/Users/private/.cache/prices",
                "reference_price": 22,
            }
            for j, cid in enumerate(IDS)
            for slip in report.SLIPPAGES
        ]
    )


def test_public_projection_retains_diagnostics_and_excludes_quotes_and_paths():
    result = report._public(metrics())
    assert "reference_price" not in result and "source_packet" not in result
    assert {
        "blocked_oversold_checks",
        "trend_break_exit_count",
        "entry_cancellation_count",
        "average_holding_hours",
    }.issubset(result)


@pytest.mark.parametrize("key", ["cash_excess_sharpe", "cagr", "max_drawdown", "basket_id"])
def test_required_projection_fields(key):
    with pytest.raises(seals.ProtocolError):
        report._public(metrics().drop(columns=key))


@pytest.mark.parametrize(
    "private", ["/Users/secret", "/home/secret", ".cache/private", "PRIVATE KEY", "api_key=x"]
)
def test_no_private_identifiers(private):
    frame = metrics()
    frame.loc[0, "basket_id"] = private
    with pytest.raises(seals.ProtocolError, match="Private"):
        report._public(frame)


def test_locked_selections_exactly_two_unique_identities():
    assert report._locked_ids({"selections": [{"basket_id": IDS[0]}, {"basket_id": IDS[2]}]}) == (
        IDS[0],
        IDS[2],
    )
    for bad in (
        {},
        {"selections": []},
        {"selections": [IDS[0], IDS[0]]},
        {"selections": [{"not_id": 1}, IDS[0]]},
    ):
        with pytest.raises(seals.ProtocolError):
            report._locked_ids(bad)


def test_tables_show_cash_undefined_and_no_net_roundtrip_claim():
    frame = metrics()
    frame.loc[0, "cash_excess_sharpe"] = np.nan
    text = "\n".join(report._table(frame.iloc[:4], costs=True))
    assert "—" in text and "Gross sleeve orders/year" in text
    assert "round trips" not in text
    for slip in report.SLIPPAGES:
        assert f" {slip} + 1 bp |" in text


def test_safe_table_identity():
    frame = metrics()
    frame.loc[0, "basket_id"] = "bad|label"
    with pytest.raises(seals.ProtocolError, match="Unsafe"):
        report._table(frame)


def test_chart_requires_all_four_baskets_and_costs(tmp_path):
    report.selection_chart(metrics(), (IDS[0], IDS[2]), tmp_path)
    assert (tmp_path / "development_filter_sharpe.png").stat().st_size > 1000
    svg = (tmp_path / "development_filter_sharpe.svg").read_text()
    assert "252-session annualization" in svg and "1-bp commission" in svg
    assert "/Users/" not in svg
    assert all(line.rstrip() == line for line in svg.splitlines())


@pytest.mark.parametrize("kind", ["missing", "duplicate", "cost", "lock"])
def test_chart_rejects_incomplete_or_unsealed_inputs(tmp_path, kind):
    frame = metrics()
    lock = (IDS[0], IDS[2])
    if kind == "missing":
        frame = frame.iloc[:-1]
    if kind == "duplicate":
        frame = pd.concat([frame.iloc[:-1], frame.iloc[:1]])
    if kind == "cost":
        frame.loc[0, "scenario_slippage_bps"] = 2
    if kind == "lock":
        lock = (IDS[0], "not_frozen")
    with pytest.raises(seals.ProtocolError):
        report.selection_chart(frame, lock, tmp_path)


def test_wealth_exposure_figure_does_not_average_returns(tmp_path):
    clock = pd.date_range("2001-01-02 22:00", periods=30, freq="D", tz="UTC")
    paths = {
        cid: {
            "equity": np.linspace(1000, 1100 + j * 100, 30),
            "exposure": np.full(30, 0.2 + j * 0.1),
        }
        for j, cid in enumerate((*IDS[:2], "buy_and_hold"))
    }
    report.wealth_exposure_chart(clock, paths, list(paths), "development", tmp_path)
    svg = (tmp_path / "development_wealth_exposure.svg").read_text()
    assert "Session-mark QQQ notional / wealth" in svg
    assert "initial $1,000" in svg
    assert "2001-01" in svg


def test_paths_exact_shape_and_wealth_bounds(tmp_path):
    path = tmp_path / "good.npz"
    np.savez(
        path,
        portfolio_id=np.array(IDS[:2]),
        timestamp=np.array(["2001-01-02T22:00:00Z", "2001-01-03T22:00:00Z"]),
        equity=np.full((2, 2), 1000.0),
        returns=np.zeros((2, 2)),
        cash_returns=np.zeros(2),
        exposure=np.zeros((2, 2)),
    )
    times, paths, cash = report._paths(path)
    assert len(times) == 2 and tuple(paths) == IDS[:2] and cash.shape == (2,)
    np.savez(
        path,
        portfolio_id=np.array(IDS[:2]),
        timestamp=np.array(["2001-01-02T22:00:00Z", "2001-01-03T22:00:00Z"]),
        equity=np.full((2, 2), 1000.0),
        returns=np.zeros((2, 2)),
        cash_returns=np.zeros(2),
        exposure=np.full((2, 2), 1.1),
    )
    with pytest.raises(seals.ProtocolError, match="wealth/exposure"):
        report._paths(path)


def test_report_has_no_price_packet_reader_or_oos_route():
    code = inspect.getsource(report)
    assert "load_packet(" not in code and "read_reconciled_packet" not in code
    assert '"historical_oos_authorized": False' in code
    assert "constituent_metrics_s" in code


def test_publish_complete_synthetic_artifact_safe_projection(tmp_path, monkeypatch):
    from trend_following import research_multi_scale_protocol as protocol

    sidecar = tmp_path / "private"
    ids = (*IDS, "portfolio_tf_000", "portfolio_tf_050", "buy_and_hold", "cash")
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
                "entry_threshold": -0.02,
                "exit_threshold": 0,
            },
        }
        for j in range(32)
    ]
    monkeypatch.setattr(protocol, "verify_study", lambda _: {"membership": membership})
    monkeypatch.setattr(protocol, "verify_results", lambda _, stage: {"stage": stage})
    monkeypatch.setattr(
        protocol,
        "verify_locked_selections",
        lambda _: {"selections": [{"basket_id": IDS[0]}, {"basket_id": IDS[2]}]},
    )
    monkeypatch.setattr(report, "ROOT", tmp_path)
    for stage, year in (("development", 2001), ("validation", 2012)):
        folder = sidecar / stage
        folder.mkdir(parents=True)
        metrics().to_csv(folder / "metrics.csv", index=False)
        controls = pd.DataFrame(
            [
                {
                    "portfolio_id": pid,
                    "basket_id": np.nan,
                    "scenario_slippage_bps": slip,
                    "cash_excess_sharpe": 0.3,
                    "cagr": 0.05,
                    "max_drawdown": -0.1,
                }
                for pid in ids[4:]
                for slip in report.SLIPPAGES
            ]
        )
        controls.to_csv(folder / "controls_metrics.csv", index=False)
        clock = np.array([f"{year}-01-02T22:00:00Z", f"{year}-01-03T22:00:00Z"])
        for slip in report.SLIPPAGES:
            constituents = pd.concat([metrics().iloc[:1]] * 128, ignore_index=True)
            constituents.to_csv(folder / f"constituent_metrics_s{slip}.csv", index=False)
        for prefix, members in (("basket", ids[:4]), ("controls", ids[4:])):
            np.savez(
                folder / f"{prefix}_s3_daily.npz",
                portfolio_id=np.array(members),
                timestamp=clock,
                equity=np.full((2, len(members)), 1000.0),
                returns=np.zeros((2, len(members))),
                cash_returns=np.zeros(2),
                exposure=np.zeros((2, len(members))),
            )
        pd.DataFrame(
            [
                {
                    "year": year,
                    "annual_return": 0.1,
                    "source_packet": "/Users/private",
                    "reference_price": 42,
                }
            ]
        ).to_csv(folder / "annual_returns.csv", index=False)
    pd.DataFrame(
        [
            {
                "basket_id": IDS[0],
                "reference_id": "buy_and_hold",
                "block_length": 5,
                "ci_lower": -0.1,
                "ci_upper": 0.1,
                "reference_price": 42,
            }
        ]
    ).to_csv(sidecar / "validation/bootstrap.csv", index=False)
    output = tmp_path / "reports/multi_scale_research/synthetic"
    result = report.publish_report(sidecar, output)
    assert result["modified_rule_count"] == 128 and result["2019_onward_prices_opened"] is False
    assert len(pd.read_csv(output / "development_constituents.csv")) == 512
    assert "source_packet" not in pd.read_csv(output / "development_annual_returns.csv")
    assert "reference_price" not in pd.read_csv(output / "validation_bootstrap.csv")
    text = (output / "README.md").read_text()
    assert "29 shorter-period" in text and "200-day SMA/EMA" in text
    assert "| SMA_DISTANCE | 50 |" in text
    assert "2019 onward remains closed" in text
    assert "nan |" not in "\n".join(report._table(controls))
