"""Mechanical v4 tests on fabricated prices only; no strategy family or real data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trend_following import research_blind_engine_v4 as engine


def market(prices, dividends=None, cash=0.0, borrow=0.0):
    ix = pd.bdate_range("2000-01-03", periods=len(prices), tz="America/New_York") + pd.Timedelta(
        hours=16
    )
    raw = pd.Series(prices, index=ix)
    div = pd.Series(0.0 if dividends is None else dividends, index=ix)
    tr = ((raw + div) / raw.shift(1) - 1).fillna(0)
    data = pd.DataFrame(
        {
            "close": raw,
            "dividend": div,
            "split": 1.0,
            "total_return": tr,
            "total_return_index": 100 * (1 + tr).cumprod(),
            "price_return": raw.pct_change().fillna(0),
        }
    )
    return engine.Market(data, pd.Series(borrow, index=ix), pd.Series(cash, index=ix))


def free_costs():
    return engine.Costs(
        fee_bps=0.0,
        slippage_bps=0.0,
        financing_spread_bps=0.0,
        expense_ratio=0.0,
        extra_drag=0.0,
        short_tax_rate=0.0,
        long_tax_rate=0.0,
        interest_tax_rate=0.0,
    )


def test_policy_only_receives_current_past_copies_and_next_close_delay():
    m = market([100, 110, 120, 130, 140])
    seen = []

    def callback(history, parameters):
        seen.append(list(history.close))
        history.iloc[-1, 0] = -1  # Cannot alter the trusted market frame.
        parameters["mutated"] = True
        return 1.0

    params = {}
    result = engine.build_targets(m, callback, params, 3, start="2000-01-03", end="2000-01-07")
    assert seen == [[100], [100, 110], [100, 110, 120], [110, 120, 130], [120, 130, 140]]
    assert result.tolist() == [0, 1, 1, 1, 1]
    assert params == {}
    assert m.data.close.tolist() == [100, 110, 120, 130, 140]


def test_future_quote_changes_cannot_change_earlier_targets():
    m = market([100, 110, 90, 120, 100, 130, 110, 140])

    def callback(history, parameters):
        return float(history.total_return_index.iloc[-1] > history.total_return_index.mean())

    a = engine.build_targets(m, callback, {}, 3, start="2000-01-03", end="2000-01-12")
    other = market([100, 110, 90, 120, 100, 13000, 11000, 14000])
    b = engine.build_targets(other, callback, {}, 3, start="2000-01-03", end="2000-01-12")
    pd.testing.assert_series_equal(a.iloc[:6], b.iloc[:6])


def test_buy_after_completed_return_no_same_close_profit():
    m = market([100, 200, 200, 200])
    targets = pd.Series([0.0, 1.0, 1.0, 1.0], index=m.data.index)
    result = engine.evaluate_targets(
        m, targets, candidate_id="fabricated", costs=free_costs(), layers=("gross",)
    )
    assert result.metrics.iloc[0].terminal_equity == pytest.approx(1000)
    assert result.paths["gross"].orders.iloc[0].timestamp == m.data.index[1]


def test_daily_reset_funding_charged_once_and_bankruptcy_not_clipped():
    m = market([100, 100, 110, 120, 110], borrow=0.001)
    targets = pd.Series([0.0, 1.0, 1.0, 1.0, 1.0], index=m.data.index)
    result = engine.evaluate_targets(
        m, targets, candidate_id="fabricated", costs=free_costs(), layers=("gross", "financing")
    )
    gross = np.prod(1 + 3 * m.data.total_return.iloc[2:])
    net = np.prod(1 + 3 * m.data.total_return.iloc[2:] - 0.002)
    assert result.metrics.iloc[0].terminal_equity == pytest.approx(1000 * gross)
    assert result.metrics.iloc[1].terminal_equity == pytest.approx(1000 * net)
    bankrupt = market([100, 100, 50, 40])
    with pytest.raises(ValueError, match="bankruptcy"):
        engine.evaluate_targets(
            bankrupt,
            targets.iloc[:4].set_axis(bankrupt.data.index),
            candidate_id="bad",
            costs=free_costs(),
        )


def test_cash_reference_can_include_negative_yields():
    m = market([100] * 5, cash=-0.001)
    targets = pd.Series(0.0, index=m.data.index)
    result = engine.evaluate_targets(m, targets, candidate_id="cash", costs=free_costs())
    assert result.metrics.iloc[0].terminal_equity == pytest.approx(1000 * (0.999**4))
    assert np.isnan(result.metrics.iloc[0].cash_excess_Sharpe)


def test_dividend_drip_no_retroactive_interest_and_terminal_clone():
    m = market([100, 100, 99, 100], dividends=[0, 0, 1, 0], cash=0.01)
    targets = pd.Series([0.0, 1.0, 1.0, 1.0], index=m.data.index)
    result = engine.evaluate_targets(
        m, targets, candidate_id="QQQ", asset="QQQ", costs=free_costs(), layers=("gross", "tax")
    )
    path = result.paths["gross"]
    assert path.cash_interest.iloc[2] == pytest.approx(0)
    assert result.metrics.iloc[0].terminal_equity == pytest.approx((1010 / 100) * (100 / 99) * 100)
    assert path.holdings.iloc[-1].quantity > 0  # Continuous account not liquidated.
    for layer in result.daily_returns:
        assert np.prod(1 + result.daily_returns[layer]) * 1000 == pytest.approx(
            result.terminals[layer]["liquidated_equity"]
        )


@pytest.mark.parametrize(
    "code",
    [
        "import os",
        "import sys",
        "open('x')",
        "np.load('x')",
        "pd.read_csv('x')",
        "x.__class__",
        "getattr(x,'y')",
    ],
)
def test_policy_rejects_file_clock_introspection_surface(tmp_path, code):
    file = tmp_path / "policy.py"
    file.write_text(
        "import numpy as np\nimport pandas as pd\n"
        + code
        + "\ndef target(history, parameters): return 0.\n"
    )
    with pytest.raises(ValueError):
        engine.load_policy(file)


def test_raw_loader_ignores_no_external_adjusted_price_and_has_known_rates(tmp_path):
    dates = pd.bdate_range("2000-01-03", periods=5)
    pd.DataFrame(
        {
            "session": dates,
            "close": [100, 100, 49, 49, 50],
            "dividend_amount": [0, 0, 1, 0, 0],
            "split_coefficient": [1, 1, 2, 1, 1],
        }
    ).to_csv(tmp_path / "QQQ.csv", index=False)
    for name in ["DFF", "DTB3"]:
        pd.DataFrame(
            {
                "available_at": pd.date_range("1999-12-31", periods=10, tz="UTC"),
                "annual_rate": 0.0,
                "series_id": name,
            }
        ).to_csv(tmp_path / f"{name}_available.csv", index=False)
    m = engine.load_market(tmp_path, free_costs())
    assert m.data.total_return_index.iloc[:4].tolist() == pytest.approx([100] * 4)
    assert not any("adj" in c for c in m.data)


def test_artificial_policy_is_loaded(tmp_path):
    path = Path(tmp_path / "policy.py")
    path.write_text('def target(history, parameters):\n    return float(parameters["weight"])\n')
    assert engine.load_policy(path)(None, {"weight": 0.5}) == 0.5
