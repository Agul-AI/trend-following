"""Synthetic gates for same development candidates and a newly bounded resplit."""

import copy
import inspect
import json

import numpy as np
import pandas as pd
import pytest

from trend_following import research_intersection_validation as original
from trend_following import research_resplit_validation as study


def authorization(tmp_path, **changes):
    value = {
        "schema": "trend_following_resplit_intersection_validation_authorization_v1",
        "authorized_stages": ["validation_only"],
        "validation_only": True,
        "explicit_user_approval": True,
        "authorized_by": "human_user",
        "user_request": study.USER_REQUEST,
        "historical_oos_authorized": False,
        "shared_wait_hours": [2.0, 3.0, 4.0],
        "cash_nominal_annual_rate": 0.03,
        "validation_period": dict(study.PERIOD),
        "historical_oos_period": dict(study.HISTORICAL_OOS_PERIOD),
        **changes,
    }
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(value))
    return path


def test_explicit_human_resplit_authorizes_only_validation(tmp_path):
    result = study._authorization(authorization(tmp_path))
    assert result["validation_period"] == {"start": "2012-01-01", "end": "2018-12-31"}
    assert result["historical_oos_period"] == {"start": "2019-01-01", "end": "2026-05-28"}
    assert result["historical_oos_authorized"] is False


@pytest.mark.parametrize(
    "changes",
    [
        {"authorized_stages": ["validation_only", "historical_oos"]},
        {"historical_oos_authorized": True},
        {"historical_oos_authorized": 0},
        {"explicit_user_approval": 1},
        {"explicit_user_approval": False},
        {"validation_only": 1},
        {"validation_only": False},
        {"validation_period": {"start": "2012-01-01", "end": "2015-12-31"}},
        {"historical_oos_period": {"start": "2016-01-01", "end": "2026-05-28"}},
        {"shared_wait_hours": [2, 4]},
        {"cash_nominal_annual_rate": 0.02},
        {"user_request": "run OOS"},
        {"authorized_by": "agent"},
    ],
)
def test_authorization_rejects_wrong_period_stage_wait_cash_and_boolean(tmp_path, changes):
    with pytest.raises(study.Error):
        study._authorization(authorization(tmp_path, **changes))


def test_prior_selection_requires_exact_same_membership_and_reads_no_old_outcomes(
    tmp_path, monkeypatch
):
    calls = []
    pairs, candidates, dev_freeze, dev_result = [], [], {"source": 1}, {"source": 2}
    payload = {
        "schema": "trend_following_constant_cash_intersection_validation_v1",
        "pairs": pairs,
        "pairs_sha256": study.protocol.canonical_hash(pairs),
        "candidates": candidates,
        "candidate_set_sha256": study.protocol.canonical_hash(candidates),
        "configuration_count_per_wait": 159,
        "configuration_count": 477,
        "wait_hours": [2.0, 3.0, 4.0],
        "development_freeze_sha256": study.protocol.canonical_hash(dev_freeze),
        "development_result_sha256": study.protocol.canonical_hash(dev_result),
        "prior_validation_or_OOS_outcomes_used_for_selection": False,
        "historical_oos_authorized": False,
        "source_hashes": {},
        "runtime": {"python": "synthetic"},
        "candidate_set_sealed_at": "2026-10-08T17:00:00Z",
    }
    monkeypatch.setattr(study, "PAIR_SHA256", payload["pairs_sha256"])
    monkeypatch.setattr(study.protocol, "_runtime", lambda: payload["runtime"])
    monkeypatch.setattr(
        study.protocol, "_read_seal", lambda p: calls.append(p) or copy.deepcopy(payload)
    )
    assert study._original_selection(tmp_path, pairs, candidates, dev_freeze, dev_result) == payload
    assert calls == [tmp_path / "freeze.json"]
    payload["candidates"] = [{"candidate_id": "replacement"}]
    with pytest.raises(study.Error, match="exact original"):
        study._original_selection(tmp_path, pairs, candidates, dev_freeze, dev_result)


def test_resplit_source_does_not_call_old_validation_verifier_or_old_loader():
    source = inspect.getsource(study)
    assert "original.verify_results(" not in source
    assert "original.verify_study(" not in source
    assert "original.load_validation_packet(" not in source
    assert "protocol._load_stage(" not in source
    assert "historical-oos" not in inspect.getsource(study.main)


def test_preserved_selection_hash_and_count_are_fixed():
    assert study.PAIR_COUNT == 159
    assert study.CANDIDATE_COUNT == 477
    assert study.PAIR_SHA256 == "c6a7c803561fcf7073b50b8e40496a2304c57ba9b587f526124816f08f73cbe6"
    assert study.SOURCE_FILES[: len(original.SOURCE_FILES)] == original.SOURCE_FILES


def test_separate_tree_rejects_old_intersection_and_ancestors(tmp_path):
    source, development, prior = (tmp_path / item for item in ("source", "development", "prior"))
    for target in (prior, prior / "nested", source, development, tmp_path):
        with pytest.raises(study.Error):
            study._safe_roots(target, source, development, prior)
    study._safe_roots(tmp_path / "new", source, development, prior)


def test_prepare_cannot_parse_prices_before_intact_selection(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        study, "verify_study", lambda *_: (_ for _ in ()).throw(study.Error("selection missing"))
    )
    monkeypatch.setattr(
        study.prefix, "prepare_validation_prefix", lambda *_a, **_k: calls.append("prices")
    )
    with pytest.raises(study.Error, match="selection missing"):
        study.prepare_study(tmp_path / "new")
    assert not calls
    assert (tmp_path / "new" / "attempts.jsonl").is_file()


def test_evaluate_cannot_parse_prices_before_candidate_and_prepared_seals(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        study, "verify_study", lambda *_: {"candidate_set_sealed_at": "2026-10-08T17:00Z"}
    )
    monkeypatch.setattr(
        study, "verify_prepared", lambda *_: (_ for _ in ()).throw(study.Error("prepared missing"))
    )
    monkeypatch.setattr(study, "load_validation_packet", lambda *_: calls.append("prices"))
    with pytest.raises(study.Error, match="prepared missing"):
        study.evaluate_study(tmp_path / "new")
    assert not calls


def test_reader_refuses_OOS_and_wrong_period_before_prepared_access(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(study, "verify_prepared", lambda *_: calls.append("prepared"))
    valid = {
        "stage": "validation",
        "validation_only": True,
        "historical_oos_authorized": False,
        "period": dict(study.PERIOD),
    }
    for changes in (
        {"stage": "historical_oos"},
        {"historical_oos_authorized": True},
        {"period": {"start": "2012-01-01", "end": "2019-01-01"}},
        {"validation_only": 1},
    ):
        with pytest.raises(study.Error):
            study.load_validation_packet(tmp_path, {**valid, **changes})
    assert not calls


def test_new_period_config_is_copied_not_written_back(tmp_path, monkeypatch):
    config = {"segments": {"validation": {"start": "2012-01-01", "end": "2015-12-31"}}}
    old = copy.deepcopy(config)
    daily = pd.DataFrame({"daily_price_available_at": [pd.Timestamp("2018-12-31T22:00Z")]})
    calendar = pd.DataFrame({"market_open": [pd.Timestamp("1999-11-01T14:30Z")]})
    empty = pd.DataFrame()
    monkeypatch.setattr(study, "verify_prepared", lambda *_: {"manifest_sha256": "sealed"})
    monkeypatch.setattr(
        study.prefix,
        "read_validation_prefix",
        lambda *_a, **_k: (daily, empty, calendar, empty, empty, {}),
    )
    monkeypatch.setattr(study.protocol, "load_config", lambda *_: config)
    monkeypatch.setattr(study.protocol, "_read_seal", lambda *_: {"gap_authorization": {}})
    observed = []
    monkeypatch.setattr(
        study.protocol,
        "_check_prefix",
        lambda *_a: observed.append(_a[-1]["segments"]["validation"]),
    )
    monkeypatch.setattr(study.protocol, "_quality_gate", lambda *_: None)
    payload = {
        "stage": "validation",
        "validation_only": True,
        "historical_oos_authorized": False,
        "period": dict(study.PERIOD),
        "config_path": str(tmp_path / "immutable.yaml"),
        "source_study": str(tmp_path / "source"),
    }
    result = study.load_validation_packet(tmp_path, payload)
    assert config == old
    assert result[-1]["segments"]["validation"] == study.PERIOD
    assert observed == [study.PERIOD]
    assert result[-1]["segments"]["historical-oos"] == study.HISTORICAL_OOS_PERIOD


def test_counts_retain_all_candidates_and_medians_over_all_pairs():
    rules = list(study.fixed.candidates_for_wait(2))[:5]
    pairs = [
        {
            "pair_id": original._pair_id(c),
            "entry_rule": c.entry_rule.to_dict(),
            "exit_rule": c.exit_rule.to_dict(),
        }
        for c in rules
    ]
    pairs = sorted(pairs, key=lambda pair: pair["pair_id"])
    metrics = pd.DataFrame(
        {
            "candidate_id": c.candidate_id,
            "entry_rule": c.entry_rule.rule_id,
            "exit_rule": c.exit_rule.rule_id,
            "entry_confirmation_hours": hours,
            "exit_confirmation_hours": hours,
            "cash_excess_sharpe": [0.7, np.nan, np.inf, 0.5, 0.3][index],
            "trade_count": index * 10,
            "cagr": index / 100,
            "max_drawdown": -index / 10,
            "annualized_volatility": index / 20,
            "exposure_fraction": index / 10,
        }
        for hours in study.WAIT_HOURS
        for index, c in enumerate(study.candidates_from_pairs(pairs, hours))
    )
    counts = study.counts_against_bh(metrics, 0.5, pairs)
    assert counts.configuration_count.eq(5).all()
    assert counts.beat_BH_Sharpe_count.eq(1).all()
    assert counts.undefined_Sharpe_count.eq(2).all()
    assert counts.median_trade_count.eq(20).all()
    assert counts.median_cagr.eq(0.02).all()
    assert counts.median_max_drawdown.eq(-0.2).all()
    metrics.loc[0, "cagr"] = np.nan
    with pytest.raises(study.Error, match="Nonfinite validation"):
        study.counts_against_bh(metrics, 0.5, pairs)
