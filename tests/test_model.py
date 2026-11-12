"""Slower tests: fit the Bayesian model on synthetic data with known truth.

Uses fewer draws than a real run so the suite finishes in about a minute.
"""

import json
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from rechargefc.config import load
from rechargefc.data import load_history
from rechargefc.demo import generate
from rechargefc.forecast import run
from rechargefc.model import fit
from rechargefc.simulate import sensitivity, simulate


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    truth = generate(out)
    settings = replace(load(out / "settings.toml"), draws=400, chains=2)
    history = load_history(out / "history.csv")
    return out, truth, settings, history


@pytest.fixture(scope="module")
def fitted(setup):
    _, _, settings, history = setup
    return fit(history, settings)


def interval(x, lo=0.05, hi=0.95):
    return np.quantile(x, lo, axis=0), np.quantile(x, hi, axis=0)


def test_sampler_is_healthy(fitted):
    d = fitted.diagnostics
    assert d["divergences"] == 0
    assert d["max_rhat"] < 1.02


def test_recovers_known_effects(fitted, setup):
    truth = setup[1]
    lo, hi = interval(fitted.draws("salary"))
    true_salary = np.array(list(truth["salary"].values()))
    assert ((lo <= true_salary) & (true_salary <= hi)).mean() >= 0.8
    lo, hi = interval(fitted.draws("level"))
    assert lo <= truth["level"] <= hi
    lo, hi = interval(fitted.draws("sigma"))
    assert lo <= truth["sigma"] <= hi
    # salary day is clearly the strongest positive effect
    assert np.median(fitted.draws("salary"), axis=0).argmax() == 1


def test_outage_day_does_not_drag_level(fitted):
    # one day 30% down should not pull the typical level down by more than ~1%
    mid = np.median(fitted.draws("level"))
    assert abs(mid - np.log(0.040)) < 0.05


def test_true_december_inside_forecast_range(fitted, setup):
    _, truth, settings, _ = setup
    sim = simulate(fitted, settings)
    p10, p50, p90 = sim.quantiles(sim.total_revenue)
    assert p10 <= truth["december_2026_revenue"] <= p90
    assert p10 < p50 < p90
    assert sim.revenue.shape == (800, 31)
    assert not sim.is_actual.any()


def test_more_tourists_more_revenue(fitted, setup):
    _, _, settings, _ = setup
    up = replace(settings, tourism_factor=replace(settings.tourism_factor, value=2.0))
    assert np.median(simulate(fitted, up).total_revenue) > np.median(
        simulate(fitted, settings).total_revenue
    )


def test_sensitivity_value_dominates(fitted, setup):
    sens = sensitivity(fitted, setup[2])
    assert sens.iloc[0]["driver"] == "Recharge value per customer"
    churn = sens.set_index("driver").loc["Daily churn rate"]
    assert churn["high"] < 0 < churn["low"]  # more churn, less revenue


def test_nowcast_keeps_actual_days(setup, tmp_path):
    out, _, settings, history = setup
    dec = pd.read_csv(out / "truth_2026-12.csv").head(10)
    combined = pd.concat([pd.read_csv(out / "history.csv"), dec])
    combined.to_csv(tmp_path / "h.csv", index=False)
    h = load_history(tmp_path / "h.csv")
    f = fit(h, settings)
    sim = simulate(f, settings)
    assert sim.is_actual.sum() == 10
    assert np.allclose(sim.revenue[:, 0], dec["recharge_revenue"].iloc[0])
    so_far = dec["recharge_revenue"].sum()
    assert sim.total_revenue.min() > so_far


def test_cli_writes_outputs(setup, tmp_path):
    out = setup[0]
    summary = run(out / "settings.toml", out / "history.csv", tmp_path, holdout_days=0, quick=True)
    assert (tmp_path / "report.html").exists()
    daily = pd.read_csv(tmp_path / "forecast_daily.csv")
    assert len(daily) == 31
    assert (daily["revenue_p10"] <= daily["revenue_p50"]).all()
    assert (daily["revenue_p50"] <= daily["revenue_p90"]).all()
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["month"] == "2026-12" and 0 <= saved["prob_reach_target"] <= 1
    assert summary["revenue"]["p50"] == pytest.approx(saved["revenue"]["p50"])
    assert sum(b["revenue_p50"] for b in saved["brackets"]) == pytest.approx(
        saved["revenue"]["p50"]
    )


def test_history_after_month_is_rejected(setup, tmp_path):
    out, _, _, _ = setup
    h = pd.read_csv(out / "history.csv")
    extra = h.tail(1).assign(date=str(date(2027, 1, 2)))
    pd.concat([h, extra]).to_csv(tmp_path / "h.csv", index=False)
    with pytest.raises(ValueError, match="past the forecast month"):
        run(out / "settings.toml", tmp_path / "h.csv", tmp_path, holdout_days=0, quick=True)
