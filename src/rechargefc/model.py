"""The Bayesian model of daily recharging customers.

For each day d:

    recharging customers  =  active base(d) x propensity(d)  +  tourists(d)

    log propensity(d) = level                      recharge rate in the forecast month
                      + day-of-week effect
                      + salary-cycle effect        before / on / after salary days
                      + event effects              e.g. National Day, New Year's Eve
                      + season effect              other calendar months vs the forecast month
                      - drift x years before       change in behaviour from one year to the next

The observed value is compared on a log scale with Student-t noise, so a single odd
day (an outage, a data glitch) cannot drag the whole model.

Every effect has a prior: a statement of how large it can plausibly be. With only a
couple of months of history this matters. The same month last year is the main
evidence for the forecast month's level, but its level has to be carried forward a
year, and the drift prior says how far behaviour can have moved. The latest month
is the freshest evidence, but it is a different calendar month, and the season prior
says how different it can be. The model weighs the two, and the forecast range shows
how sure it can be.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rechargefc.config import Settings
from rechargefc.data import DOW_LABELS, SALARY_BINS, calendar_features, tourist_series


@dataclass
class FitResult:
    idata: object  # arviz InferenceData
    history: pd.DataFrame
    features: pd.DataFrame
    tourists: np.ndarray
    season_months: list[int]  # calendar months with their own season effect
    value_daily_sd: float  # day-to-day variation of value per recharging customer
    diagnostics: dict

    def draws(self, name: str) -> np.ndarray:
        """Posterior draws with chains flattened: shape (draws, ...)."""
        arr = self.idata.posterior[name].values
        return arr.reshape(-1, *arr.shape[2:])


def _season_index(features: pd.DataFrame, season_months: list[int]) -> np.ndarray:
    months = [int(k.split("-")[1]) for k in features["month_key"]]
    k = len(season_months)
    return np.array([season_months.index(m) if m in season_months else k for m in months])


def fit(history: pd.DataFrame, settings: Settings, progress: bool = False) -> FitResult:
    import pymc as pm
    import pytensor.tensor as pt

    days = list(history["date"])
    feats = calendar_features(days, settings)
    tourists = tourist_series(days, settings, history)
    base = history["active_base"].to_numpy(dtype=float)
    customers = history["recharging_customers"].to_numpy(dtype=float)
    if (customers <= tourists).any():
        raise ValueError("Tourist customers exceed total recharging customers on some days")

    target_month = settings.month.month
    season_months = sorted({int(k.split("-")[1]) for k in feats["month_key"]} - {target_month})
    events = [e.name for e in settings.events]
    event_matrix = feats[[f"event:{n}" for n in events]].to_numpy(dtype=float) if events else None

    resident_rate = np.log(np.median((customers - tourists) / base))
    p = settings.priors
    coords = {"weekday": DOW_LABELS, "salary_phase": SALARY_BINS[:-1]}
    if events:
        coords["event_name"] = events
    if season_months:
        coords["other_month"] = [f"{m:02d}" for m in season_months]

    with pm.Model(coords=coords):
        level = pm.Normal("level", resident_rate, 0.5)
        dow = pm.ZeroSumNormal("dow", sigma=0.2, dims="weekday")
        salary = pm.Normal("salary", 0.0, p.effect_sd, dims="salary_phase")
        drift = pm.Normal("drift", 0.0, p.year_drift_sd)
        sigma = pm.HalfNormal("sigma", 0.1)

        salary_all = pt.concatenate([salary, pt.zeros(1)])
        eta = (
            level
            + dow[feats["dow"].to_numpy()]
            + salary_all[feats["salary_bin"].to_numpy()]
            - drift * feats["years_before"].to_numpy()
        )
        if events:
            event = pm.Normal("event", 0.0, p.effect_sd, dims="event_name")
            eta = eta + pt.dot(event_matrix, event)
        if season_months:
            season = pm.Normal("season", 0.0, p.season_sd, dims="other_month")
            season_all = pt.concatenate([season, pt.zeros(1)])
            eta = eta + season_all[_season_index(feats, season_months)]

        expected = base * pt.exp(eta) + tourists
        pm.StudentT("obs", nu=5, mu=pt.log(expected), sigma=sigma, observed=np.log(customers))

        idata = pm.sample(
            draws=settings.draws,
            tune=1000,
            chains=settings.chains,
            target_accept=0.9,
            random_seed=settings.seed,
            progressbar=progress,
        )

    import arviz as az

    summary = az.summary(idata, kind="diagnostics")
    diagnostics = {
        "max_rhat": float(summary["r_hat"].max()),
        "min_ess": float(summary["ess_bulk"].min()),
        "divergences": int(idata.sample_stats["diverging"].values.sum()),
    }

    value = np.log(history["recharge_revenue"] / history["recharging_customers"])
    by_month = value.groupby(feats["month_key"].to_numpy()).transform("mean")
    value_daily_sd = float((value - by_month).std(ddof=1)) if len(value) > 2 else 0.02

    return FitResult(idata, history, feats, tourists, season_months, value_daily_sd, diagnostics)
