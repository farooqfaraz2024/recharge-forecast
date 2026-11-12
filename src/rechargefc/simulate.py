"""Simulate the forecast month thousands of times.

Each simulation takes one set of model parameters from the fitted model and one
draw of every business input (gross adds, churn, tourists, tourism factor, value per
customer), rolls the active base forward day by day, and produces a full month of
daily recharging customers and revenue. The spread of the monthly totals across all
simulations is the forecast range.

Totals are added up within each simulation before percentiles are taken: adding
daily percentiles would give a range that is far too wide.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta

import numpy as np
import pandas as pd

from rechargefc.config import Settings, Uncertain
from rechargefc.data import calendar_features, month_days
from rechargefc.model import FitResult, _season_index


@dataclass
class Simulation:
    days: list[date]
    is_actual: np.ndarray  # days already in the history (kept as actuals)
    customers: np.ndarray  # (draws, days)
    revenue: np.ndarray  # (draws, days)
    base_end: np.ndarray  # active base on the last day, per draw
    total_revenue: np.ndarray  # (draws,)
    total_customers: np.ndarray  # (draws,)

    def quantiles(self, values: np.ndarray, qs=(0.1, 0.5, 0.9)) -> list[float]:
        return [float(np.quantile(values, q)) for q in qs]


def _draw(u: Uncertain, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draws of a positive business input with its stated relative uncertainty."""
    if u.rel_sd == 0 or u.value == 0:
        return np.full(n, u.value)
    sd = np.sqrt(np.log1p(u.rel_sd**2))
    return u.value * np.exp(rng.normal(-0.5 * sd**2, sd, n))  # mean stays at u.value


def roll_base(start: float, adds: np.ndarray, churn: np.ndarray, n_days: int) -> np.ndarray:
    """Active base for days 1..n after `start`, per draw: B(t) = B(t-1)(1-churn) + adds."""
    t = np.arange(1, n_days + 1)[None, :]
    keep = (1 - churn)[:, None] ** t
    steady = np.where(churn[:, None] > 0, adds[:, None] / np.maximum(churn[:, None], 1e-12), 0)
    growth = np.where(churn[:, None] > 0, steady * (1 - keep), adds[:, None] * t)
    return start * keep + growth


def simulate(
    fitted: FitResult,
    settings: Settings,
    days: list[date] | None = None,
    known_base: np.ndarray | None = None,
    known_tourists: np.ndarray | None = None,
    seed: int | None = None,
) -> Simulation:
    """Forecast `days` (default: the whole forecast month).

    Days present in the history keep their actual values. For the others, the base
    is rolled forward from the last history day unless `known_base` is given (used
    by the holdout check, where the real base is known).
    """
    rng = np.random.default_rng(settings.seed if seed is None else seed)
    days = days or month_days(settings.month)
    hist = fitted.history.set_index("date")
    is_actual = np.array([d in hist.index for d in days])
    future = [d for d in days if d not in hist.index]

    level = fitted.draws("level")
    n = len(level)
    customers = np.zeros((n, len(days)))
    revenue = np.zeros((n, len(days)))
    for j, d in enumerate(days):
        if is_actual[j]:
            customers[:, j] = hist.loc[d, "recharging_customers"]
            revenue[:, j] = hist.loc[d, "recharge_revenue"]

    last_day = max(hist.index)
    base_end = np.full(n, float(hist.loc[last_day, "active_base"]))
    if future:
        feats = calendar_features(future, settings)
        eta = level[:, None] + fitted.draws("dow")[:, feats["dow"].to_numpy()]
        salary = np.concatenate([fitted.draws("salary"), np.zeros((n, 1))], axis=1)
        eta = eta + salary[:, feats["salary_bin"].to_numpy()]
        eta = eta - fitted.draws("drift")[:, None] * feats["years_before"].to_numpy()[None, :]
        if settings.events:
            ev = feats[[f"event:{e.name}" for e in settings.events]].to_numpy(dtype=float)
            eta = eta + fitted.draws("event") @ ev.T
        if fitted.season_months:
            season = np.concatenate([fitted.draws("season"), np.zeros((n, 1))], axis=1)
            eta = eta + season[:, _season_index(feats, fitted.season_months)]

        if known_base is not None:
            base = np.broadcast_to(np.asarray(known_base, dtype=float), (n, len(future)))
        else:
            if min(future) <= last_day:
                raise ValueError("Forecast days must come after the history")
            offsets = np.array([(d - last_day).days for d in future])
            path = roll_base(
                float(hist.loc[last_day, "active_base"]),
                _draw(settings.gross_adds_per_day, n, rng),
                _draw(settings.daily_churn_rate, n, rng),
                int(offsets.max()),
            )
            base = path[:, offsets - 1]
            base_end = path[:, -1]

        if known_tourists is not None:
            tourists = np.broadcast_to(np.asarray(known_tourists, dtype=float), (n, len(future)))
        else:
            per_day = _draw(settings.tourist_recharging_per_day, n, rng)
            tourists = (per_day * _draw(settings.tourism_factor, n, rng))[:, None]

        noise = fitted.draws("sigma")[:, None] * rng.standard_t(5, size=(n, len(future)))
        sim_customers = (base * np.exp(eta) + tourists) * np.exp(noise)
        value = _draw(settings.value_per_customer, n, rng)[:, None] * np.exp(
            rng.normal(0, fitted.value_daily_sd, size=(n, len(future)))
        )
        cols = np.flatnonzero(~is_actual)
        customers[:, cols] = sim_customers
        revenue[:, cols] = sim_customers * value

    return Simulation(
        days=days,
        is_actual=is_actual,
        customers=customers,
        revenue=revenue,
        base_end=base_end,
        total_revenue=revenue.sum(axis=1),
        total_customers=customers.sum(axis=1),
    )


SENSITIVITY_DRIVERS = {
    "gross_adds_per_day": "Gross adds per day",
    "daily_churn_rate": "Daily churn rate",
    "tourist_recharging_per_day": "Tourist recharging customers per day",
    "tourism_factor": "Tourism factor",
    "value_per_customer": "Recharge value per customer",
}


def sensitivity(fitted: FitResult, settings: Settings, change: float = 0.10) -> pd.DataFrame:
    """Change in the median monthly revenue when each input is 10% lower or higher.

    The same random numbers are reused for every run, so differences come from the
    input change alone.
    """
    base_p50 = float(np.median(simulate(fitted, settings).total_revenue))
    rows = []
    for key, label in SENSITIVITY_DRIVERS.items():
        out = {"driver": label}
        for name, factor in (("low", 1 - change), ("high", 1 + change)):
            u: Uncertain = getattr(settings, key)
            varied = replace(settings, **{key: replace(u, value=u.value * factor)})
            p50 = float(np.median(simulate(fitted, varied).total_revenue))
            out[name] = p50 - base_p50
        rows.append(out)
    df = pd.DataFrame(rows)
    df["span"] = (df["high"] - df["low"]).abs()
    return df.sort_values("span", ascending=False).reset_index(drop=True)


def benchmarks(history: pd.DataFrame, settings: Settings, base_end_median: float) -> dict:
    """Simple forecasts a planner could make without a model."""
    hist = history.copy()
    hist["key"] = [f"{d:%Y-%m}" for d in hist["date"]]
    n_days = len(month_days(settings.month))
    out = {}
    ly_key = f"{settings.month.year - 1}-{settings.month.month:02d}"
    ly = hist[hist["key"] == ly_key]
    if len(ly) >= 25:
        ratio = base_end_median / ly["active_base"].mean()
        out["Same month last year x base growth"] = float(ly["recharge_revenue"].sum() * ratio)
    full = [k for k, g in hist.groupby("key") if len(g) >= 25 and k != settings.month_key]
    if full:
        latest = hist[hist["key"] == max(full)]
        out["Last full month daily average x days"] = float(
            latest["recharge_revenue"].mean() * n_days
        )
    return out


def holdout_check(history: pd.DataFrame, settings: Settings, days: int = 10) -> dict:
    """Refit without the last `days` days, forecast them, and compare with actuals."""
    from rechargefc.model import fit

    cutoff = max(history["date"]) - timedelta(days=days)
    train, test = history[history["date"] <= cutoff], history[history["date"] > cutoff]
    fitted = fit(train.reset_index(drop=True), replace(settings, draws=500, chains=4))
    from rechargefc.data import tourist_series

    test_days = list(test["date"])
    sim = simulate(
        fitted,
        settings,
        days=test_days,
        known_base=test["active_base"].to_numpy(),
        known_tourists=tourist_series(test_days, settings, test),
    )
    actual = test["recharge_revenue"].to_numpy()
    lo, hi = np.quantile(sim.revenue, [0.1, 0.9], axis=0)
    total_lo, total_mid, total_hi = sim.quantiles(sim.total_revenue)
    naive = float(train["recharge_revenue"].tail(28).mean() * len(test))
    return {
        "days": len(test),
        "from": str(min(test_days)),
        "to": str(max(test_days)),
        "daily_coverage_80": float(np.mean((actual >= lo) & (actual <= hi))),
        "actual_total": float(actual.sum()),
        "p10": total_lo,
        "p50": total_mid,
        "p90": total_hi,
        "inside_p10_p90": bool(total_lo <= actual.sum() <= total_hi),
        "model_error_pct": float((total_mid - actual.sum()) / actual.sum()),
        "naive_error_pct": float((naive - actual.sum()) / actual.sum()),
        "daily": {
            "dates": [str(d) for d in test_days],
            "actual": actual.tolist(),
            "p10": lo.tolist(),
            "p50": np.quantile(sim.revenue, 0.5, axis=0).tolist(),
            "p90": hi.tolist(),
        },
    }
