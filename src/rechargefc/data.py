"""Daily history and calendar features.

History file (CSV), one row per day:

    date                  YYYY-MM-DD
    recharging_customers  customers who recharged that day (residents + tourists)
    recharge_revenue      total recharge value that day
    active_base           active prepaid customers that day
    tourist_customers     optional: tourist customers who recharged that day

Any months can be included. Typically: the last full month and the same month last
year. Days already passed in the forecast month can be added too; the forecast then
keeps those actuals and only forecasts the remaining days.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from rechargefc.config import ConfigError, Settings

REQUIRED = ["date", "recharging_customers", "recharge_revenue", "active_base"]

# Salary-cycle phases. "none" is the reference: an ordinary day away from salary.
SALARY_BINS = ["pre", "day0", "day1", "day2", "day3_4", "none"]
SALARY_LABELS = {
    "pre": "1-2 days before",
    "day0": "Salary day",
    "day1": "Day +1",
    "day2": "Day +2",
    "day3_4": "Day +3/4",
    "none": "Other days",
}
DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def load_history(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ConfigError(f"History file is missing columns: {', '.join(missing)}")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").reset_index(drop=True)
    if df["date"].duplicated().any():
        raise ConfigError("History file has the same date more than once")
    for col in REQUIRED[1:]:
        if (df[col] <= 0).any() or df[col].isna().any():
            raise ConfigError(f"Column {col} must be positive on every day")
    return df


def month_key(d: date) -> str:
    return f"{d:%Y-%m}"


def month_days(first: date) -> list[date]:
    n = calendar.monthrange(first.year, first.month)[1]
    return [first + timedelta(days=i) for i in range(n)]


def salary_bin(d: date, salary_dates: tuple[date, ...]) -> str:
    previous = [s for s in salary_dates if s <= d]
    upcoming = [s for s in salary_dates if s > d]
    lag = (d - previous[-1]).days if previous else None
    lead = (upcoming[0] - d).days if upcoming else None
    if lag is not None and lag <= 4:
        return {0: "day0", 1: "day1", 2: "day2"}.get(lag, "day3_4")
    if lead is not None and lead <= 2:
        return "pre"
    return "none"


def check_salary_coverage(days: list[date], salary_dates: tuple[date, ...]) -> None:
    """Every day needs a known salary date before it, or the phase is guesswork."""
    first = min(days)
    if not any(first - timedelta(days=35) <= s <= first for s in salary_dates):
        raise ConfigError(
            f"Add the salary date(s) of the month before {first:%b %Y} to salary.dates"
        )
    last = max(days)
    if not any(last < s <= last + timedelta(days=35) for s in salary_dates):
        raise ConfigError(f"Add the first salary date after {last:%d %b %Y} to salary.dates")


def calendar_features(days: list[date], settings: Settings) -> pd.DataFrame:
    """Features for any list of days: history or forecast."""
    target = settings.month
    feats = pd.DataFrame({"date": days})
    feats["dow"] = [d.weekday() for d in days]
    feats["salary_bin"] = [SALARY_BINS.index(salary_bin(d, settings.salary_dates)) for d in days]
    feats["month_key"] = [month_key(d) for d in days]
    # whole years before the forecast year, for the year-on-year drift term
    feats["years_before"] = [target.year - d.year for d in days]
    for ev in settings.events:
        feats[f"event:{ev.name}"] = [int(d in ev.dates) for d in days]
    return feats


def tourist_series(days: list[date], settings: Settings, history: pd.DataFrame | None = None):
    """Tourist recharging customers per day for history days (known values)."""
    out = []
    observed = {}
    if history is not None and "tourist_customers" in history.columns:
        observed = dict(zip(history["date"], history["tourist_customers"], strict=True))
    for d in days:
        if d in observed and not pd.isna(observed[d]):
            out.append(float(observed[d]))
            continue
        key = month_key(d)
        if key in settings.history_tourism:
            per_day, factor = settings.history_tourism[key]
            out.append(per_day * factor)
        elif key == settings.month_key:
            out.append(settings.tourist_recharging_per_day.value * settings.tourism_factor.value)
        else:
            raise ConfigError(
                f"No tourist figures for {key}: add a tourist_customers column or "
                f'[tourists.history."{key}"] to the settings'
            )
    return np.asarray(out, dtype=float)
