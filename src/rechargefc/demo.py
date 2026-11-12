"""Generate synthetic demo data with known true parameters.

    python -m rechargefc.demo            # writes demo/history.csv, demo/settings.toml,
                                         # demo/truth_2026-12.csv and demo/truth.json

The numbers describe a fictional prepaid operator. They are generated from the same
structure the model assumes, with known true effects, so the tests can check that
the model recovers them and that the forecast range contains the true December.
The history deliberately includes one outage day (14 Oct 2026, 30% below normal)
to show the model is not dragged by a single bad day.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from rechargefc.config import load
from rechargefc.data import DOW_LABELS, SALARY_BINS, calendar_features

DEMO_DIR = Path(__file__).resolve().parents[2] / "demo"

TRUTH = {
    "level": float(np.log(0.040)),  # resident recharge propensity in Dec 2026
    "dow": dict(zip(DOW_LABELS, [0.04, 0.02, 0.01, 0.03, 0.00, -0.04, -0.06], strict=True)),
    "salary": {"pre": -0.06, "day0": 0.18, "day1": 0.12, "day2": 0.06, "day3_4": 0.03},
    "event": {"National Day": 0.05, "New Year's Eve": 0.08},
    "season": {"10": -0.04},  # October vs December
    "drift": 0.03,  # propensity grew 3% in a year
    "sigma": 0.03,
    "gross_adds_per_day": 1300,
    "daily_churn_rate": 0.0006,
    "tourists": {"2025-12": (2000, 1.25), "2026-10": (2100, 1.00), "2026-12": (2200, 1.30)},
    "value": {"2025-12": 35.0, "2026-10": 35.5, "2026-12": 35.8},
    "outage": {"date": "2026-10-14", "effect": -0.30},
}

SETTINGS = """\
# Demo settings for a FICTIONAL prepaid operator. Replace every number with your own.
# Uncertainties are relative standard deviations: 0.10 = "probably within about ±10%".

[forecast]
month = "2026-12"
currency = "AED"
target_revenue = 90_000_000

[base]
# The active base is rolled forward from the last day in the history file.
gross_adds_per_day = 1300
gross_adds_uncertainty = 0.10
daily_churn_rate = 0.0006          # share of the base lost per day
daily_churn_uncertainty = 0.10

[tourists]
# Tourist (visitor) customers who recharge per day in an average month,
# and the tourism factor for the forecast month (1.30 = 30% above average).
recharging_per_day = 2200
recharging_uncertainty = 0.15
tourism_factor = 1.30
tourism_factor_uncertainty = 0.10

# Tourist figures for the history months (skip if the history file has a
# tourist_customers column).
[tourists.history."2025-12"]
recharging_per_day = 2000
tourism_factor = 1.25

[tourists.history."2026-10"]
recharging_per_day = 2100
tourism_factor = 1.00

[value]
recharge_value_per_customer = 35.75
value_uncertainty = 0.03

# Revenue bracketing: share of recharging customers and their average recharge.
[[value.brackets]]
name = "Up to 10"
avg_value = 10
share = 0.30

[[value.brackets]]
name = "11-25"
avg_value = 25
share = 0.35

[[value.brackets]]
name = "26-50"
avg_value = 50
share = 0.22

[[value.brackets]]
name = "Above 50"
avg_value = 100
share = 0.13

[salary]
# Actual pay dates, moved earlier when they fall on a weekend. Include the month
# before the earliest history month and the first date after the forecast month.
dates = [
  "2025-11-25", "2025-12-01", "2025-12-25", "2026-01-01",
  "2026-09-25", "2026-10-01", "2026-10-23", "2026-10-30",
  "2026-11-25", "2026-12-01", "2026-12-25", "2027-01-01",
]

# Events with their own effect. Confirm official holiday dates before use.
[[events]]
name = "National Day"
dates = ["2025-12-02", "2025-12-03", "2026-12-02", "2026-12-03"]

[[events]]
name = "New Year's Eve"
dates = ["2025-12-31", "2026-12-31"]

[priors]
year_drift_sd = 0.05    # recharge behaviour can move about ±5% in a year
season_sd = 0.10        # another month can differ about ±10% from the forecast month
effect_sd = 0.30

[run]
draws = 1000
chains = 4
seed = 7
"""


def _month(year: int, month: int) -> list[date]:
    first = date(year, month, 1)
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    return [first + timedelta(days=i) for i in range((nxt - first).days)]


def generate(out_dir: Path = DEMO_DIR, seed: int = 2026) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "settings.toml").write_text(SETTINGS, encoding="utf-8")
    settings = load(out_dir / "settings.toml")
    rng = np.random.default_rng(seed)

    # Base path from 1 Nov 2025 to 31 Dec 2026
    start, end = date(2025, 11, 1), date(2026, 12, 31)
    all_days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    base = {}
    b = 1_950_000.0
    for d in all_days:
        adds = rng.normal(TRUTH["gross_adds_per_day"], 80)
        b = b * (1 - TRUTH["daily_churn_rate"]) + adds
        base[d] = b

    def month_rows(year: int, month: int) -> pd.DataFrame:
        days = _month(year, month)
        key = f"{year}-{month:02d}"
        f = calendar_features(days, settings)
        eta = TRUTH["level"] + f["dow"].map(lambda i: TRUTH["dow"][DOW_LABELS[i]])
        eta = eta + f["salary_bin"].map(lambda i: TRUTH["salary"].get(SALARY_BINS[i], 0.0))
        for name, eff in TRUTH["event"].items():
            eta = eta + eff * f[f"event:{name}"]
        eta = eta + TRUTH["season"].get(f"{month:02d}", 0.0)
        eta = eta - TRUTH["drift"] * f["years_before"]
        per_day, factor = TRUTH["tourists"][key]
        tourists = per_day * factor * np.exp(rng.normal(0, 0.05, len(days)))
        b_arr = np.array([base[d] for d in days])
        noise = rng.normal(0, TRUTH["sigma"], len(days))
        outage = np.array([str(d) == TRUTH["outage"]["date"] for d in days])
        noise = noise + outage * TRUTH["outage"]["effect"]
        customers = (b_arr * np.exp(eta.to_numpy()) + tourists) * np.exp(noise)
        value = TRUTH["value"][key] * np.exp(rng.normal(0, 0.02, len(days)))
        return pd.DataFrame(
            {
                "date": [str(d) for d in days],
                "recharging_customers": np.round(customers).astype(int),
                "recharge_revenue": np.round(customers * value, 2),
                "active_base": np.round(b_arr).astype(int),
            }
        )

    history = pd.concat([month_rows(2025, 12), month_rows(2026, 10)], ignore_index=True)
    history.to_csv(out_dir / "history.csv", index=False)
    truth_month = month_rows(2026, 12)
    truth_month.to_csv(out_dir / "truth_2026-12.csv", index=False)
    truth = {**TRUTH, "december_2026_revenue": float(truth_month["recharge_revenue"].sum())}
    (out_dir / "truth.json").write_text(json.dumps(truth, indent=2, default=str))
    return truth


def main() -> None:
    truth = generate()
    print(f"Demo files written to {DEMO_DIR}")
    print(f"True December 2026 revenue (kept separate): {truth['december_2026_revenue']:,.0f}")


if __name__ == "__main__":
    main()
