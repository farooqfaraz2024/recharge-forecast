"""Command line: fit, simulate, check and report.

    python -m rechargefc.forecast --settings demo/settings.toml --history demo/history.csv

Writes to the output folder (default: out/):

    forecast_daily.csv   P10 / P50 / P90 recharging customers and revenue per day
    summary.json         monthly range, probability of reaching target, checks
    report.html          interactive report (open in any browser)

Mid-month: add the days already passed in the forecast month to the history file.
Those days are kept as actuals and only the remaining days are forecast.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from rechargefc.config import load
from rechargefc.data import (
    DOW_LABELS,
    SALARY_BINS,
    SALARY_LABELS,
    check_salary_coverage,
    load_history,
    month_days,
)
from rechargefc.model import fit
from rechargefc.simulate import benchmarks, holdout_check, sensitivity, simulate


def _interval(draws: np.ndarray) -> list[list[float]]:
    q = np.quantile(draws, [0.1, 0.5, 0.9], axis=0)
    return np.round(q.T if q.ndim > 1 else q[None, :], 4).tolist()


def learned_effects(fitted, settings) -> dict:
    out = {
        "weekday": {"labels": DOW_LABELS, "q": _interval(fitted.draws("dow"))},
        "salary": {
            "labels": [SALARY_LABELS[b] for b in SALARY_BINS[:-1]],
            "q": _interval(fitted.draws("salary")),
        },
        "drift": _interval(fitted.draws("drift"))[0],
        "sigma": _interval(fitted.draws("sigma"))[0],
    }
    if settings.events:
        out["events"] = {
            "labels": [e.name for e in settings.events],
            "q": _interval(fitted.draws("event")),
        }
    if fitted.season_months:
        out["season"] = {
            "labels": [datetime(2000, m, 1).strftime("%B") for m in fitted.season_months],
            "q": _interval(fitted.draws("season")),
        }
    return out


def run(
    settings_path: Path,
    history_path: Path,
    out_dir: Path,
    holdout_days: int = 10,
    quick: bool = False,
) -> dict:
    settings = load(settings_path)
    if quick:
        from dataclasses import replace

        settings = replace(settings, draws=300, chains=2)
    history = load_history(history_path)
    forecast_days = month_days(settings.month)
    if max(history["date"]) > forecast_days[-1]:
        raise ValueError("History runs past the forecast month")
    check_salary_coverage(sorted(set(history["date"]) | set(forecast_days)), settings.salary_dates)

    print(f"Fitting the model on {len(history)} days of history ...")
    fitted = fit(history, settings)
    print(f"Simulating {settings.month:%B %Y} ...")
    sim = simulate(fitted, settings)
    p10, p50, p90 = sim.quantiles(sim.total_revenue)
    print("Sensitivity ...")
    sens = sensitivity(fitted, settings)
    bench = benchmarks(history, settings, float(np.median(sim.base_end)))
    holdout = None
    if holdout_days and len(history) > holdout_days + 28:
        print(f"Holdout check on the last {holdout_days} days ...")
        holdout = holdout_check(history, settings, holdout_days)

    out_dir.mkdir(parents=True, exist_ok=True)
    rev_q = np.quantile(sim.revenue, [0.1, 0.5, 0.9], axis=0)
    cust_q = np.quantile(sim.customers, [0.1, 0.5, 0.9], axis=0)
    daily = pd.DataFrame(
        {
            "date": [str(d) for d in sim.days],
            "actual": sim.is_actual,
            "customers_p10": cust_q[0].round(),
            "customers_p50": cust_q[1].round(),
            "customers_p90": cust_q[2].round(),
            "revenue_p10": rev_q[0].round(2),
            "revenue_p50": rev_q[1].round(2),
            "revenue_p90": rev_q[2].round(2),
        }
    )
    daily.to_csv(out_dir / "forecast_daily.csv", index=False)

    # Same month last year, aligned by day of month, for the chart
    ly_key = f"{settings.month.year - 1}-{settings.month.month:02d}"
    ly = history[[f"{d:%Y-%m}" == ly_key for d in history["date"]]]
    ly_by_day = {d.day: float(r) for d, r in zip(ly["date"], ly["recharge_revenue"], strict=True)}

    brackets = [
        {
            "name": b.name,
            "avg_value": b.avg_value,
            "share": b.share,
            "revenue_p50": p50 * b.avg_value * b.share / settings.bracket_value,
        }
        for b in settings.brackets
    ]

    summary = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "month": settings.month_key,
        "month_label": f"{settings.month:%B %Y}",
        "currency": settings.currency,
        "target_revenue": settings.target_revenue,
        "revenue": {"p10": p10, "p50": p50, "p90": p90},
        "customers_per_day_p50": float(np.median(sim.total_customers) / len(sim.days)),
        "prob_reach_target": float(np.mean(sim.total_revenue >= settings.target_revenue))
        if settings.target_revenue
        else None,
        "actual_days": int(sim.is_actual.sum()),
        "actual_revenue_so_far": float(sim.revenue[0, sim.is_actual].sum()),
        "base_start": float(history["active_base"].iloc[-1]),
        "base_end_p50": float(np.median(sim.base_end)),
        "history": {
            "days": len(history),
            "months": sorted({f"{d:%Y-%m}" for d in history["date"]}),
        },
        "diagnostics": fitted.diagnostics,
        "benchmarks": bench,
        "holdout": holdout,
        "sensitivity": sens.to_dict(orient="records"),
        "effects": learned_effects(fitted, settings),
        "brackets": brackets,
        "bracket_value": settings.bracket_value,
        "inputs": {
            "Gross adds per day": vars(settings.gross_adds_per_day),
            "Daily churn rate": vars(settings.daily_churn_rate),
            "Tourist recharging customers per day": vars(settings.tourist_recharging_per_day),
            "Tourism factor": vars(settings.tourism_factor),
            "Recharge value per customer": vars(settings.value_per_customer),
        },
        "events": [
            {"name": e.name, "dates": [str(d) for d in e.dates if d.month == settings.month.month]}
            for e in settings.events
        ],
        "salary_dates": [
            str(d) for d in settings.salary_dates if f"{d:%Y-%m}" == settings.month_key
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    hist_mass, edges = np.histogram(sim.total_revenue, bins=40)
    chart = {
        "daily": daily.to_dict(orient="list"),
        "last_year": [ly_by_day.get(d.day) for d in sim.days],
        "histogram": {"counts": hist_mass.tolist(), "edges": edges.tolist()},
        "history": {
            "date": [str(d) for d in history["date"]],
            "revenue": history["recharge_revenue"].tolist(),
        },
    }
    from rechargefc.report import write_report

    write_report(summary, chart, out_dir / "report.html")
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Bayesian monthly recharge forecast")
    ap.add_argument("--settings", type=Path, default=Path("demo/settings.toml"))
    ap.add_argument("--history", type=Path, default=Path("demo/history.csv"))
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--holdout-days", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="fewer draws, for a fast first look")
    args = ap.parse_args(argv)
    s = run(args.settings, args.history, args.out, args.holdout_days, args.quick)
    r, cur = s["revenue"], s["currency"]
    print(
        f"\n{s['month_label']} recharge revenue ({cur}): "
        f"P10 {r['p10']:,.0f} | P50 {r['p50']:,.0f} | P90 {r['p90']:,.0f}"
    )
    if s["prob_reach_target"] is not None:
        print(f"Chance of reaching {s['target_revenue']:,.0f}: {s['prob_reach_target']:.0%}")
    print(f"Report: {args.out / 'report.html'}")


if __name__ == "__main__":
    main()
