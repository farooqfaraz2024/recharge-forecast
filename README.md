# Recharge Forecast: Bayesian monthly revenue forecast for prepaid mobile

Forecasts next month's prepaid recharge revenue as a **range with probabilities**, not a single number:

> December 2026: P50 **AED 91.9M**, 80% range **85.6M – 98.5M**, **64%** chance of reaching the 90M target.

It combines what the planning team already knows (gross adds, churn, tourism, recharge value, salary dates, holidays) with what the history says (how strongly salary day, weekends and holidays move recharges), using a Bayesian model in **PyMC**. Every input carries its own uncertainty, and the forecast range reflects all of it.

**[Interactive report on demo data →](https://YOUR-USERNAME.github.io/recharge-forecast/)** (`docs/index.html`)

All data in this repository is **synthetic**, generated for a fictional operator. No real operator data is included.

---

## Why Bayesian

A planning forecast is usually built from two months: the **same month last year** and the **latest month**. Neither is the forecast month itself:

- Last year's December has the right calendar (salary dates, National Day, New Year's Eve, tourist season), but behaviour may have moved in a year.
- Last month is current, but it is a different calendar month.

A Bayesian model states how far each can be off (a **prior**: "behaviour moves about ±5% a year", "another month differs about ±10%") and weighs the two. Its range is wide when the evidence is thin, and it narrows as more months are added. A regression on 62 rows with one point estimate would hide this uncertainty.

## The model

For each day:

```
recharging customers = active base × share who recharge + tourist customers
revenue              = recharging customers × value per customer

log(share who recharge) = level                 forecast month's recharge rate
                        + day-of-week effect
                        + salary-cycle effect   1–2 days before / salary day / +1 / +2 / +3–4
                        + event effects         National Day, New Year's Eve, ...
                        + other-month effect    e.g. October vs December
                        − drift × years before  year-on-year change
```

- **Learned from history:** the level and the effects, using Student-t noise so that one outage day doesn't distort them.
- **From the business:** the inputs below. Each is drawn with its stated uncertainty in every simulation.
- **Active base:** rolled forward daily, `B(t) = B(t−1) × (1 − daily churn) + gross adds`, starting from the last known base.
- **Simulation:** 4,000 simulated months. Totals are summed within each simulation before percentiles are taken. Adding daily percentiles would overstate the range.

| Business input | Demo value | Uncertainty |
|---|---|---|
| Gross adds per day | 1,300 | ±10% |
| Daily churn rate | 0.06% | ±10% |
| Tourist recharging customers per day | 2,200 | ±15% |
| Tourism factor (forecast month) | 1.30 | ±10% |
| Recharge value per customer | 35.75 | ±3% |
| Revenue brackets (share, avg value) | 30% @10, 35% @25, 22% @50, 13% @100 | – |
| Salary dates, events | in `settings.toml` | – |

## Checks (demo data)

| Check | Result |
|---|---|
| True December revenue (synthetic, kept aside) | 93.7M, inside the 85.6–98.5M range |
| Holdout: last 10 days hidden, then forecast | model error **+0.8%**, naive 28-day average **−3.1%**; all 10 days inside the 80% band |
| Known effects recovered | salary day +17% (true +18%), day +1 +12% (true +12%), noise 0.031 (true 0.03) |
| Benchmarks | same month LY × base growth: 89.6M; last month × days: 88.6M |
| Sampler | R-hat 1.004, 0 divergences |

The model's middle forecast sits above both rules of thumb. Both miss the December uplift from tourism and calendar effects.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m rechargefc.demo          # regenerate demo data (optional)
python -m rechargefc.forecast      # ~40 s; writes out/report.html, summary.json, forecast_daily.csv
python -m pytest -q                # 23 tests, ~25 s
```

Options: `--settings`, `--history`, `--out`, `--holdout-days 10`, `--quick` (fewer draws for a first look).

### On your own data

1. Copy `demo/settings.toml` and replace every number. Comments in the file explain each setting.
2. Export a daily history CSV. Typically this covers the latest full month and the same month last year; more months give a narrower range.

   | column | meaning |
   |---|---|
   | `date` | YYYY-MM-DD |
   | `recharging_customers` | customers who recharged that day, residents and tourists |
   | `recharge_revenue` | total recharge value that day |
   | `active_base` | active prepaid customers that day |
   | `tourist_customers` | optional; otherwise taken from `[tourists.history]` |

3. Put both files in `data/`, which is git-ignored, and run:
   `python -m rechargefc.forecast --settings data/settings.toml --history data/history.csv --out data/out`

**Mid-month update:** add the days already passed in the forecast month to the history. They are kept as actuals, and only the remaining days are forecast.

**Before running, check:**
- Salary dates are actual pay dates, shifted when they fall on a weekend. Include the month before the history and the first date after the forecast month.
- Official holiday dates are confirmed.
- Churn is a **daily** rate. Monthly churn is rejected.

## Limitations

- With two history months, year-on-year drift and the other-month difference are not separately identified. Their priors carry that doubt into the range, which is the honest outcome. Adding months narrows it.
- Events missing from the settings are not forecast: a tariff change, a campaign, or a network incident.
- Revenue by bracket is a split of the total using the configured shares. Bracket mix is not modelled separately.
- Hijri-calendar events such as Ramadan and Eid move about 11 days a year. Add them as events with their dates for each year.

## Project layout

```
src/rechargefc/
  config.py      settings (TOML) with validation
  data.py        history loading, salary-cycle and calendar features
  model.py       PyMC model
  simulate.py    month simulation, sensitivity, benchmarks, holdout
  forecast.py    command line
  report.py      interactive HTML report (Plotly)
  demo.py        synthetic data with known truth
tests/           unit tests and model tests (parameter recovery, nowcast, CLI)
demo/            synthetic history, settings, and the hidden true December
docs/            report published with GitHub Pages
```

Stack: Python, PyMC, ArviZ, NumPy, pandas, Plotly.js, pytest.

## License

MIT
