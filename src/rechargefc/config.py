"""Business inputs for the forecast, read from a TOML settings file.

Every driver is given as a best estimate plus an uncertainty. The uncertainty is a
relative standard deviation: ``0.10`` means "probably within about ±10%, almost
certainly within ±20-30%". The forecast carries every uncertainty through to the
final range, so being honest here is what makes the P10-P90 range trustworthy.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


class ConfigError(ValueError):
    """A settings file that cannot produce a sensible forecast."""


@dataclass(frozen=True)
class Uncertain:
    """A best estimate with a relative standard deviation."""

    value: float
    rel_sd: float

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ConfigError(f"Value must not be negative: {self.value}")
        if not 0 <= self.rel_sd < 1:
            raise ConfigError(f"Uncertainty must be between 0 and 1: {self.rel_sd}")


@dataclass(frozen=True)
class Bracket:
    name: str
    avg_value: float  # average recharge value of customers in this bracket
    share: float  # share of recharging customers in this bracket


@dataclass(frozen=True)
class Event:
    name: str
    dates: tuple[date, ...]


@dataclass(frozen=True)
class Priors:
    # How much recharge propensity (recharging customers per active customer) can
    # drift in one year, after base size and tourists are accounted for.
    year_drift_sd: float = 0.05
    # How different another calendar month can be from the forecast month.
    season_sd: float = 0.10
    # How big a salary-day or event effect can plausibly be (log scale).
    effect_sd: float = 0.30


@dataclass(frozen=True)
class Settings:
    month: date  # first day of the forecast month
    currency: str
    target_revenue: float | None
    gross_adds_per_day: Uncertain
    daily_churn_rate: Uncertain
    tourist_recharging_per_day: Uncertain
    tourism_factor: Uncertain
    history_tourism: dict[str, tuple[float, float]]  # "YYYY-MM" -> (per day, factor)
    value_per_customer: Uncertain
    brackets: tuple[Bracket, ...]
    salary_dates: tuple[date, ...]
    events: tuple[Event, ...]
    priors: Priors = field(default_factory=Priors)
    draws: int = 1000
    chains: int = 4
    seed: int = 7

    @property
    def month_key(self) -> str:
        return f"{self.month:%Y-%m}"

    @property
    def bracket_value(self) -> float:
        return sum(b.avg_value * b.share for b in self.brackets)


def _uncertain(section: dict, key: str, sd_key: str) -> Uncertain:
    if key not in section:
        raise ConfigError(f"Missing setting: {key}")
    return Uncertain(float(section[key]), float(section.get(sd_key, 0.0)))


def _dates(values) -> tuple[date, ...]:
    try:
        return tuple(sorted(date.fromisoformat(str(v)) for v in values))
    except ValueError as exc:
        raise ConfigError(f"Dates must be written as YYYY-MM-DD: {exc}") from exc


def load(path: Path | str) -> Settings:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    fc, base = raw.get("forecast", {}), raw.get("base", {})
    tour, value = raw.get("tourists", {}), raw.get("value", {})

    try:
        month = date.fromisoformat(str(fc["month"]) + "-01")
    except (KeyError, ValueError) as exc:
        raise ConfigError('forecast.month must look like "2026-12"') from exc

    brackets = tuple(
        Bracket(str(b["name"]), float(b["avg_value"]), float(b["share"]))
        for b in value.get("brackets", [])
    )
    if brackets and abs(sum(b.share for b in brackets) - 1) > 0.01:
        raise ConfigError("Bracket shares must add up to 1 (100%)")

    history_tourism = {
        str(k): (float(v["recharging_per_day"]), float(v["tourism_factor"]))
        for k, v in tour.get("history", {}).items()
    }
    events = tuple(Event(str(e["name"]), _dates(e["dates"])) for e in raw.get("events", []))
    p = raw.get("priors", {})
    run = raw.get("run", {})
    target = fc.get("target_revenue")
    settings = Settings(
        month=month,
        currency=str(fc.get("currency", "")),
        target_revenue=float(target) if target else None,
        gross_adds_per_day=_uncertain(base, "gross_adds_per_day", "gross_adds_uncertainty"),
        daily_churn_rate=_uncertain(base, "daily_churn_rate", "daily_churn_uncertainty"),
        tourist_recharging_per_day=_uncertain(tour, "recharging_per_day", "recharging_uncertainty"),
        tourism_factor=_uncertain(tour, "tourism_factor", "tourism_factor_uncertainty"),
        history_tourism=history_tourism,
        value_per_customer=_uncertain(value, "recharge_value_per_customer", "value_uncertainty"),
        brackets=brackets,
        salary_dates=_dates(raw.get("salary", {}).get("dates", [])),
        events=events,
        priors=Priors(
            year_drift_sd=float(p.get("year_drift_sd", 0.05)),
            season_sd=float(p.get("season_sd", 0.10)),
            effect_sd=float(p.get("effect_sd", 0.30)),
        ),
        draws=int(run.get("draws", 1000)),
        chains=int(run.get("chains", 4)),
        seed=int(run.get("seed", 7)),
    )
    if settings.daily_churn_rate.value >= 0.05:
        raise ConfigError("daily_churn_rate looks like a monthly or % figure; use e.g. 0.0006")
    if not settings.salary_dates:
        raise ConfigError("Add the salary dates (salary.dates)")
    return settings
