"""Fast tests: settings, calendar, base roll-forward, input draws, report."""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rechargefc.config import ConfigError, Uncertain, load
from rechargefc.data import (
    SALARY_BINS,
    calendar_features,
    check_salary_coverage,
    load_history,
    month_days,
    salary_bin,
    tourist_series,
)
from rechargefc.demo import SETTINGS, generate
from rechargefc.report import write_report
from rechargefc.simulate import _draw, roll_base


@pytest.fixture(scope="session")
def demo(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    truth = generate(out)
    return out, truth


@pytest.fixture
def settings(demo):
    return load(demo[0] / "settings.toml")


def write_settings(tmp_path: Path, old: str, new: str) -> Path:
    p = tmp_path / "s.toml"
    assert old in SETTINGS
    p.write_text(SETTINGS.replace(old, new))
    return p


def test_settings_load(settings):
    assert settings.month == date(2026, 12, 1)
    assert settings.bracket_value == pytest.approx(35.75)
    assert settings.history_tourism["2025-12"] == (2000, 1.25)
    assert [e.name for e in settings.events] == ["National Day", "New Year's Eve"]


def test_bracket_shares_must_sum_to_one(tmp_path):
    p = write_settings(tmp_path, "share = 0.13", "share = 0.30")
    with pytest.raises(ConfigError, match="shares"):
        load(p)


def test_churn_given_as_monthly_is_rejected(tmp_path):
    p = write_settings(tmp_path, "daily_churn_rate = 0.0006", "daily_churn_rate = 2.5")
    with pytest.raises(ConfigError, match="churn"):
        load(p)


def test_uncertain_validation():
    with pytest.raises(ConfigError):
        Uncertain(-1, 0.1)
    with pytest.raises(ConfigError):
        Uncertain(1, 1.5)


def test_salary_bins():
    pay = (date(2026, 11, 25), date(2026, 12, 25), date(2027, 1, 1))
    assert salary_bin(date(2026, 12, 25), pay) == "day0"
    assert salary_bin(date(2026, 12, 26), pay) == "day1"
    assert salary_bin(date(2026, 12, 29), pay) == "day3_4"
    assert salary_bin(date(2026, 12, 23), pay) == "pre"
    assert salary_bin(date(2026, 12, 10), pay) == "none"
    # 30 Dec: 5 days after 25 Dec, 2 days before 1 Jan
    assert salary_bin(date(2026, 12, 30), pay) == "pre"


def test_salary_coverage_requires_surrounding_dates():
    days = month_days(date(2026, 12, 1))
    with pytest.raises(ConfigError, match="before"):
        check_salary_coverage(days, (date(2026, 12, 25), date(2027, 1, 1)))
    with pytest.raises(ConfigError, match="after"):
        check_salary_coverage(days, (date(2026, 11, 25), date(2026, 12, 25)))
    check_salary_coverage(days, (date(2026, 11, 25), date(2026, 12, 25), date(2027, 1, 1)))


def test_calendar_features(settings):
    days = [date(2025, 12, 2), date(2026, 12, 31)]
    f = calendar_features(days, settings)
    assert f["years_before"].tolist() == [1, 0]
    assert f["event:National Day"].tolist() == [1, 0]
    assert f["event:New Year's Eve"].tolist() == [0, 1]
    assert SALARY_BINS[f["salary_bin"][0]] == "day1"


def test_tourist_series(settings):
    days = [date(2025, 12, 5), date(2026, 12, 5)]
    assert tourist_series(days, settings).tolist() == [2000 * 1.25, 2200 * 1.30]
    hist = pd.DataFrame({"date": days, "tourist_customers": [1234, np.nan]})
    assert tourist_series(days, settings, hist).tolist() == [1234, 2200 * 1.30]
    with pytest.raises(ConfigError, match="tourist"):
        tourist_series([date(2026, 3, 1)], settings)


def test_roll_base_matches_day_by_day_loop():
    adds, churn = np.array([1300.0, 0.0]), np.array([0.0006, 0.001])
    path = roll_base(2_000_000, adds, churn, 31)
    for k in range(2):
        b = 2_000_000.0
        for t in range(31):
            b = b * (1 - churn[k]) + adds[k]
            assert path[k, t] == pytest.approx(b)
    no_churn = roll_base(100, np.array([10.0]), np.array([0.0]), 5)
    assert no_churn[0].tolist() == [110, 120, 130, 140, 150]


def test_draw_keeps_mean_and_spread():
    rng = np.random.default_rng(0)
    x = _draw(Uncertain(1300, 0.10), 200_000, rng)
    assert x.mean() == pytest.approx(1300, rel=0.005)
    assert x.std() / x.mean() == pytest.approx(0.10, rel=0.03)
    assert (_draw(Uncertain(5, 0), 3, rng) == 5).all()


def test_history_validation(tmp_path, demo):
    h = load_history(demo[0] / "history.csv")
    assert len(h) == 62 and h["date"].is_monotonic_increasing
    bad = pd.read_csv(demo[0] / "history.csv")
    bad.loc[3, "active_base"] = 0
    bad.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ConfigError, match="positive"):
        load_history(tmp_path / "bad.csv")
    dup = pd.concat([pd.read_csv(demo[0] / "history.csv")] * 2)
    dup.to_csv(tmp_path / "dup.csv", index=False)
    with pytest.raises(ConfigError, match="same date"):
        load_history(tmp_path / "dup.csv")
    bad.drop(columns="active_base").to_csv(tmp_path / "miss.csv", index=False)
    with pytest.raises(ConfigError, match="active_base"):
        load_history(tmp_path / "miss.csv")


def test_demo_is_reproducible(tmp_path, demo):
    again = generate(tmp_path)
    assert again["december_2026_revenue"] == demo[1]["december_2026_revenue"]


def test_report_escapes_script_tags(tmp_path):
    summary = {"note": "</script><b>x</b>"}
    write_report(summary, {"x": [1]}, tmp_path / "r.html")
    html = (tmp_path / "r.html").read_text()
    assert "__DATA__" not in html
    assert "<\\/script><b>x<\\/b>" in html


def test_month_days():
    assert len(month_days(date(2026, 2, 1))) == 28
    assert month_days(date(2026, 12, 1))[-1] == date(2026, 12, 31)
    dec = month_days(date(2026, 12, 1))
    assert all(b - a == timedelta(days=1) for a, b in zip(dec[:-1], dec[1:], strict=True))
