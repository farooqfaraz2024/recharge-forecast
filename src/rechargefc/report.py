"""Write the interactive HTML report (one self-contained file plus plotly.js from a CDN)."""

from __future__ import annotations

import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("report_template.html")


def write_report(summary: dict, chart: dict, path: Path) -> None:
    payload = json.dumps({"summary": summary, "chart": chart}, default=str)
    payload = payload.replace("</", "<\\/")  # never close the script tag early
    html = TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", payload)
    path.write_text(html, encoding="utf-8")
