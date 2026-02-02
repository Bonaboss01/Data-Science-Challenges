import os
import re
import json
import logging
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime
from typing import Optional, Dict, Any, List

import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Structured pattern:
# 2026-01-31 12:45:22 | auth_service | ERROR | Invalid token
STRUCTURED_PATTERN = re.compile(
    r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+\|\s+"
    r"(?P<module>[\w\-.]+)\s+\|\s+"
    r"(?P<level>ERROR|WARNING|INFO|DEBUG)\s+\|\s+"
    r"(?P<message>.*)"
)

# Unstructured fallback examples:
# [2026-01-31 12:45:22] ERROR - Something happened
# 2026-01-31T12:45:22Z WARNING Something happened
FALLBACK_TIMESTAMP = re.compile(
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})(?:Z)?"
)
FALLBACK_LEVEL = re.compile(r"\b(ERROR|WARNING|INFO|DEBUG)\b")


def parse_line(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None

    m = STRUCTURED_PATTERN.search(line)
    if m:
        d = m.groupdict()
        d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce", utc=True)
        return d

    # fallback: try to find timestamp + level anywhere
    ts = FALLBACK_TIMESTAMP.search(line)
    lvl = FALLBACK_LEVEL.search(line)

    if ts and lvl:
        timestamp = pd.to_datetime(ts.group("timestamp"), errors="coerce", utc=True)
        level = lvl.group(1)
        return {
            "timestamp": timestamp,
            "module": "unknown",
            "level": level,
            "message": line
        }

    return None


def within_window(ts: pd.Timestamp, since: Optional[pd.Timestamp], until: Optional[pd.Timestamp]) -> bool:
    if pd.isna(ts):
        return False
    if since is not None and ts < since:
        return False
    if until is not None and ts > until:
        return False
    return True


def analyze_log(
    logfile: str,
    output_prefix: str = "log_report",
    since: Optional[str] = None,
    until: Optional[str] = None,
    grep: Optional[str] = None,
    top_n: int = 10,
    summary_only: bool = False,
    make_plots: bool = False
) -> None:

    if not os.path.exists(logfile):
        logging.error(f"File not found: {logfile}")
        return

    since_ts = pd.to_datetime(since, utc=True) if since else None
    until_ts = pd.to_datetime(until, utc=True) if until else None
    grep_re = re.compile(grep, re.IGNORECASE) if grep else None

    level_counter = Counter()
    module_counter = Counter()
    error_counter = Counter()
    warning_counter = Counter()
    daily_counts = Counter()

    records: List[Dict[str, Any]] = []

    parsed_lines = 0
    kept_lines = 0

    logging.info("Reading log file...")

    with open(logfile, "r", errors="ignore") as f:
        for line in f:
            parsed = parse_line(line)
            if not parsed:
                continue

            parsed_lines += 1

            ts = parsed["timestamp"]
            if not within_window(ts, since_ts, until_ts):
                continue

            if grep_re and not grep_re.search(parsed["message"]):
                continue

            kept_lines += 1

            level = parsed["level"]
            module = parsed["module"]
            msg = parsed["message"]

            level_counter[level] += 1
            module_counter[module] += 1

            date_key = None
            if not pd.isna(ts):
                date_key = ts.date()
                daily_counts[(date_key, level)] += 1

            if level == "ERROR":
                error_counter[msg] += 1
            elif level == "WARNING":
                warning_counter[msg] += 1

            if not summary_only:
                records.append({
                    "timestamp": ts,
                    "module": module,
                    "level": level,
                    "message": msg
                })

    if kept_lines == 0:
        logging.warning("No matching log entries found (after parsing/filters).")
        return

    logging.info(f"Parsed entries: {parsed_lines} | Kept after filters: {kept_lines}")

    # ---------------------------
    # Build summary tables
    # ---------------------------
    top_modules = module_counter.most_common(top_n)
    top_errors = error_counter.most_common(top_n)
    top_warnings = warning_counter.most_common(top_n)

    # daily summary dataframe
    daily_rows = []
    for (d, lvl), cnt in daily_counts.items():
        daily_rows.append({"date": d, "level": lvl, "count": cnt})
    daily_df = pd.DataFrame(daily_rows)
    if not daily_df.empty:
        daily_pivot = daily_df.pivot_table(index="date", columns="level", values="count", fill_value=0).reset_index()
    else:
        daily_pivot = pd.DataFrame()

    # ---------------------------
    # Save outputs
    # ---------------------------
    report = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "logfile": logfile,
        "filters": {
            "since": since,
            "until": until,
            "grep": grep,
            "top_n": top_n,
            "summary_only": summary_only
        },
        "parsed_entries": parsed_lines,
        "kept_entries": kept_lines,
        "levels_count": dict(level_counter),
        "top_modules": top_modules,
        "top_errors": top_errors,
        "top_warnings": top_warnings
    }

    with open(f"{output_prefix}_summary.json", "w") as f:
        json.dump(report, f, indent=2)
    logging.info(f"Saved JSON summary: {output_prefix}_summary.json")

    if not daily_pivot.empty:
        daily_pivot.to_csv(f"{output_prefix}_daily_summary.csv", index=False)
        logging.info(f"Saved daily summary: {output_prefix}_daily_summary.csv")

    if not summary_only and records:
        pd.DataFrame(records).to_csv(f"{output_prefix}_detailed.csv", index=False)
        logging.info(f"Saved detailed logs: {output_prefix}_detailed.csv")

    # ---------------------------
    # Optional plots
    # ---------------------------
    if make_plots and not daily_pivot.empty:
        try:
            import matplotlib.pyplot as plt  # only import if needed
            daily_pivot["date"] = pd.to_datetime(daily_pivot["date"])

            # Plot ERROR and WARNING if present
            plt.figure()
            for col in ["ERROR", "WARNING"]:
                if col in daily_pivot.columns:
                    plt.plot(daily_pivot["date"], daily_pivot[col], label=col)

            plt.xlabel("Date")
            plt.ylabel("Count")
            plt.title("Log Trend (Errors/Warnings)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_prefix}_trend.png", dpi=200)
            plt.close()
            logging.info(f"Saved plot: {output_prefix}_trend.png")
        except Exception as e:
            logging.warning(f"Plot generation failed: {e}")

    # ---------------------------
    # Console summary
    # ---------------------------
    logging.info("=== SUMMARY ===")
    logging.info(f"Levels: {dict(level_counter)}")
    logging.info("Top modules:")
    for mod, cnt in top_modules[:5]:
        logging.info(f"  {mod}: {cnt}")

    logging.info("Top errors:")
    for msg, cnt in top_errors[:5]:
        logging.info(f"  {cnt}x - {msg[:120]}")

    logging.info("Top warnings:")
    for msg, cnt in top_warnings[:5]:
        logging.info(f"  {cnt}x - {msg[:120]}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Ultra Log Analyzer (filters + reports + plots)")

    parser.add_argument("--logfile", required=True, help="Path to log file")
    parser.add_argument("--output", default="log_report", help="Output prefix for saved files")
    parser.add_argument("--since", default=None, help="Filter logs since this timestamp (e.g. '2026-01-01 00:00:00')")
    parser.add_argument("--until", default=None, help="Filter logs until this timestamp")
    parser.add_argument("--grep", default=None, help="Regex/keyword filter for messages (case-insensitive)")
    parser.add_argument("--top-n", type=int, default=10, help="Top N results for modules/errors/warnings")
    parser.add_argument("--summary-only", action="store_true", help="Do not save detailed CSV (faster)")
    parser.add_argument("--plots", action="store_true", help="Generate trend plot PNG")

    args = parser.parse_args()

    analyze_log(
        logfile=args.logfile,
        output_prefix=args.output,
        since=args.since,
        until=args.until,
        grep=args.grep,
        top_n=args.top_n,
        summary_only=args.summary_only,
        make_plots=args.plots
    )
