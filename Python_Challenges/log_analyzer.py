import os
import re
import json
import logging
from argparse import ArgumentParser
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# Example log pattern:
# 2026-01-31 12:45:22 | auth_service | ERROR | Invalid token

LOG_PATTERN = re.compile(
    r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+\|\s+"
    r"(?P<module>[\w_]+)\s+\|\s+"
    r"(?P<level>ERROR|WARNING|INFO)\s+\|\s+"
    r"(?P<message>.*)"
)


def parse_log_line(line):
    match = LOG_PATTERN.search(line)
    if match:
        return match.groupdict()
    return None


def analyze_log(file_path, output_prefix="log_report"):

    if not os.path.exists(file_path):
        logging.error(f"Log file not found: {file_path}")
        return

    logging.info("Starting log analysis...")

    records = []
    level_counter = Counter()
    module_counter = Counter()
    error_messages = Counter()

    with open(file_path, "r") as file:
        for line in file:
            parsed = parse_log_line(line)

            if not parsed:
                continue

            timestamp = pd.to_datetime(parsed["timestamp"], errors="coerce")
            module = parsed["module"]
            level = parsed["level"]
            message = parsed["message"]

            records.append({
                "timestamp": timestamp,
                "module": module,
                "level": level,
                "message": message
            })

            level_counter[level] += 1
            module_counter[module] += 1

            if level == "ERROR":
                error_messages[message] += 1

    if not records:
        logging.warning("No valid log entries found.")
        return

    df = pd.DataFrame(records)

    # ---------------------------
    # Time-based summary
    # ---------------------------
    df["date"] = df["timestamp"].dt.date

    daily_summary = (
        df.groupby(["date", "level"])
          .size()
          .unstack(fill_value=0)
          .reset_index()
    )

    # ---------------------------
    # Save detailed logs
    # ---------------------------
    df.to_csv(f"{output_prefix}_detailed.csv", index=False)

    # ---------------------------
    # Save daily summary
    # ---------------------------
    daily_summary.to_csv(f"{output_prefix}_daily_summary.csv", index=False)

    # ---------------------------
    # JSON Report
    # ---------------------------
    report = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "total_entries": len(df),
        "levels_count": dict(level_counter),
        "most_active_modules": module_counter.most_common(10),
        "most_common_errors": error_messages.most_common(10)
    }

    with open(f"{output_prefix}_summary.json", "w") as f:
        json.dump(report, f, indent=2)

    # ---------------------------
    # Console Output
    # ---------------------------
    logging.info("Analysis complete!")
    logging.info(f"Total entries: {len(df)}")
    logging.info(f"Level counts: {dict(level_counter)}")
    logging.info("Top modules:")
    for mod, count in module_counter.most_common(5):
        logging.info(f"  {mod}: {count}")

    logging.info("Top errors:")
    for msg, count in error_messages.most_common(5):
        logging.info(f"  {count}x - {msg}")


if __name__ == "__main__":

    parser = ArgumentParser(description="Advanced Log Analysis Tool")

    parser.add_argument(
        "--logfile",
        required=True,
        help="Path to log file"
    )

    parser.add_argument(
        "--output",
        default="log_report",
        help="Output file prefix"
    )

    args = parser.parse_args()

    analyze_log(
        file_path=args.logfile,
        output_prefix=args.output
    )
