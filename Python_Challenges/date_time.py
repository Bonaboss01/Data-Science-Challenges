# Investigating with different timedate packages
import datetime as dt
time_of_year = dt.datetime(year, month, day, hour, minutes, seconds)

time_to_format = dt.datetime(2020, 1, 25, 15, 25, 13)

print(time_to_format.strftime("%Y/%m/%d %H:%M:%S"))

time_str = "12:30 2019-05-19"

date_object = dt.datetime.strptime(time_str, "%H:%M %Y-%m-%d")

import pytz

my_timezone_name = 'America/New_York' 
time_now = dt.datetime.now(pytz.timezone(my_timezone_name))

time_now = dt.datetime.now()

# Birthday for someone born on 19th of May
time_birthday = dt.datetime(time_now.year, 5, 19)
time_to_birthday = time_birthday - time_now
print(time_to_birthday)


"""
Reusable date and time utilities for analytics and ML pipelines.
Designed for consistency across data ingestion, modeling, and reporting.
"""

from datetime import datetime, timedelta
from typing import Tuple


def get_utc_now() -> datetime:
    """
    Returns the current UTC datetime.
    """
    return datetime.utcnow()


def parse_date(date_str: str, fmt: str = "%Y-%m-%d") -> datetime:
    """
    Parse a date string into a datetime object.

    Args:
        date_str (str): Date string to parse
        fmt (str): Datetime format

    Returns:
        datetime
    """
    return datetime.strptime(date_str, fmt)


def date_range(
    start_date: datetime,
    end_date: datetime
) -> Tuple[datetime, ...]:
    """
    Generate an inclusive date range between two dates.

    Args:
        start_date (datetime)
        end_date (datetime)

    Returns:
        tuple of datetime objects
    """
    delta_days = (end_date - start_date).days
    return tuple(start_date + timedelta(days=i) for i in range(delta_days + 1))


def is_weekend(date: datetime) -> bool:
    """
    Check if a date falls on a weekend.
    """


from datetime import datetime

def log_run():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("run_log.txt", "a") as f:
        f.write(f"Script run at: {now}\n")

if __name__ == "__main__":
    log_run()

