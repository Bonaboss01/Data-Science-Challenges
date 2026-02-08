"""
sleeping.py

Simple script to calculate recommended sleep duration.
"""

from datetime import datetime, timedelta


def recommended_sleep(hours=8):
    """Return recommended wake-up time based on current time."""
    now = datetime.now()
    wake_time = now + timedelta(hours=hours)
    return wake_time.strftime("%Y-%m-%d %H:%M")


if __name__ == "__main__":
    print("If you sleep now, you should wake up at:")
    print(recommended_sleep())
