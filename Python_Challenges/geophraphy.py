"""
geography.py

Small geography utilities:
- Country -> continent lookup
- Simple text normalization
- Quick summaries (counts by continent)

Usage:
    python geography.py Nigeria
    python geography.py "United Kingdom"
"""

from __future__ import annotations

import sys


# Minimal, editable mapping (expand anytime)
COUNTRY_TO_CONTINENT = {
    "nigeria": "Africa",
    "ghana": "Africa",
    "kenya": "Africa",
    "south africa": "Africa",
    "egypt": "Africa",
    "united kingdom": "Europe",
    "uk": "Europe",
    "ireland": "Europe",
    "france": "Europe",
    "germany": "Europe",
    "spain": "Europe",
    "italy": "Europe",
    "united states": "North America",
    "usa": "North America",
    "canada": "North America",
    "mexico": "North America",
    "brazil": "South America",
    "argentina": "South America",
    "chile": "South America",
    "china": "Asia",
    "india": "Asia",
    "japan": "Asia",
    "south korea": "Asia",
    "australia": "Oceania",
    "new zealand": "Oceania",
}


def normalize_country_name(name: str) -> str:
    """Normalize country input for dictionary lookup."""
    return name.strip().lower()


def country_to_continent(country: str) -> str:
    """Return continent for a given country, or 'Unknown' if not found."""
    key = normalize_country_name(country)
    return COUNTRY_TO_CONTINENT.get(key, "Unknown")


def count_by_continent(countries: list[str]) -> dict[str, int]:
    """Count how many countries fall into each continent."""
    counts: dict[str, int] = {}
    for c in countries:
        cont = country_to_continent(c)
        counts[cont] = counts.get(cont, 0) + 1
    return counts


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python geography.py <country name>")
        print("Example: python geography.py Nigeria")
        sys.exit(0)

    country = " ".join(sys.argv[1:])
    continent = country_to_continent(country)
    print(f"{country} -> {continent}")


if __name__ == "__main__":
    main()
