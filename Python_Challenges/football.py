# football.py
"""
Fetch simple football data with API-Football.

Before running:
1) Get a free API key from API-Football.
2) Save it as environment variable: 
   export API_FOOTBALL_KEY="YOUR_KEY"
"""

import os
import requests

API_KEY = os.getenv("API_FOOTBALL_KEY")
BASE_URL = "https://api-football-v1.p.rapidapi.com/v3"

HEADERS = {
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
    "X-RapidAPI-Key": API_KEY
}

def get_leagues():
    """Get a list of major football leagues."""
    url = f"{BASE_URL}/leagues"
    response = requests.get(url, headers=HEADERS)
    return response.json()

def print_leagues():
    data = get_leagues()
    for item in data.get("response", []):
        league = item["league"]
        country = item["country"]["name"]
        print(f"{league['name']} ({country})")

if __name__ == "__main__":
    if not API_KEY:
        print("Set API_FOOTBALL_KEY before running")
    else:
        print_leagues()
