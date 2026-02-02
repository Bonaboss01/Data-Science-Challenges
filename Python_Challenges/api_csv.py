import requests
import pandas as pd


def fetch_api_data(url, params=None, max_pages=5, page_param="page"):
    """
    Fetch JSON data from an API that returns a list of records.
    Supports simple pagination using a page query param.
    """
    all_rows = []

    params = params or {}

    for page in range(1, max_pages + 1):
        params[page_param] = page

        response = requests.get(url, params=params, timeout=20)

        if response.status_code != 200:
            print(f"Request failed on page {page}: {response.status_code}")
            break

        data = response.json()

        # If API returns dict with a key like "results"
        if isinstance(data, dict) and "results" in data:
            rows = data["results"]
        else:
            rows = data

        # Stop if no more records
        if not rows:
            print("No more data found. Stopping pagination.")
            break

        all_rows.extend(rows)
        print(f"Fetched page {page}: {len(rows)} records")

    return all_rows


def save_to_csv(rows, output_file):
    if not rows:
        print("No data to save.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} rows to {output_file}")


if __name__ == "__main__":
    # Example public API
    api_url = "https://jsonplaceholder.typicode.com/posts"

    rows = fetch_api_data(api_url, max_pages=1)  # no pagination needed for this API
    save_to_csv(rows, "api_data.csv")

