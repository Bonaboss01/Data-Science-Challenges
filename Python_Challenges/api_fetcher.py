import requests
import json

def fetch_data(url, output_file):
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        with open(output_file, "w") as f:
            json.dump(data, f, indent=4)
        print("Data saved to", output_file)
    else:
        print("Failed to fetch data:", response.status_code)


if __name__ == "__main__":
    api_url = "https://jsonplaceholder.typicode.com/posts"
    fetch_data(api_url, "posts.json")
