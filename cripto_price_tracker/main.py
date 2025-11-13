import requests
import time
from datetime import datetime

def fetch_data(retries=3, delay=5):
    """
    Fetch Bitcoin and Ethereum prices from CoinGecko API.
    Retries automatically if there's a network/API error.
    """
    url = 'https://api.coingecko.com/api/v3/simple/price'
    params = {
        'ids': 'bitcoin,ethereum',
        'vs_currencies': 'usd'
    }

    attempt = 0
    while attempt < retries:
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()  # Raise error for HTTP issues
            data = response.json()

            # Check if expected data exists
            if 'bitcoin' not in data or 'ethereum' not in data:
                raise ValueError(f"Unexpected API response: {data}")

            btc_price = data['bitcoin']['usd']
            eth_price = data['ethereum']['usd']
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            print(f"\n[{timestamp}]")
            print(f"Bitcoin:  ${btc_price:,.2f}")
            print(f"Ethereum: ${eth_price:,.2f}")

            return True  # Success

        except (requests.exceptions.RequestException, ValueError) as e:
            attempt += 1
            wait = delay * attempt  # exponential backoff
            print(f"\n[!] Attempt {attempt}/{retries} failed: {e}")
            print(f"⏳ Retrying in {wait} seconds...\n")
            time.sleep(wait)

    print("\nAll retries failed. Skipping this cycle.\n")
    return False


while True:
    fetch_data()
    time.sleep(10)
