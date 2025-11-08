from dotenv import load_dotenv
import requests
import os

load_dotenv()

api_key = os.getenv("COIN_GECKO")
# print(f"Using CoinGecko API key: {api_key}")


# refreshs in about 1-2 minutes 
url = "https://api.coingecko.com/api/v3/simple/price"


params = {
    "ids": "ethereum",
    "vs_currencies": "usd",
}



res = requests.get(url, params=params)

if res.status_code == 200:
    data = res.json()
    print("Current Prices:")
    for coin, price_info in data.items():
        print(f"{coin.capitalize()}: ${price_info['usd']}")
else:
    print(f"Error fetching prices: {res.status_code} - {res.text}")