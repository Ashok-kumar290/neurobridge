import ccxt
import os
from dotenv import load_dotenv

load_dotenv("/media/seyominaoto/x/neurobridge/.env")
exchange = ccxt.binance({
    "apiKey": os.getenv("BINANCE_API_KEY"),
    "secret": os.getenv("BINANCE_SECRET"),
})

try:
    # Try fetching balance in sandbox mode
    exchange.set_sandbox_mode(True)
    balance_sandbox = exchange.fetch_balance()
    print("Sandbox Mode: Success")
except Exception as e:
    print(f"Sandbox Mode Failed: {e}")

try:
    # Try fetching balance in live mode
    exchange.set_sandbox_mode(False)
    balance_live = exchange.fetch_balance()
    print("Live Mode: Success")
except Exception as e:
    print(f"Live Mode Failed: {e}")
