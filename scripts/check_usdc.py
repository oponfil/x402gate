"""Check USDC balance of a Solana wallet.

Usage: python scripts/check_usdc.py [wallet_address]
Default wallet: AakQJK2ssdg1PMVDPLj3azyjTh9846RjwiEbTtqpSfXm (E2E test wallet)
Reads SOLANA_RPC_URL from .env automatically.
"""

import json
import os
import sys
import urllib.request

USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


def main():
    wallet = sys.argv[1] if len(sys.argv) > 1 else "AakQJK2ssdg1PMVDPLj3azyjTh9846RjwiEbTtqpSfXm"

    rpc_url = os.environ.get("SOLANA_RPC_URL")
    if not rpc_url:
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("SOLANA_RPC_URL="):
                        rpc_url = line.strip().split("=", 1)[1]
                        break
    if not rpc_url:
        print("ERROR: SOLANA_RPC_URL not set")
        sys.exit(1)

    req = urllib.request.Request(
        rpc_url,
        json.dumps({
            "jsonrpc": "2.0", "id": 1,
            "method": "getTokenAccountsByOwner",
            "params": [
                wallet,
                {"mint": USDC_MINT},
                {"encoding": "jsonParsed"},
            ],
        }).encode(),
        {"Content-Type": "application/json"},
    )
    r = urllib.request.urlopen(req, timeout=10)
    d = json.loads(r.read())
    accounts = d.get("result", {}).get("value", [])
    if not accounts:
        print(f"{wallet}: 0 USDC (no token account)")
    else:
        for acc in accounts:
            info = acc["account"]["data"]["parsed"]["info"]["tokenAmount"]
            print(f"{wallet}: {info['uiAmountString']} USDC")


if __name__ == "__main__":
    main()
