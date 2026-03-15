"""Check Solana facilitator wallet SOL balance."""

import json
import os
import sys
import urllib.request

# Facilitator public key (this is a public address, safe to commit — NOT a secret)
FACILITATOR_PUBKEY = "3Mco8Z9NakFiJv88XHpJ6nXeC9BogGuWYqADu3ouu1L4"


def main():
    rpc_url = os.environ.get("SOLANA_RPC_URL")
    if not rpc_url:
        # Try loading from .env
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
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "getBalance", "params": [FACILITATOR_PUBKEY]}).encode(),
        {"Content-Type": "application/json"},
    )
    r = urllib.request.urlopen(req, timeout=10)
    d = json.loads(r.read())
    sol = d["result"]["value"] / 1e9
    print(f"Facilitator {FACILITATOR_PUBKEY}: {sol:.6f} SOL")
    if sol < 0.01:
        print("⚠️  LOW BALANCE — likely cause of simulation failures!")
    else:
        print("✅ Balance OK")


if __name__ == "__main__":
    main()
