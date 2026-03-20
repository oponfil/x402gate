"""Locus + x402gate Integration Demo — Synthesis Hackathon 2026.

Demonstrates an autonomous AI agent that:
  1. Self-registers with Locus to get a Base wallet + USDC budget
  2. Discovers AI services available on x402gate
  3. Requests an LLM chat completion via x402gate
  4. Receives HTTP 402 Payment Required with on-chain payment options
  5. Signs the payment using its Locus-provisioned private key
  6. Gets the AI-generated response — fully autonomous, no API keys

Requirements:
    pip install httpx x402[evm] pyyaml eth-account

Usage:
    # Against production gateway
    python scripts/locus_demo.py

    # Against local gateway
    GATEWAY_URL=http://localhost:4021 python scripts/locus_demo.py

    # Skip Locus registration (use existing Base private key)
    BASE_PRIVATE_KEY=0x... python scripts/locus_demo.py

Environment:
    GATEWAY_URL          Gateway URL (default: https://x402gate.io)
    BASE_PRIVATE_KEY     Skip Locus registration, use this key directly
    LOCUS_API_KEY        Reuse existing Locus API key (skip registration)
"""

import asyncio
import base64
import logging
import os
import sys
import time

import httpx
from eth_account import Account
from x402 import PaymentRequired, x402Client
from x402.mechanisms.evm.exact import ExactEvmScheme
from x402.mechanisms.evm.signers import EthAccountSigner

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GATEWAY_URL = os.environ.get("GATEWAY_URL", "https://x402gate.io")
LOCUS_API_URL = "https://beta-api.paywithlocus.com/api"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("locus-demo")


# ---------------------------------------------------------------------------
# Step 0: Banner
# ---------------------------------------------------------------------------


def banner():
    """Print a banner describing the demo."""
    print(
        """
╔══════════════════════════════════════════════════════════════╗
║          🔗  Locus + x402gate — Autonomous Agent Demo       ║
║                                                              ║
║  An AI agent that self-provisions a crypto wallet,           ║
║  discovers AI services, and pays for them on-chain.          ║
║                                                              ║
║  Hackathon: The Synthesis (synthesis.md)                     ║
║  Track:     Agents that pay / Open Track / Locus Bounty      ║
╚══════════════════════════════════════════════════════════════╝
"""
    )


# ---------------------------------------------------------------------------
# Step 1: Self-Registration with Locus
# ---------------------------------------------------------------------------


async def register_with_locus(http: httpx.AsyncClient) -> dict:
    """Register a new agent wallet via Locus API.

    Returns dict with keys: apiKey, ownerPrivateKey, ownerAddress,
    claimUrl, walletStatus.
    """
    logger.info("📝 Step 1: Registering with Locus to get a Base wallet...")

    resp = await http.post(
        f"{LOCUS_API_URL}/register",
        json={"name": "x402gate-demo-agent"},
        timeout=15.0,
    )

    if resp.status_code != 200:
        logger.error("Locus registration failed: %d %s", resp.status_code, resp.text)
        sys.exit(1)

    data = resp.json().get("data", resp.json())
    logger.info("✅ Wallet created!")
    logger.info("   Address:   %s", data["ownerAddress"])
    logger.info("   Claim URL: %s", data.get("claimUrl", "N/A"))
    logger.info(
        "   ⚠️  Share the Claim URL with your human to link this wallet"
        " to the Locus dashboard and set spending limits."
    )

    return data


async def wait_for_wallet(http: httpx.AsyncClient, api_key: str) -> None:
    """Poll Locus /status until wallet is deployed."""
    logger.info("   Waiting for wallet deployment...")
    for _attempt in range(30):
        resp = await http.get(
            f"{LOCUS_API_URL}/status",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        )
        status = resp.json().get("data", resp.json()).get("walletStatus", "unknown")
        if status == "deployed":
            logger.info("   ✅ Wallet deployed and ready!")
            return
        await asyncio.sleep(2)
    logger.warning("   ⚠️  Wallet still deploying after 60s, proceeding anyway...")


async def check_locus_balance(http: httpx.AsyncClient, api_key: str) -> float:
    """Check USDC balance via Locus API. Returns balance as float."""
    try:
        resp = await http.get(
            f"{LOCUS_API_URL}/pay/balance",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        )
        data = resp.json().get("data", resp.json())
        balance = float(data.get("balanceUsdc", data.get("balance", 0)))
        return balance
    except Exception:
        return 0.0


async def request_hackathon_credits(http: httpx.AsyncClient, api_key: str) -> None:
    """Request promotional USDC credits for hackathon builders."""
    logger.info("   💰 Requesting hackathon credits from Locus...")
    try:
        resp = await http.post(
            f"{LOCUS_API_URL}/gift-code-requests",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "reason": "Building at The Synthesis hackathon — x402gate.io integration demo",
                "requestedAmountUsdc": 5,
            },
            timeout=10.0,
        )
        if resp.status_code == 200:
            logger.info("   ✅ Credit request submitted! Check Locus dashboard for approval.")
        else:
            logger.info("   ℹ️  Credit request: %s", resp.text[:200])
    except Exception as e:
        logger.info("   ℹ️  Could not request credits: %s", e)


# ---------------------------------------------------------------------------
# Step 2: Service Discovery
# ---------------------------------------------------------------------------


async def discover_services(http: httpx.AsyncClient) -> dict:
    """Query x402gate's machine-readable service discovery endpoint."""
    logger.info("\n🔍 Step 2: Discovering AI services on x402gate...")

    resp = await http.get(
        f"{GATEWAY_URL}/v1/providers",
        timeout=10.0,
    )

    if resp.status_code != 200:
        logger.error("Service discovery failed: %d", resp.status_code)
        sys.exit(1)

    manifest = resp.json()
    providers = manifest.get("providers", {})

    # Handle both list (GET /) and dict (GET /v1/providers) formats
    if isinstance(providers, list):
        logger.info("   Found %d providers: %s", len(providers), ", ".join(providers))
    else:
        logger.info("   Found %d providers:", len(providers))
        for name, info in providers.items():
            desc = info.get("description", "")
            logger.info("   • %-16s %s", name, desc)

    return manifest


# ---------------------------------------------------------------------------
# Step 3: Request AI Generation (get 402)
# ---------------------------------------------------------------------------


async def request_generation(http: httpx.AsyncClient) -> tuple[dict, str]:
    """Send a chat completion request to x402gate → get 402 with price.

    Returns (payment_data, endpoint_url).
    """
    logger.info("\n🤖 Step 3: Requesting LLM chat completion via x402gate...")

    endpoint = f"{GATEWAY_URL}/v1/openrouter/chat/completions"
    body = {
        "model": "google/gemini-2.5-flash",
        "messages": [
            {
                "role": "user",
                "content": (
                    "You are an AI agent participating in The Synthesis hackathon. "
                    "In one sentence, explain why autonomous AI agents need "
                    "decentralized payment infrastructure like x402."
                ),
            },
        ],
        "max_tokens": 256,
    }

    resp = await http.post(endpoint, json=body, timeout=15.0)

    if resp.status_code != 402:
        logger.error("Expected 402 Payment Required, got %d: %s", resp.status_code, resp.text)
        sys.exit(1)

    payment_data = resp.json()
    accepts = payment_data.get("accepts", [])

    logger.info("   💳 Payment Required!")
    for opt in accepts:
        network = opt.get("network", "?")
        amount = int(opt.get("amount", 0)) / 1e6
        logger.info("      Network: %s — $%.6f USDC", network, amount)

    return payment_data, endpoint


# ---------------------------------------------------------------------------
# Step 4: Sign Payment & Get Result
# ---------------------------------------------------------------------------


async def sign_and_pay(
    http: httpx.AsyncClient,
    payment_data: dict,
    endpoint: str,
    private_key: str,
) -> dict:
    """Sign the x402 payment with the agent's private key and get the result.

    Returns the AI-generated response.
    """
    logger.info("\n✍️  Step 4: Signing payment with agent's Base wallet...")

    # Initialize x402 client with the agent's key
    account = Account.from_key(private_key)
    signer = EthAccountSigner(account)
    x402_client = x402Client()
    x402_client.register("eip155:8453", ExactEvmScheme(signer))

    logger.info("   Signer address: %s", account.address)

    # Parse 402 response and filter to Base network only
    payment_required = PaymentRequired.model_validate(payment_data)
    base_accepts = [
        a for a in payment_required.accepts if "eip155:8453" in getattr(a, "network", "")
    ]
    if not base_accepts:
        logger.error("   ❌ No Base (eip155:8453) payment option available!")
        sys.exit(1)
    payment_required.accepts = base_accepts

    # Create and encode payment payload
    t0 = time.monotonic()
    payment_payload = await x402_client.create_payment_payload(payment_required)
    signature = base64.b64encode(payment_payload.model_dump_json(by_alias=True).encode()).decode()
    sign_time = time.monotonic() - t0
    logger.info("   ✅ Payment signed in %.2fs", sign_time)

    # Retry the request with payment
    logger.info("   📡 Sending paid request to x402gate...")
    body = {
        "model": "google/gemini-2.5-flash",
        "messages": [
            {
                "role": "user",
                "content": (
                    "You are an AI agent participating in The Synthesis hackathon. "
                    "In one sentence, explain why autonomous AI agents need "
                    "decentralized payment infrastructure like x402."
                ),
            },
        ],
        "max_tokens": 256,
    }

    t0 = time.monotonic()
    resp = await http.post(
        endpoint,
        json=body,
        headers={"PAYMENT-SIGNATURE": signature},
        timeout=60.0,
    )
    request_time = time.monotonic() - t0

    if resp.status_code != 200:
        logger.error("   ❌ Payment failed: %d %s", resp.status_code, resp.text)
        sys.exit(1)

    result = resp.json()
    data = result.get("data", result)
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = data.get("usage", {})
    model = data.get("model", "?")

    logger.info("   ✅ Response received in %.2fs!", request_time)
    logger.info("\n" + "=" * 60)
    logger.info("📝 AI Response (model: %s):", model)
    logger.info("   %s", content)
    logger.info("=" * 60)
    logger.info(
        "   Tokens: prompt=%s, completion=%s, total=%s",
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        usage.get("total_tokens"),
    )

    # Extract transaction hash if available
    tx_hash = resp.headers.get("x-payment-tx")
    if tx_hash:
        logger.info("\n🔗 On-chain proof:")
        logger.info("   Tx Hash: %s", tx_hash)
        logger.info("   BaseScan: https://basescan.org/tx/%s", tx_hash)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    """Run the full autonomous agent demo."""
    banner()
    t_start = time.monotonic()

    async with httpx.AsyncClient() as http:
        # Determine wallet source
        private_key = os.environ.get("BASE_PRIVATE_KEY")
        locus_api_key = os.environ.get("LOCUS_API_KEY")

        if private_key:
            # Direct mode: use provided key (skip Locus)
            account = Account.from_key(private_key)
            logger.info("🔑 Using provided Base wallet: %s", account.address)
            locus_api_key = None
        else:
            # Full Locus flow
            locus_data = await register_with_locus(http)
            private_key = locus_data["ownerPrivateKey"]
            locus_api_key = locus_data.get("apiKey")

            # Wait for wallet deployment
            if locus_api_key:
                await wait_for_wallet(http, locus_api_key)

                # Check balance
                balance = await check_locus_balance(http, locus_api_key)
                logger.info("   💰 Current USDC balance: $%.6f", balance)

                if balance < 0.01:
                    logger.warning("   ⚠️  Insufficient balance! The wallet needs USDC on Base.")
                    logger.info("   Options:")
                    logger.info("     1. Send USDC to %s on Base", locus_data["ownerAddress"])
                    logger.info("     2. Request hackathon credits (submitting now...)")
                    await request_hackathon_credits(http, locus_api_key)
                    logger.info(
                        "\n   ℹ️  After funding, re-run this script with:\n"
                        "      LOCUS_API_KEY=%s BASE_PRIVATE_KEY=%s python scripts/locus_demo.py",
                        locus_api_key,
                        private_key,
                    )
                    return

        # Step 2: Discover services
        await discover_services(http)

        # Step 3: Request generation (get 402 price)
        payment_data, endpoint = await request_generation(http)

        # Step 4: Sign and pay
        await sign_and_pay(http, payment_data, endpoint, private_key)

    elapsed = time.monotonic() - t_start
    print(
        f"""
╔══════════════════════════════════════════════════════════════╗
║  ✅  Demo Complete!                                         ║
║                                                              ║
║  Total time: {elapsed:.1f}s                                         ║
║                                                              ║
║  What just happened:                                         ║
║  1. Agent self-registered a Base wallet via Locus             ║
║  2. Discovered AI services on x402gate (JSON manifest)       ║
║  3. Requested LLM completion → got 402 with USDC price       ║
║  4. Signed on-chain payment → received AI response            ║
║                                                              ║
║  No API keys. No subscriptions. No intermediaries.            ║
║  Just crypto + AI, working together autonomously.             ║
║                                                              ║
║  🌐 Gateway:   {GATEWAY_URL:<44s} ║
║  📦 GitHub:    https://github.com/oponfil/x402gate            ║
║  🏆 Hackathon: https://synthesis.md                           ║
╚══════════════════════════════════════════════════════════════╝
"""
    )


if __name__ == "__main__":
    asyncio.run(main())
