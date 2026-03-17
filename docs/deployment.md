# Deployment

x402gate can be deployed using Railway (recommended), Docker, or a traditional VPS.

> [!IMPORTANT]
> **Single instance only.** x402gate currently stores pending payment settlements in process memory (`asyncio.Task`). Running multiple instances behind a load balancer may cause lost settlements if an instance crashes mid-settlement. Multi-instance support requires adding a persistent task queue (Redis, PostgreSQL, etc.) — planned for a future release.

## Railway (Recommended)

[Railway](https://railway.app) provides the simplest deployment experience.

### Steps

1. Push your code to GitHub
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**
3. Select your `x402gate` repository
4. Add environment variables in the Railway dashboard:
   - `WAVESPEED_API_KEY` — your WaveSpeed API key
   - `BASE_PAY_TO_ADDRESS` — your EVM wallet address for receiving USDC
5. Railway auto-detects the `Dockerfile` and deploys

### Custom Domain

1. Go to **Settings** → **Domains** in Railway
2. Add your domain (e.g., `x402gate.io`)
3. Configure DNS as instructed by Railway

### Environment Variables

Set these in the Railway dashboard under **Variables**:

| Variable | Required | Description |
|---|---|---|
| `WAVESPEED_API_KEY` | Yes | WaveSpeed AI API key |
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key |
| `TUNGSTEN_JWT_TOKEN` | Yes | Tungsten JWT cookie |
| `TUNGSTEN_CF_CLEARANCE` | Yes | Tungsten Cloudflare clearance cookie |
| `CLOUDCONVERT_API_KEY` | Yes | CloudConvert API key |
| `RAPIDAPI_KEY` | Yes | RapidAPI key for SocialDownload |
| `BASE_PAY_TO_ADDRESS` | Yes | EVM address for USDC payments |
| `BASE_FACILITATOR_PRIVATE_KEY` | Yes | Private key for on-chain settlement (EVM) |
| `SOLANA_PAY_TO_ADDRESS` | Solana | Solana wallet for USDC payments |
| `SOLANA_FACILITATOR_PRIVATE_KEY` | Solana | Solana private key for settlement |
| `SOLANA_RPC_URL` | Solana | Solana RPC endpoint (e.g. Helius) |
| `PORT` | No | Server port (Railway sets this automatically) |

## Docker

### Build and Run

```bash
docker build -t x402gate .
docker run -d \
  -p 4021:4021 \
  --env-file .env \
  x402gate
```

### Docker Compose

```yaml
version: "3.8"
services:
  x402gate:
    build: .
    ports:
      - "4021:4021"
    env_file:
      - .env
    restart: unless-stopped
```

```bash
docker compose up -d
```

## VPS (Manual)

### Prerequisites

- Python 3.11+
- systemd (for service management)
- nginx (for reverse proxy / TLS)

### Setup

```bash
# Clone and install
git clone https://github.com/oponfil/x402gate.git
cd x402gate
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your values

# Test run
python -m x402gate.main
```

### systemd Service

Create `/etc/systemd/system/x402gate.service`:

```ini
[Unit]
Description=x402gate API Gateway
After=network.target

[Service]
Type=simple
User=x402gate
WorkingDirectory=/opt/x402gate
EnvironmentFile=/opt/x402gate/.env
ExecStart=/opt/x402gate/.venv/bin/python -m x402gate.main
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable x402gate
sudo systemctl start x402gate
```

### nginx Reverse Proxy

```nginx
server {
    listen 443 ssl;
    server_name x402gate.io;

    ssl_certificate /etc/letsencrypt/live/x402gate.io/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/x402gate.io/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:4021;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 600s;  # Long timeout for AI tasks
    }
}
```
