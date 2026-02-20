FROM python:3.12-slim AS builder

WORKDIR /app

COPY pyproject.toml .
COPY README.md .
RUN pip install --no-cache-dir .

FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY config.yaml .
COPY x402gate/ x402gate/

EXPOSE 4021

CMD ["python", "-m", "x402gate.main"]
