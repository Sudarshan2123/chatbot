FROM python:3.12.10-alpine3.21 AS builder
WORKDIR /app
COPY requirements.txt .
RUN apk update && apk upgrade && \
    apk add --no-cache gcc musl-dev linux-headers openssl-dev && \
    pip install --no-cache-dir -r requirements.txt && \
    apk del gcc musl-dev linux-headers && \
    rm -rf /var/cache/apk/*

# Final stage
FROM python:3.12.10-alpine3.21
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn
COPY . .

RUN addgroup -g 1000 python && adduser -u 999 -G python -D python && \
    chown -R python:python /app
USER 999
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

