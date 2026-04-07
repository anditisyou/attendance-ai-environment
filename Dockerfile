FROM ghcr.io/astral-sh/uv:python3.10-bookworm

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "-m", "server.app"]
