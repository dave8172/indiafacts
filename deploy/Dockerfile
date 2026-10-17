# The indiafacts engine (engine/server.py). The corpus lives in Postgres on the host (schema indiafacts), reached
# through its mounted unix socket; secrets are mounted at run time, never baked in.
FROM python:3.12-slim
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /app
COPY engine/ engine/
COPY sources.json sites.json ./
ARG IF_VERSION=unknown
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 IF_VERSION=$IF_VERSION
CMD ["python", "-m", "engine.server"]
