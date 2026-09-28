# ---- Stage 1: install dependencies into an isolated prefix ----
FROM public.ecr.aws/docker/library/python:3.12-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 120 --retries 5 --prefix=/install -r requirements.txt

# ---- Stage 2: runtime image ----
FROM public.ecr.aws/docker/library/python:3.12-slim

# Lets Lambda talk to a normal web server. Ignored when run anywhere else.
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.1.0 /lambda-adapter /opt/extensions/lambda-adapter

# System-wide location, readable by any user (Lambda runs as non-root)
COPY --from=builder /install /usr/local

ENV PORT=8000 \
    AWS_LWA_READINESS_CHECK_PATH=/health \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /var/task
COPY app/ app/
COPY src/ src/
COPY models/ models/
COPY configs/ configs/
COPY static/ static/

EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]