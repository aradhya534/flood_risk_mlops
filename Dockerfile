# ---- Stage 1: install dependencies ----
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ---- Stage 2: runtime image ----
FROM python:3.12-slim

WORKDIR /app

# copy installed packages from the builder stage (keeps final image smaller)
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# copy only what the running app actually needs
COPY app/ app/
COPY src/ src/
COPY models/ models/
COPY configs/ configs/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

