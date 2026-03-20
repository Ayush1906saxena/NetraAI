# Stage 1: Build React dashboard
FROM node:20-alpine AS dashboard-build
WORKDIR /app/dashboard
RUN corepack enable && corepack prepare pnpm@latest --activate
COPY dashboard/package.json dashboard/pnpm-lock.yaml* ./
RUN pnpm install --frozen-lockfile || pnpm install
COPY dashboard/ ./
RUN pnpm build

# Stage 2: Python application
FROM python:3.11-slim AS runtime
WORKDIR /app

# Install system dependencies required by OpenCV and other libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e "." 2>/dev/null || pip install --no-cache-dir .

# Copy application code
COPY server/ ./server/
COPY ml/ ./ml/
COPY alembic.ini ./

# Copy dashboard build output to serve as static files
COPY --from=dashboard-build /app/dashboard/dist ./static/

EXPOSE 8000

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
