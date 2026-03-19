#!/bin/bash
# setup.sh — Full environment setup for Netra AI
#
# Run once on a fresh Mac to set up the complete development environment.
# Usage: bash scripts/setup.sh
#
# Prerequisites:
#   - macOS 14+ (Sonoma or later)
#   - Apple Silicon (M1/M2/M3/M4) recommended
#   - 100GB free disk space

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${HOME}/.venvs/netra"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "=============================================="
echo "  Netra AI — Development Environment Setup"
echo "=============================================="
echo ""
echo "Project root: $PROJECT_ROOT"
echo ""

# -------------------------------------------------------------------
# 1. Check prerequisites
# -------------------------------------------------------------------
log_info "Checking prerequisites ..."

if ! command -v brew &>/dev/null; then
    log_info "Installing Homebrew ..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Add Homebrew to PATH for Apple Silicon
    if [[ -f /opt/homebrew/bin/brew ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
else
    log_ok "Homebrew already installed"
fi

# -------------------------------------------------------------------
# 2. System dependencies
# -------------------------------------------------------------------
log_info "Installing system dependencies via Homebrew ..."

BREW_PACKAGES=(
    python@3.11
    cmake
    pkg-config
    jpeg-turbo
    libpng
    libtiff
    webp
    poppler        # PDF rendering
    tesseract      # OCR fallback
)

for pkg in "${BREW_PACKAGES[@]}"; do
    if brew list "$pkg" &>/dev/null 2>&1; then
        log_ok "$pkg already installed"
    else
        log_info "Installing $pkg ..."
        brew install "$pkg"
    fi
done

# Docker
if ! command -v docker &>/dev/null; then
    log_info "Installing Docker Desktop ..."
    brew install --cask docker
    log_warn "Please open Docker Desktop and configure: 4GB RAM, 2 CPUs"
else
    log_ok "Docker already installed"
fi

# -------------------------------------------------------------------
# 3. Python virtual environment
# -------------------------------------------------------------------
log_info "Setting up Python virtual environment ..."

PYTHON_BIN="python3.11"
if ! command -v "$PYTHON_BIN" &>/dev/null; then
    PYTHON_BIN="python3"
fi

if [[ -d "$VENV_DIR" ]]; then
    log_ok "Virtual environment already exists at $VENV_DIR"
else
    log_info "Creating virtual environment at $VENV_DIR ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"
log_ok "Activated venv: $(which python)"

# Upgrade pip
pip install --upgrade pip setuptools wheel

# -------------------------------------------------------------------
# 4. Install Python dependencies
# -------------------------------------------------------------------
log_info "Installing Python dependencies ..."

cd "$PROJECT_ROOT"

# Install the project in development mode
pip install -e ".[dev]"

log_ok "Python dependencies installed"

# -------------------------------------------------------------------
# 5. Start Docker services
# -------------------------------------------------------------------
log_info "Starting Docker services ..."

if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
    cd "$PROJECT_ROOT"
    docker compose up -d
    log_ok "Docker services started"

    # Wait for PostgreSQL to be ready
    log_info "Waiting for PostgreSQL to be ready ..."
    for i in $(seq 1 30); do
        if docker compose exec -T postgres pg_isready -U netra &>/dev/null 2>&1; then
            log_ok "PostgreSQL is ready"
            break
        fi
        if [[ $i -eq 30 ]]; then
            log_warn "PostgreSQL not ready after 30s. Check docker compose logs."
        fi
        sleep 1
    done

    # Create MinIO buckets
    if command -v mc &>/dev/null; then
        log_info "Creating MinIO buckets ..."
        mc alias set local http://localhost:9000 minioadmin minioadmin 2>/dev/null || true
        for bucket in fundus-images reports model-artifacts mlflow-artifacts training-data; do
            mc mb "local/$bucket" 2>/dev/null || true
        done
        log_ok "MinIO buckets created"
    else
        log_warn "MinIO client (mc) not found. Install with: brew install minio-mc"
        log_warn "Then run: scripts/setup.sh again or manually create buckets."
    fi
else
    log_warn "Docker is not running. Start Docker Desktop and re-run this script."
    log_warn "Continuing without Docker services ..."
fi

# -------------------------------------------------------------------
# 6. Clone RETFound model repository
# -------------------------------------------------------------------
RETFOUND_DIR="${HOME}/netra/models/RETFound"
if [[ -d "$RETFOUND_DIR" ]]; then
    log_ok "RETFound repo already cloned at $RETFOUND_DIR"
else
    log_info "Cloning RETFound repository ..."
    mkdir -p "$(dirname "$RETFOUND_DIR")"
    git clone https://github.com/rmaphoh/RETFound.git "$RETFOUND_DIR" || {
        log_warn "Failed to clone RETFound. You can clone it manually later."
    }
fi

# -------------------------------------------------------------------
# 7. Create local directories
# -------------------------------------------------------------------
log_info "Creating local directories ..."

DIRS=(
    "$PROJECT_ROOT/data/raw"
    "$PROJECT_ROOT/data/processed"
    "$PROJECT_ROOT/weights"
    "$PROJECT_ROOT/results"
    "$PROJECT_ROOT/exports"
    "$PROJECT_ROOT/logs"
)

for dir in "${DIRS[@]}"; do
    mkdir -p "$dir"
done
log_ok "Local directories created"

# -------------------------------------------------------------------
# 8. Create .env file if not exists
# -------------------------------------------------------------------
ENV_FILE="$PROJECT_ROOT/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    log_info "Creating .env file from template ..."
    cat > "$ENV_FILE" << 'ENVEOF'
# Netra AI Local Development Environment
# Do NOT commit this file.

# Database
DATABASE_URL=postgresql+asyncpg://netra:netra_dev_2026@localhost:5432/netra
DATABASE_URL_SYNC=postgresql://netra:netra_dev_2026@localhost:5432/netra

# Redis
REDIS_URL=redis://localhost:6379/0

# MinIO (S3-compatible storage)
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_IMAGES=fundus-images
MINIO_BUCKET_REPORTS=reports
MINIO_BUCKET_MODELS=model-artifacts

# JWT Auth
SECRET_KEY=dev-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=480

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin

# Model paths
DR_MODEL_PATH=weights/dr_grader_v1.pt
IQA_MODEL_PATH=weights/iqa_v1.pt

# Device
INFERENCE_DEVICE=mps
ENVEOF
    log_ok ".env file created"
else
    log_ok ".env file already exists"
fi

# -------------------------------------------------------------------
# 9. Verify installation
# -------------------------------------------------------------------
log_info "Verifying installation ..."

echo ""
echo "--- Python ---"
python --version
echo "Torch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')"
echo "MPS available: $(python -c 'import torch; print(torch.backends.mps.is_available())' 2>/dev/null || echo 'N/A')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"

echo ""
echo "--- Docker Services ---"
if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
    docker compose ps 2>/dev/null || true
else
    echo "Docker not running"
fi

echo ""
echo "=============================================="
echo "  Setup complete!"
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To start working:"
echo "  cd $PROJECT_ROOT"
echo "  bash scripts/run_local.sh"
echo ""
echo "To run tests:"
echo "  pytest tests/"
echo ""
