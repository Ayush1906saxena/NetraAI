#!/bin/bash
# run_local.sh — Start all Netra AI services locally.
#
# Starts:
#   1. Docker services (PostgreSQL, Redis, MinIO, MLflow)
#   2. FastAPI backend server
#   3. Celery worker (async task processing)
#
# Usage:
#   bash scripts/run_local.sh          # Start all services
#   bash scripts/run_local.sh --api    # Start only the API server
#   bash scripts/run_local.sh --stop   # Stop all services
#
# Prerequisites:
#   - Run scripts/setup.sh first
#   - Docker Desktop running

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${HOME}/.venvs/netra"
LOG_DIR="$PROJECT_ROOT/logs"
PID_DIR="$PROJECT_ROOT/.pids"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

mkdir -p "$LOG_DIR" "$PID_DIR"

# -------------------------------------------------------------------
# Parse arguments
# -------------------------------------------------------------------
MODE="all"
if [[ "${1:-}" == "--api" ]]; then
    MODE="api"
elif [[ "${1:-}" == "--stop" ]]; then
    MODE="stop"
elif [[ "${1:-}" == "--status" ]]; then
    MODE="status"
elif [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    echo "Usage: $0 [--api | --stop | --status | --help]"
    echo ""
    echo "  (no args)   Start all services (Docker + API + Celery)"
    echo "  --api       Start only the FastAPI server"
    echo "  --stop      Stop all running services"
    echo "  --status    Show status of all services"
    echo "  --help      Show this help message"
    exit 0
fi

# -------------------------------------------------------------------
# Activate virtual environment
# -------------------------------------------------------------------
activate_venv() {
    if [[ -f "$VENV_DIR/bin/activate" ]]; then
        source "$VENV_DIR/bin/activate"
    else
        log_error "Virtual environment not found at $VENV_DIR"
        log_error "Run 'bash scripts/setup.sh' first."
        exit 1
    fi
}

# -------------------------------------------------------------------
# Stop all services
# -------------------------------------------------------------------
stop_services() {
    log_info "Stopping all Netra AI services ..."

    # Stop FastAPI server
    if [[ -f "$PID_DIR/api.pid" ]]; then
        API_PID=$(cat "$PID_DIR/api.pid")
        if kill -0 "$API_PID" 2>/dev/null; then
            kill "$API_PID" 2>/dev/null || true
            log_ok "Stopped API server (PID $API_PID)"
        fi
        rm -f "$PID_DIR/api.pid"
    fi

    # Stop Celery worker
    if [[ -f "$PID_DIR/celery.pid" ]]; then
        CELERY_PID=$(cat "$PID_DIR/celery.pid")
        if kill -0 "$CELERY_PID" 2>/dev/null; then
            kill "$CELERY_PID" 2>/dev/null || true
            log_ok "Stopped Celery worker (PID $CELERY_PID)"
        fi
        rm -f "$PID_DIR/celery.pid"
    fi

    # Also kill any orphaned processes
    pkill -f "uvicorn server.main:app" 2>/dev/null || true
    pkill -f "celery -A server.workers" 2>/dev/null || true

    # Stop Docker services
    cd "$PROJECT_ROOT"
    if command -v docker &>/dev/null; then
        docker compose down 2>/dev/null || true
        log_ok "Docker services stopped"
    fi

    log_ok "All services stopped"
}

# -------------------------------------------------------------------
# Show status
# -------------------------------------------------------------------
show_status() {
    echo ""
    echo "=============================================="
    echo "  Netra AI — Service Status"
    echo "=============================================="
    echo ""

    # Docker services
    echo -e "${CYAN}Docker Services:${NC}"
    if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
        cd "$PROJECT_ROOT"
        docker compose ps 2>/dev/null || echo "  Docker Compose not configured"
    else
        echo "  Docker not running"
    fi

    echo ""
    echo -e "${CYAN}Application Services:${NC}"

    # API server
    if [[ -f "$PID_DIR/api.pid" ]] && kill -0 "$(cat "$PID_DIR/api.pid")" 2>/dev/null; then
        echo -e "  API Server:    ${GREEN}running${NC} (PID $(cat "$PID_DIR/api.pid"))"
    else
        echo -e "  API Server:    ${RED}stopped${NC}"
    fi

    # Celery worker
    if [[ -f "$PID_DIR/celery.pid" ]] && kill -0 "$(cat "$PID_DIR/celery.pid")" 2>/dev/null; then
        echo -e "  Celery Worker: ${GREEN}running${NC} (PID $(cat "$PID_DIR/celery.pid"))"
    else
        echo -e "  Celery Worker: ${RED}stopped${NC}"
    fi

    echo ""
    echo -e "${CYAN}URLs:${NC}"
    echo "  API:        http://localhost:8000"
    echo "  API Docs:   http://localhost:8000/docs"
    echo "  MLflow:     http://localhost:5000"
    echo "  MinIO:      http://localhost:9001"
    echo ""
}

# -------------------------------------------------------------------
# Handle modes
# -------------------------------------------------------------------
if [[ "$MODE" == "stop" ]]; then
    stop_services
    exit 0
fi

if [[ "$MODE" == "status" ]]; then
    show_status
    exit 0
fi

# -------------------------------------------------------------------
# Start services
# -------------------------------------------------------------------
echo "=============================================="
echo "  Netra AI — Starting Local Services"
echo "=============================================="
echo ""

activate_venv
cd "$PROJECT_ROOT"

# Load environment variables
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
    log_ok "Loaded .env"
fi

# -------------------------------------------------------------------
# Start Docker services (unless --api only)
# -------------------------------------------------------------------
if [[ "$MODE" == "all" ]]; then
    log_info "Starting Docker services ..."

    if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
        docker compose up -d
        log_ok "Docker services started"

        # Wait for PostgreSQL
        log_info "Waiting for PostgreSQL ..."
        for i in $(seq 1 30); do
            if docker compose exec -T postgres pg_isready -U netra &>/dev/null 2>&1; then
                log_ok "PostgreSQL ready"
                break
            fi
            sleep 1
        done

        # Wait for Redis
        log_info "Checking Redis ..."
        if docker compose exec -T redis redis-cli ping &>/dev/null 2>&1; then
            log_ok "Redis ready"
        else
            log_warn "Redis not responding"
        fi
    else
        log_warn "Docker not available. Starting without Docker services."
        log_warn "Database and cache will not be available."
    fi
fi

# -------------------------------------------------------------------
# Start Celery worker (background)
# -------------------------------------------------------------------
if [[ "$MODE" == "all" ]]; then
    log_info "Starting Celery worker ..."

    celery -A server.workers worker \
        --loglevel=info \
        --concurrency=2 \
        --pool=prefork \
        --pidfile="$PID_DIR/celery.pid" \
        > "$LOG_DIR/celery.log" 2>&1 &

    CELERY_PID=$!
    echo "$CELERY_PID" > "$PID_DIR/celery.pid"
    log_ok "Celery worker started (PID $CELERY_PID, log: $LOG_DIR/celery.log)"
fi

# -------------------------------------------------------------------
# Start FastAPI server (foreground)
# -------------------------------------------------------------------
log_info "Starting FastAPI server ..."
echo ""
echo -e "${CYAN}API server starting at http://localhost:8000${NC}"
echo -e "${CYAN}API docs at http://localhost:8000/docs${NC}"
echo -e "${CYAN}Press Ctrl+C to stop${NC}"
echo ""

# Run uvicorn in foreground so Ctrl+C works naturally
uvicorn server.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --reload-dir server \
    --log-level info \
    2>&1 | tee "$LOG_DIR/api.log" &

API_PID=$!
echo "$API_PID" > "$PID_DIR/api.pid"

# Trap Ctrl+C to clean up
cleanup() {
    echo ""
    log_info "Shutting down ..."
    kill "$API_PID" 2>/dev/null || true

    if [[ "$MODE" == "all" ]]; then
        if [[ -f "$PID_DIR/celery.pid" ]]; then
            kill "$(cat "$PID_DIR/celery.pid")" 2>/dev/null || true
        fi
    fi

    rm -f "$PID_DIR/api.pid" "$PID_DIR/celery.pid"
    log_ok "Services stopped. Docker services still running (use --stop to stop all)."
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for the API process
wait "$API_PID"
