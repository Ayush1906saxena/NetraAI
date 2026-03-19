#!/bin/bash
# seed_db.sh — Seed the development database with sample data.
#
# Creates test users, stores, patients, and screenings for local development.
# Requires Docker services to be running (PostgreSQL).
#
# Usage: bash scripts/seed_db.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

echo "=============================================="
echo "  Netra AI — Seed Development Database"
echo "=============================================="
echo ""

# -------------------------------------------------------------------
# Check prerequisites
# -------------------------------------------------------------------
log_info "Checking prerequisites ..."

# Check Docker
if ! docker compose ps postgres 2>/dev/null | grep -q "running"; then
    log_warn "PostgreSQL container is not running."
    log_info "Starting Docker services ..."
    cd "$PROJECT_ROOT"
    docker compose up -d postgres redis
    sleep 5
fi

# Check if PostgreSQL is accepting connections
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_USER="${DB_USER:-netra}"
DB_NAME="${DB_NAME:-netra}"
DB_PASS="${DB_PASS:-netra_dev_2026}"

export PGPASSWORD="$DB_PASS"

if ! command -v psql &>/dev/null; then
    log_warn "psql not found. Using docker exec instead."
    PSQL_CMD="docker compose exec -T postgres psql -U $DB_USER -d $DB_NAME"
else
    PSQL_CMD="psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME"
fi

wait_for_db() {
    log_info "Waiting for database to be ready ..."
    for i in $(seq 1 30); do
        if docker compose exec -T postgres pg_isready -U "$DB_USER" &>/dev/null 2>&1; then
            log_ok "Database is ready"
            return 0
        fi
        sleep 1
    done
    log_warn "Database not ready after 30s"
    return 1
}

wait_for_db

# -------------------------------------------------------------------
# Run Alembic migrations (if available)
# -------------------------------------------------------------------
log_info "Running database migrations ..."

cd "$PROJECT_ROOT"
if [[ -f "server/migrations/env.py" ]]; then
    # Activate virtual env if it exists
    if [[ -f "${HOME}/.venvs/netra/bin/activate" ]]; then
        source "${HOME}/.venvs/netra/bin/activate"
    fi

    if command -v alembic &>/dev/null; then
        alembic upgrade head 2>/dev/null || log_warn "Alembic migration failed or not configured yet."
    else
        log_warn "Alembic not installed. Skipping migrations."
    fi
else
    log_warn "No Alembic migrations found. Creating tables via SQL ..."
fi

# -------------------------------------------------------------------
# Seed data via SQL
# -------------------------------------------------------------------
log_info "Seeding database ..."

SEED_SQL=$(cat << 'SQLEOF'
-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create tables if they don't exist (fallback if Alembic hasn't run)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'operator',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS stores (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    code VARCHAR(50) UNIQUE NOT NULL,
    address TEXT,
    city VARCHAR(100),
    state VARCHAR(100),
    pincode VARCHAR(10),
    phone VARCHAR(20),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS patients (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    full_name VARCHAR(255) NOT NULL,
    age INTEGER,
    gender VARCHAR(20),
    phone VARCHAR(20),
    abha_id VARCHAR(50),
    diabetes_type VARCHAR(20),
    diabetes_duration_years INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS screenings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id UUID REFERENCES patients(id),
    store_id UUID REFERENCES stores(id),
    operator_id UUID REFERENCES users(id),
    status VARCHAR(50) DEFAULT 'pending',
    dr_grade_left INTEGER,
    dr_grade_right INTEGER,
    confidence_left FLOAT,
    confidence_right FLOAT,
    is_referable BOOLEAN,
    reviewer_id UUID REFERENCES users(id),
    review_grade_left INTEGER,
    review_grade_right INTEGER,
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    reviewed_at TIMESTAMP
);

-- Seed users (password is 'password123' — bcrypt hash)
INSERT INTO users (email, password_hash, full_name, role) VALUES
    ('admin@netra.ai', '$2b$12$LQv3c1yqBo9SkvXS7QTJPe0b9UxFFi3yQV6MldfGM4VK5jdBtm/bS', 'Admin User', 'admin'),
    ('operator@netra.ai', '$2b$12$LQv3c1yqBo9SkvXS7QTJPe0b9UxFFi3yQV6MldfGM4VK5jdBtm/bS', 'Store Operator', 'operator'),
    ('reviewer@netra.ai', '$2b$12$LQv3c1yqBo9SkvXS7QTJPe0b9UxFFi3yQV6MldfGM4VK5jdBtm/bS', 'Dr. Sharma (Reviewer)', 'reviewer')
ON CONFLICT (email) DO NOTHING;

-- Seed stores
INSERT INTO stores (name, code, address, city, state, pincode, phone) VALUES
    ('Lenskart Connaught Place', 'LK-CP-001', '14, Block B, Connaught Place', 'New Delhi', 'Delhi', '110001', '+91-11-4000-0001'),
    ('Lenskart Koramangala', 'LK-KR-001', '80 Feet Road, Koramangala', 'Bangalore', 'Karnataka', '560034', '+91-80-4000-0002'),
    ('Lenskart Bandra', 'LK-BD-001', 'Hill Road, Bandra West', 'Mumbai', 'Maharashtra', '400050', '+91-22-4000-0003'),
    ('Vision Express Andheri', 'VE-AN-001', 'Andheri West', 'Mumbai', 'Maharashtra', '400058', '+91-22-4000-0004'),
    ('Titan Eyeplus HSR Layout', 'TE-HSR-001', 'HSR Layout', 'Bangalore', 'Karnataka', '560102', '+91-80-4000-0005')
ON CONFLICT (code) DO NOTHING;

-- Seed patients
INSERT INTO patients (full_name, age, gender, phone, diabetes_type, diabetes_duration_years) VALUES
    ('Rajesh Kumar', 52, 'male', '+91-98100-00001', 'type2', 8),
    ('Priya Sharma', 45, 'female', '+91-98100-00002', 'type2', 5),
    ('Amit Patel', 60, 'male', '+91-98100-00003', 'type2', 15),
    ('Sunita Verma', 38, 'female', '+91-98100-00004', 'type1', 20),
    ('Mohammed Khan', 55, 'male', '+91-98100-00005', 'type2', 10),
    ('Ananya Reddy', 48, 'female', '+91-98100-00006', 'type2', 7),
    ('Vikram Singh', 62, 'male', '+91-98100-00007', 'type2', 18),
    ('Lakshmi Nair', 50, 'female', '+91-98100-00008', 'type2', 12),
    ('Ravi Gupta', 44, 'male', '+91-98100-00009', 'type2', 3),
    ('Deepa Joshi', 57, 'female', '+91-98100-00010', 'type2', 9)
ON CONFLICT DO NOTHING;

-- Seed screenings (using subqueries for IDs)
INSERT INTO screenings (patient_id, store_id, operator_id, status, dr_grade_left, dr_grade_right, confidence_left, confidence_right, is_referable, created_at)
SELECT
    p.id,
    s.id,
    u.id,
    'completed',
    0, 0, 0.95, 0.93, false,
    NOW() - INTERVAL '7 days'
FROM patients p, stores s, users u
WHERE p.full_name = 'Rajesh Kumar' AND s.code = 'LK-CP-001' AND u.email = 'operator@netra.ai'
ON CONFLICT DO NOTHING;

INSERT INTO screenings (patient_id, store_id, operator_id, status, dr_grade_left, dr_grade_right, confidence_left, confidence_right, is_referable, created_at)
SELECT
    p.id,
    s.id,
    u.id,
    'completed',
    2, 1, 0.82, 0.88, true,
    NOW() - INTERVAL '5 days'
FROM patients p, stores s, users u
WHERE p.full_name = 'Amit Patel' AND s.code = 'LK-KR-001' AND u.email = 'operator@netra.ai'
ON CONFLICT DO NOTHING;

INSERT INTO screenings (patient_id, store_id, operator_id, status, dr_grade_left, dr_grade_right, confidence_left, confidence_right, is_referable, created_at)
SELECT
    p.id,
    s.id,
    u.id,
    'completed',
    3, 4, 0.78, 0.91, true,
    NOW() - INTERVAL '3 days'
FROM patients p, stores s, users u
WHERE p.full_name = 'Sunita Verma' AND s.code = 'LK-BD-001' AND u.email = 'operator@netra.ai'
ON CONFLICT DO NOTHING;

INSERT INTO screenings (patient_id, store_id, operator_id, status, dr_grade_left, dr_grade_right, confidence_left, confidence_right, is_referable, created_at)
SELECT
    p.id,
    s.id,
    u.id,
    'pending',
    NULL, NULL, NULL, NULL, NULL,
    NOW()
FROM patients p, stores s, users u
WHERE p.full_name = 'Priya Sharma' AND s.code = 'LK-CP-001' AND u.email = 'operator@netra.ai'
ON CONFLICT DO NOTHING;

SQLEOF

# Execute the seed SQL
echo "$SEED_SQL" | eval "$PSQL_CMD" 2>/dev/null && log_ok "Database seeded" || log_warn "Some seed operations may have been skipped (already exists)"

# -------------------------------------------------------------------
# Print summary
# -------------------------------------------------------------------
echo ""
log_info "Querying seed data summary ..."

SUMMARY_SQL="
SELECT 'Users' as entity, COUNT(*) as count FROM users
UNION ALL
SELECT 'Stores', COUNT(*) FROM stores
UNION ALL
SELECT 'Patients', COUNT(*) FROM patients
UNION ALL
SELECT 'Screenings', COUNT(*) FROM screenings;
"

echo "$SUMMARY_SQL" | eval "$PSQL_CMD" 2>/dev/null || log_warn "Could not query summary"

echo ""
echo "=============================================="
echo "  Database seeding complete!"
echo "=============================================="
echo ""
echo "Test credentials:"
echo "  Admin:    admin@netra.ai / password123"
echo "  Operator: operator@netra.ai / password123"
echo "  Reviewer: reviewer@netra.ai / password123"
echo ""
