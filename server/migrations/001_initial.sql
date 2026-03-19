-- NetraAI Initial Schema Migration
-- Raw SQL version for direct execution against PostgreSQL
-- Compatible with PostgreSQL 14+

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE stores (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    city VARCHAR(100) NOT NULL,
    state VARCHAR(100) NOT NULL,
    tier VARCHAR(10),
    camera_model VARCHAR(100),
    camera_serial VARCHAR(100),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('operator', 'admin', 'reviewer', 'ml_engineer')),
    store_id UUID REFERENCES stores(id),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE patients (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    abha_id VARCHAR(20) UNIQUE,
    full_name VARCHAR(255) NOT NULL,
    age INTEGER,
    gender VARCHAR(10),
    phone VARCHAR(15),
    diabetes_status VARCHAR(20) CHECK (diabetes_status IN ('confirmed', 'suspected', 'unknown')),
    diabetes_duration_years INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    consent_given BOOLEAN DEFAULT false,
    consent_timestamp TIMESTAMPTZ
);

CREATE TABLE screenings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id UUID REFERENCES patients(id) NOT NULL,
    store_id UUID REFERENCES stores(id) NOT NULL,
    operator_id UUID REFERENCES users(id) NOT NULL,
    left_eye_key VARCHAR(500),
    right_eye_key VARCHAR(500),
    left_eye_quality FLOAT,
    right_eye_quality FLOAT,
    results JSONB,
    referral JSONB,
    status VARCHAR(20) DEFAULT 'created' CHECK (status IN ('created', 'partial', 'analyzing', 'completed', 'failed', 'reviewed')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    analyzed_at TIMESTAMPTZ,
    reviewed_at TIMESTAMPTZ,
    reviewer_id UUID REFERENCES users(id),
    reviewer_grade_left JSONB,
    reviewer_grade_right JSONB,
    reviewer_agrees BOOLEAN,
    reviewer_notes TEXT
);

CREATE TABLE reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    screening_id UUID REFERENCES screenings(id) NOT NULL,
    report_key VARCHAR(500) NOT NULL,
    language VARCHAR(5) DEFAULT 'en',
    generated_at TIMESTAMPTZ DEFAULT NOW(),
    sent_via VARCHAR(20),
    sent_at TIMESTAMPTZ,
    delivery_status VARCHAR(20)
);

CREATE TABLE audit_log (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    action VARCHAR(50) NOT NULL,
    resource_type VARCHAR(50),
    resource_id UUID,
    details JSONB,
    ip_address INET,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_screenings_patient ON screenings(patient_id);
CREATE INDEX idx_screenings_store ON screenings(store_id);
CREATE INDEX idx_screenings_status ON screenings(status);
CREATE INDEX idx_screenings_created ON screenings(created_at DESC);
CREATE INDEX idx_patients_abha ON patients(abha_id) WHERE abha_id IS NOT NULL;
CREATE INDEX idx_audit_user ON audit_log(user_id);
CREATE INDEX idx_audit_created ON audit_log(created_at DESC);
