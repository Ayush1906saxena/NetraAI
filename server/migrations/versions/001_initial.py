"""Initial migration: create all tables.

Revision ID: 001_initial
Revises:
Create Date: 2026-03-18

"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable uuid-ossp extension
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    # ── stores ──────────────────────────────────────────────────────────
    op.create_table(
        "stores",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            primary_key=True,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("code", sa.String(50), unique=True, nullable=False),
        sa.Column("address", sa.Text, nullable=True),
        sa.Column("city", sa.String(100), nullable=True),
        sa.Column("state", sa.String(100), nullable=True),
        sa.Column("pincode", sa.String(10), nullable=True),
        sa.Column("phone", sa.String(20), nullable=True),
        sa.Column("email", sa.String(255), nullable=True),
        sa.Column("latitude", sa.Float, nullable=True),
        sa.Column("longitude", sa.Float, nullable=True),
        sa.Column("is_active", sa.Boolean, server_default=sa.text("true")),
        sa.Column("manager_name", sa.String(255), nullable=True),
        sa.Column("equipment_info", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("idx_stores_name", "stores", ["name"])
    op.create_index("idx_stores_code", "stores", ["code"], unique=True)

    # ── users ───────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            primary_key=True,
        ),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("phone", sa.String(20), unique=True, nullable=True),
        sa.Column("full_name", sa.String(255), nullable=False),
        sa.Column("hashed_password", sa.String(512), nullable=False),
        sa.Column("role", sa.String(50), nullable=False, server_default="operator"),
        sa.Column(
            "store_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("stores.id"),
            nullable=True,
        ),
        sa.Column("is_active", sa.Boolean, server_default=sa.text("true")),
        sa.Column("is_verified", sa.Boolean, server_default=sa.text("false")),
        sa.Column("last_login", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("idx_users_email", "users", ["email"], unique=True)
    op.create_index("idx_users_store_id", "users", ["store_id"])

    # ── patients ────────────────────────────────────────────────────────
    op.create_table(
        "patients",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            primary_key=True,
        ),
        sa.Column("full_name", sa.String(255), nullable=False),
        sa.Column("date_of_birth", sa.Date, nullable=True),
        sa.Column("age", sa.Integer, nullable=True),
        sa.Column("gender", sa.String(20), nullable=True),
        sa.Column("phone", sa.String(20), nullable=True),
        sa.Column("email", sa.String(255), nullable=True),
        sa.Column("address", sa.Text, nullable=True),
        sa.Column("city", sa.String(100), nullable=True),
        sa.Column("state", sa.String(100), nullable=True),
        sa.Column("pincode", sa.String(10), nullable=True),
        sa.Column("abha_id", sa.String(50), unique=True, nullable=True),
        sa.Column("abha_address", sa.String(255), nullable=True),
        sa.Column("health_id_number", sa.String(50), nullable=True),
        sa.Column("is_diabetic", sa.Boolean, nullable=True),
        sa.Column("diabetes_duration_years", sa.Integer, nullable=True),
        sa.Column("has_hypertension", sa.Boolean, nullable=True),
        sa.Column("medical_notes", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("idx_patients_full_name", "patients", ["full_name"])
    op.create_index("idx_patients_phone", "patients", ["phone"])
    op.create_index(
        "idx_patients_abha",
        "patients",
        ["abha_id"],
        unique=True,
        postgresql_where=sa.text("abha_id IS NOT NULL"),
    )

    # ── screenings ──────────────────────────────────────────────────────
    op.create_table(
        "screenings",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            primary_key=True,
        ),
        sa.Column(
            "patient_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("patients.id"),
            nullable=False,
        ),
        sa.Column(
            "store_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("stores.id"),
            nullable=False,
        ),
        sa.Column(
            "operator_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id"),
            nullable=False,
        ),
        sa.Column("status", sa.String(50), nullable=False, server_default="created"),
        # AI results
        sa.Column("dr_grade_left", sa.Integer, nullable=True),
        sa.Column("dr_grade_right", sa.Integer, nullable=True),
        sa.Column("dr_confidence_left", sa.Float, nullable=True),
        sa.Column("dr_confidence_right", sa.Float, nullable=True),
        sa.Column("glaucoma_prob_left", sa.Float, nullable=True),
        sa.Column("glaucoma_prob_right", sa.Float, nullable=True),
        sa.Column("amd_prob_left", sa.Float, nullable=True),
        sa.Column("amd_prob_right", sa.Float, nullable=True),
        # Risk & referral
        sa.Column("overall_risk", sa.String(20), nullable=True),
        sa.Column("referral_required", sa.Boolean, nullable=True),
        sa.Column("referral_urgency", sa.String(20), nullable=True),
        sa.Column("referral_reason", sa.Text, nullable=True),
        # Full results JSON
        sa.Column("raw_results", postgresql.JSON, nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
        # Timestamps
        sa.Column(
            "screened_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("idx_screenings_patient", "screenings", ["patient_id"])
    op.create_index("idx_screenings_store", "screenings", ["store_id"])
    op.create_index("idx_screenings_operator", "screenings", ["operator_id"])
    op.create_index("idx_screenings_status", "screenings", ["status"])
    op.create_index(
        "idx_screenings_created",
        "screenings",
        [sa.text("created_at DESC")],
    )

    # ── images ──────────────────────────────────────────────────────────
    op.create_table(
        "images",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            primary_key=True,
        ),
        sa.Column(
            "screening_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("screenings.id"),
            nullable=False,
        ),
        sa.Column("eye", sa.String(10), nullable=False),
        sa.Column(
            "image_type", sa.String(50), nullable=False, server_default="fundus"
        ),
        sa.Column("s3_key", sa.String(512), nullable=False),
        sa.Column("s3_bucket", sa.String(255), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column(
            "content_type", sa.String(100), nullable=False, server_default="image/jpeg"
        ),
        sa.Column("file_size_bytes", sa.Integer, nullable=True),
        sa.Column("iqa_score", sa.Float, nullable=True),
        sa.Column("iqa_passed", sa.Boolean, nullable=True),
        sa.Column("iqa_details", postgresql.JSON, nullable=True),
        sa.Column("gradcam_s3_key", sa.String(512), nullable=True),
        sa.Column("ai_results", postgresql.JSON, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("idx_images_screening", "images", ["screening_id"])

    # ── reports ─────────────────────────────────────────────────────────
    op.create_table(
        "reports",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            primary_key=True,
        ),
        sa.Column(
            "screening_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("screenings.id"),
            unique=True,
            nullable=False,
        ),
        sa.Column("s3_key", sa.String(512), nullable=False),
        sa.Column("s3_bucket", sa.String(255), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("format", sa.String(20), nullable=False, server_default="pdf"),
        sa.Column("language", sa.String(10), nullable=False, server_default="en"),
        sa.Column("whatsapp_sent", sa.Boolean, server_default=sa.text("false")),
        sa.Column("sms_sent", sa.Boolean, server_default=sa.text("false")),
        sa.Column("email_sent", sa.Boolean, server_default=sa.text("false")),
        sa.Column("whatsapp_sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("abdm_pushed", sa.Boolean, server_default=sa.text("false")),
        sa.Column("abdm_record_id", sa.String(255), nullable=True),
        sa.Column(
            "generated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("idx_reports_screening", "reports", ["screening_id"], unique=True)

    # ── audit_logs ──────────────────────────────────────────────────────
    op.create_table(
        "audit_logs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            primary_key=True,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id"),
            nullable=True,
        ),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("resource_type", sa.String(50), nullable=True),
        sa.Column("resource_id", sa.String(255), nullable=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("user_agent", sa.String(512), nullable=True),
        sa.Column("request_method", sa.String(10), nullable=True),
        sa.Column("request_path", sa.String(512), nullable=True),
        sa.Column("metadata_json", postgresql.JSON, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("idx_audit_user", "audit_logs", ["user_id"])
    op.create_index("idx_audit_action", "audit_logs", ["action"])
    op.create_index(
        "idx_audit_created",
        "audit_logs",
        [sa.text("created_at DESC")],
    )


def downgrade() -> None:
    op.drop_table("audit_logs")
    op.drop_table("reports")
    op.drop_table("images")
    op.drop_table("screenings")
    op.drop_table("patients")
    op.drop_table("users")
    op.drop_table("stores")
    op.execute('DROP EXTENSION IF EXISTS "uuid-ossp"')
