"""
Database seed script — populates PostgreSQL with test data.

Creates:
- 3 stores (Mumbai, Delhi, Bangalore)
- 5 users (1 admin, 2 operators, 2 reviewers) with hashed passwords
- 10 patients with realistic Indian names
- 5 sample screenings with various statuses

Run with:
    .venv/bin/python scripts/seed_db.py
"""

import asyncio
import hashlib
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from server.config import settings
from server.models.user import Base, User
from server.models.store import Store
from server.models.patient import Patient
from server.models.screening import Screening
from server.models.image import Image
from server.models.report import Report
from server.models.audit_log import AuditLog


def _hash_password(plain: str) -> str:
    """Simple SHA-256 password hash for seeding.

    In production you would use bcrypt/argon2 via passlib, but for seed
    data we just need something stored in hashed_password.
    """
    try:
        from passlib.context import CryptContext
        ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
        return ctx.hash(plain)
    except ImportError:
        # Fallback if passlib not installed
        return hashlib.sha256(plain.encode()).hexdigest()


async def seed():
    """Create test data in PostgreSQL."""
    engine = create_async_engine(
        settings.DATABASE_URL,
        pool_size=5,
        max_overflow=5,
        echo=False,
    )

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("[+] Tables created / verified.")

    session_factory = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )

    async with session_factory() as db:
        # ── Stores ────────────────────────────────────────────────────
        stores = [
            Store(
                id=uuid.uuid4(),
                name="NetraAI Vision Center - Mumbai",
                code="MUM-001",
                address="42 Marine Drive, Churchgate",
                city="Mumbai",
                state="Maharashtra",
                pincode="400020",
                phone="+919876543210",
                email="mumbai@netra.ai",
                latitude=18.9322,
                longitude=72.8264,
                is_active=True,
                manager_name="Priya Deshmukh",
                equipment_info="Topcon NW400, Canon CR-2",
            ),
            Store(
                id=uuid.uuid4(),
                name="NetraAI Vision Center - Delhi",
                code="DEL-001",
                address="15 Connaught Place, Block B",
                city="New Delhi",
                state="Delhi",
                pincode="110001",
                phone="+919876543211",
                email="delhi@netra.ai",
                latitude=28.6315,
                longitude=77.2167,
                is_active=True,
                manager_name="Amit Sharma",
                equipment_info="Zeiss VISUCAM, Optovue iCam",
            ),
            Store(
                id=uuid.uuid4(),
                name="NetraAI Vision Center - Bangalore",
                code="BLR-001",
                address="88 MG Road, Shivaji Nagar",
                city="Bangalore",
                state="Karnataka",
                pincode="560001",
                phone="+919876543212",
                email="bangalore@netra.ai",
                latitude=12.9716,
                longitude=77.5946,
                is_active=True,
                manager_name="Kavitha Rao",
                equipment_info="Topcon TRC-50DX, Nikon NM-1000",
            ),
        ]
        for s in stores:
            db.add(s)
        await db.flush()
        print(f"[+] Created {len(stores)} stores.")

        # ── Users ─────────────────────────────────────────────────────
        hashed_pw = _hash_password("netra2024")
        users = [
            User(
                id=uuid.uuid4(),
                email="admin@netra.ai",
                phone="+919900000001",
                full_name="Dr. Rajesh Kumar",
                hashed_password=hashed_pw,
                role="admin",
                store_id=None,
                is_active=True,
                is_verified=True,
            ),
            User(
                id=uuid.uuid4(),
                email="operator1@netra.ai",
                phone="+919900000002",
                full_name="Sneha Patel",
                hashed_password=hashed_pw,
                role="operator",
                store_id=stores[0].id,
                is_active=True,
                is_verified=True,
            ),
            User(
                id=uuid.uuid4(),
                email="operator2@netra.ai",
                phone="+919900000003",
                full_name="Vikram Singh",
                hashed_password=hashed_pw,
                role="operator",
                store_id=stores[1].id,
                is_active=True,
                is_verified=True,
            ),
            User(
                id=uuid.uuid4(),
                email="reviewer1@netra.ai",
                phone="+919900000004",
                full_name="Dr. Ananya Iyer",
                hashed_password=hashed_pw,
                role="doctor",
                store_id=stores[0].id,
                is_active=True,
                is_verified=True,
            ),
            User(
                id=uuid.uuid4(),
                email="reviewer2@netra.ai",
                phone="+919900000005",
                full_name="Dr. Arjun Reddy",
                hashed_password=hashed_pw,
                role="doctor",
                store_id=stores[2].id,
                is_active=True,
                is_verified=True,
            ),
        ]
        for u in users:
            db.add(u)
        await db.flush()
        print(f"[+] Created {len(users)} users (password: netra2024).")

        # ── Patients ──────────────────────────────────────────────────
        patients = [
            Patient(
                id=uuid.uuid4(),
                full_name="Ramesh Gupta",
                date_of_birth=date(1965, 3, 15),
                age=61,
                gender="male",
                phone="+919800000001",
                email="ramesh.gupta@email.com",
                city="Mumbai",
                state="Maharashtra",
                pincode="400053",
                is_diabetic=True,
                diabetes_duration_years=12,
                has_hypertension=True,
            ),
            Patient(
                id=uuid.uuid4(),
                full_name="Sunita Devi",
                date_of_birth=date(1972, 7, 22),
                age=53,
                gender="female",
                phone="+919800000002",
                city="Delhi",
                state="Delhi",
                pincode="110032",
                is_diabetic=True,
                diabetes_duration_years=8,
                has_hypertension=False,
            ),
            Patient(
                id=uuid.uuid4(),
                full_name="Mohammed Farid",
                date_of_birth=date(1958, 11, 5),
                age=67,
                gender="male",
                phone="+919800000003",
                city="Bangalore",
                state="Karnataka",
                pincode="560034",
                is_diabetic=True,
                diabetes_duration_years=20,
                has_hypertension=True,
            ),
            Patient(
                id=uuid.uuid4(),
                full_name="Lakshmi Narayanan",
                date_of_birth=date(1980, 1, 10),
                age=46,
                gender="female",
                phone="+919800000004",
                city="Mumbai",
                state="Maharashtra",
                pincode="400071",
                is_diabetic=False,
                has_hypertension=False,
            ),
            Patient(
                id=uuid.uuid4(),
                full_name="Harpreet Kaur",
                date_of_birth=date(1975, 9, 28),
                age=50,
                gender="female",
                phone="+919800000005",
                city="Delhi",
                state="Delhi",
                pincode="110019",
                is_diabetic=True,
                diabetes_duration_years=5,
                has_hypertension=True,
            ),
            Patient(
                id=uuid.uuid4(),
                full_name="Suresh Babu",
                date_of_birth=date(1960, 4, 18),
                age=65,
                gender="male",
                phone="+919800000006",
                city="Bangalore",
                state="Karnataka",
                pincode="560011",
                is_diabetic=True,
                diabetes_duration_years=15,
                has_hypertension=True,
            ),
            Patient(
                id=uuid.uuid4(),
                full_name="Meena Kumari",
                date_of_birth=date(1988, 12, 3),
                age=37,
                gender="female",
                phone="+919800000007",
                city="Mumbai",
                state="Maharashtra",
                pincode="400058",
                is_diabetic=False,
                has_hypertension=False,
            ),
            Patient(
                id=uuid.uuid4(),
                full_name="Ravi Shankar",
                date_of_birth=date(1970, 6, 14),
                age=55,
                gender="male",
                phone="+919800000008",
                city="Delhi",
                state="Delhi",
                pincode="110048",
                is_diabetic=True,
                diabetes_duration_years=10,
                has_hypertension=False,
            ),
            Patient(
                id=uuid.uuid4(),
                full_name="Deepa Krishnan",
                date_of_birth=date(1982, 2, 20),
                age=44,
                gender="female",
                phone="+919800000009",
                city="Bangalore",
                state="Karnataka",
                pincode="560001",
                is_diabetic=True,
                diabetes_duration_years=3,
                has_hypertension=False,
            ),
            Patient(
                id=uuid.uuid4(),
                full_name="Arun Joshi",
                date_of_birth=date(1955, 8, 7),
                age=70,
                gender="male",
                phone="+919800000010",
                city="Mumbai",
                state="Maharashtra",
                pincode="400001",
                is_diabetic=True,
                diabetes_duration_years=25,
                has_hypertension=True,
                medical_notes="History of cataract surgery (2019). On insulin.",
            ),
        ]
        for p in patients:
            db.add(p)
        await db.flush()
        print(f"[+] Created {len(patients)} patients.")

        # ── Screenings ────────────────────────────────────────────────
        screenings = [
            Screening(
                id=uuid.uuid4(),
                patient_id=patients[0].id,
                store_id=stores[0].id,
                operator_id=users[1].id,
                status="completed",
                dr_grade_left=2,
                dr_grade_right=1,
                dr_confidence_left=0.87,
                dr_confidence_right=0.92,
                overall_risk="moderate",
                referral_required=True,
                referral_urgency="urgent",
                referral_reason="Moderate NPDR in left eye (grade 2)",
                raw_results={"analysis_version": "v2", "model": "efficientnet_b3"},
                completed_at=datetime(2026, 3, 15, 10, 30, 0, tzinfo=timezone.utc),
            ),
            Screening(
                id=uuid.uuid4(),
                patient_id=patients[1].id,
                store_id=stores[1].id,
                operator_id=users[2].id,
                status="completed",
                dr_grade_left=0,
                dr_grade_right=0,
                dr_confidence_left=0.95,
                dr_confidence_right=0.97,
                overall_risk="low",
                referral_required=False,
                raw_results={"analysis_version": "v2", "model": "efficientnet_b3"},
                completed_at=datetime(2026, 3, 16, 14, 0, 0, tzinfo=timezone.utc),
            ),
            Screening(
                id=uuid.uuid4(),
                patient_id=patients[2].id,
                store_id=stores[2].id,
                operator_id=users[1].id,
                status="analyzing",
                notes="Patient reported blurry vision in right eye.",
            ),
            Screening(
                id=uuid.uuid4(),
                patient_id=patients[3].id,
                store_id=stores[0].id,
                operator_id=users[1].id,
                status="images_uploaded",
                notes="Routine check-up, no complaints.",
            ),
            Screening(
                id=uuid.uuid4(),
                patient_id=patients[4].id,
                store_id=stores[1].id,
                operator_id=users[2].id,
                status="created",
                notes="First screening for this patient.",
            ),
        ]
        for s in screenings:
            db.add(s)
        await db.flush()
        print(f"[+] Created {len(screenings)} screenings.")

        # Commit everything
        await db.commit()
        print("\n[OK] Database seeded successfully.")
        print("     Stores: Mumbai, Delhi, Bangalore")
        print("     Users:  admin@netra.ai, operator1@netra.ai, operator2@netra.ai,")
        print("             reviewer1@netra.ai, reviewer2@netra.ai")
        print("     Password: netra2024")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(seed())
