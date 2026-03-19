"""ABDM (Ayushman Bharat Digital Mission) client for FHIR health record integration."""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx

from server.config import settings

logger = logging.getLogger(__name__)


class ABDMClient:
    """Client for interacting with the ABDM gateway."""

    def __init__(self):
        self.base_url = settings.ABDM_BASE_URL
        self.client_id = settings.ABDM_CLIENT_ID
        self.client_secret = settings.ABDM_CLIENT_SECRET
        self.auth_url = settings.ABDM_AUTH_URL
        self.hip_id = settings.ABDM_HIP_ID
        self.hip_name = settings.ABDM_HIP_NAME
        self.callback_url = settings.ABDM_CALLBACK_URL
        self._access_token: str | None = None
        self._token_expiry: datetime | None = None

    async def _get_token(self) -> str:
        """Obtain or refresh the ABDM gateway access token."""
        if self._access_token and self._token_expiry and datetime.now(timezone.utc) < self._token_expiry:
            return self._access_token

        payload = {
            "clientId": self.client_id,
            "clientSecret": self.client_secret,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(self.auth_url, json=payload)
            response.raise_for_status()
            data = response.json()

        self._access_token = data.get("accessToken", "")
        # Token typically valid for 20 minutes
        from datetime import timedelta
        self._token_expiry = datetime.now(timezone.utc) + timedelta(minutes=18)
        logger.info("ABDM access token refreshed")
        return self._access_token

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        """Make an authenticated request to the ABDM gateway."""
        token = await self._get_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "X-CM-ID": "sbx",
            "Content-Type": "application/json",
        }
        headers.update(kwargs.pop("headers", {}))

        url = f"{self.base_url}{path}"
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.request(method, url, headers=headers, **kwargs)
            if response.status_code == 202:
                return {"status": "accepted", "code": 202}
            response.raise_for_status()
            return response.json()

    async def verify_abha(self, abha_id: str) -> dict:
        """Verify an ABHA (Ayushman Bharat Health Account) ID."""
        try:
            result = await self._request(
                "POST",
                "/v0.5/users/auth/fetch-modes",
                json={
                    "requestId": str(uuid.uuid4()),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "query": {
                        "id": abha_id,
                        "purpose": "KYC_AND_LINK",
                        "requester": {
                            "type": "HIP",
                            "id": self.hip_id,
                        },
                    },
                },
            )
            return {"verified": True, "data": result}
        except Exception as e:
            logger.error("ABHA verification failed for %s: %s", abha_id, e)
            return {"verified": False, "error": str(e)}

    async def push_health_record(
        self,
        patient_abha_id: str,
        screening_data: dict,
        care_context_reference: str,
    ) -> dict:
        """Push a health record (screening result) to ABDM."""
        fhir_bundle = self._build_fhir_bundle(screening_data)

        try:
            result = await self._request(
                "POST",
                "/v0.5/health-information/hip/on-request",
                json={
                    "requestId": str(uuid.uuid4()),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "hiRequest": {
                        "transactionId": str(uuid.uuid4()),
                        "entries": [
                            {
                                "content": fhir_bundle,
                                "media": "application/fhir+json",
                                "checksum": "",
                                "careContextReference": care_context_reference,
                            }
                        ],
                    },
                },
            )
            logger.info("Health record pushed for ABHA %s", patient_abha_id)
            return {"pushed": True, "data": result}
        except Exception as e:
            logger.error("Health record push failed for %s: %s", patient_abha_id, e)
            return {"pushed": False, "error": str(e)}

    async def link_care_context(
        self,
        patient_abha_id: str,
        screening_id: str,
        display_name: str = "Eye Screening Report",
    ) -> dict:
        """Link a care context (screening) to the patient's ABHA."""
        try:
            result = await self._request(
                "POST",
                "/v0.5/links/link/add-contexts",
                json={
                    "requestId": str(uuid.uuid4()),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "link": {
                        "accessToken": "",  # populated from consent flow
                        "patient": {
                            "referenceNumber": patient_abha_id,
                            "display": "Patient",
                            "careContexts": [
                                {
                                    "referenceNumber": screening_id,
                                    "display": display_name,
                                }
                            ],
                        },
                    },
                },
            )
            logger.info("Care context linked for ABHA %s, screening %s", patient_abha_id, screening_id)
            return {"linked": True, "data": result}
        except Exception as e:
            logger.error("Care context linking failed: %s", e)
            return {"linked": False, "error": str(e)}

    def _build_fhir_bundle(self, screening_data: dict) -> dict:
        """Build a minimal FHIR DiagnosticReport bundle from screening data."""
        now = datetime.now(timezone.utc).isoformat()
        bundle_id = str(uuid.uuid4())

        observations = []
        obs_refs = []

        # DR observations
        for eye in ("left", "right"):
            grade_key = f"dr_grade_{eye}"
            if grade_key in screening_data and screening_data[grade_key] is not None:
                obs_id = str(uuid.uuid4())
                obs_refs.append({"reference": f"Observation/{obs_id}"})
                observations.append({
                    "resourceType": "Observation",
                    "id": obs_id,
                    "status": "final",
                    "code": {
                        "coding": [
                            {
                                "system": "http://snomed.info/sct",
                                "code": "59276001",
                                "display": f"Diabetic retinopathy ({eye} eye)",
                            }
                        ]
                    },
                    "valueInteger": screening_data[grade_key],
                    "effectiveDateTime": now,
                })

        # Glaucoma observations
        for eye in ("left", "right"):
            prob_key = f"glaucoma_prob_{eye}"
            if prob_key in screening_data and screening_data[prob_key] is not None:
                obs_id = str(uuid.uuid4())
                obs_refs.append({"reference": f"Observation/{obs_id}"})
                observations.append({
                    "resourceType": "Observation",
                    "id": obs_id,
                    "status": "final",
                    "code": {
                        "coding": [
                            {
                                "system": "http://snomed.info/sct",
                                "code": "23986001",
                                "display": f"Glaucoma screening ({eye} eye)",
                            }
                        ]
                    },
                    "valueQuantity": {
                        "value": round(screening_data[prob_key], 4),
                        "unit": "probability",
                    },
                    "effectiveDateTime": now,
                })

        # DiagnosticReport
        report = {
            "resourceType": "DiagnosticReport",
            "id": str(uuid.uuid4()),
            "status": "final",
            "code": {
                "coding": [
                    {
                        "system": "http://snomed.info/sct",
                        "code": "252886007",
                        "display": "Retinal screening",
                    }
                ]
            },
            "effectiveDateTime": now,
            "conclusion": screening_data.get("referral_reason", "Screening completed"),
            "result": obs_refs,
        }

        entries = [{"resource": report, "fullUrl": f"DiagnosticReport/{report['id']}"}]
        for obs in observations:
            entries.append({"resource": obs, "fullUrl": f"Observation/{obs['id']}"})

        return {
            "resourceType": "Bundle",
            "id": bundle_id,
            "type": "collection",
            "timestamp": now,
            "entry": entries,
        }
