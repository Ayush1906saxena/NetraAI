"""WhatsApp and SMS notification service using Gupshup API."""

import logging
from typing import Any

import httpx

from server.config import settings

logger = logging.getLogger(__name__)


class WhatsAppService:
    """Send WhatsApp messages and SMS via Gupshup."""

    def __init__(self):
        self.api_key = settings.GUPSHUP_API_KEY
        self.app_name = settings.GUPSHUP_APP_NAME
        self.base_url = settings.GUPSHUP_BASE_URL
        self.source_phone = settings.GUPSHUP_SOURCE_PHONE
        self.sms_url = settings.GUPSHUP_SMS_URL
        self.sms_userid = settings.GUPSHUP_SMS_USERID
        self.sms_password = settings.GUPSHUP_SMS_PASSWORD

    async def send_template_message(
        self,
        phone: str,
        template_id: str,
        params: list[str] | None = None,
    ) -> dict:
        """Send a WhatsApp template message."""
        phone = self._normalize_phone(phone)
        payload = {
            "channel": "whatsapp",
            "source": self.source_phone,
            "destination": phone,
            "src.name": self.app_name,
            "template": f'{{"id":"{template_id}","params":{params or []}}}',
        }
        return await self._wa_request(payload)

    async def send_text_message(self, phone: str, text: str) -> dict:
        """Send a plain WhatsApp text message."""
        phone = self._normalize_phone(phone)
        payload = {
            "channel": "whatsapp",
            "source": self.source_phone,
            "destination": phone,
            "src.name": self.app_name,
            "message": f'{{"type":"text","text":"{text}"}}',
        }
        return await self._wa_request(payload)

    async def send_document_message(
        self,
        phone: str,
        document_url: str,
        filename: str,
        caption: str = "",
    ) -> dict:
        """Send a WhatsApp document (PDF, etc.)."""
        phone = self._normalize_phone(phone)
        import json

        message = json.dumps({
            "type": "document",
            "url": document_url,
            "filename": filename,
            "caption": caption,
        })
        payload = {
            "channel": "whatsapp",
            "source": self.source_phone,
            "destination": phone,
            "src.name": self.app_name,
            "message": message,
        }
        return await self._wa_request(payload)

    async def send_report(self, phone: str, report_url: str, report: Any = None) -> dict:
        """Convenience method to send a screening report via WhatsApp."""
        filename = report.filename if report and hasattr(report, "filename") else "screening_report.pdf"
        caption = "Your NetraAI eye screening report is ready. Please consult an ophthalmologist if a referral is recommended."
        return await self.send_document_message(
            phone=phone,
            document_url=report_url,
            filename=filename,
            caption=caption,
        )

    async def send_sms(self, phone: str, message: str) -> dict:
        """Send an SMS via Gupshup SMS gateway."""
        phone = self._normalize_phone(phone)
        params = {
            "method": "SendMessage",
            "send_to": phone,
            "msg": message,
            "msg_type": "TEXT",
            "userid": self.sms_userid,
            "auth_scheme": "plain",
            "password": self.sms_password,
            "v": "1.1",
            "format": "json",
        }
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(self.sms_url, params=params)
                result = response.json() if response.headers.get("content-type", "").startswith("application/json") else {"raw": response.text}
                logger.info("SMS sent to %s: %s", phone, result)
                return result
        except Exception as e:
            logger.error("SMS send failed to %s: %s", phone, e)
            return {"error": str(e)}

    async def _wa_request(self, payload: dict) -> dict:
        """Make a request to the Gupshup WhatsApp API."""
        headers = {
            "apikey": self.api_key,
            "Content-Type": "application/x-www-form-urlencoded",
        }
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{self.base_url}/msg",
                    data=payload,
                    headers=headers,
                )
                result = response.json()
                logger.info("WhatsApp API response: %s", result)
                return result
        except Exception as e:
            logger.error("WhatsApp API request failed: %s", e)
            return {"error": str(e)}

    @staticmethod
    def _normalize_phone(phone: str) -> str:
        """Normalize phone number to E.164-like format."""
        phone = phone.strip().replace(" ", "").replace("-", "")
        if not phone.startswith("+") and not phone.startswith("91"):
            phone = "91" + phone
        phone = phone.lstrip("+")
        return phone
