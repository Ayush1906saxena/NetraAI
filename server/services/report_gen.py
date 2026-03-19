"""PDF report generation using Jinja2 + WeasyPrint with QR codes."""

import io
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from server.config import settings

logger = logging.getLogger(__name__)

# DR grade human-readable labels
DR_LABELS = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}

RISK_COLORS = {
    "low": "#28a745",
    "moderate": "#ffc107",
    "high": "#fd7e14",
    "urgent": "#dc3545",
}


class ReportGenerator:
    """Generate screening report PDFs."""

    def __init__(self):
        self._template_dir = settings.REPORT_TEMPLATE_DIR

    async def generate(
        self,
        screening: Any,
        language: str = "en",
        include_gradcam: bool = True,
        db: AsyncSession | None = None,
    ) -> tuple[bytes, str]:
        """Generate a PDF report. Returns (pdf_bytes, filename)."""
        # Gather context
        context = await self._build_context(screening, include_gradcam, db)

        html_content = self._render_html(context, language)
        pdf_bytes = self._html_to_pdf(html_content)
        filename = f"netra_report_{screening.id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.pdf"

        logger.info("Generated report PDF: %s (%d bytes)", filename, len(pdf_bytes))
        return pdf_bytes, filename

    async def _build_context(
        self,
        screening: Any,
        include_gradcam: bool,
        db: AsyncSession | None,
    ) -> dict:
        """Build template context from screening data."""
        context = {
            "screening_id": str(screening.id),
            "screened_at": screening.screened_at.strftime("%d %B %Y, %I:%M %p") if screening.screened_at else "N/A",
            "status": screening.status,
            "overall_risk": screening.overall_risk or "N/A",
            "risk_color": RISK_COLORS.get(screening.overall_risk or "", "#6c757d"),
            "referral_required": screening.referral_required,
            "referral_urgency": screening.referral_urgency,
            "referral_reason": screening.referral_reason,
            "dr_grade_left": DR_LABELS.get(screening.dr_grade_left, "N/A"),
            "dr_grade_right": DR_LABELS.get(screening.dr_grade_right, "N/A"),
            "dr_confidence_left": f"{(screening.dr_confidence_left or 0) * 100:.1f}%",
            "dr_confidence_right": f"{(screening.dr_confidence_right or 0) * 100:.1f}%",
            "glaucoma_prob_left": f"{(screening.glaucoma_prob_left or 0) * 100:.1f}%",
            "glaucoma_prob_right": f"{(screening.glaucoma_prob_right or 0) * 100:.1f}%",
            "amd_prob_left": f"{(screening.amd_prob_left or 0) * 100:.1f}%",
            "amd_prob_right": f"{(screening.amd_prob_right or 0) * 100:.1f}%",
            "notes": screening.notes or "",
            "generated_at": datetime.now(timezone.utc).strftime("%d %B %Y, %I:%M %p UTC"),
            "app_name": settings.APP_NAME,
            "app_version": settings.APP_VERSION,
        }

        # Patient info
        if screening.patient:
            context.update({
                "patient_name": screening.patient.full_name,
                "patient_age": screening.patient.age or "N/A",
                "patient_gender": screening.patient.gender or "N/A",
                "patient_phone": screening.patient.phone or "N/A",
                "patient_abha": screening.patient.abha_id or "N/A",
                "is_diabetic": "Yes" if screening.patient.is_diabetic else "No" if screening.patient.is_diabetic is False else "Unknown",
            })

        # Store info
        if screening.store:
            context.update({
                "store_name": screening.store.name,
                "store_code": screening.store.code,
                "store_city": screening.store.city or "",
            })

        # QR code (base64 encoded)
        context["qr_code_b64"] = self._generate_qr_code(str(screening.id))

        return context

    def _render_html(self, context: dict, language: str) -> str:
        """Render Jinja2 template to HTML string."""
        try:
            from jinja2 import Environment, FileSystemLoader

            env = Environment(loader=FileSystemLoader(self._template_dir))
            template_name = f"report_{language}.html"
            try:
                template = env.get_template(template_name)
            except Exception:
                template = env.get_template("report_en.html")
            return template.render(**context)
        except Exception as e:
            logger.warning("Template rendering failed, using inline HTML: %s", e)
            return self._fallback_html(context)

    def _fallback_html(self, ctx: dict) -> str:
        """Fallback inline HTML when template files are not available."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: 'Helvetica Neue', Arial, sans-serif; margin: 40px; color: #333; }}
                .header {{ text-align: center; border-bottom: 3px solid #2c5282; padding-bottom: 20px; }}
                .header h1 {{ color: #2c5282; margin: 0; font-size: 28px; }}
                .header p {{ color: #666; margin: 5px 0; }}
                .section {{ margin: 25px 0; }}
                .section h2 {{ color: #2c5282; font-size: 18px; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                td, th {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
                th {{ background: #f7fafc; font-weight: 600; width: 40%; }}
                .risk-badge {{ display: inline-block; padding: 6px 16px; border-radius: 20px; color: white;
                               font-weight: bold; font-size: 16px; background: {ctx.get('risk_color', '#6c757d')}; }}
                .referral {{ background: #fff5f5; border: 1px solid #feb2b2; border-radius: 8px; padding: 15px; margin: 15px 0; }}
                .footer {{ text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #e2e8f0;
                           font-size: 12px; color: #999; }}
                .qr {{ text-align: center; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{ctx.get('app_name', 'NetraAI')} Screening Report</h1>
                <p>Report ID: {ctx.get('screening_id', '')}</p>
                <p>Generated: {ctx.get('generated_at', '')}</p>
            </div>

            <div class="section">
                <h2>Patient Information</h2>
                <table>
                    <tr><th>Name</th><td>{ctx.get('patient_name', 'N/A')}</td></tr>
                    <tr><th>Age / Gender</th><td>{ctx.get('patient_age', 'N/A')} / {ctx.get('patient_gender', 'N/A')}</td></tr>
                    <tr><th>Phone</th><td>{ctx.get('patient_phone', 'N/A')}</td></tr>
                    <tr><th>ABHA ID</th><td>{ctx.get('patient_abha', 'N/A')}</td></tr>
                    <tr><th>Diabetic</th><td>{ctx.get('is_diabetic', 'Unknown')}</td></tr>
                </table>
            </div>

            <div class="section">
                <h2>Screening Location</h2>
                <table>
                    <tr><th>Store</th><td>{ctx.get('store_name', 'N/A')} ({ctx.get('store_code', '')})</td></tr>
                    <tr><th>City</th><td>{ctx.get('store_city', 'N/A')}</td></tr>
                    <tr><th>Screened At</th><td>{ctx.get('screened_at', 'N/A')}</td></tr>
                </table>
            </div>

            <div class="section">
                <h2>AI Analysis Results</h2>
                <p>Overall Risk: <span class="risk-badge">{ctx.get('overall_risk', 'N/A').upper()}</span></p>
                <table>
                    <tr><th></th><th>Left Eye</th><th>Right Eye</th></tr>
                    <tr><td><strong>DR Grade</strong></td><td>{ctx.get('dr_grade_left', 'N/A')}</td><td>{ctx.get('dr_grade_right', 'N/A')}</td></tr>
                    <tr><td><strong>DR Confidence</strong></td><td>{ctx.get('dr_confidence_left', 'N/A')}</td><td>{ctx.get('dr_confidence_right', 'N/A')}</td></tr>
                    <tr><td><strong>Glaucoma Risk</strong></td><td>{ctx.get('glaucoma_prob_left', 'N/A')}</td><td>{ctx.get('glaucoma_prob_right', 'N/A')}</td></tr>
                    <tr><td><strong>AMD Risk</strong></td><td>{ctx.get('amd_prob_left', 'N/A')}</td><td>{ctx.get('amd_prob_right', 'N/A')}</td></tr>
                </table>
            </div>

            {"<div class='referral section'><h2>Referral Recommendation</h2><p><strong>Urgency:</strong> " + (ctx.get('referral_urgency') or 'N/A') + "</p><p><strong>Reason:</strong> " + (ctx.get('referral_reason') or 'N/A') + "</p></div>" if ctx.get('referral_required') else ""}

            {f'<div class="qr"><img src="data:image/png;base64,{ctx.get("qr_code_b64", "")}" width="120" height="120" /><p>Scan to view results online</p></div>' if ctx.get('qr_code_b64') else ''}

            <div class="footer">
                <p>This report was generated by {ctx.get('app_name', 'NetraAI')} v{ctx.get('app_version', '')} and is intended as a screening aid only.</p>
                <p>It does not constitute a medical diagnosis. Please consult an ophthalmologist for a definitive assessment.</p>
            </div>
        </body>
        </html>
        """

    def _html_to_pdf(self, html: str) -> bytes:
        """Convert HTML to PDF bytes using WeasyPrint."""
        try:
            from weasyprint import HTML

            pdf_bytes = HTML(string=html).write_pdf()
            return pdf_bytes
        except ImportError:
            logger.warning("WeasyPrint not installed — returning HTML as bytes")
            return html.encode("utf-8")
        except Exception as e:
            logger.error("PDF generation failed: %s", e)
            return html.encode("utf-8")

    def _generate_qr_code(self, data: str) -> str:
        """Generate a QR code as base64-encoded PNG."""
        try:
            import base64

            import qrcode
            from qrcode.image.pil import PilImage

            qr = qrcode.QRCode(version=1, box_size=4, border=2)
            qr.add_data(data)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("ascii")
        except ImportError:
            logger.warning("qrcode library not installed — QR code skipped")
            return ""
        except Exception as e:
            logger.warning("QR code generation failed: %s", e)
            return ""
