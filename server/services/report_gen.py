"""PDF report generation using Jinja2 + WeasyPrint with QR codes."""

import base64
import io
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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

# Resolve the actual templates directory (server/templates/)
_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


class ReportGenerator:
    """Generate screening report PDFs."""

    def __init__(self):
        self._template_dir = settings.REPORT_TEMPLATE_DIR
        # Resolve the real template directory for the Jinja2 templates
        self._real_template_dir = str(_TEMPLATES_DIR)

    # ------------------------------------------------------------------
    # Original DB-backed generate (unchanged, for full app usage)
    # ------------------------------------------------------------------
    async def generate(
        self,
        screening: Any,
        language: str = "en",
        include_gradcam: bool = True,
        db: Any = None,
    ) -> tuple[bytes, str]:
        """Generate a PDF report from a DB screening object. Returns (pdf_bytes, filename)."""
        context = await self._build_context(screening, include_gradcam, db)
        html_content = self._render_html(context, language)
        pdf_bytes = self._html_to_pdf(html_content)
        filename = f"netra_report_{screening.id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.pdf"
        logger.info("Generated report PDF: %s (%d bytes)", filename, len(pdf_bytes))
        return pdf_bytes, filename

    # ------------------------------------------------------------------
    # New: generate from raw inference results (demo / standalone)
    # ------------------------------------------------------------------
    async def generate_from_inference(
        self,
        inference_result: dict[str, Any],
        fundus_image_bytes: bytes | None = None,
        quality_result: dict[str, Any] | None = None,
        patient_name: str = "Demo Patient",
        patient_age: int | None = None,
        patient_gender: str | None = None,
        language: str = "en",
    ) -> tuple[bytes, str, str]:
        """
        Generate a report from InferenceService.analyze_fundus() output.

        Returns:
            (output_bytes, filename, content_type)
            content_type is 'application/pdf' if WeasyPrint is available,
            otherwise 'text/html'.
        """
        report_id = str(uuid.uuid4())[:12].upper()
        now = datetime.now(timezone.utc)

        dr = inference_result.get("dr", {})
        referral = inference_result.get("referral", {})
        gradcam = inference_result.get("gradcam", {})
        grade = dr.get("grade", 0)
        confidence = dr.get("confidence", 0.0)

        # Map risk level to template risk classes
        risk_level = referral.get("risk_level", "minimal")
        risk_map = {
            "minimal": "low",
            "low": "low",
            "moderate": "moderate",
            "high": "high",
            "critical": "urgent",
        }
        overall_risk = risk_map.get(risk_level, "low")

        # Encode the original fundus image as base64 for embedding
        fundus_b64 = ""
        if fundus_image_bytes:
            fundus_b64 = base64.b64encode(fundus_image_bytes).decode("ascii")

        gradcam_b64 = gradcam.get("overlay_png_base64", "")
        iqa_score = None
        iqa_passed = None
        if quality_result:
            iqa_score = quality_result.get("score")
            iqa_passed = quality_result.get("passed")

        # Build context that matches the Jinja2 template variables
        # The template expects `patient` and `store` as objects, and
        # `screening` as an object with DR fields for both eyes.
        # For demo mode, we use the same image for "left eye" and mark
        # right eye as not analysed.
        context = self._build_demo_context(
            report_id=report_id,
            now=now,
            grade=grade,
            confidence=confidence,
            overall_risk=overall_risk,
            referral=referral,
            fundus_b64=fundus_b64,
            gradcam_b64=gradcam_b64,
            iqa_score=iqa_score,
            iqa_passed=iqa_passed,
            patient_name=patient_name,
            patient_age=patient_age,
            patient_gender=patient_gender,
            dr_info=dr,
        )

        # Try the full Jinja2 template first, fall back to inline HTML
        html_content = self._render_screening_html(context, language)
        output_bytes = self._html_to_pdf(html_content)

        # Detect whether we got a real PDF or HTML fallback
        is_pdf = output_bytes[:5] == b"%PDF-"
        content_type = "application/pdf" if is_pdf else "text/html"
        ext = "pdf" if is_pdf else "html"
        filename = f"netra_report_{report_id}_{now.strftime('%Y%m%d_%H%M%S')}.{ext}"

        logger.info("Generated demo report: %s (%d bytes, %s)", filename, len(output_bytes), content_type)
        return output_bytes, filename, content_type

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_demo_context(
        self,
        report_id: str,
        now: datetime,
        grade: int,
        confidence: float,
        overall_risk: str,
        referral: dict,
        fundus_b64: str,
        gradcam_b64: str,
        iqa_score: float | None,
        iqa_passed: bool | None,
        patient_name: str,
        patient_age: int | None,
        patient_gender: str | None,
        dr_info: dict,
    ) -> dict:
        """Build the full template context for demo / inference-based reports."""
        # The screening_report_en.html template expects object-like access
        # for patient, screening, store — we use SimpleNamespace.
        patient = SimpleNamespace(
            full_name=patient_name,
            age=patient_age,
            gender=patient_gender,
            abha_id=None,
            is_diabetic=None,
            has_hypertension=None,
            diabetes_duration_years=None,
        )

        screening = SimpleNamespace(
            dr_grade_left=grade,
            dr_confidence_left=confidence,
            glaucoma_prob_left=None,
            amd_prob_left=None,
            dr_grade_right=None,
            dr_confidence_right=None,
            glaucoma_prob_right=None,
            amd_prob_right=None,
        )

        store = SimpleNamespace(
            name="Demo Screening",
            city="",
            code="DEMO",
        )

        # Build findings list from DR result
        findings = []
        if grade > 0:
            severity_map = {1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}
            findings.append({
                "condition": "Diabetic Retinopathy",
                "severity": severity_map.get(grade, "Unknown"),
                "description": dr_info.get("description", DR_LABELS.get(grade, "")),
            })

        is_referable = referral.get("is_referable", False)
        urgency = referral.get("urgency", "none")
        # Map inference urgency to template urgency values
        urgency_map = {
            "emergency": "emergency",
            "urgent": "urgent",
            "soon": "routine",
            "routine": "routine",
            "none": "routine",
        }
        template_urgency = urgency_map.get(urgency, "routine")

        return {
            # Header
            "report_id": report_id,
            "screening_date": now.strftime("%d %B %Y, %I:%M %p UTC"),
            "overall_risk": overall_risk,
            # Patient (object)
            "patient": patient,
            # Store (object)
            "store": store,
            # Screening (object)
            "screening": screening,
            # Eye images (base64)
            "left_fundus_base64": fundus_b64,
            "left_gradcam_base64": gradcam_b64,
            "left_iqa_score": iqa_score,
            "left_iqa_passed": iqa_passed,
            "right_fundus_base64": "",
            "right_gradcam_base64": "",
            "right_iqa_score": None,
            "right_iqa_passed": None,
            # Findings
            "findings": findings,
            # Referral
            "referral_required": is_referable,
            "referral_urgency": template_urgency,
            "referral_reason": referral.get("recommendation", ""),
            "referral_message": referral.get("recommendation", ""),
            # Footer
            "screening_id": report_id,
            "generated_at": now.strftime("%d %B %Y, %I:%M %p UTC"),
            "app_version": settings.APP_VERSION,
            "app_name": settings.APP_NAME,
            "qr_code_base64": self._generate_qr_code(report_id),
            # Logo
            "logo_base64": "",
        }

    def _render_screening_html(self, context: dict, language: str) -> str:
        """Render the screening_report Jinja2 template with includes."""
        try:
            from jinja2 import Environment, FileSystemLoader

            env = Environment(loader=FileSystemLoader(self._real_template_dir))
            template_name = f"screening_report_{language}.html"
            try:
                template = env.get_template(template_name)
            except Exception:
                template = env.get_template("screening_report_en.html")
            return template.render(**context)
        except Exception as e:
            logger.warning("Template rendering failed, using fallback HTML: %s", e)
            return self._fallback_demo_html(context)

    def _fallback_demo_html(self, ctx: dict) -> str:
        """Fallback inline HTML for demo reports when template files are not available."""
        patient = ctx.get("patient", SimpleNamespace(full_name="N/A", age=None, gender=None))
        grade = getattr(ctx.get("screening"), "dr_grade_left", None)
        confidence = getattr(ctx.get("screening"), "dr_confidence_left", 0)
        grade_name = DR_LABELS.get(grade, "N/A") if grade is not None else "N/A"
        risk = ctx.get("overall_risk", "low")
        risk_color = RISK_COLORS.get(risk, "#6c757d")
        findings = ctx.get("findings", [])

        findings_html = ""
        if findings:
            rows = "".join(
                f"<tr><td><strong>{f['condition']}</strong></td><td>{f['severity']}</td><td>{f['description']}</td></tr>"
                for f in findings
            )
            findings_html = f"""
            <table>
                <tr><th>Condition</th><th>Severity</th><th>Description</th></tr>
                {rows}
            </table>"""
        else:
            findings_html = '<p style="color:#166534;background:#dcfce7;padding:10px;border-radius:6px;text-align:center;">No significant findings detected.</p>'

        fundus_img = ""
        if ctx.get("left_fundus_base64"):
            fundus_img = f'<img src="data:image/png;base64,{ctx["left_fundus_base64"]}" style="max-width:200px;border-radius:8px;border:1px solid #ccc;" />'

        gradcam_img = ""
        if ctx.get("left_gradcam_base64"):
            gradcam_img = f'<img src="data:image/png;base64,{ctx["left_gradcam_base64"]}" style="max-width:200px;border-radius:8px;border:1px solid #ccc;" />'

        referral_html = ""
        if ctx.get("referral_required"):
            referral_html = f"""
            <div class="section" style="background:#fff5f5;border:1px solid #feb2b2;border-radius:8px;padding:15px;">
                <h2>Referral Required</h2>
                <p><strong>Urgency:</strong> {ctx.get('referral_urgency', 'N/A').upper()}</p>
                <p>{ctx.get('referral_reason', '')}</p>
            </div>"""
        else:
            referral_html = """
            <div class="section" style="background:#dcfce7;border:1px solid #86efac;border-radius:8px;padding:15px;">
                <p>No immediate referral required. Routine follow-up in 12 months.</p>
            </div>"""

        qr_html = ""
        qr_b64 = ctx.get("qr_code_base64", "")
        if qr_b64:
            qr_html = f'<div style="text-align:center;margin:20px 0;"><img src="data:image/png;base64,{qr_b64}" width="80" height="80" /></div>'

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        @page {{ size: A4; margin: 15mm; }}
        body {{ font-family: 'Helvetica Neue', Arial, sans-serif; margin: 40px; color: #333; font-size: 11pt; }}
        .header {{ text-align: center; border-bottom: 3px solid #0052cc; padding-bottom: 20px; margin-bottom: 20px; }}
        .header h1 {{ color: #0052cc; margin: 0; font-size: 24px; }}
        .header p {{ color: #666; margin: 5px 0; font-size: 10pt; }}
        .section {{ margin: 20px 0; }}
        .section h2 {{ color: #0052cc; font-size: 16px; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        td, th {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background: #f7fafc; font-weight: 600; }}
        .risk-badge {{ display: inline-block; padding: 6px 16px; border-radius: 20px; color: white;
                       font-weight: bold; font-size: 14px; background: {risk_color}; }}
        .images {{ display: flex; gap: 20px; justify-content: center; margin: 15px 0; }}
        .img-box {{ text-align: center; }}
        .img-box p {{ font-size: 9pt; color: #666; margin-top: 4px; }}
        .footer {{ text-align: center; margin-top: 30px; padding-top: 15px; border-top: 1px solid #e2e8f0;
                   font-size: 9pt; color: #999; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>NetraAI Eye Screening Report</h1>
        <p>Report ID: {ctx.get('report_id', '')}</p>
        <p>{ctx.get('generated_at', '')}</p>
    </div>

    <div class="section">
        <h2>Patient Information</h2>
        <table>
            <tr><th style="width:35%;">Name</th><td>{patient.full_name}</td></tr>
            <tr><th>Age</th><td>{patient.age or 'N/A'}</td></tr>
            <tr><th>Gender</th><td>{patient.gender or 'N/A'}</td></tr>
        </table>
    </div>

    <div class="section">
        <h2>AI Analysis Results</h2>
        <p>Overall Risk: <span class="risk-badge">{risk.upper()}</span></p>
        <table>
            <tr><th style="width:35%;">DR Grade</th><td>{grade_name}</td></tr>
            <tr><th>Confidence</th><td>{confidence * 100:.1f}%</td></tr>
        </table>
    </div>

    <div class="images">
        <div class="img-box">{fundus_img}<p>Fundus Image</p></div>
        <div class="img-box">{gradcam_img}<p>GradCAM Heatmap</p></div>
    </div>

    <div class="section">
        <h2>Key Findings</h2>
        {findings_html}
    </div>

    {referral_html}
    {qr_html}

    <div class="footer">
        <p>Generated by {ctx.get('app_name', 'NetraAI')} v{ctx.get('app_version', '')} | This is an AI screening aid, not a medical diagnosis.</p>
    </div>
</body>
</html>"""

    # ------------------------------------------------------------------
    # Shared helpers (used by both paths)
    # ------------------------------------------------------------------

    async def _build_context(
        self,
        screening: Any,
        include_gradcam: bool,
        db: Any,
    ) -> dict:
        """Build template context from screening data (DB-backed)."""
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
        """Render Jinja2 template to HTML string (legacy path)."""
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
        """Fallback inline HTML when template files are not available (legacy)."""
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
                .footer {{ text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #e2e8f0;
                           font-size: 12px; color: #999; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{ctx.get('app_name', 'NetraAI')} Screening Report</h1>
                <p>Report ID: {ctx.get('screening_id', '')}</p>
                <p>Generated: {ctx.get('generated_at', '')}</p>
            </div>
            <div class="section">
                <h2>AI Analysis Results</h2>
                <p>Overall Risk: <span class="risk-badge">{ctx.get('overall_risk', 'N/A').upper()}</span></p>
            </div>
            <div class="footer">
                <p>Generated by {ctx.get('app_name', 'NetraAI')} v{ctx.get('app_version', '')}</p>
            </div>
        </body>
        </html>
        """

    def _html_to_pdf(self, html: str) -> bytes:
        """Convert HTML to PDF bytes using WeasyPrint, with HTML fallback."""
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
            import qrcode

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
