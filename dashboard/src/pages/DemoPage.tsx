import { useState } from "react";
import FundusUploader from "../components/FundusUploader";
import ResultCard from "../components/ResultCard";
import GradcamViewer from "../components/GradcamViewer";
import ProbabilityChart from "../components/ProbabilityChart";
import ReferralBanner from "../components/ReferralBanner";
import { analyzeImage, type AnalysisResult, type ConditionScreening } from "../api/client";

/* ─── Per-grade clinical content ─────────────────────────────────────── */

const GRADE_INFO: Record<number, {
  title: string;
  severity: string;
  gradient: string;
  gradientSubtle: string;
  textColor: string;
  accentColor: string;
  dotColor: string;
  what: string;
  meaning: string;
  action: string[];
  timeline: string;
  lifestyle: string[];
}> = {
  0: {
    title: "No Diabetic Retinopathy Detected",
    severity: "Healthy",
    gradient: "from-emerald-500 via-green-500 to-teal-500",
    gradientSubtle: "from-emerald-50 to-green-50",
    textColor: "text-emerald-800",
    accentColor: "text-emerald-600",
    dotColor: "bg-emerald-500",
    what: "Your retinal scan shows no signs of diabetic retinopathy. The blood vessels in your retina appear healthy with no visible damage from diabetes.",
    meaning: "This is great news! However, diabetes can still affect your eyes over time, so regular screening is essential.",
    action: [
      "Continue annual retinal screening",
      "Keep your blood sugar (HbA1c) below 7%",
      "Maintain regular check-ups with your diabetes doctor",
    ],
    timeline: "Next screening recommended in 12 months",
    lifestyle: [
      "Control blood sugar levels \u2014 this is the #1 way to prevent eye damage",
      "Keep blood pressure under 130/80 mmHg",
      "Don't skip your diabetes medications",
      "Exercise regularly (30 min/day, 5 days/week)",
      "Annual dilated eye exam even if this screening is normal",
    ],
  },
  1: {
    title: "Mild Non-Proliferative Diabetic Retinopathy",
    severity: "Mild",
    gradient: "from-lime-500 via-green-500 to-emerald-500",
    gradientSubtle: "from-lime-50 to-green-50",
    textColor: "text-green-800",
    accentColor: "text-green-600",
    dotColor: "bg-green-500",
    what: "Small balloon-like swellings (microaneurysms) were detected in the tiny blood vessels of your retina. This is the earliest stage of diabetic eye disease.",
    meaning: "At this stage, you likely have no vision problems. But this is an early warning sign that diabetes is starting to affect your eyes. With good diabetes control, this can stabilize or even improve.",
    action: [
      "See an eye doctor within 3 months for a detailed exam",
      "Focus on tightening blood sugar control (target HbA1c < 7%)",
      "Rescreen every 6 months to monitor for changes",
    ],
    timeline: "Follow-up screening in 6 months. Eye doctor visit within 3 months.",
    lifestyle: [
      "Strictly control blood sugar \u2014 this is the most important thing you can do",
      "Monitor and control blood pressure (target < 130/80)",
      "Reduce salt intake and manage cholesterol",
      "Stop smoking if applicable \u2014 smoking accelerates retinal damage",
      "Report any vision changes (blurriness, floaters) to your doctor immediately",
    ],
  },
  2: {
    title: "Moderate Non-Proliferative Diabetic Retinopathy",
    severity: "Moderate",
    gradient: "from-amber-500 via-orange-500 to-amber-600",
    gradientSubtle: "from-amber-50 to-orange-50",
    textColor: "text-orange-800",
    accentColor: "text-orange-600",
    dotColor: "bg-orange-500",
    what: "Some blood vessels in your retina are becoming blocked or swollen. There are more microaneurysms and possibly some small areas of bleeding (hemorrhages) or fluid leakage (exudates).",
    meaning: "The disease is progressing. While you may still have normal vision, the retina is under stress. Without intervention, there is a significant risk of progressing to more severe stages.",
    action: [
      "See an ophthalmologist within 1 month",
      "Get a comprehensive dilated eye exam with OCT scan",
      "Intensify diabetes management with your endocrinologist",
      "May need more frequent monitoring (every 3-4 months)",
    ],
    timeline: "Ophthalmologist visit within 1 month. Monitoring every 3-4 months.",
    lifestyle: [
      "Aggressive blood sugar control is critical \u2014 work with your doctor to optimize medications",
      "Blood pressure control is equally important at this stage",
      "Consider seeing a diabetes educator for lifestyle management",
      "Avoid heavy lifting or straining (can worsen retinal bleeding)",
      "Ensure adequate sleep and stress management",
      "Do NOT delay your ophthalmologist appointment",
    ],
  },
  3: {
    title: "Severe Non-Proliferative Diabetic Retinopathy",
    severity: "Severe",
    gradient: "from-red-500 via-rose-500 to-red-600",
    gradientSubtle: "from-red-50 to-rose-50",
    textColor: "text-red-800",
    accentColor: "text-red-600",
    dotColor: "bg-red-500",
    what: "Many blood vessels in your retina are blocked, depriving large areas of the retina of blood supply. The retina is sending distress signals to grow new blood vessels.",
    meaning: "This is a serious stage. There is a high risk (>50% within 1 year) of progressing to proliferative diabetic retinopathy, which can cause sudden vision loss. Urgent medical attention is needed.",
    action: [
      "See a retina specialist within 1 week",
      "You may need laser treatment (panretinal photocoagulation) or anti-VEGF injections",
      "Do NOT delay \u2014 timely treatment can prevent vision loss",
      "Frequent monitoring every 1-2 months",
    ],
    timeline: "URGENT: Retina specialist within 1 week.",
    lifestyle: [
      "This is a medical priority \u2014 treat it with the same urgency as a heart problem",
      "Strictly follow all prescribed medications",
      "Avoid activities that increase eye pressure (heavy lifting, straining, inverted positions)",
      "Report any sudden vision changes, floaters, or flashes of light IMMEDIATELY",
      "Keep all follow-up appointments \u2014 do not reschedule",
      "Emotional support is important \u2014 talk to family about your condition",
    ],
  },
  4: {
    title: "Proliferative Diabetic Retinopathy",
    severity: "Critical",
    gradient: "from-red-600 via-red-700 to-rose-800",
    gradientSubtle: "from-red-100 to-rose-100",
    textColor: "text-red-900",
    accentColor: "text-red-700",
    dotColor: "bg-red-700",
    what: "New, abnormal blood vessels are growing on the surface of your retina (neovascularization). These fragile new vessels can bleed into the eye (vitreous hemorrhage) or cause the retina to detach, leading to severe vision loss or blindness.",
    meaning: "This is the most advanced stage of diabetic retinopathy. Without treatment, there is a very high risk of significant and permanent vision loss. However, modern treatments are highly effective when started promptly.",
    action: [
      "EMERGENCY: See a retina specialist within 24-48 hours",
      "Treatment options include anti-VEGF injections, laser surgery, or vitrectomy",
      "Treatment can preserve and often improve vision if started early",
      "Will need ongoing treatment and monitoring",
    ],
    timeline: "EMERGENCY: Retina specialist within 24-48 hours.",
    lifestyle: [
      "Seek treatment immediately \u2014 every day matters at this stage",
      "Avoid any strenuous activity until seen by specialist",
      "If you experience sudden vision loss, flashing lights, or a 'curtain' over your vision, go to the emergency room",
      "You will likely need multiple treatment sessions \u2014 commit to the full treatment plan",
      "Blood sugar and blood pressure control remain critical even during treatment",
      "Seek support \u2014 organizations like the diabetic retinopathy foundation can help",
    ],
  },
};

/* ─── Loading animation ──────────────────────────────────────────────── */

function ScanAnimation() {
  return (
    <div className="flex flex-col items-center justify-center py-20">
      <div className="relative w-40 h-40">
        {/* Concentric rings */}
        {[0, 1, 2, 3].map((i) => (
          <div
            key={i}
            className="absolute inset-0 rounded-full border-2 border-indigo-300/30"
            style={{
              transform: `scale(${0.4 + i * 0.2})`,
              animation: `expandRing 2.5s ease-out ${i * 0.4}s infinite`,
            }}
          />
        ))}
        {/* Center eye */}
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-16 h-16 rounded-full gradient-brand flex items-center justify-center shadow-[0_0_30px_rgba(99,102,241,0.3)]"
               style={{ animation: "spinSlow 8s linear infinite" }}>
            <svg className="w-8 h-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            </svg>
          </div>
        </div>
        {/* Scanning line */}
        <div className="absolute inset-x-4 top-1/2 -translate-y-1/2 h-0.5 overflow-hidden">
          <div
            className="h-full w-full bg-gradient-to-r from-transparent via-indigo-400 to-transparent"
            style={{ animation: "scanLine 2s ease-in-out infinite" }}
          />
        </div>
      </div>
      <div className="mt-8 text-center">
        <p className="text-lg font-semibold text-gray-800">Analyzing retinal image</p>
        <p className="text-sm text-gray-400 mt-1">AI model is examining vascular patterns...</p>
        <div className="flex items-center justify-center gap-1.5 mt-4">
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              className="w-2 h-2 rounded-full bg-indigo-400"
              style={{
                animation: "float 1.2s ease-in-out infinite",
                animationDelay: `${i * 0.2}s`,
              }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

/* ─── Info card component ────────────────────────────────────────────── */

function InfoCard({ icon, title, children, className = "", delay = "" }: {
  icon: React.ReactNode;
  title: string;
  children: React.ReactNode;
  className?: string;
  delay?: string;
}) {
  return (
    <div className={`card-elevated p-6 fade-in-up ${delay} ${className}`}>
      <div className="flex items-center gap-3 mb-4">
        {icon}
        <h3 className="text-base font-semibold text-gray-900">{title}</h3>
      </div>
      {children}
    </div>
  );
}

/* ─── Expandable section ─────────────────────────────────────────────── */

function ExpandableSection({ title, icon, children, defaultOpen = false }: {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  return (
    <details className="card-elevated overflow-hidden group" open={defaultOpen || undefined}>
      <summary className="px-6 py-5 cursor-pointer flex items-center gap-3 hover:bg-gray-50/50 transition-colors select-none list-none [&::-webkit-details-marker]:hidden">
        {icon}
        <span className="text-base font-semibold text-gray-900 flex-1">{title}</span>
        <svg
          className="w-5 h-5 text-gray-400 transition-transform duration-300 group-open:rotate-180"
          fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </summary>
      <div className="px-6 pb-6 pt-1">{children}</div>
    </details>
  );
}

/* ─── Risk level badge helper ────────────────────────────────────────── */

const RISK_BADGE_STYLES: Record<string, string> = {
  none: "bg-gray-100 text-gray-600",
  low: "bg-emerald-100 text-emerald-700",
  moderate: "bg-amber-100 text-amber-700",
  high: "bg-red-100 text-red-700",
  very_high: "bg-red-200 text-red-800",
};

function RiskBadge({ level }: { level: string }) {
  const style = RISK_BADGE_STYLES[level] || RISK_BADGE_STYLES.none;
  const label = level === "very_high" ? "Very High" : level.charAt(0).toUpperCase() + level.slice(1);
  return (
    <span className={`inline-block px-2.5 py-1 rounded-lg text-xs font-semibold ${style}`}>
      {label}
    </span>
  );
}

/* ─── Condition screening row ────────────────────────────────────────── */

function ConditionRow({ condition }: { condition: ConditionScreening }) {
  return (
    <details className="group border border-gray-100 rounded-xl overflow-hidden">
      <summary className="px-5 py-4 cursor-pointer flex items-center gap-3 hover:bg-gray-50/50 transition-colors select-none list-none [&::-webkit-details-marker]:hidden">
        <RiskBadge level={condition.risk_level} />
        <span className="text-sm font-semibold text-gray-800 flex-1">{condition.condition_name}</span>
        <svg
          className="w-4 h-4 text-gray-400 transition-transform duration-300 group-open:rotate-180"
          fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </summary>
      <div className="px-5 pb-4 pt-1 space-y-3">
        <p className="text-xs text-gray-500 leading-relaxed">{condition.description}</p>
        {condition.findings.length > 0 && (
          <ul className="space-y-1.5">
            {condition.findings.map((f, i) => (
              <li key={i} className="text-sm text-gray-600 flex items-start gap-2">
                <span className="mt-1.5 w-1.5 h-1.5 rounded-full flex-shrink-0 bg-indigo-400" />
                {f}
              </li>
            ))}
          </ul>
        )}
        <div className="p-3 rounded-lg bg-indigo-50 border border-indigo-100">
          <p className="text-xs font-medium text-indigo-700">{condition.recommendation}</p>
        </div>
      </div>
    </details>
  );
}

/* ─── Page ───────────────────────────────────────────────────────────── */

export default function DemoPage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const handleFile = async (file: File) => {
    setLoading(true);
    setError(null);
    setResult(null);
    setPreviewUrl(URL.createObjectURL(file));

    try {
      const data = await analyzeImage(file);
      setResult(data);
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : "Analysis failed. Is the backend running?";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setResult(null);
    setPreviewUrl(null);
    setError(null);
  };

  const info = result ? GRADE_INFO[result.grade] ?? GRADE_INFO[0] : null;

  return (
    <div className="max-w-4xl mx-auto">

      {/* ── Hero / Upload state ──────────────────────────────────────── */}
      {!result && !loading && (
        <div className="fade-in-up">
          {/* Hero header */}
          <div className="text-center mb-10 pt-4">
            <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-indigo-50 border border-indigo-100 text-indigo-600 text-xs font-semibold mb-5 tracking-wide uppercase">
              <span className="w-1.5 h-1.5 rounded-full bg-indigo-500 animate-pulse" />
              AI-Powered Screening
            </div>
            <h1 className="text-4xl sm:text-5xl font-extrabold text-gray-900 tracking-tight leading-tight">
              Diabetic Retinopathy
              <br />
              <span className="bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                Screening
              </span>
            </h1>
            <p className="text-lg text-gray-500 mt-4 max-w-xl mx-auto leading-relaxed">
              Upload a fundus image and our AI will analyze it for signs of
              diabetic retinopathy in seconds.
            </p>
          </div>

          <FundusUploader onFileSelected={handleFile} disabled={loading} />
        </div>
      )}

      {/* ── Loading state ────────────────────────────────────────────── */}
      {loading && <ScanAnimation />}

      {/* ── Error state ──────────────────────────────────────────────── */}
      {error && (
        <div className="mt-6 card-elevated border-red-200 bg-red-50/50 p-5 text-sm fade-in-up">
          <div className="flex items-start gap-3">
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-red-100 flex items-center justify-center">
              <svg className="w-4 h-4 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </div>
            <div>
              <p className="font-semibold text-red-800">Analysis Failed</p>
              <p className="text-red-600 mt-1">{error}</p>
              <button
                onClick={reset}
                className="mt-3 text-sm font-medium text-red-600 hover:text-red-700 underline underline-offset-2"
              >
                Try again
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Results ──────────────────────────────────────────────────── */}
      {result && info && previewUrl && (
        <div className="space-y-6 pb-8">

          {/* Full-width gradient banner */}
          <div className={`relative overflow-hidden rounded-2xl bg-gradient-to-r ${info.gradient} p-8 sm:p-10 fade-in-up`}>
            {/* Decorative circles */}
            <div className="absolute top-0 right-0 w-64 h-64 rounded-full bg-white/10 -translate-y-1/3 translate-x-1/4" />
            <div className="absolute bottom-0 left-0 w-32 h-32 rounded-full bg-white/10 translate-y-1/2 -translate-x-1/4" />

            <div className="relative z-10">
              <div className="flex items-center gap-2 mb-3">
                <span className="px-3 py-1 rounded-full bg-white/20 text-white text-xs font-semibold backdrop-blur-sm">
                  DR Grade {result.grade} of 4
                </span>
                <span className="px-3 py-1 rounded-full bg-white/20 text-white text-xs font-semibold backdrop-blur-sm">
                  {info.severity}
                </span>
              </div>
              <h2 className="text-2xl sm:text-3xl font-bold text-white leading-snug max-w-2xl">
                {info.title}
              </h2>
              <p className="text-white/80 text-sm mt-2">
                AI Confidence: {(result.confidence * 100).toFixed(1)}%
              </p>
            </div>
          </div>

          {/* Plain English Summary */}
          {result.summary && (
            <div className="card-elevated p-5 bg-indigo-50/50 border border-indigo-100 fade-in-up">
              <div className="flex items-start gap-3">
                <div className="flex-shrink-0 w-9 h-9 rounded-xl bg-indigo-100 flex items-center justify-center">
                  <svg className="w-5 h-5 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-indigo-900 mb-1">Summary</h3>
                  <p className="text-sm text-indigo-800 leading-relaxed">{result.summary}</p>
                </div>
              </div>
            </div>
          )}

          {/* Confidence gauge + Referral side by side */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <ResultCard
              grade={result.grade}
              gradeLabel={result.grade_label}
              confidence={result.confidence}
            />
            {result.referral && result.referral.recommendation && (
              <ReferralBanner
                urgency={result.referral.urgency}
                recommendation={result.referral.recommendation}
              />
            )}
          </div>

          {/* Referable DR + Progression Risk side by side */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Referable DR Card */}
            {result.referable_dr && (
              <InfoCard
                icon={
                  <div className={`w-9 h-9 rounded-xl flex items-center justify-center ${
                    result.referable_dr.is_referable ? "bg-red-100" : "bg-emerald-100"
                  }`}>
                    <svg className={`w-5 h-5 ${result.referable_dr.is_referable ? "text-red-600" : "text-emerald-600"}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                      <path strokeLinecap="round" strokeLinejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                    </svg>
                  </div>
                }
                title="Referable DR"
              >
                <div className="space-y-3">
                  <div className="flex items-center gap-3">
                    <span className={`text-2xl font-bold ${
                      result.referable_dr.is_referable ? "text-red-600" : "text-emerald-600"
                    }`}>
                      {result.referable_dr.is_referable ? "Yes" : "No"}
                    </span>
                    <RiskBadge level={result.referable_dr.confidence_level === "high" ? (result.referable_dr.is_referable ? "high" : "low") : "moderate"} />
                  </div>
                  <div>
                    <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
                      <span>Referable Probability</span>
                      <span className="font-semibold text-gray-700">{(result.referable_dr.referable_probability * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full h-2 bg-gray-100 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${
                          result.referable_dr.is_referable
                            ? "bg-gradient-to-r from-red-400 to-red-600"
                            : "bg-gradient-to-r from-emerald-400 to-emerald-600"
                        }`}
                        style={{ width: `${Math.min(result.referable_dr.referable_probability * 100, 100)}%` }}
                      />
                    </div>
                  </div>
                  <p className="text-xs text-gray-500 leading-relaxed">{result.referable_dr.clinical_action}</p>
                </div>
              </InfoCard>
            )}

            {/* Progression Risk Card */}
            {result.progression && (
              <InfoCard
                icon={
                  <div className="w-9 h-9 rounded-xl bg-orange-100 flex items-center justify-center">
                    <svg className="w-5 h-5 text-orange-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                    </svg>
                  </div>
                }
                title="Progression Risk"
              >
                <div className="space-y-3">
                  <div className="flex items-center gap-3 mb-1">
                    <RiskBadge level={result.progression.risk_level} />
                    <span className="text-xs text-gray-500">
                      Rescreen in {result.progression.recommended_rescreen_months} months
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-3 rounded-lg bg-gray-50 text-center">
                      <p className="text-2xl font-bold text-gray-800">
                        {(result.progression.progression_risk_1yr * 100).toFixed(0)}%
                      </p>
                      <p className="text-xs text-gray-500 mt-0.5">1-Year Risk</p>
                    </div>
                    <div className="p-3 rounded-lg bg-gray-50 text-center">
                      <p className="text-2xl font-bold text-gray-800">
                        {(result.progression.progression_risk_5yr * 100).toFixed(0)}%
                      </p>
                      <p className="text-xs text-gray-500 mt-0.5">5-Year Risk</p>
                    </div>
                  </div>
                  {result.progression.risk_factors.length > 0 && (
                    <ul className="space-y-1">
                      {result.progression.risk_factors.slice(0, 3).map((factor, i) => (
                        <li key={i} className="text-xs text-gray-500 flex items-start gap-1.5">
                          <span className="mt-1 w-1 h-1 rounded-full flex-shrink-0 bg-orange-400" />
                          {factor}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </InfoCard>
            )}
          </div>

          {/* Three info cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            <InfoCard
              icon={
                <div className="w-9 h-9 rounded-xl bg-blue-100 flex items-center justify-center">
                  <svg className="w-5 h-5 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                </div>
              }
              title="What We Found"
              delay="stagger-1"
            >
              <p className="text-sm text-gray-600 leading-relaxed">{info.what}</p>
            </InfoCard>

            <InfoCard
              icon={
                <div className="w-9 h-9 rounded-xl bg-purple-100 flex items-center justify-center">
                  <svg className="w-5 h-5 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
              }
              title="Risk Level"
              delay="stagger-2"
            >
              <p className="text-sm text-gray-600 leading-relaxed">{info.meaning}</p>
            </InfoCard>

            <InfoCard
              icon={
                <div className="w-9 h-9 rounded-xl bg-green-100 flex items-center justify-center">
                  <svg className="w-5 h-5 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
                  </svg>
                </div>
              }
              title="Next Steps"
              delay="stagger-3"
            >
              <div className={`inline-block px-3 py-1.5 rounded-lg text-xs font-semibold mb-3 ${
                result.grade >= 3 ? "bg-red-100 text-red-700" : result.grade >= 2 ? "bg-amber-100 text-amber-700" : "bg-emerald-100 text-emerald-700"
              }`}>
                {info.timeline}
              </div>
              <ul className="space-y-1.5">
                {info.action.slice(0, 3).map((item, i) => (
                  <li key={i} className="text-sm text-gray-600 flex items-start gap-2">
                    <span className={`mt-1 w-1.5 h-1.5 rounded-full flex-shrink-0 ${info.dotColor}`} />
                    {item}
                  </li>
                ))}
              </ul>
            </InfoCard>
          </div>

          {/* Multi-Condition Screening */}
          {result.conditions && result.conditions.length > 0 && (
            <ExpandableSection
              title="Multi-Condition Screening"
              icon={
                <div className="w-8 h-8 rounded-lg bg-violet-100 flex items-center justify-center">
                  <svg className="w-4.5 h-4.5 text-violet-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m5.231 13.481L15 17.25m-4.5-15H5.625c-.621 0-1.125.504-1.125 1.125v16.5c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9zm3.75 11.625a2.625 2.625 0 11-5.25 0 2.625 2.625 0 015.25 0z" />
                  </svg>
                </div>
              }
              defaultOpen={result.conditions.some(c => c.risk_level === "high")}
            >
              <p className="text-xs text-gray-500 mb-4 leading-relaxed">
                Heuristic screening for additional ocular conditions based on the fundus analysis.
                These are not definitive diagnoses but flags for further clinical investigation.
              </p>
              <div className="space-y-3">
                {result.conditions.map((condition, i) => (
                  <ConditionRow key={i} condition={condition} />
                ))}
              </div>
            </ExpandableSection>
          )}

          {/* Expandable: Action Plan */}
          <ExpandableSection
            title="Detailed Action Plan"
            icon={
              <div className="w-8 h-8 rounded-lg bg-indigo-100 flex items-center justify-center">
                <svg className="w-4.5 h-4.5 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
            }
            defaultOpen={result.grade >= 3}
          >
            <div className={`p-4 rounded-xl bg-gradient-to-r ${info.gradientSubtle} mb-5`}>
              <p className={`text-sm font-semibold ${info.textColor}`}>
                Timeline: {info.timeline}
              </p>
            </div>
            <ol className="space-y-3">
              {info.action.map((item, i) => (
                <li key={i} className="flex items-start gap-3">
                  <span className={`flex-shrink-0 w-6 h-6 rounded-full bg-gradient-to-r ${info.gradient} flex items-center justify-center text-white text-xs font-bold`}>
                    {i + 1}
                  </span>
                  <span className="text-sm text-gray-700 leading-relaxed">{item}</span>
                </li>
              ))}
            </ol>
          </ExpandableSection>

          {/* Expandable: Lifestyle */}
          <ExpandableSection
            title="Lifestyle Recommendations"
            icon={
              <div className="w-8 h-8 rounded-lg bg-emerald-100 flex items-center justify-center">
                <svg className="w-4.5 h-4.5 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                </svg>
              </div>
            }
          >
            <ul className="space-y-3">
              {info.lifestyle.map((tip, i) => (
                <li key={i} className="flex items-start gap-3 text-sm text-gray-700">
                  <svg className="w-5 h-5 text-indigo-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span className="leading-relaxed">{tip}</span>
                </li>
              ))}
            </ul>
          </ExpandableSection>

          {/* GradCAM */}
          {result.gradcam_base64 && (
            <GradcamViewer
              originalSrc={previewUrl}
              gradcamBase64={result.gradcam_base64}
            />
          )}

          {/* Probability chart */}
          <ProbabilityChart
            probabilities={result.probabilities}
            predictedGrade={result.grade}
          />

          {/* Disclaimer */}
          <div className="card-elevated bg-gray-50/50 border border-gray-100 p-5 fade-in-up stagger-4">
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-amber-100 flex items-center justify-center">
                <svg className="w-4 h-4 text-amber-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
                </svg>
              </div>
              <div>
                <p className="text-sm font-semibold text-gray-700 mb-1">Important Disclaimer</p>
                <p className="text-xs text-gray-500 leading-relaxed">
                  This is an AI-assisted screening result, not a clinical diagnosis.
                  This report should be reviewed by a qualified ophthalmologist for clinical decision-making.
                  AI screening may not detect all retinal conditions. Regular dilated eye exams by an eye care
                  professional remain essential regardless of this screening result.
                </p>
              </div>
            </div>
          </div>

          {/* Action buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-2">
            <button
              onClick={reset}
              className="group inline-flex items-center gap-2.5 px-8 py-3.5 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-semibold shadow-[0_4px_16px_rgba(99,102,241,0.3)] hover:shadow-[0_8px_24px_rgba(99,102,241,0.4)] hover:scale-[1.02] active:scale-[0.98] transition-all duration-200"
            >
              <svg className="w-5 h-5 transition-transform group-hover:-translate-x-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Screen Another Image
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
