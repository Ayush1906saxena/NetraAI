import { useState } from "react";
import FundusUploader from "../components/FundusUploader";
import ResultCard from "../components/ResultCard";
import GradcamViewer from "../components/GradcamViewer";
import ProbabilityChart from "../components/ProbabilityChart";
import { analyzeImage, type AnalysisResult } from "../api/client";

const GRADE_INFO: Record<number, {
  title: string;
  emoji: string;
  color: string;
  bgColor: string;
  borderColor: string;
  what: string;
  meaning: string;
  action: string[];
  timeline: string;
  lifestyle: string[];
}> = {
  0: {
    title: "No Diabetic Retinopathy Detected",
    emoji: "✅",
    color: "text-green-800",
    bgColor: "bg-green-50",
    borderColor: "border-green-400",
    what: "Your retinal scan shows no signs of diabetic retinopathy. The blood vessels in your retina appear healthy with no visible damage from diabetes.",
    meaning: "This is great news! However, diabetes can still affect your eyes over time, so regular screening is essential.",
    action: [
      "Continue annual retinal screening",
      "Keep your blood sugar (HbA1c) below 7%",
      "Maintain regular check-ups with your diabetes doctor",
    ],
    timeline: "Next screening recommended in 12 months",
    lifestyle: [
      "Control blood sugar levels — this is the #1 way to prevent eye damage",
      "Keep blood pressure under 130/80 mmHg",
      "Don't skip your diabetes medications",
      "Exercise regularly (30 min/day, 5 days/week)",
      "Annual dilated eye exam even if this screening is normal",
    ],
  },
  1: {
    title: "Mild Non-Proliferative Diabetic Retinopathy",
    emoji: "🟡",
    color: "text-amber-800",
    bgColor: "bg-amber-50",
    borderColor: "border-amber-400",
    what: "Small balloon-like swellings (microaneurysms) were detected in the tiny blood vessels of your retina. This is the earliest stage of diabetic eye disease.",
    meaning: "At this stage, you likely have no vision problems. But this is an early warning sign that diabetes is starting to affect your eyes. With good diabetes control, this can stabilize or even improve.",
    action: [
      "See an eye doctor within 3 months for a detailed exam",
      "Focus on tightening blood sugar control (target HbA1c < 7%)",
      "Rescreen every 6 months to monitor for changes",
    ],
    timeline: "Follow-up screening in 6 months. Eye doctor visit within 3 months.",
    lifestyle: [
      "Strictly control blood sugar — this is the most important thing you can do",
      "Monitor and control blood pressure (target < 130/80)",
      "Reduce salt intake and manage cholesterol",
      "Stop smoking if applicable — smoking accelerates retinal damage",
      "Report any vision changes (blurriness, floaters) to your doctor immediately",
    ],
  },
  2: {
    title: "Moderate Non-Proliferative Diabetic Retinopathy",
    emoji: "🟠",
    color: "text-orange-800",
    bgColor: "bg-orange-50",
    borderColor: "border-orange-400",
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
      "Aggressive blood sugar control is critical — work with your doctor to optimize medications",
      "Blood pressure control is equally important at this stage",
      "Consider seeing a diabetes educator for lifestyle management",
      "Avoid heavy lifting or straining (can worsen retinal bleeding)",
      "Ensure adequate sleep and stress management",
      "Do NOT delay your ophthalmologist appointment",
    ],
  },
  3: {
    title: "Severe Non-Proliferative Diabetic Retinopathy",
    emoji: "🔴",
    color: "text-red-800",
    bgColor: "bg-red-50",
    borderColor: "border-red-400",
    what: "Many blood vessels in your retina are blocked, depriving large areas of the retina of blood supply. The retina is sending distress signals to grow new blood vessels.",
    meaning: "This is a serious stage. There is a high risk (>50% within 1 year) of progressing to proliferative diabetic retinopathy, which can cause sudden vision loss. Urgent medical attention is needed.",
    action: [
      "See a retina specialist within 1 week",
      "You may need laser treatment (panretinal photocoagulation) or anti-VEGF injections",
      "Do NOT delay — timely treatment can prevent vision loss",
      "Frequent monitoring every 1-2 months",
    ],
    timeline: "URGENT: Retina specialist within 1 week.",
    lifestyle: [
      "This is a medical priority — treat it with the same urgency as a heart problem",
      "Strictly follow all prescribed medications",
      "Avoid activities that increase eye pressure (heavy lifting, straining, inverted positions)",
      "Report any sudden vision changes, floaters, or flashes of light IMMEDIATELY",
      "Keep all follow-up appointments — do not reschedule",
      "Emotional support is important — talk to family about your condition",
    ],
  },
  4: {
    title: "Proliferative Diabetic Retinopathy",
    emoji: "🚨",
    color: "text-red-900",
    bgColor: "bg-red-100",
    borderColor: "border-red-600",
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
      "Seek treatment immediately — every day matters at this stage",
      "Avoid any strenuous activity until seen by specialist",
      "If you experience sudden vision loss, flashing lights, or a 'curtain' over your vision, go to the emergency room",
      "You will likely need multiple treatment sessions — commit to the full treatment plan",
      "Blood sugar and blood pressure control remain critical even during treatment",
      "Seek support — organizations like the diabetic retinopathy foundation can help",
    ],
  },
};

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
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-[#1B4F72]">
          Diabetic Retinopathy Screening
        </h1>
        <p className="text-gray-500 mt-2">
          Upload a fundus image for AI-powered DR grading
        </p>
      </div>

      {/* Upload area */}
      {!result && !loading && (
        <FundusUploader onFileSelected={handleFile} disabled={loading} />
      )}

      {/* Loading */}
      {loading && (
        <div className="flex flex-col items-center justify-center py-16">
          <div className="w-12 h-12 border-4 border-[#1B4F72] border-t-transparent rounded-full animate-spin" />
          <p className="mt-4 text-gray-500 text-sm">Analyzing retinal image...</p>
          <p className="mt-1 text-gray-400 text-xs">This usually takes 3-5 seconds</p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mt-6 bg-red-50 border border-red-200 text-red-700 rounded-xl p-4 text-sm">
          {error}
          <button onClick={reset} className="block mt-2 underline text-red-600">Try again</button>
        </div>
      )}

      {/* Results */}
      {result && info && previewUrl && (
        <div className="space-y-6 mt-2">

          {/* Main Result Banner */}
          <div className={`${info.bgColor} border-2 ${info.borderColor} rounded-2xl p-6`}>
            <div className="flex items-start gap-4">
              <span className="text-4xl">{info.emoji}</span>
              <div className="flex-1">
                <h2 className={`text-xl font-bold ${info.color}`}>{info.title}</h2>
                <div className="flex items-center gap-3 mt-1">
                  <span className="text-sm text-gray-600">DR Grade {result.grade} of 4</span>
                  <span className="text-sm text-gray-400">|</span>
                  <span className="text-sm text-gray-600">AI Confidence: {(result.confidence * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>
          </div>

          {/* What This Means */}
          <div className="bg-white border rounded-2xl p-6">
            <h3 className="text-lg font-semibold text-[#1B4F72] mb-3">What Was Found</h3>
            <p className="text-gray-700 leading-relaxed">{info.what}</p>
            <h3 className="text-lg font-semibold text-[#1B4F72] mt-5 mb-3">What This Means For You</h3>
            <p className="text-gray-700 leading-relaxed">{info.meaning}</p>
          </div>

          {/* What To Do Next */}
          <div className={`${info.bgColor} border-2 ${info.borderColor} rounded-2xl p-6`}>
            <h3 className={`text-lg font-semibold ${info.color} mb-3`}>
              {result.grade >= 3 ? "⚠ Urgent: " : ""}What To Do Next
            </h3>
            <ul className="space-y-2">
              {info.action.map((item, i) => (
                <li key={i} className="flex items-start gap-3">
                  <span className={`mt-1 w-5 h-5 rounded-full flex items-center justify-center text-white text-xs font-bold ${
                    result.grade <= 1 ? "bg-green-500" : result.grade === 2 ? "bg-orange-500" : "bg-red-500"
                  }`}>{i + 1}</span>
                  <span className="text-gray-800">{item}</span>
                </li>
              ))}
            </ul>
            <div className={`mt-4 p-3 rounded-lg ${result.grade >= 3 ? "bg-red-100" : "bg-white/60"}`}>
              <p className={`text-sm font-semibold ${info.color}`}>
                Timeline: {info.timeline}
              </p>
            </div>
          </div>

          {/* Lifestyle & Prevention */}
          <div className="bg-white border rounded-2xl p-6">
            <h3 className="text-lg font-semibold text-[#1B4F72] mb-3">
              Lifestyle Recommendations
            </h3>
            <ul className="space-y-3">
              {info.lifestyle.map((tip, i) => (
                <li key={i} className="flex items-start gap-3 text-gray-700">
                  <span className="text-blue-500 mt-0.5">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </span>
                  <span>{tip}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* AI Analysis Details (collapsible) */}
          <details className="bg-white border rounded-2xl">
            <summary className="p-6 cursor-pointer text-lg font-semibold text-[#1B4F72] hover:text-[#154360]">
              AI Analysis Details
            </summary>
            <div className="px-6 pb-6 space-y-6">
              {/* Grade Card + Probabilities */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <ResultCard
                  grade={result.grade}
                  gradeLabel={result.grade_label}
                  confidence={result.confidence}
                />
                <div>
                  <ProbabilityChart probabilities={result.probabilities} />
                </div>
              </div>

              {/* GradCAM */}
              {result.gradcam_base64 && (
                <div>
                  <h4 className="text-sm font-semibold text-gray-600 mb-2">
                    AI Attention Map (GradCAM)
                  </h4>
                  <p className="text-xs text-gray-400 mb-3">
                    The highlighted areas show where the AI focused when making its assessment. Warm colors (red/yellow) indicate regions of interest.
                  </p>
                  <GradcamViewer
                    originalSrc={previewUrl}
                    gradcamBase64={result.gradcam_base64}
                  />
                </div>
              )}
            </div>
          </details>

          {/* Disclaimer */}
          <div className="bg-yellow-50 border border-yellow-200 rounded-2xl p-4">
            <p className="text-xs text-yellow-800 leading-relaxed">
              <strong>Important Disclaimer:</strong> This is an AI-assisted screening result, not a clinical diagnosis.
              This report should be reviewed by a qualified ophthalmologist for clinical decision-making.
              AI screening may not detect all retinal conditions. Regular dilated eye exams by an eye care
              professional remain essential regardless of this screening result.
            </p>
          </div>

          {/* Try Another */}
          <div className="text-center pb-8">
            <button
              onClick={reset}
              className="px-8 py-3 bg-[#1B4F72] text-white rounded-lg font-medium hover:bg-[#154360] transition-colors"
            >
              Screen Another Image
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
