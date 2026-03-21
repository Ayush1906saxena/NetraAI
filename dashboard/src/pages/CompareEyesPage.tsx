import { useState, useCallback, useRef } from "react";
import GradcamViewer from "../components/GradcamViewer";
import ProbabilityChart from "../components/ProbabilityChart";
import ResultCard from "../components/ResultCard";
import ReferralBanner from "../components/ReferralBanner";
import { compareEyes, type CompareEyesResponse, type CompareEyeResult } from "../api/client";

/* ── Grade color config ──────────────────────────────────────────────── */

const GRADE_COLORS: Record<number, { gradient: string; badge: string; text: string }> = {
  0: { gradient: "from-emerald-500 to-green-500", badge: "bg-emerald-100 text-emerald-700", text: "text-emerald-700" },
  1: { gradient: "from-lime-500 to-emerald-500", badge: "bg-green-100 text-green-700", text: "text-green-700" },
  2: { gradient: "from-amber-500 to-orange-500", badge: "bg-amber-100 text-amber-700", text: "text-amber-700" },
  3: { gradient: "from-red-500 to-rose-500", badge: "bg-red-100 text-red-700", text: "text-red-700" },
  4: { gradient: "from-red-600 to-red-800", badge: "bg-red-200 text-red-800", text: "text-red-800" },
};

/* ── Upload zone component ───────────────────────────────────────────── */

function EyeUploadZone({
  label,
  side,
  file,
  previewUrl,
  onFile,
  disabled,
}: {
  label: string;
  side: "left" | "right";
  file: File | null;
  previewUrl: string | null;
  onFile: (f: File) => void;
  disabled: boolean;
}) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (f: File) => {
      if (!f.type.startsWith("image/")) return;
      onFile(f);
    },
    [onFile],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const f = e.dataTransfer.files[0];
      if (f) handleFile(f);
    },
    [handleFile],
  );

  return (
    <div className="flex-1">
      <p className="text-sm font-semibold text-gray-700 mb-2 text-center">{label}</p>
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => !disabled && inputRef.current?.click()}
        className={`relative rounded-2xl p-6 text-center cursor-pointer transition-all duration-300 border-2 border-dashed ${
          previewUrl
            ? "border-indigo-300 bg-indigo-50/20"
            : dragOver
              ? "border-indigo-400 bg-indigo-50/60 scale-[1.01]"
              : "border-gray-200 hover:border-indigo-300 hover:bg-indigo-50/30"
        } ${disabled ? "opacity-50 cursor-not-allowed pointer-events-none" : ""}`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
          className="hidden"
          disabled={disabled}
        />

        {previewUrl ? (
          <div className="space-y-3">
            <img
              src={previewUrl}
              alt={`${side} eye preview`}
              className="w-full aspect-square object-cover rounded-xl"
            />
            <p className="text-xs text-gray-500 truncate">{file?.name}</p>
          </div>
        ) : (
          <div className="py-8">
            <div className={`mx-auto w-16 h-16 rounded-full flex items-center justify-center mb-4 ${
              side === "left" ? "bg-blue-100" : "bg-purple-100"
            }`}>
              <svg className={`w-8 h-8 ${side === "left" ? "text-blue-500" : "text-purple-500"}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
            </div>
            <p className="text-sm font-medium text-gray-600">
              {dragOver ? "Drop image here" : `Upload ${label}`}
            </p>
            <p className="text-xs text-gray-400 mt-1">Drag & drop or click</p>
          </div>
        )}
      </div>
    </div>
  );
}

/* ── Single eye result panel ─────────────────────────────────────────── */

function EyeResultPanel({
  label,
  result,
  previewUrl,
}: {
  label: string;
  result: CompareEyeResult;
  previewUrl: string;
}) {
  const colors = GRADE_COLORS[result.grade] ?? GRADE_COLORS[0];

  return (
    <div className="flex-1 space-y-4">
      <div className={`text-center py-3 px-4 rounded-xl bg-gradient-to-r ${colors.gradient} text-white`}>
        <p className="text-xs font-semibold uppercase tracking-wider opacity-80">{label}</p>
        <p className="text-lg font-bold">Grade {result.grade} — {result.grade_label}</p>
        <p className="text-sm opacity-80">Confidence: {(result.confidence * 100).toFixed(1)}%</p>
      </div>

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

      {result.gradcam_base64 && (
        <GradcamViewer
          originalSrc={previewUrl}
          gradcamBase64={result.gradcam_base64}
        />
      )}

      <ProbabilityChart
        probabilities={result.probabilities}
        predictedGrade={result.grade}
      />
    </div>
  );
}

/* ── Main page ───────────────────────────────────────────────────────── */

export default function CompareEyesPage() {
  const [leftFile, setLeftFile] = useState<File | null>(null);
  const [rightFile, setRightFile] = useState<File | null>(null);
  const [leftPreview, setLeftPreview] = useState<string | null>(null);
  const [rightPreview, setRightPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<CompareEyesResponse | null>(null);

  const handleLeftFile = useCallback((f: File) => {
    setLeftFile(f);
    setLeftPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  }, []);

  const handleRightFile = useCallback((f: File) => {
    setRightFile(f);
    setRightPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  }, []);

  const handleCompare = async () => {
    if (!leftFile || !rightFile) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await compareEyes(leftFile, rightFile);
      setResult(data);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Comparison failed. Is the backend running?";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setLeftFile(null);
    setRightFile(null);
    setLeftPreview(null);
    setRightPreview(null);
    setResult(null);
    setError(null);
  };

  const comparison = result?.comparison;

  return (
    <div className="max-w-5xl mx-auto">
      {/* Header */}
      <div className="text-center mb-8 pt-4 fade-in-up">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-purple-50 border border-purple-100 text-purple-600 text-xs font-semibold mb-5 tracking-wide uppercase">
          <span className="w-1.5 h-1.5 rounded-full bg-purple-500 animate-pulse" />
          Bilateral Comparison
        </div>
        <h1 className="text-4xl sm:text-5xl font-extrabold text-gray-900 tracking-tight leading-tight">
          Left vs Right Eye
          <br />
          <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            Comparison
          </span>
        </h1>
        <p className="text-lg text-gray-500 mt-4 max-w-xl mx-auto leading-relaxed">
          Upload both eyes to detect asymmetric disease progression
          and bilateral pathology patterns.
        </p>
      </div>

      {/* Upload zones */}
      {!result && (
        <div className="fade-in-up">
          <div className="flex gap-6">
            <EyeUploadZone
              label="Left Eye (OS)"
              side="left"
              file={leftFile}
              previewUrl={leftPreview}
              onFile={handleLeftFile}
              disabled={loading}
            />
            <EyeUploadZone
              label="Right Eye (OD)"
              side="right"
              file={rightFile}
              previewUrl={rightPreview}
              onFile={handleRightFile}
              disabled={loading}
            />
          </div>

          {/* Compare button */}
          <div className="text-center mt-8">
            <button
              onClick={handleCompare}
              disabled={!leftFile || !rightFile || loading}
              className={`inline-flex items-center gap-2.5 px-10 py-4 rounded-xl font-semibold shadow-lg transition-all duration-200 ${
                leftFile && rightFile && !loading
                  ? "bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:shadow-xl hover:scale-[1.02] active:scale-[0.98]"
                  : "bg-gray-200 text-gray-400 cursor-not-allowed"
              }`}
            >
              {loading ? (
                <>
                  <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Analyzing Both Eyes...
                </>
              ) : (
                <>
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                  </svg>
                  Compare Eyes
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mt-6 card-elevated border-red-200 bg-red-50/50 p-5 text-sm fade-in-up">
          <div className="flex items-start gap-3">
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-red-100 flex items-center justify-center">
              <svg className="w-4 h-4 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </div>
            <div>
              <p className="font-semibold text-red-800">Comparison Failed</p>
              <p className="text-red-600 mt-1">{error}</p>
              <button onClick={reset} className="mt-3 text-sm font-medium text-red-600 hover:text-red-700 underline underline-offset-2">
                Try again
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Results */}
      {result && comparison && leftPreview && rightPreview && (
        <div className="space-y-6 pb-8 fade-in-up">

          {/* Asymmetry banner */}
          {comparison.asymmetry_flag && (
            <div className="rounded-2xl bg-gradient-to-r from-red-500 to-rose-600 p-6 text-white shadow-lg">
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-12 h-12 rounded-full bg-white/20 flex items-center justify-center">
                  <svg className="w-7 h-7 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-lg font-bold">Asymmetry Detected</h3>
                  <p className="text-white/90 text-sm mt-1">{comparison.clinical_significance}</p>
                  {comparison.asymmetry_details.length > 0 && (
                    <ul className="mt-3 space-y-1">
                      {comparison.asymmetry_details.map((detail, i) => (
                        <li key={i} className="text-sm text-white/80 flex items-start gap-2">
                          <span className="mt-1 w-1.5 h-1.5 rounded-full bg-white/60 flex-shrink-0" />
                          {detail}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Comparison summary card */}
          <div className="card-elevated p-6">
            <h3 className="text-base font-semibold text-gray-900 mb-4">Comparison Summary</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center p-3 rounded-xl bg-gray-50">
                <p className="text-xs text-gray-500 mb-1">Grade Difference</p>
                <p className={`text-2xl font-bold ${comparison.grade_difference >= 2 ? "text-red-600" : "text-gray-900"}`}>
                  {comparison.grade_difference}
                </p>
              </div>
              <div className="text-center p-3 rounded-xl bg-gray-50">
                <p className="text-xs text-gray-500 mb-1">Worse Eye</p>
                <p className="text-2xl font-bold text-gray-900 capitalize">{comparison.worse_eye}</p>
              </div>
              <div className="text-center p-3 rounded-xl bg-gray-50">
                <p className="text-xs text-gray-500 mb-1">Left Grade</p>
                <p className={`text-2xl font-bold ${(GRADE_COLORS[comparison.left_grade] ?? GRADE_COLORS[0]).text}`}>
                  {comparison.left_grade}
                </p>
                <p className="text-xs text-gray-400">{comparison.left_grade_name}</p>
              </div>
              <div className="text-center p-3 rounded-xl bg-gray-50">
                <p className="text-xs text-gray-500 mb-1">Right Grade</p>
                <p className={`text-2xl font-bold ${(GRADE_COLORS[comparison.right_grade] ?? GRADE_COLORS[0]).text}`}>
                  {comparison.right_grade}
                </p>
                <p className="text-xs text-gray-400">{comparison.right_grade_name}</p>
              </div>
            </div>

            {!comparison.asymmetry_flag && (
              <div className="mt-4 p-4 rounded-xl bg-emerald-50 border border-emerald-200">
                <p className="text-sm text-emerald-800">
                  <span className="font-semibold">No significant asymmetry detected.</span>{" "}
                  {comparison.clinical_significance}
                </p>
              </div>
            )}

            {comparison.cdr_difference !== null && (
              <div className="mt-4 p-3 rounded-xl bg-blue-50 border border-blue-200">
                <p className="text-sm text-blue-800">
                  <span className="font-semibold">CDR Difference:</span> {comparison.cdr_difference.toFixed(3)}
                  {comparison.cdr_difference >= 0.1
                    ? " — asymmetric CDR may indicate glaucoma risk"
                    : " — within normal range"}
                </p>
              </div>
            )}
          </div>

          {/* Side-by-side detailed results */}
          <div className="flex gap-6 flex-col lg:flex-row">
            <EyeResultPanel
              label="Left Eye (OS)"
              result={result.left_results}
              previewUrl={leftPreview}
            />
            <EyeResultPanel
              label="Right Eye (OD)"
              result={result.right_results}
              previewUrl={rightPreview}
            />
          </div>

          {/* Disclaimer */}
          <div className="card-elevated bg-gray-50/50 border border-gray-100 p-5">
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-amber-100 flex items-center justify-center">
                <svg className="w-4 h-4 text-amber-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
                </svg>
              </div>
              <div>
                <p className="text-sm font-semibold text-gray-700 mb-1">Important Disclaimer</p>
                <p className="text-xs text-gray-500 leading-relaxed">
                  This is an AI-assisted bilateral comparison, not a clinical diagnosis.
                  Asymmetry detection is based on AI grading and may not capture all
                  clinically relevant differences. This report should be reviewed by a
                  qualified ophthalmologist for clinical decision-making.
                </p>
              </div>
            </div>
          </div>

          {/* Reset button */}
          <div className="flex justify-center pt-2">
            <button
              onClick={reset}
              className="group inline-flex items-center gap-2.5 px-8 py-3.5 rounded-xl bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold shadow-[0_4px_16px_rgba(99,102,241,0.3)] hover:shadow-[0_8px_24px_rgba(99,102,241,0.4)] hover:scale-[1.02] active:scale-[0.98] transition-all duration-200"
            >
              <svg className="w-5 h-5 transition-transform group-hover:-translate-x-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Compare Another Pair
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
