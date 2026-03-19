import { useState } from "react";
import FundusUploader from "../components/FundusUploader";
import ResultCard from "../components/ResultCard";
import GradcamViewer from "../components/GradcamViewer";
import ProbabilityChart from "../components/ProbabilityChart";
import ReferralBanner from "../components/ReferralBanner";
import { analyzeImage, type AnalysisResult } from "../api/client";

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

      {/* Upload area - shown when no result yet */}
      {!result && (
        <FundusUploader onFileSelected={handleFile} disabled={loading} />
      )}

      {/* Loading spinner */}
      {loading && (
        <div className="flex flex-col items-center justify-center py-16">
          <div className="w-12 h-12 border-4 border-[#1B4F72] border-t-transparent rounded-full animate-spin" />
          <p className="mt-4 text-gray-500 text-sm">Analyzing image...</p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mt-6 bg-red-50 border border-red-200 text-red-700 rounded-xl p-4 text-sm">
          {error}
        </div>
      )}

      {/* Results */}
      {result && previewUrl && (
        <div className="space-y-6 mt-6">
          {/* Grade + Referral */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <ResultCard
              grade={result.grade}
              gradeLabel={result.grade_label}
              confidence={result.confidence}
            />
            <ReferralBanner
              urgency={result.referral.urgency}
              recommendation={result.referral.recommendation}
            />
          </div>

          {/* Probability chart */}
          <div className="bg-white border rounded-xl p-6">
            <ProbabilityChart probabilities={result.probabilities} />
          </div>

          {/* GradCAM */}
          <div className="bg-white border rounded-xl p-6">
            <GradcamViewer
              originalSrc={previewUrl}
              gradcamBase64={result.gradcam_base64}
            />
          </div>

          {/* Reset button */}
          <div className="text-center">
            <button
              onClick={reset}
              className="px-6 py-3 bg-[#1B4F72] text-white rounded-lg font-medium hover:bg-[#154360] transition-colors"
            >
              Try Another Image
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
