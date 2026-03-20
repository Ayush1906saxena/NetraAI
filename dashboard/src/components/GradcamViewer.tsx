import { useState } from "react";

interface Props {
  originalSrc: string;
  gradcamBase64: string;
}

export default function GradcamViewer({ originalSrc, gradcamBase64 }: Props) {
  const [showHeatmap, setShowHeatmap] = useState(true);

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-semibold text-gray-600">Retinal Image Analysis</h4>
        <button
          onClick={() => setShowHeatmap(!showHeatmap)}
          className={`text-xs px-3 py-1.5 rounded-full font-medium transition-all ${
            showHeatmap
              ? "bg-[#1B4F72] text-white"
              : "bg-gray-100 text-gray-600 hover:bg-gray-200"
          }`}
        >
          {showHeatmap ? "Showing AI Heatmap" : "Show AI Heatmap"}
        </button>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="relative group">
          <span className="absolute top-2 left-2 bg-black/50 text-white text-[10px] px-2 py-0.5 rounded-full z-10">
            Original
          </span>
          <img
            src={originalSrc}
            alt="Original fundus"
            className="w-full rounded-xl shadow-md border border-gray-100 transition-transform group-hover:scale-[1.01]"
          />
        </div>
        <div className="relative group">
          <span className="absolute top-2 left-2 bg-black/50 text-white text-[10px] px-2 py-0.5 rounded-full z-10">
            AI Attention Map
          </span>
          {showHeatmap && gradcamBase64 ? (
            <img
              src={`data:image/png;base64,${gradcamBase64}`}
              alt="GradCAM heatmap"
              className="w-full rounded-xl shadow-md border border-gray-100 transition-transform group-hover:scale-[1.01]"
            />
          ) : (
            <img
              src={originalSrc}
              alt="Original fundus"
              className="w-full rounded-xl shadow-md border border-gray-100 opacity-60"
            />
          )}
        </div>
      </div>
      <p className="text-[11px] text-gray-400 mt-2 text-center">
        Red/yellow areas show where the AI detected pathological features. Blue areas are normal.
      </p>
    </div>
  );
}
