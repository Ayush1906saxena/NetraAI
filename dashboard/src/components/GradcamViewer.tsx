import { useState } from "react";

interface Props {
  originalSrc: string;
  gradcamBase64: string;
}

export default function GradcamViewer({ originalSrc, gradcamBase64 }: Props) {
  const [overlay, setOverlay] = useState(0.6);

  return (
    <div className="card-elevated overflow-hidden fade-in-up stagger-2">
      {/* Header */}
      <div className="px-6 pt-6 pb-4 flex items-center justify-between">
        <div>
          <h4 className="text-base font-semibold text-gray-900">AI Attention Map</h4>
          <p className="text-xs text-gray-400 mt-0.5">GradCAM visualization of model focus regions</p>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-gray-400 font-medium">Original</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={overlay}
            onChange={(e) => setOverlay(parseFloat(e.target.value))}
            className="w-24 h-1.5 bg-gray-200 rounded-full appearance-none cursor-pointer accent-indigo-500"
          />
          <span className="text-xs text-gray-400 font-medium">Heatmap</span>
        </div>
      </div>

      {/* Image with overlay */}
      <div className="relative mx-6 mb-4 rounded-xl overflow-hidden bg-gray-900 aspect-square max-w-md">
        {/* Original image */}
        <img
          src={originalSrc}
          alt="Original fundus"
          className="absolute inset-0 w-full h-full object-cover"
        />
        {/* GradCAM overlay */}
        {gradcamBase64 && (
          <img
            src={`data:image/png;base64,${gradcamBase64}`}
            alt="GradCAM heatmap"
            className="absolute inset-0 w-full h-full object-cover transition-opacity duration-300"
            style={{ opacity: overlay }}
          />
        )}
      </div>

      {/* Color legend */}
      <div className="px-6 pb-5 flex items-center justify-center gap-6">
        <div className="flex items-center gap-2">
          <div className="w-4 h-2.5 rounded-full bg-gradient-to-r from-blue-500 to-cyan-400" />
          <span className="text-xs text-gray-500">Normal</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-2.5 rounded-full bg-gradient-to-r from-green-400 to-yellow-400" />
          <span className="text-xs text-gray-500">Low attention</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-2.5 rounded-full bg-gradient-to-r from-orange-400 to-red-500" />
          <span className="text-xs text-gray-500">High attention</span>
        </div>
      </div>
    </div>
  );
}
