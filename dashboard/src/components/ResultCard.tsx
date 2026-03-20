import { useEffect, useState } from "react";

interface Props {
  grade: number;
  gradeLabel: string;
  confidence: number;
}

const GRADE_GRADIENTS: Record<number, { gradient: string; ring: string; text: string; label: string }> = {
  0: { gradient: "from-emerald-400 to-green-500", ring: "#10b981", text: "text-emerald-700", label: "Healthy" },
  1: { gradient: "from-lime-400 to-emerald-500", ring: "#22c55e", text: "text-green-700", label: "Mild" },
  2: { gradient: "from-amber-400 to-orange-500", ring: "#f59e0b", text: "text-amber-700", label: "Moderate" },
  3: { gradient: "from-orange-400 to-red-500", ring: "#ef4444", text: "text-red-700", label: "Severe" },
  4: { gradient: "from-red-500 to-red-700", ring: "#dc2626", text: "text-red-800", label: "Proliferative" },
};

export default function ResultCard({ grade, gradeLabel, confidence }: Props) {
  const colors = GRADE_GRADIENTS[grade] ?? GRADE_GRADIENTS[0];
  const pct = confidence * 100;
  const [animatedPct, setAnimatedPct] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => setAnimatedPct(pct), 100);
    return () => clearTimeout(timer);
  }, [pct]);

  // SVG circle gauge
  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (animatedPct / 100) * circumference;

  return (
    <div className="card-elevated p-6 fade-in-up">
      <div className="flex items-center gap-6">
        {/* Circular confidence gauge */}
        <div className="relative flex-shrink-0">
          <svg width="132" height="132" viewBox="0 0 132 132" className="transform -rotate-90">
            {/* Background ring */}
            <circle
              cx="66"
              cy="66"
              r={radius}
              fill="none"
              stroke="#f1f5f9"
              strokeWidth="10"
            />
            {/* Animated progress ring */}
            <circle
              cx="66"
              cy="66"
              r={radius}
              fill="none"
              stroke={colors.ring}
              strokeWidth="10"
              strokeLinecap="round"
              strokeDasharray={circumference}
              strokeDashoffset={strokeDashoffset}
              style={{ transition: "stroke-dashoffset 1.2s cubic-bezier(0.4, 0, 0.2, 1)" }}
            />
          </svg>
          {/* Center text */}
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-2xl font-bold text-gray-900">{pct.toFixed(1)}%</span>
            <span className="text-[10px] font-medium text-gray-400 uppercase tracking-wider">Confidence</span>
          </div>
        </div>

        {/* Grade info */}
        <div className="flex-1 min-w-0">
          <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full bg-gradient-to-r ${colors.gradient} text-white text-xs font-semibold mb-2`}>
            Grade {grade}
          </div>
          <h3 className="text-xl font-bold text-gray-900 leading-tight">{gradeLabel}</h3>
          <p className="text-sm text-gray-400 mt-1">AI classification result</p>
        </div>
      </div>
    </div>
  );
}
