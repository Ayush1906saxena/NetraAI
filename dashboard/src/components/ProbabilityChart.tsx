import { useEffect, useState } from "react";

interface Props {
  probabilities: Record<string, number>;
  predictedGrade?: number;
}

const GRADE_LABELS = [
  "No DR",
  "Mild NPDR",
  "Moderate NPDR",
  "Severe NPDR",
  "Proliferative DR",
];

const SHORT_LABELS = ["No DR", "Mild", "Moderate", "Severe", "PDR"];

const BAR_COLORS = [
  { bar: "#10b981", bg: "rgba(16, 185, 129, 0.1)" },
  { bar: "#22c55e", bg: "rgba(34, 197, 94, 0.1)" },
  { bar: "#f59e0b", bg: "rgba(245, 158, 11, 0.1)" },
  { bar: "#ef4444", bg: "rgba(239, 68, 68, 0.1)" },
  { bar: "#dc2626", bg: "rgba(220, 38, 38, 0.1)" },
];

export default function ProbabilityChart({ probabilities, predictedGrade }: Props) {
  const [animate, setAnimate] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setAnimate(true), 200);
    return () => clearTimeout(timer);
  }, []);

  const values = Object.values(probabilities);
  const data = SHORT_LABELS.map((label, i) => {
    const val = probabilities[GRADE_LABELS[i]] ?? probabilities[label] ?? probabilities[String(i)] ?? values[i] ?? 0;
    return {
      label,
      fullLabel: GRADE_LABELS[i],
      value: val * 100,
      isPredicted: predictedGrade === i,
    };
  });

  const maxVal = Math.max(...data.map((d) => d.value), 1);

  return (
    <div className="card-elevated p-6 fade-in-up stagger-3">
      <h4 className="text-base font-semibold text-gray-900 mb-1">Class Probabilities</h4>
      <p className="text-xs text-gray-400 mb-5">Model confidence distribution across DR grades</p>

      <div className="space-y-3.5">
        {data.map((d, i) => (
          <div key={i} className="group">
            <div className="flex items-center justify-between mb-1.5">
              <div className="flex items-center gap-2">
                <span
                  className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                  style={{ backgroundColor: BAR_COLORS[i].bar }}
                />
                <span className={`text-sm font-medium ${d.isPredicted ? "text-gray-900" : "text-gray-600"}`}>
                  {d.label}
                </span>
                {d.isPredicted && (
                  <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-indigo-100 text-indigo-600 uppercase tracking-wide">
                    Predicted
                  </span>
                )}
              </div>
              <span className={`text-sm tabular-nums font-semibold ${d.isPredicted ? "text-gray-900" : "text-gray-500"}`}>
                {d.value.toFixed(1)}%
              </span>
            </div>
            <div
              className="w-full h-2.5 rounded-full overflow-hidden"
              style={{ backgroundColor: BAR_COLORS[i].bg }}
            >
              <div
                className="h-full rounded-full transition-all duration-1000 ease-out"
                style={{
                  width: animate ? `${(d.value / maxVal) * 100}%` : "0%",
                  backgroundColor: BAR_COLORS[i].bar,
                  transitionDelay: `${i * 100}ms`,
                  opacity: d.isPredicted ? 1 : 0.7,
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
