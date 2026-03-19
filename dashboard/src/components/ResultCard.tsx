interface Props {
  grade: number;
  gradeLabel: string;
  confidence: number;
}

const GRADE_COLORS: Record<number, { bg: string; text: string; dot: string }> = {
  0: { bg: "bg-green-50", text: "text-[#28a745]", dot: "bg-[#28a745]" },
  1: { bg: "bg-green-50", text: "text-[#28a745]", dot: "bg-[#28a745]" },
  2: { bg: "bg-amber-50", text: "text-[#ffc107]", dot: "bg-[#ffc107]" },
  3: { bg: "bg-red-50", text: "text-[#dc3545]", dot: "bg-[#dc3545]" },
  4: { bg: "bg-red-50", text: "text-[#dc3545]", dot: "bg-[#dc3545]" },
};

export default function ResultCard({ grade, gradeLabel, confidence }: Props) {
  const colors = GRADE_COLORS[grade] ?? GRADE_COLORS[0];
  const pct = (confidence * 100).toFixed(1);

  return (
    <div className={`rounded-xl p-6 ${colors.bg} border`}>
      <div className="flex items-center gap-3 mb-3">
        <span className={`w-4 h-4 rounded-full ${colors.dot}`} />
        <h3 className={`text-2xl font-bold ${colors.text}`}>{gradeLabel}</h3>
      </div>
      <p className="text-gray-600 text-sm">DR Grade {grade}</p>
      <div className="mt-4">
        <div className="flex justify-between text-sm mb-1">
          <span className="text-gray-500">Confidence</span>
          <span className="font-semibold">{pct}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2.5">
          <div
            className={`h-2.5 rounded-full ${colors.dot}`}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>
    </div>
  );
}
