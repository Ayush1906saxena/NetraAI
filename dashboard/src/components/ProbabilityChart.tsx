import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

interface Props {
  probabilities: Record<string, number>;
}

const GRADE_LABELS = [
  "No DR",
  "Mild",
  "Moderate",
  "Severe",
  "Proliferative",
];

const BAR_COLORS = ["#28a745", "#28a745", "#ffc107", "#dc3545", "#dc3545"];

export default function ProbabilityChart({ probabilities }: Props) {
  const data = GRADE_LABELS.map((label, i) => ({
    name: label,
    probability: Number(
      ((probabilities[label] ?? probabilities[String(i)] ?? 0) * 100).toFixed(1)
    ),
  }));

  return (
    <div>
      <h4 className="text-sm font-medium text-gray-500 mb-3">
        Class Probabilities
      </h4>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} layout="vertical" margin={{ left: 80 }}>
          <XAxis type="number" domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
          <YAxis type="category" dataKey="name" width={80} />
          <Tooltip formatter={(v) => `${v}%`} />
          <Bar dataKey="probability" radius={[0, 4, 4, 0]}>
            {data.map((_, i) => (
              <Cell key={i} fill={BAR_COLORS[i]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
