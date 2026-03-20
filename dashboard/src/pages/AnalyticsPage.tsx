import { useState } from "react";
import type { PieLabelRenderProps } from "recharts";
import {
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

// ── Mock Data ─────────────────────────────────────────────────────────────

const SUMMARY_STATS = {
  totalScreenings: 2847,
  referralRate: 0.312,
  avgConfidence: 0.891,
  iqaRejectionRate: 0.043,
};

const DR_DISTRIBUTION = [
  { name: "No DR", value: 1256, color: "#22c55e" },
  { name: "Mild NPDR", value: 498, color: "#84cc16" },
  { name: "Moderate NPDR", value: 612, color: "#f59e0b" },
  { name: "Severe NPDR", value: 289, color: "#ef4444" },
  { name: "Proliferative DR", value: 192, color: "#991b1b" },
];

const SCREENINGS_OVER_TIME = [
  { month: "Oct 2025", screenings: 312, referrals: 89 },
  { month: "Nov 2025", screenings: 387, referrals: 114 },
  { month: "Dec 2025", screenings: 421, referrals: 138 },
  { month: "Jan 2026", screenings: 498, referrals: 162 },
  { month: "Feb 2026", screenings: 563, referrals: 171 },
  { month: "Mar 2026", screenings: 666, referrals: 215 },
];

const REFERRAL_URGENCY = [
  { urgency: "None", count: 1754, color: "#22c55e" },
  { urgency: "Routine", count: 498, color: "#84cc16" },
  { urgency: "Soon", count: 314, color: "#f59e0b" },
  { urgency: "Urgent", count: 189, color: "#ef4444" },
  { urgency: "Emergency", count: 92, color: "#991b1b" },
];

const STORE_PERFORMANCE = [
  {
    store: "Apollo Clinic, Bangalore",
    screenings: 834,
    referralRate: 0.29,
    avgConfidence: 0.903,
    iqaRejectRate: 0.032,
  },
  {
    store: "Sankara Nethralaya, Chennai",
    screenings: 712,
    referralRate: 0.34,
    avgConfidence: 0.887,
    iqaRejectRate: 0.041,
  },
  {
    store: "LV Prasad Eye, Hyderabad",
    screenings: 621,
    referralRate: 0.31,
    avgConfidence: 0.895,
    iqaRejectRate: 0.038,
  },
  {
    store: "Aravind Eye, Madurai",
    screenings: 458,
    referralRate: 0.28,
    avgConfidence: 0.901,
    iqaRejectRate: 0.051,
  },
  {
    store: "AIIMS Eye Centre, Delhi",
    screenings: 222,
    referralRate: 0.36,
    avgConfidence: 0.872,
    iqaRejectRate: 0.062,
  },
];

// ── Components ────────────────────────────────────────────────────────────

function StatCard({
  title,
  value,
  subtitle,
  icon,
  color,
}: {
  title: string;
  value: string;
  subtitle: string;
  icon: React.ReactNode;
  color: string;
}) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5 flex items-start gap-4 shadow-sm hover:shadow-md transition-shadow">
      <div
        className="flex-shrink-0 w-12 h-12 rounded-lg flex items-center justify-center"
        style={{ backgroundColor: `${color}15` }}
      >
        <div style={{ color }}>{icon}</div>
      </div>
      <div className="min-w-0">
        <p className="text-sm text-gray-500 font-medium">{title}</p>
        <p className="text-2xl font-bold text-gray-900 mt-0.5">{value}</p>
        <p className="text-xs text-gray-400 mt-1">{subtitle}</p>
      </div>
    </div>
  );
}

const CUSTOM_TOOLTIP_STYLE = {
  backgroundColor: "#fff",
  border: "1px solid #e5e7eb",
  borderRadius: "8px",
  padding: "8px 12px",
  boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.1)",
};

function PieLabel(props: PieLabelRenderProps) {
  const cx = Number(props.cx ?? 0);
  const cy = Number(props.cy ?? 0);
  const midAngle = Number(props.midAngle ?? 0);
  const innerRadius = Number(props.innerRadius ?? 0);
  const outerRadius = Number(props.outerRadius ?? 0);
  const percent = Number(props.percent ?? 0);
  const name = String(props.name ?? "");

  const RADIAN = Math.PI / 180;
  const radius = innerRadius + (outerRadius - innerRadius) * 1.4;
  const x = cx + radius * Math.cos(-midAngle * RADIAN);
  const y = cy + radius * Math.sin(-midAngle * RADIAN);

  if (percent < 0.05) return null;

  return (
    <text
      x={x}
      y={y}
      fill="#374151"
      textAnchor={x > cx ? "start" : "end"}
      dominantBaseline="central"
      fontSize={12}
    >
      {name} ({(percent * 100).toFixed(0)}%)
    </text>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────

type TimeRange = "7d" | "30d" | "90d" | "all";

export default function AnalyticsPage() {
  const [timeRange, setTimeRange] = useState<TimeRange>("all");

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-[#1B4F72]">Analytics</h1>
          <p className="text-sm text-gray-500 mt-1">
            Screening performance and population health insights
          </p>
        </div>
        <div className="flex gap-1 bg-gray-100 rounded-lg p-1">
          {(["7d", "30d", "90d", "all"] as TimeRange[]).map((range) => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                timeRange === range
                  ? "bg-white text-[#1B4F72] shadow-sm"
                  : "text-gray-500 hover:text-gray-700"
              }`}
            >
              {range === "all" ? "All Time" : range.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Screenings"
          value={SUMMARY_STATS.totalScreenings.toLocaleString()}
          subtitle="Across all centres"
          color="#1B4F72"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            </svg>
          }
        />
        <StatCard
          title="Referral Rate"
          value={`${(SUMMARY_STATS.referralRate * 100).toFixed(1)}%`}
          subtitle="Grade >= Moderate NPDR"
          color="#ef4444"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
          }
        />
        <StatCard
          title="Avg Confidence"
          value={`${(SUMMARY_STATS.avgConfidence * 100).toFixed(1)}%`}
          subtitle="Model prediction confidence"
          color="#22c55e"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          }
        />
        <StatCard
          title="IQA Rejection Rate"
          value={`${(SUMMARY_STATS.iqaRejectionRate * 100).toFixed(1)}%`}
          subtitle="Failed image quality check"
          color="#f59e0b"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          }
        />
      </div>

      {/* Charts Row 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* DR Grade Distribution Pie Chart */}
        <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            DR Grade Distribution
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={DR_DISTRIBUTION}
                cx="50%"
                cy="50%"
                labelLine={true}
                label={PieLabel}
                outerRadius={100}
                innerRadius={45}
                dataKey="value"
                strokeWidth={2}
                stroke="#fff"
              >
                {DR_DISTRIBUTION.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={CUSTOM_TOOLTIP_STYLE}
                formatter={(value, name) => [
                  `${Number(value).toLocaleString()} (${((Number(value) / SUMMARY_STATS.totalScreenings) * 100).toFixed(1)}%)`,
                  String(name),
                ]}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Screenings Over Time */}
        <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Screenings Over Time
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={SCREENINGS_OVER_TIME}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis
                dataKey="month"
                tick={{ fontSize: 12 }}
                tickLine={false}
              />
              <YAxis tick={{ fontSize: 12 }} tickLine={false} axisLine={false} />
              <Tooltip contentStyle={CUSTOM_TOOLTIP_STYLE} />
              <Legend />
              <Line
                type="monotone"
                dataKey="screenings"
                stroke="#1B4F72"
                strokeWidth={2.5}
                dot={{ r: 4, fill: "#1B4F72" }}
                activeDot={{ r: 6 }}
                name="Total Screenings"
              />
              <Line
                type="monotone"
                dataKey="referrals"
                stroke="#ef4444"
                strokeWidth={2}
                dot={{ r: 3, fill: "#ef4444" }}
                strokeDasharray="5 5"
                name="Referrals"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Charts Row 2 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Referral Urgency Breakdown */}
        <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Referral Urgency Breakdown
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={REFERRAL_URGENCY} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 12 }} tickLine={false} axisLine={false} />
              <YAxis
                dataKey="urgency"
                type="category"
                tick={{ fontSize: 12 }}
                tickLine={false}
                axisLine={false}
                width={80}
              />
              <Tooltip
                contentStyle={CUSTOM_TOOLTIP_STYLE}
                formatter={(value) => [Number(value).toLocaleString(), "Patients"]}
              />
              <Bar dataKey="count" radius={[0, 6, 6, 0]} barSize={28}>
                {REFERRAL_URGENCY.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Quick Stats Panel */}
        <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Model Performance
          </h2>
          <div className="space-y-4">
            {[
              { label: "Quadratic Weighted Kappa", value: 0.912, benchmark: 0.85 },
              { label: "AUC-ROC (macro)", value: 0.964, benchmark: 0.90 },
              { label: "Sensitivity (Referable DR)", value: 0.947, benchmark: 0.90 },
              { label: "Specificity (Referable DR)", value: 0.891, benchmark: 0.85 },
            ].map((metric) => (
              <div key={metric.label}>
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-sm text-gray-600">{metric.label}</span>
                  <span className="text-sm font-semibold text-gray-900">
                    {(metric.value * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-100 rounded-full h-2.5 relative">
                  <div
                    className="h-2.5 rounded-full transition-all duration-500"
                    style={{
                      width: `${metric.value * 100}%`,
                      backgroundColor:
                        metric.value >= metric.benchmark ? "#22c55e" : "#ef4444",
                    }}
                  />
                  {/* Benchmark marker */}
                  <div
                    className="absolute top-0 h-2.5 w-0.5 bg-gray-400"
                    style={{ left: `${metric.benchmark * 100}%` }}
                    title={`Benchmark: ${(metric.benchmark * 100).toFixed(0)}%`}
                  />
                </div>
                <p className="text-xs text-gray-400 mt-0.5">
                  Benchmark: {(metric.benchmark * 100).toFixed(0)}%
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Store Performance Table */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-100">
          <h2 className="text-lg font-semibold text-gray-900">
            Centre Performance
          </h2>
          <p className="text-sm text-gray-500 mt-0.5">
            Screening metrics by deployment location
          </p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-50 text-left">
                <th className="px-6 py-3 font-medium text-gray-500 uppercase tracking-wider text-xs">
                  Centre
                </th>
                <th className="px-6 py-3 font-medium text-gray-500 uppercase tracking-wider text-xs text-right">
                  Screenings
                </th>
                <th className="px-6 py-3 font-medium text-gray-500 uppercase tracking-wider text-xs text-right">
                  Referral Rate
                </th>
                <th className="px-6 py-3 font-medium text-gray-500 uppercase tracking-wider text-xs text-right">
                  Avg Confidence
                </th>
                <th className="px-6 py-3 font-medium text-gray-500 uppercase tracking-wider text-xs text-right">
                  IQA Reject Rate
                </th>
                <th className="px-6 py-3 font-medium text-gray-500 uppercase tracking-wider text-xs text-center">
                  Status
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {STORE_PERFORMANCE.map((store, idx) => (
                <tr
                  key={idx}
                  className="hover:bg-gray-50 transition-colors"
                >
                  <td className="px-6 py-4 font-medium text-gray-900">
                    {store.store}
                  </td>
                  <td className="px-6 py-4 text-right text-gray-700 tabular-nums">
                    {store.screenings.toLocaleString()}
                  </td>
                  <td className="px-6 py-4 text-right">
                    <span
                      className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${
                        store.referralRate > 0.35
                          ? "bg-red-100 text-red-700"
                          : store.referralRate > 0.3
                          ? "bg-amber-100 text-amber-700"
                          : "bg-green-100 text-green-700"
                      }`}
                    >
                      {(store.referralRate * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-6 py-4 text-right text-gray-700 tabular-nums">
                    {(store.avgConfidence * 100).toFixed(1)}%
                  </td>
                  <td className="px-6 py-4 text-right text-gray-700 tabular-nums">
                    {(store.iqaRejectRate * 100).toFixed(1)}%
                  </td>
                  <td className="px-6 py-4 text-center">
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-700">
                      <span className="w-1.5 h-1.5 rounded-full bg-green-500" />
                      Active
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Footer note */}
      <p className="text-xs text-gray-400 text-center pb-4">
        Data shown is mock data for development. Connect to the analytics API for live metrics.
      </p>
    </div>
  );
}
