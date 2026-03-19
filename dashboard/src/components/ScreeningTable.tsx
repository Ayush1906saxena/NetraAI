import type { Screening } from "../api/client";

interface Props {
  screenings: Screening[];
  onSelect: (s: Screening) => void;
}

const GRADE_DOT: Record<number, string> = {
  0: "bg-[#28a745]",
  1: "bg-[#28a745]",
  2: "bg-[#ffc107]",
  3: "bg-[#dc3545]",
  4: "bg-[#dc3545]",
};

export default function ScreeningTable({ screenings, onSelect }: Props) {
  if (screenings.length === 0) {
    return (
      <div className="text-center py-12 text-gray-400">
        No screenings found.
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-200 text-left text-gray-500">
            <th className="py-3 px-4 font-medium">ID</th>
            <th className="py-3 px-4 font-medium">Date</th>
            <th className="py-3 px-4 font-medium">Patient</th>
            <th className="py-3 px-4 font-medium">DR Grade</th>
            <th className="py-3 px-4 font-medium">Confidence</th>
            <th className="py-3 px-4 font-medium">Status</th>
          </tr>
        </thead>
        <tbody>
          {screenings.map((s) => (
            <tr
              key={s.id}
              onClick={() => onSelect(s)}
              className="border-b border-gray-100 hover:bg-gray-50 cursor-pointer transition-colors"
            >
              <td className="py-3 px-4 font-mono text-xs text-gray-500">
                {s.id.slice(0, 8)}
              </td>
              <td className="py-3 px-4">
                {new Date(s.created_at).toLocaleDateString()}
              </td>
              <td className="py-3 px-4 font-medium">{s.patient_name}</td>
              <td className="py-3 px-4">
                <span className="flex items-center gap-2">
                  <span
                    className={`w-2.5 h-2.5 rounded-full ${GRADE_DOT[s.grade] ?? "bg-gray-300"}`}
                  />
                  {s.grade_label}
                </span>
              </td>
              <td className="py-3 px-4">
                {(s.confidence * 100).toFixed(1)}%
              </td>
              <td className="py-3 px-4">
                <span
                  className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${
                    s.status === "completed"
                      ? "bg-green-100 text-green-700"
                      : "bg-gray-100 text-gray-600"
                  }`}
                >
                  {s.status}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
