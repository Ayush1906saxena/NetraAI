import { useEffect, useState } from "react";
import ScreeningTable from "../components/ScreeningTable";
import { fetchScreenings, type Screening } from "../api/client";

export default function DashboardPage() {
  const [screenings, setScreenings] = useState<Screening[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Screening | null>(null);

  useEffect(() => {
    fetchScreenings()
      .then(setScreenings)
      .catch(() => setError("Failed to load screenings. Is the backend running?"))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div>
      <h1 className="text-2xl font-bold text-[#1B4F72] mb-6">
        Screening History
      </h1>

      {loading && (
        <div className="flex justify-center py-12">
          <div className="w-8 h-8 border-4 border-[#1B4F72] border-t-transparent rounded-full animate-spin" />
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 rounded-xl p-4 text-sm">
          {error}
        </div>
      )}

      {!loading && !error && (
        <div className="bg-white border rounded-xl">
          <ScreeningTable screenings={screenings} onSelect={setSelected} />
        </div>
      )}

      {/* Detail modal */}
      {selected && (
        <div
          className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4"
          onClick={() => setSelected(null)}
        >
          <div
            className="bg-white rounded-xl shadow-xl max-w-md w-full p-6"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="text-lg font-bold text-[#1B4F72] mb-4">
              Screening Details
            </h2>
            <dl className="space-y-3 text-sm">
              <div className="flex justify-between">
                <dt className="text-gray-500">ID</dt>
                <dd className="font-mono">{selected.id}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-500">Date</dt>
                <dd>{new Date(selected.created_at).toLocaleString()}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-500">Patient</dt>
                <dd className="font-medium">{selected.patient_name}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-500">DR Grade</dt>
                <dd className="font-medium">{selected.grade_label}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-500">Confidence</dt>
                <dd>{(selected.confidence * 100).toFixed(1)}%</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-500">Status</dt>
                <dd>{selected.status}</dd>
              </div>
            </dl>
            <button
              onClick={() => setSelected(null)}
              className="mt-6 w-full py-2 bg-[#1B4F72] text-white rounded-lg font-medium hover:bg-[#154360] transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
