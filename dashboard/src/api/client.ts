import axios from "axios";

const api = axios.create({
  baseURL: "/v1",
});

// Attach JWT token if present
api.interceptors.request.use((config) => {
  const token = localStorage.getItem("netra_token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export interface AnalysisResult {
  grade: number;
  grade_label: string;
  confidence: number;
  probabilities: Record<string, number>;
  gradcam_base64: string;
  referral: {
    urgency: string;
    recommendation: string;
  };
}

export interface Screening {
  id: string;
  created_at: string;
  patient_name: string;
  grade_label: string;
  grade: number;
  confidence: number;
  status: string;
}

export async function analyzeImage(file: File): Promise<AnalysisResult> {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post("/demo/analyze", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  // Map the nested API response to our flat AnalysisResult shape
  const dr = data.analysis?.dr || data.dr || data;
  const referral = data.analysis?.referral || data.referral || {};
  const gradcam = data.analysis?.gradcam || data.gradcam || {};

  return {
    grade: dr.grade ?? 0,
    grade_label: dr.grade_name || dr.grade_label || "Unknown",
    confidence: dr.confidence ?? 0,
    probabilities: dr.probabilities || {},
    gradcam_base64: gradcam.overlay_png_base64 || dr.gradcam_base64 || "",
    referral: {
      urgency: referral.urgency || "none",
      recommendation: referral.recommendation || "",
    },
  };
}

export async function fetchScreenings(): Promise<Screening[]> {
  const { data } = await api.get<Screening[]>("/screenings/");
  return data;
}

export async function login(
  username: string,
  password: string
): Promise<string> {
  const form = new URLSearchParams();
  form.append("username", username);
  form.append("password", password);
  const { data } = await api.post<{ access_token: string }>(
    "/auth/login",
    form,
    { headers: { "Content-Type": "application/x-www-form-urlencoded" } }
  );
  localStorage.setItem("netra_token", data.access_token);
  return data.access_token;
}

export default api;
