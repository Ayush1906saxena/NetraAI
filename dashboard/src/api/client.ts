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
  const { data } = await api.post<AnalysisResult>("/demo/analyze", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
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
