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

export interface ReferableDR {
  is_referable: boolean;
  referable_probability: number;
  confidence_level: string;
  clinical_action: string;
  explanation: string;
}

export interface ProgressionRisk {
  progression_risk_1yr: number;
  progression_risk_5yr: number;
  risk_level: string;
  risk_factors: string[];
  recommended_rescreen_months: number;
  explanation: string;
}

export interface ConditionScreening {
  condition_name: string;
  risk_level: string;
  findings: string[];
  description: string;
  recommendation: string;
}

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
  referable_dr: ReferableDR | null;
  progression: ProgressionRisk | null;
  conditions: ConditionScreening[];
  summary: string;
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
    referable_dr: data.referable_dr || null,
    progression: data.progression || null,
    conditions: data.conditions || [],
    summary: data.summary || "",
  };
}

export interface CompareEyeResult {
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

export interface CompareEyesResponse {
  status: string;
  left_filename: string;
  right_filename: string;
  left_results: CompareEyeResult;
  right_results: CompareEyeResult;
  comparison: {
    grade_difference: number;
    cdr_difference: number | null;
    asymmetry_flag: boolean;
    asymmetry_details: string[];
    worse_eye: string;
    clinical_significance: string;
    left_grade: number;
    left_grade_name: string;
    right_grade: number;
    right_grade_name: string;
  };
}

function _mapEyeResult(raw: Record<string, unknown>): CompareEyeResult {
  const analysis = (raw.analysis ?? {}) as Record<string, unknown>;
  const dr = (analysis.dr ?? {}) as Record<string, unknown>;
  const referral = (analysis.referral ?? raw.referral ?? {}) as Record<string, unknown>;
  const gradcam = (analysis.gradcam ?? raw.gradcam ?? {}) as Record<string, unknown>;

  return {
    grade: (dr.grade as number) ?? 0,
    grade_label: (dr.grade_name as string) || (dr.grade_label as string) || "Unknown",
    confidence: (dr.confidence as number) ?? 0,
    probabilities: (dr.probabilities as Record<string, number>) || {},
    gradcam_base64: (gradcam.overlay_png_base64 as string) || "",
    referral: {
      urgency: (referral.urgency as string) || "none",
      recommendation: (referral.recommendation as string) || "",
    },
  };
}

export async function compareEyes(leftFile: File, rightFile: File): Promise<CompareEyesResponse> {
  const form = new FormData();
  form.append("left_eye", leftFile);
  form.append("right_eye", rightFile);
  const { data } = await api.post("/demo/compare-eyes", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  return {
    status: data.status,
    left_filename: data.left_filename,
    right_filename: data.right_filename,
    left_results: _mapEyeResult(data.left_results || {}),
    right_results: _mapEyeResult(data.right_results || {}),
    comparison: data.comparison,
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
