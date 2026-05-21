import { authFetch, BASE, parseError } from "./client";

export interface CloudCheck {
  name: string;
  result: "PASS" | "FAIL" | "ERROR";
  value: number | null;
  limit: number;
  detail?: string;
}

export interface CloudValidationResult {
  id: string;
  name: string;
  status: string;
  is: {
    ic_mean: number | null;
    ic_ir: number | null;
    turnover: number | null;
    sharpe: number | null;
    fitness: number | null;
    ic_decay: number[] | null;
    coverage: number | null;
    data_days: number | null;
    max_correlation: number | null;
  };
  checks: CloudCheck[] | null;
  reject_reason: string | null;
}

export async function checkCloudStatus(): Promise<{ configured: boolean; cloud_url: string }> {
  const res = await authFetch(`${BASE}/api/v1/cloud/status`);
  if (!res.ok) return { configured: false, cloud_url: "" };
  return res.json();
}

export interface CloudUploadRequest {
  expression: string;
  universe: string;
  name?: string;
  claimed_ic_mean?: number;
  claimed_ic_ir?: number;
  factor_values_data: Array<{ date: string; values: Record<string, number> }>;
}

export async function uploadToCloud(req: CloudUploadRequest): Promise<CloudValidationResult> {
  const res = await authFetch(`${BASE}/api/v1/cloud/upload`, {
    method: "POST",
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}
