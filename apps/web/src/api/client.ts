import type { ExplainableReport, ScanPayload, ScanSummary } from "@consentshield/shared";
import { API_ROUTES } from "@consentshield/shared";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const token = localStorage.getItem("cs_token");
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(init?.headers || {}),
    },
  });
  if (!res.ok) {
    throw new Error(await res.text());
  }
  return res.json() as Promise<T>;
}

export const api = {
  listScans: () => request<ScanSummary[]>(API_ROUTES.scan),
  createScan: (payload: ScanPayload) =>
    request<ScanSummary>(API_ROUTES.scan, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  getReport: (id: string) => request<ExplainableReport>(API_ROUTES.report(id)),
  rulesCatalog: () => request<{ rules: unknown[]; count: number }>(API_ROUTES.rulesCatalog),
  modelsRegistry: () => request<{ stub_mode: boolean; modules: unknown[] }>(API_ROUTES.modelsRegistry),
};
