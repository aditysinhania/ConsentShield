/**
 * Inference provider — Remote API with progress-friendly exports.
 */
import { loadSettings } from "./settings";

export interface InferenceProvider {
  analyze(payload: unknown): Promise<unknown>;
}

export class RemoteApiProvider implements InferenceProvider {
  constructor(private baseUrl: string, private token?: string) {}

  async analyze(payload: unknown): Promise<unknown> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...(this.token ? { Authorization: `Bearer ${this.token}` } : {}),
    };
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 120_000);
    try {
      const res = await fetch(`${this.baseUrl}/api/v1/scan`, {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Scan failed (${res.status}): ${text}`);
      }
      const scan = await res.json();
      const reportRes = await fetch(`${this.baseUrl}/api/v1/report/${scan.id}`, {
        headers: this.token ? { Authorization: `Bearer ${this.token}` } : {},
        signal: controller.signal,
      });
      if (!reportRes.ok) {
        throw new Error(`Report fetch failed (${reportRes.status})`);
      }
      const report = await reportRes.json();
      return { ...report, scan_id: report.scan_id || scan.id };
    } catch (e) {
      if ((e as Error).name === "AbortError") {
        throw new Error("Scan timed out");
      }
      throw e;
    } finally {
      clearTimeout(timer);
    }
  }
}

export async function getProvider(): Promise<InferenceProvider> {
  const { apiBaseUrl, apiToken } = await loadSettings();
  return new RemoteApiProvider(apiBaseUrl || "http://localhost:8000", apiToken || undefined);
}

export async function pingBackend(baseUrl?: string): Promise<boolean> {
  const settings = await loadSettings();
  const url = (baseUrl || settings.apiBaseUrl || "http://localhost:8000").replace(/\/$/, "");
  try {
    const res = await fetch(`${url}/health`, { signal: AbortSignal.timeout(4000) });
    return res.ok;
  } catch {
    return false;
  }
}
