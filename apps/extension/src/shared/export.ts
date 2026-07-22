/** Export helpers — fetch report formats from the API. */

import type { ExportFormat } from "./settings";
import { loadSettings } from "./settings";

export async function exportReport(
  scanId: string,
  format: ExportFormat,
): Promise<{ blob: Blob; filename: string }> {
  const { apiBaseUrl, apiToken } = await loadSettings();
  const headers: Record<string, string> = {};
  if (apiToken) headers.Authorization = `Bearer ${apiToken}`;

  if (format === "json") {
    const res = await fetch(`${apiBaseUrl}/api/v1/report/${scanId}`, { headers });
    if (!res.ok) throw new Error(`Export failed (${res.status})`);
    const data = await res.json();
    return {
      blob: new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
      filename: `consentshield-${scanId}.json`,
    };
  }

  const path =
    format === "html"
      ? `/api/v1/report/${scanId}/html`
      : format === "markdown"
        ? `/api/v1/report/${scanId}/markdown`
        : `/api/v1/report/${scanId}/pdf`;

  const res = await fetch(`${apiBaseUrl}${path}`, { headers });
  if (!res.ok) throw new Error(`Export failed (${res.status})`);
  const blob = await res.blob();
  const ext = format === "markdown" ? "md" : format;
  return { blob, filename: `consentshield-${scanId}.${ext}` };
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1500);
}
