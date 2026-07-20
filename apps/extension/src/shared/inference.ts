/**
 * Inference provider abstraction — Remote API now; local on-device later.
 */
export interface InferenceProvider {
  analyze(payload: unknown): Promise<unknown>;
}

export class RemoteApiProvider implements InferenceProvider {
  constructor(private baseUrl: string) {}

  async analyze(payload: unknown): Promise<unknown> {
    const { apiToken } = await chrome.storage.local.get(["apiToken"]);
    const res = await fetch(`${this.baseUrl}/api/v1/scan`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(apiToken ? { Authorization: `Bearer ${apiToken}` } : {}),
      },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Scan failed (${res.status}): ${text}`);
    }
    const scan = await res.json();
    const reportRes = await fetch(`${this.baseUrl}/api/v1/report/${scan.id}`, {
      headers: {
        ...(apiToken ? { Authorization: `Bearer ${apiToken}` } : {}),
      },
    });
    if (!reportRes.ok) {
      throw new Error(`Report fetch failed (${reportRes.status})`);
    }
    return reportRes.json();
  }
}

/** Future: run models inside the extension / WASM / native messaging. */
export class LocalOnDeviceProvider implements InferenceProvider {
  async analyze(_payload: unknown): Promise<unknown> {
    throw new Error("Local inference is not implemented yet.");
  }
}

export async function getProvider(): Promise<InferenceProvider> {
  const { apiBaseUrl } = await chrome.storage.local.get(["apiBaseUrl"]);
  return new RemoteApiProvider(apiBaseUrl || "http://localhost:8000");
}
