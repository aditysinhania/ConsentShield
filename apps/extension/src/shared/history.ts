/** Scan history — latest 25 entries in chrome.storage.local. */

import type { ExplainableReport } from "@consentshield/shared";

export const HISTORY_KEY = "scanHistory";
export const HISTORY_LIMIT = 25;

export interface HistoryEntry {
  id: string;
  url: string;
  title: string;
  hostname: string;
  scannedAt: string;
  riskScore: number;
  category: string;
  confidence: number;
  scanDurationMs: number;
  patterns: string[];
  report: ExplainableReport;
  screenshotDataUrl?: string | null;
  cssSnapshot?: Record<string, unknown> | null;
  limitedByIframe?: boolean;
}

function hostnameOf(url: string): string {
  try {
    return new URL(url).hostname;
  } catch {
    return url;
  }
}

export async function loadHistory(): Promise<HistoryEntry[]> {
  const { scanHistory } = await chrome.storage.local.get(HISTORY_KEY);
  return Array.isArray(scanHistory) ? (scanHistory as HistoryEntry[]) : [];
}

export async function saveHistoryEntry(entry: Omit<HistoryEntry, "id" | "hostname"> & { id?: string }): Promise<HistoryEntry> {
  const list = await loadHistory();
  const full: HistoryEntry = {
    ...entry,
    id: entry.id || crypto.randomUUID(),
    hostname: hostnameOf(entry.url),
  };
  const next = [full, ...list.filter((e) => e.url !== full.url || e.scannedAt !== full.scannedAt)].slice(
    0,
    HISTORY_LIMIT,
  );
  await chrome.storage.local.set({ [HISTORY_KEY]: next });
  return full;
}

export async function getHistoryEntry(id: string): Promise<HistoryEntry | null> {
  const list = await loadHistory();
  return list.find((e) => e.id === id) || null;
}

export async function deleteHistoryEntry(id: string): Promise<void> {
  const list = await loadHistory();
  await chrome.storage.local.set({ [HISTORY_KEY]: list.filter((e) => e.id !== id) });
}

export async function clearHistory(): Promise<void> {
  await chrome.storage.local.set({ [HISTORY_KEY]: [] });
}
