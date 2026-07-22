/** ConsentShield extension settings (chrome.storage.local). */

export type ThemeMode = "light" | "dark" | "system";
export type ExportFormat = "pdf" | "html" | "markdown" | "json";

export interface ExtensionSettings {
  apiBaseUrl: string;
  apiToken: string;
  enableAi: boolean;
  enableScreenshots: boolean;
  enableHistory: boolean;
  theme: ThemeMode;
  exportFormat: ExportFormat;
  confidenceThreshold: number;
  automaticScan: boolean;
  showAdvancedDetails: boolean;
  developerMode: boolean;
}

export const DEFAULT_SETTINGS: ExtensionSettings = {
  apiBaseUrl: "http://localhost:8000",
  apiToken: "",
  enableAi: true,
  enableScreenshots: true,
  enableHistory: true,
  theme: "system",
  exportFormat: "pdf",
  confidenceThreshold: 0.35,
  automaticScan: false,
  showAdvancedDetails: false,
  developerMode: false,
};

export async function loadSettings(): Promise<ExtensionSettings> {
  const stored = await chrome.storage.local.get(Object.keys(DEFAULT_SETTINGS));
  return { ...DEFAULT_SETTINGS, ...(stored as Partial<ExtensionSettings>) };
}

export async function saveSettings(patch: Partial<ExtensionSettings>): Promise<ExtensionSettings> {
  const next = { ...(await loadSettings()), ...patch };
  await chrome.storage.local.set(next);
  return next;
}

export async function resetSettings(): Promise<ExtensionSettings> {
  await chrome.storage.local.set(DEFAULT_SETTINGS);
  return { ...DEFAULT_SETTINGS };
}
