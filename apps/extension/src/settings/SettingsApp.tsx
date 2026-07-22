import { useEffect, useState } from "react";
import {
  DEFAULT_SETTINGS,
  loadSettings,
  resetSettings,
  saveSettings,
  type ExtensionSettings,
  type ExportFormat,
  type ThemeMode,
} from "../shared/settings";
import { applyTheme } from "../shared/theme";
import { pingBackend } from "../shared/inference";
import "../shared/ui.css";
import "./settings.css";

export function SettingsApp() {
  const [settings, setSettings] = useState<ExtensionSettings>(DEFAULT_SETTINGS);
  const [toast, setToast] = useState<string | null>(null);
  const [health, setHealth] = useState<"unknown" | "ok" | "down">("unknown");

  useEffect(() => {
    void (async () => {
      const s = await loadSettings();
      setSettings(s);
      applyTheme(s.theme);
      setHealth((await pingBackend(s.apiBaseUrl)) ? "ok" : "down");
    })();
  }, []);

  async function update<K extends keyof ExtensionSettings>(key: K, value: ExtensionSettings[K]) {
    const next = await saveSettings({ [key]: value });
    setSettings(next);
    if (key === "theme") applyTheme(next.theme);
    setToast("Saved");
    setTimeout(() => setToast(null), 1400);
  }

  return (
    <main className="settings-shell">
      <header className="fade-in">
        <p className="brand">ConsentShield</p>
        <h1>Settings</h1>
        <p className="meta">
          Backend:{" "}
          {health === "ok" ? "online" : health === "down" ? "unreachable" : "checking…"} · {settings.apiBaseUrl}
        </p>
      </header>

      <section className="card fade-in">
        <h2>Connection</h2>
        <label className="field">
          API base URL
          <input
            value={settings.apiBaseUrl}
            onChange={(e) => setSettings({ ...settings, apiBaseUrl: e.target.value })}
            onBlur={() => void update("apiBaseUrl", settings.apiBaseUrl)}
            aria-label="API base URL"
          />
        </label>
        <label className="field">
          API token (optional)
          <input
            type="password"
            value={settings.apiToken}
            onChange={(e) => setSettings({ ...settings, apiToken: e.target.value })}
            onBlur={() => void update("apiToken", settings.apiToken)}
            aria-label="API token"
            autoComplete="off"
          />
        </label>
        <button
          type="button"
          className="btn btn-secondary"
          onClick={() => void pingBackend(settings.apiBaseUrl).then((ok) => setHealth(ok ? "ok" : "down"))}
        >
          Test connection
        </button>
      </section>

      <section className="card fade-in">
        <h2>Features</h2>
        <Toggle
          label="Enable AI"
          checked={settings.enableAi}
          onChange={(v) => void update("enableAi", v)}
          hint="Uses backend NLP/Vision when AI_STUB_MODE=false"
        />
        <Toggle label="Enable screenshots" checked={settings.enableScreenshots} onChange={(v) => void update("enableScreenshots", v)} />
        <Toggle label="Enable history" checked={settings.enableHistory} onChange={(v) => void update("enableHistory", v)} />
        <Toggle label="Automatic scan on open" checked={settings.automaticScan} onChange={(v) => void update("automaticScan", v)} />
        <Toggle
          label="Show advanced details by default"
          checked={settings.showAdvancedDetails}
          onChange={(v) => void update("showAdvancedDetails", v)}
        />
        <Toggle label="Developer mode" checked={settings.developerMode} onChange={(v) => void update("developerMode", v)} />
      </section>

      <section className="card fade-in">
        <h2>Appearance & export</h2>
        <label className="field">
          Theme
          <select
            value={settings.theme}
            onChange={(e) => void update("theme", e.target.value as ThemeMode)}
            aria-label="Theme"
          >
            <option value="system">System</option>
            <option value="light">Light</option>
            <option value="dark">Dark</option>
          </select>
        </label>
        <label className="field">
          Default export format
          <select
            value={settings.exportFormat}
            onChange={(e) => void update("exportFormat", e.target.value as ExportFormat)}
            aria-label="Export format"
          >
            <option value="pdf">PDF</option>
            <option value="html">HTML</option>
            <option value="markdown">Markdown</option>
            <option value="json">JSON</option>
          </select>
        </label>
        <label className="field">
          Confidence threshold ({Math.round(settings.confidenceThreshold * 100)}%)
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={settings.confidenceThreshold}
            onChange={(e) => void update("confidenceThreshold", Number(e.target.value))}
            aria-label="Confidence threshold"
          />
        </label>
      </section>

      <section className="card fade-in">
        <h2>Reset</h2>
        <button
          type="button"
          className="btn btn-danger"
          onClick={() =>
            void resetSettings().then((s) => {
              setSettings(s);
              applyTheme(s.theme);
              setToast("Settings reset");
            })
          }
        >
          Reset settings
        </button>
      </section>

      {toast && <div className="toast">{toast}</div>}
    </main>
  );
}

function Toggle({
  label,
  checked,
  onChange,
  hint,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  hint?: string;
}) {
  const id = label.toLowerCase().replace(/\s+/g, "-");
  return (
    <div className="toggle-row">
      <div>
        <label htmlFor={id}>{label}</label>
        {hint && <p className="meta">{hint}</p>}
      </div>
      <input id={id} type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
    </div>
  );
}
