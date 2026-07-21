import { useEffect, useState } from "react";
import type { ExplainableReport } from "@consentshield/shared";

export function PopupApp() {
  const [apiBaseUrl, setApiBaseUrl] = useState("http://localhost:8000");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<ExplainableReport | null>(null);

  useEffect(() => {
    void chrome.storage.local.get(["apiBaseUrl"]).then((v) => {
      if (v.apiBaseUrl) setApiBaseUrl(v.apiBaseUrl);
    });
  }, []);

  async function saveSettings() {
    await chrome.storage.local.set({ apiBaseUrl });
  }

  async function runScan() {
    setLoading(true);
    setError(null);
    setReport(null);
    await saveSettings();
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab?.id) throw new Error("No active tab");
      const response = await chrome.runtime.sendMessage({ type: "RUN_SCAN", tabId: tab.id });
      if (!response?.ok) throw new Error(response?.error || "Scan failed");
      setReport(response.report as ExplainableReport);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="shell">
      <header>
        <h1>ConsentShield</h1>
        <p>Consent dark-pattern scanner</p>
      </header>

      <label className="field">
        API URL
        <input value={apiBaseUrl} onChange={(e) => setApiBaseUrl(e.target.value)} />
      </label>

      <button className="primary" disabled={loading} onClick={() => void runScan()}>
        {loading ? "Scanning…" : "Scan this page"}
      </button>

      {error && <div className="error">{error}</div>}

      {report && (
        <section className="report">
          <div className="score">
            <strong>{Math.round(report.risk_score)}</strong>
            <span>Risk</span>
          </div>
          <p className="category">{report.category}</p>
          <p className="meta">Confidence {(report.confidence * 100).toFixed(0)}%</p>
          {report.confidence_breakdown && (
            <p className="meta">
              Breakdown t/v/l/cmp{" "}
              {(report.confidence_breakdown.text * 100).toFixed(0)}/
              {(report.confidence_breakdown.visual * 100).toFixed(0)}/
              {(report.confidence_breakdown.layout * 100).toFixed(0)}/
              {(report.confidence_breakdown.cmp * 100).toFixed(0)}
            </p>
          )}
          <ul>
            {(report.evidence || []).map((item) => (
              <li key={item.id}>
                <div>{item.statement}</div>
                {item.explanation && <div className="meta">{item.explanation}</div>}
                {item.recommendation && (
                  <div className="meta">Rec: {item.recommendation}</div>
                )}
              </li>
            ))}
          </ul>
          {(report.rule_traces || []).length > 0 && (
            <details>
              <summary>Rule traces ({report.rule_traces!.length})</summary>
              <ul>
                {report.rule_traces!.map((t) => (
                  <li key={t.rule_id}>
                    {t.rule_id} · risk {t.risk_contribution} · features{" "}
                    {t.features_used.slice(0, 3).join(", ")}
                  </li>
                ))}
              </ul>
            </details>
          )}
          {(report.pipeline_notes || []).length > 0 && (
            <details>
              <summary>Pipeline notes</summary>
              <ul>
                {report.pipeline_notes!.map((n) => (
                  <li key={n}>{n}</li>
                ))}
              </ul>
            </details>
          )}
        </section>
      )}
    </div>
  );
}
