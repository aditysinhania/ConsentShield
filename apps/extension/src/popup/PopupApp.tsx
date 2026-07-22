import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ExplainableReport } from "@consentshield/shared";
import { ScoreRing } from "./components/ScoreRing";
import { ProgressSteps } from "./components/ProgressSteps";
import { classifyError } from "../shared/errors";
import {
  buildFindings,
  cmpInfo,
  detectLimitedIframe,
  iframeLimitationCopy,
  modelsUsed,
} from "../shared/explanations";
import { downloadBlob, exportReport } from "../shared/export";
import {
  clearHistory,
  deleteHistoryEntry,
  loadHistory,
  saveHistoryEntry,
  type HistoryEntry,
} from "../shared/history";
import { riskLabel } from "../shared/risk";
import { loadSettings, type ExtensionSettings } from "../shared/settings";
import { applyTheme } from "../shared/theme";
import "../shared/ui.css";
import "./popup.css";

type View = "home" | "history";

export function PopupApp() {
  const [settings, setSettings] = useState<ExtensionSettings | null>(null);
  const [view, setView] = useState<View>("home");
  const [loading, setLoading] = useState(false);
  const [progressIndex, setProgressIndex] = useState(0);
  const [progressMsg, setProgressMsg] = useState("");
  const [error, setError] = useState<{ title: string; message: string } | null>(null);
  const [report, setReport] = useState<ExplainableReport | null>(null);
  const [screenshot, setScreenshot] = useState<string | null>(null);
  const [cssSnapshot, setCssSnapshot] = useState<Record<string, unknown> | null>(null);
  const [pageUrl, setPageUrl] = useState("");
  const [pageTitle, setPageTitle] = useState("");
  const [scanMs, setScanMs] = useState(0);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [toast, setToast] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  const autoStarted = useRef(false);

  useEffect(() => {
    void (async () => {
      const s = await loadSettings();
      setSettings(s);
      applyTheme(s.theme);
      setHistory(await loadHistory());
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      setPageUrl(tab?.url || "");
      setPageTitle(tab?.title || "");
    })();
  }, []);

  const showToast = useCallback((msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 2400);
  }, []);

  const limitedIframe = useMemo(
    () => (report ? detectLimitedIframe(report, cssSnapshot) : false),
    [report, cssSnapshot],
  );

  const findings = useMemo(() => (report ? buildFindings(report) : []), [report]);
  const patterns = useMemo(() => {
    const clusters = (report?.pattern_clusters || [])
      .map((c) => String((c as { cluster?: string }).cluster || ""))
      .filter(Boolean);
    if (clusters.length) return clusters;
    return findings.slice(0, 6).map((f) => f.title);
  }, [report, findings]);

  const cmp = useMemo(() => cmpInfo(cssSnapshot), [cssSnapshot]);
  const models = useMemo(() => (report ? modelsUsed(report) : []), [report]);

  const displayConfidence = useMemo(() => {
    if (!report) return 0;
    if (limitedIframe) return Math.min(report.confidence, 0.45);
    return report.confidence;
  }, [report, limitedIframe]);

  const runScan = useCallback(async () => {
    setLoading(true);
    setError(null);
    setReport(null);
    setProgressIndex(0);
    setProgressMsg("Scanning page…");
    const t0 = performance.now();
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab?.id) throw new Error("No active tab");
      setPageUrl(tab.url || "");
      setPageTitle(tab.title || "");

      const port = chrome.runtime.connect({ name: "scan" });
      const result = await new Promise<{
        ok: boolean;
        report?: ExplainableReport;
        screenshotDataUrl?: string | null;
        cssSnapshot?: Record<string, unknown> | null;
        collectionDurationMs?: number;
        error?: string;
      }>((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error("Scan timed out"));
          port.disconnect();
        }, 130_000);

        port.onMessage.addListener((msg) => {
          if (msg?.type === "SCAN_PROGRESS") {
            setProgressIndex(Number(msg.index) || 0);
            setProgressMsg(String(msg.message || ""));
          }
          if (msg?.type === "SCAN_RESULT") {
            clearTimeout(timeout);
            resolve(msg);
            port.disconnect();
          }
        });
        port.onDisconnect.addListener(() => {
          clearTimeout(timeout);
        });
        port.postMessage({ type: "START_SCAN", tabId: tab.id });
      });

      if (!result.ok || !result.report) throw new Error(result.error || "Scan failed");

      const nextReport = result.report;
      const css = result.cssSnapshot || null;
      const shot = result.screenshotDataUrl || null;
      setReport(nextReport);
      setCssSnapshot(css);
      setScreenshot(shot);
      setScanMs(performance.now() - t0);

      const s = settings || (await loadSettings());
      if (s.enableHistory) {
        await saveHistoryEntry({
          url: tab.url || nextReport.scan_id || "unknown",
          title: tab.title || "",
          scannedAt: new Date().toISOString(),
          riskScore: nextReport.risk_score,
          category: String(nextReport.category),
          confidence: nextReport.confidence,
          scanDurationMs: performance.now() - t0,
          patterns: (nextReport.pattern_clusters || []).map((c) =>
            String((c as { cluster?: string }).cluster || ""),
          ),
          report: nextReport,
          screenshotDataUrl: shot,
          cssSnapshot: css,
          limitedByIframe: detectLimitedIframe(nextReport, css),
        });
        setHistory(await loadHistory());
      }
    } catch (e) {
      setError(classifyError(e));
    } finally {
      setLoading(false);
    }
  }, [settings]);

  useEffect(() => {
    if (!settings?.automaticScan || autoStarted.current) return;
    autoStarted.current = true;
    void runScan();
  }, [settings, runScan]);

  async function openDetails() {
    if (!report) return;
    const payload = {
      report,
      screenshotDataUrl: screenshot,
      cssSnapshot,
      pageUrl,
      pageTitle,
      scanMs,
      limitedByIframe: limitedIframe,
    };
    await chrome.storage.session.set({ detailsPayload: payload });
    await chrome.tabs.create({ url: chrome.runtime.getURL("details.html") });
  }

  async function openSettings() {
    await chrome.tabs.create({ url: chrome.runtime.getURL("settings.html") });
  }

  async function doExport() {
    if (!report?.scan_id) {
      showToast("Scan ID missing — re-scan to export");
      return;
    }
    setExporting(true);
    try {
      const s = settings || (await loadSettings());
      const { blob, filename } = await exportReport(report.scan_id, s.exportFormat);
      downloadBlob(blob, filename);
      showToast(`${s.exportFormat.toUpperCase()} exported`);
    } catch (e) {
      setError(classifyError(e));
    } finally {
      setExporting(false);
    }
  }

  async function openHistoryEntry(entry: HistoryEntry) {
    setReport(entry.report);
    setScreenshot(entry.screenshotDataUrl || null);
    setCssSnapshot(entry.cssSnapshot || null);
    setPageUrl(entry.url);
    setPageTitle(entry.title);
    setScanMs(entry.scanDurationMs);
    setView("home");
  }

  const host = useMemo(() => {
    try {
      return pageUrl ? new URL(pageUrl).hostname : "—";
    } catch {
      return pageUrl || "—";
    }
  }, [pageUrl]);

  const iframeCopy = iframeLimitationCopy(cmp.vendor);

  return (
    <div className="popup-shell">
      <header className="popup-header fade-in">
        <div>
          <p className="brand" aria-hidden="true">
            ConsentShield
          </p>
          <h1 className="sr-only">ConsentShield</h1>
          <p className="site" title={pageUrl}>
            {host}
          </p>
        </div>
        <div className="header-actions">
          <button
            type="button"
            className="btn btn-ghost"
            onClick={() => setView(view === "history" ? "home" : "history")}
            aria-label="Scan history"
            aria-pressed={view === "history"}
          >
            History
          </button>
          <button type="button" className="btn btn-ghost" onClick={() => void openSettings()} aria-label="Open settings">
            Settings
          </button>
        </div>
      </header>

      {view === "history" ? (
        <section className="card fade-in history-panel" aria-label="Scan history">
          <div className="row-between">
            <h2>Recent scans</h2>
            <button
              type="button"
              className="btn btn-danger"
              onClick={() => void clearHistory().then(async () => setHistory(await loadHistory()))}
            >
              Clear
            </button>
          </div>
          {history.length === 0 && <p className="meta">No scans yet.</p>}
          <ul className="history-list">
            {history.map((h) => (
              <li key={h.id}>
                <button type="button" className="history-item" onClick={() => void openHistoryEntry(h)}>
                  <strong>{h.hostname}</strong>
                  <span className="meta">
                    {new Date(h.scannedAt).toLocaleString()} · risk {Math.round(h.riskScore)} ·{" "}
                    {Math.round(h.confidence * 100)}%
                  </span>
                </button>
                <button
                  type="button"
                  className="btn btn-ghost"
                  aria-label={`Delete scan of ${h.hostname}`}
                  onClick={() => void deleteHistoryEntry(h.id).then(async () => setHistory(await loadHistory()))}
                >
                  ✕
                </button>
              </li>
            ))}
          </ul>
        </section>
      ) : (
        <>
          {!loading && !report && !error && (
            <section className="card fade-in hero-empty">
              <p>Scan the active tab for consent dark patterns using the Rule Engine and optional AI models.</p>
              <button type="button" className="btn btn-primary" onClick={() => void runScan()}>
                Analyze this page
              </button>
            </section>
          )}

          {loading && <ProgressSteps activeIndex={progressIndex} message={progressMsg} />}

          {error && !loading && (
            <div className="card error-card fade-in" role="alert">
              <strong>{error.title}</strong>
              <p>{error.message}</p>
              <button type="button" className="btn btn-secondary" onClick={() => void runScan()}>
                Try again
              </button>
            </div>
          )}

          {report && !loading && (
            <div className="results fade-in">
              {limitedIframe && (
                <section className="card iframe-card" role="status">
                  <span className="chip">Limited analysis</span>
                  <h2>{iframeCopy.title}</h2>
                  <p className="emphasis">{iframeCopy.body}</p>
                  <p className="meta">{iframeCopy.reason}</p>
                  <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={() => {
                      const md = `# ConsentShield — Limited analysis\n\n**Site:** ${pageUrl}\n\n## ${iframeCopy.title}\n\n${iframeCopy.body}\n\n**Reason:** ${iframeCopy.reason}\n\n**Confidence:** ${Math.round(displayConfidence * 100)}% (reduced due to iframe isolation)\n\n**CMP:** ${cmp.vendor}\n`;
                      downloadBlob(new Blob([md], { type: "text/markdown" }), "consentshield-iframe-limitation.md");
                      showToast("Limitation note exported");
                    }}
                  >
                    Export explanation
                  </button>
                </section>
              )}

              <section className="card score-card">
                <ScoreRing score={report.risk_score} />
                <div className="score-meta">
                  <p className="risk-tier">{riskLabel(report.risk_score)}</p>
                  <h2>{String(report.category)}</h2>
                  <p className="meta">
                    Confidence {Math.round(displayConfidence * 100)}%
                    {limitedIframe ? " · reduced (iframe)" : ""}
                  </p>
                  {cmp.vendor !== "Unknown" && (
                    <p className="meta">
                      CMP: <strong>{cmp.vendor}</strong> ({Math.round(cmp.confidence * 100)}%)
                    </p>
                  )}
                  <p className="meta">Scan time {(scanMs / 1000).toFixed(1)}s</p>
                </div>
              </section>

              <section className="card">
                <h3>Detected patterns</h3>
                {patterns.length === 0 ? (
                  <p className="meta">
                    {limitedIframe ? "Patterns unavailable due to iframe isolation." : "No pattern clusters triggered."}
                  </p>
                ) : (
                  <ul className="chip-row">
                    {patterns.map((p) => (
                      <li key={p} className="chip">
                        {p}
                      </li>
                    ))}
                  </ul>
                )}
              </section>

              <section className="card">
                <h3>Models used</h3>
                <ul className="chip-row">
                  {models.map((m) => (
                    <li key={m} className="chip chip-muted">
                      {m}
                    </li>
                  ))}
                </ul>
              </section>

              {findings.length > 0 && (
                <section className="card">
                  <h3>Key findings</h3>
                  <ul className="finding-list">
                    {findings.slice(0, 4).map((f) => (
                      <li key={f.id}>
                        <div className="finding-title">{f.title}</div>
                        <div className="meta">{f.detail}</div>
                        <span className="chip chip-muted">{f.source}</span>
                      </li>
                    ))}
                  </ul>
                </section>
              )}

              <div className="actions">
                <button type="button" className="btn btn-primary" onClick={() => void runScan()}>
                  Analyze again
                </button>
                <button type="button" className="btn btn-secondary" onClick={() => void openDetails()}>
                  View details
                </button>
                <button type="button" className="btn btn-secondary" disabled={exporting} onClick={() => void doExport()}>
                  Export report
                </button>
              </div>
            </div>
          )}
        </>
      )}

      {toast && (
        <div className="toast" role="status" aria-live="polite">
          {toast}
        </div>
      )}
    </div>
  );
}
