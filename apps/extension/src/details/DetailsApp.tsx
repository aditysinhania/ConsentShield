import { useEffect, useMemo, useState, type ReactNode } from "react";
import type { ExplainableReport } from "@consentshield/shared";
import { AnnotatedScreenshot } from "../popup/components/AnnotatedScreenshot";
import { buildAnnotations } from "../shared/annotations";
import { buildFindings, cmpInfo, iframeLimitationCopy, modelsUsed } from "../shared/explanations";
import { downloadBlob, exportReport } from "../shared/export";
import { classifyError } from "../shared/errors";
import { loadSettings } from "../shared/settings";
import { applyTheme } from "../shared/theme";
import "../shared/ui.css";
import "./details.css";

interface DetailsPayload {
  report: ExplainableReport;
  screenshotDataUrl?: string | null;
  cssSnapshot?: Record<string, unknown> | null;
  pageUrl?: string;
  pageTitle?: string;
  scanMs?: number;
  limitedByIframe?: boolean;
}

function Section({ title, children, defaultOpen = false }: { title: string; children: ReactNode; defaultOpen?: boolean }) {
  return (
    <details className="collapsible card section" open={defaultOpen}>
      <summary>{title}</summary>
      <div className="section-body">{children}</div>
    </details>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span className="meta">{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export function DetailsApp() {
  const [payload, setPayload] = useState<DetailsPayload | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    void (async () => {
      const s = await loadSettings();
      applyTheme(s.theme);
      const { detailsPayload } = await chrome.storage.session.get("detailsPayload");
      if (detailsPayload) setPayload(detailsPayload as DetailsPayload);
    })();
  }, []);

  const report = payload?.report;
  const findings = useMemo(() => (report ? buildFindings(report) : []), [report]);
  const boxes = useMemo(() => buildAnnotations(payload?.cssSnapshot || null), [payload]);
  const cmp = useMemo(() => cmpInfo(payload?.cssSnapshot || null), [payload]);
  const perf = (report?.performance || {}) as Record<string, unknown>;
  const stages = (perf.stages || {}) as Record<string, number>;
  const bd = report?.confidence_breakdown;

  async function doExport(format: "pdf" | "html" | "markdown" | "json") {
    if (!report?.scan_id) {
      setErr("Missing scan id");
      return;
    }
    try {
      const { blob, filename } = await exportReport(report.scan_id, format);
      downloadBlob(blob, filename);
      setToast(`${format.toUpperCase()} exported`);
      setTimeout(() => setToast(null), 2200);
    } catch (e) {
      setErr(classifyError(e).message);
    }
  }

  if (!payload || !report) {
    return (
      <main className="details-shell">
        <p className="meta">No scan loaded. Run Analyze from the popup, then open View Details.</p>
      </main>
    );
  }

  const iframeNote = payload.limitedByIframe ? iframeLimitationCopy(cmp.vendor) : null;
  const text = report.text as { status?: string; evidence_spans?: Array<Record<string, unknown>>; message?: string } | null;
  const vision = report.vision as { status?: string; banner_detected?: boolean; layout?: Record<string, unknown>; message?: string } | null;
  const fusion = report.fusion as { message?: string; sources_used?: string[]; risk_score?: number } | null;
  const banner = (payload.cssSnapshot?.banner || {}) as Record<string, unknown>;
  const buttons = ((payload.cssSnapshot?.buttons || []) as Array<Record<string, unknown>>).slice(0, 20);

  return (
    <main className="details-shell">
      <header className="details-header fade-in">
        <div>
          <p className="brand">ConsentShield</p>
          <h1>Detailed analysis</h1>
          <p className="meta">{payload.pageTitle || payload.pageUrl}</p>
        </div>
        <div className="export-row">
          {(["pdf", "html", "markdown", "json"] as const).map((f) => (
            <button key={f} type="button" className="btn btn-secondary" onClick={() => void doExport(f)}>
              {f.toUpperCase()}
            </button>
          ))}
        </div>
      </header>

      {iframeNote && (
        <section className="card iframe-card fade-in" role="status">
          <span className="chip">Limited analysis</span>
          <h2>{iframeNote.title}</h2>
          <p className="emphasis">{iframeNote.body}</p>
          <p className="meta">{iframeNote.reason}</p>
        </section>
      )}

      {err && (
        <div className="card error-card" role="alert">
          {err}
        </div>
      )}

      <Section title="Summary" defaultOpen>
        <div className="metric-grid">
          <Metric label="Risk" value={String(Math.round(report.risk_score))} />
          <Metric label="Category" value={String(report.category)} />
          <Metric
            label="Confidence"
            value={`${Math.round((payload.limitedByIframe ? Math.min(report.confidence, 0.45) : report.confidence) * 100)}%`}
          />
          <Metric label="Scan time" value={`${((payload.scanMs || 0) / 1000).toFixed(1)}s`} />
          <Metric label="CMP" value={cmp.vendor} />
          <Metric label="Models" value={modelsUsed(report).join(", ")} />
        </div>
      </Section>

      <Section title="Detected patterns" defaultOpen>
        <ul className="chip-row">
          {(report.pattern_clusters || []).map((c, i) => (
            <li key={i} className="chip">
              {String((c as { cluster?: string }).cluster || "Pattern")}
            </li>
          ))}
          {(report.pattern_clusters || []).length === 0 && <li className="meta">No clusters.</li>}
        </ul>
      </Section>

      <Section title="Rule Engine findings" defaultOpen>
        <ul className="finding-list">
          {findings
            .filter((f) => f.source === "Rule Engine")
            .map((f) => (
              <li key={f.id}>
                <div className="finding-title">{f.title}</div>
                <div className="meta">{f.detail}</div>
                <span className="chip chip-muted">{f.source}</span>
              </li>
            ))}
          {findings.filter((f) => f.source === "Rule Engine").length === 0 && (
            <li className="meta">No rule hits.</li>
          )}
        </ul>
      </Section>

      <Section title="NLP findings">
        <p className="meta">Status: {text?.status || "unavailable"}</p>
        <p className="meta">{text?.message}</p>
        <ul className="finding-list">
          {(text?.evidence_spans || []).map((s, i) => (
            <li key={i}>
              <div className="finding-title">
                AI found semantic similarity to {String(s.predicted_pattern || "pattern")}.
              </div>
              <div className="meta">
                Example: “{String(s.matched_example || "")}” · similarity{" "}
                {Math.round(Number(s.embedding_similarity || s.confidence || 0) * 100)}%
              </div>
              <span className="chip chip-muted">Sentence Transformer</span>
            </li>
          ))}
        </ul>
      </Section>

      <Section title="Vision findings">
        <p className="meta">Status: {vision?.status || "unavailable"} · Banner: {String(vision?.banner_detected)}</p>
        <p className="meta">{vision?.message}</p>
        <pre className="code-block">{JSON.stringify(vision?.layout || {}, null, 2)}</pre>
      </Section>

      <Section title="Fusion decision">
        <p>{fusion?.message || "—"}</p>
        <p className="meta">Sources: {(fusion?.sources_used || []).join(", ") || "—"}</p>
        {(report.rules as { hits?: unknown[] } | null)?.hits?.length &&
          (text?.status === "ready" || vision?.status === "ready") && (
            <p className="emphasis">The Rule Engine and AI agree on consent-risk signals.</p>
          )}
      </Section>

      <Section title="Confidence breakdown">
        {bd ? (
          <div className="metric-grid">
            <Metric label="Rules" value={`${Math.round((bd.rules || 0) * 100)}%`} />
            <Metric label="NLP" value={`${Math.round((bd.nlp || bd.text || 0) * 100)}%`} />
            <Metric label="Vision" value={`${Math.round((bd.vision_model || bd.visual || 0) * 100)}%`} />
            <Metric label="CMP" value={`${Math.round((bd.cmp || 0) * 100)}%`} />
            <Metric label="Fusion" value={`${Math.round((bd.fusion || bd.final || 0) * 100)}%`} />
            <Metric label="Agreement" value={`${Math.round((bd.agreement || 0) * 100)}%`} />
          </div>
        ) : (
          <p className="meta">No breakdown.</p>
        )}
      </Section>

      <Section title="Consent provider">
        <p>
          <strong>{cmp.vendor}</strong> · confidence {Math.round(cmp.confidence * 100)}%
        </p>
      </Section>

      <Section title="Screenshot preview" defaultOpen>
        <AnnotatedScreenshot imageUrl={payload.screenshotDataUrl} boxes={boxes} />
      </Section>

      <Section title="Detected controls">
        <table className="data-table">
          <thead>
            <tr>
              <th>Text</th>
              <th>Role</th>
              <th>Size</th>
            </tr>
          </thead>
          <tbody>
            {buttons.map((b, i) => (
              <tr key={i}>
                <td>{String(b.text || "—")}</td>
                <td>{String(b.consentRole || "—")}</td>
                <td>
                  {String(b.width || 0)}×{String(b.height || 0)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Section>

      <Section title="Banner information">
        <pre className="code-block">{JSON.stringify(banner || {}, null, 2)}</pre>
      </Section>

      <Section title="Timing information">
        <div className="metric-grid">
          <Metric label="Collection" value={`${Number(perf.collection_ms || stages.collection || 0).toFixed(0)} ms`} />
          <Metric label="Rule engine" value={`${Number(perf.rule_engine_ms || stages.rule_engine || 0).toFixed(0)} ms`} />
          <Metric label="NLP" value={`${Number(stages.text || 0).toFixed(0)} ms`} />
          <Metric label="Vision" value={`${Number(stages.vision || 0).toFixed(0)} ms`} />
          <Metric label="Fusion" value={`${Number(perf.fusion_ms || stages.fusion || 0).toFixed(0)} ms`} />
          <Metric label="Total" value={`${Number(perf.total_ms || payload.scanMs || 0).toFixed(0)} ms`} />
        </div>
      </Section>

      <Section title="Performance metrics">
        <pre className="code-block">{JSON.stringify(perf, null, 2)}</pre>
        {"memory" in performance && (
          <p className="meta">
            JS heap:{" "}
            {Math.round((((performance as unknown as { memory?: { usedJSHeapSize: number } }).memory?.usedJSHeapSize || 0) / 1048576) * 10) / 10}{" "}
            MB
          </p>
        )}
      </Section>

      <Section title="Raw AI confidence">
        <pre className="code-block">{JSON.stringify(report.ai_analysis || {}, null, 2)}</pre>
      </Section>

      <Section title="DOM evidence">
        <ul className="finding-list">
          {(report.evidence || []).map((e) => (
            <li key={e.id}>
              <div className="finding-title">{e.statement}</div>
              <div className="meta">
                {e.css_selector || e.xpath || "—"} · {e.source}
              </div>
            </li>
          ))}
        </ul>
      </Section>

      <Section title="Reasoning">
        <ul className="finding-list">
          {findings.map((f) => (
            <li key={f.id}>
              <div className="finding-title">{f.title}</div>
              <div className="meta">{f.detail}</div>
              <span className="chip chip-muted">{f.source}</span>
            </li>
          ))}
        </ul>
        {(report.pipeline_notes || []).map((n) => (
          <p key={n} className="meta">
            {n}
          </p>
        ))}
      </Section>

      {toast && <div className="toast">{toast}</div>}
    </main>
  );
}
