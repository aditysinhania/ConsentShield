/** Human-readable explanations + detection source labels. */

import type { ExplainableReport } from "@consentshield/shared";

export type DetectorSource = "Rule Engine" | "Sentence Transformer" | "CLIP" | "Fusion" | "DOM";

export interface ExplainedFinding {
  id: string;
  title: string;
  detail: string;
  source: DetectorSource;
  severity?: number;
}

export function detectLimitedIframe(report: ExplainableReport, css?: Record<string, unknown> | null): boolean {
  const snap = css || {};
  const iframes = (snap.iframes as Array<Record<string, unknown>>) || [];
  const crossCmp = iframes.some((f) => f.cross_origin && (f.likely_cmp || f.cmp_vendor));
  const debug = report.debug as { reasons?: string[]; cross_origin_cmp_iframes?: number } | null | undefined;
  const reasonHit = (debug?.reasons || []).some((r) => /iframe/i.test(r));
  const noHits = (report.evidence || []).filter((e) => e.source === "rules").length === 0;
  const cmpDetected = Boolean((snap.cmp as { detected?: boolean } | undefined)?.detected);
  return Boolean(crossCmp || reasonHit || (cmpDetected && noHits && report.risk_score === 0 && (debug?.cross_origin_cmp_iframes || 0) > 0));
}

export function buildFindings(report: ExplainableReport): ExplainedFinding[] {
  const out: ExplainedFinding[] = [];

  for (const item of report.evidence || []) {
    const source: DetectorSource =
      item.source === "rules"
        ? "Rule Engine"
        : item.source === "text"
          ? "Sentence Transformer"
          : item.source === "vision"
            ? "CLIP"
            : item.source === "fusion"
              ? "Fusion"
              : "DOM";
    out.push({
      id: item.id,
      title: item.statement,
      detail:
        item.explanation ||
        item.user_impact ||
        "Detected from collected page evidence.",
      source,
      severity: item.severity,
    });
  }

  const ai = report.ai_analysis as
    | {
        nlp?: { findings?: Array<Record<string, unknown>> };
        vision?: { findings?: Array<Record<string, unknown>> };
      }
    | null
    | undefined;

  for (const f of ai?.nlp?.findings || []) {
    const pattern = String(f.predicted_pattern || "");
    if (!pattern) continue;
    if (out.some((o) => o.detail.includes(pattern))) continue;
    out.push({
      id: `nlp:${pattern}`,
      title: `AI found semantic similarity to ${formatPattern(pattern)}.`,
      detail: `Matched example: “${String(f.matched_example || "")}” (similarity ${pct(f.embedding_similarity ?? f.confidence)}).`,
      source: "Sentence Transformer",
      severity: Number(f.confidence) || 0.4,
    });
  }

  const vision = report.vision as { banner_detected?: boolean; layout?: { overlay_score?: number } } | null;
  if (vision?.banner_detected && !out.some((o) => /banner|overlay/i.test(o.title))) {
    const cov = vision.layout?.overlay_score;
    out.push({
      id: "vision:banner",
      title:
        typeof cov === "number" && cov > 0
          ? `Consent banner / overlay signals detected (coverage ~${Math.round(cov * 100)}%).`
          : "CLIP / vision detected a consent banner or dialog.",
      detail: "Visual analysis corroborated the presence of a consent interface.",
      source: "CLIP",
      severity: 0.45,
    });
  }

  if (report.fusion?.message && /agree|assist|rule-dominant/i.test(String(report.fusion.message))) {
    const rulesReady = (report.rules as { hits?: unknown[] } | null)?.hits?.length;
    const aiReady =
      (report.text as { status?: string } | null)?.status === "ready" ||
      (report.vision as { status?: string } | null)?.status === "ready";
    if (rulesReady && aiReady) {
      out.push({
        id: "fusion:agree",
        title: "The Rule Engine and AI agree.",
        detail: String(report.fusion.message),
        source: "Fusion",
        severity: 0.3,
      });
    }
  }

  return out;
}

export function formatPattern(id: string): string {
  return id
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function pct(v: unknown): string {
  const n = Number(v);
  if (Number.isNaN(n)) return "—";
  return `${Math.round(n * 100)}%`;
}

export function modelsUsed(report: ExplainableReport): string[] {
  const fromReport = (report as { models_used?: string[] }).models_used;
  if (Array.isArray(fromReport) && fromReport.length > 0) {
    return fromReport;
  }
  const models: string[] = ["Rule Engine"];
  const text = report.text as {
    status?: string;
    backend?: string;
    message?: string;
  } | null | undefined;
  const vision = report.vision as { status?: string } | null | undefined;
  if (text?.status === "ready") {
    const blob = `${text.backend || ""} ${text.message || ""}`;
    if (/finetuned.?minilm|Fine-tuned MiniLM/i.test(blob)) {
      models.push("NLP (Fine-tuned MiniLM)");
    } else {
      models.push("Sentence Transformer");
    }
  } else if (text?.status === "not_loaded") {
    models.push("NLP (stub)");
  }
  if (vision?.status === "ready") models.push("CLIP");
  else if (vision?.status === "not_loaded") models.push("Vision (stub)");
  models.push("Fusion");
  return models;
}

export function cmpInfo(css?: Record<string, unknown> | null): { vendor: string; confidence: number } {
  const cmp = (css?.cmp || {}) as { detected?: boolean; vendor?: string; name?: string; confidence?: number };
  if (!cmp.detected && !cmp.vendor && !cmp.name) {
    return { vendor: "Unknown", confidence: 0 };
  }
  return {
    vendor: String(cmp.vendor || cmp.name || "Unknown"),
    confidence: Number(cmp.confidence) || (cmp.detected ? 0.7 : 0),
  };
}

export function iframeLimitationCopy(vendor?: string): { title: string; body: string; reason: string } {
  return {
    title: "Consent interface detected",
    body: "Analysis limited by browser security",
    reason:
      `This consent interface${vendor && vendor !== "Unknown" ? ` (${vendor})` : ""} is embedded inside a protected cross-origin iframe that Chrome extensions cannot inspect.`,
  };
}
