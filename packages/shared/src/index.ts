export type DarkPatternCategory =
  | "No Dark Pattern"
  | "Cookie Consent Manipulation"
  | "Hidden Subscription"
  | "Hidden Billing"
  | "Confirmshaming"
  | "Misleading Free Trial"
  | "Mixed Consent Manipulation"
  | "Unknown";

export interface EvidenceItem {
  id: string;
  statement: string;
  severity: number;
  source: "rules" | "vision" | "text" | "fusion" | "dom" | string;
  rule_id?: string | null;
  metadata?: Record<string, unknown>;
}

export interface CssButtonSnapshot {
  text?: string;
  ariaLabel?: string;
  width?: number;
  height?: number;
  fontSizePx?: number;
  fontWeight?: number | string;
  backgroundColor?: string;
  color?: string;
  display?: string;
  visibility?: string;
  opacity?: number;
  textDecoration?: string;
}

export interface CssSnapshot {
  buttons?: CssButtonSnapshot[];
  checkboxes?: Array<Record<string, unknown>>;
  toggles?: Array<Record<string, unknown>>;
  banner?: Record<string, unknown>;
  billingSection?: Record<string, unknown>;
  primaryCta?: Record<string, unknown>;
  subscriptionOptions?: Array<Record<string, unknown>>;
  body?: { fontSizePx?: number };
  layoutHints?: string[];
}

export interface ScanPayload {
  url: string;
  title?: string | null;
  html?: string | null;
  css_snapshot?: CssSnapshot | null;
  visible_text?: string | null;
  screenshot_base64?: string | null;
  viewport?: { width: number; height: number } | null;
  collected_at?: string | null;
}

export interface ExplainableReport {
  scan_id?: string;
  risk_score: number;
  category: DarkPatternCategory | string;
  confidence: number;
  evidence: EvidenceItem[];
  vision?: Record<string, unknown> | null;
  text?: Record<string, unknown> | null;
  rules?: Record<string, unknown> | null;
  fusion?: Record<string, unknown> | null;
  pipeline_notes?: string[];
  screenshot_path?: string | null;
}

export interface ScanSummary {
  id: string;
  url: string;
  title?: string | null;
  status: string;
  created_at: string;
  completed_at?: string | null;
}

export const API_ROUTES = {
  health: "/health",
  login: "/api/v1/auth/login",
  register: "/api/v1/auth/register",
  scan: "/api/v1/scan",
  report: (id: string) => `/api/v1/report/${id}`,
  rulesCatalog: "/api/v1/rules/catalog",
  modelsRegistry: "/api/v1/models/registry",
  feedback: "/api/v1/feedback",
} as const;
