export type DarkPatternCategory =
  | "No Dark Pattern"
  | "Cookie Consent Manipulation"
  | "Hidden Subscription"
  | "Hidden Billing"
  | "Confirmshaming"
  | "Misleading Free Trial"
  | "Mixed Consent Manipulation"
  | "Unknown";

export interface BoundingRect {
  x: number;
  y: number;
  width: number;
  height: number;
  top: number;
  left: number;
  right: number;
  bottom: number;
}

export interface ConfidenceBreakdown {
  text: number;
  visual: number;
  layout: number;
  cmp: number;
  agreement: number;
  final: number;
}

export interface RuleTrace {
  rule_id: string;
  features_used: string[];
  visual_score: number;
  text_score: number;
  layout_score: number;
  confidence_breakdown: ConfidenceBreakdown;
  risk_contribution: number;
}

export interface EvidenceItem {
  id: string;
  statement: string;
  severity: number;
  source: "rules" | "vision" | "text" | "fusion" | "dom" | string;
  rule_id?: string | null;
  metadata?: Record<string, unknown>;
  explanation?: string | null;
  user_impact?: string | null;
  gdpr_relevance?: string | null;
  recommendation?: string | null;
  xpath?: string | null;
  css_selector?: string | null;
  dom_path?: string | null;
  bounding_rect?: BoundingRect | null;
  viewport?: { width: number; height: number } | null;
  scroll_position?: { x: number; y: number } | null;
  html_snippet?: string | null;
  computed_styles?: Record<string, unknown> | null;
  timestamp?: string | null;
  url?: string | null;
  page_title?: string | null;
}

export interface CssButtonSnapshot {
  text?: string;
  ariaLabel?: string;
  width?: number;
  height?: number;
  x?: number;
  y?: number;
  top?: number;
  left?: number;
  right?: number;
  bottom?: number;
  boundingRect?: BoundingRect;
  fontSizePx?: number;
  fontWeight?: number | string;
  backgroundColor?: string;
  color?: string;
  display?: string;
  visibility?: string;
  opacity?: number;
  textDecoration?: string;
  xpath?: string;
  cssSelector?: string;
  domPath?: string;
  htmlSnippet?: string;
  computedStyles?: Record<string, unknown>;
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
  cmp?: Record<string, unknown>;
}

export interface ScanPayload {
  url: string;
  title?: string | null;
  html?: string | null;
  css_snapshot?: CssSnapshot | null;
  visible_text?: string | null;
  screenshot_base64?: string | null;
  viewport?: { width: number; height: number } | null;
  scroll_position?: { x: number; y: number } | null;
  collected_at?: string | null;
  collection_duration_ms?: number | null;
}

export interface ExplainableReport {
  scan_id?: string;
  risk_score: number;
  category: DarkPatternCategory | string;
  confidence: number;
  confidence_breakdown?: ConfidenceBreakdown | null;
  evidence: EvidenceItem[];
  rule_traces?: RuleTrace[];
  severity?: Record<string, unknown> | null;
  pattern_clusters?: Array<Record<string, unknown>>;
  accessibility?: Record<string, unknown> | null;
  timeline?: Array<Record<string, unknown>>;
  performance?: Record<string, unknown> | null;
  vision?: Record<string, unknown> | null;
  text?: Record<string, unknown> | null;
  rules?: Record<string, unknown> | null;
  fusion?: Record<string, unknown> | null;
  pipeline_notes?: string[];
  screenshot_path?: string | null;
  annotated_screenshot_path?: string | null;
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
  ruleTraces: (id: string) => `/api/v1/rules/${id}/traces`,
  modelsRegistry: "/api/v1/models/registry",
  feedback: "/api/v1/feedback",
} as const;
