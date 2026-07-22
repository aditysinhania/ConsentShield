/** Client-side screenshot annotation overlay helpers. */

export type AnnotationKind = "banner" | "accept" | "reject" | "settings" | "overlay" | "checkbox" | "other";

export interface AnnotationBox {
  id: string;
  kind: AnnotationKind;
  label: string;
  confidence: number;
  x: number;
  y: number;
  width: number;
  height: number;
}

const COLORS: Record<AnnotationKind, string> = {
  banner: "#2563eb",
  accept: "#16a34a",
  reject: "#dc2626",
  settings: "#d97706",
  overlay: "#7c3aed",
  checkbox: "#0891b2",
  other: "#64748b",
};

export function annotationColor(kind: AnnotationKind): string {
  return COLORS[kind];
}

export function buildAnnotations(css?: Record<string, unknown> | null): AnnotationBox[] {
  if (!css) return [];
  const boxes: AnnotationBox[] = [];
  const banner = css.banner as Record<string, unknown> | undefined;
  if (banner && (banner.width || banner.boundingRect)) {
    const br = (banner.boundingRect || banner) as Record<string, number>;
    boxes.push({
      id: "banner",
      kind: Number(banner.coverage) >= 0.45 ? "overlay" : "banner",
      label: "Cookie Banner",
      confidence: 0.85,
      x: Number(br.x ?? br.left ?? 0),
      y: Number(br.y ?? br.top ?? 0),
      width: Number(br.width ?? 0),
      height: Number(br.height ?? 0),
    });
  }

  const buttons = (css.buttons as Array<Record<string, unknown>>) || [];
  buttons.forEach((btn, i) => {
    const role = String(btn.consentRole || "").toLowerCase();
    const text = String(btn.text || btn.ariaLabel || "Control");
    let kind: AnnotationKind = "other";
    if (role === "accept" || /\baccept|agree|allow\b/i.test(text)) kind = "accept";
    else if (role === "reject" || /\breject|decline|no.?thank|necessary\b/i.test(text)) kind = "reject";
    else if (role === "settings" || /\bmanage|settings|preferences|custom/i.test(text)) kind = "settings";
    else if (!(btn.inConsentContainer || role)) return;

    const br = (btn.boundingRect || btn) as Record<string, number>;
    const w = Number(br.width ?? btn.width ?? 0);
    const h = Number(br.height ?? btn.height ?? 0);
    if (w <= 0 || h <= 0) return;
    boxes.push({
      id: `btn-${i}`,
      kind,
      label: text.slice(0, 40),
      confidence: kind === "other" ? 0.5 : 0.8,
      x: Number(br.x ?? br.left ?? btn.x ?? 0),
      y: Number(br.y ?? br.top ?? btn.y ?? 0),
      width: w,
      height: h,
    });
  });

  const checks = (css.checkboxes as Array<Record<string, unknown>>) || [];
  checks.slice(0, 12).forEach((c, i) => {
    boxes.push({
      id: `chk-${i}`,
      kind: "checkbox",
      label: String(c.label || c.name || "Checkbox").slice(0, 40),
      confidence: c.checked ? 0.75 : 0.55,
      x: 0,
      y: 0,
      width: 0,
      height: 0,
    });
  });

  return boxes.filter((b) => b.kind === "checkbox" || (b.width > 0 && b.height > 0));
}
