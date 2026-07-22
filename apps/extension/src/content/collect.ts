import type { CssSnapshot, ScanPayload } from "@consentshield/shared";

/** Consent vocabulary — mirrors backend rule patterns (normalized, not site-specific). */
const ACCEPT_RE =
  /\b(accept(\s+all|\s+cookies|\s+selected)?|allow(\s+all|\s+cookies)?|agree(\s+to\s+all|\s+and\s+continue)?|i\s+agree|yes[,\s]+i\s+accept|continue\s+with\s+cookies|confirm\s+choices|got\s+it|ok[,\s]+i\s+agree)\b/i;
const REJECT_RE =
  /\b(reject(\s+all|\s+cookies|\s+non[- ]essential)?|decline(\s+all|\s+cookies)?|refuse(\s+all)?|deny(\s+all)?|disagree|no[,\s]+thank\s*you|no\s+thanks|only\s+necessary|necessary\s+only|essential\s+only|required\s+only)\b/i;
const SETTINGS_RE =
  /\b(manage\s*(preferences|cookies|settings|options)?|cookie\s*settings|privacy\s*settings|customise|customize|preferences|more\s*options|save\s+preferences|confirm\s+my\s+choices|cookie\s+preferences|consent\s+settings)\b/i;
const CONSENT_TOKEN_RE =
  /\b(cookie|consent|gdpr|privacy|cmp|ot-sdk|onetrust|sourcepoint|didomi|trustarc|cookiebot|usercentrics|quantcast|osano)\b/i;

type CmpSignature = {
  vendor: string;
  selectors?: string[];
  scriptPatterns?: RegExp[];
  iframePatterns?: RegExp[];
  globals?: string[];
};

const CMP_SIGNATURES: CmpSignature[] = [
  {
    vendor: "Sourcepoint",
    selectors: ["[id*='sp_message']", "[class*='sp_message']", "#sp_message_container", "[id^='sp_message_']"],
    scriptPatterns: [/sourcepoint/i, /sp-prod\.net/i, /spclient/i],
    iframePatterns: [/sp-prod\.net/i, /sourcepoint/i],
    globals: ["_sp_", "__tcfapi"],
  },
  {
    vendor: "OneTrust",
    selectors: ["#onetrust-banner-sdk", "#onetrust-consent-sdk", ".ot-sdk-container", "#ot-sdk-btn"],
    scriptPatterns: [/onetrust/i, /cookielaw\.org/i, /cdn\.cookielaw/i],
    iframePatterns: [/onetrust/i, /cookielaw/i],
    globals: ["OneTrust", "Optanon"],
  },
  {
    vendor: "Cookiebot",
    selectors: ["#CybotCookiebotDialog", ".CybotCookiebotDialog"],
    scriptPatterns: [/cookiebot/i],
    iframePatterns: [/cookiebot/i],
    globals: ["Cookiebot"],
  },
  {
    vendor: "Didomi",
    selectors: ["#didomi-host", ".didomi-popup", "[class*='didomi']"],
    scriptPatterns: [/didomi/i],
    iframePatterns: [/didomi/i],
    globals: ["Didomi"],
  },
  {
    vendor: "TrustArc",
    selectors: ["#truste-consent-track", ".truste_box_overlay", "#consent-banner"],
    scriptPatterns: [/trustarc/i, /truste\.com/i],
    iframePatterns: [/trustarc/i, /truste/i],
    globals: ["truste", "TrustArc"],
  },
  {
    vendor: "Usercentrics",
    selectors: ["#usercentrics-root", "[data-testid='uc-container']", ".uc-banner"],
    scriptPatterns: [/usercentrics/i],
    iframePatterns: [/usercentrics/i],
    globals: ["UC_UI", "usercentrics"],
  },
  {
    vendor: "Quantcast Choice",
    selectors: [".qc-cmp2-container", "#qc-cmp2-container", ".qc-cmp-ui"],
    scriptPatterns: [/quantcast/i, /qc-cmp/i],
    iframePatterns: [/quantcast/i],
    globals: ["__cmp", "__tcfapi"],
  },
  {
    vendor: "Civic Cookie Control",
    selectors: ["#ccc", ".ccc-module", "#ccc-module"],
    scriptPatterns: [/cookiecontrol/i, /civiccomputing/i],
    iframePatterns: [/civiccomputing/i],
    globals: ["CookieControl"],
  },
  {
    vendor: "Osano",
    selectors: [".osano-cm-window", "#osano-cm-dom-info"],
    scriptPatterns: [/osano/i],
    iframePatterns: [/osano/i],
    globals: ["Osano"],
  },
  {
    vendor: "CookieYes",
    selectors: [".cky-consent-container", "#cky-consent", ".cky-modal"],
    scriptPatterns: [/cookieyes/i],
    iframePatterns: [/cookieyes/i],
    globals: ["cookieyes"],
  },
  {
    vendor: "CookieScript",
    selectors: ["#cookiescript_injected", ".cookiescript_badge"],
    scriptPatterns: [/cookie-script/i, /cookiescript/i],
    iframePatterns: [/cookie-script/i],
    globals: ["CookieScript"],
  },
  {
    vendor: "CookieHub",
    selectors: [".ch2", "#ch2", ".cookiehub"],
    scriptPatterns: [/cookiehub/i],
    iframePatterns: [/cookiehub/i],
    globals: ["cookiehub"],
  },
];

function styleOf(el: Element): CSSStyleDeclaration {
  return window.getComputedStyle(el);
}

function cssEscape(value: string): string {
  if (typeof CSS !== "undefined" && typeof CSS.escape === "function") {
    return CSS.escape(value);
  }
  return value.replace(/([ !"#$%&'()*+,./:;<=>?@[\\\]^`{|}~])/g, "\\$1");
}

function buildXPath(el: Element): string {
  if (el.id) return `//*[@id="${el.id}"]`;
  const parts: string[] = [];
  let node: Element | null = el;
  while (node && node.nodeType === Node.ELEMENT_NODE) {
    let index = 1;
    let sibling = node.previousElementSibling;
    while (sibling) {
      if (sibling.nodeName === node.nodeName) index += 1;
      sibling = sibling.previousElementSibling;
    }
    parts.unshift(`${node.nodeName.toLowerCase()}[${index}]`);
    node = node.parentElement;
    if (node?.tagName.toLowerCase() === "html") {
      parts.unshift("html[1]");
      break;
    }
  }
  return "/" + parts.join("/");
}

function buildCssSelector(el: Element): string {
  if (el.id) return `#${cssEscape(el.id)}`;
  const parts: string[] = [];
  let node: Element | null = el;
  while (node && node.nodeType === Node.ELEMENT_NODE && parts.length < 6) {
    let part = node.nodeName.toLowerCase();
    if (node.classList.length > 0) {
      part += Array.from(node.classList)
        .slice(0, 2)
        .map((c) => `.${cssEscape(c)}`)
        .join("");
    }
    const parent: Element | null = node.parentElement;
    if (parent) {
      const same = Array.from(parent.children).filter(
        (c: Element) => c.nodeName === node!.nodeName,
      );
      if (same.length > 1) {
        part += `:nth-of-type(${same.indexOf(node) + 1})`;
      }
    }
    parts.unshift(part);
    if (node.id) {
      parts[0] = `#${cssEscape(node.id)}`;
      break;
    }
    node = parent;
  }
  return parts.join(" > ");
}

function buildDomPath(el: Element): string {
  const parts: string[] = [];
  let node: Element | null = el;
  while (node && node.nodeType === Node.ELEMENT_NODE) {
    let label = node.nodeName.toLowerCase();
    if (node.id) label += `#${node.id}`;
    else if (node.classList.length) label += `.${Array.from(node.classList).slice(0, 2).join(".")}`;
    parts.unshift(label);
    node = node.parentElement;
  }
  return parts.join(" > ");
}

function rectSnapshot(el: HTMLElement) {
  const rect = el.getBoundingClientRect();
  return {
    width: Math.round(rect.width),
    height: Math.round(rect.height),
    x: Math.round(rect.x),
    y: Math.round(rect.y),
    top: Math.round(rect.top),
    left: Math.round(rect.left),
    right: Math.round(rect.right),
    bottom: Math.round(rect.bottom),
    boundingRect: {
      x: Math.round(rect.x),
      y: Math.round(rect.y),
      width: Math.round(rect.width),
      height: Math.round(rect.height),
      top: Math.round(rect.top),
      left: Math.round(rect.left),
      right: Math.round(rect.right),
      bottom: Math.round(rect.bottom),
    },
  };
}

function isVisible(el: HTMLElement): boolean {
  const style = styleOf(el);
  const rect = el.getBoundingClientRect();
  if (style.display === "none" || style.visibility === "hidden") return false;
  if (parseFloat(style.opacity || "1") < 0.05) return false;
  return rect.width > 0 && rect.height > 0;
}

function controlLabel(el: HTMLElement): string {
  return (
    el.innerText ||
    el.getAttribute("aria-label") ||
    el.getAttribute("value") ||
    el.getAttribute("title") ||
    ""
  )
    .trim()
    .slice(0, 160);
}

function consentRole(label: string): "accept" | "reject" | "settings" | "other" {
  if (ACCEPT_RE.test(label)) return "accept";
  if (REJECT_RE.test(label)) return "reject";
  if (SETTINGS_RE.test(label)) return "settings";
  return "other";
}

function buttonSnapshot(
  el: HTMLElement,
  extras: { inConsentContainer?: boolean; frameOrigin?: string } = {},
) {
  const style = styleOf(el);
  const label = controlLabel(el);
  const geom = rectSnapshot(el);
  return {
    text: label,
    ariaLabel: el.getAttribute("aria-label") || undefined,
    tagName: el.tagName.toLowerCase(),
    role: el.getAttribute("role") || undefined,
    ...geom,
    fontSizePx: parseFloat(style.fontSize) || 0,
    fontWeight: parseInt(style.fontWeight, 10) || 400,
    backgroundColor: style.backgroundColor,
    color: style.color,
    display: style.display,
    visibility: style.visibility,
    opacity: parseFloat(style.opacity) || 1,
    textDecoration: style.textDecorationLine,
    xpath: buildXPath(el),
    cssSelector: buildCssSelector(el),
    domPath: buildDomPath(el),
    htmlSnippet: el.outerHTML.slice(0, 500),
    consentRole: consentRole(label),
    inConsentContainer: Boolean(extras.inConsentContainer),
    frameOrigin: extras.frameOrigin,
    computedStyles: {
      fontSizePx: parseFloat(style.fontSize) || 0,
      fontWeight: parseInt(style.fontWeight, 10) || 400,
      backgroundColor: style.backgroundColor,
      color: style.color,
      display: style.display,
      visibility: style.visibility,
      opacity: parseFloat(style.opacity) || 1,
      textDecoration: style.textDecorationLine,
      position: style.position,
      zIndex: style.zIndex,
    },
  };
}

function detectCmp(): Record<string, unknown> {
  const html = document.documentElement.innerHTML.slice(0, 400_000);
  const scripts = Array.from(document.scripts)
    .map((s) => s.src || s.textContent?.slice(0, 200) || "")
    .join("\n");
  const iframes = Array.from(document.querySelectorAll("iframe"));
  const win = window as unknown as Record<string, unknown>;

  let best: { vendor: string; method: string; confidence: number } | null = null;

  for (const sig of CMP_SIGNATURES) {
    let method = "";
    let confidence = 0;

    for (const sel of sig.selectors || []) {
      try {
        if (document.querySelector(sel)) {
          method = `dom:${sel}`;
          confidence = Math.max(confidence, 0.9);
          break;
        }
      } catch {
        /* invalid selector */
      }
    }

    for (const re of sig.scriptPatterns || []) {
      if (re.test(scripts) || re.test(html)) {
        method = method || `script:${re.source}`;
        confidence = Math.max(confidence, 0.75);
      }
    }

    for (const frame of iframes) {
      const blob = `${frame.src} ${frame.name} ${frame.title}`;
      for (const re of sig.iframePatterns || []) {
        if (re.test(blob)) {
          method = method || `iframe:${re.source}`;
          confidence = Math.max(confidence, 0.85);
        }
      }
    }

    for (const g of sig.globals || []) {
      if (win[g] != null) {
        method = method || `global:${g}`;
        confidence = Math.max(confidence, 0.8);
      }
    }

    if (confidence > 0 && (!best || confidence > best.confidence)) {
      best = { vendor: sig.vendor, method, confidence };
    }
  }

  if (!best) {
    return { detected: false, vendor: null, confidence: 0, detection_method: null };
  }
  return {
    detected: true,
    vendor: best.vendor,
    version: null,
    detection_method: best.method,
    confidence: best.confidence,
  };
}

function scoreContainer(el: HTMLElement): number {
  let score = 0;
  const style = styleOf(el);
  const rect = el.getBoundingClientRect();
  const vw = window.innerWidth || 1;
  const vh = window.innerHeight || 1;
  const coverage = (rect.width * rect.height) / (vw * vh);
  const idClass = `${el.id} ${el.className}`.toLowerCase();
  const aria = (el.getAttribute("aria-label") || "").toLowerCase();
  const role = (el.getAttribute("role") || "").toLowerCase();

  if (role === "dialog" || role === "alertdialog") score += 40;
  if (el.getAttribute("aria-modal") === "true") score += 25;
  if (CONSENT_TOKEN_RE.test(idClass) || CONSENT_TOKEN_RE.test(aria)) score += 35;
  if (/sp_message|onetrust|cookiebot|didomi|truste|qc-cmp|cky-|osano|usercentrics|cookiehub/.test(idClass)) {
    score += 45;
  }
  if (style.position === "fixed" || style.position === "sticky") score += 15;
  const z = parseInt(style.zIndex || "0", 10);
  if (!Number.isNaN(z) && z >= 1000) score += 15;
  if (coverage >= 0.2) score += 20;
  if (coverage >= 0.45) score += 15;
  if (rect.bottom >= vh - 8 && rect.height < vh * 0.55) score += 10; // bottom banner
  if (rect.top <= 8 && rect.height < vh * 0.4) score += 8; // top banner
  if (!isVisible(el)) score -= 50;

  const text = (el.innerText || "").slice(0, 800);
  if (ACCEPT_RE.test(text) || REJECT_RE.test(text) || SETTINGS_RE.test(text)) score += 25;
  if (CONSENT_TOKEN_RE.test(text)) score += 10;
  return score;
}

function findConsentContainers(root: ParentNode = document): HTMLElement[] {
  const candidates = new Set<HTMLElement>();
  const selectors = [
    "[role='dialog']",
    "[role='alertdialog']",
    "[aria-modal='true']",
    "[id*='cookie' i]",
    "[class*='cookie' i]",
    "[id*='consent' i]",
    "[class*='consent' i]",
    "[aria-label*='cookie' i]",
    "[aria-label*='consent' i]",
    "[data-testid*='consent' i]",
    "[data-testid*='cookie' i]",
  ];
  for (const sig of CMP_SIGNATURES) {
    for (const sel of sig.selectors || []) selectors.push(sel);
  }
  for (const sel of selectors) {
    try {
      root.querySelectorAll(sel).forEach((n) => candidates.add(n as HTMLElement));
    } catch {
      /* ignore */
    }
  }

  // Fixed / high-z overlays that look like consent UI
  const all = Array.from(root.querySelectorAll("div, section, aside, form")) as HTMLElement[];
  for (const el of all.slice(0, 400)) {
    const style = styleOf(el);
    const z = parseInt(style.zIndex || "0", 10);
    if (
      (style.position === "fixed" || style.position === "sticky") &&
      !Number.isNaN(z) &&
      z >= 999 &&
      isVisible(el)
    ) {
      const text = (el.innerText || "").slice(0, 500);
      if (CONSENT_TOKEN_RE.test(text) || ACCEPT_RE.test(text) || REJECT_RE.test(text)) {
        candidates.add(el);
      }
    }
  }

  return Array.from(candidates)
    .map((el) => ({ el, score: scoreContainer(el) }))
    .filter((x) => x.score >= 40)
    .sort((a, b) => b.score - a.score)
    .map((x) => x.el);
}

function bannerFromElement(el: HTMLElement): Record<string, unknown> {
  const style = styleOf(el);
  const geom = rectSnapshot(el);
  const vw = window.innerWidth || 1;
  const vh = window.innerHeight || 1;
  const coverage = (geom.width * geom.height) / (vw * vh);
  let placement = "overlay";
  if (coverage >= 0.7) placement = "fullscreen";
  else if (geom.bottom >= vh - 12 && geom.height < vh * 0.55) placement = "bottom_banner";
  else if (geom.top <= 12 && geom.height < vh * 0.4) placement = "top_banner";
  else if (style.position === "fixed" && geom.width < vw * 0.5) placement = "floating_popup";
  else if (roleIsSidebar(geom, vw, vh)) placement = "sidebar";

  return {
    ...geom,
    xpath: buildXPath(el),
    cssSelector: buildCssSelector(el),
    domPath: buildDomPath(el),
    htmlSnippet: el.outerHTML.slice(0, 800),
    placement,
    coverage: Math.round(coverage * 1000) / 1000,
    visible: isVisible(el),
    computedStyles: {
      display: style.display,
      visibility: style.visibility,
      opacity: parseFloat(style.opacity) || 1,
      position: style.position,
      zIndex: style.zIndex,
    },
  };
}

function roleIsSidebar(geom: { width: number; left: number; height: number }, vw: number, vh: number): boolean {
  return geom.width < vw * 0.45 && geom.height > vh * 0.5 && (geom.left < 24 || geom.left + geom.width > vw - 24);
}

function collectControlsInRoot(
  root: ParentNode,
  opts: { inConsentContainer: boolean; frameOrigin?: string; limit: number },
): ReturnType<typeof buttonSnapshot>[] {
  const selector =
    "button, [role='button'], input[type='submit'], input[type='button'], a.button, a.btn, .btn, a[href='#']";
  const nodes = Array.from(root.querySelectorAll(selector)) as HTMLElement[];
  // Also action-like links inside consent containers
  const extraLinks = opts.inConsentContainer
    ? (Array.from(root.querySelectorAll("a")).filter((a) => {
        const t = controlLabel(a as HTMLElement);
        return ACCEPT_RE.test(t) || REJECT_RE.test(t) || SETTINGS_RE.test(t);
      }) as HTMLElement[])
    : [];

  const seen = new Set<HTMLElement>();
  const out: ReturnType<typeof buttonSnapshot>[] = [];
  for (const el of [...nodes, ...extraLinks]) {
    if (seen.has(el)) continue;
    seen.add(el);
    if (!isVisible(el) && !opts.inConsentContainer) continue;
    const rect = el.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) continue;
    out.push(buttonSnapshot(el, { inConsentContainer: opts.inConsentContainer, frameOrigin: opts.frameOrigin }));
    if (out.length >= opts.limit) break;
  }
  return out;
}

function collectIframes(): Array<Record<string, unknown>> {
  return Array.from(document.querySelectorAll("iframe")).map((frame) => {
    const src = frame.src || "";
    const name = frame.name || "";
    const title = frame.title || "";
    let crossOrigin = false;
    let accessible = false;
    let sameOriginButtons: ReturnType<typeof buttonSnapshot>[] = [];
    let sameOriginBanner: Record<string, unknown> | undefined;
    let cmpVendor: string | null = null;

    for (const sig of CMP_SIGNATURES) {
      for (const re of sig.iframePatterns || []) {
        if (re.test(`${src} ${name} ${title}`)) {
          cmpVendor = sig.vendor;
          break;
        }
      }
      if (cmpVendor) break;
    }

    try {
      const doc = frame.contentDocument;
      if (doc) {
        accessible = true;
        const containers = findConsentContainers(doc);
        if (containers[0]) sameOriginBanner = bannerFromElement(containers[0]);
        sameOriginButtons = collectControlsInRoot(doc, {
          inConsentContainer: true,
          frameOrigin: src || location.origin,
          limit: 30,
        });
      } else {
        crossOrigin = true;
      }
    } catch {
      crossOrigin = true;
    }

    return {
      src: src.slice(0, 500),
      name,
      title,
      sandbox: frame.getAttribute("sandbox"),
      cross_origin: crossOrigin,
      accessible,
      likely_cmp: Boolean(cmpVendor) || CONSENT_TOKEN_RE.test(`${src} ${name} ${title}`),
      cmp_vendor: cmpVendor,
      button_count: sameOriginButtons.length,
      buttons: sameOriginButtons,
      banner: sameOriginBanner,
    };
  });
}

function buildConsentState(
  banner: Record<string, unknown> | undefined,
  containers: HTMLElement[],
): Record<string, unknown> {
  const visible = Boolean(banner && banner.visible !== false && ((banner.width as number) || 0) > 0);
  const placement = String(banner?.placement || "none");
  const cookieTextPresent = CONSENT_TOKEN_RE.test(document.body?.innerText?.slice(0, 5000) || "");

  return {
    banner_visible: visible,
    banner_dismissed: !visible && cookieTextPresent === false && containers.length === 0,
    banner_hidden: !visible,
    modal: placement === "overlay" || placement === "fullscreen",
    overlay: placement === "overlay" || placement === "fullscreen",
    sidebar: placement === "sidebar",
    floating_popup: placement === "floating_popup",
    bottom_banner: placement === "bottom_banner",
    top_banner: placement === "top_banner",
    fullscreen_dialog: placement === "fullscreen",
    placement: visible ? placement : "none",
  };
}

function collectCssSnapshot(): CssSnapshot {
  const cmp = detectCmp();
  const containers = findConsentContainers(document);
  const bannerEl = containers[0];
  let banner = bannerEl ? bannerFromElement(bannerEl) : undefined;

  const consentButtons = bannerEl
    ? collectControlsInRoot(bannerEl, { inConsentContainer: true, limit: 40 })
    : [];

  // Also pull consent-role controls from full document (not first-N only)
  const pageConsent = collectControlsInRoot(document, { inConsentContainer: false, limit: 80 }).filter(
    (b) => b.consentRole !== "other" || CONSENT_TOKEN_RE.test(b.text || ""),
  );

  const pageOther = collectControlsInRoot(document, { inConsentContainer: false, limit: 30 }).filter(
    (b) => b.consentRole === "other",
  );

  const iframes = collectIframes();
  const iframeButtons: ReturnType<typeof buttonSnapshot>[] = [];
  for (const frame of iframes) {
    if (frame.banner && !banner) banner = frame.banner as Record<string, unknown>;
    const btns = (frame.buttons as ReturnType<typeof buttonSnapshot>[]) || [];
    for (const b of btns) {
      iframeButtons.push({ ...b, inConsentContainer: true });
    }
  }

  const dedupeKey = (b: ReturnType<typeof buttonSnapshot>) =>
    `${b.text}|${b.xpath}|${b.width}x${b.height}`;
  const seen = new Set<string>();
  const buttons: ReturnType<typeof buttonSnapshot>[] = [];
  for (const b of [...consentButtons, ...iframeButtons, ...pageConsent, ...pageOther]) {
    const k = dedupeKey(b);
    if (seen.has(k)) continue;
    seen.add(k);
    buttons.push(b);
    if (buttons.length >= 60) break;
  }

  const checkboxes = Array.from(document.querySelectorAll("input[type='checkbox']")).map((el) => {
    const input = el as HTMLInputElement;
    const label =
      input.labels?.[0]?.innerText || input.getAttribute("aria-label") || input.name || "";
    return {
      label: label.trim().slice(0, 120),
      checked: input.checked,
      required: input.required,
      name: input.name,
      xpath: buildXPath(input),
      cssSelector: buildCssSelector(input),
      domPath: buildDomPath(input),
    };
  });

  const toggles = Array.from(
    document.querySelectorAll("[role='switch'], input[type='checkbox'][role='switch']"),
  ).map((el) => {
    const anyEl = el as HTMLElement & { checked?: boolean };
    return {
      label: (anyEl.getAttribute("aria-label") || anyEl.innerText || "").trim().slice(0, 120),
      checked: anyEl.getAttribute("aria-checked") === "true" || Boolean(anyEl.checked),
      ariaChecked: anyEl.getAttribute("aria-checked") === "true",
      xpath: buildXPath(anyEl),
      cssSelector: buildCssSelector(anyEl),
      domPath: buildDomPath(anyEl),
    };
  });

  const consent_state = buildConsentState(banner, containers);
  const bodyStyle = styleOf(document.body);

  return {
    buttons,
    checkboxes,
    toggles,
    banner,
    cmp,
    consent_state,
    iframes: iframes.map(({ buttons: _b, ...meta }) => meta),
    body: { fontSizePx: parseFloat(bodyStyle.fontSize) || 16 },
    primaryCta: buttons.find((b) => b.consentRole === "accept") || buttons[0],
    layoutHints: [],
  };
}

function visibleText(limit = 12000): string {
  const text = document.body?.innerText || "";
  return text.replace(/\s+/g, " ").trim().slice(0, limit);
}

export function collectPageData(): Omit<ScanPayload, "screenshot_base64"> {
  return {
    url: location.href,
    title: document.title,
    html: document.documentElement.outerHTML.slice(0, 200_000),
    css_snapshot: collectCssSnapshot(),
    visible_text: visibleText(),
    viewport: { width: window.innerWidth, height: window.innerHeight },
    scroll_position: { x: Math.round(window.scrollX), y: Math.round(window.scrollY) },
    collected_at: new Date().toISOString(),
  };
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "PING") {
    sendResponse({ ok: true });
    return true;
  }
  if (message?.type === "COLLECT_PAGE") {
    try {
      sendResponse({ ok: true, data: collectPageData() });
    } catch (error) {
      sendResponse({ ok: false, error: String(error) });
    }
  }
  return true;
});
