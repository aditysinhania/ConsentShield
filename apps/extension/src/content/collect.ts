import type { CssSnapshot, ScanPayload } from "@consentshield/shared";

function styleOf(el: Element): CSSStyleDeclaration {
  return window.getComputedStyle(el);
}

function buttonSnapshot(el: HTMLElement) {
  const rect = el.getBoundingClientRect();
  const style = styleOf(el);
  return {
    text: (el.innerText || el.getAttribute("aria-label") || "").trim().slice(0, 120),
    ariaLabel: el.getAttribute("aria-label") || undefined,
    width: Math.round(rect.width),
    height: Math.round(rect.height),
    fontSizePx: parseFloat(style.fontSize) || 0,
    fontWeight: parseInt(style.fontWeight, 10) || 400,
    backgroundColor: style.backgroundColor,
    color: style.color,
    display: style.display,
    visibility: style.visibility,
    opacity: parseFloat(style.opacity) || 1,
    textDecoration: style.textDecorationLine,
  };
}

function collectCssSnapshot(): CssSnapshot {
  const candidates = Array.from(
    document.querySelectorAll("button, [role='button'], input[type='submit'], a.button, .btn"),
  ) as HTMLElement[];

  const buttons = candidates
    .filter((el) => {
      const rect = el.getBoundingClientRect();
      return rect.width > 0 && rect.height > 0;
    })
    .slice(0, 40)
    .map(buttonSnapshot);

  const checkboxes = Array.from(
    document.querySelectorAll("input[type='checkbox']"),
  ).map((el) => {
    const input = el as HTMLInputElement;
    const label =
      input.labels?.[0]?.innerText ||
      input.getAttribute("aria-label") ||
      input.name ||
      "";
    return {
      label: label.trim().slice(0, 120),
      checked: input.checked,
      required: input.required,
      name: input.name,
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
    };
  });

  const bannerEl =
    (document.querySelector("[id*='cookie' i], [class*='cookie' i], [aria-label*='cookie' i]") as HTMLElement) ||
    null;
  let banner: Record<string, unknown> | undefined;
  if (bannerEl) {
    const rect = bannerEl.getBoundingClientRect();
    banner = { width: Math.round(rect.width), height: Math.round(rect.height) };
  }

  const bodyStyle = styleOf(document.body);
  return {
    buttons,
    checkboxes,
    toggles,
    banner,
    body: { fontSizePx: parseFloat(bodyStyle.fontSize) || 16 },
    primaryCta: buttons[0],
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
    collected_at: new Date().toISOString(),
  };
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "COLLECT_PAGE") {
    try {
      sendResponse({ ok: true, data: collectPageData() });
    } catch (error) {
      sendResponse({ ok: false, error: String(error) });
    }
  }
  return true;
});
