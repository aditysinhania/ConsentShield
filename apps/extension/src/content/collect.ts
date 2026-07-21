import type { CssSnapshot, ScanPayload } from "@consentshield/shared";

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
  if (el.id) {
    return `//*[@id="${el.id}"]`;
  }
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
  if (el.id) {
    return `#${cssEscape(el.id)}`;
  }
  const parts: string[] = [];
  let node: Element | null = el;
  while (node && node.nodeType === Node.ELEMENT_NODE && parts.length < 6) {
    let part = node.nodeName.toLowerCase();
    if (node.classList.length > 0) {
      const cls = Array.from(node.classList)
        .slice(0, 2)
        .map((c) => `.${cssEscape(c)}`)
        .join("");
      part += cls;
    }
    const parent: Element | null = node.parentElement;
    if (parent) {
      const same = Array.from(parent.children).filter(
        (c: Element) => c.nodeName === node!.nodeName,
      );
      if (same.length > 1) {
        const idx = same.indexOf(node) + 1;
        part += `:nth-of-type(${idx})`;
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

function buttonSnapshot(el: HTMLElement) {
  const rect = el.getBoundingClientRect();
  const style = styleOf(el);
  return {
    text: (el.innerText || el.getAttribute("aria-label") || "").trim().slice(0, 120),
    ariaLabel: el.getAttribute("aria-label") || undefined,
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

  const bannerEl =
    (document.querySelector("[id*='cookie' i], [class*='cookie' i], [aria-label*='cookie' i]") as HTMLElement) ||
    null;
  let banner: Record<string, unknown> | undefined;
  if (bannerEl) {
    const rect = bannerEl.getBoundingClientRect();
    banner = {
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
      xpath: buildXPath(bannerEl),
      cssSelector: buildCssSelector(bannerEl),
      domPath: buildDomPath(bannerEl),
      htmlSnippet: bannerEl.outerHTML.slice(0, 800),
      computedStyles: {
        display: styleOf(bannerEl).display,
        visibility: styleOf(bannerEl).visibility,
        opacity: parseFloat(styleOf(bannerEl).opacity) || 1,
        position: styleOf(bannerEl).position,
        zIndex: styleOf(bannerEl).zIndex,
      },
    };
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
    scroll_position: { x: Math.round(window.scrollX), y: Math.round(window.scrollY) },
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
