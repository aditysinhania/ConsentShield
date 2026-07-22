/** User-facing error messages — never crash the UI. */

export type ErrorKind =
  | "backend"
  | "ai"
  | "screenshot"
  | "permission"
  | "iframe"
  | "timeout"
  | "network"
  | "unknown";

export interface FriendlyError {
  kind: ErrorKind;
  title: string;
  message: string;
}

export function classifyError(raw: unknown): FriendlyError {
  const text = String(raw ?? "Unknown error");
  const lower = text.toLowerCase();

  if (lower.includes("failed to fetch") || lower.includes("networkerror") || lower.includes("net::")) {
    return {
      kind: "network",
      title: "Network error",
      message: "Could not reach the ConsentShield API. Check that the backend is running and the API URL is correct.",
    };
  }
  if (lower.includes("401") || lower.includes("403") || lower.includes("permission")) {
    return {
      kind: "permission",
      title: "Permission denied",
      message: "The API rejected this request. Check your API token in Settings, or try without authentication in development.",
    };
  }
  if (lower.includes("screenshot") || lower.includes("capturevisibletab")) {
    return {
      kind: "screenshot",
      title: "Screenshot failed",
      message: "Chrome could not capture this tab. Try refreshing the page or granting the activeTab permission again.",
    };
  }
  if (lower.includes("timeout") || lower.includes("timed out")) {
    return {
      kind: "timeout",
      title: "Scan timed out",
      message: "The scan took too long. The page may be heavy, or the AI models are still loading on first use.",
    };
  }
  if (lower.includes("iframe") || lower.includes("cross-origin")) {
    return {
      kind: "iframe",
      title: "Cross-origin consent interface",
      message:
        "This consent interface is embedded inside a protected cross-origin iframe that Chrome extensions cannot fully inspect.",
    };
  }
  if (lower.includes("502") || lower.includes("503") || lower.includes("econnrefused") || lower.includes("scan failed (5")) {
    return {
      kind: "backend",
      title: "Backend unavailable",
      message: "The ConsentShield API is not responding. Start the API (scripts/dev-api) and try again.",
    };
  }
  if (lower.includes("ai") || lower.includes("model") || lower.includes("stub")) {
    return {
      kind: "ai",
      title: "AI unavailable",
      message: "AI models are offline or in stub mode. Rule Engine results still apply. Enable AI in Settings / .env when ready.",
    };
  }
  if (lower.includes("receiving end does not exist") || lower.includes("could not establish connection")) {
    return {
      kind: "permission",
      title: "Page not ready",
      message: "Refresh the tab and try again. Content scripts may not be injected on this page yet.",
    };
  }

  return {
    kind: "unknown",
    title: "Scan failed",
    message: text.replace(/^Error:\s*/i, "").slice(0, 280),
  };
}
