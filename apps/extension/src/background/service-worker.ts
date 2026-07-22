import { getProvider } from "../shared/inference";
import { loadSettings } from "../shared/settings";

type ProgressStage =
  | "scanning"
  | "extracting"
  | "banner"
  | "rules"
  | "nlp"
  | "vision"
  | "fusion"
  | "report"
  | "done"
  | "error";

function postProgress(port: chrome.runtime.Port | null, stage: ProgressStage, message: string, index: number) {
  try {
    port?.postMessage({ type: "SCAN_PROGRESS", stage, message, index });
  } catch {
    /* popup may have closed */
  }
}

chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== "scan") return;

  port.onMessage.addListener((message) => {
    if (message?.type !== "START_SCAN") return;
    void (async () => {
      try {
        const tabId = message.tabId as number;
        if (!tabId) throw new Error("No active tab");

        const settings = await loadSettings();
        postProgress(port, "scanning", "Scanning page…", 0);

        // Ensure content script is available
        try {
          await chrome.tabs.sendMessage(tabId, { type: "PING" });
        } catch {
          await chrome.scripting.executeScript({
            target: { tabId },
            files: ["content.js"],
          });
        }

        postProgress(port, "extracting", "Extracting DOM…", 1);
        const collectStart = performance.now();
        const collected = await chrome.tabs.sendMessage(tabId, { type: "COLLECT_PAGE" });
        if (!collected?.ok) throw new Error(collected?.error || "Collection failed");
        const collectionDurationMs = performance.now() - collectStart;

        postProgress(port, "banner", "Detecting consent banner…", 2);

        let screenshot_base64: string | undefined;
        if (settings.enableScreenshots) {
          try {
            screenshot_base64 = await chrome.tabs.captureVisibleTab({ format: "png" });
          } catch {
            /* Screenshot may fail on restricted pages — continue without it */
          }
        }

        const payload = {
          ...collected.data,
          screenshot_base64: settings.enableScreenshots ? screenshot_base64 : undefined,
          collection_duration_ms: collectionDurationMs,
        };

        postProgress(port, "rules", "Running Rule Engine…", 3);
        // Remaining stages are backend-side; surface optimistic progress while awaiting
        const progressTimer = windowSetIntervals(port);

        const provider = await getProvider();
        const report = (await provider.analyze(payload)) as Record<string, unknown>;
        clearIntervals(progressTimer);

        postProgress(port, "done", "Finished", 7);
        port.postMessage({
          ok: true,
          type: "SCAN_RESULT",
          report,
          screenshotDataUrl: screenshot_base64 || null,
          cssSnapshot: collected.data?.css_snapshot || null,
          collectionDurationMs,
          url: collected.data?.url,
          title: collected.data?.title,
        });
      } catch (error) {
        port.postMessage({ ok: false, type: "SCAN_RESULT", error: String(error) });
      }
    })();
  });
});

/** Fallback for popup one-shot messages (compatibility). */
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "RUN_SCAN") {
    const portLike = {
      postMessage: () => undefined,
    } as unknown as chrome.runtime.Port;
    void (async () => {
      try {
        const tabId = message.tabId as number;
        if (!tabId) throw new Error("No active tab");
        const settings = await loadSettings();
        try {
          await chrome.tabs.sendMessage(tabId, { type: "PING" });
        } catch {
          await chrome.scripting.executeScript({ target: { tabId }, files: ["content.js"] });
        }
        const collectStart = performance.now();
        const collected = await chrome.tabs.sendMessage(tabId, { type: "COLLECT_PAGE" });
        if (!collected?.ok) throw new Error(collected?.error || "Collection failed");
        let screenshot_base64: string | undefined;
        if (settings.enableScreenshots) {
          try {
            screenshot_base64 = await chrome.tabs.captureVisibleTab({ format: "png" });
          } catch {
            /* ignore */
          }
        }
        const payload = {
          ...collected.data,
          screenshot_base64,
          collection_duration_ms: performance.now() - collectStart,
        };
        const provider = await getProvider();
        const report = await provider.analyze(payload);
        sendResponse({
          ok: true,
          report,
          screenshotDataUrl: screenshot_base64 || null,
          cssSnapshot: collected.data?.css_snapshot || null,
          collectionDurationMs: payload.collection_duration_ms,
          url: collected.data?.url,
          title: collected.data?.title,
        });
        void portLike;
      } catch (error) {
        sendResponse({ ok: false, error: String(error) });
      }
    })();
    return true;
  }
  return false;
});

function windowSetIntervals(port: chrome.runtime.Port): number[] {
  const ids: number[] = [];
  ids.push(
    setTimeout(() => postProgress(port, "nlp", "Running NLP…", 4), 400) as unknown as number,
  );
  ids.push(
    setTimeout(() => postProgress(port, "vision", "Running Vision…", 5), 900) as unknown as number,
  );
  ids.push(
    setTimeout(() => postProgress(port, "fusion", "Combining results…", 6), 1400) as unknown as number,
  );
  ids.push(
    setTimeout(() => postProgress(port, "report", "Generating report…", 7), 2000) as unknown as number,
  );
  return ids;
}

function clearIntervals(ids: number[]) {
  ids.forEach((id) => clearTimeout(id));
}
