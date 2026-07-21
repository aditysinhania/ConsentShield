import { getProvider } from "../shared/inference";

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type === "RUN_SCAN") {
    void (async () => {
      try {
        const tabId = sender.tab?.id ?? message.tabId;
        if (!tabId) throw new Error("No active tab");

        const [{ result: page }] = await chrome.scripting.executeScript({
          target: { tabId },
          func: () => {
            // Fallback if content script not injected yet
            return null;
          },
        });

        const collectStart = performance.now();
        const collected = await chrome.tabs.sendMessage(tabId, { type: "COLLECT_PAGE" });
        if (!collected?.ok) throw new Error(collected?.error || "Collection failed");
        const collectionDurationMs = performance.now() - collectStart;

        const dataUrl = await chrome.tabs.captureVisibleTab({ format: "png" });
        const payload = {
          ...collected.data,
          screenshot_base64: dataUrl,
          collection_duration_ms: collectionDurationMs,
        };

        const provider = await getProvider();
        const report = await provider.analyze(payload);
        sendResponse({ ok: true, report, unused: page });
      } catch (error) {
        sendResponse({ ok: false, error: String(error) });
      }
    })();
    return true;
  }
  return false;
});
