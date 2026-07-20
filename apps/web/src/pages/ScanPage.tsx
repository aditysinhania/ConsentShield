import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { api } from "../api/client";

export function ScanPage() {
  const navigate = useNavigate();
  const [url, setUrl] = useState("https://example.com");
  const [visibleText, setVisibleText] = useState(
    "Accept all cookies. Manage preferences. This subscription auto-renews monthly.",
  );
  const [error, setError] = useState<string | null>(null);

  const mutation = useMutation({
    mutationFn: () =>
      api.createScan({
        url,
        title: "Manual dashboard scan",
        visible_text: visibleText,
        css_snapshot: {
          buttons: [
            {
              text: "Accept all",
              width: 160,
              height: 44,
              fontWeight: 700,
              backgroundColor: "rgb(11, 110, 79)",
              fontSizePx: 16,
            },
            {
              text: "Manage preferences",
              width: 140,
              height: 36,
              fontWeight: 400,
              backgroundColor: "transparent",
              fontSizePx: 14,
            },
          ],
        },
        viewport: { width: 1280, height: 720 },
      }),
    onSuccess: (scan) => navigate(`/scans/${scan.id}`),
    onError: (e: Error) => setError(e.message),
  });

  return (
    <div className="max-w-2xl space-y-6">
      <div>
        <h1 className="font-display text-4xl font-bold">Website Scan</h1>
        <p className="mt-2 text-ink/70">
          Submit a payload for analysis. Production scans should come from the Chrome extension
          with live DOM, CSS, and screenshot capture.
        </p>
      </div>

      <label className="block space-y-2">
        <span className="text-sm font-semibold">URL</span>
        <input
          className="w-full rounded-xl border border-ink/10 bg-white/80 px-4 py-3"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
        />
      </label>

      <label className="block space-y-2">
        <span className="text-sm font-semibold">Visible text sample</span>
        <textarea
          className="min-h-32 w-full rounded-xl border border-ink/10 bg-white/80 px-4 py-3"
          value={visibleText}
          onChange={(e) => setVisibleText(e.target.value)}
        />
      </label>

      {error && <p className="text-sm text-clay">{error}</p>}

      <button
        className="rounded-xl bg-moss px-5 py-3 text-sm font-semibold text-white disabled:opacity-60"
        disabled={mutation.isPending}
        onClick={() => {
          setError(null);
          mutation.mutate();
        }}
      >
        {mutation.isPending ? "Running pipeline…" : "Run multimodal scan"}
      </button>
    </div>
  );
}
