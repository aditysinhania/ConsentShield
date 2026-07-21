import { useQuery } from "@tanstack/react-query";
import { useParams } from "react-router-dom";
import { api } from "../api/client";

export function ExplanationPage() {
  const { id = "" } = useParams();
  const { data, isLoading, error } = useQuery({
    queryKey: ["report", id],
    queryFn: () => api.getReport(id),
    enabled: Boolean(id),
  });

  if (isLoading) return <p>Loading explanation…</p>;
  if (error || !data) return <p className="text-clay">Report not found.</p>;

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-display text-4xl font-bold">AI Explanation</h1>
          <p className="mt-2 text-ink/70">Evidence-backed multimodal report</p>
        </div>
        <div className="rounded-2xl bg-moss px-5 py-3 text-white">
          <div className="text-xs uppercase tracking-wide opacity-80">Risk score</div>
          <div className="font-display text-3xl font-bold">{Math.round(data.risk_score)}</div>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <Panel title="Category" body={String(data.category)} />
        <Panel title="Confidence" body={`${(data.confidence * 100).toFixed(0)}%`} />
        <Panel title="Evidence items" body={String(data.evidence?.length ?? 0)} />
      </div>

      {data.confidence_breakdown && (
        <section className="rounded-2xl border border-ink/10 bg-white/70 p-6">
          <h2 className="font-display text-2xl font-semibold">Confidence breakdown</h2>
          <div className="mt-4 grid gap-3 sm:grid-cols-3 lg:grid-cols-6 text-sm">
            {(
              [
                ["text", data.confidence_breakdown.text],
                ["visual", data.confidence_breakdown.visual],
                ["layout", data.confidence_breakdown.layout],
                ["cmp", data.confidence_breakdown.cmp],
                ["agreement", data.confidence_breakdown.agreement],
                ["final", data.confidence_breakdown.final],
              ] as const
            ).map(([label, value]) => (
              <div key={label}>
                <div className="text-ink/60 capitalize">{label}</div>
                <div className="font-display text-lg font-semibold">
                  {(value * 100).toFixed(0)}%
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      <section className="rounded-2xl border border-ink/10 bg-white/70 p-6">
        <h2 className="font-display text-2xl font-semibold">Evidence</h2>
        <ul className="mt-4 space-y-3">
          {(data.evidence || []).map((item) => (
            <li key={item.id} className="flex gap-3 text-sm">
              <span className="mt-1 h-2 w-2 shrink-0 rounded-full bg-moss" />
              <div>
                <div className="font-semibold">{item.statement}</div>
                {item.explanation && <div className="mt-1 text-ink/80">{item.explanation}</div>}
                {item.user_impact && (
                  <div className="mt-1 text-ink/60">Impact: {item.user_impact}</div>
                )}
                {item.gdpr_relevance && (
                  <div className="mt-1 text-ink/60">GDPR: {item.gdpr_relevance}</div>
                )}
                {item.recommendation && (
                  <div className="mt-1 text-ink/60">Recommendation: {item.recommendation}</div>
                )}
                <div className="text-ink/50">
                  {item.source}
                  {item.rule_id ? ` · ${item.rule_id}` : ""} · severity {item.severity}
                  {item.css_selector ? ` · ${item.css_selector}` : ""}
                </div>
              </div>
            </li>
          ))}
          {(data.evidence || []).length === 0 && (
            <li className="text-ink/60">No dark-pattern evidence triggered for this scan.</li>
          )}
        </ul>
      </section>

      {(data.rule_traces || []).length > 0 && (
        <section className="rounded-2xl border border-ink/10 bg-white/70 p-6">
          <h2 className="font-display text-2xl font-semibold">Rule traces</h2>
          <ul className="mt-4 space-y-2 text-sm">
            {data.rule_traces!.map((t) => (
              <li key={t.rule_id} className="border-b border-ink/5 pb-2">
                <div className="font-semibold">{t.rule_id}</div>
                <div className="text-ink/60">
                  risk {t.risk_contribution} · visual {t.visual_score.toFixed(2)} · text{" "}
                  {t.text_score.toFixed(2)} · layout {t.layout_score.toFixed(2)}
                </div>
                <div className="text-ink/50">features: {t.features_used.join(", ")}</div>
              </li>
            ))}
          </ul>
        </section>
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        <JsonPanel title="Rule engine" data={data.rules} />
        <JsonPanel title="Vision" data={data.vision} />
        <JsonPanel title="Text / NLP" data={data.text} />
        <JsonPanel title="Fusion" data={data.fusion} />
      </div>

      {(data.pipeline_notes || []).length > 0 && (
        <section className="rounded-2xl border border-dashed border-ink/20 bg-white/40 p-5 text-sm text-ink/70">
          <h3 className="mb-2 font-semibold text-ink">Pipeline notes</h3>
          <ul className="list-disc space-y-1 pl-5">
            {data.pipeline_notes!.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}

function Panel({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-2xl border border-ink/10 bg-white/70 p-5">
      <div className="text-sm text-ink/60">{title}</div>
      <div className="mt-1 font-display text-xl font-semibold">{body}</div>
    </div>
  );
}

function JsonPanel({ title, data }: { title: string; data: unknown }) {
  return (
    <section className="rounded-2xl border border-ink/10 bg-white/70 p-5">
      <h3 className="font-display text-lg font-semibold">{title}</h3>
      <pre className="mt-3 max-h-64 overflow-auto rounded-xl bg-ink/[0.04] p-3 text-xs leading-relaxed">
        {JSON.stringify(data ?? { status: "unavailable" }, null, 2)}
      </pre>
    </section>
  );
}
