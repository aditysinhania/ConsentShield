import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";

export function SettingsPage() {
  const [token, setToken] = useState("");
  const { data: catalog } = useQuery({ queryKey: ["rules"], queryFn: api.rulesCatalog });
  const { data: registry } = useQuery({ queryKey: ["models"], queryFn: api.modelsRegistry });

  useEffect(() => {
    setToken(localStorage.getItem("cs_token") || "");
  }, []);

  return (
    <div className="max-w-2xl space-y-8">
      <div>
        <h1 className="font-display text-4xl font-bold">Settings</h1>
        <p className="mt-2 text-ink/70">API auth token and module status.</p>
      </div>

      <label className="block space-y-2">
        <span className="text-sm font-semibold">JWT access token</span>
        <input
          className="w-full rounded-xl border border-ink/10 bg-white/80 px-4 py-3"
          value={token}
          onChange={(e) => setToken(e.target.value)}
          placeholder="Optional — scans work anonymously in development"
        />
      </label>
      <button
        className="rounded-xl bg-moss px-5 py-3 text-sm font-semibold text-white"
        onClick={() => localStorage.setItem("cs_token", token)}
      >
        Save token
      </button>

      <section className="rounded-2xl border border-ink/10 bg-white/70 p-5">
        <h2 className="font-display text-xl font-semibold">Rule catalog</h2>
        <p className="mt-1 text-sm text-ink/60">{catalog?.count ?? 0} configurable rules loaded</p>
      </section>

      <section className="rounded-2xl border border-ink/10 bg-white/70 p-5">
        <h2 className="font-display text-xl font-semibold">Model registry</h2>
        <pre className="mt-3 overflow-auto rounded-xl bg-ink/[0.04] p-3 text-xs">
          {JSON.stringify(registry ?? {}, null, 2)}
        </pre>
      </section>
    </div>
  );
}
