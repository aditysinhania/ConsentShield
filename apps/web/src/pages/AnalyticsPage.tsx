import { useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api/client";

export function AnalyticsPage() {
  const { data: scans } = useQuery({ queryKey: ["scans"], queryFn: api.listScans });
  const { data: registry } = useQuery({ queryKey: ["models"], queryFn: api.modelsRegistry });

  const byStatus = Object.entries(
    (scans || []).reduce<Record<string, number>>((acc, s) => {
      acc[s.status] = (acc[s.status] || 0) + 1;
      return acc;
    }, {}),
  ).map(([name, value]) => ({ name, value }));

  return (
    <div className="space-y-8">
      <div>
        <h1 className="font-display text-4xl font-bold">Analytics</h1>
        <p className="mt-2 text-ink/70">Aggregate scan activity and model readiness.</p>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <Stat label="Total scans" value={String(scans?.length ?? 0)} />
        <Stat label="Stub mode" value={registry?.stub_mode ? "On" : "Off"} />
        <Stat label="Modules" value={String(registry?.modules?.length ?? 0)} />
      </div>

      <div className="h-72 rounded-2xl border border-ink/10 bg-white/70 p-4">
        <h2 className="mb-4 font-display text-xl font-semibold">Scans by status</h2>
        <ResponsiveContainer width="100%" height="85%">
          <BarChart data={byStatus.length ? byStatus : [{ name: "none", value: 0 }]}>
            <CartesianGrid strokeDasharray="3 3" stroke="#d9e5de" />
            <XAxis dataKey="name" />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="value" fill="#0b6e4f" radius={[8, 8, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-ink/10 bg-white/70 p-5">
      <div className="text-sm text-ink/60">{label}</div>
      <div className="mt-1 font-display text-3xl font-bold">{value}</div>
    </div>
  );
}
