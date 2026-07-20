import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api/client";

export function HistoryPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["scans"],
    queryFn: api.listScans,
  });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-4xl font-bold">Scan History</h1>
        <p className="mt-2 text-ink/70">Past website analyses and their status.</p>
      </div>

      {isLoading && <p>Loading…</p>}
      {error && <p className="text-clay">Failed to load scans. Is the API running?</p>}

      <div className="overflow-hidden rounded-2xl border border-ink/10 bg-white/70">
        <table className="w-full text-left text-sm">
          <thead className="bg-mist/60 text-ink/70">
            <tr>
              <th className="px-4 py-3 font-semibold">URL</th>
              <th className="px-4 py-3 font-semibold">Status</th>
              <th className="px-4 py-3 font-semibold">Created</th>
            </tr>
          </thead>
          <tbody>
            {(data || []).map((scan) => (
              <tr key={scan.id} className="border-t border-ink/5">
                <td className="px-4 py-3">
                  <Link className="font-semibold text-moss hover:underline" to={`/scans/${scan.id}`}>
                    {scan.title || scan.url}
                  </Link>
                  <div className="text-xs text-ink/50">{scan.url}</div>
                </td>
                <td className="px-4 py-3">{scan.status}</td>
                <td className="px-4 py-3">{new Date(scan.created_at).toLocaleString()}</td>
              </tr>
            ))}
            {data?.length === 0 && (
              <tr>
                <td className="px-4 py-8 text-ink/60" colSpan={3}>
                  No scans yet. Run one from the Scan page or Chrome extension.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
