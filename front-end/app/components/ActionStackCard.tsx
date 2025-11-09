"use client";

import { Recommendation } from "../types/dashboard";

type ActionStackCardProps = {
  recommendations: Recommendation[];
  recStatus: "idle" | "loading" | "error";
  lastRunAt: string | null;
  formatPercent: (value: number | null | undefined) => string;
};

export function ActionStackCard({
  recommendations,
  recStatus,
  lastRunAt,
  formatPercent,
}: ActionStackCardProps) {
  return (
    <div
      className="card-animate rounded-2xl border border-white/10 bg-black/30 p-4"
      style={{ animationDelay: "0.05s" }}
    >
      <div className="flex items-center justify-between text-xs uppercase tracking-[0.4em] text-slate-400">
        <span>Action stack</span>
        {lastRunAt && (
          <span className="text-[10px] normal-case text-slate-500">
            {new Date(lastRunAt).toLocaleTimeString()}
          </span>
        )}
      </div>
      <div className="mt-3 space-y-3">
        {recStatus === "loading" && (
          <div className="space-y-2">
            {[0, 1, 2].map((idx) => (
              <div
                key={idx}
                className="h-10 animate-pulse rounded-xl bg-slate-800/40"
              />
            ))}
          </div>
        )}
        {recStatus !== "loading" && recommendations.length === 0 && (
          <p className="text-sm text-slate-400">
            No actions yet -- adjust the slider or run the signal to see
            recommendations.
          </p>
        )}
        {recStatus !== "loading" &&
          recommendations.map((rec, index) => (
            <div
              key={rec.action}
              className="flex items-center justify-between rounded-2xl border border-white/10 bg-slate-900/40 px-4 py-3"
            >
              <div>
                <p className="text-xs uppercase tracking-[0.5em] text-slate-500">
                  {index + 1 < 10 ? `0${index + 1}` : index + 1}
                </p>
                <p className="text-lg font-semibold text-white">{rec.action}</p>
                {rec.rationale && (
                  <p className="text-xs text-slate-400">{rec.rationale}</p>
                )}
              </div>
              <span className="text-base font-semibold text-emerald-300">
                {formatPercent(rec.probability)}
              </span>
            </div>
          ))}
        {recStatus === "error" && (
          <p className="text-sm text-rose-300">
            Unable to refresh recommendations. Check the model API.
          </p>
        )}
      </div>
    </div>
  );
}
