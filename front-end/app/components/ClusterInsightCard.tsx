"use client";

import { ClusterInsight } from "../types/dashboard";

type ClusterInsightCardProps = {
  clusterInsight: ClusterInsight | null;
  recStatus: "idle" | "loading" | "error";
  formatMetricValue: (value: number | null | undefined) => string;
};

export function ClusterInsightCard({
  clusterInsight,
  recStatus,
  formatMetricValue,
}: ClusterInsightCardProps) {
  return (
    <div
      className="card-animate rounded-2xl border border-white/10 bg-black/30 p-4"
      style={{ animationDelay: "0.08s" }}
    >
      <p className="text-xs uppercase tracking-[0.4em] text-slate-400">
        Cluster regime
      </p>
      {clusterInsight ? (
        <div className="mt-3 space-y-3">
          <div>
            <p className="text-2xl font-semibold text-white">
              Cluster {clusterInsight.id}
            </p>
            <p className="text-sm text-slate-300">
              {clusterInsight.label ?? "Unlabeled regime"}
            </p>
          </div>
          {clusterInsight.description && (
            <p className="text-sm text-slate-300">
              {clusterInsight.description}
            </p>
          )}
          {clusterInsight.drivers && clusterInsight.drivers.length > 0 && (
            <ul className="space-y-2 text-sm text-slate-300">
              {clusterInsight.drivers.map((driver) => (
                <li
                  key={driver}
                  className="flex items-start gap-2 text-xs text-slate-400"
                >
                  <span className="mt-1 h-1.5 w-1.5 rounded-full bg-emerald-400" />
                  {driver}
                </li>
              ))}
            </ul>
          )}
          {clusterInsight.metrics && (
            <div className="grid grid-cols-2 gap-3 text-xs text-slate-300">
              {Object.entries(clusterInsight.metrics)
                .slice(0, 4)
                .map(([key, value]) => (
                  <div
                    key={key}
                    className="rounded-xl border border-white/10 bg-slate-900/40 p-3"
                  >
                    <p className="uppercase tracking-[0.3em] text-slate-500">
                      {key.replace(/_/g, " ")}
                    </p>
                    <p className="text-base font-semibold text-white">
                      {formatMetricValue(value)}
                    </p>
                  </div>
                ))}
            </div>
          )}
        </div>
      ) : recStatus === "loading" ? (
        <div className="mt-4 h-32 animate-pulse rounded-2xl bg-slate-800/50" />
      ) : (
        <p className="mt-3 text-sm text-slate-400">
          Adjust the slider or run the signal to classify the current regime.
        </p>
      )}
    </div>
  );
}
