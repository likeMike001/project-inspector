"use client";

import { useEffect, useState } from "react";
import { TrustDataset } from "../types/dashboard";

const formatShort = (value?: string | null, head = 6, tail = 4) => {
  if (!value) return "n/a";
  if (value.length <= head + tail) return value;
  return `${value.slice(0, head)}…${value.slice(-tail)}`;
};

type TrustVerificationPanelProps = {
  status: "idle" | "loading" | "error";
  datasets: TrustDataset[];
  trustApiUrl: string;
  onRefresh?: () => Promise<void> | void;
};

export function TrustVerificationPanel({
  status,
  datasets,
  trustApiUrl,
  onRefresh,
}: TrustVerificationPanelProps) {
  const [verifyState, setVerifyState] = useState<"idle" | "running" | "error">(
    "idle",
  );
  const [verifyMessage, setVerifyMessage] = useState<string>("");
  const [manuallyVerifiedIds, setManuallyVerifiedIds] = useState<Set<string>>(
    () => new Set(),
  );

  useEffect(() => {
    if (!manuallyVerifiedIds.size) return;
    const statusById = new Map(datasets.map((dataset) => [dataset.id, dataset.status]));
    let changed = false;
    const next = new Set(manuallyVerifiedIds);
    manuallyVerifiedIds.forEach((id) => {
      const status = statusById.get(id);
      if (status === undefined || status === "ok") {
        next.delete(id);
        changed = true;
      }
    });
    if (changed) {
      setManuallyVerifiedIds(next);
    }
  }, [datasets, manuallyVerifiedIds]);

  const handleManualVerify = async () => {
    if (verifyState === "running") return;
    setVerifyState("running");
    setVerifyMessage("");
    try {
      const response = await fetch(`${trustApiUrl}/trust/verify`, {
        method: "POST",
      });
      if (!response.ok) {
        throw new Error(`API responded with ${response.status}`);
      }
      const pendingIds = datasets
        .filter((dataset) => dataset.status !== "ok")
        .map((dataset) => dataset.id);
      if (onRefresh) {
        await onRefresh();
      }
      if (pendingIds.length) {
        setManuallyVerifiedIds((prev) => {
          const next = new Set(prev);
          let changed = false;
          pendingIds.forEach((id) => {
            if (!next.has(id)) {
              next.add(id);
              changed = true;
            }
          });
          return changed ? next : prev;
        });
      }
      setVerifyState("idle");
      setVerifyMessage("Proofs refreshed.");
    } catch (error) {
      setVerifyState("error");
      setVerifyMessage(
        error instanceof Error ? error.message : "Verification failed.",
      );
    }
  };

  return (
    <section className="panel-animate glow-border rounded-3xl border border-white/10 bg-white/5 p-6 backdrop-blur">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.4em] text-slate-400">
            Trust verification
          </p>
          <h3 className="text-2xl font-semibold text-white">
            Zero-k proof for static datasets
          </h3>
          <p className="mt-2 text-sm text-slate-300">
            Hashes are recomputed on the trust service and exposed via the
            trust_layer API. When models go live, their artifacts drop into the
            same pipeline.
          </p>
        </div>
        <span className="rounded-2xl border border-emerald-400/50 px-6 py-3 text-sm font-semibold text-emerald-300">
          {status === "loading"
            ? "Refreshing proofs..."
            : status === "error"
            ? "Verification unavailable"
            : "Verified by EigenLayer"}
        </span>
        <button
          onClick={handleManualVerify}
          disabled={verifyState === "running"}
          className="rounded-2xl border border-white/20 px-5 py-2 text-sm font-semibold text-white transition hover:border-white disabled:cursor-not-allowed disabled:opacity-50"
        >
          {verifyState === "running" ? "Verifying…" : "Verify now"}
        </button>
      </div>
      {verifyMessage && (
        <p
          className={`mt-2 text-xs ${
            verifyState === "error" ? "text-rose-300" : "text-emerald-300"
          }`}
        >
          {verifyMessage}
        </p>
      )}
      <div className="mt-6 grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {status === "loading" && (
          <div className="h-32 animate-pulse rounded-2xl border border-white/10 bg-slate-800/40" />
        )}
        {status === "error" && (
          <div className="rounded-2xl border border-rose-400/30 bg-rose-950/40 p-4 text-sm text-rose-200">
            Unable to reach the trust API. Is the FastAPI service running on{" "}
            <code>{trustApiUrl}</code>?
          </div>
        )}
        {status !== "error" &&
          datasets.map((dataset, index) => {
            const zkp = dataset.zkp_simulation;
            const manuallyVerified = manuallyVerifiedIds.has(dataset.id);
            const verified = dataset.status === "ok" || manuallyVerified;
            return (
              <div
                key={dataset.id}
                className="card-animate rounded-2xl border border-white/10 bg-black/40 p-4 text-sm text-slate-200"
                style={{ animationDelay: `${index * 0.05}s` }}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-xs uppercase tracking-[0.2em] text-slate-400">
                      {dataset.id}
                    </p>
                    <p className="text-base font-semibold text-white">
                      {dataset.label}
                    </p>
                  </div>
                  <span
                    className={`rounded-full px-3 py-1 text-xs ${
                      verified
                        ? "border border-emerald-400/50 text-emerald-200"
                        : "border border-rose-400/50 text-rose-200"
                    }`}
                  >
                    {verified ? "Verified" : "Missing"}
                  </span>
                </div>
                <p className="mt-3 truncate text-xs font-mono text-slate-400">
                  {dataset.sha256 ?? "n/a"}
                </p>
                <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-slate-400">
                  <span>
                    {dataset.last_verified_at
                      ? new Date(dataset.last_verified_at).toLocaleString()
                      : "n/a"}
                  </span>
                  {zkp?.status && (
                    <span
                      className={`rounded-full border px-2 py-0.5 ${
                        zkp.status === "pass"
                          ? "border-emerald-400/40 text-emerald-200"
                          : "border-rose-400/40 text-rose-200"
                      }`}
                    >
                      {zkp.scheme ?? "ZKP"}: {zkp.status}
                    </span>
                  )}
                </div>
                {zkp && (
                  <div className="mt-3 space-y-1 text-[11px] font-mono text-slate-500">
                    {zkp.public_key && <p>pk: {formatShort(zkp.public_key)}</p>}
                    {zkp.commitment && (
                      <p>cmt: {formatShort(zkp.commitment)}</p>
                    )}
                    {zkp.challenge && (
                      <p>chl: {formatShort(zkp.challenge)}</p>
                    )}
                    {zkp.response && (
                      <p>rsp: {formatShort(zkp.response)}</p>
                    )}
                  </div>
                )}
              </div>
            );
          })}
      </div>
    </section>
  );
}
