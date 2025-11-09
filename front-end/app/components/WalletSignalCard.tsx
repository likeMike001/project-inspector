"use client";

type WalletSignalCardProps = {
  wallet: string;
  onWalletChange: (value: string) => void;
  onRunSignal: () => void;
  recStatus: "idle" | "loading" | "error";
  disabled: boolean;
  modelMessage: string | null;
  walletSummaryCopy: string | null;
  walletSummaryError: string | null;
};

export function WalletSignalCard({
  wallet,
  onWalletChange,
  onRunSignal,
  recStatus,
  disabled,
  modelMessage,
  walletSummaryCopy,
  walletSummaryError,
}: WalletSignalCardProps) {
  return (
    <div
      className="card-animate mt-6 rounded-2xl bg-black/30 p-4"
      style={{ animationDelay: "0.1s" }}
    >
      <label className="text-sm font-semibold text-slate-200">
        Optional wallet signal
        <span className="ml-2 rounded-full border border-emerald-400/30 px-2 py-0.5 text-[11px] font-normal text-emerald-200">
          runs wallet snapshot
        </span>
      </label>
      <div className="mt-2 flex flex-col gap-3 sm:flex-row">
        <input
          value={wallet}
          onChange={(event) => onWalletChange(event.target.value)}
          placeholder="0x... (executed against wallet_snapshot.py)"
          className="flex-1 rounded-xl border border-white/10 bg-black/40 px-4 py-3 text-sm text-white outline-none transition focus:border-emerald-400/70"
          disabled={disabled}
        />
        <button
          onClick={onRunSignal}
          disabled={recStatus === "loading" || disabled}
          className={`rounded-xl px-6 py-3 text-sm font-semibold shadow-2xl shadow-emerald-500/30 transition ${
            recStatus === "loading"
              ? "cursor-not-allowed bg-slate-600 text-slate-300"
              : "bg-gradient-to-r from-emerald-400 to-cyan-400 text-slate-900"
          }`}
          type="button"
        >
          {recStatus === "loading" ? "Updating..." : "Run signal"}
        </button>
      </div>
      <p className="mt-2 text-xs text-slate-400">
        Entering an address runs the wallet tracking script, pipes the JSON
        summary to Claude, and annotates model rationale.
      </p>
      {modelMessage && (
        <p className="mt-3 text-xs text-slate-400">
          {recStatus === "error" ? "Heads up: " : "Latest: "}
          {modelMessage}
        </p>
      )}
      {walletSummaryCopy && (
        <p className="mt-2 text-xs text-emerald-300">{walletSummaryCopy}</p>
      )}
      {walletSummaryError && (
        <p className="mt-2 text-xs text-rose-300">
          Wallet snapshot error: {walletSummaryError}
        </p>
      )}
    </div>
  );
}
