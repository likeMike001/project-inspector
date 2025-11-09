"use client";

type NarrativePanelProps = {
  narrativeCopy: string;
};

export function NarrativePanel({ narrativeCopy }: NarrativePanelProps) {
  return (
    <div className="panel-animate glow-border rounded-3xl border border-white/10 bg-gradient-to-b from-black/60 to-slate-900/40 p-6">
      <h3 className="text-sm uppercase tracking-[0.4em] text-slate-400">
        Live takeaways
      </h3>
      <div
        className="card-animate mt-4 rounded-2xl border border-white/10 bg-black/30 p-4 text-sm text-slate-200"
        style={{ animationDelay: "0.05s" }}
      >
        <p>{narrativeCopy}</p>
      </div>
      <div
        className="card-animate mt-6 rounded-2xl border border-white/10 bg-black/30 p-5 text-sm text-slate-300"
        style={{ animationDelay: "0.15s" }}
      >
        <p className="text-xs uppercase tracking-[0.4em] text-emerald-300">
          Model status
        </p>
        <p className="mt-2 text-lg font-semibold text-white">
          Streaming synthetic news + CoinGecko spot feed
        </p>
        <p className="mt-1 text-slate-400">
          Last Claude batch - 32 articles - 58% positive tone
        </p>
      </div>
    </div>
  );
}
