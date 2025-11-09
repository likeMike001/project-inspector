"use client";

type PipelinePreviewCardProps = {
  refCallback: (node: HTMLDivElement | null) => void;
};

export function PipelinePreviewCard({
  refCallback,
}: PipelinePreviewCardProps) {
  return (
    <section
      ref={refCallback}
      className="panel-animate glow-border rounded-3xl border border-white/10 bg-gradient-to-r from-slate-900/70 to-black/50 p-6 backdrop-blur"
    >
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.4em] text-slate-400">
            Pipeline preview
          </p>
          <h3 className="text-2xl font-semibold text-white">
            Synthetic news + sentiment + feature store + PKL predictions
          </h3>
          <p className="mt-2 text-sm text-slate-300">
            Once the random-forest + FinBERT stack is promoted, the PKL artifact
            will plug directly into this UI. Today&apos;s view is mocked data so
            we can finalize layout and interactions.
          </p>
        </div>
        <button className="rounded-2xl border border-emerald-400/50 px-6 py-3 text-sm font-semibold text-emerald-300 transition hover:border-emerald-300 hover:text-white">
          View integration checklist
        </button>
      </div>
    </section>
  );
}
