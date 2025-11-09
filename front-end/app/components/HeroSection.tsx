"use client";

type HeroSectionProps = {
  narrativeCopy: string;
  effectiveBias: number;
};

export function HeroSection({
  narrativeCopy,
  effectiveBias,
}: HeroSectionProps) {
  return (
    <section className="hero-copy glow-border rounded-3xl border border-white/10 bg-white/5 p-8 backdrop-blur-md lg:p-12">
      <p className="mb-4 text-sm uppercase tracking-[0.4em] text-slate-300">
        Ethereum Alpha Console
      </p>
      <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
        <div className="space-y-4">
          <h1 className="text-4xl font-semibold leading-tight tracking-tight text-white sm:text-5xl">
            Blend price predictions with semantic intelligence in real time.
          </h1>
          <p className="text-lg text-slate-200 lg:max-w-2xl">
            Nudge the slider toward pure quant or narrative-driven signals, then
            review how the model reconciles the two -- complete with live
            projections, tone analysis, and confidence bands.
          </p>
        </div>
      </div>
    </section>
  );
}
