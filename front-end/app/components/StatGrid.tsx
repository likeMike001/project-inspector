"use client";

type StatCard = {
  label: string;
  value: string;
  meta: string;
};

type StatGridProps = {
  cards: StatCard[];
  onCardMount: (index: number, node: HTMLDivElement | null) => void;
};

export function StatGrid({ cards, onCardMount }: StatGridProps) {
  return (
    <section className="grid gap-6 lg:grid-cols-3">
      {cards.map((card, idx) => (
        <div
          key={card.label}
          ref={(node) => onCardMount(idx, node)}
          className="card-animate rounded-3xl border border-white/10 bg-white/5 p-6 backdrop-blur"
          style={{ animationDelay: `${idx * 0.08}s` }}
        >
          <p className="text-sm uppercase tracking-[0.3em] text-slate-400">
            {card.label}
          </p>
          <p className="mt-3 text-4xl font-semibold text-white">{card.value}</p>
          <p className="mt-1 text-sm text-slate-300">{card.meta}</p>
        </div>
      ))}
    </section>
  );
}
