"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type SentimentPoint = {
  label: string;
  value: number;
};

type SentimentChartProps = {
  data: SentimentPoint[];
  showCharts: boolean;
  refCallback: (node: HTMLDivElement | null) => void;
};

export function SentimentChart({
  data,
  showCharts,
  refCallback,
}: SentimentChartProps) {
  return (
    <div
      ref={refCallback}
      className="panel-animate glow-border rounded-3xl border border-white/10 bg-white/5 p-6 backdrop-blur"
    >
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.4em] text-slate-400">
            Sentiment fabric
          </p>
          <p className="text-lg font-medium text-white">
            Claude + FinBERT merge
          </p>
        </div>
        <span className="rounded-full border border-white/10 px-3 py-1 text-xs text-slate-300">
          32 articles
        </span>
      </div>
      <div className="mt-6 h-64">
        {showCharts ? (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data}>
              <defs>
                <linearGradient id="sentimentGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#34d399" />
                  <stop offset="100%" stopColor="#0ea5e9" />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
              <XAxis dataKey="label" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip
                contentStyle={{
                  background: "#020617",
                  border: "1px solid #1f2937",
                  borderRadius: "12px",
                }}
              />
              <Bar
                dataKey="value"
                radius={[8, 8, 0, 0]}
                fill="url(#sentimentGradient)"
              />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-full animate-pulse rounded-2xl bg-slate-800/50" />
        )}
      </div>
      <div
        className="card-animate mt-4 rounded-2xl border border-white/10 bg-black/30 p-4 text-sm text-slate-300"
        style={{ animationDelay: "0.1s" }}
      >
        <p>
          Neutral cluster is shrinking as wallet narratives skew bullish. Expect
          sentiment weight to stay elevated unless regulatory risk resurfaces.
        </p>
      </div>
    </div>
  );
}
