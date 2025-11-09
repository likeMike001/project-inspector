"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { gsap } from "gsap";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type FocusMode = "price" | "sentiment";

type TrustDataset = {
  id: string;
  label: string;
  status: string;
  sha256: string | null;
  last_verified_at: string | null;
  eigenlayer_attestation?: {
    simulated?: boolean;
    proof_id?: string;
    confidence?: number;
  };
  zkp_simulation?: {
    status?: string;
    scheme?: string;
  };
};

const TRUST_API_URL =
  process.env.NEXT_PUBLIC_TRUST_API_URL ?? "http://localhost:8000";

const mockPriceSeries = [
  { label: "Now", spot: 3540, projection: 3540 },
  { label: "+4h", spot: 3562, projection: 3590 },
  { label: "+8h", spot: 3551, projection: 3625 },
  { label: "+12h", spot: 3538, projection: 3655 },
  { label: "+24h", spot: 3570, projection: 3712 },
  { label: "+36h", spot: 3559, projection: 3750 },
];

const mockSentiment = [
  { label: "Positive", value: 58 },
  { label: "Neutral", value: 27 },
  { label: "Negative", value: 15 },
];

const insightBullets = [
  "LLM tone is trending optimistic as macro liquidity stabilizes.",
  "Price model spots a 5.6% upside with tight volatility bands.",
  "Neutral news weight is rising; keep an eye on regulatory chatter.",
];

export default function Home() {
  const [focus, setFocus] = useState<FocusMode>("price");
  const [weight, setWeight] = useState(65); // 0 sentiment bias — 100 price bias
  const [wallet, setWallet] = useState("");
  const [hydrated, setHydrated] = useState(false);
  const [trustDatasets, setTrustDatasets] = useState<TrustDataset[]>([]);
  const [trustStatus, setTrustStatus] =
    useState<"idle" | "loading" | "error">("idle");

  const heroRef = useRef<HTMLDivElement | null>(null);
  const highlightRef = useRef<HTMLSpanElement | null>(null);
  const cardRefs = useRef<HTMLDivElement[]>([]);
  const chartRefs = useRef<HTMLDivElement[]>([]);

  useEffect(() => {
    const id = requestAnimationFrame(() => setHydrated(true));
    return () => cancelAnimationFrame(id);
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function fetchTrust() {
      try {
        setTrustStatus("loading");
        const response = await fetch(`${TRUST_API_URL}/trust/datasets`, {
          cache: "no-store",
        });
        if (!response.ok) {
          throw new Error(`Trust API error: ${response.status}`);
        }
        const payload = await response.json();
        if (!cancelled) {
          setTrustDatasets(payload.datasets ?? []);
          setTrustStatus("idle");
        }
      } catch (error) {
        console.error("Failed to load trust datasets", error);
        if (!cancelled) {
          setTrustStatus("error");
        }
      }
    }
    fetchTrust();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!heroRef.current) return;

    const ctx = gsap.context(() => {
      gsap.from(".hero-copy", {
        opacity: 0,
        y: 30,
        duration: 0.8,
        ease: "power3.out",
      });

      gsap.from(cardRefs.current, {
        opacity: 0,
        y: 25,
        duration: 0.6,
        stagger: 0.1,
        delay: 0.2,
        ease: "power3.out",
      });

      gsap.from(chartRefs.current, {
        opacity: 0,
        y: 40,
        duration: 0.7,
        stagger: 0.15,
        delay: 0.3,
        ease: "power3.out",
      });
    }, heroRef);

    return () => ctx.revert();
  }, []);

  useEffect(() => {
    if (!highlightRef.current) return;
    gsap.to(highlightRef.current, {
      xPercent: focus === "price" ? 0 : 100,
      duration: 0.35,
      ease: "power2.out",
    });
  }, [focus]);

  const handleFocusChange = (mode: FocusMode) => {
    setFocus(mode);
    setWeight(mode === "price" ? 75 : 25);
  };

  const confidenceScore = useMemo(() => {
    const priceBias = weight / 100;
    const sentimentBias = 1 - priceBias;
    const base = focus === "price" ? 0.62 : 0.55;
    return Math.round((base + priceBias * 0.25 + sentimentBias * 0.2) * 100);
  }, [focus, weight]);

  const tiltCopy =
    focus === "price"
      ? "ML is prioritizing on-chain microstructure and order flow."
      : "LLM-derived tone will steer allocations for the next window.";

  const statCards = [
    {
      label: "Projected Move",
      value: "+5.6%",
      meta: "24h blended target",
    },
    {
      label: "Sentiment Tilt",
      value: "+0.42",
      meta: "FinBERT score",
    },
    {
      label: "Confidence",
      value: `${confidenceScore}%`,
      meta: `${weight}% price · ${100 - weight}% sentiment`,
    },
  ];

  const showCharts = hydrated;

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 text-white">
      <main
        ref={heroRef}
        className="mx-auto flex max-w-6xl flex-col gap-10 px-6 py-12 lg:py-16"
      >
        <section className="hero-copy rounded-3xl border border-white/10 bg-white/5 p-8 backdrop-blur-md lg:p-12">
          <p className="mb-4 text-sm uppercase tracking-[0.4em] text-slate-300">
            Ethereum Alpha Console
          </p>
          <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
            <div className="space-y-4">
              <h1 className="text-4xl font-semibold leading-tight tracking-tight text-white sm:text-5xl">
                Blend price predictions with semantic intelligence in real time.
              </h1>
              <p className="text-lg text-slate-200 lg:max-w-2xl">
                Nudge the slider toward pure quant or narrative-driven signals,
                then review how the model reconciles the two—complete with live
                projections, tone analysis, and confidence bands.
              </p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-black/30 px-6 py-5">
              <p className="text-sm uppercase tracking-[0.3em] text-emerald-300">
                Active window
              </p>
              <p className="text-2xl font-semibold">LSTM x FinBERT v0.3</p>
              <p className="text-xs text-slate-300">Next refresh in 17 min</p>
            </div>
          </div>
        </section>

        <section className="grid gap-6 lg:grid-cols-[1.2fr,0.8fr]">
          <div className="rounded-3xl border border-white/10 bg-white/5 p-6 backdrop-blur">
            <div className="flex flex-col gap-4">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold">Preference Control</h2>
                <span className="text-sm text-slate-300">{tiltCopy}</span>
              </div>
              <div className="relative grid grid-cols-2 rounded-2xl border border-white/10 bg-black/30 p-1 text-sm font-medium text-slate-300">
                <span
                  ref={highlightRef}
                  className="absolute inset-y-1 left-1 z-0 w-1/2 rounded-xl bg-gradient-to-r from-emerald-400/80 to-cyan-400/80 mix-blend-screen"
                />
                {["price", "sentiment"].map((mode) => (
                  <button
                    key={mode}
                    className={`z-10 rounded-xl px-4 py-2 transition-colors ${
                      focus === mode ? "text-black" : "text-slate-300"
                    }`}
                    onClick={() => handleFocusChange(mode as FocusMode)}
                  >
                    {mode === "price" ? "Price precision" : "Semantic context"}
                  </button>
                ))}
              </div>
              <div className="space-y-2">
                <label className="flex items-center justify-between text-sm text-slate-300">
                  <span>Weight toward price action</span>
                  <span className="font-semibold text-white">{weight}%</span>
                </label>
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={weight}
                  onChange={(e) => {
                    const next = Number(e.target.value);
                    setWeight(next);
                    setFocus(next >= 50 ? "price" : "sentiment");
                  }}
                  className="h-1 w-full appearance-none rounded-full bg-slate-700 accent-emerald-400"
                />
              </div>
            </div>
            <div className="mt-6 rounded-2xl bg-black/30 p-4">
              <label className="text-sm text-slate-300">
                Optional wallet signal
              </label>
              <div className="mt-2 flex flex-col gap-3 sm:flex-row">
                <input
                  value={wallet}
                  onChange={(e) => setWallet(e.target.value)}
                  placeholder="0x… (coming soon to modeling layer)"
                  className="flex-1 rounded-xl border border-white/10 bg-black/40 px-4 py-3 text-sm text-white outline-none transition focus:border-emerald-400/70"
                />
                <button className="rounded-xl bg-gradient-to-r from-emerald-400 to-cyan-400 px-6 py-3 text-sm font-semibold text-slate-900 shadow-2xl shadow-emerald-500/30">
                  Queue signal
                </button>
              </div>
              <p className="mt-2 text-xs text-slate-400">
                We’ll incorporate per-wallet flows once the feature store is
                wired to on-chain traces.
              </p>
            </div>
          </div>
          <div className="rounded-3xl border border-white/10 bg-gradient-to-b from-black/60 to-slate-900/40 p-6">
            <h3 className="text-sm uppercase tracking-[0.4em] text-slate-400">
              Live takeaways
            </h3>
            <ul className="mt-4 space-y-4 text-sm text-slate-200">
              {insightBullets.map((line) => (
                <li key={line} className="flex gap-3">
                  <span className="mt-1 h-2 w-2 rounded-full bg-emerald-400" />
                  <p>{line}</p>
                </li>
              ))}
            </ul>
            <div className="mt-6 rounded-2xl border border-white/10 bg-black/30 p-5 text-sm text-slate-300">
              <p className="text-xs uppercase tracking-[0.4em] text-emerald-300">
                Model status
              </p>
              <p className="mt-2 text-lg font-semibold text-white">
                Streaming synthetic news + CoinGecko spot feed
              </p>
              <p className="mt-1 text-slate-400">
                Last Claude batch · 32 articles · 58% positive tone
              </p>
            </div>
          </div>
        </section>

        <section className="grid gap-6 lg:grid-cols-3">
          {statCards.map((card, idx) => (
            <div
              key={card.label}
              ref={(el) => {
                if (el) cardRefs.current[idx] = el;
              }}
              className="rounded-3xl border border-white/10 bg-white/5 p-6 backdrop-blur"
            >
              <p className="text-sm uppercase tracking-[0.3em] text-slate-400">
                {card.label}
              </p>
              <p className="mt-3 text-4xl font-semibold text-white">
                {card.value}
              </p>
              <p className="mt-1 text-sm text-slate-300">{card.meta}</p>
            </div>
          ))}
        </section>

        <section className="grid gap-6 lg:grid-cols-2">
          <div
            ref={(el) => {
              if (el) chartRefs.current[0] = el;
            }}
            className="rounded-3xl border border-white/10 bg-white/5 p-6 backdrop-blur"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.4em] text-slate-400">
                  Price projection
                </p>
                <p className="text-lg font-medium text-white">
                  ML track vs live spot
                </p>
              </div>
              <span className="rounded-full border border-white/10 px-3 py-1 text-xs text-slate-300">
                Horizon · 36h
              </span>
            </div>
            <div className="mt-6 h-64">
              {showCharts ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={mockPriceSeries}>
                    <defs>
                      <linearGradient id="projectionStroke" x1="0" x2="1" y1="0" y2="0">
                        <stop offset="0%" stopColor="#34d399" />
                        <stop offset="100%" stopColor="#0ea5e9" />
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
                    <XAxis dataKey="label" stroke="#94a3b8" />
                    <YAxis
                      stroke="#94a3b8"
                      tickFormatter={(value) => `$${value}`}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "#020617",
                        border: "1px solid #1f2937",
                        borderRadius: "12px",
                      }}
                    />
                    <Line
                      type="monotone"
                      dataKey="spot"
                      stroke="#64748b"
                      strokeWidth={2}
                      dot={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="projection"
                      stroke="url(#projectionStroke)"
                      strokeWidth={3}
                      dot={{
                        stroke: "#34d399",
                        strokeWidth: 2,
                        r: 4,
                      }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full animate-pulse rounded-2xl bg-slate-800/50" />
              )}
            </div>
          </div>

          <div
            ref={(el) => {
              if (el) chartRefs.current[1] = el;
            }}
            className="rounded-3xl border border-white/10 bg-white/5 p-6 backdrop-blur"
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
                  <BarChart data={mockSentiment}>
                    <defs>
                      <linearGradient
                        id="sentimentGradient"
                        x1="0"
                        y1="0"
                        x2="0"
                        y2="1"
                      >
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
            <div className="mt-4 rounded-2xl border border-white/10 bg-black/30 p-4 text-sm text-slate-300">
              <p>
                Neutral cluster is shrinking as wallet narratives skew bullish.
                Expect sentiment weight to stay elevated unless regulatory risk
                resurfaces.
              </p>
            </div>
          </div>
        </section>

        <section
          ref={(el) => {
            if (el) chartRefs.current[2] = el;
          }}
          className="rounded-3xl border border-white/10 bg-gradient-to-r from-slate-900/70 to-black/50 p-6 backdrop-blur"
        >
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.4em] text-slate-400">
                Pipeline preview
              </p>
              <h3 className="text-2xl font-semibold text-white">
                Synthetic news → sentiment → feature store → PKL predictions
              </h3>
              <p className="mt-2 text-sm text-slate-300">
                Once the random-forest + FinBERT stack is promoted, the PKL
                artifact will plug directly into this UI. Today’s view is mocked
                data so we can finalize layout and interactions.
              </p>
            </div>
            <button className="rounded-2xl border border-emerald-400/50 px-6 py-3 text-sm font-semibold text-emerald-300 transition hover:border-emerald-300 hover:text-white">
              View integration checklist
            </button>
          </div>
        </section>

        <section className="rounded-3xl border border-white/10 bg-white/5 p-6 backdrop-blur">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.4em] text-slate-400">
                Trust verification
              </p>
              <h3 className="text-2xl font-semibold text-white">
                EigenLayer-style proofs for static datasets
              </h3>
              <p className="mt-2 text-sm text-slate-300">
                Hashes are recomputed on the trust service and exposed via the
                `trust_layer` API. When models go live, their artifacts drop
                into the same pipeline.
              </p>
            </div>
            <span className="rounded-2xl border border-emerald-400/50 px-6 py-3 text-sm font-semibold text-emerald-300">
              {trustStatus === "loading"
                ? "Refreshing proofs…"
                : trustStatus === "error"
                ? "Verification unavailable"
                : "Verified by EigenLayer"}
            </span>
          </div>
          <div className="mt-6 grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {trustStatus === "loading" && (
              <div className="h-32 animate-pulse rounded-2xl border border-white/10 bg-slate-800/40" />
            )}
            {trustStatus === "error" && (
              <div className="rounded-2xl border border-rose-400/30 bg-rose-950/40 p-4 text-sm text-rose-200">
                Unable to reach the trust API. Is the FastAPI service running on{" "}
                <code>{TRUST_API_URL}</code>?
              </div>
            )}
            {trustStatus !== "error" &&
              trustDatasets.map((dataset) => {
                const verified = dataset.status === "ok";
                return (
                  <div
                    key={dataset.id}
                    className="rounded-2xl border border-white/10 bg-black/40 p-4 text-sm text-slate-200"
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
                      {dataset.sha256 ?? "—"}
                    </p>
                    <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-slate-400">
                      <span>
                        {dataset.last_verified_at
                          ? new Date(dataset.last_verified_at).toLocaleString()
                          : "n/a"}
                      </span>
                      {dataset.zkp_simulation?.status && (
                        <span className="rounded-full border border-white/10 px-2 py-0.5 text-emerald-200">
                          ZKP: {dataset.zkp_simulation.status}
                        </span>
                      )}
                    </div>
                  </div>
                );
              })}
          </div>
        </section>
      </main>
    </div>
  );
}
