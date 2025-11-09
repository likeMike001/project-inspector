"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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

type Recommendation = {
  action: string;
  probability: number;
  rationale?: string;
};

type ClusterInsight = {
  id: number;
  label?: string;
  description?: string;
  drivers?: string[];
  metrics?: Record<string, number>;
};

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
const MODEL_API_URL =
  process.env.NEXT_PUBLIC_MODEL_API_URL ?? "http://localhost:8001";
const USE_DEMO_SIGNALS =
  (process.env.NEXT_PUBLIC_DEMO_SIGNALS ?? "false").toLowerCase() === "true";
const DEMO_WEIGHT = Number(process.env.NEXT_PUBLIC_DEMO_WEIGHT ?? 65);
const snapWeight = (value: number) => (value < 50 ? 50 : 60);
const formatMetricValue = (value: number | null | undefined) => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "—";
  }
  if (Math.abs(value) >= 1) {
    return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }
  return value.toFixed(4);
};

const DEMO_RECOMMENDATIONS: Recommendation[] = [
  { action: "restake", probability: 0.58, rationale: "Price-weighted snapshot favours compounding rewards." },
  { action: "stake", probability: 0.29, rationale: "Neutral stance keeps capital deployed in baseline pools." },
  { action: "liquid_stake", probability: 0.13, rationale: "Liquidity optionality remains a secondary hedge." },
];

const DEMO_CLUSTER: ClusterInsight = {
  id: 0,
  label: "Restake skew",
  description: "APR momentum trending higher alongside elevated withdrawer counts.",
  drivers: [
    "Daily APR sits near local highs with positive netflows.",
    "Depositors outpace withdrawers after semantic boost.",
  ],
  metrics: {
    daily_apr: 0.0259,
    withdraw: -3200,
    deposit: 12500,
    daily_netflow: 9300,
    withdrawers: 58,
    depositors: 72,
  },
};

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

const DEFAULT_NARRATIVE =
  "Models are syncing signals—adjust the slider to see how focus shifts recommendations.";
const DEMO_NARRATIVE =
  "Claude notes a restake tilt with liquidity hedges kept light; watch deposits outpacing withdrawals.";

export default function Home() {
  const [focus, setFocus] = useState<FocusMode>("price");
  const [weight, setWeight] = useState(DEMO_WEIGHT); // 0 sentiment bias — 100 price bias
  const [effectiveBias, setEffectiveBias] = useState(snapWeight(DEMO_WEIGHT));
  const [wallet, setWallet] = useState("");
  const [hydrated, setHydrated] = useState(false);
  const [trustDatasets, setTrustDatasets] = useState<TrustDataset[]>([]);
  const [trustStatus, setTrustStatus] =
    useState<"idle" | "loading" | "error">("idle");
  const [recStatus, setRecStatus] =
    useState<"idle" | "loading" | "error">("idle");
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [clusterInsight, setClusterInsight] = useState<ClusterInsight | null>(
    null,
  );
  const [modelMessage, setModelMessage] = useState<string | null>(null);
  const [lastRunAt, setLastRunAt] = useState<string | null>(null);
  const [narrativeCopy, setNarrativeCopy] = useState<string>(DEFAULT_NARRATIVE);

  const heroRef = useRef<HTMLDivElement | null>(null);
  const highlightRef = useRef<HTMLSpanElement | null>(null);
  const cardRefs = useRef<HTMLDivElement[]>([]);
  const chartRefs = useRef<HTMLDivElement[]>([]);
  const inferenceController = useRef<AbortController | null>(null);
  const walletRef = useRef(wallet);
  const weightRef = useRef(effectiveBias);

  useEffect(() => {
    walletRef.current = wallet;
  }, [wallet]);

  useEffect(() => {
    weightRef.current = effectiveBias;
  }, [effectiveBias]);

  useEffect(() => {
    return () => {
      inferenceController.current?.abort();
    };
  }, []);

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
  const formatPercent = (value: number | null | undefined) => {
    if (typeof value !== "number" || Number.isNaN(value)) {
      return "—";
    }
    return `${(value * 100).toFixed(1)}%`;
  };

  const runInference = useCallback(
    async ({
      trigger = "manual",
      bias,
      includeWallet = false,
    }: {
      trigger?: "auto" | "manual";
      bias?: number;
      includeWallet?: boolean;
    } = {}) => {
      const effectiveBias =
        typeof bias === "number" ? bias : weightRef.current ?? weight;
      const priceWeight = Math.min(Math.max(effectiveBias / 100, 0), 1);
      const sentimentWeight = 1 - priceWeight;
      const payload = {
        price_weight: priceWeight,
        sentiment_weight: sentimentWeight,
        wallet:
          includeWallet && walletRef.current.trim().length > 0
            ? walletRef.current.trim()
            : null,
      };

      if (USE_DEMO_SIGNALS) {
        setRecommendations(DEMO_RECOMMENDATIONS);
        setClusterInsight(DEMO_CLUSTER);
        setModelMessage("Demo snapshot loaded locally.");
        setLastRunAt(new Date().toISOString());
         setNarrativeCopy(DEMO_NARRATIVE);
        setRecStatus("idle");
        return;
      }

      inferenceController.current?.abort();
      const controller = new AbortController();
      inferenceController.current = controller;

      try {
        setRecStatus("loading");
        setModelMessage(
          trigger === "auto"
            ? "Updating signals to reflect the slider."
            : "Fetching the freshest signal mix.",
        );
        const response = await fetch(`${MODEL_API_URL}/signals`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`Model API error: ${response.status}`);
        }
        const body = await response.json();
        if (controller.signal.aborted) return;
        setRecommendations(body.recommendations ?? []);
        setClusterInsight(body.cluster ?? null);
        setModelMessage(body.message ?? null);
        setLastRunAt(body.generated_at ?? new Date().toISOString());
        setNarrativeCopy(body.narrative ?? DEFAULT_NARRATIVE);
        setRecStatus("idle");
      } catch (error) {
        if (controller.signal.aborted) return;
        console.error("Failed to fetch model signal", error);
        setRecStatus("error");
        setRecommendations(DEMO_RECOMMENDATIONS);
        setClusterInsight(DEMO_CLUSTER);
        setModelMessage(
          "Falling back to demo snapshot while the model API is unreachable.",
        );
        setLastRunAt(new Date().toISOString());
        setNarrativeCopy(DEFAULT_NARRATIVE);
      }
    },
    [],
  );

  useEffect(() => {
    const debounce = setTimeout(() => {
      runInference({
        trigger: USE_DEMO_SIGNALS ? "manual" : "auto",
        bias: effectiveBias,
      });
    }, USE_DEMO_SIGNALS ? 0 : 600);
    return () => clearTimeout(debounce);
  }, [effectiveBias, runInference]);

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
            {/* <div className="rounded-2xl border border-white/10 bg-black/30 p-4">
              <p className="text-xs uppercase tracking-[0.4em] text-slate-400">
                Claude insight
              </p>
              <p className="mt-3 text-base text-slate-100">{narrativeCopy}</p>
              <p className="mt-4 text-xs text-slate-400">
                Slider bias locked at {effectiveBias}% price ·{" "}
                {100 - effectiveBias}% sentiment.
              </p>
            </div> */}
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
                    const snapped = snapWeight(next);
                    setFocus(snapped >= 50 ? "price" : "sentiment");
                    setEffectiveBias(snapped);
                    console.log("Preference updated", {
                      sliderValue: next,
                      appliedPriceWeight: snapped / 100,
                    });
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
                  disabled={USE_DEMO_SIGNALS}
                />
                <button
                  onClick={() =>
                    runInference({ trigger: "manual", includeWallet: true })
                  }
                  disabled={recStatus === "loading" || USE_DEMO_SIGNALS}
                  className={`rounded-xl px-6 py-3 text-sm font-semibold shadow-2xl shadow-emerald-500/30 transition ${
                    recStatus === "loading"
                      ? "cursor-not-allowed bg-slate-600 text-slate-300"
                      : "bg-gradient-to-r from-emerald-400 to-cyan-400 text-slate-900"
                  }`}
                >
                  {recStatus === "loading" ? "Updating…" : "Run signal"}
                </button>
              </div>
              <p className="mt-2 text-xs text-slate-400">
                Wallet routing remains optional; we’ll pass it to the modeling
                layer when available.
              </p>
              {modelMessage && (
                <p className="mt-3 text-xs text-slate-400">
                  {recStatus === "error" ? "⚠️ " : "ℹ️ "}
                  {modelMessage}
                </p>
              )}
            </div>
            <div className="mt-6 grid gap-4 lg:grid-cols-2">
              <div className="rounded-2xl border border-white/10 bg-black/30 p-4">
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
                      No actions yet—adjust the slider or run the signal to see
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
                          <p className="text-lg font-semibold text-white">
                            {rec.action}
                          </p>
                          {rec.rationale && (
                            <p className="text-xs text-slate-400">
                              {rec.rationale}
                            </p>
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
              <div className="rounded-2xl border border-white/10 bg-black/30 p-4">
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
                    Adjust the slider or run the signal to classify the current
                    regime.
                  </p>
                )}
              </div>
            </div>
          </div>
          <div className="rounded-3xl border border-white/10 bg-gradient-to-b from-black/60 to-slate-900/40 p-6">
            <h3 className="text-sm uppercase tracking-[0.4em] text-slate-400">
              Live takeaways
            </h3>
            <div className="mt-4 rounded-2xl border border-white/10 bg-black/30 p-4 text-sm text-slate-200">
              <p>{narrativeCopy}</p>
            </div>
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
