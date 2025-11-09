"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { gsap } from "gsap";
import { HeroSection } from "./components/HeroSection";
import { FocusControls } from "./components/FocusControls";
import { WalletSignalCard } from "./components/WalletSignalCard";
import { ActionStackCard } from "./components/ActionStackCard";
import { ClusterInsightCard } from "./components/ClusterInsightCard";
import { NarrativePanel } from "./components/NarrativePanel";
import { StatGrid } from "./components/StatGrid";
import { PriceProjectionChart } from "./components/PriceProjectionChart";
import { SentimentChart } from "./components/SentimentChart";
import { PipelinePreviewCard } from "./components/PipelinePreviewCard";
import { TrustVerificationPanel } from "./components/TrustVerificationPanel";
import {
  ClusterInsight,
  FocusMode,
  Recommendation,
  TrustDataset,
} from "./types/dashboard";

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
    return "--";
  }
  if (Math.abs(value) >= 1) {
    return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }
  return value.toFixed(4);
};

const DEMO_RECOMMENDATIONS: Recommendation[] = [
  {
    action: "restake",
    probability: 0.58,
    rationale: "Price-weighted snapshot favours compounding rewards.",
  },
  {
    action: "stake",
    probability: 0.29,
    rationale: "Neutral stance keeps capital deployed in baseline pools.",
  },
  {
    action: "liquid_stake",
    probability: 0.13,
    rationale: "Liquidity optionality remains a secondary hedge.",
  },
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
  "Models are syncing signals -- adjust the slider to see how focus shifts recommendations.";
const DEMO_NARRATIVE =
  "Claude notes a restake tilt with liquidity hedges kept light; watch deposits outpacing withdrawals.";
const DEMO_WALLET_SUMMARY =
  "Demo wallet mirrors EtherFi eETH + Pendle LP flow (refs: https://etherscan.io/token/0x35fa164735182de50811e8e2e824cfb9b6118ac2, https://etherscan.io/token/0xcd5fe23c85820f7b72d0926fc9b05b43e359b7ee, https://etherscan.io/token/0xa89a7420b19679b5a124cd1499787a6122e800d7).";

export default function Home() {
  const [focus, setFocus] = useState<FocusMode>("price");
  const [weight, setWeight] = useState(DEMO_WEIGHT); // 0 sentiment bias -> 100 price bias
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
  const [walletSummaryCopy, setWalletSummaryCopy] = useState<string | null>(
    null,
  );
  const [walletSummaryError, setWalletSummaryError] = useState<string | null>(
    null,
  );

  const fetchTrust = useCallback(async () => {
    try {
      setTrustStatus("loading");
      const response = await fetch(`${TRUST_API_URL}/trust/datasets`, {
        cache: "no-store",
      });
      if (!response.ok) {
        throw new Error(`Trust API error: ${response.status}`);
      }
      const payload = await response.json();
      setTrustDatasets(payload.datasets ?? []);
      setTrustStatus("idle");
    } catch (error) {
      console.error("Failed to load trust datasets", error);
      setTrustStatus("error");
    }
  }, [TRUST_API_URL]);

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
    fetchTrust();
  }, [fetchTrust]);

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
    const nextWeight = mode === "price" ? 75 : 25;
    setWeight(nextWeight);
    setEffectiveBias(snapWeight(nextWeight));
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
      return "--";
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
      const includeWalletSignal =
        includeWallet && walletRef.current.trim().length > 0;
      const payload = {
        price_weight: priceWeight,
        sentiment_weight: sentimentWeight,
        wallet: includeWalletSignal ? walletRef.current.trim() : null,
      };

      if (USE_DEMO_SIGNALS) {
        setRecommendations(DEMO_RECOMMENDATIONS);
        setClusterInsight(DEMO_CLUSTER);
        setModelMessage("Demo snapshot loaded locally.");
        setLastRunAt(new Date().toISOString());
        setNarrativeCopy(DEMO_NARRATIVE);
        setRecStatus("idle");
        if (includeWalletSignal) {
          setWalletSummaryCopy(DEMO_WALLET_SUMMARY);
          setWalletSummaryError(null);
        }
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
        if (includeWalletSignal) {
          if (body.wallet_summary) {
            setWalletSummaryCopy(
              `${body.wallet_summary.holdings_text ?? ""} ${
                body.wallet_summary.recent_activity_text ?? ""
              }`.trim(),
            );
          } else {
            setWalletSummaryCopy(null);
          }
          setWalletSummaryError(body.wallet_summary_error ?? null);
        }
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
        if (includeWalletSignal) {
          setWalletSummaryCopy(null);
          setWalletSummaryError(
            "Wallet inspector unavailable; showing cached model copy.",
          );
        }
      }
    },
    [weight],
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

  const handleWeightChange = (value: number) => {
    setWeight(value);
    const snapped = snapWeight(value);
    setFocus(snapped >= 50 ? "price" : "sentiment");
    setEffectiveBias(snapped);
    console.log("Preference updated", {
      sliderValue: value,
      appliedPriceWeight: snapped / 100,
    });
  };

  const handleWalletChange = (value: string) => {
    setWallet(value);
  };

  const handleRunSignal = () =>
    runInference({ trigger: "manual", includeWallet: true });

  const handleStatCardMount = (
    index: number,
    node: HTMLDivElement | null,
  ) => {
    if (node) {
      cardRefs.current[index] = node;
    }
  };

  const createChartRef =
    (index: number) => (node: HTMLDivElement | null) => {
      if (node) {
        chartRefs.current[index] = node;
      }
    };

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
      meta: `${weight}% price / ${100 - weight}% sentiment`,
    },
  ];

  const showCharts = hydrated;

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 text-white">
      <main
        ref={heroRef}
        className="mx-auto flex max-w-6xl flex-col gap-10 px-6 py-12 lg:py-16"
      >
        <HeroSection
          narrativeCopy={narrativeCopy}
          effectiveBias={effectiveBias}
        />

        <section className="grid gap-6 lg:grid-cols-[1.2fr,0.8fr]">
          <div className="panel-animate glow-border rounded-3xl border border-white/10 bg-white/5 p-6 backdrop-blur">
            <FocusControls
              focus={focus}
              weight={weight}
              tiltCopy={tiltCopy}
              highlightRef={highlightRef}
              onFocusChange={handleFocusChange}
              onWeightChange={handleWeightChange}
            />
            <WalletSignalCard
              wallet={wallet}
              onWalletChange={handleWalletChange}
              onRunSignal={handleRunSignal}
              recStatus={recStatus}
              disabled={USE_DEMO_SIGNALS}
              modelMessage={modelMessage}
              walletSummaryCopy={walletSummaryCopy}
              walletSummaryError={walletSummaryError}
            />
            <div className="mt-6 grid gap-4 lg:grid-cols-2">
              <ActionStackCard
                recommendations={recommendations}
                recStatus={recStatus}
                lastRunAt={lastRunAt}
                formatPercent={formatPercent}
              />
              <ClusterInsightCard
                clusterInsight={clusterInsight}
                recStatus={recStatus}
                formatMetricValue={formatMetricValue}
              />
            </div>
          </div>
          <NarrativePanel narrativeCopy={narrativeCopy} />
        </section>

        <StatGrid cards={statCards} onCardMount={handleStatCardMount} />

        <section className="grid gap-6 lg:grid-cols-2">
          <PriceProjectionChart
            data={mockPriceSeries}
            showCharts={showCharts}
            refCallback={createChartRef(0)}
          />
          <SentimentChart
            data={mockSentiment}
            showCharts={showCharts}
            refCallback={createChartRef(1)}
          />
        </section>

        <PipelinePreviewCard refCallback={createChartRef(2)} />

        <TrustVerificationPanel
          status={trustStatus}
          datasets={trustDatasets}
          trustApiUrl={TRUST_API_URL}
          onRefresh={fetchTrust}
        />
      </main>
    </div>
  );
}
