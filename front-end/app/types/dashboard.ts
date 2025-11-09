export type FocusMode = "price" | "sentiment";

export type Recommendation = {
  action: string;
  probability: number;
  rationale?: string;
};

export type ClusterInsight = {
  id: number;
  label?: string;
  description?: string;
  drivers?: string[];
  metrics?: Record<string, number>;
};

export type TrustDataset = {
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
    public_key?: string | null;
    commitment?: string | null;
    challenge?: string | null;
    response?: string | null;
  };
};
