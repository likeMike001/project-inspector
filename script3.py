"""
Generate future-return style target labels for EtherFi dataset.

Logic:
- Reads etherfi_combined.csv
- Computes the next-period change in avg_7day_apr (future_apr_change)
- Defines binary target_label:
      1  → if future APR expected to increase
      0  → if future APR expected to decrease or stay the same
- Saves as etherfi_combined_labeled.csv
"""

import pandas as pd

INPUT = "etherfi_combined.csv"
OUTPUT = "etherfi_combined_labeled.csv"

def main():
    df = pd.read_csv(INPUT)
    
    # Ensure timestamp is parsed
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values("timestamp")

    # Handle missing APR safely
    if "avg_7day_apr" not in df.columns:
        raise ValueError("❌ 'avg_7day_apr' column not found in etherfi_combined.csv")

    # Forward-fill missing APR values (some rows might be <nil>)
    df["avg_7day_apr"] = pd.to_numeric(df["avg_7day_apr"], errors="coerce").fillna(method="ffill")

    # Compute future APR change (next row minus current)
    df["future_apr_change"] = df["avg_7day_apr"].shift(-1) - df["avg_7day_apr"]

    # Compute percent change
    df["future_apr_change_pct"] = df["future_apr_change"] / df["avg_7day_apr"]

    # Define binary target label
    df["target_label"] = (df["future_apr_change_pct"] > 0).astype(int)

    # Drop last row (no future value available)
    df = df.dropna(subset=["future_apr_change", "future_apr_change_pct"])

    # Save output
    df.to_csv(OUTPUT, index=False)

    print(f"✅ Labeled dataset saved to {OUTPUT}")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    print("\nPreview:")
    print(df[["timestamp", "avg_7day_apr", "future_apr_change", "future_apr_change_pct", "target_label"]].head(10))

if __name__ == "__main__":
    main()