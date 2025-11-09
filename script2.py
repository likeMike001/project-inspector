"""
Merge EtherFi + eETH datasets into a single time-aligned CSV.

Expected input files in the ROOT folder:

1) eETH_APR.csv
   Columns:
     day,block_day,daily_apr,apr_padding_high,avg_7day_apr,avg_30day_apr,
     apr_padding_low,all_time_apr,oracle_rate,dex_rate,pad_high,nav_ratio,
     constant_one,pad_low,premium_or_discount,premium_or_discount_perc,
     premium_or_discount_bips,avg_7day_apr_cnt,avg_30day_apr_cnt,
     premium_or_discount_perc_cnt,buffer_eth,pending_withdrawals,cum_withdraw,
     cum_deposit,cum_netflow,withdraw,deposit,daily_netflow,withdrawers,
     depositors,first_time_deposits,repeat_deposits,first_time_depositors,
     repeat_depositors

2) eETH_Active_Holder.csv
   Columns:
     granularity_day,holder_class,addresses,token_balance_usd,active_address,
     balance_tracked,num_month,month_change,total_unique_holders

3) etherFI_Liquid_Valut_Deposit_retention.csv
   Columns:
     date,total_deposits_t,weighted_retention_rate,cohort,total_deposits,
     week_1_retention,...,week_12_retention

4) etherFI_deposit_Retention.csv
   Columns:
     date,total_deposits_t,weighted_retention_rate,cohort,total_deposits,
     week_1_retention,...,week_12_retention

5) etherFI_Holder_retention.csv
   Columns:
     date,total_wallets_per_cohort_t,weighted_retention_rate,cohort,
     total_wallets_per_cohort,month_1_retention,...,month_12_retention

Output:
  etherfi_combined.csv
"""

import pandas as pd


def load_eeth_apr(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 'day' → timestamp
    df["timestamp"] = pd.to_datetime(df["day"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    # Keep a focused subset of useful numeric features
    keep_cols = [
        "timestamp",
        "daily_apr",
        "avg_7day_apr",
        "avg_30day_apr",
        "all_time_apr",
        "oracle_rate",
        "dex_rate",
        "nav_ratio",
        "premium_or_discount",
        "premium_or_discount_perc",
        "cum_withdraw",
        "cum_deposit",
        "cum_netflow",
        "withdraw",
        "deposit",
        "daily_netflow",
        "withdrawers",
        "depositors",
        "first_time_deposits",
        "repeat_deposits",
        "first_time_depositors",
        "repeat_depositors",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]
    print(f"✅ Loaded eETH_APR.csv with {len(df)} rows")
    return df


def load_eeth_active_holders(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 'granularity_day' → timestamp
    df["timestamp"] = pd.to_datetime(df["granularity_day"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    keep_cols = [
        "timestamp",
        "holder_class",
        "addresses",
        "token_balance_usd",
        "active_address",
        "balance_tracked",
        "num_month",
        "month_change",
        "total_unique_holders",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    # Aggregate per day over holder_class (summing or averaging)
    # So we get one row per timestamp
    agg = {
        "addresses": "sum",
        "token_balance_usd": "sum",
        "active_address": "sum",
        "balance_tracked": "sum",
        "num_month": "max",
        "month_change": "mean",
        "total_unique_holders": "max",
    }
    df_agg = df.groupby("timestamp", as_index=False).agg(
        {k: v for k, v in agg.items() if k in df.columns}
    )

    print(f"✅ Loaded eETH_Active_Holder.csv with {len(df_agg)} aggregated rows")
    return df_agg


def load_retention_weekly(path: str, prefix: str) -> pd.DataFrame:
    """
    For:
      etherFI_Liquid_Valut_Deposit_retention.csv
      etherFI_deposit_Retention.csv

    - Parse 'date' as timestamp
    - Keep retention-related columns and prefix them so they don't collide
    """
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    base_cols = ["timestamp", "weighted_retention_rate"]
    week_cols = [
        "week_1_retention",
        "week_2_retention",
        "week_3_retention",
        "week_4_retention",
        "week_8_retention",
        "week_12_retention",
    ]
    keep_cols = [c for c in base_cols + week_cols if c in df.columns]
    df = df[keep_cols]

    # Prefix all non-timestamp columns
    rename_map = {c: f"{prefix}_{c}" for c in df.columns if c != "timestamp"}
    df = df.rename(columns=rename_map)

    print(f"✅ Loaded {path} with {len(df)} rows (prefix={prefix})")
    return df


def load_retention_monthly(path: str, prefix: str) -> pd.DataFrame:
    """
    For:
      etherFI_Holder_retention.csv

    - Parse 'date' as timestamp
    - Keep monthly retention metrics, prefixed
    """
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    base_cols = ["timestamp", "weighted_retention_rate"]
    month_cols = [
        "month_1_retention",
        "month_2_retention",
        "month_3_retention",
        "month_4_retention",
        "month_5_retention",
        "month_6_retention",
        "month_12_retention",
    ]
    keep_cols = [c for c in base_cols + month_cols if c in df.columns]
    df = df[keep_cols]

    rename_map = {c: f"{prefix}_{c}" for c in df.columns if c != "timestamp"}
    df = df.rename(columns=rename_map)

    print(f"✅ Loaded {path} with {len(df)} rows (prefix={prefix})")
    return df


def main():
    # Load all five datasets from root
    eeth_apr = load_eeth_apr("eETH_APR.csv")
    eeth_holders = load_eeth_active_holders("eETH_Active_Holder.csv")

    liquid_ret = load_retention_weekly(
        "etherFI_Liquid_Valut_Deposit_retention.csv", prefix="liquid"
    )
    deposit_ret = load_retention_weekly(
        "etherFI_deposit_Retention.csv", prefix="deposit"
    )
    holder_ret = load_retention_monthly(
        "etherFI_Holder_retention.csv", prefix="holder"
    )

    # Start merged frame from the highest-frequency data: APR (daily)
    merged = eeth_apr.sort_values("timestamp")

    # Merge active holder data by nearest timestamp (backward in time)
    merged = pd.merge_asof(
        merged.sort_values("timestamp"),
        eeth_holders.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )

    # Merge each retention dataset also by nearest timestamp
    merged = pd.merge_asof(
        merged.sort_values("timestamp"),
        holder_ret.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )

    merged = pd.merge_asof(
        merged.sort_values("timestamp"),
        deposit_ret.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )

    merged = pd.merge_asof(
        merged.sort_values("timestamp"),
        liquid_ret.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )

    # Fill gaps forward and then backward (for early rows)
    merged = merged.sort_values("timestamp")
    merged = merged.ffill().bfill()

    out_path = "etherfi_combined.csv"
    merged.to_csv(out_path, index=False)

    print("\n✅ Final merged dataset saved to", out_path)
    print(f"   Rows: {len(merged)} | Columns: {len(merged.columns)}")
    print("\nPreview:")
    print(merged.head(10))


if __name__ == "__main__":
    main()