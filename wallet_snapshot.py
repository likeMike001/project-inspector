#!/usr/bin/env python3
"""
wallet_snapshot.py

Lightweight script that fetches Ethereum mainnet wallet information (native ETH
balance, ERC-20 transfers, token balances, and a textual summary) so the data
can be forwarded to downstream consumers such as Claude.

Example:
    python wallet_snapshot.py 0xf0bb20865277aBd641a307eCe5Ee04E79073416C \
        --lookback-blocks 75000 --max-events 150
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from web3 import Web3

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional for environments without python-dotenv
    load_dotenv = None


ERC20_MIN_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function",
    },
]


@dataclass
class TokenBalance:
    token_address: str
    symbol: str
    decimals: int
    balance: float


@dataclass
class TransferEvent:
    token_address: str
    symbol: str
    tx_hash: str
    block_number: int
    timestamp: str
    from_address: str
    to_address: str
    value_raw: str
    value_human: float


def _load_env_file(root: Path) -> None:
    if load_dotenv is None:
        return
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if value == "":
        raise SystemExit(f"❌ Environment variable {name} is required for wallet_snapshot.py")
    return value


def _parse_tracked_tokens(raw: str) -> List[Dict[str, Any]]:
    tokens: List[Dict[str, Any]] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split("|")]
        try:
            address = checksum(parts[0])
        except Exception:
            continue
        decimals_hint = None
        if len(parts) > 2:
            try:
                decimals_hint = int(parts[2])
            except ValueError:
                decimals_hint = None
        tokens.append(
            {
                "address": address,
                "symbol_hint": parts[1] if len(parts) > 1 else None,
                "decimals_hint": decimals_hint,
            }
        )
    return tokens


def _parse_address_list(raw: str) -> List[str]:
    result: List[str] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            result.append(checksum(chunk))
        except Exception:
            continue
    return result


def configure() -> Dict[str, Any]:
    root = Path(__file__).resolve().parent
    _load_env_file(root)
    return {
        "rpc_url": _require_env("RPC_URL"),
        "etherscan_api_key": _require_env("ETHERSCAN_API_KEY"),
        "etherscan_base_url": os.getenv("ETHERSCAN_BASE_URL", "https://api.etherscan.io/api"),
        "etherscan_page_size": int(os.getenv("ETHERSCAN_TOKENTX_PAGE_SIZE", "100")),
        "etherscan_max_pages": int(os.getenv("ETHERSCAN_TOKENTX_MAX_PAGES", "5")),
        "etherscan_throttle_seconds": float(os.getenv("ETHERSCAN_THROTTLE_SECONDS", "0.2")),
        "tracked_tokens": _parse_tracked_tokens(os.getenv("TRACKED_TOKEN_CONTRACTS", "")),
        "staking_contracts": _parse_address_list(os.getenv("STAKING_CONTRACTS", "")),
    }


_SNAPSHOT_CONFIG: Dict[str, Any] | None = None
_SNAPSHOT_WEB3: Optional[Web3] = None


def _runtime() -> Tuple[Dict[str, Any], Web3]:
    global _SNAPSHOT_CONFIG, _SNAPSHOT_WEB3
    if _SNAPSHOT_CONFIG is None or _SNAPSHOT_WEB3 is None:
        _SNAPSHOT_CONFIG = configure()
        _SNAPSHOT_WEB3 = init_web3(_SNAPSHOT_CONFIG["rpc_url"])
    return _SNAPSHOT_CONFIG, _SNAPSHOT_WEB3  # type: ignore[return-value]


def _default_lookback() -> int:
    return int(os.getenv("LOG_LOOKBACK_BLOCKS", "50000"))


def _default_max_events() -> int:
    return int(os.getenv("MAX_EVENTS", "200"))


def init_web3(rpc_url: str) -> Web3:
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        raise SystemExit(f"❌ Could not connect to RPC at {rpc_url}")
    return w3


def checksum(address: str) -> str:
    return Web3.to_checksum_address(address)


def get_eth_balance(w3: Web3, wallet: str) -> float:
    balance_wei = w3.eth.get_balance(checksum(wallet))
    return w3.from_wei(balance_wei, "ether")


def etherscan_tokentx(
    wallet: str,
    start_block: int,
    end_block: int,
    *,
    base_url: str,
    api_key: str,
    page_size: int,
    max_pages: int,
    throttle: float,
) -> List[TransferEvent]:
    normalized = wallet.lower()
    events: List[TransferEvent] = []

    for page in range(1, max_pages + 1):
        params = {
            "module": "account",
            "action": "tokentx",
            "address": normalized,
            "startblock": start_block,
            "endblock": end_block,
            "page": page,
            "offset": page_size,
            "sort": "desc",
            "apikey": api_key,
        }
        try:
            response = requests.get(base_url, params=params, timeout=20)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            print(f"[warn] Etherscan request failed on page {page}: {exc}")
            break

        if payload.get("status") != "1" or not payload.get("result"):
            if page == 1:
                print(f"[warn] Etherscan responded with no data: {payload.get('message')}")
            break

        for entry in payload["result"]:
            event = _etherscan_entry_to_event(entry)
            if event:
                events.append(event)

        if len(payload["result"]) < page_size:
            break

        if throttle > 0:
            time.sleep(throttle)

    return events


def _etherscan_entry_to_event(entry: Dict[str, Any]) -> Optional[TransferEvent]:
    try:
        block_number = int(entry["blockNumber"])
        timestamp = datetime.fromtimestamp(int(entry["timeStamp"]), tz=timezone.utc).isoformat()
        value_raw_str = entry.get("value", "0")
        value_raw = int(value_raw_str)
        decimals = int(entry.get("tokenDecimal") or 18)
        value_h = human_value(value_raw, decimals)
        return TransferEvent(
            token_address=checksum(entry["contractAddress"]),
            symbol=entry.get("tokenSymbol") or "UNKNOWN",
            tx_hash=entry["hash"],
            block_number=block_number,
            timestamp=timestamp,
            from_address=entry["from"],
            to_address=entry["to"],
            value_raw=value_raw_str,
            value_human=value_h,
        )
    except Exception:
        return None


def human_value(raw: int, decimals: int) -> float:
    decimals = int(decimals)
    scale = 10 ** decimals
    return float(raw) / scale if scale else float(raw)


def build_token_balances(
    w3: Web3,
    wallet: str,
    transfers: Iterable[TransferEvent],
    *,
    extra_addresses: Optional[Iterable[str]] = None,
    tracked_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[TokenBalance]:
    token_addresses = {t.token_address for t in transfers}
    if extra_addresses:
        token_addresses.update(extra_addresses)
    token_addresses = sorted(token_addresses)
    balances: List[TokenBalance] = []
    checksummed_wallet = checksum(wallet)

    for token_address in token_addresses:
        contract = w3.eth.contract(address=token_address, abi=ERC20_MIN_ABI)
        symbol = safe_call(contract.functions.symbol().call, default="UNKNOWN")
        decimals = safe_call(contract.functions.decimals().call, default=18)
        meta = None
        if tracked_metadata:
            meta = tracked_metadata.get(token_address.lower())
        if symbol in (None, "", "UNKNOWN") and meta and meta.get("symbol_hint"):
            symbol = meta["symbol_hint"]
        if decimals is None and meta and meta.get("decimals_hint") is not None:
            decimals = meta["decimals_hint"]
        elif isinstance(decimals, str) and decimals.isdigit():
            decimals = int(decimals)
        if decimals is None:
            decimals = 18
        raw_balance = safe_call(lambda: contract.functions.balanceOf(checksummed_wallet).call(), default=0)
        balance = human_value(raw_balance, decimals)
        balances.append(
            TokenBalance(
                token_address=token_address,
                symbol=symbol,
                decimals=int(decimals),
                balance=balance,
            )
        )

    return balances


def safe_call(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def summarize(
    wallet: str,
    transfers: List[TransferEvent],
    eth_balance: float,
    holdings_text: str,
) -> Dict[str, Any]:
    if transfers:
        latest = transfers[0]
        recent_activity = (
            f"Last transfer on {latest.timestamp}: {latest.value_human:.4f} "
            f"{latest.symbol} from {latest.from_address} to {latest.to_address} "
            f"(tx {latest.tx_hash})."
        )
    else:
        recent_activity = "No ERC-20 transfers found in the requested block window."

    return {
        "wallet": wallet,
        "snapshot_at": datetime.now(timezone.utc).isoformat(),
        "eth_balance": eth_balance,
        "recent_activity_text": recent_activity,
        "holdings_text": holdings_text,
        "transfer_count": len(transfers),
    }


def summarize_tracked_tokens(balances: List[TokenBalance], tracked_addresses: List[str]) -> List[Dict[str, Any]]:
    tracked_set = {addr.lower() for addr in tracked_addresses}
    tracked: List[Dict[str, Any]] = []
    for token in balances:
        if token.token_address.lower() in tracked_set:
            tracked.append(
                {
                    "token_address": token.token_address,
                    "symbol": token.symbol,
                    "balance": token.balance,
                    "decimals": token.decimals,
                }
            )
    return tracked


def summarize_staking_activity(
    wallet: str, transfers: List[TransferEvent], staking_contracts: List[str]
) -> List[Dict[str, Any]]:
    wallet_l = wallet.lower()
    staking_map = {addr.lower(): addr for addr in staking_contracts}
    if not staking_map:
        return []
    notes: List[Dict[str, Any]] = []
    for event in transfers:
        from_l = event.from_address.lower()
        to_l = event.to_address.lower()
        direction = None
        counterparty = None
        if from_l == wallet_l and to_l in staking_map:
            direction = "deposit"
            counterparty = staking_map[to_l]
        elif to_l == wallet_l and from_l in staking_map:
            direction = "withdraw"
            counterparty = staking_map[from_l]
        if direction:
            notes.append(
                {
                    "tx_hash": event.tx_hash,
                    "timestamp": event.timestamp,
                    "token_symbol": event.symbol,
                    "direction": direction,
                    "contract": counterparty,
                    "amount": event.value_human,
                }
            )
    return notes[:10]


def format_holdings_text(eth_balance: float, token_balances: List[TokenBalance]) -> str:
    top_tokens = sorted(
        token_balances,
        key=lambda tb: tb.balance,
        reverse=True,
    )
    excerpts = [f"ETH: {eth_balance:.4f}"]
    for token in top_tokens:
        if token.balance <= 0:
            continue
        excerpts.append(f"{token.symbol or 'token'}: {token.balance:.4f}")
        if len(excerpts) >= 5:
            break
    return ", ".join(excerpts)


def format_tracked_token_text(tracked_tokens: List[Dict[str, Any]]) -> str:
    if not tracked_tokens:
        return ""
    excerpts = []
    for token in tracked_tokens:
        symbol = token.get("symbol") or "token"
        balance = token.get("balance")
        if balance is None:
            continue
        excerpts.append(f"{symbol}: {balance:.4f}")
    return ", ".join(excerpts)


def compute_block_window(w3: Web3, lookback_blocks: int) -> Tuple[int, int]:
    end_block = w3.eth.block_number
    start_block = max(0, end_block - lookback_blocks)
    return start_block, end_block


def fetch_wallet_snapshot(
    w3: Web3,
    wallet: str,
    *,
    lookback_blocks: int,
    max_events: int,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    start_block, end_block = compute_block_window(w3, lookback_blocks)
    print(
        f"[info] Fetching wallet snapshot for {wallet} "
        f"(blocks {start_block} -> {end_block}, max {max_events} events)"
    )

    transfers = etherscan_tokentx(
        wallet,
        start_block,
        end_block,
        base_url=config["etherscan_base_url"],
        api_key=config["etherscan_api_key"],
        page_size=config["etherscan_page_size"],
        max_pages=config["etherscan_max_pages"],
        throttle=config["etherscan_throttle_seconds"],
    )
    transfers.sort(key=lambda e: e.block_number, reverse=True)
    transfers = transfers[:max_events]

    eth_balance = float(get_eth_balance(w3, wallet))
    extra_addresses = [token["address"] for token in config["tracked_tokens"]]
    tracked_meta = {token["address"].lower(): token for token in config["tracked_tokens"]}
    token_balances = build_token_balances(
        w3,
        wallet,
        transfers,
        extra_addresses=extra_addresses,
        tracked_metadata=tracked_meta,
    )

    holdings_text = format_holdings_text(eth_balance, token_balances)
    summary = summarize(wallet, transfers, eth_balance, holdings_text)
    tracked_token_summary = summarize_tracked_tokens(
        token_balances, extra_addresses
    )
    staking_activity = summarize_staking_activity(
        wallet,
        transfers,
        config["staking_contracts"],
    )

    tracked_token_text = format_tracked_token_text(tracked_token_summary)

    payload = {
        "summary": summary,
        "transfers": [asdict(t) for t in transfers],
        "token_balances": [asdict(tb) for tb in token_balances],
        "tracked_tokens": tracked_token_summary,
        "tracked_tokens_text": tracked_token_text,
        "staking_activity": staking_activity,
        "metadata": {
            "lookback_blocks": lookback_blocks,
            "max_events": max_events,
            "rpc_url": config["rpc_url"],
            "etherscan_base_url": config["etherscan_base_url"],
        },
    }
    return payload


def snapshot_wallet(
    wallet: str,
    lookback_blocks: Optional[int] = None,
    max_events: Optional[int] = None,
) -> Dict[str, Any]:
    if not Web3.is_address(wallet):
        raise ValueError(f"{wallet} is not a valid Ethereum address")

    config, w3 = _runtime()
    lb = lookback_blocks if lookback_blocks is not None else _default_lookback()
    me = max_events if max_events is not None else _default_max_events()

    payload = fetch_wallet_snapshot(
        w3,
        wallet,
        lookback_blocks=lb,
        max_events=me,
        config=config,
    )
    summary = payload["summary"]
    wallet_addr = checksum(wallet)
    response = {
        "wallet": wallet_addr,
        "fetched_at": summary["snapshot_at"],
        "eth_balance": summary["eth_balance"],
        "tokens": payload["token_balances"],
        "staking_events_inferred": payload["staking_activity"],
        "tracked_tokens": payload["tracked_tokens"],
        "transfers": payload["transfers"],
        "metadata": payload["metadata"],
        "summary_for_claude": {
            "holdings_text": summary["holdings_text"],
            "recent_activity_text": summary["recent_activity_text"],
            "tracked_token_highlights": payload["tracked_tokens"],
            "tracked_token_text": payload["tracked_tokens_text"],
        },
        "errors": [],
    }
    return response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Ethereum wallet information from on-chain + Etherscan.")
    parser.add_argument("wallet", help="Wallet address (0x...) to inspect.")
    parser.add_argument(
        "--lookback-blocks",
        type=int,
        default=int(os.getenv("LOG_LOOKBACK_BLOCKS", "50000")),
        help="Number of blocks to look back for ERC-20 transfers (default 50k).",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=int(os.getenv("MAX_EVENTS", "200")),
        help="Maximum number of ERC-20 transfers to keep (default 200).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write the JSON payload. When omitted, prints to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not Web3.is_address(args.wallet):
        raise SystemExit(f"❌ {args.wallet} is not a valid Ethereum address")

    payload = snapshot_wallet(
        args.wallet,
        lookback_blocks=args.lookback_blocks,
        max_events=args.max_events,
    )

    json_payload = json.dumps(payload, indent=2)
    if args.out:
        args.out.write_text(json_payload)
        print(f"[info] Snapshot written to {args.out}")
    else:
        print(json_payload)


if __name__ == "__main__":
    main()
