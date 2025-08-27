#!/usr/bin/env python3
# bnb_liq_benchmark_hardened.py
#
# BNB Chain on-chain quoting benchmark (extra-hardened).
#
# Venues:
#   - Pancake v2
#   - Pancake v3 (single + 2-hop)
#   - Uniswap v3 (single + 2-hop)
#   - THENA/Algebra (single)
#   - Wombat (stable leg, USDT<->USDC)
#   - DODO v2 helper (optional), DODO direct pools, DODO factory discovery (view)
#
# Improvements:
#   - DODO discovery via factory view calls (getDODOPoolBidirection) per docs
#   - Optional old log-scan retained, but only used if explicitly requested
#   - Robust revert decoding: supports tuple-wrapped uint256[] (offset=0x20) and plain arrays
#   - Always dumps undecoded payloads (console + raw_payloads.csv)
#   - Safer CLI parsing for --dodo-pools (warn & ignore bad items)
#
# DODO "KJUDGE_ERROR" context (FeeRateDIP3Impl):
#   For some pool versions, if K==0 and base/quote reserve is 0, getFeeRate reverts KJUDGE_ERROR.
#   We skip RouteHelper when it errors and quote pools directly (querySellBase/querySellQuote)
#   after checking reserves > 0.
from __future__ import annotations

import os
import random, time
import math
import csv
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

from web3 import Web3
from web3.types import FilterParams
from eth_abi import decode as abi_decode
import json
# === Oracle + DB + PoolCache plumbing ===================================
# (1) ORACLE DEFINITIONS -------------------------------------------------------

from dataclasses import dataclass
import sqlite3, json, pathlib

# --- ABIs --------------------------------------------------------------------
# ---- Chainlink minimal ABI ----
AGG_V3_ABI = [
    {"inputs":[],"name":"decimals","outputs":[{"internalType":"uint8","name":"","type":"uint8"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"description","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"latestRoundData","outputs":[
        {"internalType":"uint80","name":"roundId","type":"uint80"},
        {"internalType":"int256","name":"answer","type":"int256"},
        {"internalType":"uint256","name":"startedAt","type":"uint256"},
        {"internalType":"uint256","name":"updatedAt","type":"uint256"},
        {"internalType":"uint80","name":"answeredInRound","type":"uint80"}],
     "stateMutability":"view","type":"function"}
]

BAND_STDREF_ABI = [
    {"inputs":[{"name":"_base","type":"string"},{"name":"_quote","type":"string"}],
     "name":"getReferenceData","outputs":[
        {"name":"rate","type":"uint256"},
        {"name":"lastUpdatedBase","type":"uint256"},
        {"name":"lastUpdatedQuote","type":"uint256"}],
     "stateMutability":"view","type":"function"}
]

# Pyth EVM (read-only getters on on-chain cached prices)
IPYTH_ABI = [
    {"inputs":[{"internalType":"bytes32","name":"id","type":"bytes32"}],
     "name":"getPriceUnsafe",
     "outputs":[{"components":[
         {"internalType":"int64","name":"price","type":"int64"},
         {"internalType":"uint64","name":"conf","type":"uint64"},
         {"internalType":"int32","name":"expo","type":"int32"},
         {"internalType":"uint256","name":"publishTime","type":"uint256"}],
         "internalType":"struct PythStructs.Price","name":"","type":"tuple"}],
     "stateMutability":"view","type":"function"}
]

DIA_ORACLE_ABI = [
    {"inputs":[{"name":"key","type":"string"}],
     "name":"getValue",
     "outputs":[{"name":"value","type":"uint128"},{"name":"timestamp","type":"uint128"}],
     "stateMutability":"view","type":"function"}
]

# Binance Oracle - Feed Adapter (Chainlink-like, exposes latestRoundData/decimals/...)
BINANCE_FEED_ADAPTER_ABI = [
    {"inputs":[],"name":"decimals","outputs":[{"internalType":"uint8","name":"","type":"uint8"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"description","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"version","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"latestRoundData","outputs":[
        {"internalType":"uint80","name":"roundId","type":"uint80"},
        {"internalType":"int256","name":"answer","type":"int256"},
        {"internalType":"uint256","name":"startedAt","type":"uint256"},
        {"internalType":"uint256","name":"updatedAt","type":"uint256"},
        {"internalType":"uint80","name":"answeredInRound","type":"uint80"}],
     "stateMutability":"view","type":"function"},
    {"inputs":[],"name":"latestAnswer","outputs":[{"internalType":"int256","name":"","type":"int256"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"latestTimestamp","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"}
]

# Binance Oracle - Feed Registry (name-based queries + synthetic pairs)
BINANCE_FEED_REGISTRY_ABI = [
    {"inputs":[{"internalType":"string","name":"base","type":"string"},{"internalType":"string","name":"quote","type":"string"}],
     "name":"decimalsByName","outputs":[{"internalType":"uint8","name":"","type":"uint8"}],
     "stateMutability":"view","type":"function"},
    {"inputs":[{"internalType":"string","name":"base","type":"string"},{"internalType":"string","name":"quote","type":"string"}],
     "name":"latestRoundDataByName","outputs":[
        {"internalType":"uint80","name":"roundId","type":"uint80"},
        {"internalType":"int256","name":"answer","type":"int256"},
        {"internalType":"uint256","name":"startedAt","type":"uint256"},
        {"internalType":"uint256","name":"updatedAt","type":"uint256"},
        {"internalType":"uint80","name":"answeredInRound","type":"uint80"}],
     "stateMutability":"view","type":"function"}
]

# --- Uniswap/Pancake V3 minimal ABIs ----
UNIV3_POOL_ABI = [
    {"name":"token0","inputs":[],"outputs":[{"type":"address"}],"stateMutability":"view","type":"function"},
    {"name":"token1","inputs":[],"outputs":[{"type":"address"}],"stateMutability":"view","type":"function"},
    {"name":"observe","inputs":[{"type":"uint32[]","name":"secondsAgos"}],
     "outputs":[{"type":"int56[]","name":"tickCumulatives"},{"type":"uint160[]","name":"secondsPerLiquidityCumulativeX128"}],
     "stateMutability":"view","type":"function"},
]

UNIV3_FACTORY_ABI = [
    {"name":"getPool","inputs":[{"type":"address","name":"tokenA"},{"type":"address","name":"tokenB"},{"type":"uint24","name":"fee"}],
     "outputs":[{"type":"address"}],"stateMutability":"view","type":"function"},
]

# --- Tellor minimal ABI (Flex & legacy-friendly) ---
# Preferred on modern Tellor: getDataBefore(queryId, timestamp) -> (bool, bytes, uint256)
# Legacy fallback:           getCurrentValue(queryId)          -> (bool, bytes, uint256)
TELLOR_ABI_MIN = [
    {"name":"getDataBefore","inputs":[{"type":"bytes32","name":"_queryId"},{"type":"uint256","name":"_timestamp"}],
     "outputs":[{"type":"bool","name":"_ifRetrieve"},{"type":"bytes","name":"_value"},{"type":"uint256","name":"_timestampRetrieved"}],
     "stateMutability":"view","type":"function"},
    {"name":"getCurrentValue","inputs":[{"type":"bytes32","name":"_queryId"}],
     "outputs":[{"type":"bool","name":"_ifRetrieve"},{"type":"bytes","name":"_value"},{"type":"uint256","name":"_timestampRetrieved"}],
     "stateMutability":"view","type":"function"},
]

ARCHIVE_BSC_RPC = os.environ.get("ARCHIVE_BSC_RPC", "").strip()
_ARCHIVE_URLS = [u.strip() for u in os.environ.get("BSC_ARCHIVE_RPC","").split(",") if u.strip()]

def _switch_to_archive_tier(reason: str=""):
    global _RPC_URLS
    if not _ARCHIVE_URLS:
        raise RuntimeError("Pinned read needs archive but ARCHIVE_BSC_RPC is empty")
    _RPC_URLS = list(_ARCHIVE_URLS)
    _init_rpc_health(_RPC_URLS)
    _update_active_w3(_best_rpc())
    if RPC_DEBUG:
        print(f"[rpc] switched to ARCHIVE tier ({reason})")

@dataclass
class PinnedContext:
    url: str
    block_number: int
    block_hash: str

def get_pinned_context() -> PinnedContext:
    # snapshot on the *current* endpoint
    blk = with_retries(lambda: w3.eth.get_block('latest'), label="getBlock(latest)")
    ctx = PinnedContext(ACTIVE_RPC_URL, int(blk.number), blk.hash.hex())
    return ctx

def with_retries_no_rotate(fn, *, attempts=3, base_delay=0.25, max_delay=1.5, jitter=0.25, label=""):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last = e
            if not _is_transient_jsonrpc_error(e):
                raise
            if i == attempts - 1:
                break
            delay = min(max_delay, base_delay * (2 ** i))
            delay *= (1.0 + jitter * (random.random() - 0.5))
            if RPC_DEBUG or label:
                print(f"[retry-no-rotate] {label} attempt {i+1}/{attempts}")
            time.sleep(delay)
    raise last

def call_pinned_strict(fn, ctx: PinnedContext, *, label: str):
    # ensure we are on the same endpoint
    if ACTIVE_RPC_URL != ctx.url:
        _update_active_w3(ctx.url)
    try:
        return with_retries_no_rotate(lambda: fn.call(block_identifier=ctx.block_hash), label=f"{label}[hash]")
    except Exception as e:
        if _is_pruned_or_blockgap(e):
            # escalate: this endpoint cannot serve the pinned block -> switch to archive-tier
            _mark_rpc_failure(ctx.url)
            # choose a known archive endpoint here (see C)
            _switch_to_archive_tier(reason=f"{label} pruned")
            _update_active_w3(ACTIVE_RPC_URL)
            return with_retries(lambda: fn.call(block_identifier=ctx.block_hash), label=f"{label}[hash-archive]")
        raise

@dataclass
class OracleSpec:
    src: str          # "chainlink" | "band" | "pyth" | "dia" | "chainlink_ratio"
    address: str      # feed contract (or stdref/pyth core/dia reader)
    pair: str         # e.g. "ETH/USD", "USDT/USD" (we normalize to ETHb/USDT later)
    extra: dict       # source-specific (e.g., priceId for Pyth)

@dataclass
class OracleReading:
    src: str
    pair: str
    price: float | None  # normalized to "ETHb/USDT" (or requested pair) as float
    updated_at: int | None
    stale: bool
    raw_note: str


def _is_transient_jsonrpc_error(e: Exception) -> bool:
    s = str(e).lower()
    return any(x in s for x in (
        "timeout", "timed out", "connection reset", "econnreset",
        "502 bad gateway", "503 service unavailable", "429 too many requests",
        "max retries exceeded", "temporarily unavailable", "ssl",
    ))

def _is_pruned_or_blockgap(e: Exception) -> bool:
    s = str(e).lower()
    return any(x in s for x in (
        "missing trie node", "state unavailable", "unknown block",
        "header not found", "pruned", "ancient block",
        "not found for block", "snapshot not found",
    ))

def _parse_oracle_arg(s: str) -> OracleSpec:
    """
    CLI formats (repeatable --oracle):
      chainlink:PAIR@0xfeed
      chainlink_ratio:BASE@0xbase,USDT@0xusdt     # BASE/USDT via USD legs (addresses point to */USD feeds)
      band:PAIR@0xstdref
      pyth:PAIR@0xpyth:PRICE_ID_HEX
      dia:PAIR@0xdia:KEY_STRING
      binance:PAIR@0xfeed_adapter                 # Binance FeedAdapter (*/USD)
      binance_registry:PAIR@0xregistry:BASE,QUOTE # Binance FeedRegistry by token names (e.g., ETH,USD)
    """
    typ, rest = s.split(":", 1)

    if typ == "chainlink_ratio":
        # Expect exactly "BASE@0x...,USDT@0x..."
        if "," not in rest:
            raise ValueError("chainlink_ratio requires 'BASE@addr,QUOTE@addr'")
        base_part, quote_part = rest.split(",", 1)
        if "@" not in base_part or "@" not in quote_part:
            raise ValueError("chainlink_ratio legs must be like 'ETH@0x..,USDT@0x..'")
        base_sym, base_addr = base_part.split("@", 1)
        quote_sym, quote_addr = quote_part.split("@", 1)
        pair = f"{base_sym}/{quote_sym}"
        return OracleSpec("chainlink_ratio", "", pair, {
            "base_sym": base_sym.upper(),
            "quote_sym": quote_sym.upper(),
            "base_addr": CS(base_addr),
            "quote_addr": CS(quote_addr),
        })

    pair, addr_and_extra = rest.split("@", 1)

    if typ == "pyth":
        addr, price_id = addr_and_extra.split(":", 1)
        return OracleSpec("pyth", CS(addr), pair, {"priceId": price_id})

    if typ == "dia":
        addr, key = addr_and_extra.split(":", 1)
        return OracleSpec("dia", CS(addr), pair, {"key": key})

    if typ == "binance":
        # feed adapter per pair (e.g., ETH/USD)
        return OracleSpec("binance", CS(addr_and_extra), pair, {})

    if typ == "binance_registry":
        # @0xregistry:BASE,QUOTE
        addr, names = addr_and_extra.split(":", 1)
        base_name, quote_name = [t.strip() for t in names.split(",", 1)]
        return OracleSpec("binance_registry", CS(addr), pair,
                          {"base_name": base_name, "quote_name": quote_name})

    # band / chainlink (simple)
    return OracleSpec(typ, CS(addr_and_extra), pair, {})

def _is_stale(ts: Optional[int], stale_secs: int) -> bool:
    if ts is None: 
        return True
    try:
        return (int(time.time()) - int(ts)) > int(stale_secs)
    except Exception:
        return True

def _stale_flag(ts: Optional[int], stale_secs: int) -> tuple[bool, int]:
    return _is_stale(ts, stale_secs), (int(ts) if ts is not None else 0)

def _pair_to_tuple(p: str) -> tuple[str,str]:
    base, quote = [t.strip().upper() for t in p.split("/", 1)]
    return base, quote

# --- Label helpers  -------------------------
def _addr6(a: str) -> str:
    try:
        return CS(a)[-6:]
    except Exception:
        return (a or "")[-6:]

def _lab(name: str, addr: str, ctx: str | None = None) -> str:
    # "name@a6[ctx]"   e.g. chainlink.latestRoundData@c0ffee[ETH/USD]
    base = f"{name}@{_addr6(addr)}"
    return f"{base}[{ctx}]" if ctx else base

def _ensure_code(addr: str):
    code = eth_get_code_safe(CS(addr))
    if not code or len(code) == 0:
        raise RuntimeError(f"no code at {addr} (EOA or wrong network)")
# ---------------------------------------------------------------------------

def _read_chainlink_feed(addr: str, *, block_number: int | None = None):
    """
    Returns (answer, decimals, updatedAt) for a Chainlink feed.
    Rejects incomplete/invalid rounds (answer<=0 or answeredInRound<roundId).
    """
    _ensure_code(addr)
    c = w3.eth.contract(address=CS(addr), abi=AGG_V3_ABI)
    if block_number is not None:
        rd  = call_at(
            c.functions.latestRoundData(),
            block_number=block_number,
            label=_lab("chainlink.latestRoundData", addr)
        )
        dec = call_at(
            c.functions.decimals(),
            block_number=block_number,
            label=_lab("chainlink.decimals", addr)
        )
    else:
        rd  = c.functions.latestRoundData().call()
        dec = c.functions.decimals().call()

    round_id, answer, _started_at, updated_at, answered_in_round = rd
    if int(answer) <= 0:
        raise RuntimeError("chainlink answer <= 0")
    if int(answered_in_round) < int(round_id):
        raise RuntimeError("chainlink round incomplete")
    return int(answer), int(dec), int(updated_at)


def _read_band_pair(stdref_addr: str, base: string, quote: string, *, block_number: int | None = None):
    _ensure_code(stdref_addr)
    c = w3.eth.contract(address=CS(stdref_addr), abi=BAND_STDREF_ABI)
    if block_number is not None:
        rate, tB, tQ = call_at(
            c.functions.getReferenceData(base, quote),
            block_number=block_number,
            label=_lab("band.getReferenceData", stdref_addr, f"{base}/{quote}")
        )
    else:
        rate, tB, tQ = c.functions.getReferenceData(base, quote).call()
    return float(rate) / 1e18, int(max(tB, tQ))


def _read_pyth_price(pyth_addr: str, price_id_hex: str, *, block_number: int | None = None):
    from hexbytes import HexBytes
    _ensure_code(pyth_addr)
    c = w3.eth.contract(address=CS(pyth_addr), abi=IPYTH_ABI)
    pid = HexBytes(price_id_hex)  # bytes32
    pid_short = price_id_hex[:10]
    if block_number is not None:
        px = call_at(
            c.functions.getPriceUnsafe(pid),
            block_number=block_number,
            label=_lab("pyth.getPriceUnsafe", pyth_addr, pid_short)
        )
    else:
        px = c.functions.getPriceUnsafe(pid).call()
    price, expo, ts = int(px[0]), int(px[2]), int(px[3])
    val = float(price) * (10.0 ** expo)  # expo can be negative (scale)
    return val, ts


def _read_dia_value(dia_addr: str, key: str, *, block_number: int | None = None):
    """
    DIA: getValue(string) -> (uint128 value, uint128 timestamp)
    Common failure modes:
      - wrong chain or address => no code (caught early)
      - ABI mismatch (some proxies expose different selectors)
      - key mismatch (returns 0/old ts on some deployments; others revert)
    """
    _ensure_code(dia_addr)
    c = w3.eth.contract(address=CS(dia_addr), abi=DIA_ORACLE_ABI)
    key_short = key if len(key) <= 24 else (key[:24] + "…")
    if block_number is not None:
        val, ts = call_at(
            c.functions.getValue(key),
            block_number=block_number,
            label=_lab("dia.getValue", dia_addr, key_short)
        )
    else:
        val, ts = c.functions.getValue(key).call()
    return int(val), int(ts)


def _read_binance_adapter(addr: str, *, block_number: int | None = None):
    _ensure_code(addr)
    c = w3.eth.contract(address=CS(addr), abi=BINANCE_FEED_ADAPTER_ABI)
    if block_number is not None:
        dec = call_at(
            c.functions.decimals(), block_number=block_number,
            label=_lab("binance.adapter.decimals", addr)
        )
        rd  = call_at(
            c.functions.latestRoundData(), block_number=block_number,
            label=_lab("binance.adapter.latestRoundData", addr)
        )
    else:
        dec = c.functions.decimals().call()
        rd  = c.functions.latestRoundData().call()
    answer, ts = int(rd[1]), int(rd[3])
    val = float(answer) / (10 ** int(dec))
    return val, ts


def _read_binance_registry_by_name(reg_addr: str, base: str, quote: str, *, block_number: int | None = None):
    _ensure_code(reg_addr)
    c = w3.eth.contract(address=CS(reg_addr), abi=BINANCE_FEED_REGISTRY_ABI)
    pair = f"{base}/{quote}"
    if block_number is not None:
        dec = call_at(
            c.functions.decimalsByName(base, quote),
            block_number=block_number,
            label=_lab("binance.registry.decimalsByName", reg_addr, pair)
        )
        rd  = call_at(
            c.functions.latestRoundDataByName(base, quote),
            block_number=block_number,
            label=_lab("binance.registry.latestRoundDataByName", reg_addr, pair)
        )
    else:
        dec = c.functions.decimalsByName(base, quote).call()
        rd  = c.functions.latestRoundDataByName(base, quote).call()
    answer, ts = int(rd[1]), int(rd[3])
    val = float(answer) / (10 ** int(dec))
    return val, ts


def fetch_oracles(
    oracles: list[OracleSpec],
    want_pair: str,
    *,
    block_number: int,
    stale_secs: int = 600,
    allow_usdt_peg: bool = True,
    usdt_peg_value: float = 1.0,
    usdt_peg_tol: float = 0.005,     # ±0.5% peg window
    min_provider_quorum: int = 2,    # need at least 2 providers; else fallback to pooled
    pooled_guardrail_tol: float = 0.20,   # if provider-median disagrees with pooled by >20%, prefer pooled
    base_alias_map: dict[str, str] | None = None  # e.g. {"WBETH":"ETH","BETH":"ETH"}; None = no alias
) -> tuple[list[OracleReading], Optional[float]]:
    """
    Returns (readings, chosen_median) with all prices normalized to want_pair (BASE/USDT).
    """
    tgt_base, tgt_quote = _pair_to_tuple(want_pair)
    assert tgt_quote.upper() == "USDT", "fetch_oracles is specialized for */USDT targets."

    base_accept: set[str] = {tgt_base.upper()}
    alias_used_for_log = None
    if base_alias_map:
        alias = base_alias_map.get(tgt_base.upper())
        if alias:
            base_accept.add(alias.upper())
            if alias.upper() != tgt_base.upper():
                alias_used_for_log = alias.upper()

    readings: list[OracleReading] = []
    provider_data: dict[str, dict[str, list[tuple[float, Optional[int]]]]] = {}

    def _provider_of(src_tag: str) -> str:
        root = src_tag.split(":", 1)[0].lower()
        if root in ("binance_registry", "binance"):
            return "binance"
        if root == "chainlink_ratio":
            return "chainlink"
        return root

    def _rec_prov(prov: str):
        provider_data.setdefault(prov, {
            "baseusd_norm": [], "usdtusd_norm": [],
            "baseusd_raw":  [], "usdtusd_raw":  [],
            "direct": []  # (val, ts, note)
        })

    def _stale_flag2(ts: Optional[int]) -> tuple[bool, int]:
        return _stale_flag(ts, stale_secs)

    def _note_reading(src: str, pair: str, val: Optional[float], ts: Optional[int], note: str):
        stale, uts = _stale_flag2(ts)
        readings.append(OracleReading(src, pair, (None if stale else val), uts, stale, note))
        if val is None:
            return
        base_sym, quote_sym = _pair_to_tuple(pair)
        prov = _provider_of(src); _rec_prov(prov)

        if quote_sym.upper() == "USD":
            if base_sym.upper() in base_accept:
                if "raw_scale" in note:
                    provider_data[prov]["baseusd_raw"].append((float(val), ts))
                else:
                    provider_data[prov]["baseusd_norm"].append((float(val), ts))
            if base_sym.upper() == "USDT":
                if "raw_scale" in note:
                    provider_data[prov]["usdtusd_raw"].append((float(val), ts))
                else:
                    provider_data[prov]["usdtusd_norm"].append((float(val), ts))

    def _maybe_candidate(val: Optional[float], ts: Optional[int], src: str, note: str):
        stale, uts = _stale_flag2(ts)
        readings.append(OracleReading(src, want_pair, (None if stale else val), uts, stale, note))
        prov = _provider_of(src); _rec_prov(prov)
        if (not stale) and (val is not None):
            provider_data[prov]["direct"].append((float(val), ts, note))

    # ---- Read/normalize all specs ----
    for spec in oracles:
        try:
            if spec.src == "chainlink":
                base, quote = _pair_to_tuple(spec.pair)
                ans, dec, ts = _read_chainlink_feed(spec.address, block_number=block_number)
                val = float(ans) / (10 ** dec)
                tag = f"chainlink:{spec.address[-6:]}"
                if quote.upper() == "USD" and (base.upper() in base_accept or base.upper() == "USDT"):
                    _note_reading(tag, f"{base}/USD", val, ts, "kept_for_ratio")
                if f"{base}/{quote}".upper() == want_pair.upper():
                    _maybe_candidate(val, ts, tag, "direct")

            elif spec.src == "chainlink_ratio":
                bSym = spec.extra["base_sym"]; qSym = spec.extra["quote_sym"]
                bAns,bDec,bTs = _read_chainlink_feed(spec.extra["base_addr"], block_number=block_number)
                qAns,qDec,qTs = _read_chainlink_feed(spec.extra["quote_addr"], block_number=block_number)
                bVal = float(bAns) / (10 ** bDec)
                qVal = float(qAns) / (10 ** qDec)

                if bSym.upper() in base_accept:
                    _note_reading("chainlink_ratio", f"{bSym}/USD", bVal, bTs, "kept_for_ratio")
                if qSym.upper() == "USDT":
                    _note_reading("chainlink_ratio", "USDT/USD", qVal, qTs, "kept_for_ratio")

                if f"{bSym}/{qSym}".upper() == want_pair.upper():
                    cand_val = bVal / max(qVal, 1e-18)
                    cand_ts  = min(bTs, qTs)
                    stale, uts = _stale_flag(cand_ts, stale_secs)
                    readings.append(OracleReading("chainlink_ratio", want_pair, (None if stale else cand_val), uts, stale, "base/quote via USD legs"))

            elif spec.src == "band":
                base, quote = _pair_to_tuple(spec.pair)
                rate, ts = _read_band_pair(spec.address, base, quote, block_number=block_number)
                tag = f"band:{spec.address[-6:]}"
                if quote.upper() == "USD" and (base.upper() in base_accept or base.upper() == "USDT"):
                    _note_reading(tag, f"{base}/USD", rate, ts, "kept_for_ratio")
                if f"{base}/{quote}".upper() == want_pair.upper():
                    _maybe_candidate(rate, ts, tag, "direct")

            elif spec.src == "pyth":
                base, quote = _pair_to_tuple(spec.pair)
                try:
                    val, ts = _read_pyth_price(spec.address, spec.extra["priceId"], block_number=block_number)
                except Exception as e:
                    _note_reading(f"pyth:{spec.address[-6:]}", f"{base}/{quote}", None, None, f"error:{e}")
                    continue
                tag = f"pyth:{spec.address[-6:]}"
                if quote.upper() == "USD" and (base.upper() in base_accept or base.upper() == "USDT"):
                    _note_reading(tag, f"{base}/USD", val, ts, "kept_for_ratio")
                if f"{base}/{quote}".upper() == want_pair.upper():
                    _maybe_candidate(val, ts, tag, "direct")

            elif spec.src == "dia":
                base, quote = _pair_to_tuple(spec.pair)
                raw, ts = _read_dia_value(spec.address, spec.extra["key"], block_number=block_number)
                v = float(raw)
                tag = f"dia:{spec.address[-6:]}"
                if quote.upper() == "USD" and (base.upper() in base_accept or base.upper() == "USDT"):
                    _note_reading(tag, f"{base}/USD", v, ts, "kept_for_ratio(raw_scale)")
                if f"{base}/{quote}".upper() == want_pair.upper():
                    stale, uts = _stale_flag2(ts)
                    readings.append(OracleReading(tag, want_pair, (None if stale else v), uts, stale, "direct_raw"))

            elif spec.src in ("binance", "binance_registry"):
                if spec.src == "binance":
                    base, quote = _pair_to_tuple(spec.pair)
                    val, ts = _read_binance_adapter(spec.address, block_number=block_number)
                    tag = f"binance:{spec.address[-6:]}"
                else:
                    base, quote = spec.extra["base_name"], spec.extra["quote_name"]
                    val, ts = _read_binance_registry_by_name(spec.address, base, quote, block_number=block_number)
                    tag = "binance_registry"
                if quote.upper() == "USD" and (base.upper() in base_accept or base.upper() == "USDT"):
                    _note_reading(tag, f"{base.upper()}/USD", val, ts, "kept_for_ratio")
                if f"{base.upper()}/{quote.upper()}" == want_pair.upper():
                    _maybe_candidate(val, ts, tag, "direct")

        except Exception as e:
            readings.append(OracleReading(spec.src, spec.pair, None, 0, True, f"error:{e}"))

    # ---- helpers ----
    def _fresh(vals_ts: list[tuple[float, Optional[int]]]) -> tuple[Optional[float], Optional[int]]:
        best_v, best_ts = None, None
        for v, ts in vals_ts:
            if ts is None or _is_stale(ts, stale_secs):
                continue
            if (best_ts is None) or (ts > best_ts):
                best_v, best_ts = float(v), int(ts)
        return best_v, best_ts

    def _median(xs: list[float]) -> Optional[float]:
        if not xs: return None
        ys = sorted(xs); n = len(ys)
        return ys[n//2] if (n % 2 == 1) else 0.5*(ys[n//2-1] + ys[n//2])

    def _mad_trim(xs: list[float], k: float = 5.0) -> list[float]:
        if len(xs) < 3: return xs[:]
        m = _median(xs)
        if m is None: return xs[:]
        dev = [abs(x - m) for x in xs]
        mad = _median(dev)
        if not mad: return xs[:]
        factor = 1.4826
        return [x for x in xs if abs(x - m) / (factor*mad) <= k]

    # ---- per-provider candidates ----
    provider_candidates: list[tuple[str,float,int]] = []

    for prov, data in provider_data.items():
        # coverage/debug snapshot for console later
        cov_note = (
            f"direct={len(data['direct'])} "
            f"baseusd_norm={len(data['baseusd_norm'])} usdtusd_norm={len(data['usdtusd_norm'])} "
            f"baseusd_raw={len(data['baseusd_raw'])} usdtusd_raw={len(data['usdtusd_raw'])}"
        )
        readings.append(OracleReading(f"coverage({prov})", want_pair, None, int(time.time()), False, cov_note))

        # 1) freshest direct BASE/USDT
        best_direct = None
        for v, ts, _note in data["direct"]:
            if ts is None or _is_stale(ts, stale_secs): 
                continue
            if (best_direct is None) or (ts > best_direct[1]):
                best_direct = (float(v), int(ts))
        if best_direct:
            v, ts = best_direct
            readings.append(OracleReading(f"provider_median({prov})", want_pair, v, ts, False, "direct"))
            provider_candidates.append((prov, v, ts))
            continue

        # 2) ratio from same provider legs: (BASE/USD) / (USDT/USD)
        base_v, base_ts = _fresh(data["baseusd_norm"])
        usdt_v, usdt_ts = _fresh(data["usdtusd_norm"])

        used_peg = False
        if allow_usdt_peg:
            if usdt_v is None:
                usdt_v, usdt_ts = float(usdt_peg_value), int(time.time())
                used_peg = True
            elif abs(usdt_v - float(usdt_peg_value)) <= float(usdt_peg_tol):
                usdt_v = float(usdt_peg_value)
                used_peg = True

        if (base_v is not None) and (usdt_v is not None):
            ts = min(base_ts or usdt_ts, usdt_ts or base_ts)
            val = base_v / max(usdt_v, 1e-18)
            stale, uts = _stale_flag2(ts)
            readings.append(OracleReading(
                f"provider_ratio({prov})", want_pair, (None if stale else val), uts, stale,
                f"legs_norm{' (peg)' if used_peg else ''}{' alias='+alias_used_for_log if alias_used_for_log else ''}"
            ))
            if not stale:
                provider_candidates.append((prov, float(val), uts))
            continue

        # 3) raw-scale DIA fallback per provider
        base_vr, base_tsr = _fresh(data["baseusd_raw"])
        usdt_vr, usdt_tsr = _fresh(data["usdtusd_raw"])
        if base_vr is not None and usdt_vr is not None:
            ts = min(base_tsr or usdt_tsr, usdt_tsr or base_tsr)
            val = base_vr / max(usdt_vr, 1e-18)
            stale, uts = _stale_flag2(ts)
            readings.append(OracleReading(
                f"provider_ratio({prov})", want_pair, (None if stale else val), uts, stale,
                f"legs_raw{' alias='+alias_used_for_log if alias_used_for_log else ''}"
            ))
            if not stale:
                provider_candidates.append((prov, float(val), uts))

    # ---- pooled via-USD (median(BASE/USD) / median(USDT/USD)) ----
    all_base_norm, all_usdt_norm = [], []
    for _prov, data in provider_data.items():
        for v, ts in data["baseusd_norm"]:
            if ts is not None and not _is_stale(ts, stale_secs):
                all_base_norm.append(float(v))
        for v, ts in data["usdtusd_norm"]:
            if ts is not None and not _is_stale(ts, stale_secs):
                all_usdt_norm.append(float(v))
    base_med = _median(_mad_trim(all_base_norm))
    usdt_med = _median(_mad_trim(all_usdt_norm))

    pooled = None
    if base_med is not None and (usdt_med is not None or allow_usdt_peg):
        if usdt_med is None or abs(usdt_med - float(usdt_peg_value)) <= float(usdt_peg_tol):
            usdt_med = float(usdt_peg_value)
            peg_tag = " (peg)"
        else:
            peg_tag = ""
        pooled = float(base_med) / max(float(usdt_med), 1e-18)
        now_ts = int(time.time())
        readings.append(OracleReading(
            "ratio_via_USD(pooled_median)", want_pair, pooled, now_ts, False,
            f"base_med={base_med} usdt_med={usdt_med}{peg_tag}{' alias='+alias_used_for_log if alias_used_for_log else ''}"
        ))

    # ---- finalize pick ----
    prov_to_val = {prov: v for (prov, v, _ts) in provider_candidates}
    vals_all = list(prov_to_val.values())
    vals_kept = _mad_trim(vals_all) if vals_all else []
    kept_set = {prov for prov, v in prov_to_val.items() if v in vals_kept}
    trimmed_set = set(prov_to_val.keys()) - kept_set

    now_ts = int(time.time())
    for prov in sorted(trimmed_set):
        readings.append(OracleReading(f"provider_outlier({prov})", want_pair, prov_to_val[prov], now_ts, False, "trimmed_by_MAD"))
    readings.append(OracleReading("provider_kept", want_pair, None, now_ts, False, ",".join(sorted(kept_set))))
    if trimmed_set:
        readings.append(OracleReading("provider_trimmed", want_pair, None, now_ts, False, ",".join(sorted(trimmed_set))))

    provider_median = None
    if kept_set:
        kept_vals = [prov_to_val[p] for p in kept_set]
        kept_vals.sort()
        n = len(kept_vals)
        provider_median = kept_vals[n//2] if (n % 2 == 1) else 0.5*(kept_vals[n//2-1] + kept_vals[n//2])

    choice = None
    choice_src = ""
    if (provider_median is None) or (len(kept_set) < int(min_provider_quorum)):
        choice = pooled
        choice_src = "pooled_via_USD(fallback)"
    else:
        choice = provider_median
        choice_src = "provider_median"
        if (pooled is not None) and (abs(choice - pooled) / max(pooled, 1e-18) > float(pooled_guardrail_tol)):
            choice = pooled
            choice_src = "pooled_via_USD(guardrail)"

    if choice is not None:
        readings.append(OracleReading(f"final({choice_src})", want_pair, choice, int(time.time()), False, "selected"))

    return readings, choice

def _fetch_and_persist_oracles(oracle_specs, sym: str, blkno: int, blkt: int, args, db):
    """Always prints a line and writes JSONL, even on failure or when no specs provided."""
    want_pair = f"{sym}/USDT"

    def _prov_of(tag: str) -> str:
        root = tag.split(":", 1)[0].lower()
        return "binance" if root in ("binance", "binance_registry") else ("chainlink" if root == "chainlink_ratio" else root)

    # Default values if oracles are disabled or failed
    readings, median, status_note = [], None, ""
    if not oracle_specs:
        status_note = "(none configured)"
        _print_now(f"[oracles] {want_pair} {status_note}")
    else:
        try:
            readings, median = fetch_oracles(
                oracle_specs, want_pair, block_number=blkno, stale_secs=args.oracle_stale_secs
            )

            # --- Build a detailed summary from readings ---
            fresh = [r for r in readings if (not r.stale) and (r.price is not None)]
            # Which path was chosen?
            final = next((r for r in readings if r.src.startswith("final(")), None)
            via = final.src[6:-1] if final else "n/a"

            # Provider candidates (kept) and outliers (trimmed)
            candidates = [r for r in readings
                          if r.pair == want_pair and not r.stale and r.price is not None
                          and (r.src.startswith("provider_ratio(") or r.src.startswith("provider_median("))]
            trimmed = [r for r in readings if r.src.startswith("provider_outlier(")]
            trimmed_names = [r.src.split("(",1)[1].split(")",1)[0] for r in trimmed]
            kept_note = next((r.raw_note for r in readings if r.src == "provider_kept"), "")
            kept_names = [x for x in kept_note.split(",") if x] if kept_note else sorted({_prov_of(c.src.split("(",1)[1].split(")",1)[0]) for c in candidates})

            # Pooled info (if any)
            pooled = next((r for r in readings if r.src == "ratio_via_USD(pooled_median)"), None)
            pooled_note = pooled.raw_note if pooled else ""
            # parse base_med/usdt_med quickly if present
            base_med, usdt_med = None, None
            if pooled_note:
                try:
                    # example: "base_med=2345.6 usdt_med=1.0 (peg) alias=ETH"
                    parts = pooled_note.replace("(", " ").replace(")", " ").split()
                    bm = next((p for p in parts if p.startswith("base_med=")), None)
                    um = next((p for p in parts if p.startswith("usdt_med=")), None)
                    if bm: base_med = float(bm.split("=",1)[1])
                    if um:
                        v = um.split("=",1)[1]
                        usdt_med = float(v)
                except Exception:
                    pass

            # Leg coverage (how many BASE/USD and USDT/USD legs per provider)
            leg_counts = {}
            for r in readings:
                if "kept_for_ratio" not in r.raw_note:
                    continue
                prov = _prov_of(r.src)
                leg_counts.setdefault(prov, {"baseusd": 0, "usdtusd": 0})
                if r.pair.upper().endswith("/USD"):
                    base = r.pair.split("/")[0].upper()
                    if base == "USDT":
                        leg_counts[prov]["usdtusd"] += 1
                    else:
                        leg_counts[prov]["baseusd"] += 1

            # Build source strings like: chainlink=ratio(peg):4596.90
            def _src_label(r):
                if r.src.startswith("provider_ratio("):
                    prov = r.src.split("(",1)[1].split(")",1)[0]
                    peg = "peg" if "peg" in r.raw_note.lower() else ""
                    return f"{prov}=ratio{('(peg)' if peg else '')}:{r.price:.6f}"
                else:
                    prov = r.src.split("(",1)[1].split(")",1)[0]
                    return f"{prov}=direct:{r.price:.6f}"

            src_bits = "  ".join(_src_label(r) for r in sorted(candidates, key=lambda x: x.src))

            # First line: headline
            med_note = f"{median:.6f}" if median is not None else "n/a"
            _print_now(
                f"[oracles] {want_pair} median ≈ {med_note}  |  via={via}  providers={len(candidates)} "
                f"kept={len(kept_names)} trimmed={len(trimmed_names)}"
            )

            # Second line: per-provider + pooled info
            pooled_bit = ""
            if pooled:
                pooled_bit = f" | pooled={pooled.price:.6f}"
                if base_med is not None and usdt_med is not None:
                    pooled_bit += f" (base_med={base_med:.6f} usdt_med={usdt_med:.6f})"
                elif pooled.raw_note:
                    pooled_bit += f" ({pooled.raw_note})"

            _print_now(f"[oracles] sources: {src_bits}{pooled_bit}")

            # Third line: coverage and legs
            cov_parts = []
            for prov in sorted(leg_counts.keys()):
                bc = leg_counts[prov]["baseusd"]
                uc = leg_counts[prov]["usdtusd"]
                cov_parts.append(f"{prov}:BASE/USD={bc},USDT/USD={uc}")
            if cov_parts:
                _print_now(f"[oracles] legs: " + "  |  ".join(cov_parts))

            # Fourth line: trimmed (if any)
            if trimmed_names:
                _print_now(f"[oracles] trimmed: {', '.join(sorted(trimmed_names))}")

            # If nothing fresh at all
            if not fresh:
                _print_now(f"[oracles] {want_pair} no fresh normalized readings")

        except Exception as e:
            status_note = f"failed: {e}"
            _print_now(f"[oracles] {want_pair} {status_note}")

    # Persist to DB + JSONL even if empty
    if db:
        try:
            db.insert_oracles(blkno, readings)
        except Exception:
            pass
    try:
        log_jsonl(args.out_oracles_jsonl, {
            "block_number": blkno,
            "block_timestamp": blkt,
            "pair": want_pair,
            "median": median,
            "status": status_note,
            "readings": [
                {"src": r.src, "pair": r.pair, "price": r.price,
                 "updated_at": r.updated_at, "stale": r.stale, "note": r.raw_note}
                for r in readings
            ]
        })
    except Exception:
        pass

    return median

# (2) WORKING-POOL CACHE -------------------------------------------------------
class PoolCache:
    def __init__(self, path: str):
        self.path = path
        self.data = {"v3": {"pancake": [], "uniswap": []}, "thena": {"single": [], "via_wbnb": []}}
        try:
            if path and pathlib.Path(path).exists():
                self.data = json.loads(pathlib.Path(path).read_text())
        except Exception:
            pass

    def _list(self, venue: str) -> list:
        v, name = ("v3", venue) if venue in ("pancake","uniswap") else ("thena", venue)
        return self.data[v][name]

    def add_v3(self, venue_short: str, token_in: str, token_out: str, fee: int):
        lst = self._list(venue_short)
        rec = [token_in.lower(), token_out.lower(), int(fee)]
        if rec not in lst: lst.append(rec)

    def has_v3(self, venue_short: str, token_in: str, token_out: str, fee: int) -> bool:
        return [token_in.lower(), token_out.lower(), int(fee)] in self._list(venue_short)

    def add_thena_single(self, a: str, b: str):
        lst = self._list("single")
        rec = [a.lower(), b.lower()]
        if rec not in lst and rec[::-1] not in lst: lst.append(rec)

    def add_thena_via_wbnb(self, a: str, b: str):
        lst = self._list("via_wbnb")
        rec = [a.lower(), b.lower()]
        if rec not in lst and rec[::-1] not in lst: lst.append(rec)

    def save(self):
        if not self.path: return
        pathlib.Path(self.path).write_text(json.dumps(self.data, indent=2))

# (3) SQLITE SINK --------------------------------------------------------------
class BenchDB:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path)
        self._init()

    def _init(self):
        c = self.conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS blocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            number INTEGER UNIQUE,
            ts INTEGER
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS oracles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            block_number INTEGER,
            source TEXT,
            pair TEXT,
            price REAL,
            updated_at INTEGER,
            stale INTEGER,
            note TEXT
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS quotes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            block_number INTEGER,
            venue TEXT,
            path TEXT,
            amount_in TEXT,
            out_token TEXT,
            amount_out TEXT,
            usdt_equiv REAL,
            gas_estimate INTEGER,
            ticks_crossed INTEGER,
            latency_ms REAL
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS best_by_block (
            block_number INTEGER PRIMARY KEY,
            best_venue TEXT,
            best_path TEXT,
            best_token TEXT,
            best_amount_out TEXT,
            best_usdt_equiv REAL,
            oracle_median REAL
        )""")
        self.conn.commit()

    def upsert_block(self, number: int, ts: int):
        self.conn.execute("INSERT OR IGNORE INTO blocks(number,ts) VALUES(?,?)", (number, ts))
        self.conn.commit()

    def insert_oracles(self, block_number: int, readings: list[OracleReading]):
        cur = self.conn.cursor()
        cur.executemany(
            "INSERT INTO oracles(block_number,source,pair,price,updated_at,stale,note) VALUES(?,?,?,?,?,?,?)",
            [(block_number, r.src, r.pair, r.price, r.updated_at, 1 if r.stale else 0, r.raw_note) for r in readings]
        )
        self.conn.commit()

    def insert_quotes(self, block_number: int, q: 'Quote', usdt_equiv: float | None):
        self.conn.execute(
            "INSERT INTO quotes(block_number,venue,path,amount_in,out_token,amount_out,usdt_equiv,gas_estimate,ticks_crossed,latency_ms) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (block_number, q.venue, q.path, str(q.amount_in), symbol_of(q.out_token), str(q.amount_out),
             usdt_equiv, q.gas_estimate if q.gas_estimate is not None else None, q.ticks_crossed, q.ms)
        )
        self.conn.commit()

    def upsert_best(self, block_number: int, q: 'Quote', usdt_equiv: float, oracle_median: float | None):
        self.conn.execute(
            "INSERT INTO best_by_block(block_number,best_venue,best_path,best_token,best_amount_out,best_usdt_equiv,oracle_median) "
            "VALUES(?,?,?,?,?,?,?) ON CONFLICT(block_number) DO UPDATE SET "
            "best_venue=excluded.best_venue,best_path=excluded.best_path,best_token=excluded.best_token,"
            "best_amount_out=excluded.best_amount_out,best_usdt_equiv=excluded.best_usdt_equiv,oracle_median=excluded.oracle_median",
            (block_number, q.venue, q.path, symbol_of(q.out_token), str(q.amount_out), usdt_equiv, oracle_median)
        )
        self.conn.commit()

# --- Utils / shared ---
ZERO = "0x0000000000000000000000000000000000000000"

def _to_cs(w3, a: str) -> str:
    try:
        return w3.to_checksum_address(a)
    except Exception:
        return a

def _decode_revert_reason(data: bytes) -> str:
    # std Error(string) selector 0x08c379a0
    if not data or len(data) < 4:
        return ""
    try:
        sel = data[:4].hex()
        if sel == "08c379a0":  # Error(string)
            # strip selector, decode ABI-encoded string
            import eth_abi
            return str(eth_abi.decode(["string"], data[4:])[0])
    except Exception:
        pass
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""
def _normalize_pool_inputs(w3, pools_in: Any, verbose: bool=False) -> List[Tuple[str, Optional[str]]]:
    """
    Accepts:
      - "0xaddr"
      - ("0xaddr", "DVM"/"DPP"/"DSP"/<kind>)
      - {"address": "0xaddr", "kind": "DVM"}
      - nested lists/sets/tuples
    Returns a de-duped list of (address, kind_or_None)
    """
    out: List[Tuple[str, Optional[str]]] = []
    seen: Set[str] = set()

    def walk(x: Any):
        if x is None:
            return
        if isinstance(x, str):
            a = _to_cs(w3, x)
            if a.lower() not in seen:
                seen.add(a.lower()); out.append((a, None))
            return
        if isinstance(x, (list, tuple, set)):
            if len(x) == 2 and isinstance(x[0], str) and isinstance(x[1], str) and x[0].startswith("0x"):
                a = _to_cs(w3, x[0]); kind = x[1]
                if a.lower() not in seen:
                    seen.add(a.lower()); out.append((a, kind))
                return
            for y in x:
                walk(y)
            return
        if isinstance(x, dict):
            a = x.get("address") or x.get("pool") or x.get("addr")
            k = x.get("kind") or x.get("type")
            if isinstance(a, str) and a.startswith("0x"):
                a = _to_cs(w3, a)
                if a.lower() not in seen:
                    seen.add(a.lower()); out.append((a, k if isinstance(k, str) else None))
            return
        # ignore anything else

    walk(pools_in)
    if verbose:
        print(f"[DODO] normalized {len(out)} pool(s)")
    return out
    
# ---------------------------
# RPC / chain setup (hardened with multi-endpoint rotation)
# ---------------------------
DEFAULT_BSC_RPC = os.environ.get(
    "BSC_RPC",
    # comma-separated fallbacks from BNB Chain guidance
    "https://bsc-dataseed.binance.org,"
    "https://bsc-dataseed1.defibit.io,"
    "https://bsc-dataseed1.ninicoin.io"
)

# Global pool & state
_RPC_URLS: list[str] = []
_RPC_I: int = 0
ACTIVE_RPC_URL: str = ""  # for diagnostics

_RPC_HEALTH = {}  # url -> {"failures": int, "cooldown_until": float}
_RPC_COOLDOWN_BASE = 8.0
_RPC_COOLDOWN_MAX  = 60.0
RPC_DEBUG = str(os.environ.get("RPC_DEBUG","0")).lower() in ("1","true","yes")

RPC_DEBUG = str(os.environ.get("RPC_DEBUG", "0")).lower() in ("1", "true", "yes")

def _mk_single_web3(rpc_url: str) -> Web3:
    w3_ = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={'timeout': 30}))
    # PoA middleware (BSC) + disable ENS entirely
    try:
        from web3.middleware import ExtraDataToPOAMiddleware  # web3.py v6+
        w3_.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    except Exception:
        try:
            from web3.middleware import geth_poa_middleware  # web3.py v5 fallback
            w3_.middleware_onion.inject(geth_poa_middleware, layer=0)
        except Exception:
            pass
    w3_.ens = None
    return w3_

def _update_active_w3(rpc_url: str):
    global w3, ACTIVE_RPC_URL
    w3 = _mk_single_web3(rpc_url)
    ACTIVE_RPC_URL = rpc_url

def mk_web3(rpc_urls: str | list[str]) -> Web3:
    """
    Accepts a single URL or a comma-separated list. Initializes the pool and returns the first w3.
    """
    global _RPC_URLS, _RPC_I
    if isinstance(rpc_urls, str):
        # allow comma-separated
        _RPC_URLS = [u.strip() for u in rpc_urls.split(",") if u.strip()]
    else:
        _RPC_URLS = [u.strip() for u in rpc_urls if u.strip()]

    if not _RPC_URLS:
        raise ValueError("No RPC URLs configured")

    _RPC_I = 0
    _update_active_w3(_RPC_URLS[_RPC_I])
    return w3



def _init_rpc_health(urls):
    now = time.monotonic()
    for u in urls:
        _RPC_HEALTH.setdefault(u, {"failures": 0, "cooldown_until": now})

def _mark_rpc_failure(url: str):
    h = _RPC_HEALTH.setdefault(url, {"failures": 0, "cooldown_until": 0.0})
    h["failures"] += 1
    backoff = min(_RPC_COOLDOWN_MAX, _RPC_COOLDOWN_BASE * (2 ** max(0, h["failures"] - 1)))
    h["cooldown_until"] = time.monotonic() + backoff
    if RPC_DEBUG:
        print(f"[rpc-health] {url} fail#{h['failures']} cooldown {backoff:.1f}s")

def _mark_rpc_success(url: str):
    h = _RPC_HEALTH.setdefault(url, {"failures": 0, "cooldown_until": 0.0})
    if h["failures"] > 0:
        h["failures"] -= 1
    h["cooldown_until"] = time.monotonic()

def _best_rpc(current: Optional[str] = None) -> str:
    now = time.monotonic()
    candidates = []
    for u in _RPC_URLS:
        h = _RPC_HEALTH.get(u, {"failures": 0, "cooldown_until": 0.0})
        cooled = h["cooldown_until"] <= now
        candidates.append((cooled, h["failures"], h["cooldown_until"], u))
    candidates.sort(key=lambda x: (not x[0], x[1], x[2]))
    best = candidates[0][3]
    if current and best == current and len(candidates) > 1:
        best = candidates[1][3]
    return best

def _rotate_rpc(reason: str = ""):
    global ACTIVE_RPC_URL
    prev = ACTIVE_RPC_URL
    nxt = _best_rpc(prev)
    if nxt == prev:
        return
    _update_active_w3(nxt)
    if RPC_DEBUG or reason:
        print(f"[rpc] switch {prev} → {nxt} ({reason or 'retry'})")

def _is_transient_jsonrpc_error(e: Exception) -> bool:
    s = str(e).lower()
    return any(x in s for x in (
        "timeout", "timed out", "connection reset", "econnreset",
        "502 ", "503 ", "429 ", "max retries exceeded",
        "temporarily unavailable", "network", "ssl",
        # pruned / historical gaps should also be retried on another endpoint:
        "missing trie node", "state unavailable", "unknown block",
        "header not found", "pruned", "ancient block", "not found for block",
    ))
RPC_DEBUG = str(os.environ.get("RPC_DEBUG", "0")).lower() in ("1", "true", "yes")

def _is_abi_decode_error(e: Exception) -> bool:
    s = (str(e) or "").lower()
    if ("could not decode contract function call" in s) or ("invalid return data" in s):
        return True
    try:
        # eth_abi DecodingError class (avoid import at top level if not present)
        from eth_abi.exceptions import DecodingError
        if isinstance(e, DecodingError):
            return True
    except Exception:
        pass
    return False


# --- helpers  ---
def _is_badfunctioncalloutput(e: Exception) -> bool:
    # web3.exceptions.BadFunctionCallOutput subclasses ValueError in some versions,
    # but the message text is stable enough:
    s = (str(e) or "").lower()
    return ("could not transact with/call contract function" in s) or \
           ("badfunctioncalloutput" in s)

def _label_for_fn(fn, base: str) -> str:
    # Try to enrich logs with the target address suffix
    try:
        # web3py v5: fn.address; v6: fn._parent.address
        addr = getattr(fn, "address", None) or getattr(getattr(fn, "_parent", None), "address", None)
        if addr:
            return f"{base}@{str(addr)[-6:]}"
    except Exception:
        pass
    return base

def with_retries(fn, *, label: str = "", attempts: int = 5,
                 base_delay: float = 0.35, max_delay: float = 3.0, jitter: float = 0.20):
    """
    Call fn() with backoff on transient net/JSON-RPC/pruned-state errors.
    - ABI decode errors are HARD (no retry): wrong ABI / wrong contract type.
    - Pruned/missing-trie/unknown-block are TRANSIENT (rotate + backoff).
    """
    try:
        from web3.exceptions import ContractLogicError
    except Exception:
        class ContractLogicError(Exception): pass

    last_exc = None
    for i in range(attempts):
        try:
            res = fn()
            try: _mark_rpc_success(ACTIVE_RPC_URL)
            except Exception: pass
            return res
        except ContractLogicError:
            raise
        except Exception as e:
            last_exc = e
            # classify
            hard = _is_abi_decode_error(e)
            transient = _is_transient_jsonrpc_error(e) or _is_pruned_or_blockgap(e)
            if hard and not transient:
                if RPC_DEBUG or label:
                    print(f"[retry] {label}: hard ABI/decode error, aborting: {e}")
                raise
            if not transient:
                if RPC_DEBUG or label:
                    print(f"[retry] {label}: hard error, aborting: {e}")
                raise
            # transient path
            try:
                _mark_rpc_failure(ACTIVE_RPC_URL)
                _rotate_rpc(f"{label or 'call'}: transient rpc")
            except Exception:
                pass
            delay = min(max_delay, base_delay * (2 ** i))
            delay *= (1.0 + jitter * (random.random() - 0.5))
            if RPC_DEBUG or label:
                print(f"[retry] {label}: attempt {i+1}/{attempts} sleeping {delay:.2f}s")
            time.sleep(max(0.05, delay))

    if last_exc:
        raise last_exc


# Initialize on import
w3 = mk_web3(DEFAULT_BSC_RPC)
CS = Web3.to_checksum_address

# classify transient network errors that merit rotation
# ---- Robust network exception set + retry helper (add once) ----
import time, socket

# requests / urllib3 exceptions (canonical names)
try:
    from requests.exceptions import (
        ConnectionError as RequestsConnectionError,
        Timeout as RequestsTimeout,
        ReadTimeout,
        SSLError,
        ChunkedEncodingError,
        RequestException,
    )
except Exception:
    # If requests isn't present in your env, define dummies
    class _Dummy(Exception): pass
    RequestsConnectionError = _Dummy
    RequestsTimeout = _Dummy
    ReadTimeout = _Dummy
    SSLError = _Dummy
    ChunkedEncodingError = _Dummy
    RequestException = _Dummy

try:
    from urllib3.exceptions import (
        NameResolutionError,
        MaxRetryError,
        ProtocolError,
        NewConnectionError,
    )
except Exception:
    # Fallback dummies
    class _UDummy(Exception): pass
    NameResolutionError = _UDummy
    MaxRetryError = _UDummy
    ProtocolError = _UDummy
    NewConnectionError = _UDummy

# A tuple of network-ish transient errors worth retrying
NETWORK_ERRORS = (
    RequestsConnectionError,
    RequestsTimeout,
    ReadTimeout,
    SSLError,
    ChunkedEncodingError,
    RequestException,
    NameResolutionError,
    MaxRetryError,
    ProtocolError,
    NewConnectionError,
    socket.gaierror,
    socket.timeout,
    TimeoutError,
)

TRANSIENT_RPC_SUBSTRINGS = tuple(s.lower() for s in [
    "missing trie node",          # Erigon state pruning
    "header not found",
    "state not available",
    "pruned",                     # "pruned state"
    "ancient block sync",
    "unknown block",
    "timeout",
    "context deadline exceeded",
    "try again later",
    "oversized request",
    "backend unhealthy",
])

def _is_transient_jsonrpc_error(e: Exception) -> bool:
    s = str(e).lower()
    return any(x in s for x in (
        "timeout", "timed out", "connection reset", "econnreset",
        "502 bad gateway", "503 service unavailable", "429 too many requests",
        "max retries exceeded", "temporarily unavailable", "ssl",
        "gateway timeout", "bad gateway",
    ))

# ---- DROP-IN: hardened with_retries (quiet by default) ----------------------

def eth_get_code_safe(addr: str) -> bytes:
    """Resilient wrapper for w3.eth.get_code."""
    return with_retries(lambda: w3.eth.get_code(addr), label=f"eth_getCode {addr[-6:]}")

# ---- Block snapshot + call wrappers ----
def ensure_rpc_ready(expect_chain_id: int | None = 56, *, tries: int = 2):
    """
    Probe current RPC; rotate until one responds sanely to chainId + blockNumber.
    """
    errors = []
    for _ in range(len(_RPC_URLS) * max(1, tries)):
        try:
            cid = with_retries(lambda: w3.eth.chain_id, label="eth_chainId", attempts=2)
            if expect_chain_id is not None and cid != expect_chain_id:
                raise RuntimeError(f"unexpected chainId {cid} from {ACTIVE_RPC_URL}")
            with_retries(lambda: w3.eth.block_number, label="eth_blockNumber", attempts=2)
            return
        except Exception as e:
            errors.append(str(e))
            try: _rotate_rpc("rpc warmup")
            except Exception: pass
            time.sleep(0.2)
    raise SystemExit(f"No working RPC after probing pool. Last errors: {errors[-3:]}")


def get_block_snapshot(*, attempts: int = 6, max_skew_secs: int = 60) -> dict:
    """
    Robustly capture a consistent block context:
      - try eth_getBlockByNumber('latest')
      - on failure, try eth_blockNumber then get that block
      - rotates RPCs on transient errors
      - sanity-checks timestamp skew
    """
    def _latest():
        blk = w3.eth.get_block('latest')
        return int(blk.number), int(blk.timestamp)

    def _by_number():
        n = with_retries(lambda: w3.eth.block_number, label="eth_blockNumber", attempts=3)
        blk = with_retries(lambda: w3.eth.get_block(n), label=f"eth_getBlockByNumber({n})", attempts=3)
        return int(blk.number), int(blk.timestamp)

    last_err = None
    for i in range(attempts):
        try:
            # First try 'latest'
            number, ts = with_retries(_latest, label="eth_getBlockByNumber(latest)", attempts=3)
        except Exception as e:
            last_err = e
            # Fallback path
            try:
                number, ts = _by_number()
            except Exception as e2:
                last_err = e2
                # rotate and backoff before next loop
                try: _rotate_rpc("snapshot failure")
                except Exception: pass
                time.sleep(min(1.5, 0.2 * (2 ** i)))
                continue

        # timestamp sanity (bad node clock)
        now = int(time.time())
        if ts > now + max_skew_secs:
            try: _rotate_rpc("clock skew")
            except Exception: pass
            time.sleep(0.25)
            continue

        return {"number": number, "timestamp": ts}

    # out of attempts
    raise SystemExit(f"Could not obtain a block snapshot. Last error: {last_err}")


# ---- pinned call wrapper with safe fallback ------------------------
def call_at(fn, block_number=None, label="eth_call"):
    """
    Try pinned first; on pruned/missing/generic BadFunctionCallOutput, rotate and retry unpinned.
    """
    nice_label = _label_for_fn(fn, label)

    if block_number is None:
        return with_retries(lambda: fn.call(), label=nice_label)

    try:
        return with_retries(lambda: fn.call(block_identifier=block_number), label=f"{nice_label}[pinned]")
    except Exception as e:
        # Treat pruned/missing + the generic badfunctioncalloutput as candidates for unpinned fallback
        if _is_pruned_or_blockgap(e) or _is_transient_jsonrpc_error(e) or _is_badfunctioncalloutput(e):
            try: _rotate_rpc(f"{nice_label}: pinned->unpinned fallback")
            except Exception: pass
            return with_retries(lambda: fn.call(), label=f"{nice_label}[unpinned]")
        raise

# ---------------------------
# Addresses (checksummed)
# ---------------------------
WBNB = CS("0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c")
ETHb = CS("0x2170Ed0880ac9A755fd29B2688956BD959F933F8")
USDT = CS("0x55d398326f99059fF775485246999027B3197955")
USDC = CS("0x8ac76a51cc950d9822d68b83fe1ad97b32cd580d")

# ETH-like variants (optional; extend via CLI)
BETH   = CS("0x250632378E573c6Be1AC2f97FCdf00515d0Aa91B")
WBETH  = CS("0xa2e3356610840701bdf5611a53974510ae27e2e1")
WSTETH = CS("0x26c5e01524d2E6280A48F2c50fF6De7e52E9611C")

PANCAKE_V2_ROUTER  = CS("0x10ED43C718714eb63d5aA57B78B54704E256024E")
PANCAKE_V2_FACTORY = CS("0xCa143Ce32Fe78f1f7019d7d551a6402fC5350c73")

PANCAKE_V3_FACTORY   = CS("0x0BFbCF9fa4f9C56B0F40a671Ad40E0805A091865")
PANCAKE_V3_QUOTER_V2 = CS("0xB048Bbc1Ee6b733FFfCFb9e9CeF7375518e25997")

UNISWAP_V3_FACTORY   = CS("0xdB1d10011AD0Ff90774D0C6Bb92e5C5c8b4461F7")
UNISWAP_V3_QUOTER_V2 = CS("0x78D78E420Da98ad378D7799bE8f4AF69033EB077")

THENA_QUOTER = CS("0xeA68020D6A9532EeC42D4dB0f92B83580c39b2cA")
# THENA / Algebra v1.0 (from official docs)
THENA_FACTORY = CS("0x306F06C147f064A010530292A1EB6737c3e378e4")  # AlgebraFactory (BNB)

WOMBAT_ROUTER    = CS("0x19609B03C976CCA288fbDae5c21d4290e9a4aDD7")
WOMBAT_MAIN_POOL = CS("0x312Bc7eAAF93f1C60Dc5AfC115FcCDE161055fb0")

# DODO v2 (BSC, from docs)
DODO_V2_ROUTE_HELPER = CS("0xb48eE7B874Af8bC0e068036e55e33b5DC91C3a65")
DODO_ZOO       = CS("0xCA459456a45e300AA7EF447DBB60F87CCcb42828")  # v1 registry (unused here)
DODO_DVM_FACTORY = CS("0x790B4A80Fb1094589A3c0eFC8740aA9b0C1733fB")
DODO_DPP_FACTORY = CS("0xd9CAc3D964327e47399aebd8e1e6dCC4c251DaAE")
DODO_DSP_FACTORY = CS("0x0fb9815938Ad069Bf90E14FE6C596c514BEDe767")
DODO_UPCP_FACTORY= CS("0x4F57F6929E58AE564F6AB090fE0AEEcb39B0f270")  # unified CP (rare on BSC)

MULTICALL3 = CS("0xcA11bde05977b3631167028862Be2a173976CA11")
QUOTE_USER = CS("0x1111111111111111111111111111111111111111")

# === KyberSwap Elastic (universal EVM addresses; valid on BNB Chain) ===
KYBER_ELASTIC_FACTORY   = CS("0x5F1ddd3c321B1A8aD8B3cd408e933B56A472290A")
KYBER_ELASTIC_QUOTER_V2 = CS("0x0D125c15E98d7b7564da0e2b5fB09F8E8E88e3e9")
# Optional (not used for quoting here):
KYBER_ELASTIC_ROUTER    = CS("0xC1e7dF4070B4A9F7Cd8E21BBaCE7aDEaA120F1BB")

# --- Biswap (v2-style) ---
BISWAP_V2_FACTORY = CS("0x858E3312ed3A876947EA49d572A7C42DE08af7EE")
BISWAP_V2_ROUTER  = CS("0x3a6d8cA21D1CF76F653A67577FA0D27453350dD8")
# Biswap v2 (Uniswap v2-compatible router)
BISWAP_V2_ROUTER = CS("0x8bD8432880c0C603e9489BC913d376231185016a")
ISWAP_V2_ROUTER = Web3.to_checksum_address("0x3a6d8cA21D1CF76F653A67577FA0D27453350dD8")
# ---------------------------
# Tuning
# ---------------------------
TARGET_TOTAL_USD = 1000_000
SLICE_USD        = 1_000

UNI_V3_FEES = [100, 500, 3000] #, 10000]
PCS_V3_FEES = [100, 500, 2500]#, 10000]

# Kyber Elastic fee IDs (Uniswap v3 units = hundredths of a bip)
# Use Kyber fees in **bps**, not Uniswap units
KYBER_FEES_BPS = [1, 4, 10, 25, 30, 60, 100]  # tight set: 0.01%, 0.04%, 0.10%, 0.25%, 0.30%, 0.60%, 1.00%

INTERMEDIATES = [WBNB, USDC]

GAS_PCS_V2_SWAP = 150_000
GAS_WOMBAT_SWAP = 200_000
GAS_DODO_QUERY  = 120_000

# ---------------------------
# Minimal ABIs
# ---------------------------
ERC20_ABI = [
    {"constant": True,"inputs": [],"name": "decimals","outputs": [{"name":"","type":"uint8"}],"type":"function"},
    {"constant": True,"inputs": [],"name": "symbol","outputs": [{"name":"","type":"string"}],"type":"function"},
]
ERC20_SYMBOL_BYTES32_ABI = [{"constant": True,"inputs": [],"name": "symbol","outputs": [{"name":"","type":"bytes32"}],"type":"function"}]

V2_ROUTER_ABI = [
  {"name":"getAmountsOut","outputs":[{"name":"amounts","type":"uint256[]"}],
   "inputs":[{"name":"amountIn","type":"uint256"},{"name":"path","type":"address[]"}],
   "stateMutability":"view","type":"function"},
  {"name":"getAmountsIn","outputs":[{"name":"amounts","type":"uint256[]"}],
   "inputs":[{"name":"amountOut","type":"uint256"},{"name":"path","type":"address[]"}],
   "stateMutability":"view","type":"function"}
]
V2_FACTORY_ABI = [{
 "name":"getPair","type":"function","stateMutability":"view",
 "inputs":[{"name":"tokenA","type":"address"},{"name":"tokenB","type":"address"}],
 "outputs":[{"name":"pair","type":"address"}]
}]

V3_QUOTER_V2_ABI = [
  { "name": "quoteExactInputSingle","type": "function","stateMutability":"nonpayable",
    "inputs": [{ "components": [
        {"name":"tokenIn","type":"address"},
        {"name":"tokenOut","type":"address"},
        {"name":"fee","type":"uint24"},
        {"name":"amountIn","type":"uint256"},
        {"name":"sqrtPriceLimitX96","type":"uint160"}], "name":"params","type":"tuple"}],
    "outputs":[
      {"name":"amountOut","type":"uint256"},
      {"name":"sqrtPriceX96After","type":"uint160"},
      {"name":"initializedTicksCrossed","type":"uint32"},
      {"name":"gasEstimate","type":"uint256"}] },
  { "name": "quoteExactInput","type": "function","stateMutability":"nonpayable",
    "inputs": [{"name":"path","type":"bytes"},{"name":"amountIn","type":"uint256"}],
    "outputs":[
      {"name":"amountOut","type":"uint256"},
      {"name":"sqrtPriceX96AfterList","type":"uint160[]"},
      {"name":"initializedTicksCrossedList","type":"uint32[]"},
      {"name":"gasEstimate","type":"uint256"}] }
]
V3_FACTORY_ABI = [{
 "name":"getPool","type":"function","stateMutability":"view",
 "inputs":[{"name":"tokenA","type":"address"},{"name":"tokenB","type":"address"},{"name":"fee","type":"uint24"}],
 "outputs":[{"name":"pool","type":"address"}]
}]
V3_POOL_SLOT0_ABI = [{
 "name":"slot0","type":"function","stateMutability":"view",
 "inputs":[],
 "outputs":[
   {"name":"sqrtPriceX96","type":"uint160"},
   {"name":"tick","type":"int24"},
   {"name":"observationIndex","type":"uint16"},
   {"name":"observationCardinality","type":"uint16"},
   {"name":"observationCardinalityNext","type":"uint16"},
   {"name":"feeProtocol","type":"uint8"},
   {"name":"unlocked","type":"bool"}]}]

WOMBAT_ROUTER_ABI = [{
 "name":"getAmountOut","type":"function","stateMutability":"view",
 "inputs":[{"name":"tokenPath","type":"address[]"},{"name":"poolPath","type":"address[]"},{"name":"amountIn","type":"int256"}],
 "outputs":[{"name":"amountOut","type":"uint256"}]
}]

THENA_QUOTER_ABI = [{
 "name":"quoteExactInputSingle","type":"function","stateMutability":"nonpayable",
 "inputs":[{"components":[
     {"name":"tokenIn","type":"address"},
     {"name":"tokenOut","type":"address"},
     {"name":"amountIn","type":"uint256"},
     {"name":"limitSqrtPrice","type":"uint160"}
 ],"name":"params","type":"tuple"}],
 "outputs":[{"name":"amountOut","type":"uint256"},{"name":"sqrtPriceX96After","type":"uint160"},{"name":"initializedTicksCrossed","type":"uint32"},{"name":"gasEstimate","type":"uint256"}]
}]

MULTICALL3_ABI = [{
 "name":"tryAggregate","type":"function","stateMutability":"nonpayable",
 "inputs":[
   {"name":"requireSuccess","type":"bool"},
   {"components":[{"name":"target","type":"address"},{"name":"callData","type":"bytes"}],"name":"calls","type":"tuple[]"}],
 "outputs":[{"components":[{"name":"success","type":"bool"},{"name":"returnData","type":"bytes"}],"name":"returnData","type":"tuple[]"}]
}]

MULTICALL3_ABI_EX = MULTICALL3_ABI + [{
  "name":"tryBlockAndAggregate","type":"function","stateMutability":"nonpayable",
  "inputs":[{"name":"requireSuccess","type":"bool"},
            {"components":[{"name":"target","type":"address"},{"name":"callData","type":"bytes"}],
             "name":"calls","type":"tuple[]"}],
  "outputs":[{"name":"blockNumber","type":"uint256"},
             {"name":"returnData","type":"bytes[]"}]
}]

# DODO bits
DODO_V2_ROUTE_HELPER_ABI = [{
 "name":"getPairDetail","type":"function","stateMutability":"view",
 "inputs":[{"name":"token0","type":"address"},{"name":"token1","type":"address"},{"name":"userAddr","type":"address"}],
 "outputs":[{"components":[
    {"name":"i","type":"uint256"},{"name":"K","type":"uint256"},{"name":"B","type":"uint256"},{"name":"Q","type":"uint256"},
    {"name":"B0","type":"uint256"},{"name":"Q0","type":"uint256"},{"name":"R","type":"uint256"},
    {"name":"lpFeeRate","type":"uint256"},{"name":"mtFeeRate","type":"uint256"},
    {"name":"baseToken","type":"address"},{"name":"quoteToken","type":"address"},
    {"name":"curPair","type":"address"},{"name":"pairVersion","type":"uint256"}],"name":"res","type":"tuple[]"}]
}]
DODO_V2_POOL_ABI = [
  {"name":"_BASE_TOKEN_","type":"function","stateMutability":"view","inputs":[],"outputs":[{"type":"address"}]},
  {"name":"_QUOTE_TOKEN_","type":"function","stateMutability":"view","inputs":[],"outputs":[{"type":"address"}]},
  {"name":"_BASE_RESERVE_","type":"function","stateMutability":"view","inputs":[],"outputs":[{"type":"uint256"}]},
  {"name":"_QUOTE_RESERVE_","type":"function","stateMutability":"view","inputs":[],"outputs":[{"type":"uint256"}]},
  {"name":"querySellBase","type":"function","stateMutability":"view",
   "inputs":[{"name":"trader","type":"address"},{"name":"payBaseAmount","type":"uint256"}],
   "outputs":[{"name":"receiveQuoteAmount","type":"uint256"},{"name":"mtFee","type":"uint256"}]},
  {"name":"querySellQuote","type":"function","stateMutability":"view",
   "inputs":[{"name":"trader","type":"address"},{"name":"payQuoteAmount","type":"uint256"}],
   "outputs":[{"name":"receiveBaseAmount","type":"uint256"},{"name":"mtFee","type":"uint256"}]},
]

DVM_DPP_DSP_FACTORY_ABI = [
 {"name":"getDODOPool","type":"function","stateMutability":"view",
  "inputs":[{"name":"base","type":"address"},{"name":"quote","type":"address"}],
  "outputs":[{"name":"pools","type":"address[]"}]},
 {"name":"getDODOPoolBidirection","type":"function","stateMutability":"view",
  "inputs":[{"name":"token0","type":"address"},{"name":"token1","type":"address"}],
  "outputs":[
     {"name":"baseToken0Pools","type":"address[]"},
     {"name":"baseToken1Pools","type":"address[]"}]}
]
# add alongside V3_POOL_SLOT0_ABI
V3_POOL_LIQUIDITY_ABI = [{
    "name":"liquidity","type":"function","stateMutability":"view",
    "inputs":[],"outputs":[{"name":"","type":"uint128"}]
}]
# --- THENA (Algebra) helpers ---

ALGEBRA_FACTORY_ABI = [
    # Some forks:
    {"name":"poolByPair","type":"function","stateMutability":"view",
     "inputs":[{"name":"tokenA","type":"address"},{"name":"tokenB","type":"address"}],
     "outputs":[{"name":"pool","type":"address"}]},
    # Others just expose mapping pool(token0, token1)
    {"name":"pool","type":"function","stateMutability":"view",
     "inputs":[{"name":"token0","type":"address"},{"name":"token1","type":"address"}],
     "outputs":[{"name":"pool","type":"address"}]},
]

ALGEBRA_POOL_ABI = [
    {"name":"globalState","type":"function","stateMutability":"view","inputs":[],"outputs":[
        {"name":"price","type":"uint160"},
        {"name":"tick","type":"int24"},
        {"name":"lastFee","type":"uint16"},
        {"name":"pluginConfig","type":"uint8"},
    ]},
]



# ---- Uniswap/Pancake v3 TWAP using observe() directly ----
V3_OBSERVE_ABI = [{
    "name":"observe","type":"function","stateMutability":"view","inputs":[{"name":"secondsAgos","type":"uint32[]"}],
    "outputs":[{"name":"tickCumulatives","type":"int56[]"},{"name":"secondsPerLiquidityCumulativeX128s","type":"uint160[]"}]
}]

def v3_twap_tick(pool_addr: str, *, secs: int, block_number: int) -> int | None:
    """
    Returns average tick over `secs` seconds ending at `block_number`.
    """
    try:
        p = w3.eth.contract(address=CS(pool_addr), abi=V3_OBSERVE_ABI)
        # observe([secsAgoStart, secsAgoEnd]) => [cum_ticks_start, cum_ticks_end]
        tcs, _ = call_at(p.functions.observe([secs, 0]), block_number=block_number)
        dt = secs
        avg_tick = (int(tcs[1]) - int(tcs[0])) // dt
        return int(avg_tick)
    except Exception:
        return None


# ---------------------------
# Caches / helpers
# ---------------------------
_DECIMALS: Dict[str,int] = {}
_SYMBOLS: Dict[str,str] = {}

def is_addr(x: Any) -> bool:
    return isinstance(x, str) and x.startswith("0x") and len(x) == 42

def erc20(addr): return w3.eth.contract(address=addr, abi=ERC20_ABI)
def erc20_symbol_bytes32(addr): return w3.eth.contract(address=addr, abi=ERC20_SYMBOL_BYTES32_ABI)
def v2_router():  return w3.eth.contract(address=PANCAKE_V2_ROUTER,  abi=V2_ROUTER_ABI)
def v2_factory(): return w3.eth.contract(address=PANCAKE_V2_FACTORY, abi=V2_FACTORY_ABI)
def v3_quoter(addr):  return w3.eth.contract(address=addr, abi=V3_QUOTER_V2_ABI)
def v3_factory(addr): return w3.eth.contract(address=addr, abi=V3_FACTORY_ABI)
def v3_pool(addr):    return w3.eth.contract(address=addr, abi=V3_POOL_SLOT0_ABI)
def wombat_router():  return w3.eth.contract(address=WOMBAT_ROUTER, abi=WOMBAT_ROUTER_ABI)
def dodo_helper():    return w3.eth.contract(address=DODO_V2_ROUTE_HELPER, abi=DODO_V2_ROUTE_HELPER_ABI)
def dodo_pool(addr):  return w3.eth.contract(address=addr, abi=DODO_V2_POOL_ABI)
def dodo_factory(addr): return w3.eth.contract(address=addr, abi=DVM_DPP_DSP_FACTORY_ABI)
def kyber_quoter():  return w3.eth.contract(address=KYBER_ELASTIC_QUOTER_V2, abi=V3_QUOTER_V2_ABI)
def kyber_factory(): return w3.eth.contract(address=KYBER_ELASTIC_FACTORY,   abi=V3_FACTORY_ABI)

def decimals_of(addr) -> int:
    if not is_addr(addr):
        upper = str(addr).upper()
        if   upper == "USDT": addr = USDT
        elif upper == "USDC": addr = USDC
        elif upper == "WBNB": addr = WBNB
        elif upper in ("ETHB","ETH"): addr = ETHb
        else:
            raise ValueError(f"Not an address: {addr}")
    if addr not in _DECIMALS:
        _DECIMALS[addr] = erc20(addr).functions.decimals().call()
    return _DECIMALS[addr]

def symbol_of(addr) -> str:
    if is_addr(addr):
        if addr in _SYMBOLS: return _SYMBOLS[addr]
        try:
            s = erc20(addr).functions.symbol().call()
            if isinstance(s, bytes): s = s.decode('utf-8', errors='ignore')
        except Exception:
            try:
                b = erc20_symbol_bytes32(addr).functions.symbol().call()
                s = Web3.to_text(b).strip("\x00")
            except Exception:
                s = addr[-4:]
        _SYMBOLS[addr] = s
        return s
    return str(addr)

def human(amount, decimals): return amount / (10 ** decimals)

def call_with_pin_fallback(fn, *, block_number: int | None, label: str):
    """
    Try pinned call first (respecting snapshot). If the node reports pruned/missing state,
    rotate RPC and retry unpinned once.
    """
    # inner closure to evaluate .call()
    def _do(block_id):
        return fn.call(block_identifier=block_id) if block_id is not None else fn.call()

    try:
        return with_retries(lambda: _do(block_number), label=label)
    except Exception as e:
        if _is_transient_jsonrpc_error(e):
            # rotate and retry unpinned
            try: _rotate_rpc(f"{label}: pinned->unpinned fallback")
            except Exception: pass
            return with_retries(lambda: _do(None), label=f"{label} [unpinned]")
        raise

# ---- PCS v2 ---------------------------------------------------------------
def v2_getAmountsOut(amount_in, path, *, ctx: 'PinnedContext' | None = None):
    r = v2_router()
    fn = r.functions.getAmountsOut(int(amount_in), path)
    if ctx:
        return call_pinned_strict(fn, ctx, label="PCSv2.getAmountsOut")
    return with_retries(lambda: fn.call(), label="PCSv2.getAmountsOut", attempts=5)

def v2_getAmountsIn(amount_out, path, *, ctx: 'PinnedContext' | None = None):
    r = v2_router()
    fn = r.functions.getAmountsIn(int(amount_out), path)
    if ctx:
        return call_pinned_strict(fn, ctx, label="PCSv2.getAmountsIn")
    return with_retries(lambda: fn.call(), label="PCSv2.getAmountsIn", attempts=5)

# ---- V3 Quoter helpers ----------------------------------------------------
def v3_quote_single(quoter_addr, token_in, token_out, amount_in, fee, *, ctx: 'PinnedContext' | None = None):
    q = v3_quoter(quoter_addr)
    fn = q.functions.quoteExactInputSingle((token_in, token_out, int(fee), int(amount_in), 0))
    if ctx:
        return call_pinned_strict(fn, ctx, label="v3.quoteExactInputSingle")
    return with_retries(lambda: fn.call(), label="v3.quoteExactInputSingle")

def v3_quote_path(quoter_addr, tokens: list[str], fees: list[int], amount_in: int, *, ctx: 'PinnedContext' | None = None):
    q = v3_quoter(quoter_addr)
    path = encode_v3_path(tokens, fees)
    fn = q.functions.quoteExactInput(path, int(amount_in))
    if ctx:
        return call_pinned_strict(fn, ctx, label="v3.quoteExactInput")
    return with_retries(lambda: fn.call(), label="v3.quoteExactInput")

# ---- Wombat (stable router) ----------------------------------------------
def wombat_router_amount_out(
    token_in: str,
    token_out: str,
    amount_in: int,
    *,
    ctx: Optional[PinnedContext] = None,
    block_number: Optional[int] = None,
):
    """
    Get WombatRouter getAmountOut with pinning:
      - if ctx is provided -> strict pinned-by-hash
      - elif block_number is provided -> pinned-by-number with fallback
      - else -> latest
    """
    r  = wombat_router()
    fn = r.functions.getAmountOut([token_in, token_out], [WOMBAT_MAIN_POOL], int(amount_in))

    if ctx is not None:
        return call_pinned_strict(fn, ctx, label="wombat.getAmountOut")

    if block_number is not None:
        # best-effort pin by number with pruned fallback
        return call_with_pin_fallback(fn, block_number=int(block_number), label="wombat.getAmountOut")

    return with_retries(lambda: fn.call(), label="wombat.getAmountOut")



# ---- PCS v2 helpers -------------------------------------------------------
def v2_pair_exists(tokenA, tokenB, *, ctx: 'PinnedContext' | None = None) -> bool:
    f = v2_factory()
    fn = f.functions.getPair(tokenA, tokenB)
    try:
        pair = call_pinned_strict(fn, ctx, label="PCSv2.getPair") if ctx else \
               with_retries(lambda: fn.call(), label="PCSv2.getPair")
    except Exception:
        return False
    return int(pair, 16) != 0


# ---- Uniswap/Pancake v3 plumbing (pin-aware) ------------------------------
from functools import lru_cache

@lru_cache(maxsize=4096)
def _probe_slot0_ok(addr: str, block_number: Optional[int]=None) -> bool:
    """
    Raw eth_call to slot0() selector (0x3850c7bd).
    Returns True if the call succeeded and returned >= 192 bytes (6 words),
    which is compatible with Uniswap v3-like pools (6-7 fixed fields).
    """
    try:
        data = HexBytes("0x3850c7bd")
        raw  = with_retries(
            lambda: w3.eth.call({"to": CS(addr), "data": data}, block_identifier=block_number),
            label="probe.slot0", attempts=2
        )
        out = bytes(raw)
        return len(out) >= 192  # tolerate 6 or 7 words
    except Exception:
        return False

def v3_pool_has_liquidity(addr: str, block_number: Optional[int]=None) -> bool:
    c = w3.eth.contract(address=CS(addr), abi=V3_POOL_LIQUIDITY_ABI)
    try:
        fn = c.functions.liquidity()
        val = with_retries(
            lambda: (fn.call(block_identifier=block_number) if block_number is not None else fn.call()),
            label="v3.liquidity", attempts=3
        )
        return int(val) > 0
    except Exception:
        return False


def v3_pool_address(factory_addr: str, tokenA: str, tokenB: str, fee: int,
                    *, block_number: Optional[int] = None) -> Optional[str]:
    f = v3_factory(factory_addr)
    fn = f.functions.getPool(tokenA, tokenB, int(fee))
    try:
        if block_number is not None:
            pool = call_with_pin_fallback(fn, block_number=int(block_number), label="v3.getPool")
        else:
            pool = with_retries(lambda: fn.call(), label="v3.getPool")
    except Exception:
        return None
    return None if int(pool, 16) == 0 else CS(pool)

@lru_cache(maxsize=4096)
def v3_pool_status(factory_addr: str, tokenA: str, tokenB: str, fee: int,
                   *, block_number: Optional[int] = None) -> Tuple[bool, str, str]:
    addr = v3_pool_address(factory_addr, tokenA, tokenB, fee, block_number=block_number)
    if not addr:
        return (False, "no pool", "")

    code = eth_get_code_safe(addr)
    if not code:
        return (False, "pool not deployed", addr)

    # NEW: probe the raw selector first — avoids noisy decode logs
    if not _probe_slot0_ok(addr, block_number):
        return (False, "pool ABI mismatch", addr)

    # Optional: liquidity gate (best-effort)
    try:
        if not v3_pool_has_liquidity(addr, block_number):
            return (False, "no liquidity", addr)
    except Exception:
        pass

    return (True, "", addr)

def v3_quote_single(quoter_addr, token_in, token_out, amount_in, fee, *, ctx: 'PinnedContext' | None = None):
    q  = v3_quoter(quoter_addr)
    fn = q.functions.quoteExactInputSingle((token_in, token_out, int(fee), int(amount_in), 0))
    if ctx:
        return call_pinned_strict(fn, ctx, label="v3.quoteExactInputSingle")
    return with_retries(lambda: fn.call(), label="v3.quoteExactInputSingle")

def v3_quote_path(quoter_addr, tokens: list[str], fees: list[int], amount_in: int, *, ctx: 'PinnedContext' | None = None):
    q    = v3_quoter(quoter_addr)
    path = encode_v3_path(tokens, fees)
    fn   = q.functions.quoteExactInput(path, int(amount_in))
    if ctx:
        return call_pinned_strict(fn, ctx, label="v3.quoteExactInput")
    return with_retries(lambda: fn.call(), label="v3.quoteExactInput")

def encode_v3_path(tokens: List[str], fees: List[int]) -> str:
    if len(tokens) != len(fees) + 1:
        raise ValueError("len(tokens) must be len(fees)+1")
    out = b""
    for i in range(len(fees)):
        out += bytes.fromhex(tokens[i][2:])
        out += int(fees[i]).to_bytes(3, byteorder="big", signed=False)
    out += bytes.fromhex(tokens[-1][2:])
    return "0x" + out.hex()


# ---- THENA / Algebra v1.0 -------------------------------------------------
def thena_factory():
    if not THENA_FACTORY:
        return None
    return w3.eth.contract(address=CS(THENA_FACTORY), abi=ALGEBRA_FACTORY_ABI)

@lru_cache(maxsize=4096)
def _algebra_get_pool_unpinned_cached(factory_addr: str, a: str, b: str) -> str | None:
    fac = w3.eth.contract(address=CS(factory_addr), abi=ALGEBRA_FACTORY_ABI)
    for fn_name in ("poolByPair", "pool"):
        for x, y in ((a, b), (b, a)):
            try:
                addr = with_retries(lambda: getattr(fac.functions, fn_name)(x, y).call(), label=f"thena.{fn_name}")
                if int(addr, 16) != 0:
                    return CS(addr)
            except Exception:
                continue
    return None

def thena_quoter():
    """Bind THENA Quoter with the CORRECT Algebra v1.0 ABI."""
    return w3.eth.contract(address=THENA_QUOTER, abi=THENA_QUOTER_ABI)

def extract_revert_info(exc: Exception) -> tuple[str, str]:
    """(reason_text, raw_hex_short) — unchanged utility."""
    try:
        s = str(exc) or ""
    except Exception:
        s = ""
    raw_hex = ""
    if hasattr(exc, "args") and exc.args:
        a0 = exc.args[0]
        a1 = exc.args[1] if len(exc.args) > 1 else None
        if isinstance(a0, dict):
            data = a0.get("data") or a0.get("message") or ""
            if isinstance(data, str) and data.startswith("0x"):
                try:
                    b = bytes.fromhex(data[2:])
                    s2 = decode_revert_text_or_none(b)
                    if s2: s = s2
                    raw_hex = "0x" + _dump_hex(b)
                except Exception:
                    pass
        if isinstance(a1, str) and a1.startswith("0x"):
            try:
                b = bytes.fromhex(a1[2:])
                s2 = decode_revert_text_or_none(b)
                if s2: s = s2
                raw_hex = "0x" + _dump_hex(b)
            except Exception:
                pass
    if not s:
        s = "reverted"
    return s, raw_hex

@lru_cache(maxsize=4096)
def _algebra_get_pool_cached(factory_addr: str, a: str, b: str) -> Optional[str]:
    fac = w3.eth.contract(address=CS(factory_addr), abi=ALGEBRA_FACTORY_ABI)
    for fn_name in ("poolByPair", "pool"):
        for x, y in ((a, b), (b, a)):
            try:
                fn = getattr(fac.functions, fn_name)(x, y)
                addr = with_retries(lambda: fn.call(), label=f"thena.{fn_name}")
                if int(addr, 16) != 0:
                    return CS(addr)
            except Exception:
                continue
    return None

def thena_pool_status(tokenA: str, tokenB: str, *, block_number: int | None = None) -> tuple[bool, str, str | None]:
    if not THENA_FACTORY:
        return (False, "factory not configured", None)

    fac = w3.eth.contract(address=THENA_FACTORY, abi=ALGEBRA_FACTORY_ABI)

    def _call(fn):
        return fn.call(block_identifier=block_number) if block_number is not None else fn.call()

    # try poolByPair / pool, both directions
    addr = None
    for fn_name in ("poolByPair", "pool"):
        for x, y in ((tokenA, tokenB), (tokenB, tokenA)):
            try:
                candidate = _call(getattr(fac.functions, fn_name)(x, y))
                if int(candidate, 16) != 0:
                    addr = CS(candidate); break
            except Exception:
                pass
        if addr: break

    if not addr:
        return (False, "no pool", None)

    code = eth_get_code_safe(addr)
    if not code:
        return (False, "pool not deployed", addr)

    # Algebra: prefer liquidity() as initialization gate
    try:
        liq = w3.eth.contract(address=addr, abi=V3_POOL_LIQUIDITY_ABI).functions.liquidity()
        val = _call(liq)
        if int(val) <= 0:
            return (False, "no liquidity", addr)
    except Exception:
        return (False, "abi mismatch", addr)

    # Optional: try globalState() but ignore its decode errors
    try:
        gf = w3.eth.contract(address=addr, abi=ALGEBRA_POOL_ABI).functions.globalState()
        _ = _call(gf)  # succeeds on proper Algebra pools (8 outputs). 
    except Exception:
        # not fatal — we already verified liquidity
        pass

    return (True, "", addr)


# ---- v3 status compatibility wrapper --------------------------------------

def v3_status2(factory_addr: str, tokenA: str, tokenB: str, fee: int,
               *, block_number: Optional[int]=None) -> Tuple[bool, str]:
    ok, reason, _addr = v3_pool_status(factory_addr, tokenA, tokenB, fee, block_number=block_number)
    return ok, reason

# ---------------------------
# Revert parsing + raw dump
# ---------------------------
ERROR_STRING_SIG = b'\x08\xc3\x79\xa0'
PANIC_SIG        = b'\x4e\x48\x7b\x71'

def _dump_hex(data: bytes, max_bytes: int = 96) -> str:
    if not data: return ""
    hx = data.hex()
    return hx if len(hx) <= max_bytes*2 else hx[:max_bytes*2] + "..."

def decode_revert_text_or_none(data: bytes) -> Optional[str]:
    if not data or len(data) < 4: return None
    sig = data[:4]
    if sig == ERROR_STRING_SIG:
        try:
            return abi_decode(['string'], data[4:])[0]
        except Exception:
            return "error(string)"
    if sig == PANIC_SIG:
        try:
            code = int.from_bytes(data[4+28:4+32], 'big')
            return f"Panic({code})"
        except Exception:
            return "Panic"
    return None

def looks_like_uint256_array_or_tuple(data: bytes) -> Tuple[bool, bool]:
    """return (is_plain_array, is_tuple_wrapped_array)"""
    if not data or len(data) < 96: return (False, False)
    # plain array: [len][items...]
    n_plain = int.from_bytes(data[0:32],'big')
    if (len(data) >= 32 + 32*n_plain) and n_plain < 1_000:
        return (True, False)
    # tuple-wrapped: [0x20][len][items...]
    w0 = int.from_bytes(data[0:32],'big')
    if w0 == 32:
        n = int.from_bytes(data[32:64],'big')
        if (len(data) >= 64 + 32*n) and n < 1_000:
            return (False, True)
    return (False, False)

def decode_uint256_array_any(data: bytes) -> Optional[List[int]]:
    try:
        # plain array
        arr = abi_decode(['uint256[]'], data)[0]
        return [int(x) for x in arr]
    except Exception:
        pass
    try:
        # tuple-wrapped array: skip first 32 bytes
        if len(data) >= 64 and int.from_bytes(data[0:32],'big') == 32:
            arr = abi_decode(['uint256[]'], data[32:])[0]
            return [int(x) for x in arr]
    except Exception:
        pass
    try:
        # explicit tuple decode
        (arr,) = abi_decode(['(uint256[])'], data)
        return [int(x) for x in arr]
    except Exception:
        return None
def _u256_at(b: bytes, i: int) -> int:
    return int.from_bytes(b[i:i+32], 'big') if len(b) >= i+32 else 0

def unwrap_tuple_wrapped_uint256_array(data: bytes) -> Optional[bytes]:
    """
    If `data` looks like ABI-encoding of a tuple containing a single dynamic uint256[],
    i.e. [0x20][N][v0][v1]...[v{N-1}], return the inner plain-array encoding starting
    at byte 32: [N][v0][v1]...[v{N-1}]. Otherwise return None.
    """
    if not data or len(data) < 64:
        return None
    w0 = _u256_at(data, 0)         # expected 0x20
    n  = _u256_at(data, 32)        # length of array
    if w0 != 32:
        return None
    # we require at least the header [0x20][N] plus N words
    min_len = 64 + 32 * n
    if n > 4096 or len(data) < min_len:
        return None
    # inner is the array encoding: [N][v0]...[v{N-1}]
    inner = data[32:min_len]
    # quick self-check
    if _u256_at(inner, 0) != n:
        return None
    return inner

def try_decode_quoter_v2_revert(amt_in: int, payload: bytes) -> Optional[int]:
    """
    Try to pull amountOut from Quoter V2 revert payloads for quoteExactInput:
    - unwrap tuple-wrapped uint256[] if present
    - decode as uint256[] (or fall back to plain array)
    Heuristic selection:
      if arr[0] == amountIn -> amountOut = arr[1]
      else -> amountOut = arr[-1]
    """
    if not payload:
        return None

    # Prefer tuple-wrapped pattern (most common for your logs)
    inner = unwrap_tuple_wrapped_uint256_array(payload)
    candidates = []

    if inner:
        try:
            arr = abi_decode(['uint256[]'], inner)[0]
            arr = [int(x) for x in arr]
            if arr:
                candidates.append(arr)
        except Exception:
            pass

    # Plain array fallback (some nodes may already strip the wrapper)
    if not candidates:
        try:
            arr = abi_decode(['uint256[]'], payload)[0]
            arr = [int(x) for x in arr]
            if arr:
                candidates.append(arr)
        except Exception:
            pass

    if not candidates:
        # single uint256 fallback (older-style single-hop behavior)
        try:
            (val,) = abi_decode(['uint256'], payload)
            return int(val)
        except Exception:
            return None

    arr = candidates[0]
    if len(arr) == 1:
        return int(arr[0])
    # common shapes: [amountIn, amountOut, ...]  or  [..., amountOut]
    return int(arr[1]) if arr[0] == amt_in else int(arr[-1])

# ---------------------------
# Data classes
# ---------------------------
@dataclass
class Quote:
    venue: str
    path: str
    amount_in: int
    amount_out: int
    out_token: str
    ms: float
    gas_estimate: Optional[int] = None
    ticks_crossed: Optional[int] = None

@dataclass
class RawDump:
    venue: str
    desc: str
    ok: bool
    raw_len: int
    raw_head_hex: str

# ---------------------------
# DODO helpers
# ---------------------------
# ---------- DODO (BSC) addresses: from YOUR message ----------
DODO_V2_ROUTE_HELPER = CS("0xb48eE7B874Af8bC0e068036e55e33b5DC91C3a65")
DODO_DVM_FACTORY     = CS("0x790B4A80Fb1094589A3c0eFC8740aA9b0C1733fB")
DODO_DPP_FACTORY     = CS("0xd9CAc3D964327e47399aebd8e1e6dCC4c251DaAE")
DODO_DSP_FACTORY     = CS("0x0fb9815938Ad069Bf90E14FE6C596c514BEDe767")
DODO_UPCP_FACTORY    = CS("0x4F57F6929E58AE564F6AB090fE0AEEcb39B0f270")  # optional/rare on BSC

ZERO_ADDR = "0x0000000000000000000000000000000000000000"

# We will ONLY call functions present in the ABIs you already provided:
# - DVM_DPP_DSP_FACTORY_ABI
# - DODO_V2_POOL_ABI
# - DODO_V2_ROUTE_HELPER_ABI

def _checksum(a: str) -> str:
    # be permissive: accept already-checksummed or lowercased 0x-40 hex
    if not isinstance(a, str) or not a.startswith("0x") or len(a) != 42:
        raise ValueError(f"bad address string: {a}")
    return Web3.to_checksum_address(a)

def _has_fn(contract, fn_name: str) -> bool:
    try:
        getattr(contract.functions, fn_name)
        return True
    except Exception:
        return False

def dodo_discover_pools_via_factories(base_token: str, quote_token: str, verbose: bool=False) -> list[tuple[str,str]]:
    """
    Query DVM/DPP/DSP (and UPCP if you like) factories for pools between (base_token, quote_token).
    Returns [(pool_addr, kind)], deduped.
    """
    pools: list[tuple[str,str]] = []
    t0 = time.perf_counter()
    token0 = _checksum(base_token)
    token1 = _checksum(quote_token)
    fac_abi = DVM_DPP_DSP_FACTORY_ABI

    def _fetch(factory_addr: str, label: str):
        if not factory_addr:
            return [], (0, 0)
        try:
            fac = w3.eth.contract(address=factory_addr, abi=fac_abi)
            uni = fac.functions.getDODOPool(token0, token1).call()
            bi0, bi1 = fac.functions.getDODOPoolBidirection(token0, token1).call()
            # Dedup at source, keep raw counts for log "a+b"
            addrs = list({Web3.to_checksum_address(x) for x in (uni + bi0 + bi1) if isinstance(x, str) and x.startswith("0x") and len(x) == 42})
            return addrs, (len(uni), len(bi0) + len(bi1))
        except Exception as e:
            if verbose:
                print(f"[DODO-view] {label} factory query failed: {e}")
            return [], (0, 0)

    dvm, c_dvm = _fetch(DODO_DVM_FACTORY, "DVM")
    dpp, c_dpp = _fetch(DODO_DPP_FACTORY, "DPP")
    dsp, c_dsp = _fetch(DODO_DSP_FACTORY, "DSP")
    upc, c_upc = _fetch(DODO_UPCP_FACTORY, "UPCP") if DODO_UPCP_FACTORY else ([], (0, 0))

    if verbose:
        if dvm: print(f"[DODO-view] DVM {dvm[0][-6:]} returned {c_dvm[0]}+{c_dvm[1]} pools")
        if dpp: print(f"[DODO-view] DPP {dpp[0][-6:]} returned {c_dpp[0]}+{c_dpp[1]} pools")
        if dsp: print(f"[DODO-view] DSP {dsp[0][-6:]} returned {c_dsp[0]}+{c_dsp[1]} pools")
        if upc: print(f"[DODO-view] UPCP {upc[0][-6:]} returned {c_upc[0]}+{c_upc[1]} pools")
        print(f"[DODO-view] discovery total {len(dvm)+len(dpp)+len(dsp)+len(upc)} pools in {(time.perf_counter()-t0)*1000:.1f} ms")

    pools += [(p, "DVM")  for p in dvm]
    pools += [(p, "DPP")  for p in dpp]
    pools += [(p, "DSP")  for p in dsp]
    pools += [(p, "UPCP") for p in upc]
    return pools

def dodo_quote_direct_pools(base_token: str, amount_in_wei: int,
                            pools: list[tuple[str,str]], verbose: bool=False) -> tuple[list[tuple[str,int]], list[dict]]:
    """
    For each pool address, detect orientation via _BASE_TOKEN_/_QUOTE_TOKEN_.
    Then call querySellBase or querySellQuote (read-only) and collect out amounts.
    Returns (quotes, failures):
       quotes   = [(pool_addr, out_amount)]
       failures = [{"pool":addr,"kind":kind,"error":str(e)}...]
    """
    out: list[tuple[str,int]] = []
    fails: list[dict] = []
    if not pools:
        return out, fails

    base = _checksum(base_token)
    pool_abi = DODO_V2_POOL_ABI

    for (addr, kind) in pools:
        try:
            pool = w3.eth.contract(address=_checksum(addr), abi=pool_abi)
            # Only call functions we actually declared (defensive against ABI drift)
            if not (_has_fn(pool, "_BASE_TOKEN_") and _has_fn(pool, "_QUOTE_TOKEN_")):
                if verbose:
                    print(f"[DODO] skip {addr[-6:]} {kind}: pool ABI missing base/quote getters")
                continue

            b = pool.functions._BASE_TOKEN_().call()
            q = pool.functions._QUOTE_TOKEN_().call()

            if b.lower() == base.lower():
                if not _has_fn(pool, "querySellBase"):
                    if verbose: print(f"[DODO] skip {addr[-6:]} {kind}: no querySellBase")
                    continue
                recv, _fee = pool.functions.querySellBase(_checksum(ZERO_ADDR), int(amount_in_wei)).call()
                out.append((addr, int(recv)))

            elif q.lower() == base.lower():
                if not _has_fn(pool, "querySellQuote"):
                    if verbose: print(f"[DODO] skip {addr[-6:]} {kind}: no querySellQuote")
                    continue
                recv, _fee = pool.functions.querySellQuote(_checksum(ZERO_ADDR), int(amount_in_wei)).call()
                out.append((addr, int(recv)))

            else:
                if verbose:
                    print(f"[DODO] skip {addr[-6:]} {kind}: pair mismatch vs input token")
                continue

        except Exception as e:
            # Common: revert inside FeeRateDIP3Impl (KJUDGE_ERROR) or pool not initialized
            fails.append({"pool": addr, "kind": kind, "error": str(e)})
            if verbose:
                print(f"[DODO] pool {addr[-6:]} {kind} query failed: {e}")

    return out, fails

def dodo_quote_helper(base_token: str, quote_token: str, amount_in_wei: int, verbose: bool=False) -> tuple[list[tuple[str,int]], list[dict]]:
    """
    Try helper (getPairDetail). If it reverts (e.g. KJUDGE_ERROR) we swallow and return [].
    We do NOT rely on helper for amounts; we extract its pool addresses and quote each pool directly
    to avoid any ABI mismatches and to uniformly compute amounts.
    """
    results: list[tuple[str,int]] = []
    fails:   list[dict] = []
    helper_addr = DODO_V2_ROUTE_HELPER
    if not helper_addr:
        return results, fails

    try:
        helper = w3.eth.contract(address=helper_addr, abi=DODO_V2_ROUTE_HELPER_ABI)
        if not _has_fn(helper, "getPairDetail"):
            if verbose:
                print("  [dodo] helper ABI has no getPairDetail; skipping helper")
            return results, fails

        base  = _checksum(base_token)
        quote = _checksum(quote_token)

        t0 = time.perf_counter()
        details = helper.functions.getPairDetail(base, quote, _checksum(ZERO_ADDR)).call()
        if not isinstance(details, (list, tuple)):
            if verbose:
                print("  [dodo] helper returned unexpected type; skipping helper")
            return results, fails

        # Pull 'curPair' field by position (index 11 in your ABI)
        pools = []
        for item in details:
            try:
                pool_addr = _checksum(item[11])
                pools.append((pool_addr, "HELPER"))
            except Exception:
                continue

        if verbose:
            print(f"  [dodo] helper discovered {len(pools)} pool(s) in {(time.perf_counter()-t0)*1000:.1f} ms")

        # Quote each pool directly (uniform path)
        q, f = dodo_quote_direct_pools(base_token, amount_in_wei, pools, verbose=verbose)
        results.extend(q)
        fails.extend(f)

    except Exception as e:
        # This is where KJUDGE_ERROR from nested fee logic typically bubbles up
        if verbose:
            print(f"  [dodo] helper discovery failed: {e}")
        # We do NOT propagate; we just treat helper as unavailable
        # (factory discovery still runs)

    return results, fails

# ---- Optional legacy log scan (kept, but try view discovery first) ----
EVT_NEWDVM = Web3.keccak(text="NewDVM(address,address,address,address)")
EVT_NEWDPP = Web3.keccak(text="NewDPP(address,address,address,address)")
EVT_NEWDSP = Web3.keccak(text="NewDSP(address,address,address,address)")
EVT_NEWUPCP = [Web3.keccak(text="NewUpCp(address,address,address,address)"),
               Web3.keccak(text="NewCP(address,address,address,address)")]

def _extract_address_from_32(data_word: bytes) -> str:
    return CS("0x" + data_word[-20:].hex())

def _try_decode_pool_from_log(log) -> Optional[str]:
    if not log["data"]:
        return None
    b = bytes.fromhex(log["data"][2:])
    if len(b) < 32:
        return None
    return _extract_address_from_32(b[-32:])

def dodo_factory_scan(base_token: str, quote_token: str, frm: int, to: int|str,
                      *, verbose=False, step_blocks: int = 50_000,
                      max_total_logs: int = 20_000, hard_timeout_s: int = 45) -> List[str]:
    t_start = time.time()
    pools: List[str] = []
    factories = [
        (DODO_DVM_FACTORY, [EVT_NEWDVM]),
        (DODO_DPP_FACTORY, [EVT_NEWDPP]),
        (DODO_DSP_FACTORY, [EVT_NEWDSP]),
        (DODO_UPCP_FACTORY, EVT_NEWUPCP),
    ]
    latest = w3.eth.block_number
    to_block = latest if (isinstance(to, str) and to.lower()=="latest") else int(to)
    from_block = max(1, int(frm))
    total_logs = 0

    for fac, topics in factories:
        for t0 in topics:
            start = from_block
            while start <= to_block:
                if (time.time() - t_start) > hard_timeout_s:
                    if verbose: print("[DODO-scan] timeout reached; stopping scan early")
                    start = to_block + 1
                    break
                end = min(start + step_blocks - 1, to_block)
                params: FilterParams = {
                    "address": fac,
                    "fromBlock": start,
                    "toBlock": end,
                    "topics": [t0.hex()]
                }
                try:
                    logs = w3.eth.get_logs(params)
                except Exception as e:
                    if verbose: print(f"[DODO-scan] get_logs {fac} {start}-{end} failed: {e}; shrinking window")
                    step_blocks = max(2_000, step_blocks // 2)
                    if start == end: start += 1
                    continue

                total_logs += len(logs)
                if verbose:
                    print(f"[DODO-scan] {fac[-6:]} {start}-{end} → {len(logs)} logs (total {total_logs})")
                for lg in logs:
                    pool = _try_decode_pool_from_log(lg)
                    if not pool:
                        continue
                    try:
                        p = dodo_pool(pool)
                        bt = p.functions.baseToken().call()
                        qt = p.functions.quoteToken().call()
                    except Exception:
                        continue
                    if {bt.lower(), qt.lower()} != {base_token.lower(), quote_token.lower()}:
                        continue
                    pools.append(pool)
                if total_logs >= max_total_logs:
                    if verbose: print("[DODO-scan] max_total_logs reached; stopping early")
                    start = to_block + 1
                    break
                start = end + 1

    seen=set(); uniq=[]
    for p in pools:
        k=p.lower()
        if k in seen: continue
        seen.add(k); uniq.append(p)
    return uniq

# ---------------------------
# Multicall
# ---------------------------
def multicall_try_block_and_aggregate(calls: list[tuple[str, bytes]]):
    """
    Executes all calls at one block height on the current endpoint and returns:
        (blockNumber:int, returnData:List[bytes])
    Useful when you don't already have a pinned context.
    """
    mc = w3.eth.contract(address=MULTICALL3, abi=MULTICALL3_ABI_EX)
    return with_retries(
        lambda: mc.functions.tryBlockAndAggregate(False, [(c[0], c[1]) for c in calls]).call(),
        label="multicall3.tryBlockAndAggregate"
    )

# ---- DROP-IN: multicall pinned/unpinned with ctx OR block id ----------------
# ---- Hardened pinned Multicall with archive + clean fallback ----------------
STRICT_PIN = str(os.environ.get("STRICT_PIN", "0")).lower() in ("1", "true", "yes")

def _archive_urls() -> list[str]:
    raw = os.environ.get("BSC_RPC_ARCHIVE", "")
    return [u.strip() for u in raw.split(",") if u.strip()]

# ---- DROP-IN: Multicall3 pinned→unpinned with dual signature ---------------

def multicall_try_aggregate_at(calls: List[Tuple[str, bytes]],
                               *, block_number: Optional[int] = None,
                               ctx: Optional['PinnedContext'] = None):
    """
    Returns List[(success:bool, returnData:bytes)].
    Accepts either:
      - ctx=PinnedContext (strict pin to block_hash/url), or
      - block_number=… (best-effort pin by number, with pruned fallback)
    """
    mc = w3.eth.contract(address=MULTICALL3, abi=MULTICALL3_ABI)
    fn = mc.functions.tryAggregate(False, [(c[0], c[1]) for c in calls])

    label = "multicall3.tryAggregate"

    # Strict ctx pinning (your pinned-by-hash path)
    if ctx is not None:
        return call_pinned_strict(fn, ctx, label=label)

    # Legacy/block_number path: try pinned-by-number, on pruned fallback to unpinned
    if block_number is not None:
        ident = int(block_number)
        try:
            return with_retries(lambda: fn.call(block_identifier=ident),
                                label=f"{label}[pinned-id]", attempts=5)
        except Exception as e:
            if _is_pruned_or_blockgap(e) or _is_transient_jsonrpc_error(e):
                try: _rotate_rpc(f"{label}: pinned->unpinned")
                except Exception: pass
                return with_retries(lambda: fn.call(), label=f"{label}[unpinned]", attempts=5)
            raise

    # Unpinned
    return with_retries(lambda: fn.call(), label=f"{label}[latest]", attempts=5)

def build_mc_calls(amount_in_eth_like: int, base_token: str):
    """
    Build ONE Multicall3 batch + an aligned `meta` describing how to decode.

    Venues (all at the same block when executed):
      • Uniswap v3 (QuoterV2):     direct BASE->USDT/USDC + 2-hop via INTERMEDIATES
      • Pancake v3 (QuoterV2):     direct BASE->USDT/USDC + 2-hop via INTERMEDIATES
      • KyberSwap Elastic (QuoterV2): direct BASE->USDT/USDC + 2-hop via INTERMEDIATES
          IMPORTANT: Kyber uses fee **in basis points (bps)**; we prefilter with feeAmountTickDistance.
          We always call `quoteExactInput(path, amountIn)` to keep one ABI across v3-like venues.
      • Pancake v2 (Router):       baseline V2 paths
      • Biswap  v2 (Router):       baseline V2 paths (correct router => real getAmountsOut)

    Return:
      calls: [(target_addr, calldata), ...]
      meta : list of tuples aligned with results for non-skip entries:
             (kind, venue, pdesc, out_token_addr, amount_in, target_addr)
             where kind ∈ {"decode_path","decode_v2"}; we also emit ("skip",...) rows
             (they don't consume a result slot; useful for logging/DB).
    """
    calls: list[tuple[str, bytes]] = []
    meta:  list[tuple[str, str, str, str, int, str]] = []
    used:  set[tuple[str, bytes]] = set()  # de-dup exact (target, calldata)

    # ---- bind quoters/routers
    uniQ   = v3_quoter(UNISWAP_V3_QUOTER_V2)
    pcsQ   = v3_quoter(PANCAKE_V3_QUOTER_V2)
    kybQ   = v3_quoter(KYBER_ELASTIC_QUOTER_V2)

    pcsV2  = w3.eth.contract(address=PANCAKE_V2_ROUTER, abi=V2_ROUTER_ABI)
    bisV2  = w3.eth.contract(address=BISWAP_V2_ROUTER,   abi=V2_ROUTER_ABI)

    # ---- Kyber factory (min ABI incl. fee filter in **bps**)
    KYBER_FACTORY_ABI_MIN = [
        {"name":"getPool","type":"function","stateMutability":"view",
         "inputs":[{"name":"tokenA","type":"address"},{"name":"tokenB","type":"address"},{"name":"swapFeeBps","type":"uint16"}],
         "outputs":[{"name":"pool","type":"address"}]},
        {"name":"feeAmountTickDistance","type":"function","stateMutability":"view",
         "inputs":[{"name":"swapFeeBps","type":"uint16"}],
         "outputs":[{"name":"tickDistance","type":"int24"}]},
    ]
    kybF = w3.eth.contract(address=KYBER_ELASTIC_FACTORY, abi=KYBER_FACTORY_ABI_MIN)

    # ---------------- helpers ----------------
    def _sym(a: str) -> str:
        try: return symbol_of(a)
        except Exception: return a

    def _add_call(target: str, data: bytes, meta_tuple):
        key = (CS(target), data)
        if key in used: return
        used.add(key)
        calls.append(key)
        meta.append(meta_tuple)

    # ---- generic v3-like status (Uniswap/Pancake)
    def _v3_status(factory_addr: str, a: str, b: str, fee_u24: int) -> tuple[bool, str]:
        # tolerant to your helper’s 2- or 3-tuple return
        res = v3_pool_status(factory_addr, a, b, int(fee_u24))
        try: ok, reason, _addr = res
        except Exception: ok, reason = res
        return bool(ok), str(reason or "")

    # ---- Kyber status (fee **bps**)
    def _kyber_status(a: str, b: str, fee_bps: int) -> tuple[bool, str]:
        try:
            # quick “is this fee enabled at all” check
            td = kybF.functions.feeAmountTickDistance(int(fee_bps)).call()
            if int(td) <= 0:
                return False, "fee disabled"
        except Exception:
            # if factory doesn’t expose (shouldn’t happen), just attempt getPool
            pass
        try:
            pool = kybF.functions.getPool(a, b, int(fee_bps)).call()
            if int(pool, 16) == 0:
                return False, "no pool"
            if not eth_get_code_safe(pool):
                return False, "pool not deployed"
            # optional: check liquidity() > 0 if you want
            return True, ""
        except Exception as e:
            return False, f"error:{str(e)[:48]}"

    # ---- add v3-like quotes (uniform: quoteExactInput(path, amountIn))
    def _add_v3_direct_exact_input(venue: str, quoter, factory_addr: str,
                                   token_out: str, fees_u24: list[int]):
        for fee in fees_u24:
            ok, reason = _v3_status(factory_addr, base_token, token_out, fee)
            if not ok:
                meta.append(("skip", venue,
                             f"{_sym(base_token)}->{_sym(token_out)} (fee {fee})",
                             reason, amount_in_eth_like, ""))  # no target consumed
                continue
            path = encode_v3_path([base_token, token_out], [int(fee)])
            data = quoter.encodeABI("quoteExactInput", args=[path, int(amount_in_eth_like)])
            _add_call(quoter.address, data,
                      ("decode_path", venue,
                       f"{_sym(base_token)}->{_sym(token_out)} (fee {fee})",
                       token_out, int(amount_in_eth_like), quoter.address))

    def _add_v3_twohop_exact_input(venue: str, quoter, factory_addr: str,
                                   token_mid: str, token_out: str,
                                   fees1_u24: list[int], fees2_u24: list[int]):
        if token_mid.lower() in {token_out.lower(), base_token.lower()}:
            return
        for f1 in fees1_u24:
            for f2 in fees2_u24:
                ok1, r1 = _v3_status(factory_addr, base_token, token_mid, f1)
                ok2, r2 = _v3_status(factory_addr, token_mid, token_out, f2)
                if not (ok1 and ok2):
                    reason = " / ".join([x for x in (r1 if not ok1 else None, r2 if not ok2 else None) if x]) or "no pool(s)"
                    meta.append(("skip", venue,
                                 f"{_sym(base_token)}->{_sym(token_mid)}->{_sym(token_out)} ({f1}/{f2})",
                                 reason, amount_in_eth_like, ""))
                    continue
                path = encode_v3_path([base_token, token_mid, token_out], [int(f1), int(f2)])
                data = quoter.encodeABI("quoteExactInput", args=[path, int(amount_in_eth_like)])
                _add_call(quoter.address, data,
                          ("decode_path", venue,
                           f"{_sym(base_token)}->{_sym(token_mid)}->{_sym(token_out)} ({f1}/{f2})",
                           token_out, int(amount_in_eth_like), quoter.address))

    # ---- Kyber v3-like with **bps** fees
    def _add_kyber_direct(token_out: str, fees_bps: list[int]):
        for fb in fees_bps:
            ok, reason = _kyber_status(base_token, token_out, fb)
            if not ok:
                meta.append(("skip", "kyber_elastic",
                             f"{_sym(base_token)}->{_sym(token_out)} (bps {fb})",
                             reason, amount_in_eth_like, ""))
                continue
            path = encode_v3_path([base_token, token_out], [int(fb)])
            data = kybQ.encodeABI("quoteExactInput", args=[path, int(amount_in_eth_like)])
            _add_call(kybQ.address, data,
                      ("decode_path", "kyber_elastic",
                       f"{_sym(base_token)}->{_sym(token_out)} (bps {fb})",
                       token_out, int(amount_in_eth_like), kybQ.address))

    def _add_kyber_twohop(token_mid: str, token_out: str, fees1_bps: list[int], fees2_bps: list[int]):
        if token_mid.lower() in {token_out.lower(), base_token.lower()}:
            return
        for f1 in fees1_bps:
            for f2 in fees2_bps:
                ok1, r1 = _kyber_status(base_token, token_mid, f1)
                ok2, r2 = _kyber_status(token_mid, token_out, f2)
                if not (ok1 and ok2):
                    reason = " / ".join([x for x in (r1 if not ok1 else None, r2 if not ok2 else None) if x]) or "no pool(s)"
                    meta.append(("skip", "kyber_elastic",
                                 f"{_sym(base_token)}->{_sym(token_mid)}->{_sym(token_out)} ({f1}/{f2} bps)",
                                 reason, amount_in_eth_like, ""))
                    continue
                path = encode_v3_path([base_token, token_mid, token_out], [int(f1), int(f2)])
                data = kybQ.encodeABI("quoteExactInput", args=[path, int(amount_in_eth_like)])
                _add_call(kybQ.address, data,
                          ("decode_path", "kyber_elastic",
                           f"{_sym(base_token)}->{_sym(token_mid)}->{_sym(token_out)} ({f1}/{f2} bps)",
                           token_out, int(amount_in_eth_like), kybQ.address))

    # ---- V2 routers
    def _add_v2_paths(router, router_addr: str, venue_tag: str, paths: list[list[str]]):
        for path in paths:
            data = router.encodeABI("getAmountsOut", args=[int(amount_in_eth_like), path])
            _add_call(router_addr, data,
                      ("decode_v2", venue_tag, "->".join(_sym(p) for p in path),
                       path[-1], int(amount_in_eth_like), router_addr))

    # -------- assemble batch --------
    # Uniswap v3
    _add_v3_direct_exact_input("uniswap_v3", uniQ, UNISWAP_V3_FACTORY, USDT, UNI_V3_FEES)
    _add_v3_direct_exact_input("uniswap_v3", uniQ, UNISWAP_V3_FACTORY, USDC, UNI_V3_FEES)
    for mid in INTERMEDIATES:
        _add_v3_twohop_exact_input("uniswap_v3", uniQ, UNISWAP_V3_FACTORY, mid, USDT, UNI_V3_FEES, UNI_V3_FEES)
        _add_v3_twohop_exact_input("uniswap_v3", uniQ, UNISWAP_V3_FACTORY, mid, USDC, UNI_V3_FEES, UNI_V3_FEES)

    # Pancake v3
    _add_v3_direct_exact_input("pancake_v3", pcsQ, PANCAKE_V3_FACTORY, USDT, PCS_V3_FEES)
    _add_v3_direct_exact_input("pancake_v3", pcsQ, PANCAKE_V3_FACTORY, USDC, PCS_V3_FEES)
    for mid in INTERMEDIATES:
        _add_v3_twohop_exact_input("pancake_v3", pcsQ, PANCAKE_V3_FACTORY, mid, USDT, PCS_V3_FEES, PCS_V3_FEES)
        _add_v3_twohop_exact_input("pancake_v3", pcsQ, PANCAKE_V3_FACTORY, mid, USDC, PCS_V3_FEES, PCS_V3_FEES)

    # Kyber Elastic (bps fees)
    _add_kyber_direct(USDT, KYBER_FEES_BPS)
    _add_kyber_direct(USDC, KYBER_FEES_BPS)
    for mid in INTERMEDIATES:
        _add_kyber_twohop(mid, USDT, KYBER_FEES_BPS, KYBER_FEES_BPS)
        _add_kyber_twohop(mid, USDC, KYBER_FEES_BPS, KYBER_FEES_BPS)

    # V2 baselines (Pancake + Biswap)
    v2_paths = [
        [base_token, USDT],
        [base_token, WBNB, USDT],
        [base_token, USDC, USDT],
        [base_token, USDC],
        [base_token, WBNB, USDC],
        [base_token, USDT, USDC],
    ]
    _add_v2_paths(pcsV2, PANCAKE_V2_ROUTER, "pancake_v2", v2_paths)
    _add_v2_paths(bisV2, BISWAP_V2_ROUTER,  "biswap_v2",  v2_paths)

    return calls, meta

def decode_mc_results(meta, results, verbose=False) -> Tuple[List[Quote], List[Dict[str,Any]], List[RawDump]]:
    """
    Robust Multicall3 result decoder with:
      - Correct meta/results alignment (skip entries do NOT consume results)
      - QuoterV2 revert-style payload parsing (dynamic uint256[]), even when ok==True
      - Graceful fallbacks and detailed failure logging (optionally includes target addr)

    Expected meta item shape (last element optional):
        (kind, venue, pdesc, outToken, amt_in[, target_addr])
    """
    quotes: List[Quote] = []
    failures: List[Dict[str,Any]] = []
    rawdumps: List[RawDump] = []

    # ---------- small helpers ----------
    def record_dump(venue: str, desc: str, ok: bool, rdat: bytes):
        rawdumps.append(RawDump(venue, desc, ok, len(rdat), "0x" + _dump_hex(rdat)))
        if (not ok) and verbose and rdat:
            print(f"           ↳ raw0x: {_dump_hex(rdat)}")

    def emit_failure(venue: str, pdesc: str, msg: str, rdat: bytes, target: Optional[str] = None):
        suffix = (f" via {target}" if target else "")
        failures.append({
            "venue": venue,
            "path": pdesc,
            "error": f"{msg}{suffix}",
            "raw_len": len(rdat),
            "raw_hex": "0x" + _dump_hex(rdat),
            "target": target or "",
        })
        if verbose:
            print(f"  [mc-fail] {venue:11s} | {pdesc:35s} | {msg}{suffix} (len={len(rdat)})")
        record_dump(venue, pdesc, False, rdat)

    def _unpack_meta(m):
        # (kind, venue, pdesc, outToken, amt_in[, target_addr])
        if len(m) >= 6:
            return m[0], m[1], m[2], m[3], m[4], m[5]
        else:
            return m[0], m[1], m[2], m[3], m[4], None

    def _manual_parse_uint256_array(data: bytes) -> Optional[List[int]]:
        """
        Forgiving parser for ABI-encoded uint256[]:
          * canonical: [0x20][len][items...]
          * tail-only : [len][items...]
        Returns list or None. Never raises.
        """
        try:
            if not data or len(data) < 32:
                return None
            # canonical [0x20][len]
            if len(data) >= 64 and int.from_bytes(data[0:32], 'big') == 32:
                n = int.from_bytes(data[32:64], 'big')
                need = 64 + 32 * n
                if n > 4096 or len(data) < need:
                    return None
                offs = 64
            else:
                # tail-only [len]
                n = int.from_bytes(data[0:32], 'big')
                need = 32 + 32 * n
                if n > 4096 or len(data) < need:
                    return None
                offs = 32
            out = []
            for i in range(n):
                out.append(int.from_bytes(data[offs + 32*i: offs + 32*(i+1)], 'big'))
            return out
        except Exception:
            return None

    def _try_decode_quoter_v2_like(kind: str, venue: str, amt_in: int, rdat: bytes) -> Optional[int]:
        """
        Handle Uniswap/Pancake V3 QuoterV2 revert-like payloads:
          - Single-hop: first 32 bytes often hold amountOut when revert-style is used
          - Multi-hop: dynamic uint256[]; prefer [amountIn, amountOut,...] -> amountOut
        """
        if not rdat:
            return None

        # Single-hop style quick path: treat the first word as amountOut if it looks sane
        if len(rdat) in (96, 128) and kind != "decode_v2":
            try:
                first = int.from_bytes(rdat[0:32], 'big')
                if first > 0:
                    return first
            except Exception:
                pass

        # Dynamic uint256[] (multi-hop)
        arr = _manual_parse_uint256_array(rdat)
        if arr:
            if len(arr) >= 2 and arr[0] == amt_in:
                return int(arr[1])
            return int(arr[-1])
        return None

    # ---------- main loop with correct alignment ----------
    ri = 0  # index in `results`
    for meta_item in meta:
        kind, venue, pdesc, outToken, amt_in, target_addr = _unpack_meta(meta_item)

        # Skip entries do NOT consume a results slot
        if kind == "skip":
            if verbose:
                print(f"  [skip] {venue:11s} {pdesc:35s}: {outToken}")
            continue

        if ri >= len(results):
            # Defensive: fewer results than non-skip meta
            emit_failure(venue, pdesc, "multicall result underflow (meta/results mismatch)", b"", target_addr)
            continue

        ok, rdat = results[ri]
        ri += 1

        try:
            # --------------------------
            # OK path
            # --------------------------
            if ok:
                parsed = False

                # Try the expected ABI first
                try:
                    if kind == "decode_v2":
                        amounts = abi_decode(['uint256[]'], rdat)[0]
                        amount_out = int(amounts[-1])
                        quotes.append(Quote(venue, pdesc, amt_in, amount_out, outToken, 0.0, GAS_PCS_V2_SWAP))
                        parsed = True

                    elif kind == "decode_single":
                        amount_out, _s, ticks, gasEst = abi_decode(['uint256','uint160','uint32','uint256'], rdat)
                        quotes.append(Quote(venue, pdesc, amt_in, int(amount_out), outToken, 0.0, int(gasEst), int(ticks)))
                        parsed = True

                    elif kind == "decode_path":
                        # Some nodes/adapters still surface revert-like arrays even with ok==True
                        # so we try tuple first, then fallback to the revert-like array parser.
                        try:
                            amount_out, _sL, ticksL, gasEst = abi_decode(['uint256','uint160[]','uint32[]','uint256'], rdat)
                            total_ticks = sum(int(x) for x in (ticksL or []))
                            quotes.append(Quote(venue, pdesc, amt_in, int(amount_out), outToken, 0.0, int(gasEst), total_ticks))
                            parsed = True
                        except Exception:
                            amt_out = _try_decode_quoter_v2_like(kind, venue, amt_in, rdat)
                            if amt_out is not None:
                                quotes.append(Quote(venue, pdesc, amt_in, int(amt_out), outToken, 0.0, None, None))
                                if verbose:
                                    via = f" via {target_addr}" if target_addr else ""
                                    print(f"  [mc] {venue:11s} | {pdesc:35s} | ok but revert-like decoded{via} -> amountOut={amt_out}")
                                parsed = True

                except Exception:
                    # We'll try the revert-like parser below
                    pass

                if not parsed:
                    # Last resort on ok path: revert-like dynamic uint256[]
                    amt_out = _try_decode_quoter_v2_like(kind, venue, amt_in, rdat)
                    if amt_out is not None:
                        quotes.append(Quote(venue, pdesc, amt_in, int(amt_out), outToken, 0.0, None, None))
                        if verbose:
                            via = f" via {target_addr}" if target_addr else ""
                            print(f"  [mc] {venue:11s} | {pdesc:35s} | ok decoded via uint256[] fallback{via} -> amountOut={amt_out}")
                        record_dump(venue, pdesc, True, rdat)
                        continue

                    # Give up on ok path
                    emit_failure(venue, pdesc, "decode (success path) failed", rdat, target_addr)
                else:
                    record_dump(venue, pdesc, True, rdat)

                continue  # done with ok path

            # --------------------------
            # REVERT path (ok == False)
            # --------------------------
            # 1) QuoterV2 revert-like first (multi-hop etc.)
            amt_out = _try_decode_quoter_v2_like(kind, venue, amt_in, rdat)
            if amt_out is not None:
                quotes.append(Quote(venue, pdesc, amt_in, int(amt_out), outToken, 0.0, None, None))
                if verbose:
                    via = f" via {target_addr}" if target_addr else ""
                    print(f"  [mc] {venue:11s} | {pdesc:35s} | decoded from revert(bytes){via} -> amountOut={amt_out}")
                record_dump(venue, pdesc, False, rdat)
                continue

            # 2) Try “as if” return tuple (some envs flag ok=False but still return data)
            try:
                if kind == "decode_v2":
                    arr = abi_decode(['uint256[]'], rdat)[0]
                    amount_out = int(arr[-1])
                    quotes.append(Quote(venue, pdesc, amt_in, amount_out, outToken, 0.0, GAS_PCS_V2_SWAP))
                    record_dump(venue, pdesc, False, rdat)
                    continue
                elif kind == "decode_single":
                    amount_out, _s, ticks, gasEst = abi_decode(['uint256','uint160','uint32','uint256'], rdat)
                    quotes.append(Quote(venue, pdesc, amt_in, int(amount_out), outToken, 0.0, int(gasEst), int(ticks)))
                    record_dump(venue, pdesc, False, rdat)
                    continue
                elif kind == "decode_path":
                    amount_out, _sL, ticksL, gasEst = abi_decode(['uint256','uint160[]','uint32[]','uint256'], rdat)
                    total_ticks = sum(int(x) for x in (ticksL or []))
                    quotes.append(Quote(venue, pdesc, amt_in, int(amount_out), outToken, 0.0, int(gasEst), total_ticks))
                    record_dump(venue, pdesc, False, rdat)
                    continue
            except Exception:
                pass

            # 3) Heuristic: dynamic uint256[] fallback
            arr = _manual_parse_uint256_array(rdat)
            if arr:
                if len(arr) >= 2 and arr[0] == amt_in:
                    amount_out = int(arr[1])
                else:
                    amount_out = int(arr[-1])
                quotes.append(Quote(venue, pdesc, amt_in, amount_out, outToken, 0.0, None, None))
                if verbose:
                    via = f" via {target_addr}" if target_addr else ""
                    print(f"  [mc] {venue:11s} | {pdesc:35s} | decoded via uint256[] fallback{via} (n={len(arr)})")
                record_dump(venue, pdesc, False, rdat)
                continue

            # 4) Human-friendly revert reason
            msg = decode_revert_text_or_none(rdat) or ("empty revert" if not rdat else "revert")
            emit_failure(venue, pdesc, msg, rdat, target_addr)

        except Exception as e:
            emit_failure(venue, pdesc, f"decode failed: {str(e)}", rdat, target_addr)

    # Optional: warn if results had extra entries (should not happen)
    if ri != len(results) and verbose:
        print(f"[warn] decode_mc_results consumed {ri}/{len(results)} results (meta/results mismatch?)")

    return quotes, failures, rawdumps

# ---------------------------
# Quoting core
# ---------------------------
def price_wbnb_in_usdt() -> float:
    d_wbnb = decimals_of(WBNB); d_usdt = decimals_of(USDT)
    try:
        out = v2_getAmountsOut(10**d_wbnb, [WBNB, USDT])[-1]
        return human(out, d_usdt)
    except Exception:
        return 0.0

def price_in_usdt(token_in: str) -> float:
    d_in = decimals_of(token_in); d_usdt = decimals_of(USDT)
    try:
        out = v2_getAmountsOut(10**d_in, [token_in, USDT])[-1]
        return human(out, d_usdt)
    except Exception:
        return 0.0

def to_usdt_equiv_ex(
    token_addr: str,
    amount: int,
    *,
    ctx: Optional[PinnedContext] = None,
    block_number: Optional[int] = None,
    verbose: bool = False,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[float, str]:
    """
    Returns (usdt_equiv, note). Records *how* we converted.

    Paths:
      - USDT: exact.
      - USDC: Wombat USDC->USDT (pinned if ctx or block_number); on failure, 1:1 fallback (explicit).
      - other: raw decimal scaling only (explicit).
    """
    def _emit(msg: str):
        if log:
            try: log(msg)
            except Exception: pass
        if verbose:
            print(msg)

    try:
        if token_addr.lower() == USDT.lower():
            val  = human(amount, decimals_of(USDT))
            note = f"direct: token is USDT (exact) → {val:.6f} USDT"
            _emit("[to_usdt_equiv] " + note)
            return val, note

        if token_addr.lower() == USDC.lower():
            try:
                out  = wombat_router_amount_out(USDC, USDT, int(amount), ctx=ctx, block_number=block_number)
                val  = human(int(out), decimals_of(USDT))
                note = f"wombat: USDC→USDT on-chain quote ok (blk={block_number if block_number is not None else 'latest'}) → {val:.6f} USDT"
                _emit("[to_usdt_equiv] " + note)
                return val, note
            except Exception as e:
                val  = human(amount, decimals_of(USDC))
                note = f"fallback: wombat failed ({str(e) or 'error'}) — assumed USDC≈USDT 1:1 → {val:.6f} USDT"
                _emit("[to_usdt_equiv] " + note)
                return val, note

        val  = amount / (10 ** decimals_of(token_addr))
        note = (f"raw-scale: {symbol_of(token_addr)} not USDT/USDC — scaled by decimals only → {val:.6f} "
                f"(NOT a stable conversion)")
        _emit("[to_usdt_equiv] " + note)
        return val, note

    except Exception as e:
        # ultimate safety net
        try:
            val = amount / (10 ** decimals_of(token_addr))
        except Exception:
            val = float(amount)
        note = f"error: conversion crashed ({str(e) or 'error'}), returned raw-scale {val:.6f}"
        _emit("[to_usdt_equiv] " + note)
        return val, note

def to_usdt_equiv(
    token_addr: str,
    amount: int,
    *,
    ctx: Optional[PinnedContext] = None,
    block_number: Optional[int] = None,
    verbose: bool = False,
    log: Optional[Callable[[str], None]] = None,
) -> float:
    """Backward-compatible wrapper (same name as before)."""
    val, _ = to_usdt_equiv_ex(
        token_addr, amount, ctx=ctx, block_number=block_number, verbose=verbose, log=log
    )
    return val

def gas_cost_usdt(gas_units: int, wbnb_usdt: float) -> float:
    if not gas_units or wbnb_usdt <= 0:
        return 0.0
    gp = w3.eth.gas_price
    cost_bnb = gas_units * gp / 1e18
    return cost_bnb * wbnb_usdt

def find_amount_per_slice_for_target_usdt(base_token: str, target_usdt: float, *, verbose: bool=False) -> Optional[int]:
    """
    Hardened sizer:
      - tries multiple V2 paths and chooses the MINIMUM input
      - applies a price sanity check vs a reference (via WBNB or via USDC)
      - logs all candidates when verbose=True
    Returns amount_in (wei) or None.
    """
    d_in   = decimals_of(base_token)
    d_usdt = decimals_of(USDT)
    target_out = int(target_usdt * (10 ** d_usdt))

    candidates = []
    paths = [
        [base_token, USDT],
        [base_token, WBNB, USDT],
        [base_token, USDC, USDT],
    ]

    # 1) Collect all candidates that succeed
    for path in paths:
        try:
            amt_in = v2_getAmountsIn(target_out, path)[0]
            # implied price from this candidate (USDT per 1 base token)
            price_usdt_per_base = (target_usdt) / (amt_in / (10 ** d_in))
            candidates.append({
                "path": path,
                "amount_in": int(amt_in),
                "price": float(price_usdt_per_base),
            })
        except Exception:
            if verbose:
                print(f"[size]   path {'->'.join(symbol_of(x) for x in path)} failed")
            continue

    if not candidates:
        return None

    # 2) Build a reference price (prefer via WBNB, then via USDC, else median of candidates)
    ref_price = 0.0
    # try base->WBNB->USDT as ref
    try:
        one_base = 10 ** d_in
        ref_out  = v2_getAmountsOut(one_base, [base_token, WBNB, USDT])[-1]
        ref_price = human(ref_out, d_usdt)
        if verbose:
            print(f"[size] ref via WBNB: ~{ref_price:.6f} USDT/{symbol_of(base_token)}")
    except Exception:
        # try base->USDC->USDT as ref
        try:
            one_base = 10 ** d_in
            ref_out  = v2_getAmountsOut(one_base, [base_token, USDC, USDT])[-1]
            ref_price = human(ref_out, d_usdt)
            if verbose:
                print(f"[size] ref via USDC: ~{ref_price:.6f} USDT/{symbol_of(base_token)}")
        except Exception:
            # use median of candidate prices as last resort
            vals = sorted(c["price"] for c in candidates if c["price"] > 0)
            if vals:
                m = vals[len(vals)//2]
                ref_price = float(m)
                if verbose:
                    print(f"[size] ref via median(candidates): ~{ref_price:.6f} USDT/{symbol_of(base_token)}")

    # 3) Sanity filter: drop outliers w.r.t. reference
    TOL = 0.50  # accept within ±50%
    filtered = []
    for c in candidates:
        ok = True
        if ref_price > 0 and c["price"] > 0:
            if abs(c["price"] - ref_price) / ref_price > TOL:
                ok = False
        if verbose:
            p = "->".join(symbol_of(x) for x in c["path"])
            flag = "OK " if ok else "DROP"
            print(f"[size] {flag}  {p:30s} needs {human(c['amount_in'], d_in):.8f} {symbol_of(base_token)}  "
                  f"(implied ~{c['price']:.6f} vs ref ~{ref_price:.6f} USDT/{symbol_of(base_token)})")
        if ok:
            filtered.append(c)

    chosen_set = filtered if filtered else candidates
    best = min(chosen_set, key=lambda c: c["amount_in"])

    if verbose and filtered != candidates:
        print(f"[size] filtered {len(candidates)-len(filtered)} outlier path(s)")
    if verbose:
        print(f"[size] chosen path: {'->'.join(symbol_of(x) for x in best['path'])}  "
              f"amount_in ~ {human(best['amount_in'], d_in):.8f} {symbol_of(base_token)}")

    return int(best["amount_in"])

def quote_thena_single(
    base_token: str,
    out_token: str,
    amount_in: int,
    verbose: bool,
    *,
    block_number: int | None = None
) -> tuple[Optional[Quote], Optional[Dict[str, Any]]]:

    def _fmt_sym(x: str) -> str:
        try:
            return symbol_of(x)
        except Exception:
            return x

    ok, reason, pool = thena_pool_status(base_token, out_token, block_number=block_number)
    if not ok:
        if verbose:
            extra = f" @ {pool}" if pool else ""
            print(f"  [skip] THENA        {_fmt_sym(base_token)}->{_fmt_sym(out_token)} (single): {reason}{extra}")
        return None, {
            "venue": "thena_algebra_v1",
            "path": f"{_fmt_sym(base_token)}->{_fmt_sym(out_token)} (single)",
            "error": reason,
            "factory": (THENA_FACTORY or ""),
            "pool": (pool or ""),
            "target": THENA_QUOTER,
            "amount_in": int(amount_in),
            "token_in": base_token,
            "token_out": out_token,
        }

    try:
        t0 = time.perf_counter()
        quoter = w3.eth.contract(address=THENA_QUOTER, abi=THENA_QUOTER_ABI)
        fn = quoter.functions.quoteExactInputSingle((base_token, out_token, int(amount_in), 0))
        amount_out, _sqrt_after, ticks_crossed, gas_est = fn.call(block_identifier=block_number) if block_number is not None else fn.call()
        ms = (time.perf_counter() - t0) * 1000.0

        q = Quote(
            venue="thena_algebra_v1",
            path=f"{_fmt_sym(base_token)}->{_fmt_sym(out_token)} (single)",
            amount_in=int(amount_in),
            amount_out=int(amount_out),
            out_token=out_token,
            ms=ms,
            gas_estimate=int(gas_est),
            ticks_crossed=int(ticks_crossed),
        )

        if verbose:
            print(
                f"  [ok]  THENA        {_fmt_sym(base_token)}->{_fmt_sym(out_token)} (single) -> "
                f"{human(amount_out, decimals_of(out_token)):.6f} {_fmt_sym(out_token)} "
                f"({q.ms:.1f} ms; gas≈{q.gas_estimate}, ticks={q.ticks_crossed})"
            )
        return q, None

    except Exception as e:
        msg, raw_hex = extract_revert_info(e)
        if verbose:
            print(f"  [fail] THENA       (single) :: {msg}")
            if raw_hex:
                print(f"           ↳ raw0x: {raw_hex[2:]}")
        fail = {
            "venue": "thena_algebra_v1",
            "path": f"{_fmt_sym(base_token)}->{_fmt_sym(out_token)} (single)",
            "error": msg,
            "raw_hex": raw_hex,
            "factory": (THENA_FACTORY or ""),
            "pool": pool or "",
            "target": THENA_QUOTER,
            "amount_in": int(amount_in),
            "token_in": base_token,
            "token_out": out_token,
        }
        return None, fail

def quote_thena_via_wbnb(
    base_token: str,
    out_token: str,
    amount_in: int,
    verbose: bool,
    *,
    block_number: int | None = None
) -> tuple[Optional[Quote], Optional[Dict[str, Any]]]:

    def _fmt(x: str) -> str:
        try:
            return symbol_of(x)
        except Exception:
            return x

    ok1, r1, pool1 = thena_pool_status(base_token, WBNB, block_number=block_number)
    ok2, r2, pool2 = thena_pool_status(WBNB, out_token, block_number=block_number)
    if not (ok1 and ok2):
        reason = " / ".join([x for x in [r1 if not ok1 else None, r2 if not ok2 else None] if x]) or "no pool(s)"
        if verbose:
            p1 = f" @{pool1}" if pool1 else ""
            p2 = f" @{pool2}" if pool2 else ""
            print(f"  [skip] THENA        {_fmt(base_token)}->WBNB->{_fmt(out_token)}: {reason}{p1}{p2}")
        return None, {
            "venue": "thena_algebra_v1_via_wbnb",
            "path": f"{_fmt(base_token)}->WBNB->{_fmt(out_token)}",
            "error": reason,
            "factory": THENA_FACTORY,
            "pool_leg1": pool1 or "",
            "pool_leg2": pool2 or "",
            "amount_in": int(amount_in),
            "token_in": base_token,
            "token_out": out_token,
        }

    qtr = thena_quoter()
    try:
        t0 = time.perf_counter()
        fn1 = qtr.functions.quoteExactInputSingle((base_token, WBNB, int(amount_in), 0))
        out1, _s1, ticks1, gas1 = fn1.call(block_identifier=block_number) if block_number is not None else fn1.call()
        if int(out1) <= 0:
            msg = "zero out on leg1 (base->WBNB)"
            if verbose:
                print(f"  [fail] THENA        via WBNB :: {msg}")
            return None, {
                "venue": "thena_algebra_v1_via_wbnb",
                "path": f"{_fmt(base_token)}->WBNB->{_fmt(out_token)}",
                "error": msg,
                "pool_leg1": pool1 or "",
                "pool_leg2": pool2 or "",
                "amount_in": int(amount_in),
            }

        fn2 = qtr.functions.quoteExactInputSingle((WBNB, out_token, int(out1), 0))
        out2, _s2, ticks2, gas2 = fn2.call(block_identifier=block_number) if block_number is not None else fn2.call()
        ms = (time.perf_counter() - t0) * 1000.0

        q = Quote(
            venue="thena_algebra_v1_via_wbnb",
            path=f"{_fmt(base_token)}->WBNB->{_fmt(out_token)}",
            amount_in=int(amount_in),
            amount_out=int(out2),
            out_token=out_token,
            ms=ms,
            gas_estimate=int(gas1) + int(gas2),
            ticks_crossed=int(ticks1) + int(ticks2),
        )

        if verbose:
            print(
                f"  [ok]  THENA        via WBNB {_fmt(base_token)}->WBNB->{_fmt(out_token)} -> "
                f"{human(out2, decimals_of(out_token)):.6f} {_fmt(out_token)} "
                f"({q.ms:.1f} ms; gas≈{q.gas_estimate}, ticks={q.ticks_crossed})"
            )
        return q, None

    except Exception as e:
        msg, raw_hex = extract_revert_info(e)
        if verbose:
            print(f"  [fail] THENA        via WBNB :: {msg}")
            if raw_hex:
                print(f"           ↳ raw0x: {raw_hex[2:]}")
        return None, {
            "venue": "thena_algebra_v1_via_wbnb",
            "path": f"{_fmt(base_token)}->WBNB->{_fmt(out_token)}",
            "error": msg,
            "raw_hex": raw_hex,
            "factory": THENA_FACTORY,
            "pool_leg1": pool1 or "",
            "pool_leg2": pool2 or "",
            "amount_in": int(amount_in),
            "token_in": base_token,
            "token_out": out_token,
        }

def find_amount_per_slice_for_target_usdt_robust(token_in: str, target_usd: float, *, verbose: bool=False) -> Optional[int]:
    """
    Robust, *deterministic* slice sizer that avoids the broken doubling-probe.
    Strategy:
      - Sample small exact-input quotes across PCS v2, PCS v3, Uniswap v3, THENA
      - Convert to USDT/token with correct decimals
      - Filter broken/absurd readings
      - Take median price and solve amount_in for target_usd
    Returns amount_in in wei (int) or None if no viable route.
    """
    try:
        d_in   = decimals_of(token_in)
        d_usdt = decimals_of(USDT)
    except Exception:
        return None

    def human_amt(x_wei: int, d: int) -> float:
        try:
            return x_wei / (10 ** d)
        except Exception:
            return 0.0

    # --- collect price candidates -------------------------------------------------
    candidates: list[tuple[str, float]] = []

    def add(src: str, amt_in_wei: int, amt_out_wei: int):
        if amt_in_wei <= 0 or amt_out_wei <= 0:
            return
        price = human_amt(amt_out_wei, d_usdt) / human_amt(amt_in_wei, d_in)
        if price > 0:
            candidates.append((src, price))

    def _looks_flat_output(token_in: str, path: list[str]) -> bool:
        d_in = decimals_of(token_in)
        small = max(1, 10 ** d_in // 10_000)  # ~0.0001 token
        big   = max(1, 10 ** d_in // 100)     # ~0.01 token
        try:
            out_s = v2_getAmountsOut(small, path)[-1]
            out_b = v2_getAmountsOut(big,   path)[-1]
            # If out doesn't grow ~proportionally when input grows 100×, treat as broken/flat.
            return out_s == 0 or out_b / max(1, out_s) < 2
        except Exception:
            return True

    def probe_v2(path: list[str], label: str):
        if _looks_flat_output(token_in, path):
            if verbose:
                print(f"  [size] probe {label:26s} skipped: flat/broken output")
            return        
        # Use a small but non-trivial probe (0.001 token) to mitigate zero-rounding
        probe_in = max(1, 10 ** d_in // 1000)
        try:
            amounts = v2_getAmountsOut(probe_in, path)
            add(label, probe_in, int(amounts[-1]))
            if verbose:
                print(f"  [size] probe {label:26s} price≈{candidates[-1][1]:.6f} USDT/{symbol_of(token_in)}")
        except Exception as e:
            if verbose:
                print(f"  [size] probe {label:26s} failed: {str(e).splitlines()[0]}")

    # PCS v2 paths (direct + two hops)
    probe_v2([token_in, USDT],            "pcs_v2:direct")
    probe_v2([token_in, WBNB, USDT],      "pcs_v2:via_WBNB")
    probe_v2([token_in, USDC, USDT],      "pcs_v2:via_USDC")

    # V3 quoters (PCS v3 & Uniswap v3), single + via WBNB
    for venue, quoter_addr, factory_addr, fees in (
        ("pcs_v3",  PANCAKE_V3_QUOTER_V2,  PANCAKE_V3_FACTORY,  PCS_V3_FEES),
        ("uni_v3",  UNISWAP_V3_QUOTER_V2,  UNISWAP_V3_FACTORY,  UNI_V3_FEES),
    ):
        quoter = v3_quoter(quoter_addr)
        probe_in = max(1, 10 ** d_in // 1000)

        # single hop token_in -> USDT
        for fee in fees:
            ok, reason = v3_status2(factory_addr, token_in, USDT, fee)
            if not ok:
                if verbose:
                    print(f"  [size] probe {venue}:single fee {fee:<5} skipped: {reason}")
                continue
            try:
                out, _s, _ticks, _gas = quoter.functions.quoteExactInputSingle(
                    (token_in, USDT, int(fee), int(probe_in), 0)
                ).call()
                add(f"{venue}:single fee{fee}", probe_in, int(out))
                if verbose:
                    print(f"  [size] probe {venue}:single fee {fee:<5} price≈{candidates[-1][1]:.6f}")
            except Exception as e:
                if verbose:
                    print(f"  [size] probe {venue}:single fee {fee:<5} failed: {str(e).splitlines()[0]}")

        # via WBNB (two hops) token_in -> WBNB -> USDT
        for f1 in fees:
            for f2 in fees:
                ok1, r1 = v3_status2(factory_addr, token_in, WBNB, f1)
                ok2, r2 = v3_status2(factory_addr, WBNB, USDT, f2)
                if not (ok1 and ok2):
                    continue
                try:
                    path = encode_v3_path([token_in, WBNB, USDT], [f1, f2])
                    out, _sL, _tL, _gas = quoter.functions.quoteExactInput(path, int(probe_in)).call()
                    add(f"{venue}:via_WBNB {f1}/{f2}", probe_in, int(out))
                except Exception:
                    pass

    # THENA (Algebra v1.0) single + via WBNB
    try:
        qtr = thena_quoter()
        # single hop
        ok, reason, _pool = thena_pool_status(token_in, USDT)
        if ok:
            probe_in = max(1, 10 ** d_in // 1000)
            try:
                out, _s, _ticks, _gas = qtr.functions.quoteExactInputSingle(
                    (token_in, USDT, int(probe_in), 0)
                ).call()
                add("thena:single", probe_in, int(out))
                if verbose:
                    print(f"  [size] probe thena:single             price≈{candidates[-1][1]:.6f}")
            except Exception:
                pass
        # via WBNB
        ok1, r1, _p1 = thena_pool_status(token_in, WBNB)
        ok2, r2, _p2 = thena_pool_status(WBNB, USDT)
        if ok1 and ok2:
            probe_in = max(1, 10 ** d_in // 1000)
            try:
                out1, *_ = qtr.functions.quoteExactInputSingle((token_in, WBNB, int(probe_in), 0)).call()
                if int(out1) > 0:
                    out2, *_ = qtr.functions.quoteExactInputSingle((WBNB, USDT, int(out1), 0)).call()
                    add("thena:via_WBNB", probe_in, int(out2))
            except Exception:
                pass
    except Exception:
        pass

    if not candidates:
        if verbose:
            print("  [size] no viable price candidates")
        return None

    # --- filter broken/absurd candidates --------------------------------------
    prices = [p for _, p in candidates]
    prices.sort()
    median = prices[len(prices)//2]

    # remove extreme outliers (±10× median)
    filtered = [(src, p) for (src, p) in candidates if (0.1 * median) <= p <= (10.0 * median)]
    if filtered:
        candidates = filtered

    if verbose:
        print("  [size] price candidates (USDT per token):")
        for src, p in sorted(candidates, key=lambda x: x[1], reverse=True):
            print(f"    {src:26s} -> {p:.6f}")

    # choose median of filtered
    prices = [p for _, p in candidates]
    prices.sort()
    chosen = prices[len(prices)//2]

    if chosen <= 0:
        if verbose:
            print("  [size] all candidates invalid (<=0)")
        return None

    # --- solve amount_in for target_usd ---------------------------------------
    amount_in_float = target_usd / chosen  # tokens
    amount_in_wei   = int(amount_in_float * (10 ** d_in))
    amount_in_wei   = max(1, amount_in_wei)

    if verbose:
        print(f"  [size] chosen price ≈ {chosen:.6f} USDT/{symbol_of(token_in)}")
        print(f"  [size] robust result: ~{human_amt(amount_in_wei, d_in):.8f} {symbol_of(token_in)} (≈ ${target_usd:,.2f})")

    return amount_in_wei

def quote_all(
    base_token: str,
    amount_in: int,
    *,
    gas_adjusted: bool,
    use_multicall: bool,
    enable_dodo: bool,
    dodo_pools_cli: List[str],
    discovered_pools: List[str],
    verbose: bool,
    expand_v3: str = "missing",
    snap: Optional[Dict[str, int]] = None,
    slice_idx: Optional[int] = None
):
    failures: List[Dict[str, Any]] = []
    quotes: List[Quote] = []
    rawdumps: List[RawDump] = []

    # Snapshot (if not provided)
    if snap is None:
        snap = get_block_snapshot()
    blkno = snap["number"]

    # ---- helpers for emitting + caching USDT-equivalents --------------------
    usdt_equiv_cache: Dict[Tuple[str, int, int], float] = {}

    def _equiv(q: Quote) -> float:
        key = (q.out_token.lower(), int(q.amount_out), int(blkno))
        if key in usdt_equiv_cache:
            return usdt_equiv_cache[key]
        val, _note = to_usdt_equiv_ex(
            q.out_token, q.amount_out, block_number=blkno, verbose=verbose
        )
        usdt_equiv_cache[key] = val
        return val

    def _emit_with_equiv(q: Quote, *, source: str, extra: Optional[Dict[str, Any]] = None):
        val, note = to_usdt_equiv_ex(
            q.out_token, q.amount_out, block_number=blkno, verbose=verbose
        )
        usdt_equiv_cache[(q.out_token.lower(), int(q.amount_out), int(blkno))] = val
        meta = dict(extra or {})
        meta["usdt_equiv_note"] = note
        emit_quote_jsonl(q, snap=snap, usdt_equiv=val, source=source, extra=meta)

    # --- Multicall batch first (pinned to snapshot block) ---
    if use_multicall:
        calls, meta = build_mc_calls(amount_in, base_token)
        if verbose:
            tried = sum(1 for m in meta if m[0] != "skip")
            skipped = sum(1 for m in meta if m[0] == "skip")
            print(f"  [mc] dispatching {tried} view calls via Multicall3... (skipped {skipped}) at block #{blkno}")
        t0 = time.perf_counter()
        raw = multicall_try_aggregate_at([(addr, data) for (addr, data) in calls], block_number=blkno)

        dur = (time.perf_counter() - t0) * 1000.0
        mc_quotes, mc_failures, dumps = decode_mc_results(meta, raw, verbose=verbose)
        for q in mc_quotes:
            q.ms = dur  # batch latency (approx)
            _emit_with_equiv(q, source="multicall", extra={"slice": slice_idx})
        quotes.extend(mc_quotes)
        failures.extend(mc_failures)
        rawdumps.extend(dumps)
        if verbose:
            print(f"  [mc] done in {dur:.1f} ms, {len(mc_quotes)} successful, {len(mc_failures)} failed")

    # --- Pancake v2 baselines (pinned to snapshot) ---
    for path in ([base_token, USDT], [base_token, WBNB, USDT], [base_token, USDC, USDT]):
        pdesc = "->".join(symbol_of(x) for x in path)
        try:
            t0 = time.perf_counter()
            amounts = v2_getAmountsOut(amount_in, path, block_number=blkno)
            ms = (time.perf_counter() - t0) * 1000.0
            amount_out = int(amounts[-1])
            out_tok = path[-1]
            q = Quote(
                venue="pancake_v2",
                path=pdesc,
                amount_in=int(amount_in),
                amount_out=amount_out,
                out_token=out_tok,
                ms=ms,
                gas_estimate=GAS_PCS_V2_SWAP,
                ticks_crossed=None,
            )
            quotes.append(q)
            _emit_with_equiv(q, source="direct", extra={"slice": slice_idx})
        except Exception as e:
            msg, raw_hex = extract_revert_info(e)
            failures.append({
                "venue": "pancake_v2",
                "path": pdesc,
                "error": msg,
                "raw_hex": raw_hex,
                "amount_in": int(amount_in),
                "target": PANCAKE_V2_ROUTER,
            })

    # --- THENA (single & via WBNB) pinned to snapshot ---
    for out_token in (USDT, USDC):
        q, fail = quote_thena_single(base_token, out_token, amount_in, verbose, block_number=blkno)
        if q is not None:
            quotes.append(q)
            _emit_with_equiv(q, source="direct:thena", extra={"slice": slice_idx})
        if fail is not None:
            failures.append(fail)

    for out_token in (USDT, USDC):
        q2, fail2 = quote_thena_via_wbnb(base_token, out_token, amount_in, verbose, block_number=blkno)
        if q2 is not None:
            quotes.append(q2)
            _emit_with_equiv(q2, source="direct:thena_via_wbnb", extra={"slice": slice_idx})
        if fail2 is not None:
            failures.append(fail2)

    # --- V3 expansion via direct quoter calls (single & via WBNB) ---
    have_uni_any = any(q.venue == "uniswap_v3" for q in quotes)
    have_pcs_any = any(q.venue == "pancake_v3" for q in quotes)
    _expand_mode = (expand_v3 or "missing").lower()
    if _expand_mode not in ("never", "missing", "always"):
        _expand_mode = "missing"

    def _should_expand(venue: str) -> bool:
        if _expand_mode == "never":
            return False
        if _expand_mode == "always":
            return True
        return (not have_uni_any) if venue == "uniswap_v3" else (not have_pcs_any)

    def _already_have(venue_label: str, desc_label: str, token_out: str) -> bool:
        for q in quotes:
            if q.venue == venue_label and q.path == desc_label and q.out_token.lower() == token_out.lower():
                return True
        return False

    def _expand_v3(venue_label: str, quoter_addr: str, factory_addr: str, fees: list[int]) -> int:
        if not _should_expand(venue_label):
            return 0
        mode = _expand_mode
        if verbose:
            print(f"  [expand:{mode}] {venue_label}: direct quoter calls (single-hop & via WBNB) at block #{blkno}")

        added = 0

        def _emit(desc_label: str, amount_out: int, ticks: int, gasEst: int, target_out: str, t0: float):
            nonlocal added
            if _already_have(venue_label, desc_label, target_out):
                return
            ms = (time.perf_counter() - t0) * 1000.0
            q = Quote(venue_label, desc_label, amount_in, int(amount_out), target_out, ms, int(gasEst), int(ticks))
            quotes.append(q)
            added += 1
            _emit_with_equiv(q, source=f"direct:{venue_label}", extra={"slice": slice_idx})
            if verbose:
                ue = _equiv(q)
                print(
                    f"  [ok]  {venue_label:11s} {desc_label:45s} -> "
                    f"{human(amount_out, decimals_of(target_out)):.6f} {symbol_of(target_out)} "
                    f"(ticks={int(ticks)}, gas≈{int(gasEst)}, {ms:.1f} ms) | ~{ue:.6f} USDT_eq"
                )

        # single-hop
        for target_out in (USDT, USDC):
            for fee in fees:
                desc_single = f"{symbol_of(base_token)}->{symbol_of(target_out)} (fee {fee})"
                pool_addr = v3_pool_address(factory_addr, base_token, target_out, fee, block_number=blkno)
                ok, reason = v3_status2(factory_addr, base_token, target_out, fee, block_number=blkno)
                if not ok:
                    if verbose:
                        pa = f" @ {pool_addr}" if pool_addr else ""
                        print(f"  [skip] {venue_label:11s} {desc_single}{pa}: {reason}")
                    continue
                if verbose:
                    liq_note = ""
                    try:
                        liq_ok = v3_pool_has_liquidity(pool_addr, block_number=blkno) if pool_addr else False
                        liq_note = f"; liquidity={'yes' if liq_ok else 'no/unknown'}"
                    except Exception:
                        pass
                    ain_h = f"{human(amount_in, decimals_of(base_token)):.8f} {symbol_of(base_token)}"
                    print(f"  [try] {venue_label:11s} {desc_single:45s} @ {pool_addr}{liq_note} | amount_in={ain_h}")
                try:
                    t0 = time.perf_counter()
                    amount_out, _s, ticks, gasEst = v3_quote_single(quoter_addr, base_token, target_out, amount_in, fee, block_number=blkno)
                    _emit(desc_single, amount_out, ticks, gasEst, target_out, t0)
                except Exception as e:
                    if verbose:
                        pa = f" @ {pool_addr}" if pool_addr else ""
                        print(f"  [fail] {venue_label:11s} {desc_single}{pa}: {e}")
                    failures.append({"venue": venue_label, "path": desc_single, "error": str(e)})

        # two-hop via WBNB
        for target_out in (USDT, USDC):
            for f1 in fees:
                for f2 in fees:
                    desc_twohop = f"{symbol_of(base_token)}->WBNB->{symbol_of(target_out)} ({f1}/{f2})"
                    ok1, r1 = v3_status2(factory_addr, base_token, WBNB, f1, block_number=blkno)
                    ok2, r2 = v3_status2(factory_addr, WBNB, target_out, f2, block_number=blkno)
                    if not (ok1 and ok2):
                        if verbose:
                            reason = " / ".join([x for x in ([r1 if not ok1 else None, r2 if not ok2 else None]) if x]) or "no pool(s)"
                            print(f"  [skip] {venue_label:11s} {desc_twohop:45s}: {reason}")
                        continue
                    if verbose:
                        print(f"  [try] {venue_label:11s} {desc_twohop:45s}")
                    try:
                        t0 = time.perf_counter()
                        amount_out, _sL, _ticksL, gasEst = v3_quote_path(quoter_addr, [base_token, WBNB, target_out], [f1, f2], amount_in, block_number=blkno)
                        _emit(desc_twohop, amount_out, 0, gasEst, target_out, t0)
                    except Exception as e:
                        if verbose:
                            print(f"  [fail] {venue_label:11s} {desc_twohop:45s}: {e}")
                        failures.append({"venue": venue_label, "path": desc_twohop, "error": str(e)})
        return added

    _expand_v3("uniswap_v3", UNISWAP_V3_QUOTER_V2, UNISWAP_V3_FACTORY, UNI_V3_FEES)
    _expand_v3("pancake_v3",  PANCAKE_V3_QUOTER_V2, PANCAKE_V3_FACTORY,  PCS_V3_FEES)

    # --- Hybrid: best(USDT) + Wombat(USDT->USDC) at snapshot ---
    try:
        to_usdt_quotes = [q for q in quotes if q.out_token.lower() == USDT.lower()]
        if to_usdt_quotes:
            best_to_usdt = max(to_usdt_quotes, key=lambda q: q.amount_out)
            usdc_out = wombat_router_amount_out(USDT, USDC, best_to_usdt.amount_out, block_number=blkno)
            q_combo = Quote(
                "combo_best_then_wombat",
                f"{best_to_usdt.path} -> (WombatRouter) USDT->USDC",
                amount_in,
                int(usdc_out),
                USDC,
                0.0,
                GAS_WOMBAT_SWAP
            )
            quotes.append(q_combo)
            _emit_with_equiv(q_combo, source="combo", extra={"slice": slice_idx})
            if verbose:
                print(f"  [ok]  COMBO        best(USDT)+Wombat   -> {human(usdc_out, decimals_of(USDC)):.6f} USDC")
    except Exception as e:
        failures.append({"venue": "combo_best_then_wombat", "path": "best(USDT) + WombatRouter", "error": str(e)})

    # --- DODO (BSC): helper + factories + direct quotes ---
    if enable_dodo:
        total_q = 0
        total_fails = []

        # 1) Helper (optional, may revert with KJUDGE_ERROR; we catch & continue)
        t0 = time.perf_counter()
        helper_quotes, helper_fails = dodo_quote_helper(base_token, USDT, amount_in, verbose=verbose)
        for (poolAddr, outAmt) in helper_quotes:
            q = Quote("dodo_v2_helper", f"{symbol_of(base_token)}->USDT via {poolAddr[-6:]}", amount_in, int(outAmt), USDT, 0.0, GAS_DODO_QUERY)
            quotes.append(q)
            _emit_with_equiv(q, source="dodo_helper", extra={"slice": slice_idx})
        if verbose:
            print(f"  [dodo] helper pools used: {len(helper_quotes)} in {(time.perf_counter()-t0)*1000:.1f} ms")
        total_q += len(helper_quotes)
        total_fails.extend(helper_fails)

        # 2) Discover via factories, then quote directly
        discovered = dodo_discover_pools_via_factories(base_token, USDT, verbose=verbose)
        if discovered:
            t1 = time.perf_counter()
            direct_quotes, view_fails = dodo_quote_direct_pools(base_token, amount_in, discovered, verbose=verbose)
            for (addr, out_amt) in direct_quotes:
                q = Quote("dodo_v2_view", f"{symbol_of(base_token)}->USDT via {addr[-6:]}", amount_in, int(out_amt), USDT, 0.0, GAS_DODO_QUERY)
                quotes.append(q)
                _emit_with_equiv(q, source="dodo_view", extra={"slice": slice_idx})
            if verbose:
                print(f"  [dodo] factory-view pools used: {len(direct_quotes)} in {(time.perf_counter()-t1)*1000:.1f} ms")
            total_q += len(direct_quotes)
            total_fails.extend(view_fails)
        elif verbose:
            print("  [dodo] factory-view discovery empty")

        # (Optional) persist or print failures for diagnostics
        for f in total_fails:
            if verbose:
                print(f"  [dodo-fail] pool {f['pool'][-6:]} {f['kind']}: {f['error']}")


    # --- Ranking (gas-adjusted optional) ---
    if gas_adjusted:
        wbnb_px = price_wbnb_in_usdt()
        def net(q: Quote) -> float:
            try:
                return _equiv(q) - gas_cost_usdt(q.gas_estimate or 0, wbnb_px)
            except Exception:
                return float("-inf")
        quotes_sorted = sorted(quotes, key=net, reverse=True)
    else:
        quotes_sorted = sorted(quotes, key=_equiv, reverse=True)

    return quotes_sorted, failures, rawdumps

def log_jsonl(path, obj):
    """Append one JSON object as a line to `path` (creates dirs/files as needed)."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    except Exception:
        pass
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, separators=(",", ":")) + "\n")
    
def emit_quote_jsonl(q: Quote, *, snap: dict, usdt_equiv: float | None, source: str, extra: dict | None = None):
    rec = {
        "block_number": snap.get("number"),
        "block_timestamp": snap.get("timestamp"),
        "rpc_url": ACTIVE_RPC_URL,
        "venue": q.venue,
        "path": q.path,
        "amount_in": str(q.amount_in),
        "amount_out": str(q.amount_out),
        "out_token": symbol_of(q.out_token),
        "latency_ms": round(q.ms, 2),
        "gas_estimate": q.gas_estimate if q.gas_estimate is not None else None,
        "ticks_crossed": q.ticks_crossed if q.ticks_crossed is not None else None,
        "usdt_equiv": usdt_equiv,
        "source": source,
    }
    if extra:
        rec.update(extra)
    log_jsonl("quotes.jsonl", rec)
# ---------------------------
# CLI
# ---------------------------
def parse_addr_list(csv_str: str) -> Tuple[List[str], List[str]]:
    addrs, bad = [], []
    for item in (csv_str or "").split(","):
        s = item.strip()
        if not s: continue
        try:
            addrs.append(CS(s))
        except Exception:
            bad.append(s)
    return addrs, bad

def parse_cli() -> argparse.Namespace:
    """
    Centralized CLI parsing (kept 100% backward-compatible with your flags).
    Adds a couple of output paths so you can redirect CSV/JSONL without editing code.
    """
    ap = argparse.ArgumentParser(description="BNB Chain on-chain liquidation route benchmark (no off-chain APIs).")
    ap.add_argument("--rpc", type=str, default=DEFAULT_BSC_RPC, help="RPC URL (default: env BSC_RPC or public dataseed)")
    ap.add_argument("--total-usd", type=float, default=TARGET_TOTAL_USD)
    ap.add_argument("--slice-usd", type=float, default=SLICE_USD)
    ap.add_argument("--gas-adjusted", type=lambda x: str(x).lower() in ["1","true","yes"], default=False)
    ap.add_argument("--multicall", type=lambda x: str(x).lower() in ["1","true","yes"], default=False)
    ap.add_argument("--enable-dodo", type=lambda x: str(x).lower() in ["1","true","yes"], default=False)
    ap.add_argument("--verbose", type=lambda x: str(x).lower() in ["1","true","yes"], default=False)
    ap.add_argument("--log-every", type=int, default=1)

    ap.add_argument("--eth-variants", type=lambda x: str(x).lower() in ["1","true","yes"], default=False,
                    help="also quote BETH, wBETH, wstETH if slice sizing is possible")
    ap.add_argument("--extra-eth", type=str, default="",
                    help='comma-separated extra ETH-like tokens as SYMBOL:0xaddr,...')

    ap.add_argument("--dodo-pools", type=str, default="", help="comma-separated DODO pool addresses to query directly")
    ap.add_argument("--dodo-view", type=lambda x: str(x).lower() in ["1","true","yes"], default=True)
    ap.add_argument("--dodo-scan", type=lambda x: str(x).lower() in ["1","true","yes"], default=False)
    ap.add_argument("--scan-from", type=int, default=None)
    ap.add_argument("--scan-to", type=str, default=None)
    ap.add_argument("--scan-step", type=int, default=50_000)
    ap.add_argument("--scan-max-logs", type=int, default=20_000)
    ap.add_argument("--scan-timeout", type=int, default=45)

    ap.add_argument("--expand-v3", choices=["never", "missing", "always"], default="missing",
                    help="Add direct v3 quoter calls; 'missing'=only if MC returned none for venue.")
    ap.add_argument("--resnap-ms", type=int, default=800,
                    help="if a slice run exceeds this latency and block advanced, re-snapshot and retry once")

    # Modes
    ap.add_argument("--mode", choices=["slice","fixed"], default="slice",
                    help="slice=auto-size to target USD per slice; fixed=use --fixed-in of the base token")
    ap.add_argument("--fixed-in", type=float, default=0.1,
                    help="When --mode=fixed: exact base-token amount to liquidate per slice (default 0.1)")

    # Oracles
    ap.add_argument("--oracle", action="append", default=[],
                    help=("Repeatable oracle spec. Formats: "
                          "chainlink:PAIR@0xfeed | chainlink_ratio:BASE@0x,QUOTE@0x | "
                          "band:PAIR@0xstdref | pyth:PAIR@0xpyth:PRICEID | dia:PAIR@0xdia:KEY"))
    ap.add_argument("--oracle-stale-secs", type=int, default=600)

    # Persistence
    ap.add_argument("--pool-cache", type=str, default="pools_cache.json")
    ap.add_argument("--refresh-pool-cache", action="store_true")
    ap.add_argument("--db", type=str, default="quotes.sqlite",
                    help="SQLite path (set empty to disable DB writes)")

    # Output files (all on by default)
    ap.add_argument("--out-best-csv", type=str, default="onchain_benchmark.csv",
                    help="CSV with one 'best' summary row per slice")
    ap.add_argument("--out-quotes-csv", type=str, default="quotes_full.csv",
                    help="CSV with every quote (per block, per venue/path, USDT/USDC separately)")
    ap.add_argument("--out-failures-csv", type=str, default="failures_log.csv",
                    help="CSV with all failures")
    ap.add_argument("--out-raw-csv", type=str, default="raw_payloads.csv",
                    help="CSV with raw payload (head) per multicall result for auditing")
    ap.add_argument("--out-oracles-jsonl", type=str, default="oracles.jsonl",
                    help="JSONL stream of oracle readings and median per block snapshot")

    return ap.parse_args()

# ---------------------------
# Helpers for console & CSV
# ---------------------------
def _print_now(msg: str):
    print(msg, flush=True)

def _write_csv(path: str, rows: list[dict]):
    if not path or not rows:
        return
    try:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    except Exception as e:
        _print_now(f"[csv] failed to write {path}: {e}")

# ---------------------------
# Oracle plumbing (wrappers)
# ---------------------------
def _parse_oracle_specs(args) -> list:
    specs = []
    bad_specs = []
    for raw in (args.oracle or []):
        try:
            specs.append(_parse_oracle_arg(raw))
        except Exception as e:
            bad_specs.append((raw, str(e)))
    if bad_specs:
        for raw, why in bad_specs:
            _print_now(f"[oracles] bad --oracle '{raw}': {why}")
    return specs

def _summarize_fresh(readings, want_pair: str):
    fresh = [r for r in readings if (not r.stale) and (r.price is not None) and r.pair.upper()==want_pair.upper()]
    # Show at most 6 sources inline (keeps console tidy)
    inline = "  ".join(f"{r.src}:{r.price:.6f}" for r in fresh[:6])
    return fresh, inline

def _fetch_and_persist_oracles(oracle_specs, sym: str, blkno: int, blkt: int, args, db):
    """Always prints a line and writes JSONL, even on failure or when no specs provided."""
    want_pair = f"{sym}/USDT"

    # Default values if oracles are disabled or failed
    readings, median, status_note = [], None, ""
    if not oracle_specs:
        status_note = "(none configured)"
        _print_now(f"[oracles] {want_pair} {status_note}")
    else:
        try:
            readings, median = fetch_oracles(
                oracle_specs, want_pair, block_number=blkno, stale_secs=args.oracle_stale_secs
            )
            fresh, inline = _summarize_fresh(readings, want_pair)
            if fresh:
                med_note = f"{median:.6f}" if median is not None else "n/a"
                _print_now(f"[oracles] {want_pair} median ≈ {med_note}  |  sources: {inline}")
            else:
                _print_now(f"[oracles] {want_pair} no fresh normalized readings")
        except Exception as e:
            status_note = f"failed: {e}"
            _print_now(f"[oracles] {want_pair} {status_note}")

    # Persist to DB + JSONL even if empty
    if db:
        try:
            db.insert_oracles(blkno, readings)
        except Exception:
            pass
    try:
        log_jsonl(args.out_oracles_jsonl, {
            "block_number": blkno,
            "block_timestamp": blkt,
            "pair": want_pair,
            "median": median,
            "status": status_note,
            "readings": [
                {"src": r.src, "pair": r.pair, "price": r.price,
                 "updated_at": r.updated_at, "stale": r.stale, "note": r.raw_note}
                for r in readings
            ]
        })
    except Exception:
        pass

    return median

# ---------------------------
# Token discovery/sizing
# ---------------------------
def _build_eth_like(args) -> List[Tuple[str, str]]:
    eth_like: List[Tuple[str, str]] = [("ETH", ETHb)]
    if args.eth_variants:
        eth_like += [("BETH", BETH), ("wBETH", WBETH), ("wstETH", WSTETH)]
    if args.extra_eth:
        items = [x.strip() for x in args.extra_eth.split(",") if x.strip()]
        for it in items:
            try:
                sym, addr = [t.strip() for t in it.split(":", 1)]
                eth_like.append((sym, CS(addr)))
            except Exception:
                _print_now(f"[warn] bad --extra-eth item: {it}")
    return eth_like

def _size_eth_like(eth_like: List[Tuple[str,str]], args) -> list[tuple[str, str, int, int]]:
    sized: list[tuple[str, str, int, int]] = []
    for sym, addr in eth_like:
        d = decimals_of(addr)
        if args.mode == "fixed":
            per_slice = int(max(0.0, args.fixed_in) * (10 ** d))
            _print_now(f"Slice sizing (fixed): {sym:6s} = {args.fixed_in} {sym}")
        else:
            per_slice = find_amount_per_slice_for_target_usdt(addr, args.slice_usd)
            if per_slice is None:
                if args.verbose:
                    _print_now(f"[fallback] V2 sizing failed for {sym}; trying robust sizing via V3/THENA/Wombat...")
                per_slice = find_amount_per_slice_for_target_usdt_robust(addr, args.slice_usd, verbose=args.verbose)
            if per_slice is None:
                _print_now(f"[skip] could not size slice for {sym} (no viable route to USDT on any venue)")
                continue
            _print_now(f"Slice sizing: {sym:6s} ~ {human(per_slice, d):.8f} {sym} (≈ ${args.slice_usd})")
        sized.append((sym, addr, per_slice, d))
    if not sized:
        raise SystemExit("No ETH-like base token could be sized. Aborting.")
    return sized

def _discover_dodo_if_needed(sized, args):
    discovered_cache: Dict[str, List[str]] = {}
    if args.enable_dodo and args.dodo_view:
        for sym, base_addr, _amt, _d in sized:
            try:
                found = dodo_discover_by_factories(base_addr, USDT, verbose=args.verbose)
                discovered_cache[base_addr.lower()] = found
                _print_now(f"[dodo] factory-view discovery for {sym}: {len(found)} pool(s)")
            except Exception as e:
                _print_now(f"[dodo] factory-view discovery for {sym} failed: {e}")
    return discovered_cache

# ---------------------------
# Slice runner
# ---------------------------
def _run_one_slice(i, slices, sym, base, per_slice_in, d, oracle_specs, discovered_cache, args, db,
                   rows_best, rows_quotes, failures_all, rawdumps_all, total_out_usdt_equiv):
    if args.verbose:
        _print_now(f"[Slice {i+1}/{slices}] • quoting {sym} amount ≈ {human(per_slice_in, d):.8f} {sym} ...")

    # (1) Snapshot
    snap = get_block_snapshot()
    blkno, blkt = snap["number"], snap["timestamp"]
    if args.verbose:
        _print_now(f"[slice] block snapshot: #{blkno} @ {blkt} | rpc={ACTIVE_RPC_URL}")

    if db:
        try: db.upsert_block(blkno, blkt)
        except Exception: pass

    # (2) Oracles (always print + JSONL)
    median = _fetch_and_persist_oracles(oracle_specs, sym, blkno, blkt, args, db)

    # (3) Quote all venues for this snapshot
    t0 = time.perf_counter()
    discovered = discovered_cache.get(base.lower(), [])
    qs, fails, dumps = quote_all(
        base, per_slice_in,
        gas_adjusted=args.gas_adjusted,
        use_multicall=args.multicall,
        enable_dodo=args.enable_dodo,
        dodo_pools_cli=parse_addr_list(args.dodo_pools)[0],
        discovered_pools=discovered,
        verbose=args.verbose,
        expand_v3=args.expand_v3,
        snap=snap,
        slice_idx=i+1
    )
    dur_ms = (time.perf_counter() - t0) * 1000.0

    # Optional re-snapshot if slow and chain advanced
    latest_after = w3.eth.block_number
    if dur_ms > args.resnap_ms and latest_after > blkno:
        if args.verbose:
            _print_now(f"[slice] re-snapshot: run took {dur_ms:.0f} ms and chain advanced to #{latest_after}; retrying once...")
        snap2 = get_block_snapshot()
        blkno, blkt = snap2["number"], snap2["timestamp"]
        if db:
            try: db.upsert_block(blkno, blkt)
            except Exception: pass
        t1 = time.perf_counter()
        qs, fails, dumps = quote_all(
            base, per_slice_in,
            gas_adjusted=args.gas_adjusted,
            use_multicall=args.multicall,
            enable_dodo=args.enable_dodo,
            dodo_pools_cli=parse_addr_list(args.dodo_pools)[0],
            discovered_pools=discovered,
            verbose=args.verbose,
            expand_v3=args.expand_v3,
            snap=snap2,
            slice_idx=i+1
        )
        dur_ms = (time.perf_counter() - t1) * 1000.0
        if args.verbose:
            _print_now(f"[slice] retry completed in {dur_ms:.1f} ms at block #{blkno}")

    # Collect failures + raw payloads (CSV later)
    failures_all.extend([dict(f, **{"slice": i+1, "base": sym, "block": blkno, "timestamp": blkt}) for f in fails])
    rawdumps_all.extend([RawDump(dmp.venue, f"{sym}: {dmp.desc}", dmp.ok, dmp.raw_len, dmp.raw_head_hex) for dmp in dumps])

    # (4) Leaders by token (console UX)
    best_usdt = max((q for q in qs if q.out_token.lower() == USDT.lower()), key=lambda q: q.amount_out, default=None)
    best_usdc = max((q for q in qs if q.out_token.lower() == USDC.lower()), key=lambda q: q.amount_out, default=None)
    if best_usdt:
        _print_now(f"  ⇒ best USDT: {best_usdt.venue:16s} | {best_usdt.path:35s} | {human(best_usdt.amount_out, decimals_of(USDT)):.6f} USDT")
    if best_usdc:
        _print_now(f"  ⇒ best USDC: {best_usdc.venue:16s} | {best_usdc.path:35s} | {human(best_usdc.amount_out, decimals_of(USDC)):.6f} USDC")

    # (5) Persist every quote (CSV+DB), include oracle median on each row
    _usdt_cache: Dict[Tuple[str, int, int], Tuple[float, str]] = {}

    def _usdt_eq_both(token_addr: str, amount: int) -> Tuple[float, str]:
        """
        Returns (usdt_equiv, note). Caches by (token, amount, block) to avoid
        repeated Wombat conversions within the same slice.
        """
        key = (token_addr.lower(), int(amount), int(blkno))
        if key in _usdt_cache:
            return _usdt_cache[key]
        val, note = to_usdt_equiv_ex(
            token_addr, amount, block_number=blkno, verbose=args.verbose
        )
        _usdt_cache[key] = (val, note)
        return val, note

    def _usdt_eq(q_: Quote) -> float:
        val, _ = _usdt_eq_both(q_.out_token, q_.amount_out)
        return val

    def _usdt_eq_note(q_: Quote) -> str:
        _, note = _usdt_eq_both(q_.out_token, q_.amount_out)
        return note


    for q in qs:
        ue = _usdt_eq(q)
        rows_quotes.append({
            "block": blkno,
            "timestamp": blkt,
            "base": sym,
            "amount_in_base": human(per_slice_in, d),
            "venue": q.venue,
            "path": q.path,
            "out_token": symbol_of(q.out_token),
            "amount_out_human": human(q.amount_out, decimals_of(q.out_token)),
            "amount_out_wei": str(q.amount_out),
            "usdt_equiv": ue,
            "usdt_equiv_note": _usdt_eq_note(q),
            "gas_estimate": q.gas_estimate if q.gas_estimate is not None else "",
            "ticks_crossed": q.ticks_crossed if q.ticks_crossed is not None else "",
            "latency_ms": round(q.ms, 2),
            "oracle_median": (median if median is not None else "")
        })
        if db:
            try: db.insert_quotes(blkno, q, ue)
            except Exception: pass

    # choose global best (USDT-equivalent)
    if qs:
        best = max(qs, key=_usdt_eq)
        best_usdt_equiv = _usdt_eq(best)
    else:
        best, best_usdt_equiv = None, 0.0

    total_out_usdt_equiv += best_usdt_equiv

    base_v2 = next((q for q in qs if q.venue == "pancake_v2" and q.path == f"{sym}->USDT"), None)
    base_v3 = next((q for q in qs if q.venue in ("pancake_v3", "uniswap_v3") and q.path.startswith(f"{sym}->USDT")), None)
    base_v2_usdt = _usdt_eq(base_v2) if base_v2 else 0.0
    base_v3_usdt = _usdt_eq(base_v3) if base_v3 else 0.0
    edge_vs_v2_bps = ((best_usdt_equiv - base_v2_usdt) * 10_000 / base_v2_usdt) if base_v2_usdt else 0.0
    edge_vs_v3_bps = ((best_usdt_equiv - base_v3_usdt) * 10_000 / base_v3_usdt) if base_v3_usdt else 0.0

    mode_label = "gas_adjusted" if args.gas_adjusted else "raw"
    if (args.verbose or ((i + 1) % max(1, args.log_every) == 0)) and best is not None:
        _print_now(f"  → best: {best.venue:16s} | {best.path:35s} | {best_usdt_equiv:.6f} USDT_eq | edge_vs_PCSv2 {edge_vs_v2_bps:.2f} bps ({mode_label})\n")

    rows_best.append({
        "slice": i + 1,
        "block": blkno,
        "timestamp": blkt,
        "base": sym,
        "amount_in_base": f"{human(per_slice_in, d):.8f}",
        "best_venue": (best.venue if best else ""),
        "best_path": (best.path if best else ""),
        "best_out_token": (symbol_of(best.out_token) if best else ""),
        "best_out_amount_human": (f"{best_usdt_equiv:.6f} USDT_equiv" if best else ""),
        "best_gas_estimate": (best.gas_estimate if (best and best.gas_estimate is not None) else ""),
        "ticks_crossed": (best.ticks_crossed if (best and best.ticks_crossed is not None) else ""),
        "latency_ms": round(dur_ms, 2),
        "edge_vs_pancake_v2_bps": round(edge_vs_v2_bps, 2),
        "edge_vs_pancake_v3_bps": round(edge_vs_v3_bps, 2),
        "failures_this_slice": len(fails),
        "ranking_mode": mode_label,
        "used_multicall": args.multicall,
        "used_dodo": args.enable_dodo or bool(parse_addr_list(args.dodo_pools)[0]) or bool(discovered),
        "oracle_median": (median if median is not None else "")
    })

    if db:
        try: db.upsert_best(blkno, (best or None), best_usdt_equiv, median)
        except Exception: pass

    return total_out_usdt_equiv

# ---------------------------
# Main (refactored)
# ---------------------------
def main():
    args = parse_cli()

    # Re-init web3 if custom RPC given
    global w3
    if args.rpc and args.rpc != DEFAULT_BSC_RPC:
        w3 = mk_web3(args.rpc)
        
    ensure_rpc_ready(56)  # BSC chainId=56; set None to skip chainId check
    # Initialize optional helpers (persistors/caches) if available
    try:
        pool_cache = PoolCache(args.pool_cache) if args.pool_cache else PoolCache("")
    except NameError:
        pool_cache = None

    try:
        db = BenchDB(args.db) if (getattr(args, "db", "") or "") else None
    except NameError:
        db = None

    # One-time telemetry
    try:
        gas_gwei = float(w3.eth.gas_price) / 1e9
    except Exception:
        gas_gwei = 0.0
    wbnb_usdt = price_wbnb_in_usdt()
    _print_now(f"Gas price ≈ {gas_gwei:.2f} gwei | WBNB/USDT ≈ {wbnb_usdt:.4f}")

    # Build ETH-like list and size
    eth_like = _build_eth_like(args)
    sized = _size_eth_like(eth_like, args)

    # Number of slices (unchanged)
    slices = math.ceil(args.total_usd / args.slice_usd) if args.mode == "slice" else math.ceil(args.total_usd / max(1e-9, args.slice_usd))
    _print_now(f"\nSlices: {slices}\n")

    # Parse --dodo-pools safely
    dodo_pools_cli, bad = parse_addr_list(args.dodo_pools)
    if bad:
        _print_now(f"[warn] ignoring invalid addresses in --dodo-pools: {', '.join(bad)}")

    # Discover DODO pools via factory view (recommended)
    discovered_cache = _discover_dodo_if_needed(sized, args)

    # Output accumulators
    rows_best: list[dict] = []
    rows_quotes: list[dict] = []
    failures_all: list[dict] = []
    rawdumps_all: List[RawDump] = []
    total_out_usdt_equiv = 0.0

    # Oracle specs (robust parse; never silently empty due to NameError)
    oracle_specs = _parse_oracle_specs(args)

    # Slice loop
    for i in range(slices):
        for (sym, base, per_slice_in, d) in sized:
            total_out_usdt_equiv = _run_one_slice(
                i, slices, sym, base, per_slice_in, d,
                oracle_specs, discovered_cache, args, db,
                rows_best, rows_quotes, failures_all, rawdumps_all, total_out_usdt_equiv
            )

    # ---------------------------
    # Output files
    # ---------------------------
    _write_csv(args.out_quotes_csv, rows_quotes)
    _write_csv(args.out_best_csv, rows_best)

    if failures_all:
        keys = sorted({k for f in failures_all for k in f.keys()})
        try:
            with open(args.out_failures_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader(); w.writerows(failures_all)
        except Exception as e:
            _print_now(f"[csv] failed to write {args.out_failures_csv}: {e}")

    if rawdumps_all:
        try:
            with open(args.out_raw_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["venue", "desc", "ok", "raw_len", "raw_head_hex"])
                w.writeheader()
                for dmp in rawdumps_all:
                    w.writerow({"venue": dmp.venue, "desc": dmp.desc, "ok": dmp.ok, "raw_len": dmp.raw_len, "raw_head_hex": dmp.raw_head_hex})
        except Exception as e:
            _print_now(f"[csv] failed to write {args.out_raw_csv}: {e}")

    _print_now(f"Total best-out sum ≈ {total_out_usdt_equiv:,.2f} USDT_equiv across {slices} slices")
    if rows_best:   _print_now(f"Saved best CSV     -> {args.out_best_csv}")
    if rows_quotes: _print_now(f"Saved quotes CSV   -> {args.out_quotes_csv}")
    if failures_all:_print_now(f"Saved failures CSV -> {args.out_failures_csv}")
    if rawdumps_all:_print_now(f"Saved raw payloads -> {args.out_raw_csv}")


if __name__ == "__main__":
    main()
