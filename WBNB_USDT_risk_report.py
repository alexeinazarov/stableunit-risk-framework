#!/usr/bin/env python3
"""
Resilient PancakeSwap-v3 risk dashboard (BNB chain).

• Picks the pool whose last swap ≤ DAYS_BACK
• Calculates σ (annual), Hurst, VaR-99, depth @ +1 %
• Safe pagination, jittered exponential back-off, rotating log file
"""

import os, math, json, time, random, logging, itertools, requests
from decimal import Decimal, getcontext
from datetime import datetime, timezone, timedelta
import pandas as pd, numpy as np
from tabulate import tabulate
from logging.handlers import RotatingFileHandler
from typing import Tuple

# ─── CONFIG ───────────────────────────────────────────────────────
BASE, QUOTE   = "WBNB", "USDT"
DAYS_BACK     = 60
MIN_SWAPS     = 20
PAGE_SIZE     = 1_000
BREAK_EARLY   = 700_000
SESSION = requests.Session() 
MAX_ROWS      = 2_000_000      # hard cap for one swap pull (~1.2 GB JSON)
STATE_FILE    = "cursor.json"  # optional resume-point on crash
# ─── RISK-ENGINE PARAMETERS ───────────────────────────────────────
WINDOW_DAILY_MIN = 15          # historical VaR needs ≥ this many daily bars
Z_99              = 2.3263479   # 99 % one-tailed z-score
INTRADAY_BARS_DAY = 288         # 5-minute bars in one trading day

GATEWAY_KEY   = os.getenv("GRAPH_KEY", "").strip()
SUBGRAPH_ID   = "A1fvJWQLBeUAggX2WQTMm3FKjXTekNXo77ZySun4YN2m"
URL           = f"https://gateway.thegraph.com/api/{GATEWAY_KEY}/subgraphs/id/{SUBGRAPH_ID}"
HDRS          = {"Content-Type": "application/json",
                 "Authorization": f"Bearer {GATEWAY_KEY}"}

getcontext().prec = 60   # high-precision Decimal

# ─── LOGGING (console + rotating file) ────────────────────────────
root = logging.getLogger()
root.setLevel(logging.INFO)
fmt  = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s",
                         datefmt="%H:%M:%S")
sh   = logging.StreamHandler()
sh.setFormatter(fmt)
root.addHandler(sh)
fh   = RotatingFileHandler("WBNB_USDT_risk.log", maxBytes=5_000_000, backupCount=3)
fh.setFormatter(fmt)
fh.setLevel(logging.DEBUG)
root.addHandler(fh)

# ─── GRAPHQL TEXT BLOCKS ──────────────────────────────────────────
Q_TOK = """query($s:String!){tokens(first:6,orderBy:totalValueLockedUSD,
         orderDirection:desc,where:{symbol:$s}){id}}"""
Q_POOL = """query($a:String!,$b:String!){pools(first:5,orderBy:volumeUSD,
          orderDirection:desc,where:{token0:$a,token1:$b}){id feeTier volumeUSD}}"""
Q_LAST = """query($p:String!){swaps(first:1,orderBy:timestamp,
          orderDirection:desc,where:{pool:$p}){timestamp}}"""
def Q_SW(n): return f"""query($p:String!,$since:Int!,$skip:Int!){{
  swaps(first:{n},skip:$skip,orderBy:timestamp,orderDirection:asc,
        where:{{pool:$p,timestamp_gt:$since}}){{timestamp amount0 amount1}}}}"""
def Q_TK(n): return f"""query($p:String!,$skip:Int!){{
  ticks(first:{n},skip:$skip,where:{{pool:$p}}){{price1 liquidityNet}}}}"""

# ─── EXPONENTIAL BACK-OFF WITH JITTER ─────────────────────────────
def gql(q: str, v: dict, tag: str,
        max_retry_dns=8, max_retry_http=3):

    attempt_dns  = attempt_http = 0
    delay = 0.6
    while True:
        t0 = time.perf_counter()
        try:
            r = SESSION.post(URL, headers=HDRS,
                             json={"query": q, "variables": v},
                             timeout=(3, 60))               # 3s connect, 60s read
            r.raise_for_status()
            js = r.json()
            if "errors" in js:
                raise RuntimeError(js["errors"])
            logging.debug("⌛ %-14s %.0f ms", tag, (time.perf_counter()-t0)*1e3)
            return js["data"]

        except requests.exceptions.ConnectionError as e:
            attempt_dns += 1
            logging.warning("DNS/conn error %-14s retry %d/%d: %s",
                            tag, attempt_dns, max_retry_dns, str(e)[:80])
            if attempt_dns >= max_retry_dns:
                raise
            time.sleep(delay * random.uniform(1.0, 1.5))
            delay *= 1.6

        except requests.exceptions.HTTPError as e:
            attempt_http += 1
            logging.warning("HTTP %s %-14s retry %d/%d",
                            e.response.status_code if e.response else "??",
                            tag, attempt_http, max_retry_http)
            if attempt_http >= max_retry_http:
                raise
            time.sleep(delay * random.uniform(1.0, 1.3))
            delay *= 2

# ─── HELPERS ──────────────────────────────────────────────────────
def token_ids(sym):  # highest-TVL twins first
    return [t["id"] for t in gql(Q_TOK, {"s": sym}, f"tok:{sym}")["tokens"]]

def latest_swap(pid):
    try:
        ts = gql(Q_LAST, {"p": pid}, f"last:{pid[:6]}")["swaps"][0]["timestamp"]
        return int(ts)
    except Exception:
        logging.debug("latest swap query failed for %s", pid[:8])
        return 0

# --------------------------------------------------------------------
# resilient swap fetcher – widens look-back until ≥ MIN_SWAPS rows
# --------------------------------------------------------------------
def fetch_swaps(pool_id: str,
                since_ts: int,
                widen_days=(7, 30, 90, 180)) -> pd.DataFrame:
    """
    Try the requested window first; if the subgraph returns < MIN_SWAPS,
    stepwise enlarge the window until we succeed or exhaust the list.
    """
    for extra in (0, *widen_days):
        rows = page(Q_SW, pool_id, since=since_ts - extra * 86_400)
        if len(rows) >= MIN_SWAPS:
            if extra:
                logging.warning("swap window widened by %d d → %d rows",
                                extra, len(rows))
            return pd.DataFrame(rows)
    raise RuntimeError(
        f"subgraph contained < {MIN_SWAPS} swaps even after widening "
        f"to {widen_days[-1]} days.")


def choose_pool(sym0, sym1, since):
    """
    Returns   (pool_id , fee_basis_points , last_swap_ts)
    fee_basis_points is an int, e.g. 500 → 0.05 %
    """
    logging.info("🔍 discovering active %s/%s pool (≤ %d days)…",
                 sym0, sym1, DAYS_BACK)

    candidates = []
    for a, b in itertools.product(token_ids(sym0), token_ids(sym1)):
        for x, y in ((a, b), (b, a)):
            candidates += gql(Q_POOL, {"a": x, "b": y}, "pools")["pools"]

    # rank by all-time volume, then attach last-swap timestamp
    candidates.sort(key=lambda p: Decimal(p["volumeUSD"]), reverse=True)
    live = [(p, latest_swap(p["id"])) for p in candidates]

    for p, ts in live:
        logging.info("    pool %-8s fee=%s  last=%s",
                     p["id"][:8], p["feeTier"],
                     datetime.fromtimestamp(ts, timezone.utc)
                             .strftime('%Y-%m-%d %H:%M'))
        if ts >= since:
            return p["id"], int(p["feeTier"]), ts     # <── 3-tuple

    # -- no pool active in window: fall back to freshest overall
    p, ts = max(live, key=lambda t: t[1])
    logging.warning("‼️ every pool dormant – fallback to %s (last %s)",
                    p["id"][:8],
                    datetime.fromtimestamp(ts, timezone.utc)
                            .strftime('%Y-%m-%d %H:%M'))
    return p["id"], int(p["feeTier"]), ts


def page(qfun, pid, **extra):
    # resume from previous cursor if present -------------
    try:
        with open(STATE_FILE) as f:
            saved = json.load(f)
            skip  = saved.get("skip", 0)
            buf   = []              # we will append fresh chunks only
            logging.warning("Resuming pagination from skip=%d (state file)", skip)
    except FileNotFoundError:
        buf, skip = [], 0
    
    while True:
        try:
            part = gql(qfun(PAGE_SIZE),
                       dict(p=pid, skip=skip, **extra),
                       f"{qfun.__name__}[{skip}]")
        except (requests.exceptions.ConnectionError, RuntimeError):
            logging.warning(
                "NETWORK / INDEXER GLITCH – continuing with %d records fetched so far",
                len(buf)
            )
            break
  
            # partial data is usually good enough for risk metrics
            logging.warning("NETWORK GLITCH – continuing with %d records fetched so far",
                            len(buf))
            break
        items = list(part.values())[0]
        if not items:
            break
        buf.extend(items)
        # ------------------------------------------------
        # progress bookkeeping
        # ------------------------------------------------
        skip += PAGE_SIZE
        # Graph node performance drops sharply past skip≈500k
        if skip >= BREAK_EARLY:
            logging.warning("Hit skip=%d ( >500 k ) – breaking early", skip)
            break        
        if skip % BREAK_EARLY == 0:
            with open(STATE_FILE, "w") as f:
                json.dump({"skip": skip}, f)
            logging.debug("Checkpoint saved at skip=%d", skip)

        if len(buf) >= MAX_ROWS:
            logging.warning("Reached MAX_ROWS=%d → stop paging early", MAX_ROWS)
            break
        
        if len(items) < PAGE_SIZE:
            break
        time.sleep(0.1)
    return buf


# ----------------------------------------------------------------------
#  Hurst-exponent estimators
#  ---------------------------------------------------------------------
#  • hurst_rs           – classical rescaled-range (R/S)   Beran (1994)
#  • hurst_dfa          – detrended fluctuation analysis   Peng et al. (1994)
#  • hurst_local_whittle– semi-parametric periodogram      Robinson (1995)
# ----------------------------------------------------------------------

def hurst_rs(series: pd.Series) -> float:
    """
    Rescaled-range estimator (R/S).

    Beran, J. *Statistics for Long-Memory Processes*, 1994, §3.1.
    Implementation: average R/S over log-spaced lags, then slope in log-log plot.
    """
    x = series.dropna().values
    if len(x) < 64:
        return float("nan")
    lags = np.unique(np.logspace(1, int(math.log10(len(x)//2)), 12).astype(int))
    rs = []
    for lag in lags:
        seg = len(x) // lag
        Z = x[:seg*lag].reshape(seg, lag)
        Y = np.cumsum(Z - Z.mean(1, keepdims=True), 1)
        rs.append(np.nanmean((Y.max(1) - Y.min(1)) / Z.std(1, ddof=1)))
    H, _ = np.polyfit(np.log(lags[:len(rs)]), np.log(rs), 1)
    return float(H)


def hurst_dfa(series: pd.Series, min_win: int = 16, bands: int = 20) -> float:
    """
    Detrended Fluctuation Analysis (first-order).

    Peng, C.-K. et al., “Mosaic organization of DNA nucleotides”, *PR E* (1994).
    F(n) ∝ n^H; slope in log-log space gives H.
    """
    r = series.dropna().values.cumsum()
    N = len(r)
    wins = np.unique(np.logspace(math.log10(min_win),
                                 math.log10(N//4), bands).astype(int))
    F = []
    for w in wins:
        if w < 4 or w >= N:
            continue
        seg = N // w
        rms = []
        for i in range(seg):
            s = r[i*w:(i+1)*w]
            t = np.arange(w)
            coef = np.polyfit(t, s, 1)
            trend = coef[0]*t + coef[1]
            rms.append(np.sqrt(np.mean((s - trend)**2)))
        F.append(np.mean(rms))
    if len(F) < 5:
        return float("nan")
    H, _ = np.polyfit(np.log(wins[:len(F)]), np.log(F), 1)
    return float(H)


def hurst_local_whittle(series: pd.Series, m_frac: float = 0.1) -> float:
    """
    Local-Whittle semi-parametric estimator.

    Robinson, P. M. (1995) “Gaussian semiparametric estimation of long-range
    dependence”, *Annals of Statistics*.
    Minimises the Whittle likelihood over the lowest-frequency ordinates.
    """
    r = series.dropna().values
    n = len(r)
    k = np.arange(1, n//2)
    I = np.abs(np.fft.fft(r)[1:n//2])**2 / (2*math.pi*n)
    m = max(25, int(len(k)*m_frac))
    k, I = k[:m], I[:m]
    S = np.sum(np.log(k))

    def J(d):
        return np.log(np.mean(I / (k**(2*d - 1)))) + (2*d - 1) * S / m

    # golden-section line search on d ∈ (0.01, 0.99)
    a, b = 0.01, 0.99
    phi = (1 + 5**0.5) / 2
    for _ in range(36):
        c, d_ = b - (b - a)/phi, a + (b - a)/phi
        (b, a) = (d_, a) if J(c) < J(d_) else (b, c)
    return float((a + b) / 2)

    
# ─── LIQUIDITY DEPTH (+/- 2 %) ─────────────────────────────────────
def depth_plusminus(pool_id: str,
                    mid_price: float,
                    pct: float = 0.02,
                    batch: int = 1_000) -> tuple[float, float]:
    """
    Stream ticks outward from the pool, accumulating |liquidityNet|.
    Stops as soon as both bid- and ask-side walls (±pct) are crossed.
    Returns depth_bid_USD, depth_ask_USD.
    """

    target_ask = Decimal(mid_price) * Decimal(1 + pct)
    target_bid = Decimal(mid_price) / Decimal(1 + pct)

    bid_depth = ask_depth = Decimal(0)
    got_bid = got_ask = False
    skip = 0

    while not (got_bid and got_ask):

        page = gql(
            f"""query($p:String!,$s:Int!){{
                   ticks(first:{batch}, skip:$s,
                         orderBy:price1, orderDirection:asc,
                         where:{{pool:$p}}){{
                     price1 liquidityNet }} }}""",
            {"p": pool_id, "s": skip},
            f"ticks[{skip}]"
        )["ticks"]

        if not page:
            break  # exhausted the book

        for tk in page:
            price = Decimal(tk["price1"])
            liq   = abs(Decimal(tk["liquidityNet"]))

            if price <= target_bid:
                bid_depth += liq
            if price >= target_ask:
                ask_depth += liq

            # keep scanning until both sides satisfied
            got_bid = price <= target_bid
            got_ask = price >= target_ask
            if got_bid and got_ask:
                break

        if got_bid and got_ask:
            break

        skip += batch
        # no sleep needed – we stop as soon as walls are met

    return float(bid_depth), float(ask_depth)

# ----------------------------------------------------------------------
#  MAIN
# ----------------------------------------------------------------------
def main() -> None:
    """
    Fractional-Brownian risk pipeline with multi-estimator H and dual VaR
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    All constants trace to peer-reviewed sources (see inline refs).
    Report now shows:
      • counts of daily closes,
      • *both* empirical and fBm-parametric VaR-99,
      • choice flag indicating which one governs limits,
      • three H estimates (R/S, DFA, local-Whittle) + simple concordance.
    """

    # 1 ── pool discovery & swap pull ──────────────────────────────────
    since_ts = int((datetime.now(timezone.utc) - timedelta(days=DAYS_BACK))
                   .timestamp())
    pid, fee_bp, last_ts = choose_pool(BASE, QUOTE, since_ts) 

    swaps_df = fetch_swaps(pid, since_ts)
    if len(swaps_df) >= MAX_ROWS:
        logging.warning("Analysis capped at %d rows (MAX_ROWS).", MAX_ROWS)
    if len(swaps_df) < MIN_SWAPS:
        logging.error("Only %d swaps in window (<%d) – abort.",
                      len(swaps_df), MIN_SWAPS)
        return

    # 2 ── 5-minute mid-price series ──────────────────────────────────
    swaps_df["t"]  = pd.to_datetime(swaps_df["timestamp"].astype(int), unit="s")
    swaps_df["px"] = (swaps_df["amount1"].astype(float).abs() /
                      swaps_df["amount0"].astype(float).abs())

    px_5m = (swaps_df.set_index("t")["px"]
                       .resample("5min")
                       .last()
                       .ffill())
    ret_5m = px_5m.pct_change().dropna()

    # 3 ── volatility & three roughness estimators ─────────────────────
    sigma_5m = ret_5m.std(ddof=1)        # unbiased sample σ
    # --- long-memory estimators
    H_rs   = hurst_rs(ret_5m)
    H_dfa  = hurst_dfa(ret_5m)
    H_lw   = hurst_local_whittle(ret_5m)
    # simple sanity check (spread ≤ 0.1 recommended)
    H_spread = max(H_rs, H_dfa, H_lw) - min(H_rs, H_dfa, H_lw)
    if H_spread > 0.1:
        logging.warning("H estimators disagree by %.2f – investigate tails.",
                        H_spread)
    # use R/S for downstream scaling to stay consistent with prior runs
    H = H_rs
    # Lo-modified R/S asymptotic SE  (Lo 1991 Table II)
    se_H = math.sqrt(math.pi / (2 * len(ret_5m)))
    H_ci = (max(0, H - 1.96*se_H), min(1, H + 1.96*se_H))

    # fBm scaling
    sigma_day = sigma_5m * (INTRADAY_BARS_DAY ** H)
    sigma_ann = sigma_5m * ((INTRADAY_BARS_DAY * 365) ** H)

    hedge_uplift = (60/5) ** (H - 0.5)   # Hayashi–Mykland 2005 eq 3.2

    # 4 ── VaR-99: empirical vs parametric fBm ────────────────────────
    daily_ret = (px_5m.resample("1d").last()
                          .pct_change()
                          .dropna())
    n_daily = len(daily_ret)
    var99_empirical = daily_ret.quantile(0.01) if n_daily else float("nan")
    var99_param     = -Z_99 * sigma_day
    if n_daily >= WINDOW_DAILY_MIN:       # empirical dominates
        var99 = var99_empirical
        var_method = "empirical"
    else:
        var99 = var99_param
        var_method = f"parametric (n={n_daily})"

    # 5 ── liquidity depth (±1 %)  ------------------------------------
    bid_liq, ask_liq = depth_plusminus(
        pool_id   = pid,
        mid_price = px_5m.iloc[-1],
        pct       = 0.01
    )
    bid_usd = bid_liq * px_5m.iloc[-1] / 1e18
    ask_usd = ask_liq * px_5m.iloc[-1] / 1e18
    tot_usd = bid_usd + ask_usd

    # 6 ── final referee-ready report ---------------------------------
    start_dt = datetime.utcfromtimestamp(since_ts).strftime('%Y-%m-%d')
    end_dt   = swaps_df["t"].iloc[-1].strftime('%Y-%m-%d')
    last_dt  = datetime.utcfromtimestamp(last_ts).strftime('%Y-%m-%d %H:%M')
    pool_str = f"{pid[:8]}…  (fee {fee_bp/10_000:.2%}, last {last_dt} UTC)"

    print(f"\nPair   : {BASE}/{QUOTE}"
          f"\nPool   : {pool_str}"
          f"\nPeriod : {start_dt} → {end_dt}  (≤ {DAYS_BACK} days)\n")
            
    print("\n"+tabulate([{
        "pair"        : f"{BASE}/{QUOTE}",
        "σ (ann)"     : f"{sigma_ann:.2%}",
        "H_R/S"       : f"{H_rs:.2f}",
        "H_DFA"       : f"{H_dfa:.2f}",
        "H_LW"        : f"{H_lw:.2f}",
        "CI95 [R/S]"  : f"[{H_ci[0]:.2f},{H_ci[1]:.2f}]",
        "VaR-99"      : f"{var99:.2%}",
        "VaR method"  : var_method,
        "hedge ↑"     : f"{hedge_uplift:.1%}",
        "bid +1 %"    : f"${bid_usd:,.0f}",
        "ask +1 %"    : f"${ask_usd:,.0f}",
        "depth +1 %"  : f"${tot_usd:,.0f}"
    }], headers="keys", tablefmt="github"))

    # tidy up only on graceful completion
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)



if __name__ == "__main__":
    import sys
    if "--debug" in sys.argv:
        root.setLevel(logging.DEBUG)
    main()"
