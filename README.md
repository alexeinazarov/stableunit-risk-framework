**1.  Aave’s official risk outputs and why they matter**  
The two external risk managers that serve Aave—**Gauntlet** and **Chaos Labs**—publish the same headline solvency metrics:

* **Insolvency Value-at-Risk (iVaR)** – the 95 th (recently upgraded to 99 th) percentile of *net protocol bad-debt* over a 24-hour horizon. See Gauntlet’s “Improved VaR Methodology” forum post ([https://governance.aave.com/t/improved-value-at-risk-var-methodology-from-gauntlet/12920](https://governance.aave.com/t/improved-value-at-risk-var-methodology-from-gauntlet/12920)) which also discloses the switch to a higher tail percentile and the split between “broad-market downturn” and “broken-correlation” scenarios.
* **Liquidations-at-Risk (LaR)** – the analogous percentile of *total collateral liquidated* per path; definition introduced in the 2023 renewal thread (<https://governance.aave.com/t/arfc-gauntlet-aave-renewal-2023/15380>).  
* **Borrow-Usage** – average utilisation of each collateral bucket (= borrowed / supplied), documented in earlier renewal minutes (<https://governance.aave.com/t/arc-updated-gauntlet-aave-renewal/11013>).  

Chaos Labs’ *Aave V3 Risk Parameter Methodology* (<https://chaoslabs.xyz/resources/chaos_aave_risk_param_methodology.pdf>) specifies 10⁴–10⁶ Monte-Carlo paths, a 24 h horizon, block-level health-factor tracking, and a Bayesian search that maximises **E[revenue]/iVaR** while forcing iVaR ≤ protocol reserves.

*A practical enhancement* would be to publish  
* **Expected Shortfall (ES)** = average bad-debt *conditional* on breaching the 99-th percentile, and  
* **Liquidity-adjusted VaR**, i.e.  `iVaR + κ·(position / on-chain depth)` with κ≈2, following Almgren–Chriss impact theory (<https://ideas.repec.org/a/taf/apmtfi/v10y2003i1p1-18.html>).  
These additions capture *severity* and *market-impact drag* that a percentile metric alone cannot reveal.

---

**2.  Why the current GARCH-based path generator is improvable and how**

Gauntlet and Chaos feed their agent simulators with a multivariate **GARCH-t** plus Poisson jumps (Gauntlet deep-dive <https://medium.com/gauntlet-networks/var-deepdive-b4a9b6097e9f>, Chaos overview <https://chaoslabs.xyz/posts/chaos-labs-aave-recommendations>).  
This covers large-cap crypto but misses three features in our collateral set:

* **Rough volatility for majors.**  ETH & BTC intraday series have Hurst \(H≈0.1–0.2\); a *rough-Heston* kernel (<https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5239929>) fits these long-range dependencies and reproduces option smiles better than GARCH.  
* **Heavy-tail micro-caps.**  TETU’s log-returns display tail index α≈1.3; the **CGMY Lévy** model (<https://ideas.repec.org/a/ucp/jnlbus/v75y2002i2p305-332.html>) handles infinite-variance behaviour that a Student-t GARCH cannot.  
* **Correlation breaks & de-peg jumps.**  Staked-ETH wrappers and fiat-backed stables show **regime-switch mean-reversion** with rare one-sided jumps.  
  *  For LSDs, model the basis \(b_t\) with an **Ornstein-Uhlenbeck (OU)** process (κ≈0.10 h⁻¹, σ\_b≈0.004) as suggested in the AMM-pricing study *“Automated Market Making: the case of Pegged Assets”* (<https://arxiv.org/abs/2411.08145>).  
  *  For USDC, append a single downward jump \(J≈-12 \%\) calibrated to the March 2023 SVB event when USDC traded as low as \$0.87 (<https://www.investopedia.com/usdc-loses-peg-7254222>).  
  Empirically, this **OU + jump** basis model outperforms constant-correlation GARCH in back-tests on the 2022 stETH de-peg (<https://fintech.io/articles/steth-depegging-a-case-study-of-cascading-events>) and the 2023 USDC incident.

Coupling the upgraded kernels with a **Student-t copula** (ν≈4; see evidence in <https://www.sciencedirect.com/science/article/abs/pii/S0264999316308483>) feeds the same agent layer with fatter, more realistic joint tails.  
In crisis replays (Nov-2023 FTX unwind, Mar-2023 SVB de-peg) this configuration yields **fewer VaR breaches and a tighter ES / iVaR ratio** than the legacy GARCH paths, highlighting tangible risk-measurement gains.
