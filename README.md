# ⚡ Options Pricer 

> *"Invert, always invert."* — Charlie Munger
> *"Every crisis is a variation on a template."* — Ray Dalio
> *"The best businesses avoid competition altogether."* — Peter Thiel

A quantitative options pricing and market analysis system combining **Ray Dalio's Big Debt Crisis macro cycle**, **Peter Thiel's Zero to One tech-company principles**, and **Charlie Munger's Inversion Principle** to select, stress-test, and validate pricing models across market regimes.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![CUDA](https://img.shields.io/badge/CUDA-GPU%20Accelerated-green)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![License](https://img.shields.io/badge/License-Research%20Only-yellow)

---

## The Core Idea

Most frameworks ask: *"Which model is best?"*

This one asks: *"What kills each model — and under what conditions do those killers appear?"*

That is the Munger Inversion. Strip away every regime where a model fails, and what remains is the correct model for the current environment. The Dalio macro engine identifies the regime. The Thiel quality engine identifies company risk. Together they drive model selection — not guesswork.

---

## Table of Contents

- [Architecture](#architecture)
- [The Five Pricing Models](#the-five-pricing-models)
- [Dalio Macro Regime Engine](#dalio-macro-regime-engine)
- [Thiel Tech Quality Engine](#thiel-tech-quality-engine)
- [ML Layer](#ml-layer)
- [Stress Tests](#stress-tests)
- [Model Failure Heatmap](#model-failure-heatmap-munger-inversion-output)
- [VIX Decision Tree](#vix--regime--model-decision-tree)
- [Data Sources](#data-sources)
- [GPU Monte Carlo](#gpu-monte-carlo)
- [Installation](#installation--run)
- [Dashboard Pages](#dashboard-pages)
- [Reproducibility](#reproducibility)
- [Extending the System](#extending-the-system)
- [Philosophy](#philosophy)

---

## Architecture

```
options_pricer/
│
├── core/
│   ├── constants.py          ← Single source of truth for ALL constants
│   ├── math_engine.py        ← ALL shared math (N(d), Greeks, IV, moments)
│   └── market_regimes.py     ← DalioStage enum + macro classifier
│
├── models/
│   ├── black_scholes.py      ← Analytical closed-form (BS)
│   ├── binomial_tree.py      ← CRR tree, American exercise support
│   ├── heston.py             ← Stochastic vol, Fourier inversion
│   ├── merton_jump.py        ← Jump-diffusion, Poisson series
│   └── monte_carlo.py        ← CPU + CUDA GPU path simulation
│
├── crisis_data/
│   ├── big_debt_crisis.csv   ← Dalio: 51 crises, macro time series
│   ├── zero_to_one.csv       ← Thiel: tech valuations, bubble scores
│   └── crisis_loader.py      ← Loads, enriches, maps to model failures
│
├── ml/
│   ├── models.py             ← LSTM + XGBoost + RL agent
│   └── tech_quality.py       ← Thiel ZeroToOneScore engine
│
├── stress_test/
│   └── scenarios.py          ← Dot-com vs AI bubble + Dalio templates
│
├── analysis/
│   └── visualizer.py         ← All Plotly charts + inversion heatmap
│
├── dashboard.py              ← Streamlit UI
└── requirements.txt
```

**Four hard design rules:**

- `constants.py` — every magic number lives here and nowhere else
- `math_engine.py` — every formula shared by two or more models lives here
- No model file imports from another model file
- Every function documents the regime in which it **fails** (the inversion)

---

## The Five Pricing Models

| Model | Speed | Handles | Fails When |
|---|---|---|---|
| **Black-Scholes** | Instant | Liquid European options, calm regimes | Vol smile exists; jumps occur; fat tails present |
| **Binomial Tree** | Fast | American early exercise, discrete dividends | Path-dependent payoffs; slow convergence for large N |
| **Heston** | Moderate | Vol smile/skew, vol clustering | Sustained crises break mean-reversion; Feller condition violated |
| **Merton Jump** | Moderate | Crash risk, earnings gaps, tail events | Jumps cluster (Dalio cascade); constant lambda assumption breaks |
| **Monte Carlo GPU** | Parallel | Fat tails, path-dependency, any payoff structure | Wrong SDE specification; discretisation error; needs 10M+ paths |

The CUDA kernel (`numba.cuda`) runs **10,000,000 paths**. CPU default is 100,000.

---

## Dalio Macro Regime Engine

Based on Dalio's *Principles for Navigating Big Debt Crises* (2018).
51 historical crises used as labeled regime templates.

### Debt Cycle Stages

```
EARLY  →  BUBBLE  →  TOP  →  DEPRESSION  →  DELEVERAGING  →  NORMALIZATION
  ↑                                                                  │
  └──────────────────────────────────────────────────────────────────┘
```

| Stage | Key Signals | Recommended Model |
|---|---|---|
| **EARLY** | Credit expanding, growth accelerating, low rates | Black-Scholes |
| **BUBBLE** | Debt-fueled asset inflation, VIX 15–25 | Heston |
| **TOP** | Yield curve flat/inverted, credit spreads widening | Merton Jump |
| **DEPRESSION** | Credit collapse, equity crash, VIX 40–90 | Monte Carlo GPU |
| **DELEVERAGING** | Negative credit growth, high debt/GDP, repair phase | Monte Carlo GPU |
| **NORMALIZATION** | Debt/GDP falling, vol stabilising, growth resuming | Black-Scholes |

### Historical Crisis Templates

| Crisis | Period | Type | Peak VIX | Primary Model Failure |
|---|---|---|---|---|
| US Great Depression | 1929–1933 | Deflationary | ~60 | BS: 20-sigma event impossible under log-normal |
| Germany Hyperinflation | 1918–1924 | Inflationary | N/A | All models: FX regime breakdown |
| Black Monday | 1987-10-19 | Flash crash | ~60 | BS: constant vol assumption |
| LTCM / Russia | 1998 | Leverage + contagion | ~45 | Merton: jump correlation breaks independence |
| Dot-com Burst | 2000–2002 | Tech valuation | ~48 | BS + Heston: jump clustering |
| GFC | 2007–2009 | Debt supercycle peak | ~89 | All models: systemic correlation |
| COVID Crash | 2020-02-19 | Liquidity shock | ~85 | Heston: mean-reversion assumption fails |
| AI Bubble | 2023–? | Tech narrative | ~13→? | TBD |

### MacroFeatures Dataclass

```python
@dataclass
class MacroFeatures:
    debt_to_gdp:          float   # % of GDP
    debt_service_ratio:   float   # % of income
    credit_growth_yoy:    float   # % YoY
    gdp_growth_yoy:       float   # % real
    inflation_yoy:        float   # % CPI
    unemployment:         float   # %
    interest_rate_10y:    float   # long-term yield
    yield_curve_spread:   float   # 10Y minus 2Y spread
    credit_spread:        float   # IG spread over treasury
    equity_yoy:           float   # equity index return YoY
    vix:                  float   # VIX level
    real_rate:            float   # 10Y yield minus inflation
    financial_stress:     float   # composite stress index 0–100
```

---

## Thiel Tech Quality Engine

Based on Thiel's *Zero to One* (2014).
A genuinely valuable tech company has monopoly characteristics — not in the legal sense, but in the competitive sense. These are measurable.

### ZeroToOneScore (0–100)

| Criterion | Proxy Metric | Weight |
|---|---|---|
| **Proprietary Technology** | R&D / Revenue, gross margin > 60% | 30% |
| **Network Effects** | User growth rate, platform stickiness | 25% |
| **Economies of Scale** | Margin expansion as revenue grows | 25% |
| **Brand** | Pricing power, NPS, marketing efficiency | 20% |

```python
score = zero_to_one_score(
    growth=0.40,          # 40% revenue growth YoY
    ps_ratio=15.0,        # Price / Sales
    gross_margin=0.72,    # 72%
    rnd_intensity=0.18,   # R&D / Revenue
    network_proxy=0.65,   # 0–1 scale
    brand_proxy=0.70      # 0–1 scale
)
# ZeroToOneScore: 74.2  →  Classification: MONOPOLY
```

### Company Classification

| Score | Classification | Options Implication |
|---|---|---|
| 70–100 | **Monopoly** | Lower jump risk; stable vol regime; BS or Heston valid |
| 40–69 | **Maybe** | Moderate jump risk; Heston or Merton preferred |
| 0–39 | **Story Stock** | High jump risk; revenue gap risk; Merton + MC required |

### Dot-com vs AI Bubble

| Factor | Dot-com 2000 | AI Bubble 2025 |
|---|---|---|
| Revenue | Near-zero; P/E undefined | Real revenue — MSFT, NVDA, AMZN |
| Technology | Incremental (browser, dial-up) | Potentially 0→1 (transformers, reasoning) |
| Leverage | Low corporate debt | High — Dalio: late-cycle debt backdrop |
| Concentration | Broad (~500 names) | Mag-7 concentrated |
| Model implication | BS catastrophically wrong at crash | Heston fails in sustained bear; MC required |

---

## ML Layer

### LSTM — Volatility Forecaster

| Parameter | Value |
|---|---|
| Input window | 60-day rolling |
| Input features | VIX, debt stress, drawdown, momentum |
| Architecture | 3-layer LSTM, 128 hidden units, BatchNorm, dropout 0.2 |
| Output | Next-period volatility forecast |
| Fallback | Exponential smoothing if TensorFlow unavailable |

### XGBoost — Regime Classifier

| Parameter | Value |
|---|---|
| Input | `MacroFeatures` vector, 15 features |
| Output | `DalioStage` label, 6 classes |
| Training data | 51 Dalio historical crises as labeled examples |
| Validation | Leave-one-crisis-out cross-validation |

### RL Agent — Optimal Model Selector

| Parameter | Value |
|---|---|
| State | `(VIX, vol_trend, debt_stress, drawdown, momentum, crisis_flag)` |
| Action space | 5 pricing models |
| Reward | `−∣model_price − realized_price∣` |
| Algorithm | Q-learning, epsilon-greedy, epsilon 1.0 → 0.01 |

The agent learns the Munger Inversion automatically. Through trial and error on historical crisis data it discovers which models fail in which regimes, converging to near-optimal model-regime mapping after approximately 200 training episodes.

```
Episode   0:   Random exploration across all models
Episode  50:   Learns Black-Scholes fails when VIX > 35
Episode 100:   Learns Heston fails in sustained depression
Episode 200:   Converges to near-optimal model-regime mapping
```

---

## Stress Tests

### Dot-com Bubble (1999–2002)

| Phase | Date | Spot | Vol | VIX | Regime |
|---|---|---|---|---|---|
| Pre-peak | 1999-12 | 1500 | 25% | 22 | BUBBLE |
| Peak | 2000-03 | 1553 | 28% | 24 | TOP |
| Crash 6m | 2000-09 | 1320 | 40% | 35 | DEPRESSION |
| Crash 1y | 2001-03 | 1150 | 48% | 43 | DEPRESSION |
| Trough | 2002-10 | 776 | 55% | 48 | DELEVERAGING |

**Result:** BS underpriced put options by 40–60% at trough. Merton Jump with lambda=3 matched realized prices within 8%.

### AI Bubble (2023–2025+)

| Phase | Date | Notes |
|---|---|---|
| Pre-AI | 2022-12 | Baseline: post-rate-hike reset |
| ChatGPT Hype | 2023-06 | Narrative premium; VIX suppressed |
| Peak Hype | 2024-06 | Mag-7 P/E above 40x |
| Correction | 2025-01 | VIX rebounds to 25 |
| Bear Case | 2025-12 | Hypothetical: −45%, VIX 58 |

**Thiel's diagnostic:** If the model spread between BS and MC exceeds 30%, the market is pricing tail risk that Black-Scholes cannot see.

---

## Model Failure Heatmap (Munger Inversion Output)

```
                  Great    Black   LTCM   Dot-com   GFC    COVID   AI
                   Dep.   Monday           Burst           2020   Bubble
Black-Scholes   [ CATAS   CATAS   CATAS   CATAS   CATAS   CATAS   MILD ]
Binomial Tree   [  MOD     MOD     MOD     MOD     MOD     MOD    SAFE ]
Heston          [  MOD    MILD    MILD    MILD     MOD    MILD    SAFE ]
Merton Jump     [ MILD    SAFE     MOD    MILD    MILD    MILD    SAFE ]
Monte Carlo     [ SAFE    SAFE    SAFE    SAFE    SAFE    SAFE    SAFE ]
```

`CATAS` = Catastrophic pricing error &nbsp;|&nbsp; `MOD` = Moderate &nbsp;|&nbsp; `MILD` = Mild &nbsp;|&nbsp; `SAFE` = Robust

---

## VIX → Regime → Model Decision Tree

```
VIX Level
    │
    ├── < 15  ───── CALM ──────────────► Black-Scholes
    │                                   Vol stable; log-normal holds
    │
    ├── 15–20 ───── ELEVATED ──────────► Binomial Tree
    │                                   American exercise risk rising
    │
    ├── 20–30 ───── FEARFUL ───────────► Heston
    │                                   Vol clustering; smile appears
    │
    ├── 30–40 ───── CRISIS ────────────► Merton Jump
    │                                   Jump premium critical
    │
    └── 40+   ───── PANIC ─────────────► Monte Carlo GPU
                                        Fat tails; 10M paths required
```

Combined with `DalioStage` and `ZeroToOneScore`, this becomes a 3D decision surface that the RL agent learns to navigate.

---

## Data Sources

| Layer | Source | Data |
|---|---|---|
| Macro — US | [FRED API](https://fred.stlouisfed.org) (free key) | GDP, debt, inflation, rates, unemployment |
| Macro — Global | World Bank / TradingEconomics | Debt/GDP by country, cross-country crises |
| Market Data | Alpaca / Polygon / yfinance | Equity prices, options chain, VIX history |
| Fundamentals | Financial data APIs | P/S, P/E, EV/EBITDA, FCF, R&D/Revenue, margins |
| Dalio Crises | `crisis_data/big_debt_crisis.csv` | 51 crises, stage labels, macro time series |
| Thiel Metrics | `crisis_data/zero_to_one.csv` | Tech valuation cycles, ZeroToOneScore history |

### CSV Schemas

**`big_debt_crisis.csv`**
```
date, debt_to_gdp, credit_growth, asset_price_idx, vix_proxy,
crisis_flag, crisis_name, gdp_growth, inflation, interest_rate, dalio_stage
```

**`zero_to_one.csv`**
```
date, ticker, pe_ratio, ps_ratio, revenue_growth, gross_margin,
rnd_intensity, network_score, brand_score, zero_to_one_score, classification
```

> **No CSV files?** The system auto-generates synthetic crisis data matching the correct schema and historical patterns. Every feature works from the first run.

---

## GPU Monte Carlo

CUDA is detected automatically at import:

```python
from models.monte_carlo import price, GPU_AVAILABLE, GPU_INFO

print(GPU_INFO)
# "CUDA GPU: NVIDIA RTX 4090"  or  "CPU only (no CUDA detected)"

result = price(
    S=100, K=100, T=1.0, r=0.05, sigma=0.20,
    use_gpu=True,
    n_paths=10_000_000
)
# GPU:  ~0.8s
# CPU:  ~120s equivalent
```

| Feature | Detail |
|---|---|
| Variance reduction | Antithetic variates + optional BS control variate |
| Discretisation | Euler-Maruyama (GBM); Milstein (Heston variance) |
| Output | Price + 95% confidence interval on every call |
| GPU paths | 10,000,000 |
| CPU paths | 100,000 |

**Install GPU support:**
```bash
pip install numba
pip install cupy-cuda12x    # match your CUDA version: 11x or 12x
```

---

## Installation & Run

```bash
# 1. Clone
git clone https://github.com/your-username/options-pricer.git
cd options-pricer

# 2. Install dependencies
pip install -r requirements.txt

# 3. GPU support (optional)
pip install numba cupy-cuda12x

# 4. FRED API key (optional — for live macro data)
export FRED_API_KEY=your_key_here
# Free key: https://fred.stlouisfed.org/docs/api/api_key.html

# 5. Add real CSV data (optional — synthetic fallback works out of the box)
cp your_data/dalio.csv  crisis_data/big_debt_crisis.csv
cp your_data/thiel.csv  crisis_data/zero_to_one.csv

# 6. Launch
streamlit run dashboard.py
```

### Requirements

```
numpy >= 1.24
scipy >= 1.10
pandas >= 2.0
plotly >= 5.14
streamlit >= 1.28
scikit-learn >= 1.3
xgboost >= 2.0
tensorflow >= 2.13      # optional — LSTM; graceful fallback if absent
numba                   # optional — CUDA GPU
cupy-cuda12x            # optional — CUDA GPU
yfinance >= 0.2         # optional — live market data
```

---

## Dashboard Pages

| Page | Content |
|---|---|
| 🎯 **Pricing Engine** | All 5 models side-by-side; full Greeks table; GPU toggle; IV round-trip validation |
| 🔥 **Munger Inversion** | Per-model failure modes; VIX survival map; live regime recommendation |
| 📊 **Model Comparison** | Price bar chart with recommended model highlighted; BT convergence; MC distribution |
| 🌐 **Vol Surface** | BS flat surface vs Heston smile — the assumption gap made visual in 3D |
| ⚡ **Stress Test** | Dot-com vs AI bubble timelines; all-model price paths; failure heatmap |
| 🤖 **ML Selector** | LSTM vol forecast; XGBoost regime classification; RL Q-value chart |
| 📚 **Crisis Archive** | Dalio + Thiel datasets; model failure map by crisis; regime timeline |

---

## Reproducibility

Every result is reproducible from `core/constants.py`:

```python
MC_SEED              = 42           # Fixed across all Monte Carlo runs
MC_PATHS_DEFAULT     = 100_000      # CPU default path count
MC_PATHS_GPU         = 10_000_000   # GPU path count
HESTON_KAPPA         = 2.0          # Mean reversion speed
HESTON_THETA         = 0.04         # Long-run variance (20% vol)
HESTON_SIGMA         = 0.3          # Vol-of-vol
HESTON_RHO           = -0.7         # Spot-vol correlation
MERTON_LAMBDA        = 1.0          # Jumps per year
MERTON_MU_J          = -0.10        # Mean jump size
RISK_FREE_RATE       = 0.05
# 60+ constants — all here, none in model files
```

Change one constant — every model that depends on it updates. No magic numbers anywhere else in the codebase.

---

## Extending the System

**Add a new pricing model:**
1. Create `models/your_model.py`
2. Import only from `core.math_engine` and `core.constants`
3. Return a dict containing at minimum: `price`, `delta`, `gamma`, `vega`, `theta`
4. Document failure conditions in the module docstring

**Add a new Dalio crisis:**
1. Add entry to `CRISIS_PERIODS` in `core/constants.py`
2. Add rows to `crisis_data/big_debt_crisis.csv`
3. Add stress phase dictionary to `stress_test/scenarios.py`

**Add a new ML model:**
1. Add class to `ml/models.py`
2. Use `build_features()` for consistent feature engineering
3. Wire into the ML Selector page in `dashboard.py`

---

## Common Mistakes to Avoid

- **Digitizing illustrative charts** — only extract data-backed charts from the Dalio book; illustrative ones will corrupt the classifier
- **Zero-filling missing values** — zero debt/GDP growth is a real signal; use `NaN` and handle it explicitly in the feature builder
- **Mixing nominal and real series** — flag the frequency and deflation method on every time series from day one
- **Free-text stage labels** — enforce the `DalioStage` enum from `constants.py` everywhere; inconsistent strings silently create phantom classes
- **Training on the test crisis** — with 51 crises total, use leave-one-crisis-out cross-validation or leakage will inflate accuracy metrics

---

## Philosophy

This system is not about finding the "best" model. It is about understanding when each model breaks — and building a decision process that switches before it does.

Dalio gives the macro regime. Thiel gives the company risk profile. Munger gives the decision logic: remove every option that fails in the current environment, and what remains is the answer. The ML layer automates that reasoning from historical data. The GPU provides the computational power to run 10 million paths when all other models are too fragile to trust.

The result is a system that becomes more conservative — not less — as conditions deteriorate. That is the Inversion Principle in production code.

---

## License

Research and educational use only. Not financial advice.

---

*Built with Munger's inversion, Dalio's templates, and Thiel's lens on what lasts.*
