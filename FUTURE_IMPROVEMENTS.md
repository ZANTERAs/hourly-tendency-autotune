# Future Improvements

A running list of planned enhancements, roughly ordered within each section by
impact-to-effort. Items are not commitments — just the backlog.

---

## 1. Project organization (housekeeping)

**Goal:** move from a flat folder where scripts, model weights, configs, and docs
all sit together, to a structured layout that's easier to navigate.

**Proposed structure:**

```
LSTM_Pytorch/
├── src/                       # core source
│   ├── main.py
│   ├── compare.py
│   ├── predict.py
│   └── trend_3mo.py
├── models/                    # saved .pt ensemble weights (git-ignored)
│   └── <TICKER>_lstm_e*.pt
├── outputs/                   # run artifacts
│   ├── <TICKER>_model_metadata.json
│   └── <TICKER>_best_config.json
├── docs/                      # explainer document + generators
│   ├── generate_pdf.py
│   ├── generate_docx.js
│   ├── LSTM_Explained.docx
│   └── LSTM_Explained.pdf
├── README.md
├── FUTURE_IMPROVEMENTS.md
├── requirements.txt
├── .gitignore
└── package.json
```

**Watch out for:** moving files breaks the hardcoded save/load paths in the code.
When this is done, update:
- where `main.py` saves `<TICKER>_lstm_e{seed}.pt` and `<TICKER>_model_metadata.json`
  → point at `models/` and `outputs/`
- where `compare.py` writes and `predict.py` reads `<TICKER>_best_config.json`
  → point at `outputs/`
- the `.gitignore` `*.pt` rule → `models/*.pt`

A small `paths.py` (or constants at the top of each script) defining
`MODELS_DIR`, `OUTPUTS_DIR`, `DOCS_DIR` would centralize this so paths never
drift again.

**Effort:** low. **Impact:** quality-of-life, no model change.

---

## 2. Modeling improvements

### Conformal prediction calibration
Post-process the P10/P90 bands against a held-out calibration set so coverage is
*guaranteed* to hit the target (e.g. 80%) regardless of the model's raw
calibration. Pure post-hoc step, no architecture change. Directly fixes the
under-coverage seen on volatile tickers.
**Effort:** low–medium. **Impact:** high (reliability).

### Regime detection feature
Train a simple Hidden Markov Model on SPY returns to label the market as
bull / bear / high-volatility, and feed that label as an extra feature. Likely
explains a lot of the fold-to-fold IC variance (model behaves very differently
in trending vs choppy regimes).
**Effort:** medium. **Impact:** medium–high.

### Temporal Fusion Transformer (TFT)
Replace (or A/B against) the bidirectional LSTM with a TFT, which was purpose-built
for multi-horizon forecasting with uncertainty and native handling of static
metadata (sector), known-future inputs, and past observed inputs. Tends to beat
LSTMs on financial series.
**Effort:** high. **Impact:** potentially high, uncertain.

---

## 3. Tooling & usability

### Multi-ticker dashboard
Run the model across a watchlist (e.g. MSFT, NVDA, AAPL, JPM) and emit one
summary table: signal, P50, IC, coverage, Sharpe side by side. Turns the model
from "one stock at a time" into a screener.
**Effort:** medium. **Impact:** high (usability).

### Ticker via CLI argument
`compare.py` and `predict.py` already accept a ticker argument; bring the same to
`main.py` (`python main.py NVDA`) so changing stock never requires editing code.
**Effort:** trivial. **Impact:** quality-of-life.

### Config file instead of in-script constants
Move the hyperparameter block to a `config.yaml` / `config.json` so runs are
reproducible and comparable without touching source.
**Effort:** low. **Impact:** medium.

---

## 4. Robustness & data

### Cache downloaded market data
yfinance is occasionally flaky and slow. Cache SPY / sector-ETF / ^TNX / earnings
data locally (e.g. parquet with a daily TTL) to speed up repeated runs and avoid
empty-download failures.
**Effort:** low. **Impact:** medium.

### Transaction-cost-aware backtest
The current backtest is frictionless. Add a per-trade cost/slippage assumption
so the reported Sharpe reflects something closer to reality.
**Effort:** low. **Impact:** medium (honesty of results).
