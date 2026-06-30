# LSTM Stock Tendency Forecaster

An hourly-bar quantile LSTM that forecasts the **~1-week forward return** of a
stock as a calibrated **P10 / P50 / P90 band**, then **auto-tunes itself** across
four history windows (3-month, 6-month, 1-year, 2-year) and saves the most
reliable configuration to JSON.

The model is **ticker-agnostic**: pass any US-listed ticker and it auto-detects
the sector ETF, auto-widens its quantile bands for high-volatility names, and
scores its own reliability with a confidence-weighted composite.

> **Disclaimer**: This is a research and educational project, not financial
> advice. Forecasts are probabilistic and frequently wrong. Do not trade real
> money based on its output.

---

## What it does

Two-step workflow:

```bash
python src/compare.py TICKER     # ~7-9 min: 4-window sweep, picks winner, saves JSON
python src/predict.py TICKER     # ~1 min:  loads JSON, retrains winning config, forecasts
```

`compare.py` trains the same architecture against four different history windows
(3-month / 6-month / 1-year / 2-year of hourly bars), scores each on
out-of-sample reliability, and writes the winner's full hyperparameters to
`outputs/<TICKER>_best_config.json`.

`predict.py` reads that file and runs only the winning configuration — same
forecast, ~6x faster.

## Project layout

```
hourly-tendency-autotune/
├── src/                                # source modules
│   ├── compare.py                      # 4-window auto-tune (entry point)
│   ├── predict.py                      # fast forecast from saved JSON
│   ├── trend_3mo.py                    # earlier single-window variant
│   └── paths.py                        # centralized path constants
├── models/                             # saved .pt weights (git-ignored)
├── outputs/                            # per-ticker results
│   └── <TICKER>_best_config.json       # winning config + all-window comparison
├── README.md
├── FUTURE_IMPROVEMENTS.md
├── requirements.txt
└── .gitignore
```

---

## How the auto-tuner picks a winner

Each window gets a **reliability score** in `[0, 1]`:

```
raw_score   = 0.30 * dir_skill  +  0.45 * ic_skill  +  0.25 * coverage_skill
confidence  = sqrt( min(1, n_val / 300) )
final_score = raw_score * confidence
```

| Component | Formula | What it measures |
|-----------|---------|------------------|
| `dir_skill`     | `(DirAcc - 0.5) * 2` clipped to [0, 1] | Directional accuracy above coin flip |
| `ic_skill`      | `max(0, IC / 0.5)` clipped to [0, 1]   | Rank correlation between predicted and actual returns |
| `coverage_skill`| `1 - abs(coverage - target) / target`  | Calibration of the P10/P90 (or P05/P95) band |
| `confidence`    | `sqrt(n_val / 300)` clipped to 1.0     | Statistical weight from validation-set size |

**Why IC dominates the weights:** in a bull market a model can hit 80% DirAcc by
just predicting "up" every time, while having no real predictive skill. IC
(rank correlation) catches that and is much harder to game. This was validated
on MU, where the 6-Month window had 82% DirAcc but IC = -0.12 (no real skill).
The IC-heavy scoring correctly assigned it a low final score.

**Why the confidence factor:** the 3-month window has only ~84 validation
sequences vs ~834 for the 2-year window. With overlapping sequences and a
35-bar forward horizon, the effective independent sample size is tiny for short
windows. `sqrt(n)` is the natural penalty since metric standard error scales
as `1/sqrt(n)`.

### Quantile auto-calibration

Before training, the script measures the standard deviation of the training-set
forward returns and picks the band width:

| Training σ (1-week fwd) | Quantile band |
|-------------------------|---------------|
| `σ < 4 pp`              | P10 / P50 / P90 (80% band) — default |
| `σ ≥ 4 pp`              | P05 / P50 / P95 (90% band) — wider, for volatile stocks |

This keeps the band sharp for stable names and avoids under-coverage on noisy
ones — and it's computed per-window, so the same ticker can use different
bands across windows.

---

## Architecture

| Stage | Detail |
|-------|--------|
| **Bars**          | Hourly (`interval="1h"`), regular session only, 7 bars/day |
| **Input window**  | 70 bars (~2 trading weeks) |
| **Target**        | 35-bar forward percentage return (~1 trading week) |
| **Features (14)** | Close, Volume, EMA9, EMA20, EMA spread, RSI14, MACD, OBV, BB width, BB %B, ATR14, Stoch K, SPY 1-week return, sector-ETF 1-week return |
| **Backbone**      | 1-layer LSTM (hidden=32) + soft attention |
| **Head**          | Linear -> 3 quantiles (P10/P50/P90 or P05/P50/P95) |
| **Loss**          | 75% pinball (quantile) + 25% soft-Spearman on P50 |
| **Optimizer**     | Adam (lr=1e-3) + ReduceLROnPlateau + early stopping |
| **Final model**   | Ensemble of 3 seeds, averaged quantile predictions |

The model is intentionally lightweight — small enough to train 4 windows x 3
seeds on a CPU in under 10 minutes for the full sweep.

---

## Scripts

| Script | Purpose | Runtime |
|--------|---------|---------|
| `src/compare.py [TICKER]` | 4-window auto-tune -> `outputs/<TICKER>_best_config.json` | ~7-9 min |
| `src/predict.py [TICKER]` | Loads saved config, retrains winning window, forecasts | ~1-2 min |
| `src/trend_3mo.py`        | Earlier single-window 3-month variant (no auto-tune) | ~30 sec |

For the longer-horizon (20-day forward return) daily-bar variant with
walk-forward validation, earnings flags, and a 5-seed ensemble, see the
companion repo:
[ZANTERAs/daily-return-forecaster](https://github.com/ZANTERAs/daily-return-forecaster).

---

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/ZANTERAs/hourly-tendency-autotune.git
cd hourly-tendency-autotune
pip install -r requirements.txt
```

GPU is used automatically if available, but CPU is fully supported (and what
the hyperparameters were tuned for).

---

## Example results

A real sweep across four tickers, May 2026:

| Ticker | Winner | Score | Best IC | DirAcc | Verdict |
|--------|--------|-------|---------|--------|---------|
| **HWM**   | 1-Year | **0.826** | +0.69 | 79.8% | Genuine skill — clean persistent uptrend (2Y R²=0.96) |
| GOOGL  | 6-Month | 0.599 | +0.48 | 64.6% | Real skill, mean-reverting after a rally |
| MSFT   | 6-Month | 0.545 | +0.54 | 51.3% | Moderate — choppy, near coin-flip directionally |
| MU     | 2-Year | **0.355** | -0.01 | 72.7% | Essentially unpredictable — volatile, trend-dominated |

The scoring system correctly distinguishes a clean trend-follower (HWM) from a
high-DirAcc-but-no-real-skill case (MU). See `<TICKER>_best_config.json` for
full per-window breakdowns including raw skill components, confidence factor,
trend slope/R², and the saved forecast.

---

## Outputs

Per ticker, `compare.py` produces:

- `outputs/<TICKER>_best_config.json` — winning hyperparameters + all-window comparison + forecast
- Interactive Plotly chart with 3 panels: price + per-window trend lines, forecast bars, and score breakdown

`predict.py` produces:

- Same forecast chart (3 panels: price + trend + forecast cone, validation, forecast bar)
- Console summary

`trend_3mo.py` produces:

- `models/<TICKER>_trend3mo_e{0,1,2}.pt` — 3 ensemble model weights *(git-ignored)*
- Interactive Plotly chart

---

## Metrics glossary

| Metric | Meaning | Baseline / target |
|--------|---------|-------------------|
| **DirAcc** | % of times `sign(P50) == sign(actual)` | 50% = random; **caution**: high in trending markets even with no skill |
| **IC** | Information Coefficient — Pearson `corr(P50, actual)` | 0.00 = random; >0.10 useful; >0.30 strong |
| **Coverage** | % of actual returns inside the P10–P90 (or P05–P95) band | ideal ≈ 80% or 90% to match the band |
| **RMSE** | Root mean squared error of P50 in percentage points | lower is better; expect 2-5pp on calm stocks, 10+ on volatile ones |
| **Trend R²** | Goodness-of-fit of a linear regression over the window's closes | >0.7 = strong trend; <0.3 = noisy or sideways |

---

## Notes

- Model weights (`*.pt`) are intentionally **not** committed — they are regenerated by running the scripts. Everything needed to reproduce them is in the code and the saved JSON.
- Yahoo Finance hourly data has a 730-day cap; the 2-year window therefore uses ~3,438 bars in practice (not the nominal 3,528).
- Data is pulled live from Yahoo Finance via `yfinance`, so an internet connection is required and exact numbers will drift as new market data arrives.
- See [`FUTURE_IMPROVEMENTS.md`](FUTURE_IMPROVEMENTS.md) for the planned roadmap (conformal prediction calibration, regime detection, multi-ticker dashboard, etc.).
