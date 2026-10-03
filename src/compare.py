"""
compare.py — auto-tune the LSTM tendency model across 4 time windows
             (3mo / 6mo / 1y / 2y), pick the most reliable one, and
             save the winning configuration to outputs/<TICKER>_best_config.json.

Usage:
    python src/compare.py            # uses TICKER below (MSFT)
    python src/compare.py AAPL       # override ticker via CLI
"""

import sys
import time
import json
import yfinance as yf

from paths import OUTPUTS_DIR

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

import pandas as pd
import numpy as np
import ta
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Config (shared across all runs) ───────────────────────────────────────────
TICKER       = "MSFT"        # CLI override is read in main(); autotune() takes it as argument

INTERVAL     = "1h"
BARS_PER_DAY = 7        # regular-session hourly bars per trading day
WINDOW       = 70       # look-back: 10 trading days × 7 bars
HORIZON      = 35       # forecast: 5 trading days × 7 bars (~1 week)
TRAIN_RATIO  = 0.75
QUANTILES_DEFAULT = [0.10, 0.50, 0.90]   # 80% band (low-vol stocks)
QUANTILES_WIDE    = [0.05, 0.50, 0.95]   # 90% band (high-vol stocks)
QUANTILES         = list(QUANTILES_DEFAULT)   # mutated per-window in run_window()
N_Q               = len(QUANTILES)
QUANTILE_WIDEN_VOL = 4.0    # std (pp) of fwd HORIZON returns that triggers wider bands
EPOCHS       = 200
BATCH_SIZE   = 32
HIDDEN_DIM   = 32
NUM_LAYERS   = 1
DROPOUT      = 0.30
SPEARMAN_W   = 0.25
LR           = 1e-3
LR_PATIENCE  = 10
ES_PATIENCE  = 30
N_ENSEMBLE   = 3

# ── Windows to compare (use_bars clipped if yfinance returns less) ────────────
WINDOWS = [
    {"label": "3-Month", "use_bars": 441},    # 63 trading days × 7
    {"label": "6-Month", "use_bars": 882},    # 126 trading days × 7
    {"label": "1-Year",  "use_bars": 1764},   # 252 trading days × 7
    {"label": "2-Year",  "use_bars": 3528},   # 504 trading days × 7
]

# Reliability score weights (must sum to 1.0)
# IC weighted highest — it's the most stable rank-correlation skill measure
# DirAcc is noisier (binary outcome), Coverage measures calibration quality
SCORE_WEIGHTS = {"dir_acc": 0.30, "ic": 0.45, "coverage": 0.25}

# Confidence factor — smaller val sets get a sqrt-scaled haircut
# n_val >= this gets full confidence (1.0); below scales as sqrt(n_val / CONF_FULL_AT)
CONFIDENCE_FULL_AT = 300

SECTOR_TO_ETF = {
    "Technology": "XLK", "Financial Services": "XLF", "Energy": "XLE",
    "Utilities": "XLU", "Healthcare": "XLV", "Consumer Defensive": "XLP",
    "Consumer Cyclical": "XLY", "Industrials": "XLI", "Real Estate": "XLRE",
    "Basic Materials": "XLB", "Communication Services": "XLC",
}

def _get_sector_etf(ticker):
    try:
        sector = yf.Ticker(ticker).info.get("sector", "")
        etf    = SECTOR_TO_ETF.get(sector, "")
        if etf:
            return sector, etf
    except Exception:
        pass
    return "Unknown", "SPY"

# Resolved in main() / autotune() (no network at import time).
_SECTOR_NAME, SECTOR_ETF = "Unknown", "SPY"

# ── Data ──────────────────────────────────────────────────────────────────────

def _fetch_close(ticker, period):
    raw = yf.download(ticker, period=period, interval=INTERVAL,
                      prepost=False, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    return raw["Close"].squeeze()


def load_full_data() -> pd.DataFrame:
    """Download max available hourly data and compute indicators once."""
    print(f"\n[1/3] Downloading {TICKER} (period=2y, interval={INTERVAL})...")
    raw = pd.DataFrame()
    for attempt in range(1, 4):
        raw = yf.download(TICKER, period="2y", interval=INTERVAL,
                          prepost=False, auto_adjust=True)
        if not raw.empty:
            break
        print(f"      attempt {attempt}/3 empty — retrying in 3s...")
        time.sleep(3)
    if raw.empty:
        raise RuntimeError(f"No data for '{TICKER}' after 3 attempts.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    print(f"[3/3] Fetching market context: SPY / {SECTOR_ETF}...", end="", flush=True)
    spy  = _fetch_close("SPY",      "2y")
    sect = _fetch_close(SECTOR_ETF, "2y")
    print(" done")
    return build_hourly_features(raw, spy, sect)


def build_hourly_features(ohlcv_1h: pd.DataFrame, spy_1h: pd.Series,
                          sect_1h: pd.Series) -> pd.DataFrame:
    """Pure feature engineering on hourly bars (no network). Same indicators as before."""
    df = ohlcv_1h[["High", "Low", "Close", "Volume"]].dropna().copy()
    print(f"      {len(df)} bars  [{df.index[0]} – {df.index[-1]}]")

    high, low, close, volume = df["High"], df["Low"], df["Close"], df["Volume"]

    print("[2/3] Computing indicators...", end="", flush=True)
    df["EMA_9"]      = ta.trend.EMAIndicator(close=close, window=9).ema_indicator()
    df["EMA_20"]     = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
    df["EMA_spread"] = df["EMA_9"] - df["EMA_20"]
    df["RSI_14"]     = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    df["MACD"]       = ta.trend.MACD(close=close).macd_diff()
    df["OBV"]        = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
    bb               = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    df["BB_width"]   = bb.bollinger_wband()
    df["BB_pos"]     = bb.bollinger_pband()
    df["ATR_14"]     = ta.volatility.AverageTrueRange(
                           high=high, low=low, close=close, window=14).average_true_range()
    df["STOCH_K"]    = ta.momentum.StochasticOscillator(
                           high=high, low=low, close=close, window=14, smooth_window=3).stoch()
    print(" done")

    df["SPY_ret"]  = spy_1h.pct_change(HORIZON).mul(100).reindex(df.index)
    df["SECT_ret"] = sect_1h.pct_change(HORIZON).mul(100).reindex(df.index)

    return df.dropna()


FEAT_COLS = [
    "Close", "Volume", "EMA_9", "EMA_20", "EMA_spread",
    "RSI_14", "MACD", "OBV", "BB_width", "BB_pos", "ATR_14", "STOCH_K",
    "SPY_ret", "SECT_ret",
]

# ── Sequences ─────────────────────────────────────────────────────────────────

def make_sequences(scaled, close_raw):
    max_i = len(scaled) - WINDOW - HORIZON
    if max_i <= 0:
        raise ValueError(f"Not enough data: {len(scaled)} bars (need >{WINDOW+HORIZON}).")
    X    = np.stack([scaled[i:i+WINDOW] for i in range(max_i)]).astype(np.float32)
    base = close_raw[WINDOW-1         : WINDOW-1         + max_i]
    fwd  = close_raw[WINDOW-1+HORIZON : WINDOW-1+HORIZON + max_i]
    y    = (100.0 * (fwd - base) / base).astype(np.float32)
    return X, y

def to_tensors(X, y):
    return torch.tensor(X), torch.tensor(y).unsqueeze(1)

# ── Model ─────────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1, bias=False)
    def forward(self, lstm_out):
        w = torch.softmax(self.score(lstm_out), dim=1)
        return (w * lstm_out).sum(dim=1)

class LSTMTrendModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, HIDDEN_DIM, NUM_LAYERS, batch_first=True)
        self.attn = Attention(HIDDEN_DIM)
        self.drop = nn.Dropout(DROPOUT)
        self.fc   = nn.Linear(HIDDEN_DIM, N_Q)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.drop(self.attn(out)))

# ── Loss ──────────────────────────────────────────────────────────────────────

def pinball_loss(pred, true):
    total = 0.0
    for i, q in enumerate(QUANTILES):
        err    = true - pred[:, i:i+1]
        total += torch.where(err >= 0, q * err, (q-1) * err).mean()
    return total / N_Q

def soft_spearman_loss(pred_p50, true):
    p   = pred_p50.squeeze(1)
    t   = true.squeeze(1)
    tau = (p.detach().std() + 1e-4).item()
    r_p = torch.sigmoid((p.unsqueeze(0) - p.unsqueeze(1)) / tau).sum(dim=1)
    r_t = torch.sigmoid((t.unsqueeze(0) - t.unsqueeze(1)) / tau).sum(dim=1)
    rp_c = r_p - r_p.mean()
    rt_c = r_t - r_t.mean()
    rho  = (rp_c * rt_c).sum() / (torch.sqrt((rp_c**2).sum() * (rt_c**2).sum()) + 1e-8)
    return 1.0 - rho

def combined_loss(pred, true):
    return ((1 - SPEARMAN_W) * pinball_loss(pred, true)
            + SPEARMAN_W    * soft_spearman_loss(pred[:, 1:2], true))

# ── Training ──────────────────────────────────────────────────────────────────

def train_model(model, tr_ld, va_ld, label="") -> float:
    opt  = torch.optim.Adam(model.parameters(), lr=LR)
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=LR_PATIENCE, factor=0.5)
    best_val, best_w, no_imp = float("inf"), None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss = 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = combined_loss(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            t_loss += loss.item() * len(xb)
        t_loss /= max(len(tr_ld.dataset), 1)
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_ld:
                xb, yb = xb.to(device), yb.to(device)
                v_loss += combined_loss(model(xb), yb).item() * len(xb)
        v_loss /= max(len(va_ld.dataset), 1)
        sch.step(v_loss)
        bar = "#" * int(20*epoch/EPOCHS) + "-" * (20 - int(20*epoch/EPOCHS))
        print(f"\r    {label}  [{bar}] ep {epoch:3d}/{EPOCHS}  val={v_loss:.4f}",
              end="", flush=True)
        if v_loss < best_val:
            best_val, best_w, no_imp = v_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= ES_PATIENCE:
                print(); break
    else:
        print()
    model.load_state_dict(best_w)
    return best_val

def compute_metrics(pred_q, true):
    p10, p50, p90 = pred_q[:,0], pred_q[:,1], pred_q[:,2]
    return dict(
        rmse     = float(np.sqrt(np.mean((p50 - true)**2))),
        dir_acc  = float(np.mean(np.sign(p50) == np.sign(true))),
        ic       = float(np.corrcoef(p50, true)[0, 1]) if len(p50) > 1 else 0.0,
        coverage = float(np.mean((p10 <= true) & (true <= p90))),
    )

def predict_ensemble(models, X_t):
    preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            preds.append(m(X_t.to(device)).cpu().numpy())
    return np.mean(preds, axis=0)

def _trend_slope(close_arr):
    x = np.arange(len(close_arr), dtype=float)
    slope, intercept = np.polyfit(x, close_arr, 1)
    yhat   = slope * x + intercept
    ss_res = np.sum((close_arr - yhat) ** 2)
    ss_tot = np.sum((close_arr - close_arr.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return 100.0 * slope / close_arr[0] * BARS_PER_DAY, r2, slope, intercept


def select_quantiles(train_close_arr: np.ndarray) -> tuple:
    """
    Auto-pick quantile band based on training-period forward-return volatility.

    Low-vol stocks (σ < QUANTILE_WIDEN_VOL pp)  → [0.10, 0.50, 0.90]  (80% band)
    High-vol stocks (σ ≥ QUANTILE_WIDEN_VOL pp) → [0.05, 0.50, 0.95]  (90% band)

    Wider bands avoid under-coverage for noisy stocks; narrower bands stay sharp
    for stable ones. Computed only on the training portion → no leakage.
    """
    if len(train_close_arr) < HORIZON + 1:
        return list(QUANTILES_DEFAULT), 0.0
    base = train_close_arr[:-HORIZON]
    fwd  = train_close_arr[HORIZON:]
    rets = 100.0 * (fwd - base) / base
    ret_std = float(np.std(rets))
    chosen = list(QUANTILES_WIDE if ret_std >= QUANTILE_WIDEN_VOL else QUANTILES_DEFAULT)
    return chosen, ret_std

# ── Reliability scoring ───────────────────────────────────────────────────────

def reliability_score(met: dict, n_val: int, quantiles: list) -> tuple:
    """
    Composite reliability score in [0, 1] — higher = more trustworthy.
    Returns (final_score, components_dict).

    Skill components (raw_score = weighted sum, all in [0, 1]):
      • dir_score : (DirAcc − 0.5) × 2                         — directional skill
      • ic_score  : max(0, IC / 0.5)                           — rank correlation skill
                    (IC > 0.5 is exceptional in finance; negative IC = 0)
      • cov_score : 1 − |coverage − target| / target           — calibration quality
                    (target derived from the window's actual quantile band width)

    Confidence factor (penalizes small validation sets):
      confidence  = sqrt( min(1, n_val / CONFIDENCE_FULL_AT) )
      final_score = raw_score × confidence

    Why sqrt: standard-error of metrics like DirAcc scales as 1/sqrt(n), so a
    sqrt-shaped penalty is statistically natural — a window with 4× the val
    sequences is 2× more trustworthy, not 4×.
    """
    if not met:
        return 0.0, {}
    dir_acc  = met.get("dir_acc", 0.5)
    ic       = met.get("ic", 0.0)
    coverage = met.get("coverage", 0.0)
    target_cov = quantiles[-1] - quantiles[0]   # 0.80 or 0.90 depending on band

    dir_score = max(0.0, min(1.0, (dir_acc - 0.5) * 2.0))
    ic_score  = max(0.0, min(1.0, ic / 0.5))
    cov_score = max(0.0, 1.0 - abs(coverage - target_cov) / target_cov)

    raw_score = (SCORE_WEIGHTS["dir_acc"]  * dir_score +
                 SCORE_WEIGHTS["ic"]       * ic_score  +
                 SCORE_WEIGHTS["coverage"] * cov_score)

    confidence  = float(np.sqrt(min(1.0, n_val / CONFIDENCE_FULL_AT)))
    final_score = raw_score * confidence

    components = {
        "dir_score":  dir_score,
        "ic_score":   ic_score,
        "cov_score":  cov_score,
        "raw_score":  raw_score,
        "confidence": confidence,
    }
    return final_score, components

# ── Per-window run ────────────────────────────────────────────────────────────

def run_window(df_full: pd.DataFrame, use_bars: int, label: str) -> dict:
    global QUANTILES, N_Q

    use_bars  = min(use_bars, len(df_full))
    df        = df_full.iloc[-use_bars:]
    values    = df[FEAT_COLS].values
    close_raw = df["Close"].values
    n, n_feat = values.shape

    slope_pct, r2, _, _ = _trend_slope(close_raw)
    trend_dir = "UP" if slope_pct > 0 else "DOWN"

    # ── Auto-select quantile band from training-period return volatility ──────
    # Use only the first TRAIN_RATIO of close prices → no leakage
    tr_close = close_raw[:int(len(close_raw) * TRAIN_RATIO)]
    chosen_q, ret_std = select_quantiles(tr_close)
    QUANTILES = chosen_q          # mutate global — used by pinball_loss
    N_Q       = len(QUANTILES)
    band_pct  = int((QUANTILES[-1] - QUANTILES[0]) * 100)
    q_lo_lbl  = f"P{int(QUANTILES[0]*100):02d}"
    q_hi_lbl  = f"P{int(QUANTILES[-1]*100):02d}"

    scaler = MinMaxScaler()
    scaler.fit(values)
    scaled = scaler.transform(values)

    X, y   = make_sequences(scaled, close_raw)
    n_seq  = len(X)
    tr_end = int(n_seq * TRAIN_RATIO)
    X_tr, y_tr = X[:tr_end], y[:tr_end]
    X_va, y_va = X[tr_end:], y[tr_end:]

    tr_ld = DataLoader(TensorDataset(*to_tensors(X_tr, y_tr)), BATCH_SIZE, shuffle=True)
    va_ld = DataLoader(TensorDataset(*to_tensors(X_va, y_va)), BATCH_SIZE)

    print(f"\n{'='*78}")
    print(f"  {label}  |  {n:,} bars  [{df.index[0]}  →  {df.index[-1]}]")
    print(f"  Trend  : {trend_dir}  {slope_pct:+.3f}%/day  (R²={r2:.3f})")
    print(f"  Return : σ={ret_std:.2f}pp ({HORIZON}-bar fwd)  "
          f"→  Quantiles: {q_lo_lbl}/P50/{q_hi_lbl} ({band_pct}% band)")
    print(f"  Seqs   : {n_seq:,} total  |  {len(X_tr):,} train / {len(X_va):,} val")
    print(f"{'='*78}")

    t0 = time.time()
    models = []
    for seed in range(N_ENSEMBLE):
        torch.manual_seed(seed)
        np.random.seed(seed)
        m  = LSTMTrendModel(n_feat).to(device)
        bv = train_model(m, tr_ld, va_ld, label=f"{label} m{seed+1}/{N_ENSEMBLE}")
        print(f"    => best_val={bv:.4f}")
        models.append(m)
    elapsed = time.time() - t0

    met = {}
    if len(X_va) >= 2:
        X_va_t, y_va_t = to_tensors(X_va, y_va)
        pred_q_val = predict_ensemble(models, X_va_t)
        true_r_val = y_va_t.squeeze(1).numpy()
        met = compute_metrics(pred_q_val, true_r_val)

    score, score_parts = reliability_score(met, n_val=len(X_va), quantiles=QUANTILES)

    current_close = float(df["Close"].iloc[-1])
    last_t = torch.tensor(scaled[-WINDOW:], dtype=torch.float32).unsqueeze(0)
    fc_q   = predict_ensemble(models, last_t).squeeze(0)
    fc_p10, fc_p50, fc_p90 = float(fc_q[0]), float(fc_q[1]), float(fc_q[2])

    print(f"  >>> Score = {score:.3f}  "
          f"(raw={score_parts.get('raw_score', 0):.3f} × "
          f"conf={score_parts.get('confidence', 0):.3f})  "
          f"[dir={score_parts.get('dir_score', 0):.2f} "
          f"ic={score_parts.get('ic_score', 0):.2f} "
          f"cov={score_parts.get('cov_score', 0):.2f}]")

    return dict(
        label       = label,
        bars        = n,
        n_seq       = n_seq,
        n_train     = len(X_tr),
        n_val       = len(X_va),
        slope_pct   = slope_pct,
        r2          = r2,
        trend_dir   = trend_dir,
        ret_std     = ret_std,
        quantiles   = list(QUANTILES),
        band_pct    = band_pct,
        met         = met,
        score       = score,
        score_parts = score_parts,
        fc_p10      = fc_p10,
        fc_p50      = fc_p50,
        fc_p90      = fc_p90,
        current     = current_close,
        signal      = "BULLISH" if fc_p50 > 0 else "BEARISH",
        elapsed     = elapsed,
        df          = df,
    )

# ── JSON persistence ──────────────────────────────────────────────────────────

def save_best_config(results: list, best: dict, out_path=None) -> str:
    """Save the winner's full config + a summary of all runs to JSON."""
    payload = {
        "ticker": TICKER,
        "sector":  _SECTOR_NAME,
        "sector_etf": SECTOR_ETF,
        "selected_at_utc": pd.Timestamp.utcnow().isoformat(),
        "best_window":    best["label"],
        "best_score":     round(best["score"], 4),
        "score_weights":  SCORE_WEIGHTS,
        "confidence_full_at": CONFIDENCE_FULL_AT,
        "quantile_widen_vol": QUANTILE_WIDEN_VOL,
        "model_config": {
            "interval":     INTERVAL,
            "bars_per_day": BARS_PER_DAY,
            "use_bars":     best["bars"],
            "window":       WINDOW,
            "horizon":      HORIZON,
            "train_ratio":  TRAIN_RATIO,
            "quantiles":    best["quantiles"],
            "band_pct":     best["band_pct"],
            "hidden_dim":   HIDDEN_DIM,
            "num_layers":   NUM_LAYERS,
            "dropout":      DROPOUT,
            "batch_size":   BATCH_SIZE,
            "epochs":       EPOCHS,
            "n_ensemble":   N_ENSEMBLE,
            "spearman_w":   SPEARMAN_W,
            "lr":           LR,
            "feat_cols":    FEAT_COLS,
        },
        "best_trend": {
            "direction":         best["trend_dir"],
            "slope_pct_per_day": round(best["slope_pct"], 4),
            "r2":                round(best["r2"], 4),
            "return_std_pp":     round(best["ret_std"], 4),
        },
        "best_metrics": {k: round(v, 4) for k, v in best["met"].items()},
        "best_score_components": {k: round(v, 4) for k, v in best["score_parts"].items()},
        "best_forecast": {
            "p10":          round(best["fc_p10"], 4),
            "p50":          round(best["fc_p50"], 4),
            "p90":          round(best["fc_p90"], 4),
            "current_close": round(best["current"], 4),
            "implied_low":  round(best["current"] * (1 + best["fc_p10"]/100), 4),
            "implied_mid":  round(best["current"] * (1 + best["fc_p50"]/100), 4),
            "implied_high": round(best["current"] * (1 + best["fc_p90"]/100), 4),
            "signal":       best["signal"],
            "horizon_bars": HORIZON,
            "horizon_desc": "~1 week (5 trading days × 7 hourly bars)",
        },
        "all_windows": [
            {
                "label":      r["label"],
                "bars":       r["bars"],
                "n_val":      r["n_val"],
                "quantiles":  r["quantiles"],
                "band_pct":   r["band_pct"],
                "score":      round(r["score"], 4),
                "score_parts":{k: round(v, 4) for k, v in r["score_parts"].items()},
                "trend": {
                    "direction":         r["trend_dir"],
                    "slope_pct_per_day": round(r["slope_pct"], 4),
                    "r2":                round(r["r2"], 4),
                    "return_std_pp":     round(r["ret_std"], 4),
                },
                "metrics":    {k: round(v, 4) for k, v in r["met"].items()} if r["met"] else {},
                "forecast": {
                    "p10":    round(r["fc_p10"], 4),
                    "p50":    round(r["fc_p50"], 4),
                    "p90":    round(r["fc_p90"], 4),
                    "signal": r["signal"],
                },
                "is_winner":  r["label"] == best["label"],
            }
            for r in results
        ],
    }

    filepath = out_path or (OUTPUTS_DIR / f"{TICKER}_best_config.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return str(filepath)

# ── Comparison plot ───────────────────────────────────────────────────────────

COLORS = ["#4fc3f7", "#ffb74d", "#a5d6a7", "#f48fb1"]   # blue, orange, green, pink

def comparison_plot(results: list, best: dict) -> None:
    import plotly.graph_objects as go            # lazy: batch runs never plot
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            "Price history with per-window trend lines (winner = solid bold)",
            f"Forecast comparison — P10 / P50 / P90  (next {HORIZON} bars, ~1 week)",
            "Reliability score breakdown",
        ],
        row_heights=[0.50, 0.25, 0.25],
        vertical_spacing=0.10,
    )

    # Panel 1 — price (longest window) + one trend line per window
    df_long = max(results, key=lambda r: r["bars"])["df"]
    fig.add_trace(go.Scatter(
        x=df_long.index, y=df_long["Close"],
        name="Close", line=dict(color="#90caf9", width=1.5),
    ), row=1, col=1)

    for r, col in zip(results, COLORS):
        close_arr = r["df"]["Close"].values
        x_arr     = np.arange(len(close_arr), dtype=float)
        slope, intercept = np.polyfit(x_arr, close_arr, 1)
        trend_y   = slope * x_arr + intercept
        is_best   = r["label"] == best["label"]
        s = "+" if r["slope_pct"] >= 0 else ""
        crown = "  ★ WINNER" if is_best else ""
        fig.add_trace(go.Scatter(
            x=r["df"].index, y=trend_y,
            name=f"{r['label']} trend  {s}{r['slope_pct']:.2f}%/d  R²={r['r2']:.2f}  score={r['score']:.2f}{crown}",
            line=dict(color=col, width=3.5 if is_best else 1.8,
                      dash="solid" if is_best else "dash"),
        ), row=1, col=1)

    # Panel 2 — forecast bars
    for r, col in zip(results, COLORS):
        is_best = r["label"] == best["label"]
        s = "+" if r["fc_p50"] >= 0 else ""
        lbl = r["label"] + ("  ★" if is_best else "")
        fig.add_trace(go.Bar(
            x=[r["fc_p50"]], y=[lbl], orientation="h",
            marker=dict(color=col, line=dict(color="white", width=2 if is_best else 0)),
            name=f"{r['label']} P50",
            text=f"{r['signal']}  P50 {s}{r['fc_p50']:.2f}%",
            textposition="inside", width=0.35,
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=[r["fc_p10"], r["fc_p90"]], y=[lbl, lbl],
            mode="markers+text",
            marker=dict(symbol="line-ns-open", size=14, color=col, line_width=2),
            text=[f"P10 {r['fc_p10']:+.1f}%", f"P90 {r['fc_p90']:+.1f}%"],
            textposition=["bottom center", "bottom center"],
            showlegend=False,
        ), row=2, col=1)
    fig.add_vline(x=0, line_dash="dot", line_color="#ef5350", row=2, col=1)

    # Panel 3 — Score breakdown (grouped bars: skills + raw + confidence + final)
    metric_labels = ["Dir", "IC", "Coverage", "Raw score", "Confidence", "FINAL"]
    for r, col in zip(results, COLORS):
        sp = r["score_parts"]
        vals = [sp.get("dir_score",  0),
                sp.get("ic_score",   0),
                sp.get("cov_score",  0),
                sp.get("raw_score",  0),
                sp.get("confidence", 0),
                r["score"]]
        is_best = r["label"] == best["label"]
        fig.add_trace(go.Bar(
            name=r["label"] + ("  ★" if is_best else ""),
            x=metric_labels, y=vals,
            marker=dict(color=col, line=dict(color="white", width=2 if is_best else 0)),
            text=[f"{v:.2f}" for v in vals], textposition="outside",
        ), row=3, col=1)

    winner_text = (f"WINNER: {best['label']}  (score={best['score']:.3f})  "
                   f"→  {best['signal']}  P50 {best['fc_p50']:+.2f}%  "
                   f"(${best['current']*(1+best['fc_p10']/100):.2f}–"
                   f"${best['current']*(1+best['fc_p90']/100):.2f})")
    fig.update_layout(
        title=f"{TICKER}  |  4-Window Auto-Tune  |  {winner_text}",
        template="plotly_dark",
        height=980,
        barmode="group",
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_xaxes(title_text="Predicted ~1-week return (%)", row=2, col=1)
    fig.update_yaxes(title_text="Score (0-1)", row=3, col=1, range=[0, 1.05])
    fig.show()

# ── Main ──────────────────────────────────────────────────────────────────────

def autotune(df_full: pd.DataFrame, ticker: str, sector: str, sector_etf: str,
             out_path=None) -> dict:
    """
    Pure 4-window sweep on an already-built hourly feature frame (see
    build_hourly_features): trains every window, picks the best reliability score
    and saves <ticker>_best_config.json to `out_path` (default outputs/).
    Returns {"payload": saved JSON, "results": per-window runs, "best": winner}.
    """
    global TICKER, _SECTOR_NAME, SECTOR_ETF
    TICKER, _SECTOR_NAME, SECTOR_ETF = ticker, sector, sector_etf
    results = []
    for cfg in WINDOWS:
        if cfg["use_bars"] > len(df_full):
            print(f"\n  [skip] {cfg['label']} needs {cfg['use_bars']} bars, "
                  f"only {len(df_full)} available")
        r = run_window(df_full, cfg["use_bars"], cfg["label"])
        results.append(r)

    best = max(results, key=lambda r: r["score"])
    path = save_best_config(results, best, out_path)
    print(f"\n  ✓ Best config saved → {path}")
    payload = json.loads(open(path, encoding="utf-8").read())
    return {"payload": payload, "results": results, "best": best}


def main():
    global TICKER, _SECTOR_NAME, SECTOR_ETF
    if len(sys.argv) > 1:
        TICKER = sys.argv[1].upper()
    _SECTOR_NAME, SECTOR_ETF = _get_sector_etf(TICKER)
    print(f"\n{'#'*78}")
    print(f"# 4-Window Auto-Tune  —  {TICKER}  ({_SECTOR_NAME} / {SECTOR_ETF})")
    print(f"# Windows  : {', '.join(w['label'] for w in WINDOWS)}")
    print(f"# Score    : ({SCORE_WEIGHTS['dir_acc']*100:.0f}% Dir + "
          f"{SCORE_WEIGHTS['ic']*100:.0f}% IC + "
          f"{SCORE_WEIGHTS['coverage']*100:.0f}% Cov) × √(min(1, n_val/{CONFIDENCE_FULL_AT}))")
    print(f"# Quantile : auto-widen P05/P95 if σ(fwd) ≥ {QUANTILE_WIDEN_VOL}pp, else P10/P90")
    print(f"{'#'*78}")

    df_full = load_full_data()
    print(f"\n  Full dataset : {len(df_full):,} hourly bars  "
          f"[{df_full.index[0]} – {df_full.index[-1]}]")
    print(f"  Device       : {device}")

    out     = autotune(df_full, TICKER, _SECTOR_NAME, SECTOR_ETF)
    results, best = out["results"], out["best"]

    # ── Comparison table ──────────────────────────────────────────────────────
    W = 18
    print(f"\n{'='*88}")
    print(f"  AUTO-TUNE RESULTS — {TICKER}  (interval={INTERVAL}, WINDOW={WINDOW}h, HORIZON={HORIZON}h)")
    print(f"{'='*88}")
    header = f"  {'Metric':<24}" + "".join(
        f"{r['label'] + (' ★' if r['label']==best['label'] else ''):>{W}}" for r in results)
    print(header)
    print("  " + "-"*24 + "".join("-"*W for _ in results))

    def _band_lbl(r):
        return f"P{int(r['quantiles'][0]*100):02d}/P{int(r['quantiles'][-1]*100):02d}"

    rows = [
        ("Bars used",         lambda r: f"{r['bars']:,}"),
        ("Date range start",  lambda r: str(r["df"].index[0])[:10]),
        ("Val sequences",     lambda r: f"{r['n_val']:,}"),
        ("Return σ (1-wk)",   lambda r: f"{r['ret_std']:.2f}pp"),
        ("Quantile band",     lambda r: f"{_band_lbl(r)} ({r['band_pct']}%)"),
        ("Trend direction",   lambda r: r["trend_dir"]),
        ("Trend %/day",       lambda r: f"{r['slope_pct']:+.3f}%"),
        ("Trend R²",          lambda r: f"{r['r2']:.3f}"),
        ("Val RMSE",          lambda r: f"{r['met'].get('rmse', 0):.2f}pp"),
        ("Val DirAcc",        lambda r: f"{r['met'].get('dir_acc', 0)*100:.1f}%"),
        ("Val IC",            lambda r: f"{r['met'].get('ic', 0):+.3f}"),
        ("Val Coverage",      lambda r: f"{r['met'].get('coverage', 0)*100:.0f}%"),
        ("  → Dir skill",     lambda r: f"{r['score_parts'].get('dir_score',  0):.2f}"),
        ("  → IC skill",      lambda r: f"{r['score_parts'].get('ic_score',   0):.2f}"),
        ("  → Coverage skill",lambda r: f"{r['score_parts'].get('cov_score',  0):.2f}"),
        ("  Raw score",       lambda r: f"{r['score_parts'].get('raw_score',  0):.3f}"),
        ("  × Confidence",    lambda r: f"{r['score_parts'].get('confidence', 0):.3f}"),
        ("⚡ FINAL SCORE",    lambda r: f"{r['score']:.3f}"),
        ("Forecast P50",      lambda r: f"{r['fc_p50']:+.2f}% (${r['current']*(1+r['fc_p50']/100):.2f})"),
        ("Signal",            lambda r: r["signal"]),
        ("Train time",        lambda r: f"{r['elapsed']:.0f}s"),
    ]
    for name, fn in rows:
        print(f"  {name:<24}" + "".join(f"{fn(r):>{W}}" for r in results))
    print(f"{'='*88}")

    # ── Winner announcement ──────────────────────────────────────────────────
    print(f"\n{'★'*78}")
    print(f"  WINNER: {best['label']}  (score={best['score']:.3f})")
    print(f"  Signal: {best['signal']}  P50 {best['fc_p50']:+.2f}%  "
          f"→  ${best['current']*(1+best['fc_p50']/100):.2f}")
    print(f"  Range : ${best['current']*(1+best['fc_p10']/100):.2f}  –  "
          f"${best['current']*(1+best['fc_p90']/100):.2f}  (P10–P90)")
    if best["score"] < 0.30:
        print(f"  ⚠  Best score is low ({best['score']:.2f}) — this stock may be "
              "hard to predict with current settings.")
    print(f"{'★'*78}")

    comparison_plot(results, best)


if __name__ == "__main__":
    main()
