"""
predict.py — load <TICKER>_best_config.json (from compare.py) and run a
             fresh forecast using ONLY the winning config (no full sweep).

Usage:
    python predict.py             # loads MSFT_best_config.json
    python predict.py AAPL        # loads AAPL_best_config.json

This trains a fresh ensemble (~30-60s vs ~7min for compare.py) using the
exact hyperparameters saved by compare.py's auto-tuner.
"""

import sys
import time
import json
import yfinance as yf

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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SECTOR_TO_ETF = {
    "Technology": "XLK", "Financial Services": "XLF", "Energy": "XLE",
    "Utilities": "XLU", "Healthcare": "XLV", "Consumer Defensive": "XLP",
    "Consumer Cyclical": "XLY", "Industrials": "XLI", "Real Estate": "XLRE",
    "Basic Materials": "XLB", "Communication Services": "XLC",
}

# ── Load saved config ─────────────────────────────────────────────────────────
TICKER = sys.argv[1].upper() if len(sys.argv) > 1 else "MSFT"
CFG_PATH = f"{TICKER}_best_config.json"

try:
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        SAVED = json.load(f)
except FileNotFoundError:
    print(f"\n❌ No config found at '{CFG_PATH}'.")
    print(f"   Run compare.py first:   python compare.py {TICKER}\n")
    sys.exit(1)

mc = SAVED["model_config"]
INTERVAL     = mc["interval"]
BARS_PER_DAY = mc["bars_per_day"]
USE_BARS     = mc["use_bars"]
WINDOW       = mc["window"]
HORIZON      = mc["horizon"]
TRAIN_RATIO  = mc["train_ratio"]
QUANTILES    = mc["quantiles"]
N_Q          = len(QUANTILES)
HIDDEN_DIM   = mc["hidden_dim"]
NUM_LAYERS   = mc["num_layers"]
DROPOUT      = mc["dropout"]
BATCH_SIZE   = mc["batch_size"]
EPOCHS       = mc["epochs"]
N_ENSEMBLE   = mc["n_ensemble"]
SPEARMAN_W   = mc["spearman_w"]
LR           = mc["lr"]
FEAT_COLS    = mc["feat_cols"]
LR_PATIENCE  = 10
ES_PATIENCE  = 30

# ── Sector lookup ─────────────────────────────────────────────────────────────

def _get_sector_etf(ticker):
    try:
        sector = yf.Ticker(ticker).info.get("sector", "")
        etf    = SECTOR_TO_ETF.get(sector, "")
        if etf:
            return sector, etf
    except Exception:
        pass
    return "Unknown", "SPY"

_SECTOR_NAME, SECTOR_ETF = _get_sector_etf(TICKER)

# ── Data ──────────────────────────────────────────────────────────────────────

def _fetch_close(ticker, period):
    raw = yf.download(ticker, period=period, interval=INTERVAL,
                      prepost=False, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    return raw["Close"].squeeze()


def load_data():
    # Download enough to cover USE_BARS plus indicator warm-up
    # Use "2y" — yfinance hourly cap; we'll slice to USE_BARS at the end
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
        raise RuntimeError(f"No data for '{TICKER}'.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    df = raw[["High", "Low", "Close", "Volume"]].dropna()
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

    print(f"[3/3] Fetching market context: SPY / {SECTOR_ETF}...", end="", flush=True)
    spy  = _fetch_close("SPY",      "2y")
    sect = _fetch_close(SECTOR_ETF, "2y")
    print(" done")
    df["SPY_ret"]  = spy.pct_change(HORIZON).mul(100).reindex(df.index)
    df["SECT_ret"] = sect.pct_change(HORIZON).mul(100).reindex(df.index)

    df = df.dropna()
    if len(df) > USE_BARS:
        df = df.iloc[-USE_BARS:]
    print(f"      Using last {len(df)} bars  [{df.index[0]} – {df.index[-1]}]")
    return df

# ── Sequences / model / training / loss (identical to compare.py) ─────────────

def make_sequences(scaled, close_raw):
    max_i = len(scaled) - WINDOW - HORIZON
    if max_i <= 0:
        raise ValueError(f"Not enough data: {len(scaled)} bars.")
    X    = np.stack([scaled[i:i+WINDOW] for i in range(max_i)]).astype(np.float32)
    base = close_raw[WINDOW-1         : WINDOW-1         + max_i]
    fwd  = close_raw[WINDOW-1+HORIZON : WINDOW-1+HORIZON + max_i]
    y    = (100.0 * (fwd - base) / base).astype(np.float32)
    return X, y

def to_tensors(X, y):
    return torch.tensor(X), torch.tensor(y).unsqueeze(1)


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


def train_model(model, tr_ld, va_ld, label=""):
    opt  = torch.optim.Adam(model.parameters(), lr=LR)
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=LR_PATIENCE, factor=0.5)
    best_val, best_w, no_imp = float("inf"), None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = combined_loss(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
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


def predict_ensemble(models, X_t):
    preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            preds.append(m(X_t.to(device)).cpu().numpy())
    return np.mean(preds, axis=0)


def compute_metrics(pred_q, true):
    p10, p50, p90 = pred_q[:,0], pred_q[:,1], pred_q[:,2]
    return dict(
        rmse     = float(np.sqrt(np.mean((p50 - true)**2))),
        dir_acc  = float(np.mean(np.sign(p50) == np.sign(true))),
        ic       = float(np.corrcoef(p50, true)[0, 1]) if len(p50) > 1 else 0.0,
        coverage = float(np.mean((p10 <= true) & (true <= p90))),
    )


def _trend_slope(close_arr):
    x = np.arange(len(close_arr), dtype=float)
    slope, intercept = np.polyfit(x, close_arr, 1)
    yhat   = slope * x + intercept
    ss_res = np.sum((close_arr - yhat) ** 2)
    ss_tot = np.sum((close_arr - close_arr.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return 100.0 * slope / close_arr[0] * BARS_PER_DAY, r2, slope, intercept


def plot(df, true_r_val, pred_q_val, val_dates,
         fc_p10, fc_p50, fc_p90, current_close):
    slope_pct, r2, slope, intercept = _trend_slope(df["Close"].values)
    trend_line = slope * np.arange(len(df)) + intercept
    trend_dir  = "UP" if slope_pct > 0 else "DOWN"
    t_p10 = current_close * (1 + fc_p10/100)
    t_p50 = current_close * (1 + fc_p50/100)
    t_p90 = current_close * (1 + fc_p90/100)
    signal = "BULLISH" if fc_p50 > 0 else "BEARISH"
    q_lo  = f"P{int(QUANTILES[0]*100):02d}"
    q_hi  = f"P{int(QUANTILES[-1]*100):02d}"

    future_dates = pd.bdate_range(start=df.index[-1] + pd.Timedelta(hours=1),
                                  periods=HORIZON, freq="bh")

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            f"Price — {len(df)} bars  |  Trend: {trend_dir}  {slope_pct:+.2f}%/day  (R²={r2:.2f})",
            f"Validation — P50 vs actual {HORIZON}-bar return",
            f"Forecast: {q_lo}/P50/{q_hi} (~1 week)",
        ],
        vertical_spacing=0.10,
        row_heights=[0.55, 0.22, 0.23],
    )

    fig.add_trace(go.Scatter(x=df.index, y=df["Close"],
                             name="Close", line=dict(color="#4fc3f7", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA_9"],
                             name="EMA 9", line=dict(color="#ffb74d", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA_20"],
                             name="EMA 20", line=dict(color="#ef5350", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=trend_line,
                             name=f"Trend {slope_pct:+.2f}%/d",
                             line=dict(color="#b39ddb", width=1.5, dash="dash")), row=1, col=1)

    fc_x    = [df.index[-1]] + list(future_dates)
    fc_high = [current_close] + [t_p90] * HORIZON
    fc_low  = [current_close] + [t_p10] * HORIZON
    fc_mid  = [current_close] + [t_p50] * HORIZON
    fig.add_trace(go.Scatter(
        x=fc_x + list(reversed(fc_x)),
        y=fc_high + list(reversed(fc_low)),
        fill="toself", fillcolor="rgba(126,87,194,0.18)",
        line=dict(color="rgba(0,0,0,0)"), name=f"{q_lo}–{q_hi} cone", hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=fc_x, y=fc_mid, name="P50 forecast",
                             line=dict(color="#7e57c2", width=2, dash="dash")), row=1, col=1)

    if pred_q_val is not None and len(pred_q_val) >= 2:
        p10v, p50v, p90v = pred_q_val[:,0], pred_q_val[:,1], pred_q_val[:,2]
        vd = list(val_dates)
        fig.add_trace(go.Scatter(
            x=vd + list(reversed(vd)),
            y=list(p90v) + list(reversed(p10v)),
            fill="toself", fillcolor="rgba(255,183,77,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name=f"{q_lo}–{q_hi} band", hoverinfo="skip",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(x=val_dates, y=true_r_val, name="Actual",
                                 line=dict(color="#4fc3f7", width=1.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=val_dates, y=p50v, name="P50",
                                 line=dict(color="#ffb74d", dash="dash", width=1.5)), row=2, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

    s = "+" if fc_p50 >= 0 else ""
    fig.add_trace(go.Bar(
        x=[fc_p50], y=["Next ~1wk"], orientation="h",
        marker_color="#7e57c2",
        text=f"P50 {s}{fc_p50:.2f}%  →  ${t_p50:.2f}",
        textposition="inside", width=0.4,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=[fc_p10, fc_p90], y=["Next ~1wk", "Next ~1wk"],
        mode="markers+text",
        marker=dict(symbol="line-ns-open", size=16, color="white", line_width=2),
        text=[f"{q_lo} {fc_p10:+.1f}%  (${t_p10:.2f})",
              f"{q_hi} {fc_p90:+.1f}%  (${t_p90:.2f})"],
        textposition=["bottom center", "bottom center"],
        showlegend=False,
    ), row=3, col=1)
    fig.add_vline(x=0, line_dash="dot", line_color="#ef5350", row=3, col=1)

    fig.update_layout(
        title=(f"{TICKER}  |  Loaded config: {SAVED['best_window']}  |  "
               f"Trend {trend_dir} {slope_pct:+.2f}%/d  |  "
               f"Signal: {signal}  {s}{fc_p50:.2f}%  |  "
               f"Range ${t_p10:.2f}–${t_p90:.2f}"),
        template="plotly_dark", height=900, hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Return (%)", row=2, col=1)
    fig.update_xaxes(title_text=f"Predicted ~1-week return (%)", row=3, col=1)
    fig.show()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    band_pct = int((QUANTILES[-1] - QUANTILES[0]) * 100)
    q_lo = f"P{int(QUANTILES[0]*100):02d}"
    q_hi = f"P{int(QUANTILES[-1]*100):02d}"

    print(f"\n{'#'*78}")
    print(f"# Predict (saved config)  —  {TICKER}  ({_SECTOR_NAME} / {SECTOR_ETF})")
    print(f"# Source   : {CFG_PATH}")
    print(f"# Window   : {SAVED['best_window']}  (use_bars={USE_BARS})")
    print(f"# Quantile : {q_lo}/P50/{q_hi}  ({band_pct}% band)  "
          f"(σ_train was {SAVED['best_trend'].get('return_std_pp', 0):.2f}pp)")
    print(f"# Saved score (last sweep): {SAVED['best_score']:.3f}")
    print(f"{'#'*78}")

    df        = load_data()
    values    = df[FEAT_COLS].values
    close_raw = df["Close"].values
    n, n_feat = values.shape

    scaler = MinMaxScaler()
    scaler.fit(values)
    scaled = scaler.transform(values)

    X, y   = make_sequences(scaled, close_raw)
    tr_end = int(len(X) * TRAIN_RATIO)
    X_tr, y_tr = X[:tr_end],  y[:tr_end]
    X_va, y_va = X[tr_end:],  y[tr_end:]

    tr_ld = DataLoader(TensorDataset(*to_tensors(X_tr, y_tr)), BATCH_SIZE, shuffle=True)
    va_ld = DataLoader(TensorDataset(*to_tensors(X_va, y_va)), BATCH_SIZE)

    print(f"\n  Sequences : {len(X):,} total  |  {len(X_tr):,} train / {len(X_va):,} val")
    print(f"  Device    : {device}\n")

    t0 = time.time()
    models = []
    for seed in range(N_ENSEMBLE):
        torch.manual_seed(seed)
        np.random.seed(seed)
        m  = LSTMTrendModel(n_feat).to(device)
        bv = train_model(m, tr_ld, va_ld, label=f"m{seed+1}/{N_ENSEMBLE}")
        print(f"    => best_val={bv:.4f}")
        models.append(m)
    elapsed = time.time() - t0

    # Validation eval
    pred_q_val, true_r_val, val_dates = None, None, []
    if len(X_va) >= 2:
        X_va_t, y_va_t = to_tensors(X_va, y_va)
        pred_q_val = predict_ensemble(models, X_va_t)
        true_r_val = y_va_t.squeeze(1).numpy()
        met        = compute_metrics(pred_q_val, true_r_val)
        val_dates  = df.index[tr_end + WINDOW - 1 : tr_end + WINDOW - 1 + len(y_va)]
        print(f"\n  Val metrics  RMSE={met['rmse']:.2f}pp  "
              f"DirAcc={met['dir_acc']*100:.1f}%  IC={met['ic']:+.3f}  "
              f"Coverage={met['coverage']*100:.0f}%")

    # Forecast
    current_close = float(df["Close"].iloc[-1])
    last_t = torch.tensor(scaled[-WINDOW:], dtype=torch.float32).unsqueeze(0)
    fc_q   = predict_ensemble(models, last_t).squeeze(0)
    fc_p10, fc_p50, fc_p90 = float(fc_q[0]), float(fc_q[1]), float(fc_q[2])
    t_p10 = current_close * (1 + fc_p10/100)
    t_p50 = current_close * (1 + fc_p50/100)
    t_p90 = current_close * (1 + fc_p90/100)
    signal = "BULLISH" if fc_p50 > 0 else "BEARISH"

    slope_pct, r2, _, _ = _trend_slope(close_raw)

    print(f"\n{'='*78}")
    print(f"  FORECAST  —  {TICKER}  from {df.index[-1]}  ({elapsed:.0f}s training)")
    print(f"  Trend       : {'UP' if slope_pct > 0 else 'DOWN'}  "
          f"{slope_pct:+.3f}%/day  (R²={r2:.2f})")
    print(f"  Current     : ${current_close:.2f}")
    print(f"  {q_lo}/P50/{q_hi} : {fc_p10:+.2f}% / {fc_p50:+.2f}% / {fc_p90:+.2f}%")
    print(f"  Range       : ${t_p10:.2f} – ${t_p90:.2f}  (mid: ${t_p50:.2f})")
    print(f"  Signal      : {signal}")
    print(f"{'='*78}")

    plot(df, true_r_val, pred_q_val, val_dates,
         fc_p10, fc_p50, fc_p90, current_close)


if __name__ == "__main__":
    main()
