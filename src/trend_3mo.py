import sys
import time
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

from paths import MODELS_DIR

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Config ────────────────────────────────────────────────────────────────────
# ↓ The ONLY line you need to change when switching companies ↓
TICKER        = "MSFT"
# ─────────────────────────────────────────────────────────────────────────────
INTERVAL      = "1h"     # yfinance bar interval (supports 1h for up to 730 days)
BARS_PER_DAY  = 7        # regular-session hourly bars per trading day (9:30–3:30 ET)
WARMUP_PERIOD = "1y"     # download period — extra history so indicators warm up
USE_BARS      = 882      # last N hourly bars used (~6 calendar months, 126d × 7)
WINDOW        = 70       # look-back window (10 trading days × 7 bars)
HORIZON       = 35       # forecast horizon (5 trading days × 7 bars ≈ 1 week)
TRAIN_RATIO   = 0.75     # fraction of sequences for training; rest = validation

QUANTILES     = [0.10, 0.50, 0.90]
N_Q           = len(QUANTILES)

EPOCHS        = 200
BATCH_SIZE    = 32       # more data now, larger batches are fine
HIDDEN_DIM    = 32
NUM_LAYERS    = 1
DROPOUT       = 0.30
SPEARMAN_W    = 0.25
LR            = 1e-3
LR_PATIENCE   = 10
ES_PATIENCE   = 30
N_ENSEMBLE    = 3

SECTOR_TO_ETF = {
    "Technology":             "XLK",
    "Financial Services":     "XLF",
    "Energy":                 "XLE",
    "Utilities":              "XLU",
    "Healthcare":             "XLV",
    "Consumer Defensive":     "XLP",
    "Consumer Cyclical":      "XLY",
    "Industrials":            "XLI",
    "Real Estate":            "XLRE",
    "Basic Materials":        "XLB",
    "Communication Services": "XLC",
}


def _get_sector_etf(ticker: str) -> tuple:
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

def _fetch_close(ticker: str, period: str) -> pd.Series:
    raw = yf.download(ticker, period=period, interval=INTERVAL,
                      prepost=False, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    return raw["Close"].squeeze()


def load_data(ticker: str) -> pd.DataFrame:
    print(f"\n[1/3] Downloading {ticker} ({WARMUP_PERIOD}, interval={INTERVAL})...")
    raw = pd.DataFrame()
    for attempt in range(1, 4):
        raw = yf.download(ticker, period=WARMUP_PERIOD, interval=INTERVAL,
                          prepost=False, auto_adjust=True)
        if not raw.empty:
            break
        print(f"      attempt {attempt}/3 empty — retrying in 3s...")
        time.sleep(3)
    if raw.empty:
        raise RuntimeError(
            f"yfinance returned no data for '{ticker}' after 3 attempts. "
            "Check the ticker symbol and your internet connection."
        )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    df = raw[["High", "Low", "Close", "Volume"]].dropna()
    print(f"      {len(df)} bars  [{df.index[0]} – {df.index[-1]}]")

    high, low, close, volume = df["High"], df["Low"], df["Close"], df["Volume"]

    # Short-period indicators suited for 3-month analysis
    print("[2/3] Computing technical indicators...", end="", flush=True)
    df["EMA_9"]      = ta.trend.EMAIndicator(close=close, window=9).ema_indicator()
    df["EMA_20"]     = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
    df["EMA_spread"] = df["EMA_9"] - df["EMA_20"]   # positive = short-term bullish
    df["RSI_14"]     = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    df["MACD"]       = ta.trend.MACD(close=close).macd_diff()
    df["OBV"]        = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
    bb               = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    df["BB_width"]   = bb.bollinger_wband()
    df["BB_pos"]     = bb.bollinger_pband()   # %B: position within band (0=low, 1=high)
    df["ATR_14"]     = ta.volatility.AverageTrueRange(
                           high=high, low=low, close=close, window=14).average_true_range()
    df["STOCH_K"]    = ta.momentum.StochasticOscillator(
                           high=high, low=low, close=close, window=14, smooth_window=3).stoch()
    print(" done  (EMA9/20, RSI, MACD, OBV, BB, ATR, STOCH)")

    # 5-day market context (1-week momentum — more relevant than 20-day for short-term)
    print(f"[3/3] Fetching market context: SPY / {SECTOR_ETF}...", end="", flush=True)
    spy  = _fetch_close("SPY",      WARMUP_PERIOD)
    sect = _fetch_close(SECTOR_ETF, WARMUP_PERIOD)
    print(" done")

    # 5-day equiv in hourly bars = HORIZON (35) for momentum alignment
    df["SPY_ret"]  = spy.pct_change(HORIZON).mul(100).reindex(df.index)
    df["SECT_ret"] = sect.pct_change(HORIZON).mul(100).reindex(df.index)

    df = df.dropna()

    # Slice to last USE_BARS hourly bars
    if len(df) > USE_BARS:
        df = df.iloc[-USE_BARS:]
    print(f"      Using last {len(df)} bars  [{df.index[0]} – {df.index[-1]}]")

    return df


FEAT_COLS = [
    "Close", "Volume",
    "EMA_9", "EMA_20", "EMA_spread",
    "RSI_14", "MACD", "OBV",
    "BB_width", "BB_pos", "ATR_14", "STOCH_K",
    "SPY_ret", "SECT_ret",
]


def make_sequences(scaled: np.ndarray, close_raw: np.ndarray,
                   window: int, horizon: int):
    max_i = len(scaled) - window - horizon
    if max_i <= 0:
        raise ValueError(
            f"Not enough data: {len(scaled)} rows but need at least {window + horizon + 1}. "
            "Increase USE_DAYS or reduce WINDOW/HORIZON."
        )
    X    = np.stack([scaled[i : i + window] for i in range(max_i)]).astype(np.float32)
    base = close_raw[window - 1           : window - 1           + max_i]
    fwd  = close_raw[window - 1 + horizon : window - 1 + horizon + max_i]
    y    = (100.0 * (fwd - base) / base).astype(np.float32)
    return X, y


def to_tensors(X: np.ndarray, y: np.ndarray):
    return torch.tensor(X), torch.tensor(y).unsqueeze(1)


# ── Model ─────────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(lstm_out), dim=1)
        return (w * lstm_out).sum(dim=1)


class LSTMTrendModel(nn.Module):
    """Lightweight LSTM + attention for short-term (3-month) tendency detection."""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 dropout: float, n_quantiles: int = N_Q):
        super().__init__()
        # dropout only applies between LSTM layers; set to 0 if num_layers=1
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=lstm_dropout)
        self.attn = Attention(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_dim, n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(self.drop(self.attn(out)))


# ── Losses ────────────────────────────────────────────────────────────────────

def pinball_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    total = 0.0
    for i, q in enumerate(QUANTILES):
        err    = true - pred[:, i : i + 1]
        total += torch.where(err >= 0, q * err, (q - 1) * err).mean()
    return total / N_Q


def soft_spearman_loss(pred_p50: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    p   = pred_p50.squeeze(1)
    t   = true.squeeze(1)
    tau = (p.detach().std() + 1e-4).item()
    r_p = torch.sigmoid((p.unsqueeze(0) - p.unsqueeze(1)) / tau).sum(dim=1)
    r_t = torch.sigmoid((t.unsqueeze(0) - t.unsqueeze(1)) / tau).sum(dim=1)
    rp_c = r_p - r_p.mean()
    rt_c = r_t - r_t.mean()
    rho  = (rp_c * rt_c).sum() / (
        torch.sqrt((rp_c ** 2).sum() * (rt_c ** 2).sum()) + 1e-8
    )
    return 1.0 - rho


def combined_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return ((1 - SPEARMAN_W) * pinball_loss(pred, true)
            + SPEARMAN_W    * soft_spearman_loss(pred[:, 1:2], true))


# ── Training ──────────────────────────────────────────────────────────────────

def train(model: nn.Module, train_loader: DataLoader,
          val_loader: DataLoader, label: str = "") -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=LR_PATIENCE, factor=0.5)
    best_val, best_w, no_imp = float("inf"), None, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = combined_loss(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item() * len(xb)
        t_loss /= max(len(train_loader.dataset), 1)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                v_loss += combined_loss(model(xb), yb).item() * len(xb)
        v_loss /= max(len(val_loader.dataset), 1)
        scheduler.step(v_loss)

        bar_done = int(20 * epoch / EPOCHS)
        bar      = "#" * bar_done + "-" * (20 - bar_done)
        print(f"\r    {label}  [{bar}] ep {epoch:3d}/{EPOCHS}"
              f"  train={t_loss:.4f}  val={v_loss:.4f}",
              end="", flush=True)

        if v_loss < best_val:
            best_val = v_loss
            best_w   = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp   = 0
        else:
            no_imp += 1
            if no_imp >= ES_PATIENCE:
                print()
                break
    else:
        print()

    model.load_state_dict(best_w)
    return best_val


# ── Evaluation ────────────────────────────────────────────────────────────────

def compute_metrics(pred_q: np.ndarray, true: np.ndarray) -> dict:
    p10, p50, p90 = pred_q[:, 0], pred_q[:, 1], pred_q[:, 2]
    return dict(
        rmse     = float(np.sqrt(np.mean((p50 - true) ** 2))),
        dir_acc  = float(np.mean(np.sign(p50) == np.sign(true))),
        ic       = float(np.corrcoef(p50, true)[0, 1]) if len(p50) > 1 else 0.0,
        coverage = float(np.mean((p10 <= true) & (true <= p90))),
    )


def predict_ensemble(models: list, X_t: torch.Tensor) -> np.ndarray:
    preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            preds.append(m(X_t.to(device)).cpu().numpy())
    return np.mean(preds, axis=0)


# ── Trend analysis ────────────────────────────────────────────────────────────

def _trend_slope(close_arr: np.ndarray) -> tuple:
    """
    Linear regression over close_arr (hourly bars).
    Returns (slope_pct_per_day, R², raw_slope, intercept).
    Converts %/bar → %/day by multiplying by BARS_PER_DAY for readability.
    """
    x = np.arange(len(close_arr), dtype=float)
    slope, intercept = np.polyfit(x, close_arr, 1)
    yhat   = slope * x + intercept
    ss_res = np.sum((close_arr - yhat) ** 2)
    ss_tot = np.sum((close_arr - close_arr.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    pct_per_day = 100.0 * slope / close_arr[0] * BARS_PER_DAY
    return pct_per_day, r2, slope, intercept


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(df: pd.DataFrame,
         pred_q_val, true_r_val,
         fc_p10: float, fc_p50: float, fc_p90: float,
         current_close: float, val_signal_dates) -> None:

    slope_pct, r2, slope, intercept = _trend_slope(df["Close"].values)
    trend_line = slope * np.arange(len(df)) + intercept
    trend_dir  = "UP" if slope_pct > 0 else "DOWN"

    target_p10 = current_close * (1 + fc_p10 / 100)
    target_p50 = current_close * (1 + fc_p50 / 100)
    target_p90 = current_close * (1 + fc_p90 / 100)
    signal     = "BULLISH" if fc_p50 > 0 else "BEARISH"

    last_date    = df.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(hours=1),
                                  periods=HORIZON, freq="bh")

    n_panels = 3 if (pred_q_val is not None and len(pred_q_val) >= 2) else 2
    subplot_titles = [
        f"Price — last {len(df)} hourly bars (~6 months)  |  Trend: {trend_dir}  {slope_pct:+.2f}%/day  (R²={r2:.2f})",
    ]
    row_heights = [0.55]
    if n_panels == 3:
        subplot_titles.append(f"Validation — P50 predicted vs actual {HORIZON}-bar return (%)")
        row_heights.append(0.22)
    subplot_titles.append(f"Ensemble forecast: next {HORIZON} hourly bars (~1 week)")
    row_heights.append(0.23)

    fig = make_subplots(
        rows=n_panels, cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.10,
        row_heights=row_heights,
    )

    # Panel 1: Price + EMAs + linear trend + forecast cone
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"],
                             name="Close", line=dict(color="#4fc3f7", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA_9"],
                             name="EMA 9",
                             line=dict(color="#ffb74d", width=1.2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA_20"],
                             name="EMA 20",
                             line=dict(color="#ef5350", width=1.2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=trend_line,
                             name=f"Trend ({slope_pct:+.2f}%/day)",
                             line=dict(color="#b39ddb", width=1.5, dash="dash")), row=1, col=1)

    if len(future_dates) > 0:
        fc_x    = [last_date] + list(future_dates)
        fc_high = [current_close] + [target_p90] * HORIZON
        fc_low  = [current_close] + [target_p10] * HORIZON
        fc_mid  = [current_close] + [target_p50] * HORIZON
        fig.add_trace(go.Scatter(
            x=fc_x + list(reversed(fc_x)),
            y=fc_high + list(reversed(fc_low)),
            fill="toself", fillcolor="rgba(126,87,194,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="P10–P90 cone", hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=fc_x, y=fc_mid,
                                 name="P50 forecast",
                                 line=dict(color="#7e57c2", width=2, dash="dash")), row=1, col=1)

    # Panel 2 (optional): Validation predictions vs actuals
    val_row = 2
    if n_panels == 3 and pred_q_val is not None:
        p10v, p50v, p90v = pred_q_val[:, 0], pred_q_val[:, 1], pred_q_val[:, 2]
        fd = list(val_signal_dates)
        fig.add_trace(go.Scatter(
            x=fd + list(reversed(fd)),
            y=list(p90v) + list(reversed(p10v)),
            fill="toself", fillcolor="rgba(255,183,77,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="P10–P90 band", hoverinfo="skip",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(x=val_signal_dates, y=true_r_val,
                                 name="Actual return",
                                 line=dict(color="#4fc3f7", width=1.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=val_signal_dates, y=p50v,
                                 name="P50 predicted",
                                 line=dict(color="#ffb74d", dash="dash", width=1.5)), row=2, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)
        val_row = 3

    # Final panel: forecast bar
    s = "+" if fc_p50 >= 0 else ""
    fig.add_trace(go.Bar(
        x=[fc_p50], y=["Next 5d"], orientation="h",
        marker_color="#7e57c2",
        name=f"P50: {s}{fc_p50:.2f}%",
        text=f"P50  {s}{fc_p50:.2f}%  (~1 week)", textposition="inside", width=0.4,
    ), row=val_row, col=1)
    fig.add_trace(go.Scatter(
        x=[fc_p10, fc_p90], y=["Next 5d", "Next 5d"],
        mode="markers+text",
        marker=dict(symbol="line-ns-open", size=16, color="white", line_width=2),
        text=[f"P10 {fc_p10:+.1f}%", f"P90 {fc_p90:+.1f}%"],
        textposition=["bottom center", "bottom center"],
        name="P10 / P90",
    ), row=val_row, col=1)
    fig.add_vline(x=0, line_dash="dot", line_color="#ef5350", row=val_row, col=1)

    fig.update_layout(
        title=(f"{TICKER}  |  3-Month Tendency  |  "
               f"Trend: {trend_dir} {slope_pct:+.2f}%/day  |  "
               f"Signal: {signal}  {s}{fc_p50:.2f}% ({HORIZON}d)  |  "
               f"Range ${target_p10:.2f}–${target_p90:.2f}"),
        template="plotly_dark",
        height=900,
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    if n_panels == 3:
        fig.update_yaxes(title_text="Return (%)", row=2, col=1)
    fig.update_xaxes(title_text=f"Predicted {HORIZON}-bar (~1 week) return (%)", row=val_row, col=1)
    fig.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df        = load_data(TICKER)
    values    = df[FEAT_COLS].values
    close_raw = df["Close"].values
    n, n_feat = values.shape

    slope_pct, r2, _, _ = _trend_slope(close_raw)
    trend_dir = "UP" if slope_pct > 0 else "DOWN"

    print("=" * 70)
    print(f"  Ticker      : {TICKER}")
    print(f"  Sector ETF  : {SECTOR_ETF}  "
          f"({'SPY fallback' if _SECTOR_NAME == 'Unknown' else _SECTOR_NAME})")
    print(f"  Data window : {n} hourly bars  [{df.index[0]} – {df.index[-1]}]")
    print(f"  Features    : {n_feat}")
    print(f"  3-mo trend  : {trend_dir}  {slope_pct:+.3f}%/day  (R²={r2:.2f})")
    print("=" * 70)

    # Scaler fitted on all USE_DAYS rows — we are forecasting, not strict backtesting
    scaler = MinMaxScaler()
    scaler.fit(values)
    scaled = scaler.transform(values)

    X, y   = make_sequences(scaled, close_raw, WINDOW, HORIZON)
    n_seq  = len(X)
    tr_end = int(n_seq * TRAIN_RATIO)

    X_tr, y_tr = X[:tr_end], y[:tr_end]
    X_va, y_va = X[tr_end:], y[tr_end:]

    print(f"\n  Sequences : {n_seq} total  |  {len(X_tr)} train / {len(X_va)} val  "
          f"(window={WINDOW}h, horizon={HORIZON}h)")
    if len(X_tr) < 5 or len(X_va) < 2:
        print("  WARNING: very few sequences — forecast direction is indicative only.")

    tr_ld = DataLoader(TensorDataset(*to_tensors(X_tr, y_tr)), BATCH_SIZE, shuffle=True)
    va_ld = DataLoader(TensorDataset(*to_tensors(X_va, y_va)), BATCH_SIZE)

    print(f"\n{'=' * 70}")
    print(f"Training ensemble ({N_ENSEMBLE} seeds)  —  device: {device}")
    print(f"{'=' * 70}")

    models = []
    for seed in range(N_ENSEMBLE):
        torch.manual_seed(seed)
        np.random.seed(seed)
        m  = LSTMTrendModel(n_feat, HIDDEN_DIM, NUM_LAYERS, DROPOUT, N_Q).to(device)
        bv = train(m, tr_ld, va_ld, label=f"Model {seed + 1}/{N_ENSEMBLE}")
        print(f"    => best_val={bv:.4f}")
        models.append(m)
        torch.save(m.state_dict(), MODELS_DIR / f"{TICKER}_trend3mo_e{seed}.pt")

    # Validation metrics
    pred_q_val, true_r_val, val_signal_dates = None, None, []
    if len(X_va) >= 2:
        X_va_t, y_va_t = to_tensors(X_va, y_va)
        pred_q_val     = predict_ensemble(models, X_va_t)
        true_r_val     = y_va_t.squeeze(1).numpy()
        met            = compute_metrics(pred_q_val, true_r_val)
        val_signal_dates = df.index[tr_end + WINDOW - 1 : tr_end + WINDOW - 1 + len(y_va)]
        print(f"\nValidation  RMSE={met['rmse']:.2f}pp  DirAcc={met['dir_acc']*100:.1f}%"
              f"  IC={met['ic']:.3f}  Coverage={met['coverage']*100:.0f}%"
              f"  ({len(X_va)} sequences)")

    # Forecast
    current_close = float(df["Close"].iloc[-1])
    last_window   = torch.tensor(scaled[-WINDOW:], dtype=torch.float32).unsqueeze(0)
    fc_q          = predict_ensemble(models, last_window).squeeze(0)
    fc_p10, fc_p50, fc_p90 = float(fc_q[0]), float(fc_q[1]), float(fc_q[2])

    t_p10  = current_close * (1 + fc_p10 / 100)
    t_p50  = current_close * (1 + fc_p50 / 100)
    t_p90  = current_close * (1 + fc_p90 / 100)
    signal = "BULLISH" if fc_p50 > 0 else "BEARISH"

    print(f"\n{'=' * 70}")
    print(f"  3-Month Tendency  →  {trend_dir}  {slope_pct:+.3f}%/day  (R²={r2:.2f})")
    print(f"  LSTM Forecast ({HORIZON}-bar / ~1 week from {df.index[-1]}):")
    print(f"    P10 / P50 / P90 : {fc_p10:+.2f}% / {fc_p50:+.2f}% / {fc_p90:+.2f}%")
    print(f"    Current close   : ${current_close:.2f}")
    print(f"    Implied range   : ${t_p10:.2f} – ${t_p90:.2f}  (mid: ${t_p50:.2f})")
    print(f"    Signal          : {signal}")
    print(f"{'=' * 70}")

    plot(df, pred_q_val, true_r_val, fc_p10, fc_p50, fc_p90,
         current_close, val_signal_dates)


if __name__ == "__main__":
    main()
