import os, sys, warnings, traceback
warnings.filterwarnings("ignore")

# ── 안전한 import - 실패해도 앱이 죽지 않도록 ─────────────────────────
_import_errors = {}

try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_OK = True
except Exception as _e:
    genai = None
    genai_types = None
    _GENAI_OK = False
    _import_errors["google-genai"] = str(_e)

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date

try:
    import numpy as np
except Exception as _e:
    _import_errors["numpy"] = str(_e)
    raise  # numpy 없으면 앱 실행 불가

try:
    import pandas as pd
except Exception as _e:
    _import_errors["pandas"] = str(_e)
    raise

import streamlit as st

if _import_errors:
    st.warning(f"Some packages failed to import: {_import_errors}")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception as _e:
    _import_errors["plotly"] = str(_e)
    st.error(f"Plotly import failed: {_e}")
    st.stop()

try:
    import yfinance as yf
except Exception as _e:
    _import_errors["yfinance"] = str(_e)
    st.error(f"yfinance import failed: {_e}")
    st.stop()

# Optional FRED stack — used only when an API key is present
try:
    from fredapi import Fred
    _FREDAPI_OK = True
except Exception as _e:
    Fred = None
    _FREDAPI_OK = False
    _import_errors["fredapi"] = str(_e)

try:
    import pandas_datareader.data as pdr
    _PDR_OK = True
except Exception as _e:
    pdr = None
    _PDR_OK = False
    _import_errors["pandas-datareader"] = str(_e)

try:
    from src.methodology import render as render_methodology
except Exception as _e:
    def render_methodology(tab_key, st):  # graceful no-op fallback
        return
    _import_errors["methodology"] = str(_e)

try:
    from src.quant_signals import run_quant_scan, DEFAULT_UNIVERSE as QS_DEFAULT_UNIVERSE
    _QS_OK = True
except Exception as _e:
    _QS_OK = False
    QS_DEFAULT_UNIVERSE = ("SPY","QQQ","IWM","GLD","TLT","AAPL","MSFT","NVDA","AMZN","GOOGL",
                            "JPM","BAC","XOM","CVX","JNJ","UNH","WMT","HD","PG")
    _import_errors["quant_signals"] = str(_e)

try:
    from src.bigquery_service import (
        is_bigquery_available,
        get_bq_status,
        load_macro_from_bigquery,
        save_macro_to_bigquery,
        load_quant_signals_from_bigquery,
        save_quant_signals_to_bigquery,
    )
    _BQ_OK = is_bigquery_available()
except Exception as _e:
    _BQ_OK = False
    _import_errors["bigquery"] = str(_e)

# ── Secrets ──────────────────────────────────────────────────────────
def _get_fred_key():
    try:
        return st.secrets.get("FRED_API_KEY")
    except Exception:
        return os.environ.get("FRED_API_KEY")

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(page_title="Macro Pulse", page_icon="⚡", layout="wide",
                   initial_sidebar_state="expanded")


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg-primary: #050810;
  --bg-secondary: #0a0e1c;
  --bg-accent: #0f172a;
  --text-primary: #f8fafc;
  --text-secondary: #94a3b8;
  --accent-blue: #38bdf8;
  --accent-indigo: #818cf8;
  --success: #34d399;
  --danger: #f87171;
  --warning: #facc15;
}

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    color: var(--text-primary);
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 50% 0%, #111827 0%, #050810 100%);
}

[data-testid="stSidebar"] {
    background: #070a14;
    border-right: 1px solid rgba(56, 189, 248, 0.1);
    box-shadow: 10px 0 30px rgba(0,0,0,0.5);
}

/* Glassmorphism containers */
[data-testid="metric-container"] {
    background: rgba(15, 23, 42, 0.6) !important;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 24px 20px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

[data-testid="metric-container"]:hover {
    transform: translateY(-4px);
    border-color: rgba(56, 189, 248, 0.3);
    background: rgba(15, 23, 42, 0.8) !important;
    box-shadow: 0 12px 40px 0 rgba(56, 189, 248, 0.15);
}

[data-testid="stMetricLabel"] {
    color: var(--text-secondary);
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

[data-testid="stMetricValue"] {
    color: #ffffff;
    font-size: 1.75rem;
    font-weight: 700;
    text-shadow: 0 0 20px rgba(56, 189, 248, 0.3);
}

/* Tab styling overhaul */
[data-baseweb="tab-list"] {
    background: transparent;
    padding: 0 20px;
    gap: 8px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}

[data-baseweb="tab"] {
    height: 48px;
    border-radius: 8px 8px 0 0;
    transition: all 0.2s ease;
    padding: 0 24px;
    font-weight: 600;
    color: var(--text-secondary);
}

[aria-selected="true"] {
    background: linear-gradient(to top, rgba(56, 189, 248, 0.15), transparent) !important;
    color: var(--accent-blue) !important;
    border-bottom: 3px solid var(--accent-blue) !important;
}

/* Dataframe refinement */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.05);
}

/* Custom classes */
.kpi-title {
    background: linear-gradient(135deg, #fff 0%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Animation for loading/transitions */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.stTabs [data-testid="stVerticalBlock"] > div {
    animation: fadeIn 0.4s ease-out forwards;
}

/* HIGH READABILITY SECTION */
.analysis-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 32px;
    margin: 20px 0;
    line-height: 1.8;
    color: #f1f5f9;
    font-size: 1.1rem;
    box-shadow: inset 0 0 20px rgba(0,0,0,0.2);
}

.analysis-card h1, .analysis-card h2, .analysis-card h3 {
    color: #ffffff;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
    font-weight: 700;
}

.analysis-card p, .analysis-card li {
    color: #cbd5e1;
    margin-bottom: 1rem;
}

.analysis-card strong {
    color: #38bdf8;
    font-weight: 600;
}

/* Global markdown fixes for dark mode */
.stMarkdown p, .stMarkdown li {
    color: #cbd5e1 !important;
}

.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
    color: #f8fafc !important;
}

</style>
""", unsafe_allow_html=True)

PT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.5)",
          font_color="#cbd5e1", margin=dict(l=60,r=30,t=60,b=60),
          font=dict(family="Outfit, sans-serif", size=13))

def update_axes(fig, xtitle="", ytitle=""):
    fig.update_xaxes(title_text=xtitle, gridcolor="rgba(255,255,255,0.05)", title_font=dict(size=14, color="#94a3b8"), tickfont=dict(size=12))
    fig.update_yaxes(title_text=ytitle, gridcolor="rgba(255,255,255,0.05)", title_font=dict(size=14, color="#94a3b8"), tickfont=dict(size=12), zerolinecolor="rgba(255,255,255,0.1)")
    return fig

COLORS = ["#38bdf8","#818cf8","#34d399","#fb923c","#f472b6","#facc15","#a78bfa"]

# ══════════════════════════════════════════
# DATA FETCHING (cached)
# ══════════════════════════════════════════
@st.cache_data(ttl=86400, show_spinner=False)
def _try_load_fred_series(start_ts, end_ts):
    """Return (credit_spread_pct, yc_slope_pct) series from FRED if reachable,
    else (None, None). credit_spread = BAA-AAA, slope = T10Y2Y. Both monthly.
    Resilient: returns None on any failure so the proxy is used."""
    try:
        start = pd.Timestamp(start_ts) - pd.DateOffset(months=2)
        end   = pd.Timestamp(end_ts)
        key = _get_fred_key()
        if key and _FREDAPI_OK:
            fr = Fred(api_key=key)
            baa = pd.Series(fr.get_series("BAA", observation_start=start, observation_end=end))
            aaa = pd.Series(fr.get_series("AAA", observation_start=start, observation_end=end))
            slope = pd.Series(fr.get_series("T10Y2Y", observation_start=start, observation_end=end))
        elif _PDR_OK:
            baa = pdr.DataReader("BAA", "fred", start, end).iloc[:, 0]
            aaa = pdr.DataReader("AAA", "fred", start, end).iloc[:, 0]
            slope = pdr.DataReader("T10Y2Y", "fred", start, end).iloc[:, 0]
        else:
            return None, None

        spread = (baa - aaa).dropna()
        spread.index = pd.to_datetime(spread.index)
        slope.index  = pd.to_datetime(slope.index)
        spread = spread.resample("ME").last()
        slope  = slope.resample("ME").last()
        spread.index = spread.index.to_period("M").to_timestamp()
        slope.index  = slope.index.to_period("M").to_timestamp()
        if spread.dropna().empty or slope.dropna().empty:
            return None, None
        return spread, slope
    except Exception:
        return None, None

@st.cache_data(ttl=3600, show_spinner="📡 Fetching macro data…")
def load_macro() -> pd.DataFrame:
    """Pull S&P500, VIX, 10Y yield via BigQuery or yfinance fallback."""
    if _BQ_OK:
        try:
            bq_df = load_macro_from_bigquery()
            if bq_df is not None and not bq_df.empty and len(bq_df) > 100:
                if "regime" not in bq_df.columns or "regime_score" not in bq_df.columns:
                    cs = (bq_df["credit_spread"] - bq_df["credit_spread"].mean()) / bq_df["credit_spread"].std()
                    rv = (bq_df["realized_vol_12m"] - bq_df["realized_vol_12m"].mean()) / bq_df["realized_vol_12m"].std()
                    bq_df["regime_score"] = cs.fillna(0) + rv.fillna(0)
                    bq_df["regime"] = np.select([bq_df["regime_score"]<-0.5, bq_df["regime_score"]>0.5],
                                              ["Risk-On 🟢","Risk-Off 🔴"], default="Neutral 🟡")
                bq_df["_data_source"] = "GCP BigQuery"
                return bq_df.dropna(subset=["sp500"])
        except Exception:
            pass

    tmap = {"sp500":"^GSPC","vix":"^VIX","dgs10":"^TNX","gold":"GLD","oil":"USO"}
    frames = {}
    for col, tkr in tmap.items():
        for attempt in range(3):
            try:
                raw = yf.download(tkr, start="2005-01-01", auto_adjust=True,
                                  progress=False, multi_level_index=False, timeout=20)
                if raw.empty:
                    break
                close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                close = close.dropna().resample("ME").last()
                close.index = close.index.to_period("M").to_timestamp()
                frames[col] = close
                break
            except Exception:
                if attempt == 2:
                    pass
    if not frames or "sp500" not in frames:
        idx = pd.date_range("2010-01-01", periods=180, freq="ME")
        frames["sp500"]  = pd.Series([2000 + i*15.0 for i in range(180)], index=idx)
        frames["vix"]    = pd.Series(20.0, index=idx)
        frames["dgs10"]  = pd.Series(4.0, index=idx)
        frames["gold"]   = pd.Series(160.0, index=idx)
        frames["oil"]    = pd.Series(60.0, index=idx)
        frames["_is_demo"] = pd.Series(1.0, index=pd.date_range("2010-01-01", periods=len(frames["sp500"]), freq="ME"))
    df = pd.DataFrame({k:v for k,v in frames.items() if k != "_is_demo"}).sort_index()
    df["_is_demo"] = "_is_demo" in frames   # 데모 여부 컬럼으로 표시
    df.index.name = "date"
    df["sp500_ret_m"]    = np.log(df["sp500"]).diff()
    df["realized_vol_12m"] = df["sp500_ret_m"].rolling(12).std() * np.sqrt(12)
    df["realized_vol_3m"]  = df["sp500_ret_m"].rolling(3).std()  * np.sqrt(12)
    df["momentum_12_1"]    = df["sp500_ret_m"].rolling(11).sum().shift(1)
    df["cumret"]           = np.exp(df["sp500_ret_m"].cumsum()) * 100
    df["drawdown"]         = df["cumret"] / df["cumret"].cummax() - 1
    df["dgs10"]            = df["dgs10"].ffill()

    # Real FRED series when available; deterministic proxy otherwise.
    fred_credit, fred_slope = _try_load_fred_series(df.index.min(), df.index.max())
    if fred_credit is not None:
        df["credit_spread"] = fred_credit.reindex(df.index).ffill() / 100.0
        df["_credit_source"] = "FRED:BAA-AAA"
    else:
        df["credit_spread"] = (df["dgs10"] * 0.35 + 1.5 - df["dgs10"] * 0.1).abs() / 100
        df["_credit_source"] = "proxy"
    if fred_slope is not None:
        df["yc_slope"] = fred_slope.reindex(df.index).ffill() / 100.0
        df["_slope_source"] = "FRED:T10Y2Y"
    else:
        df["yc_slope"]  = (df["dgs10"] - df["dgs10"] * 0.65) / 100
        df["_slope_source"] = "proxy"
    cs = (df["credit_spread"] - df["credit_spread"].mean()) / df["credit_spread"].std()
    rv = (df["realized_vol_12m"] - df["realized_vol_12m"].mean()) / df["realized_vol_12m"].std()
    df["regime_score"] = cs.fillna(0) + rv.fillna(0)
    df["regime"] = np.select([df["regime_score"]<-0.5, df["regime_score"]>0.5],
                              ["Risk-On 🟢","Risk-Off 🔴"], default="Neutral 🟡")
    df["_data_source"] = "Live yfinance"
    if _BQ_OK:
        try:
            save_macro_to_bigquery(df)
        except Exception:
            pass
    return df.dropna(subset=["sp500"])


@st.cache_data(ttl=300, show_spinner="📈 Computing indicators…")
def fetch_stock(ticker: str, start: str, end: str):
    tkr = yf.Ticker(ticker)
    df  = tkr.history(start=start, end=end, auto_adjust=True)
    if df.empty:
        return None, {}
    try:
        fi = tkr.fast_info
        info = {"name": getattr(fi,"display_name",ticker),
                "sector": getattr(fi,"sector","—"),
                "mktcap": getattr(fi,"market_cap",None),
                "pe": getattr(fi,"p_e_ratio",None),
                "52wk_high": getattr(fi,"year_high",None),
                "52wk_low":  getattr(fi,"year_low",None)}
    except Exception:
        info = {}
    # ── Indicators ──
    df["SMA20"]  = df["Close"].rolling(20).mean()
    df["SMA50"]  = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["EMA12"]  = df["Close"].ewm(span=12).mean()
    df["EMA26"]  = df["Close"].ewm(span=26).mean()
    df["MACD"]   = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["Hist"]   = df["MACD"] - df["Signal"]
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"]    = 100 - 100/(1 + gain/(loss+1e-9))
    tp   = (df["High"]+df["Low"]+df["Close"])/3
    df["VWAP"]   = (tp * df["Volume"]).groupby(df.index.date).cumsum() / \
                   df["Volume"].groupby(df.index.date).cumsum()
    std20 = df["Close"].rolling(20).std()
    df["BB_upper"] = df["SMA20"] + 2*std20
    df["BB_lower"] = df["SMA20"] - 2*std20
    df["OBV"]    = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    df["ROC"]    = df["Close"].pct_change(10) * 100
    # Volume colors (vectorized — fix #5)
    df["vol_color"] = np.where(df["Close"] >= df["Open"], "#00c853", "#ff1744")
    return df, info

# ── Screener helper ──────────────────────────────────────────────────
def _screen_one(ticker: str, period_days: int):
    try:
        tkr = yf.Ticker(ticker)
        fi  = tkr.fast_info          # fix #8: use fast_info
        last_price = getattr(fi,"last_price", None)
        if last_price is None:
            return None
        # raw float columns — fix #2
        hist = tkr.history(period=f"{period_days}d", auto_adjust=True)
        if hist.empty or len(hist) < 5:
            return None
        ret_1m_raw = hist["Close"].pct_change(21).iloc[-1]
        ret_5d     = hist["Close"].pct_change(5).iloc[-1]
        # YTD — fix #1
        now = datetime.now()
        ytd_hist = tkr.history(start=f"{now.year}-01-01", auto_adjust=True)
        if not ytd_hist.empty:
            ytd_raw = (last_price - float(ytd_hist["Close"].iloc[0])) / float(ytd_hist["Close"].iloc[0])
        else:
            ytd_raw = np.nan
        mktcap = getattr(fi,"market_cap", None)
        pe     = getattr(fi,"p_e_ratio", None)
        # SMA guard — fix #3
        sma200 = hist["Close"].rolling(200).mean().iloc[-1]
        if np.isnan(sma200):
            sma200_signal = "⚠️ Insufficient"
        else:
            sma200_signal = "↑ Above" if last_price > sma200 else "↓ Below"
        delta = hist["Close"].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi   = (100 - 100/(1+gain/(loss+1e-9))).iloc[-1]
        return {
            "Ticker": ticker,
            "Price": last_price,
            "1M Ret": ret_1m_raw,   # raw float
            "5D Ret": ret_5d,
            "YTD":    ytd_raw,      # raw float
            "RSI":    round(rsi, 1),
            "SMA200": sma200_signal,
            "Mkt Cap": mktcap,
            "P/E": pe or tkr.info.get("trailingPE") if pe is None else pe,
        }
    except Exception:
        return None

@st.cache_data(ttl=300, show_spinner=False)
def run_screener(tickers: tuple, period_days: int):
    results = []
    # fix #7: parallel fetch
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_screen_one, t, period_days): t for t in tickers}
        for f in as_completed(futs):
            r = f.result()
            if r:
                results.append(r)
    return pd.DataFrame(results)

# ── Buy-Zone Scanner (weekly bars) ─────────────────────────────────────
def _zone_for(price, ma20, ma50, rsi):
    """Deterministic weekly zone classification. Returns (zone_label, buy_lo, buy_hi, note)."""
    if not (np.isfinite(price) and np.isfinite(ma20) and np.isfinite(ma50) and np.isfinite(rsi)):
        return "⚪ Insufficient data", float("nan"), float("nan"), "Need ≥ 50 weekly bars"
    if rsi >= 70 or price > ma20 * 1.10:
        lo, hi = ma20 * 0.95, ma20
        return "🔴 Avoid (Extended)", lo, hi, f"Wait for pullback to ${lo:,.2f}–${hi:,.2f}"
    if price <= ma50 * 1.02 and rsi <= 40:
        lo, hi = ma50 * 0.97, ma50 * 1.02
        return "🟢 Strong Buy (Accumulation)", lo, hi, f"Accumulate ${lo:,.2f}–${hi:,.2f}"
    if ma50 < price <= ma20 and 40 < rsi < 60:
        lo, hi = ma50, ma20
        return "🟢 Pullback Buy", lo, hi, f"Buy on pullback ${lo:,.2f}–${hi:,.2f}"
    if ma20 < price <= ma20 * 1.05 and 50 <= rsi < 70:
        lo, hi = ma20, ma20 * 1.05
        return "🟡 Trend Continuation", lo, hi, f"Add on retest ${lo:,.2f}–${hi:,.2f}"
    return "⚪ Watch / No Edge", float("nan"), float("nan"), f"No clean entry; key level ${ma20:,.2f}"

def _scan_one_ticker(ticker: str, years: int):
    try:
        raw = yf.download(ticker, period=f"{years}y", auto_adjust=True,
                          progress=False, multi_level_index=False, timeout=20)
        if raw is None or raw.empty:
            return {"Ticker": ticker, "_err": "no data"}
        close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        wk = close.resample("W-FRI").last().dropna()
        if len(wk) < 55:
            return {"Ticker": ticker, "_err": f"only {len(wk)} weekly bars"}
        sma20 = wk.rolling(20).mean()
        sma50 = wk.rolling(50).mean()
        delta = wk.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi   = 100 - 100 / (1 + gain / (loss + 1e-9))
        ema12 = wk.ewm(span=12, adjust=False).mean()
        ema26 = wk.ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        sig   = macd.ewm(span=9, adjust=False).mean()
        hist  = macd - sig

        price  = float(wk.iloc[-1])
        ma20_v = float(sma20.iloc[-1])
        ma50_v = float(sma50.iloc[-1])
        rsi_v  = float(rsi.iloc[-1])
        macd_v = float(macd.iloc[-1])
        sig_v  = float(sig.iloc[-1])
        h_v    = float(hist.iloc[-1])
        h_prev = float(hist.iloc[-2]) if len(hist) > 1 else h_v

        if   macd_v > sig_v and h_v > h_prev: macd_state = "↑ confirming"
        elif macd_v < sig_v and h_v < h_prev: macd_state = "↓ diverging"
        else:                                  macd_state = "→ flat"

        zone, lo, hi, note = _zone_for(price, ma20_v, ma50_v, rsi_v)
        return {
            "Ticker": ticker,
            "Price":  price,
            "RSI(W)": round(rsi_v, 1),
            "MA20W":  round(ma20_v, 2),
            "MA50W":  round(ma50_v, 2),
            "MA20 dist %": (price/ma20_v - 1) * 100 if ma20_v else float("nan"),
            "MACD(W)": round(macd_v, 3),
            "MACD State": macd_state,
            "Zone": zone,
            "Buy Range": note,
            "_lo": lo, "_hi": hi,
        }
    except Exception as e:
        return {"Ticker": ticker, "_err": str(e)[:80]}

@st.cache_data(ttl=900, show_spinner="🎯 Computing weekly buy zones…")
def scan_buy_zones(tickers: tuple, years: int = 3):
    rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_scan_one_ticker, t, years): t for t in tickers}
        for f in as_completed(futs):
            rows.append(f.result())
    return pd.DataFrame(rows)

# ── Risk metrics ──────────────────────────────────────────────────────
def compute_hf_metrics(rets: pd.Series, bench_rets: pd.Series | None = None):
    ann_ret  = rets.mean() * 12
    ann_vol  = rets.std() * np.sqrt(12)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else np.nan
    neg      = rets[rets < 0]
    sortino  = ann_ret / (neg.std()*np.sqrt(12)) if len(neg) > 0 else np.nan
    cum      = np.exp(rets.cumsum())
    mdd      = (cum / cum.cummax() - 1).min()
    calmar   = ann_ret / abs(mdd) if mdd < 0 else np.nan
    win_rate = (rets > 0).mean()
    avg_win  = rets[rets > 0].mean() if (rets > 0).any() else 0
    avg_loss = rets[rets < 0].mean() if (rets < 0).any() else 0
    alpha    = np.nan
    if bench_rets is not None:
        aligned = rets.align(bench_rets, join="inner")
        if len(aligned[0]) > 12:
            cov   = np.cov(aligned[0], aligned[1])
            beta  = cov[0,1] / cov[1,1] if cov[1,1] > 0 else np.nan
            alpha = (ann_ret - beta * bench_rets.mean()*12) if not np.isnan(beta) else np.nan
    return dict(ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe, sortino=sortino,
                mdd=mdd, calmar=calmar, win_rate=win_rate, avg_win=avg_win,
                avg_loss=avg_loss, alpha=alpha)

# ── Chart helpers ─────────────────────────────────────────────────────
def area(df, col, title, color="#38bdf8", ytitle="Value"):
    s = df[col].dropna()
    fig = go.Figure(go.Scatter(x=s.index, y=s.values, fill="tozeroy", mode="lines",
                               line=dict(color=color, width=1.5),
                               fillcolor=f"rgba(56,189,248,.08)"))
    fig.update_layout(title=title, hovermode="x unified", **PT)
    return update_axes(fig, "Date", ytitle)

def multiline(df, cols, title, yformat=".2f", ytitle="Value"):
    fig = go.Figure()
    for i,c in enumerate(cols):
        if c not in df: continue
        s = df[c].dropna()
        fig.add_trace(go.Scatter(x=s.index, y=s, name=c, mode="lines",
                                 line=dict(color=COLORS[i%len(COLORS)], width=2)))
    fig.update_layout(title=title, hovermode="x unified", **PT)
    fig.update_yaxes(tickformat=yformat)
    return update_axes(fig, "Date", ytitle)

# ══════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ Macro Pulse")
    st.markdown("---")
    st.markdown("**📅 Date Range**")
    col_a, col_b = st.columns(2)
    with col_a:
        d_start = st.date_input("Start", value=date(2015,1,1), max_value=date.today())
    with col_b:
        d_end   = st.date_input("End",   value=date.today(),   max_value=date.today())
    st.markdown("---")
    st.markdown("**🎲 Monte Carlo**")
    mc_mu  = st.slider("E[R] Annual (%)", -5, 30, 8)
    mc_vol = st.slider("σ Annual (%)",     5, 40, 16)
    mc_n   = st.selectbox("Paths", [1000,5000,10000], index=1)
    st.markdown("---")
    if _BQ_OK:
        st.markdown("**☁️ GCP Data Lakehouse**")
        bq_stat = get_bq_status()
        if bq_stat.get("connected"):
            st.success(f"**BigQuery**: Connected\n\nProject: `{bq_stat['project']}`")
            if bq_stat.get("tables"):
                st.caption(f"Active Marts: {', '.join(bq_stat['tables'])}")
        else:
            st.caption("BigQuery: Standby (local/fallback)")
        st.markdown("---")
    if st.button("🔄 Reload Data"):
        st.cache_data.clear(); st.rerun()
    st.markdown("<small style='color:#475569'>Data: yfinance · FRED · BigQuery<br>© 2026 HF Research</small>",
                unsafe_allow_html=True)


# ══════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════
try:
    df_raw = load_macro()
    if "_is_demo" in df_raw.columns and bool(df_raw["_is_demo"].iloc[0]):
        st.warning("⚠️ Live market data unavailable — showing demo data. Click 🔄 Reload to retry.")
    df = df_raw[(df_raw.index >= pd.Timestamp(d_start)) &
                (df_raw.index <= pd.Timestamp(d_end))].copy()
    if df.empty or len(df) < 2:
        raise ValueError(f"Filtered DataFrame is empty (start={d_start}, end={d_end})")
except Exception as _startup_err:
    st.error(f"🚨 Startup error: {_startup_err}")
    st.code(str(type(_startup_err).__name__), language="text")
    import traceback
    st.code(traceback.format_exc(), language="text")
    st.stop()

# ── SPY benchmark ─────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_spy(start, end):
    try:
        raw = yf.download("SPY", start=start, end=str(end), auto_adjust=True,
                          progress=False, multi_level_index=False, timeout=20)
        if raw.empty:
            return pd.Series(dtype=float)
        close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.resample("ME").last()
        close.index = close.index.to_period("M").to_timestamp()
        return np.log(close).diff().dropna()
    except Exception:
        return pd.Series(dtype=float)

try:
    spy_rets = load_spy(str(d_start), d_end)
    m = compute_hf_metrics(df["sp500_ret_m"].dropna(), spy_rets)
except Exception as e:
    spy_rets = pd.Series(dtype=float)
    m = dict(ann_ret=0, ann_vol=0, sharpe=np.nan, sortino=np.nan,
             mdd=0, calmar=np.nan, win_rate=0, avg_win=0, avg_loss=0, alpha=np.nan)

# ── KPI row deltas (MoM) — fix #17 ──────────────────────────────────
try:
    last = df.iloc[-1]; prev = df.iloc[-2] if len(df)>1 else df.iloc[-1]
    sp_d   = (last["sp500"]/prev["sp500"]-1)*100
    dg_d   = last["dgs10"] - prev["dgs10"]
    cur_yr = df[df.index.year == datetime.now().year]
    ytd = (last["sp500"] / float(cur_yr["sp500"].iloc[0]) - 1)*100 if not cur_yr.empty else np.nan
except Exception:
    last = df.iloc[-1] if len(df)>0 else pd.Series({"sp500":0,"dgs10":0,"regime":"—","regime_score":0})
    prev = last
    sp_d, dg_d, ytd = 0.0, 0.0, np.nan

# ── Title ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
  <span style="font-size:2.2rem;">⚡</span>
  <div>
    <h1 style="margin:0;background:linear-gradient(135deg,#38bdf8,#818cf8);-webkit-background-clip:text;
      -webkit-text-fill-color:transparent;font-size:2rem;font-weight:800;line-height:1.1;">
      Macro Pulse
    </h1>
    <p style="margin:0;color:#64748b;font-size:.83rem;">
      Institution-grade macro analysis · Regime classification · Portfolio risk simulation
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI metrics ────────────────────────────────────────────────────────
c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(8)
c1.metric("S&P 500",      f"{last['sp500']:,.0f}",  f"{sp_d:+.2f}%")
c2.metric("10Y Yield",    f"{last['dgs10']:.2f}%",  f"{dg_d*100:+.0f}bps MoM")
c3.metric("YTD",          f"{ytd:.1f}%"   if np.isfinite(ytd) else "—")
c4.metric("Ann Return",   f"{m['ann_ret']*100:.1f}%")
c5.metric("Sharpe",       f"{m['sharpe']:.2f}"   if np.isfinite(m['sharpe']) else "—")
c6.metric("Sortino",      f"{m['sortino']:.2f}"  if np.isfinite(m['sortino']) else "—")  # fix #9
c7.metric("Calmar",       f"{m['calmar']:.2f}"   if np.isfinite(m['calmar']) else "—")   # fix #9
c8.metric("Max DD",       f"{m['mdd']*100:.1f}%")

# ── Regime banner ──────────────────────────────────────────────────────
cr = df["regime"].iloc[-1]
bg = {"Risk-On 🟢":"#0d3320","Neutral 🟡":"#332e00","Risk-Off 🔴":"#3b0e0e"}.get(cr,"#131d35")
bd = {"Risk-On 🟢":"#34d399","Neutral 🟡":"#facc15","Risk-Off 🔴":"#f87171"}.get(cr,"#38bdf8")
st.markdown(f"""<div style="background:{bg}; border-left:4px solid {bd}; padding:12px 20px; 
border-radius:8px; margin:12px 0; font-size:.95rem; color:#ffffff; font-weight:500; 
box-shadow: 0 4px 15px rgba(0,0,0,0.3); display:flex; align-items:center;">
<span style="opacity:0.9;">Current Regime:</span>&nbsp;<b style="color:{bd}; font-size:1.05rem;">{cr}</b> 
&nbsp;&nbsp;|&nbsp;&nbsp; <span style="opacity:0.9;">Score:</span>&nbsp;<b>{df['regime_score'].iloc[-1]:.2f}</b>
&nbsp;&nbsp;|&nbsp;&nbsp; <span style="opacity:0.9;">as of</span>&nbsp;<b>{df.index[-1].strftime('%b %Y')}</b></div>""", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════
# TABS
# ══════════════════════════════════════════
tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10,tab11 = st.tabs([
    "📈 Performance","🌍 Macro & Rates","🔍 Regime","🤖 Expected Returns",
    "📊 Screener","📉 Technical","🎲 Risk Sim","✨ Gemini AI Analyst",
    "🔥 NVDA Danger Zone","📊 Strategy Backtest","🎯 Quant Signals"])

# ─── Tab 1: Performance ──────────────────────────────────────────────
with tab1:
    render_methodology("performance", st)
    # SPY overlay — fix #10
    spy_cum = load_spy(str(d_start), d_end)
    spy_idx = np.exp(spy_cum.cumsum()) * 100
    sp5_idx = df["cumret"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sp5_idx.index, y=sp5_idx, name="S&P 500", mode="lines",
                             line=dict(color="#38bdf8",width=2), fill="tozeroy",
                             fillcolor="rgba(56,189,248,.06)"))
    if not spy_idx.empty:
        fig.add_trace(go.Scatter(x=spy_idx.index, y=spy_idx, name="SPY (benchmark)",
                                 mode="lines", line=dict(color="#facc15",width=1.5,dash="dot")))
    fig.update_layout(title="Cumulative Return (Base=100)", hovermode="x unified", **PT)
    update_axes(fig, "Date", "Indexed Return")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ How to read Performance charts"):
        st.write("""
        * **Cumulative Return**: Shows the growth of $100 invested at the start. Compare the blue line (S&P 500) vs the dashed yellow line (SPY) to see relative outperformance.
        * **Rolling Drawdown**: Indicates the 'pain' of holding the asset. It shows the peak-to-trough decline. A drawdown of -20% means you lost 20% from the previous high.
        * **Return Distribution**: A histogram showing how frequent different monthly returns occur. A 'bell curve' shifted to the right is ideal.
        """)

    c1,c2 = st.columns(2)
    with c1:
        dd = df["drawdown"] * 100
        fig2 = go.Figure(go.Scatter(x=dd.index, y=dd, fill="tozeroy", mode="lines",
                                    line=dict(color="#f87171",width=1.5),
                                    fillcolor="rgba(248,113,113,.12)", name="Drawdown %"))
        fig2.update_layout(title="Rolling Drawdown %", hovermode="x unified", **PT)
        update_axes(fig2, "Date", "Drawdown Percentage")
        fig2.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        mr = df["sp500_ret_m"].dropna()*100
        fig3 = go.Figure(go.Histogram(x=mr, nbinsx=60, marker_color="#38bdf8", opacity=.75))
        fig3.update_layout(title="Monthly Return Distribution (%)", **PT)
        update_axes(fig3, "Monthly Return %", "Frequency")
        fig3.update_xaxes(ticksuffix="%")
        st.plotly_chart(fig3, use_container_width=True)

    # Extended tear sheet — fix #9
    st.markdown("#### 📋 Strategy Tear Sheet")
    wr = m['win_rate']; aw = m['avg_win']; al = m['avg_loss']
    profit_factor = abs(aw/al) if al !=0 else np.nan
    tear = pd.DataFrame({
        "Metric": ["Ann Return","Ann Vol","Sharpe","Sortino","Calmar","Max Drawdown",
                   "Win Rate","Avg Win (monthly)","Avg Loss (monthly)","Profit Factor","Alpha vs SPY"],
        "Value": [f"{m['ann_ret']*100:.2f}%", f"{m['ann_vol']*100:.2f}%",
                  f"{m['sharpe']:.3f}", f"{m['sortino']:.3f}" if np.isfinite(m['sortino']) else "—",
                  f"{m['calmar']:.3f}" if np.isfinite(m['calmar']) else "—",
                  f"{m['mdd']*100:.2f}%", f"{wr*100:.1f}%",
                  f"{aw*100:.2f}%", f"{al*100:.2f}%",
                  f"{profit_factor:.2f}" if np.isfinite(profit_factor) else "—",
                  f"{m['alpha']*100:.2f}%" if np.isfinite(m.get('alpha',np.nan)) else "—"]
    })
    st.dataframe(tear, use_container_width=True, hide_index=True)

# ─── Tab 2: Macro & Rates ────────────────────────────────────────────
with tab2:
    render_methodology("macro_rates", st)
    st.info("📊 **Macro Framework**: These indicators track long-term debt costs, corporate default risk, and market fear (volatility).")
    c1,c2 = st.columns(2)
    with c1:
        fig = multiline(df, ["dgs10"], "10Y Treasury Yield (%)", ytitle="Yield %")
        fig.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = area(df,"credit_spread","Credit Spread (Proxy)",color="#fb923c", ytitle="Spread %")
        fig.update_yaxes(tickformat=".2%")
        st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("ℹ️ How to interpret Macro & Rates"):
        st.write("""
        * **10Y Yield**: The 'risk-free' rate. Rising yields discounted future cash flows, often hurting growth stocks.
        * **Credit Spread**: The extra yield required by lenders for corporate risk. Spikes signal 'Risk-Off' and economic stress.
        * **Yield Curve Slope**: Measured as the difference between long and short-term rates. Inversions (negative) often precede recessions.
        * **Realized Volatility**: Clusters of high volatility suggest a regime shift towards a defensive or unstable market.
        """)
        
    c1,c2 = st.columns(2)
    with c1:
        fig = area(df,"yc_slope","Yield Curve Slope (10Y - 2Y)",color="#818cf8", ytitle="Slope %")
        fig.add_hline(y=0,line_dash="dash",line_color="#f87171",annotation_text="Inversion", annotation_position="top left")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = multiline(df,["realized_vol_12m","realized_vol_3m"],"Realized Volatility (12M vs 3M)", ytitle="Vol %")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    latest = {
        "10Y Yield":f"{last['dgs10']:.2f}%",
        "Credit Spread":f"{last['credit_spread']*100:.2f}%",
        "YC Slope":f"{last['yc_slope']*100:.2f}%",
        "12M Vol":f"{last['realized_vol_12m']*100:.1f}%",
        "Momentum":f"{last['momentum_12_1']*100:.2f}%",
        "Regime Score":f"{last['regime_score']:.2f}",
    }
    st.markdown("#### 📋 Latest Macro Snapshot")
    st.dataframe(pd.DataFrame(latest.items(), columns=["Indicator","Value"]),
                 use_container_width=True, hide_index=True)

# ─── Tab 3: Regime ──────────────────────────────────────────────────
with tab3:
    render_methodology("regime", st)
    # Color-coded price by regime — fix #12 (labeled by macro characteristics)
    # Color-coded price by regime
    color_map = {"Risk-On 🟢":"#34d399","Neutral 🟡":"#facc15","Risk-Off 🔴":"#f87171"}
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["sp500"], name="S&P 500",
                             mode="lines", line=dict(color="#475569",width=1)))
    for reg,col in color_map.items():
        mask = df["regime"]==reg
        fig.add_trace(go.Scatter(x=df.index[mask], y=df["sp500"][mask],
                                  name=reg, mode="markers",
                                  marker=dict(color=col,size=4,opacity=.7)))
    fig.update_layout(title="S&P 500 Price — Macro Regime Classification", hovermode="x unified",**PT)
    update_axes(fig, "Date", "S&P 500 Level")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ Understanding Macro Regimes"):
        st.write("""
        * **Regime Classification**: Our GMM-inspired model clusters market states based on the *Macro Stress Score*.
        * **Macro Stress Score**: A composite Z-score of Credit Spreads and Realized Volatility. 
        * **Thresholds**: 
            * Score > 0.5 = **Risk-Off** (Markets are volatile, credit is tight).
            * Score < -0.5 = **Risk-On** (Stability, liquidity, growth bias).
        """)

    c1,c2 = st.columns(2)
    with c1:
        rc = df["regime"].value_counts()
        fig2 = go.Figure(go.Pie(labels=rc.index, values=rc.values,
                                marker_colors=["#34d399","#facc15","#f87171"], hole=.55))
        fig2.update_layout(title="Sample Period Regime Distribution", **PT)
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        fig3 = area(df,"regime_score","Macro Stress Score (Credit + Vol Z-Score)",color="#818cf8", ytitle="Z-Score")
        fig3.add_hline(y=0.5,line_dash="dash",line_color="#f87171",annotation_text="Fear Threshold", annotation_position="top left")
        fig3.add_hline(y=-0.5,line_dash="dash",line_color="#34d399",annotation_text="Confidence", annotation_position="bottom left")
        st.plotly_chart(fig3, use_container_width=True)

    # fix #12: labels with macro context
    st.markdown("#### 📊 Regime Characteristics")
    st.info("""
**Risk-On 🟢** — Credit spreads compressed, vol below average → equity-friendly environment  
**Neutral 🟡** — Balanced conditions, no strong macro signal  
**Risk-Off 🔴** — Credit spreads elevated, vol above average → defensive positioning warranted
""")
    stats = df.groupby("regime")["sp500_ret_m"].agg(
        Count="count",
        Ann_Ret=lambda x: f"{x.mean()*12*100:.1f}%",
        Ann_Vol=lambda x: f"{x.std()*np.sqrt(12)*100:.1f}%",
        Sharpe=lambda x:  f"{(x.mean()*12)/(x.std()*np.sqrt(12)+1e-9):.2f}",
        WinRate=lambda x: f"{(x>0).mean()*100:.0f}%",
    ).reset_index()
    st.dataframe(stats, use_container_width=True, hide_index=True)

# ─── Tab 4: Expected Returns ─────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="🤖 Training expected return model…")
def compute_expected_returns(df_hash: str, macro_data: pd.DataFrame):
    """Expanding-window Ridge regression on yfinance macro data.
    No FRED API or parquet files required.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    d = macro_data.copy()
    # Forward 12m log return (target)
    d["y_fwd_12m"] = d["sp500_ret_m"].rolling(12).sum().shift(-12)
    X_cols = [c for c in ["dgs10","credit_spread","yc_slope",
                           "realized_vol_12m","momentum_12_1","regime_score",
                           "realized_vol_3m"] if c in d.columns]
    d = d.dropna(subset=X_cols + ["y_fwd_12m","sp500_ret_m"])
    if len(d) < 60:
        return None

    model = Pipeline([("sc", StandardScaler()), ("ridge", Ridge(alpha=5.0))])
    preds = pd.Series(index=d.index, dtype=float)
    errors = []
    min_train = 48  # at least 4 years

    for i in range(min_train, len(d)):
        train = d.iloc[:i]
        test  = d.iloc[i:i+1]
        model.fit(train[X_cols], train["y_fwd_12m"])
        pred = model.predict(test[X_cols])[0]
        preds.iloc[i] = pred
        errors.append(pred - train["y_fwd_12m"].iloc[-1])  # in-sample residual proxy

    out = d.copy()
    out["exp_ann_return"] = np.expm1(preds)   # log-return → simple annualized
    out["pred_std"] = pd.Series(errors, dtype=float).rolling(24, min_periods=12).std().reindex(out.index)
    return out[["exp_ann_return","sp500_ret_m","pred_std"]].dropna(subset=["exp_ann_return"])

with tab4:
    render_methodology("expected_returns", st)
    # Try FRED parquet first; fall back to live yfinance model
    exp = None
    source_label = ""

    if os.path.exists("exp_return_estimates.parquet"):
        try:
            raw = pd.read_parquet("exp_return_estimates.parquet")
            exp = raw[["exp_ann_return","sp500_ret_m"]].dropna()
            exp["pred_std"] = exp["exp_ann_return"].rolling(12).std()
            source_label = "FRED macro model"
        except Exception:
            pass

    if exp is None or exp.empty:
        # Compute from yfinance data directly
        # Use a hash of the index to bust cache correctly
        _hash = str(df_raw.index[-1]) + str(len(df_raw))
        exp = compute_expected_returns(_hash, df_raw)
        source_label = "yfinance macro model (live)"

    if exp is not None and not exp.empty:
        roll_std = exp["pred_std"].fillna(exp["exp_ann_return"].rolling(12).std())
        latest_pred = exp["exp_ann_return"].dropna().iloc[-1]

        # KPI row
        c1, c2, c3 = st.columns(3)
        c1.metric("📌 12M Prediction", f"{latest_pred*100:.1f}%")
        c2.metric("Model", source_label)
        c3.metric("Data Points", f"{len(exp):,}")

        # Main chart with confidence band + realized
        fig = go.Figure()
        upper = (exp["exp_ann_return"] + roll_std).ffill()
        lower = (exp["exp_ann_return"] - roll_std).ffill()
        fig.add_trace(go.Scatter(
            x=list(exp.index) + list(exp.index[::-1]),
            y=list(upper) + list(lower[::-1]),
            fill="toself", fillcolor="rgba(56,189,248,.10)", line=dict(width=0),
            name="±1σ Confidence Band", showlegend=True))
        fig.add_trace(go.Scatter(
            x=exp.index, y=exp["exp_ann_return"],
            name="Model Predicted (12M)", line=dict(color="#38bdf8", width=2)))
        realized = exp["sp500_ret_m"].rolling(12).sum()
        fig.add_trace(go.Scatter(
            x=realized.index, y=realized,
            name="Realized 12M Return", line=dict(color="#34d399", width=1.5, dash="dot")))
        fig.add_hline(y=0, line_dash="dot", line_color="#475569", annotation_text="Zero")
        fig.update_layout(
            title=f"Model-Implied Expected 12M Return — {source_label}",
            hovermode="x unified", **PT)
        update_axes(fig, "Date", "Expected 12M Return %")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("ℹ️ How to read predictive charts"):
            st.write("""
            * **Model Predicted (12M)**: The central estimate from our Ridge regression model, forecasting the S&P 500's return over the next 12 months based on current macro features.
            * **Realized 12M Return**: The actual trailing 12-month return of the S&P 500. This tracks how well the model performed in hindsight.
            * **Confidence Band**: Represents ±1 standard deviation of historical model errors. When the green line stays within the blue shade, the model is performing within normal statistical bounds.
            """)

        # Feature importance (last fitted coefficients)
        st.markdown("#### 🔬 Model Info")
        st.caption("Expanding-window Ridge Regression | Min 48 months training | Features: yield, spread, vol, momentum, regime score")
    else:
        st.warning("Not enough data to compute expected return model. Try extending the date range.")

    st.markdown("#### 📐 Momentum Signal (12-1)")
    fig4 = area(df, "momentum_12_1", "12-1 Momentum", color="#facc15")
    fig4.add_hline(y=0, line_dash="dot", line_color="#475569")
    st.plotly_chart(fig4, use_container_width=True)

# ─── Tab 5: Screener ─────────────────────────────────────────────────
with tab5:
    render_methodology("screener", st)
    col_u, col_p, col_s = st.columns([2,1,1])
    with col_u:
        # fix #19: custom ticker input
        custom_raw = st.text_input("Custom tickers (comma-separated)",
                                   placeholder="e.g. NVDA, META, TSLA")
        default_universe = QS_DEFAULT_UNIVERSE  # shared with the Quant Signals tab
        if custom_raw.strip():
            universe = tuple(t.strip().upper() for t in custom_raw.split(",") if t.strip())
        else:
            universe = default_universe
    with col_p:
        screen_period = st.selectbox("Screen Period", ["1mo","3mo","6mo","1yr"], index=1)
        period_map = {"1mo":30,"3mo":90,"6mo":180,"1yr":365}
        period_days = period_map[screen_period]
    with col_s:
        sort_col = st.selectbox("Sort By", ["1M Ret","YTD","RSI","5D Ret"], index=0)

    run_btn = st.button("🔎 Run Screener", type="primary")
    if run_btn:
        with st.spinner("🔎 Scanning tickers in parallel…"):
            sc_df = run_screener(universe, period_days)
    else:
        sc_df = pd.DataFrame()

    if not sc_df.empty:
        sc_df = sc_df.sort_values(sort_col, ascending=False)
        display = sc_df.copy()
        # Format for display — keep raw floats in sc_df for chart
        display["1M Ret"] = display["1M Ret"].apply(lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "—")
        display["5D Ret"] = display["5D Ret"].apply(lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "—")
        display["YTD"]    = display["YTD"].apply(lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "—")
        display["Mkt Cap"]= display["Mkt Cap"].apply(lambda x: f"${x/1e9:.1f}B" if pd.notna(x) else "—")
        display["P/E"]    = display["P/E"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
        display["Price"]  = display["Price"].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "—")

        # fix #4: use .map() not .applymap()
        def color_ret(val):
            try:
                v = float(str(val).replace("%","").replace("+",""))
                return "color:#34d399" if v>0 else "color:#f87171" if v<0 else ""
            except: return ""
        styled = display.style.map(color_ret, subset=["1M Ret","5D Ret","YTD"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # fix #20: export button
        csv = sc_df.to_csv(index=False).encode()
        st.download_button("⬇️ Export CSV", csv, "screener_results.csv", "text/csv")

        # Bar chart using raw floats (fix #2)
        top10 = sc_df.nlargest(min(10,len(sc_df)), "1M Ret")
        bar_colors = np.where(top10["1M Ret"] >= 0, "#34d399", "#f87171").tolist()
        fig = go.Figure(go.Bar(x=top10["Ticker"], y=top10["1M Ret"]*100,
                               marker_color=bar_colors, text=[f"{v*100:+.1f}%" for v in top10["1M Ret"]],
                               textposition="outside"))
        fig.update_layout(title="Top Movers — 1M Return (%)", **PT)
        fig.update_yaxes(ticksuffix="%", gridcolor="#1e2d4a"); fig.update_xaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

# ─── Tab 6: Technical Analysis ───────────────────────────────────────
with tab6:
    render_methodology("technical", st)
    # ══════════════════════════════════════════════════════════════════
    # 🎯 Multi-Ticker Buy Zone Scanner — weekly 20/50 MA · RSI · MACD
    # ══════════════════════════════════════════════════════════════════
    st.markdown("### 🎯 Buy Zone Scanner — Weekly 20MA · 50MA · RSI · MACD")
    st.caption(
        "Deterministic translation of weekly 20w/50w SMAs and RSI into a "
        "zone label and a suggested *technical* buy range. **Reference levels, "
        "not investment advice — trade size, stops, fundamentals, and macro "
        "context are out of scope here.**"
    )

    BZS_DEFAULT = ("NVDA","MSFT","TSM","ASML","AMZN","GOOGL","AVGO","LLY","V","COST")
    cA, cB = st.columns([3, 1])
    with cA:
        bzs_input = st.text_input("Tickers (comma-separated)",
                                   ", ".join(BZS_DEFAULT), key="bzs_tickers")
    with cB:
        bzs_years = st.selectbox("Lookback", [2, 3, 5, 7], index=1, key="bzs_years")
    bzs_tickers = tuple(t.strip().upper() for t in bzs_input.split(",") if t.strip())[:30]

    if bzs_tickers:
        bzs_df = scan_buy_zones(bzs_tickers, years=int(bzs_years))
        ok = bzs_df[~bzs_df.get("Price", pd.Series(dtype=float)).isna()] if "Price" in bzs_df else pd.DataFrame()
        bad = bzs_df[bzs_df.get("Price", pd.Series(dtype=float)).isna()] if "Price" in bzs_df else bzs_df

        if not ok.empty:
            # KPI row — count of names per zone
            zone_counts = ok["Zone"].value_counts().to_dict()
            cs = st.columns(5)
            cs[0].metric("🟢 Strong Buy",   zone_counts.get("🟢 Strong Buy (Accumulation)", 0))
            cs[1].metric("🟢 Pullback",     zone_counts.get("🟢 Pullback Buy", 0))
            cs[2].metric("🟡 Trend Cont.",  zone_counts.get("🟡 Trend Continuation", 0))
            cs[3].metric("🔴 Avoid",         zone_counts.get("🔴 Avoid (Extended)", 0))
            cs[4].metric("⚪ Watch",         zone_counts.get("⚪ Watch / No Edge", 0))

            display_cols = ["Ticker","Price","RSI(W)","MA20W","MA50W","MA20 dist %",
                            "MACD(W)","MACD State","Zone","Buy Range"]
            show = ok.copy()
            # zone-priority sort: Strong Buy → Pullback → Trend → Watch → Avoid
            zone_order = {"🟢 Strong Buy (Accumulation)":0, "🟢 Pullback Buy":1,
                          "🟡 Trend Continuation":2, "⚪ Watch / No Edge":3,
                          "🔴 Avoid (Extended)":4, "⚪ Insufficient data":5}
            show["_ord"] = show["Zone"].map(zone_order).fillna(9)
            show = show.sort_values(["_ord","RSI(W)"]).drop(columns=["_ord"])
            show = show[display_cols]
            show["Price"]       = show["Price"].map(lambda v: f"${v:,.2f}")
            show["MA20W"]       = show["MA20W"].map(lambda v: f"${v:,.2f}")
            show["MA50W"]       = show["MA50W"].map(lambda v: f"${v:,.2f}")
            show["MA20 dist %"] = show["MA20 dist %"].map(lambda v: f"{v:+.1f}%")
            st.dataframe(show, use_container_width=True, hide_index=True)

            st.download_button(
                "📥 Export buy-zone scan to CSV",
                ok.drop(columns=[c for c in ("_lo","_hi","_err") if c in ok.columns])
                  .to_csv(index=False).encode("utf-8"),
                file_name="buy_zone_scanner.csv",
                mime="text/csv",
                key="bzs_csv",
            )

        if not bad.empty:
            with st.expander(f"⚠️ Skipped {len(bad)} ticker(s)"):
                st.dataframe(bad[["Ticker"] + [c for c in bad.columns if c == "_err"]],
                             use_container_width=True, hide_index=True)

        with st.expander("ℹ️ How the Buy Zone Scanner works (and what it isn't)"):
            st.markdown("""
**Inputs.** Daily closes from yfinance, resampled to weekly Friday closes.
Three indicators are computed on the weekly series:

* **20-week SMA** — short-term trend backbone (~ 100 trading days). Holding above it is the canonical *uptrend confirmed* filter.
* **50-week SMA** — long-term trend (~ 250 trading days). Bounces here are textbook accumulation entries.
* **Weekly RSI(14)** — momentum oscillator. < 40 oversold, > 70 overbought.
* **Weekly MACD(12-26-9)** — confirmation only. The *MACD State* column reads the direction of the histogram vs its signal line: **↑ confirming** = momentum accelerating, **↓ diverging** = momentum fading, **→ flat** = neutral.

**Deterministic zone rules.**

| Zone | Trigger | Suggested technical buy range |
|------|---------|-------------------------------|
| 🟢 Strong Buy (Accumulation) | price ≤ 50w × 1.02 **AND** RSI ≤ 40 | 50w × 0.97 → 50w × 1.02 |
| 🟢 Pullback Buy | 50w < price ≤ 20w **AND** 40 < RSI < 60 | 50w → 20w |
| 🟡 Trend Continuation | 20w < price ≤ 20w × 1.05 **AND** 50 ≤ RSI < 70 | 20w → 20w × 1.05 |
| 🔴 Avoid (Extended) | RSI ≥ 70 **OR** price > 20w × 1.10 | wait for pullback to 20w × 0.95 → 20w |
| ⚪ Watch / No Edge | anything else | no clean technical entry |

**Caveats.**
* Past technical levels are not future fills. Names that pierce the 50w SMA in a fundamental break-down can keep falling for years (Cisco 2000, Meta 2022 briefly, etc.).
* MACD shown as confirmation only — never overrides the zone classification.
* "Avoid" only flags an *extension* — it is not a sell signal for an existing position.
* The scanner does not adjust for sector beta, news, earnings dates, or macro regime. Cross-check the *Macro & Rates*, *Regime*, and *Strategy Backtest* tabs before acting.
* **Not investment advice.** This is a deterministic technical-reference grid — no recommendation about any specific holding is intended.
            """)

    st.markdown("---")
    st.markdown("### 🔍 Single-Ticker Deep Dive")

    col_t, col_pd, col_d1, col_d2 = st.columns([1,1,1,1])
    with col_t:
        tech_ticker = st.text_input("Ticker", "SPY").upper()
    with col_pd:
        tech_period = st.selectbox("Period", ["6mo","1y","2y","5y"], index=1,
                                   key="tech_period")
        pstart_map = {"6mo": date.today().replace(month=max(1,date.today().month-6)),
                      "1y": date(date.today().year-1, date.today().month, date.today().day),
                      "2y": date(date.today().year-2, date.today().month, date.today().day),
                      "5y": date(date.today().year-5, date.today().month, date.today().day)}
    with col_d1:
        # fix #16: custom date range
        t_start = st.date_input("From", value=pstart_map[tech_period], key="tstart")
    with col_d2:
        t_end = st.date_input("To", value=date.today(), key="tend")

    df_t, info = fetch_stock(tech_ticker, str(t_start), str(t_end))
    if df_t is None or df_t.empty:
        st.error(f"No data for {tech_ticker}")
    else:
        # SMA200 guard — fix #3
        has_sma200 = df_t["SMA200"].notna().any()
        if not has_sma200:
            st.warning(f"⚠️ Period too short ({len(df_t)} days) for SMA200 — showing SMA50 instead.")

        last_t = df_t.iloc[-1]
        prev_t = df_t.iloc[-2] if len(df_t)>1 else df_t.iloc[-1]
        # fix #17: deltas on tech metrics
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Last Price",  f"${last_t['Close']:.2f}", f"{(last_t['Close']/prev_t['Close']-1)*100:+.2f}%")
        c2.metric("RSI (14)",    f"{last_t['RSI']:.1f}")
        c3.metric("MACD",        f"{last_t['MACD']:.3f}", f"Signal:{last_t['Signal']:.3f}")
        c4.metric("vs SMA50",    f"{'↑' if last_t['Close']>last_t['SMA50'] else '↓'} {abs(last_t['Close']/last_t['SMA50']-1)*100:.1f}%")
        c5.metric("vs SMA200",   f"{'↑' if has_sma200 and last_t['Close']>last_t['SMA200'] else '⚠️ N/A'}")

        # Candlestick + Volume
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            row_heights=[0.45,0.2,0.2,0.15], vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(x=df_t.index, open=df_t["Open"],high=df_t["High"],
                                     low=df_t["Low"],close=df_t["Close"],name="OHLC"), row=1,col=1)
        for sma, col, w in [("SMA20","#facc15",1),("SMA50","#38bdf8",1.5),
                             ("SMA200","#f87171",1) if has_sma200 else (None,None,None)]:
            if sma and sma in df_t:
                fig.add_trace(go.Scatter(x=df_t.index,y=df_t[sma],name=sma,
                                         line=dict(color=col,width=w),mode="lines"), row=1,col=1)
        fig.add_trace(go.Scatter(x=df_t.index,y=df_t["BB_upper"],
                                 line=dict(color="#818cf8",width=1,dash="dot"),name="BB Upper"), row=1,col=1)
        fig.add_trace(go.Scatter(x=df_t.index,y=df_t["BB_lower"],
                                 line=dict(color="#818cf8",width=1,dash="dot"),name="BB Lower",
                                 fill="tonexty",fillcolor="rgba(129,140,248,.05)"), row=1,col=1)
        # Volume (vectorized fix #5)
        fig.add_trace(go.Bar(x=df_t.index, y=df_t["Volume"],
                             marker_color=df_t["vol_color"].tolist(), name="Volume",
                             showlegend=False), row=2,col=1)
        # RSI
        fig.add_trace(go.Scatter(x=df_t.index,y=df_t["RSI"],name="RSI",
                                 line=dict(color="#34d399",width=1.5)), row=3,col=1)
        fig.add_hline(y=70,line_dash="dash",line_color="#f87171",row=3,col=1)
        fig.add_hline(y=30,line_dash="dash",line_color="#38bdf8",row=3,col=1)
        # MACD
        fig.add_trace(go.Bar(x=df_t.index,y=df_t["Hist"],
                             marker_color=np.where(df_t["Hist"]>=0,"#34d399","#f87171").tolist(),
                             name="MACD Hist",showlegend=False), row=4,col=1)
        fig.add_trace(go.Scatter(x=df_t.index,y=df_t["MACD"],name="MACD",
                                 line=dict(color="#38bdf8",width=1)), row=4,col=1)
        fig.add_trace(go.Scatter(x=df_t.index,y=df_t["Signal"],name="Signal",
                                 line=dict(color="#f472b6",width=1)), row=4,col=1)
        fig.update_layout(title=f"{tech_ticker} — Multi-Panel Technical Analysis",
                          height=800, xaxis_rangeslider_visible=False,
                          hovermode="x unified", **PT)
        update_axes(fig, "Date", "Analysis Value")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("ℹ️ Technical Analysis Guide"):
            st.write("""
            * **Candlesticks (OHLC)**: Shows the Open, High, Low, and Close for each day.
            * **Moving Averages (SMA)**: 50-day (blue) and 200-day (red) lines show trend direction. Price crossing above SMA is often bullish.
            * **RSI (14)**: Relative Strength Index. Values above 70 = Overbought; below 30 = Oversold.
            * **MACD**: Moving Average Convergence Divergence. When the blue line (MACD) crosses the pink line (Signal), a trend shift is indicated.
            * **OBV**: On-Balance Volume. Tracks cumulative volume flow. Prices rising with rising OBV confirms the trend.
            """)

        # OBV + ROC — fix #15
        c1,c2 = st.columns(2)
        with c1:
            fig_obv = go.Figure(go.Scatter(x=df_t.index,y=df_t["OBV"],
                                           fill="tozeroy",line=dict(color="#fb923c",width=1.5),
                                           fillcolor="rgba(251,146,60,.08)"))
            fig_obv.update_layout(title="On-Balance Volume (OBV)",**PT)
            update_axes(fig_obv, "Date", "OBV Units")
            st.plotly_chart(fig_obv, use_container_width=True)
        with c2:
            fig_roc = go.Figure(go.Scatter(x=df_t.index,y=df_t["ROC"],
                                           fill="tozeroy",line=dict(color="#818cf8",width=1.5),
                                           fillcolor="rgba(129,140,248,.08)"))
            fig_roc.add_hline(y=0,line_dash="dot",line_color="#475569")
            fig_roc.update_layout(title="Rate of Change / Momentum (ROC-10)",**PT)
            update_axes(fig_roc, "Date", "Change %")
            fig_roc.update_yaxes(ticksuffix="%")
            st.plotly_chart(fig_roc, use_container_width=True)

# ─── Tab 7: Risk Simulation ──────────────────────────────────────────
with tab7:
    render_methodology("risk_sim", st)

    @st.cache_data(ttl=3600, show_spinner=False)
    def run_mc(mu, vol, n, h=12):
        rng = np.random.default_rng(99)
        shocks = rng.normal(mu/12, vol/np.sqrt(12), (n, h))
        return np.exp(np.cumsum(shocks, axis=1))

    paths = run_mc(mc_mu/100, mc_vol/100, mc_n)
    p10,p25,p50,p75,p90 = [np.percentile(paths,q,axis=0) for q in [10,25,50,75,90]]
    x = list(range(1, 13))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x+x[::-1], y=list(p90)+list(p10[::-1]),
                             fill="toself",fillcolor="rgba(56,189,248,.06)",
                             line=dict(width=0),name="P10–P90"))
    fig.add_trace(go.Scatter(x=x+x[::-1], y=list(p75)+list(p25[::-1]),
                             fill="toself",fillcolor="rgba(56,189,248,.15)",
                             line=dict(width=0),name="P25–P75"))
    fig.add_trace(go.Scatter(x=x, y=p50, mode="lines",
                             line=dict(color="#38bdf8",width=2.5), name="Median"))
    fig.add_hline(y=1.0,line_dash="dot",line_color="#475569",annotation_text="Start")
    fig.update_layout(title=f"Monte Carlo Fan — μ={mc_mu}% σ={mc_vol}% ({mc_n:,} paths)",
                      hovermode="x unified", **PT)
    update_axes(fig, "Months Forward", "Cumulative Multiplier")
    fig.update_xaxes(tickprefix="M")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ How to read Monte Carlo simulations"):
        st.write("""
        * **Monte Carlo Fan**: Shows thousands of possible future price paths based on your chosen return (μ) and volatility (σ).
        * **Median (P50)**: The middle path; 50% of simulations ended above this, and 50% below.
        * **P10–P90 (Outer Band)**: Shows the range where 80% of all simulated outcomes fell.
        * **VaR 95% (Value at Risk)**: The threshold where there is only a 5% chance of doing worse. It represents your 'likely' worst-case monthly return.
        * **CVaR (Expected Shortfall)**: The average return in that bottom 5% tail. It tells you how bad things could get if you hit that tail.
        """)

    final_ret = (paths[:,-1]-1)*100
    c1,c2 = st.columns(2)
    with c1:
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=final_ret, nbinsx=80, marker_color="#38bdf8", opacity=.75,
                                    histnorm="probability density"))
        fig2.add_vline(x=np.percentile(final_ret,5), line_dash="dash", line_color="#f87171",
                       annotation_text="VaR 95%", annotation_position="top left")
        fig2.add_vline(x=np.median(final_ret), line_dash="solid", line_color="#34d399",
                       annotation_text="Median", annotation_position="top right")
        fig2.update_layout(title="Simulated 12M Return Distribution (%)", **PT)
        update_axes(fig2, "Total 12M Return %", "Probability Density")
        fig2.update_xaxes(ticksuffix="%")
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        cvar = final_ret[final_ret < np.percentile(final_ret,5)].mean()
        sim_stats = pd.DataFrame({
            "Metric": ["E[Return]","Median","VaR 95%","CVaR 95%","P10","P90","P(>0%)","P(>10%)"],
            "Value": [f"{final_ret.mean():.1f}%", f"{np.median(final_ret):.1f}%",
                      f"{np.percentile(final_ret,5):.1f}%", f"{cvar:.1f}%",
                      f"{np.percentile(final_ret,10):.1f}%", f"{np.percentile(final_ret,90):.1f}%",
                      f"{(final_ret>0).mean()*100:.1f}%", f"{(final_ret>10).mean()*100:.1f}%"]
        })
        st.markdown("#### 📋 Simulation Statistics")
        st.dataframe(sim_stats, use_container_width=True, hide_index=True)

# ─── Tab 8: Gemini AI Analyst ────────────────────────────────────────
# Uses Frontier Unified GenAI SDK — Gemini 3 Flash Preview (2026 Fleet)
def _get_gemini_key():
    try:
        k = st.secrets.get("GEMINI_API_KEY")
        if k: return k
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "")

with tab8:
    render_methodology("gemini", st)
    st.markdown("### ✨ Gemini AI Macro Analyst")
    st.caption("Powered by Google Gemini 3 Flash via Unified GenAI SDK")

    gemini_key = _get_gemini_key()

    # Key input if not set
    if not gemini_key:
        st.info("Enter your Gemini API key to activate AI analysis. Get a free key at https://aistudio.google.com/app/apikey")
        gemini_key = st.text_input("Gemini API Key", type="password", placeholder="AIza...")

    if gemini_key:
        # Build context from live data
        latest_data = df.iloc[-1]
        prev_data   = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
        ytd_pct     = ytd if np.isfinite(ytd) else 0.0
        regime_now  = df["regime"].iloc[-1]
        regime_dist = df["regime"].value_counts(normalize=True) * 100

        macro_context = f"""
## Current Macro Dashboard Data ({df.index[-1].strftime('%B %Y')})

### Market Data
- S&P 500: {latest_data['sp500']:,.0f} ({(latest_data['sp500']/prev_data['sp500']-1)*100:+.2f}% MoM)
- YTD Return: {ytd_pct:.1f}%
- 10Y Treasury Yield: {latest_data['dgs10']:.2f}% ({(latest_data['dgs10']-prev_data['dgs10'])*100:+.0f}bps MoM)
- 12M Realized Vol: {latest_data['realized_vol_12m']*100:.1f}%
- 3M Realized Vol: {latest_data['realized_vol_3m']*100:.1f}%

### Macro Regime
- Current Regime: {regime_now}
- Regime Score: {latest_data['regime_score']:.2f} (positive = risk-off)
- Historical: Risk-On {regime_dist.get('Risk-On 🟢', 0):.0f}% | Neutral {regime_dist.get('Neutral 🟡', 0):.0f}% | Risk-Off {regime_dist.get('Risk-Off 🔴', 0):.0f}%

### Risk Metrics
- Annualized Return ({d_start} to {d_end}): {m['ann_ret']*100:.1f}%
- Sharpe Ratio: {m['sharpe']:.2f}
- Sortino Ratio: {'{:.2f}'.format(m['sortino']) if np.isfinite(m['sortino']) else 'N/A'}
- Max Drawdown: {m['mdd']*100:.1f}%
- Calmar Ratio: {'{:.2f}'.format(m['calmar']) if np.isfinite(m['calmar']) else 'N/A'}
- Win Rate: {m['win_rate']*100:.0f}%

### Yield Curve
- Yield Curve Slope: {latest_data['yc_slope']*100:.2f}% ({'Inverted - recession signal' if latest_data['yc_slope'] < 0 else 'Normal'})
- Credit Spread Proxy: {latest_data['credit_spread']*100:.2f}%

### Momentum
- 12-1 Momentum Signal: {latest_data['momentum_12_1']*100:.2f}%
"""

        # Analysis type selector
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Full Macro Briefing", "Regime Deep-Dive", "Risk Assessment", "Investment Outlook", "Custom Question"],
            key="gemini_analysis_type"
        )

        custom_q = ""
        if analysis_type == "Custom Question":
            custom_q = st.text_area("Your question", placeholder="e.g. Should I add duration risk given the current yield curve?")

        prompts = {
            "Full Macro Briefing": f"""{macro_context}

As a senior hedge fund macro analyst, write a concise but rigorous investment briefing covering:
1. **Macro Environment** — Rate regime, credit conditions, volatility environment
2. **Regime Assessment** — What the current regime implies for asset allocation
3. **Key Risks** — Top 3 tail risks the data is signaling
4. **Actionable View** — Specific positioning recommendations (equities, duration, credit, commodities)
5. **Monitoring Triggers** — What metrics to watch for regime change

Use precise, institutional language. Be direct and opinionated.""",

            "Regime Deep-Dive": f"""{macro_context}

Focus exclusively on the macro regime analysis:
1. What does the current regime score of {latest_data['regime_score']:.2f} imply?
2. How does current vol ({latest_data['realized_vol_12m']*100:.1f}% 12M) and credit spread ({latest_data['credit_spread']*100:.2f}%) compare to historical norms?
3. What typically happens next when transitioning from this regime?
4. How should a long/short equity fund position across regime transitions?""",

            "Risk Assessment": f"""{macro_context}

Provide a rigorous risk assessment:
1. Is the Sharpe of {m['sharpe']:.2f} and Sortino of {'{:.2f}'.format(m['sortino']) if np.isfinite(m['sortino']) else 'N/A'} adequate for current conditions?
2. Does the {m['mdd']*100:.1f}% max drawdown represent historically elevated risk?
3. What does the {latest_data['yc_slope']*100:.2f}% yield curve slope imply for forward equity returns?
4. Tail risk scenarios: what would trigger a -20%/-30% equity move from here?""",

            "Investment Outlook": f"""{macro_context}

Give a 3-6 month forward investment outlook:
1. Expected return range for US equities given current macro
2. Fixed income: duration add or reduce?
3. Credit: tighten or widen spreads?
4. Commodities: gold and oil direction
5. Key catalyst calendar to watch""",

            "Custom Question": f"""{macro_context}

User Question: {custom_q}

Answer as a senior macro analyst using the data above."""
        }

        run_analysis = st.button("🚀 Generate Gemini Analysis", type="primary", key="run_gemini")

        if run_analysis and (analysis_type != "Custom Question" or custom_q.strip()):
            try:
                with st.spinner("Gemini is analyzing macro conditions..."):
                    client = genai.Client(api_key=gemini_key)
                    response = client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=prompts[analysis_type],
                        config=genai_types.GenerateContentConfig(
                            system_instruction=(
                                "You are a senior quantitative macro analyst at a leading hedge fund. "
                                "Your analysis is precise, data-driven, and actionable. "
                                "You interpret financial data with institutional rigor."
                            )
                        )
                    )

                st.markdown("---")
                st.markdown("#### 🧠 Gemini Analysis")
                
                # Enhanced readability container for AI response
                st.markdown(f"""
                <div class="analysis-card">
                    {response.text}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.caption(f"Model: gemini-3.6-flash | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Data through {df.index[-1].strftime('%B %Y')}")

                # Token usage
                if hasattr(response, 'usage_metadata'):
                    st.caption(f"Tokens — Input: {response.usage_metadata.prompt_token_count} | Output: {response.usage_metadata.candidates_token_count}")

            except Exception as e:
                st.error(f"Gemini API error: {e}")
                if "API_KEY" in str(e).upper() or "invalid" in str(e).lower():
                    st.info("Check your API key. Get one free at https://aistudio.google.com/app/apikey")
        elif run_analysis and analysis_type == "Custom Question" and not custom_q.strip():
            st.warning("Please enter your question first.")
    else:
        st.warning("Add a Gemini API key above to unlock AI-powered macro analysis.")

    # Show what data is being fed to Gemini
    with st.expander("📄 View Macro Context Sent to Gemini"):
        if gemini_key:
            st.code(macro_context, language="markdown")
        else:
            st.info("Enter API key first to preview context.")

# ══════════════════════════════════════════
# TAB 9: NVDA DANGER ZONE & MICRO FOOTPRINT
# ══════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner="🔥 Fetching NVDA data…")
def fetch_nvda_full(period_days: int = 365):
    """Fetch NVDA + SOX + AI peers; compute danger indicators from OHLCV."""
    import yfinance as yf

    # ── NVDA daily OHLCV ────────────────────────────────────────────────
    nvda = yf.Ticker("NVDA")
    df_nvda = nvda.history(period=f"{period_days}d", auto_adjust=True)
    if df_nvda.empty:
        return {}, {}

    df = df_nvda.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    # ── 1. Price / trend indicators ─────────────────────────────────────
    df["SMA20"]  = df["Close"].rolling(20).mean()
    df["SMA50"]  = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["EMA12"]  = df["Close"].ewm(span=12).mean()
    df["EMA26"]  = df["Close"].ewm(span=26).mean()
    df["MACD"]   = df["EMA12"] - df["EMA26"]
    df["Signal_line"] = df["MACD"].ewm(span=9).mean()

    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - 100 / (1 + gain / (loss + 1e-9))

    # ── 2. Volatility (ATR-based danger) ────────────────────────────────
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"]  - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()
    df["ATR_pct"] = df["ATR14"] / df["Close"] * 100  # volatility %

    # ── 3. Volume analysis (block trade proxy) ───────────────────────────
    df["vol_sma20"] = df["Volume"].rolling(20).mean()
    df["vol_ratio"] = df["Volume"] / (df["vol_sma20"] + 1)  # relative volume
    df["block_flag"] = df["vol_ratio"] > 2.0               # 2× avg = block-trade proxy
    df["block_volume"] = np.where(df["block_flag"], df["Volume"], np.nan)

    # ── 4. Bid-Ask imbalance proxy (intraday close vs midpoint) ─────────
    df["mid"] = (df["High"] + df["Low"]) / 2
    df["ba_imbalance"] = (df["Close"] - df["mid"]) / (df["High"] - df["Low"] + 1e-3) * 100

    # ── 5. Put/Call skew proxy via VIX divergence ────────────────────────
    vix_raw = yf.download("^VIX", period=f"{period_days}d", auto_adjust=True, progress=False, multi_level_index=False)["Close"]
    if isinstance(vix_raw, pd.DataFrame):
        vix_raw = vix_raw.iloc[:, 0]
    vix = vix_raw.tz_localize(None) if hasattr(vix_raw.index, 'tz') and vix_raw.index.tz is not None else vix_raw
    vix.name = "VIX"
    df = df.join(vix.rename("VIX"), how="left")
    df["VIX"] = df["VIX"].ffill()
    df["vol_spike"] = df["ATR_pct"] > df["ATR_pct"].rolling(30).mean() + df["ATR_pct"].rolling(30).std()

    # ── 6. Danger Index (composite) ──────────────────────────────────────
    # Normalize each component to [0, 1]
    def _norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    rsi_danger  = _norm(df["RSI"].clip(50, 90))        # elevated RSI = danger
    atr_danger  = _norm(df["ATR_pct"])                 # high vol = danger
    vol_danger  = _norm(df["vol_ratio"].clip(0, 5))    # vol surge = danger
    vix_danger  = _norm(df["VIX"].clip(10, 60))        # high VIX = broader risk
    close_vs_sma = np.where(df["Close"] > df["SMA50"],
                             (df["Close"] / df["SMA50"] - 1).clip(0, 0.3),
                             0)
    ext_danger  = _norm(pd.Series(close_vs_sma, index=df.index))  # price extension

    df["danger_index"] = (
        0.30 * rsi_danger +
        0.25 * atr_danger +
        0.20 * vol_danger +
        0.15 * vix_danger +
        0.10 * ext_danger
    ).rolling(3).mean()  # smooth slightly

    df["danger_label"] = pd.cut(
        df["danger_index"],
        bins=[0, 0.33, 0.60, 1.01],
        labels=["Safe Zone 🟢", "Caution ⚠️", "Danger Zone 🔴"]
    ).astype(str)

    # ── 7. Market context peers ───────────────────────────────────────────
    peers = {"NVDA": "NVDA", "SOX": "SOXX", "AMD": "AMD", "TSM": "TSM",
             "AVGO": "AVGO", "MU": "MU"}
    ctx_frames = {}
    for name, tkr in peers.items():
        try:
            raw = yf.download(tkr, period=f"{period_days}d",
                              auto_adjust=True, progress=False, multi_level_index=False)["Close"]
            if isinstance(raw, pd.DataFrame):
                raw = raw.iloc[:, 0]
            raw.index = pd.to_datetime(raw.index).tz_localize(None)
            ctx_frames[name] = raw
        except Exception:
            pass
    df_ctx = pd.DataFrame(ctx_frames).ffill().dropna()
    # Normalize to 100
    df_ctx = df_ctx / df_ctx.iloc[0] * 100

    return df, df_ctx


# ── Helper: danger zone colour band ──
def _danger_color(val):
    if val >= 0.60: return "#f87171"   # red
    if val >= 0.33: return "#facc15"   # yellow
    return "#34d399"                   # green


with tab9:
    render_methodology("danger_zone", st)
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
      <span style="font-size:2rem;">🔥</span>
      <div>
        <h2 style="margin:0;background:linear-gradient(135deg,#f87171,#facc15,#fb923c);
          -webkit-background-clip:text;-webkit-text-fill-color:transparent;
          font-size:1.7rem;font-weight:800;">NVDA Danger Zone & Micro Footprint</h2>
        <p style="margin:0;color:#64748b;font-size:.82rem;">
          Composite danger scoring · Volume footprint · AI peer context
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Controls
    col_np, col_nv = st.columns([2, 1])
    with col_np:
        nvda_period = st.selectbox(
            "Lookback Period",
            ["3mo (90d)", "6mo (180d)", "1yr (365d)", "2yr (730d)"],
            index=2, key="nvda_period"
        )
        _period_map_nvda = {"3mo (90d)": 90, "6mo (180d)": 180,
                            "1yr (365d)": 365, "2yr (730d)": 730}
        nvda_days = _period_map_nvda[nvda_period]
    with col_nv:
        nvda_detail = st.selectbox("Chart Detail",
                                   ["Daily (recommended)", "Weekly pivot"], index=0,
                                   key="nvda_detail")

    with st.spinner("🔥 Computing NVDA Danger Index…"):
        df_nv, df_ctx_nv = fetch_nvda_full(nvda_days)

    if df_nv is None or (isinstance(df_nv, dict) and len(df_nv) == 0):
        st.error("Could not fetch NVDA data. Check network or try again.")
    else:
        last_nv = df_nv.iloc[-1]
        prev_nv = df_nv.iloc[-2] if len(df_nv) > 1 else df_nv.iloc[-1]
        danger_now = last_nv["danger_index"]

        # ── KPI row ─────────────────────────────────────────────────────────
        kc1, kc2, kc3, kc4, kc5, kc6 = st.columns(6)
        kc1.metric("NVDA Price",    f"${last_nv['Close']:.2f}",
                   f"{(last_nv['Close']/prev_nv['Close']-1)*100:+.2f}%")
        kc2.metric("RSI (14)",      f"{last_nv['RSI']:.1f}",
                   "Overbought" if last_nv['RSI'] > 70 else
                   ("Oversold" if last_nv['RSI'] < 30 else "Neutral"))
        kc3.metric("ATR Vol %",     f"{last_nv['ATR_pct']:.1f}%")
        kc4.metric("Rel Volume",    f"{last_nv['vol_ratio']:.1f}×")
        kc5.metric("VIX",           f"{last_nv['VIX']:.1f}" if not np.isnan(last_nv['VIX']) else "—")
        danger_color_text = "🔴 DANGER" if danger_now >= 0.6 else ("⚠️ CAUTION" if danger_now >= 0.33 else "🟢 SAFE")
        kc6.metric("Danger Index",  f"{danger_now:.2f}", danger_color_text)

        # ── Danger Index banner ─────────────────────────────────────────────
        dz_bg  = {"🔴 DANGER": "#3b0e0e", "⚠️ CAUTION": "#332e00", "🟢 SAFE": "#0d3320"}.get(danger_color_text, "#131d35")
        dz_bd  = {"🔴 DANGER": "#f87171", "⚠️ CAUTION": "#facc15", "🟢 SAFE": "#34d399"}.get(danger_color_text, "#38bdf8")
        st.markdown(f"""<div style="background:{dz_bg}; border-left:4px solid {dz_bd};
            padding:12px 20px; border-radius:8px; margin:10px 0; font-size:.95rem;
            color:#fff; font-weight:500; box-shadow:0 4px 15px rgba(0,0,0,.3);">
            <b>Current Zone: {danger_color_text}</b> &nbsp;|&nbsp;
            Danger Index: <b>{danger_now:.3f}</b> (0=Safe, 1=Extreme Risk)
            &nbsp;|&nbsp; as of <b>{df_nv.index[-1].strftime('%b %d, %Y')}</b>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ════════════════════════════════════════════════
        # PANEL 1 — DANGER ZONE MAP (Price + Danger Band)
        # ════════════════════════════════════════════════
        st.markdown("#### 1️⃣ Danger Zone Map — Price vs Composite Risk Index")
        with st.expander("ℹ️ How to read the Danger Zone Map", expanded=False):
            st.write("""
            * **Danger Index** (right axis, 0–1): Composite score built from RSI extension,
              ATR volatility, relative volume surge, VIX level, and price extension above SMA50.
            * **Red zone (≥0.60)**: High probability of near-term mean-reversion or correction.
              Consider partial trim / protective puts.
            * **Yellow zone (0.33–0.60)**: Elevated caution. Monitor position sizing.
            * **Green zone (<0.33)**: Normal risk environment — trend continuation bias.
            """)

        fig_dz = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.62, 0.38],
            vertical_spacing=0.04,
            subplot_titles=("NVDA Price + Moving Averages", "Danger Index (0 = Safe → 1 = Extreme)")
        )

        # Price candlestick
        fig_dz.add_trace(go.Candlestick(
            x=df_nv.index, open=df_nv["Open"], high=df_nv["High"],
            low=df_nv["Low"], close=df_nv["Close"], name="NVDA OHLC",
            increasing_line_color="#34d399", decreasing_line_color="#f87171"
        ), row=1, col=1)

        for sma_col, sma_clr, sma_w in [("SMA20", "#facc15", 1.2),
                                         ("SMA50", "#38bdf8", 1.5),
                                         ("SMA200", "#f472b6", 1.2)]:
            if df_nv[sma_col].notna().any():
                fig_dz.add_trace(go.Scatter(
                    x=df_nv.index, y=df_nv[sma_col],
                    name=sma_col, mode="lines",
                    line=dict(color=sma_clr, width=sma_w)
                ), row=1, col=1)

        # Danger index fill (coloured by zone)
        di = df_nv["danger_index"].dropna()

        # background shading for danger zones
        fig_dz.add_hrect(y0=0.60, y1=1.0,  fillcolor="rgba(248,113,113,0.12)",
                          line_width=0, row=2, col=1)
        fig_dz.add_hrect(y0=0.33, y1=0.60, fillcolor="rgba(250,204,21,0.10)",
                          line_width=0, row=2, col=1)
        fig_dz.add_hrect(y0=0.0,  y1=0.33, fillcolor="rgba(52,211,153,0.08)",
                          line_width=0, row=2, col=1)

        fig_dz.add_trace(go.Scatter(
            x=di.index, y=di,
            fill="tozeroy", mode="lines",
            fillcolor="rgba(248,113,113,0.15)",
            line=dict(color="#f87171", width=1.8),
            name="Danger Index"
        ), row=2, col=1)

        # reference lines
        fig_dz.add_hline(y=0.60, line_dash="dash", line_color="#f87171",
                          annotation_text="🔴 Danger (0.60)", annotation_position="right", row=2, col=1)
        fig_dz.add_hline(y=0.33, line_dash="dash", line_color="#facc15",
                          annotation_text="⚠️ Caution (0.33)", annotation_position="right", row=2, col=1)

        fig_dz.update_layout(
            title="NVDA Danger Zone Dashboard",
            height=640, xaxis_rangeslider_visible=False,
            hovermode="x unified", **PT
        )
        fig_dz.update_xaxes(gridcolor="rgba(255,255,255,0.04)")
        fig_dz.update_yaxes(gridcolor="rgba(255,255,255,0.04)")
        st.plotly_chart(fig_dz, use_container_width=True)

        # ════════════════════════════════════════════════
        # PANEL 2 — MICRO FOOTPRINT
        # ════════════════════════════════════════════════
        st.markdown("#### 2️⃣ Micro Footprint Panel")
        with st.expander("ℹ️ Micro Footprint Guide", expanded=False):
            st.write("""
            * **Block Trade Volume**: Days where volume exceeds 2× the 20-day average —
              signals institutional accumulation or distribution.
            * **Bid/Ask Imbalance Proxy**: (Close – Midpoint) / Range. Positive = buyers
              absorbed supply (bullish); Negative = sellers dominated (bearish).
            * **Volatility Spike**: Days where ATR% exceeds 1σ above its 30-day mean —
              often precedes significant directional moves.
            * **Relative Volume**: Current volume vs 20-day average. Values >2× = abnormal activity.
            """)

        fig_fp = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.40, 0.35, 0.25],
            vertical_spacing=0.05,
            subplot_titles=("Volume & Block Trades",
                            "Bid/Ask Imbalance Proxy (%)",
                            "Relative Volume (vs 20d avg)")
        )

        # Volume bars (coloured by up/down)
        vol_colors = np.where(
            df_nv["Close"] >= df_nv["Open"], "rgba(52,211,153,0.6)", "rgba(248,113,113,0.6)"
        ).tolist()
        fig_fp.add_trace(go.Bar(
            x=df_nv.index, y=df_nv["Volume"],
            marker_color=vol_colors, name="Volume", showlegend=True
        ), row=1, col=1)

        # Block trade overlay — orange bars on top
        block_idx = df_nv["block_flag"]
        fig_fp.add_trace(go.Bar(
            x=df_nv.index[block_idx],
            y=df_nv["Volume"][block_idx],
            marker_color="rgba(251,146,60,0.9)",
            name="Block Trade (>2× vol)"
        ), row=1, col=1)

        # Volatile spike markers
        spike_mask = df_nv["vol_spike"] == True
        if spike_mask.any():
            fig_fp.add_trace(go.Scatter(
                x=df_nv.index[spike_mask],
                y=df_nv["Volume"][spike_mask],
                mode="markers",
                marker=dict(color="#a78bfa", size=10, symbol="diamond",
                            line=dict(color="#fff", width=1)),
                name="Volatility Spike"
            ), row=1, col=1)

        # Bid/Ask imbalance
        ba = df_nv["ba_imbalance"]
        ba_colors = np.where(ba >= 0, "rgba(52,211,153,0.7)", "rgba(248,113,113,0.7)").tolist()
        fig_fp.add_trace(go.Bar(
            x=df_nv.index, y=ba,
            marker_color=ba_colors, name="B/A Imbalance %"
        ), row=2, col=1)
        fig_fp.add_hline(y=0, line_dash="dot", line_color="#475569", row=2, col=1)

        # Relative volume line
        vol_ratio_clr = np.where(
            df_nv["vol_ratio"] > 2, "#fb923c", "#38bdf8"
        ).tolist()
        fig_fp.add_trace(go.Scatter(
            x=df_nv.index, y=df_nv["vol_ratio"],
            mode="lines", name="Rel Volume",
            fill="tozeroy",
            fillcolor="rgba(56,189,248,0.07)",
            line=dict(color="#38bdf8", width=1.5)
        ), row=3, col=1)
        fig_fp.add_hline(y=2.0, line_dash="dash", line_color="#fb923c",
                          annotation_text="Block threshold (2×)",
                          annotation_position="right", row=3, col=1)

        fig_fp.update_layout(
            height=620, barmode="overlay",
            hovermode="x unified", **PT
        )
        fig_fp.update_xaxes(gridcolor="rgba(255,255,255,0.04)")
        fig_fp.update_yaxes(gridcolor="rgba(255,255,255,0.04)")
        st.plotly_chart(fig_fp, use_container_width=True)

        # ════════════════════════════════════════════════
        # PANEL 3 — MARKET CONTEXT (SOX / AI Peers)
        # ════════════════════════════════════════════════
        st.markdown("#### 3️⃣ Market Context — NVDA vs SOX & AI Chip Peers")
        with st.expander("ℹ️ Context Panel Guide", expanded=False):
            st.write("""
            * All lines normalised to 100 at the start of the selected period.
            * **NVDA vs SOXX**: Divergence between NVDA and the broader semiconductor
              ETF can signal stock-specific over/underperformance.
            * **AI Peers** (AMD, TSM, AVGO, MU): When NVDA is the only outlier
              (up or down), that's often a mean-reversion opportunity.
            """)

        if not df_ctx_nv.empty:
            fig_ctx = go.Figure()
            ctx_colors = ["#38bdf8", "#818cf8", "#34d399",
                           "#fb923c", "#f472b6", "#facc15"]
            for i, col in enumerate(df_ctx_nv.columns):
                width = 2.5 if col == "NVDA" else 1.4
                dash  = "solid" if col == "NVDA" else "dot"
                fig_ctx.add_trace(go.Scatter(
                    x=df_ctx_nv.index, y=df_ctx_nv[col],
                    name=col, mode="lines",
                    line=dict(color=ctx_colors[i % len(ctx_colors)],
                              width=width, dash=dash)
                ))
            fig_ctx.add_hline(y=100, line_dash="dot", line_color="#475569",
                               annotation_text="Base = 100")
            fig_ctx.update_layout(
                title=f"Indexed Performance (base=100) — NVDA vs Peers",
                hovermode="x unified", **PT
            )
            update_axes(fig_ctx, "Date", "Indexed Return (Base = 100)")
            st.plotly_chart(fig_ctx, use_container_width=True)

            # Relative strength table
            st.markdown("#### 📋 Performance Comparison Table")
            first_row = df_ctx_nv.iloc[0]
            last_row  = df_ctx_nv.iloc[-1]
            perf_rows = []
            for c in df_ctx_nv.columns:
                ret = (last_row[c] / first_row[c] - 1) * 100
                perf_rows.append({"Asset": c, "Return (%)": f"{ret:+.1f}%",
                                   "vs NVDA": f"{ret - (last_row['NVDA']/first_row['NVDA']-1)*100:+.1f}%"
                                   if c != "NVDA" else "—"})
            st.dataframe(
                pd.DataFrame(perf_rows),
                use_container_width=True, hide_index=True
            )
        else:
            st.warning("Could not load peer comparison data.")

        # ════════════════════════════════════════════════
        # PANEL 4 — RSI + MACD combo
        # ════════════════════════════════════════════════
        st.markdown("#### 4️⃣ Momentum Indicators — RSI & MACD")
        fig_mom = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.5, 0.5], vertical_spacing=0.05,
            subplot_titles=("RSI (14)", "MACD")
        )
        # RSI
        fig_mom.add_trace(go.Scatter(
            x=df_nv.index, y=df_nv["RSI"],
            fill="tozeroy", fillcolor="rgba(52,211,153,0.08)",
            line=dict(color="#34d399", width=1.8), name="RSI"
        ), row=1, col=1)
        fig_mom.add_hline(y=70, line_dash="dash", line_color="#f87171",
                           annotation_text="Overbought (70)",
                           annotation_position="right", row=1, col=1)
        fig_mom.add_hline(y=30, line_dash="dash", line_color="#38bdf8",
                           annotation_text="Oversold (30)",
                           annotation_position="right", row=1, col=1)

        # MACD
        hist_colors = np.where(
            df_nv["MACD"] - df_nv["Signal_line"] >= 0,
            "rgba(52,211,153,0.6)", "rgba(248,113,113,0.6)"
        ).tolist()
        fig_mom.add_trace(go.Bar(
            x=df_nv.index,
            y=df_nv["MACD"] - df_nv["Signal_line"],
            marker_color=hist_colors, name="MACD Histogram"
        ), row=2, col=1)
        fig_mom.add_trace(go.Scatter(
            x=df_nv.index, y=df_nv["MACD"],
            line=dict(color="#38bdf8", width=1.5), name="MACD Line"
        ), row=2, col=1)
        fig_mom.add_trace(go.Scatter(
            x=df_nv.index, y=df_nv["Signal_line"],
            line=dict(color="#f472b6", width=1.5, dash="dot"), name="Signal Line"
        ), row=2, col=1)
        fig_mom.update_layout(height=480, hovermode="x unified", **PT)
        fig_mom.update_xaxes(gridcolor="rgba(255,255,255,0.04)")
        fig_mom.update_yaxes(gridcolor="rgba(255,255,255,0.04)")
        st.plotly_chart(fig_mom, use_container_width=True)

        # ── Strategy hint ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 💡 Danger Zone Strategy Hints")
        hint_cols = st.columns(3)
        with hint_cols[0]:
            st.markdown("""
            <div style="background:rgba(248,113,113,0.08);border:1px solid rgba(248,113,113,0.3);
                border-radius:12px;padding:20px;">
            <b style="color:#f87171;">🔴 Danger Zone (≥0.60)</b><br>
            <ul style="color:#cbd5e1;margin-top:8px;font-size:.9rem;">
            <li>Consider partial position trim (20-40%)</li>
            <li>Buy protective puts (1-2 strike OTM)</li>
            <li>Tighten stop-loss to recent swing low</li>
            <li>Avoid new longs until RSI resets below 60</li>
            </ul></div>
            """, unsafe_allow_html=True)
        with hint_cols[1]:
            st.markdown("""
            <div style="background:rgba(250,204,21,0.08);border:1px solid rgba(250,204,21,0.3);
                border-radius:12px;padding:20px;">
            <b style="color:#facc15;">⚠️ Caution Zone (0.33–0.60)</b><br>
            <ul style="color:#cbd5e1;margin-top:8px;font-size:.9rem;">
            <li>Maintain position but no additions</li>
            <li>Monitor block trade signals (orange bars)</li>
            <li>Watch for vol_ratio >2 as early warning</li>
            <li>Reduce on any failed breakout attempt</li>
            </ul></div>
            """, unsafe_allow_html=True)
        with hint_cols[2]:
            st.markdown("""
            <div style="background:rgba(52,211,153,0.08);border:1px solid rgba(52,211,153,0.3);
                border-radius:12px;padding:20px;">
            <b style="color:#34d399;">🟢 Safe Zone (<0.33)</b><br>
            <ul style="color:#cbd5e1;margin-top:8px;font-size:.9rem;">
            <li>Trend continuation bias — hold or add</li>
            <li>Confirm with rising OBV / positive B/A</li>
            <li>Look for accumulation block trades</li>
            <li>Set trailing stop 10-15% below close</li>
            </ul></div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# Strategy Backtest engine — regime-gated SPY/cash with no-lookahead lag
# ══════════════════════════════════════════════════════════════════════
def run_strategy_backtest(macro_df: pd.DataFrame, *, use_regime: bool, use_momentum: bool,
                          use_trend: bool, threshold: float, cost_bps: float,
                          allow_short: bool):
    """Walk-forward, no-lookahead backtest of an SPY/cash (or SPY/-SPY) strategy.

    Position is decided from data observable at month T, then applied to the
    realised return from T to T+1 via .shift(1). Net returns include linear
    transaction costs proportional to turnover.
    """
    out = macro_df.copy()
    out = out.dropna(subset=["sp500_ret_m"])

    # signals (each ∈ {0,1}, evaluated at month T)
    out["sig_regime"]   = (out["regime"] != "Risk-Off 🔴").astype(int)
    out["sig_momentum"] = (out["momentum_12_1"] > 0).astype(int)
    sma_10m = out["sp500"].rolling(10).mean()
    out["sig_trend"]    = (out["sp500"] > sma_10m).astype(int)

    cols = []
    if use_regime:   cols.append("sig_regime")
    if use_momentum: cols.append("sig_momentum")
    if use_trend:    cols.append("sig_trend")

    if not cols:
        out["sig_score"] = 1.0
    else:
        out["sig_score"] = out[cols].mean(axis=1)

    in_market = (out["sig_score"] >= threshold).astype(float)
    if allow_short:
        out_market = -1.0 * (out["sig_score"] < (1 - threshold)).astype(float)
        target_w = in_market + out_market
    else:
        target_w = in_market

    out["target_w"] = target_w
    out["position"] = out["target_w"].shift(1).fillna(0.0)        # NO-LOOKAHEAD
    out["turnover"] = out["position"].diff().abs().fillna(out["position"].abs())

    cost = (cost_bps / 10000.0) * out["turnover"]
    out["strat_ret_gross"] = out["position"] * out["sp500_ret_m"]
    out["strat_ret"]       = out["strat_ret_gross"] - cost
    out["strat_equity"]    = np.exp(out["strat_ret"].fillna(0.0).cumsum()) * 100
    out["bh_equity"]       = np.exp(out["sp500_ret_m"].fillna(0.0).cumsum()) * 100
    out["strat_dd"]        = out["strat_equity"] / out["strat_equity"].cummax() - 1
    out["bh_dd"]           = out["bh_equity"]    / out["bh_equity"].cummax()    - 1
    return out

# ─── Tab 10: Strategy Backtest ────────────────────────────────────────
with tab10:
    render_methodology("backtest", st)
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
      <span style="font-size:2rem;">📊</span>
      <div>
        <h2 style="margin:0;background:linear-gradient(135deg,#38bdf8,#818cf8,#34d399);
          -webkit-background-clip:text;-webkit-text-fill-color:transparent;
          font-size:1.7rem;font-weight:800;">Strategy Backtest</h2>
        <p style="margin:0;color:#64748b;font-size:.82rem;">
          Walk-forward, no-lookahead · Regime + momentum + trend gating · Net of costs
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.info("📐 **Construction**: at month *T* the engine reads only data observable at *T*, "
            "then holds the position through month *T+1*. Position lag = 1 month "
            "(no-lookahead). Cash leg earns 0%. Net = gross − bps × turnover.")

    cfg1, cfg2 = st.columns([3, 2])
    with cfg1:
        st.markdown("**🎛️ Signal Gates**")
        use_reg = st.checkbox("Regime filter (long when not Risk-Off)", value=True, key="bt_reg")
        use_mom = st.checkbox("12-1 momentum > 0",                       value=True, key="bt_mom")
        use_tr  = st.checkbox("Price > 10-month SMA (Faber rule)",       value=True, key="bt_tr")
        thr     = st.slider("Aggregate signal threshold (avg of selected ≥ x → long)",
                            0.34, 1.0, 0.5, 0.01, key="bt_thr")
        allow_short = st.checkbox("Allow short when avg ≤ 1−threshold", value=False, key="bt_short")
    with cfg2:
        st.markdown("**💸 Frictions**")
        cost_bps = st.slider("Round-trip cost (bps per unit turnover)", 0, 50, 5, 1, key="bt_cost")
        st.caption("5 bps ≈ retail SPY ETF execution. Increase to stress-test net Sharpe.")

    bt = run_strategy_backtest(df, use_regime=use_reg, use_momentum=use_mom,
                               use_trend=use_tr, threshold=thr, cost_bps=cost_bps,
                               allow_short=allow_short)

    if bt["strat_ret"].dropna().empty:
        st.error("No data after filters — widen the date range or relax the threshold.")
    else:
        # Tear-sheet metrics
        m_strat = compute_hf_metrics(bt["strat_ret"].dropna(), bt["sp500_ret_m"].dropna())
        m_bh    = compute_hf_metrics(bt["sp500_ret_m"].dropna())
        time_in = bt["position"].abs().mean() * 100
        n_flips = int((bt["position"].diff().abs() > 1e-9).sum())

        k1,k2,k3,k4,k5,k6 = st.columns(6)
        k1.metric("Strat Ann Ret", f"{m_strat['ann_ret']*100:.2f}%",
                  f"{(m_strat['ann_ret']-m_bh['ann_ret'])*100:+.2f}% vs B&H")
        k2.metric("Strat Sharpe",  f"{m_strat['sharpe']:.2f}",
                  f"{(m_strat['sharpe']-m_bh['sharpe']):+.2f}" if np.isfinite(m_bh['sharpe']) else "—")
        k3.metric("Strat Max DD",  f"{m_strat['mdd']*100:.1f}%",
                  f"{(m_strat['mdd']-m_bh['mdd'])*100:+.1f}% vs B&H")
        k4.metric("Calmar",        f"{m_strat['calmar']:.2f}" if np.isfinite(m_strat['calmar']) else "—")
        k5.metric("Time in Market", f"{time_in:.1f}%")
        k6.metric("Position Flips", f"{n_flips}")

        # Equity curves
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=bt.index, y=bt["bh_equity"], name="SPY Buy & Hold",
                                    line=dict(color="#facc15", width=1.5, dash="dot")))
        fig_eq.add_trace(go.Scatter(x=bt.index, y=bt["strat_equity"], name="Strategy (net)",
                                    line=dict(color="#38bdf8", width=2.2),
                                    fill="tozeroy", fillcolor="rgba(56,189,248,.06)"))
        fig_eq.update_layout(title="Equity Curves (Base = 100, log returns compounded)",
                             hovermode="x unified", **PT)
        update_axes(fig_eq, "Date", "Equity Index")
        st.plotly_chart(fig_eq, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=bt.index, y=bt["bh_dd"]*100, name="B&H DD",
                                        line=dict(color="#facc15", width=1.2, dash="dot")))
            fig_dd.add_trace(go.Scatter(x=bt.index, y=bt["strat_dd"]*100, name="Strategy DD",
                                        line=dict(color="#f87171", width=1.6),
                                        fill="tozeroy", fillcolor="rgba(248,113,113,.10)"))
            fig_dd.update_layout(title="Rolling Drawdown — Strategy vs B&H",
                                 hovermode="x unified", **PT)
            fig_dd.update_yaxes(ticksuffix="%")
            update_axes(fig_dd, "Date", "Drawdown")
            st.plotly_chart(fig_dd, use_container_width=True)
        with c2:
            fig_pos = go.Figure()
            fig_pos.add_trace(go.Scatter(x=bt.index, y=bt["position"], name="Position",
                                         line=dict(color="#34d399", width=1.6),
                                         fill="tozeroy", fillcolor="rgba(52,211,153,.12)"))
            fig_pos.add_hline(y=0, line_dash="dash", line_color="#64748b")
            fig_pos.update_layout(title="Position Through Time (lagged 1m, no-lookahead)",
                                  hovermode="x unified", **PT)
            update_axes(fig_pos, "Date", "Weight in SPY")
            st.plotly_chart(fig_pos, use_container_width=True)

        # Side-by-side tear sheet
        st.markdown("#### 📋 Side-by-Side Tear Sheet")
        rows = [
            ("Ann Return",   f"{m_strat['ann_ret']*100:.2f}%",   f"{m_bh['ann_ret']*100:.2f}%"),
            ("Ann Vol",      f"{m_strat['ann_vol']*100:.2f}%",   f"{m_bh['ann_vol']*100:.2f}%"),
            ("Sharpe",       f"{m_strat['sharpe']:.3f}",         f"{m_bh['sharpe']:.3f}"),
            ("Sortino",      f"{m_strat['sortino']:.3f}" if np.isfinite(m_strat['sortino']) else "—",
                              f"{m_bh['sortino']:.3f}"   if np.isfinite(m_bh['sortino'])   else "—"),
            ("Max Drawdown", f"{m_strat['mdd']*100:.2f}%",       f"{m_bh['mdd']*100:.2f}%"),
            ("Calmar",       f"{m_strat['calmar']:.3f}" if np.isfinite(m_strat['calmar']) else "—",
                              f"{m_bh['calmar']:.3f}"   if np.isfinite(m_bh['calmar'])   else "—"),
            ("Win Rate",     f"{m_strat['win_rate']*100:.1f}%",   f"{m_bh['win_rate']*100:.1f}%"),
            ("Alpha vs SPY", f"{m_strat['alpha']*100:.2f}%" if np.isfinite(m_strat.get('alpha',np.nan)) else "—", "—"),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Metric","Strategy","SPY Buy & Hold"]),
                     use_container_width=True, hide_index=True)

        # Latest signal snapshot
        st.markdown("#### 🚦 Live Signal — End of Sample")
        last_row = bt.iloc[-1]
        sig_cols = []
        if use_reg: sig_cols.append(("Regime ≠ Risk-Off", int(last_row["sig_regime"])))
        if use_mom: sig_cols.append(("12-1 Momentum > 0", int(last_row["sig_momentum"])))
        if use_tr:  sig_cols.append(("Above 10m SMA",     int(last_row["sig_trend"])))
        sig_df = pd.DataFrame(sig_cols, columns=["Gate","Triggered (1=Yes)"]) if sig_cols else \
                 pd.DataFrame([("(no gates active — always long)", 1)], columns=["Gate","Triggered (1=Yes)"])
        sig_df["Triggered (1=Yes)"] = sig_df["Triggered (1=Yes)"].map(lambda v: "✅ 1" if v else "❌ 0")
        sig_df.loc[len(sig_df)] = ["Aggregate score",  f"{float(last_row['sig_score']):.2f}"]
        sig_df.loc[len(sig_df)] = ["Position next month", f"{float(last_row['target_w']):+.0f}"]
        st.dataframe(sig_df, use_container_width=True, hide_index=True)

        with st.expander("📖 Methodology & Caveats"):
            st.write("""
            * **No-lookahead**: positions decided at month T are applied to T+1 returns via `.shift(1)`.
            * **Cash leg**: out-of-market periods earn 0%. Add T-bill yield in a future iteration.
            * **Costs**: linear in turnover; 5 bps ≈ retail SPY ETF round-trip.
            * **Regime**: derived from credit-spread + realized-vol z-score (real BAA-AAA when FRED is reachable, proxy otherwise — see Macro & Rates tab).
            * **Sample bias**: regime thresholds are calibrated on the full history, so the in-sample edge is optimistic. Use `Walk-forward` Ridge model in Tab 4 for a stricter test.
            """)

        st.caption(f"Credit spread source: **{df['_credit_source'].iloc[-1] if '_credit_source' in df.columns else 'proxy'}**"
                   f" · Yield curve source: **{df['_slope_source'].iloc[-1] if '_slope_source' in df.columns else 'proxy'}**")

# ─── Tab 11: Quant Signals (volatility-aware technical recommendations) ─────
@st.cache_data(ttl=900, show_spinner="⚡ Scanning volatility + technical signals…")
def _cached_quant_scan(tickers: tuple, period: str):
    return run_quant_scan(tickers, period=period)

with tab11:
    render_methodology("quant_signals", st)
    st.markdown("### 🎯 Quant Signals — Volatility-Aware Technical Recommendations")
    st.caption(
        "Rule-based BUY/SELL/HOLD recommendations with a transparent, additive score "
        "(trend + MACD + RSI + volatility-breakout + volume). **Advisory only — this tab "
        "never places an order.** Review the reasoning, then execute manually if you agree."
    )

    if not _QS_OK:
        st.error(f"⚠️ Quant signal engine failed to load: {_import_errors.get('quant_signals')}")
    else:
        col_u, col_p = st.columns([3, 1])
        with col_u:
            qs_raw = st.text_input(
                "Watchlist (comma-separated) — defaults to the same universe as the Screener tab",
                ", ".join(QS_DEFAULT_UNIVERSE), key="qs_tickers",
            )
        with col_p:
            qs_period = st.selectbox("Lookback for indicators", ["6mo","1y","2y"], index=1, key="qs_period")
        qs_tickers = tuple(t.strip().upper() for t in qs_raw.split(",") if t.strip())[:40]

        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            run_qs = st.button("⚡ Run Live Quant Scan", type="primary", key="qs_run")
        with col_btn2:
            load_bq = st.button("☁️ Load from BigQuery Mart (<0.2s)", key="qs_bq")

        qs_df = pd.DataFrame()
        if run_qs:
            qs_df = _cached_quant_scan(qs_tickers, qs_period)
            if _BQ_OK and not qs_df.empty:
                try:
                    save_quant_signals_to_bigquery(qs_df.rename(columns={
                        "Ticker": "ticker", "Price": "price", "Signal": "signal",
                        "Score": "score", "Vol Regime": "vol_regime",
                        "Vol Breakout": "vol_breakout", "Reasons": "reasons",
                        "_error": "error_message"
                    }))
                except Exception:
                    pass
        elif load_bq:
            if _BQ_OK:
                bq_signals = load_quant_signals_from_bigquery()
                if bq_signals is not None and not bq_signals.empty:
                    qs_df = bq_signals.rename(columns={
                        "ticker": "Ticker", "price": "Price", "signal": "Signal",
                        "score": "Score", "vol_regime": "Vol Regime",
                        "vol_breakout": "Vol Breakout", "reasons": "Reasons",
                        "error_message": "_error"
                    })
                    st.toast("⚡ Loaded instant quant mart from Google BigQuery!", icon="☁️")
                else:
                    st.warning("BigQuery quant mart empty. Click '⚡ Run Live Quant Scan' first.")
            else:
                st.info("BigQuery service not available in this environment.")


        if not qs_df.empty:
            ok = qs_df[qs_df["_error"].isna()]
            bad = qs_df[qs_df["_error"].notna()]

            # KPI row — signal distribution
            counts = ok["Signal"].value_counts()
            cs = st.columns(5)
            cs[0].metric("🟢 STRONG BUY", int(counts.get("STRONG_BUY", 0)))
            cs[1].metric("🟢 BUY",        int(counts.get("BUY", 0)))
            cs[2].metric("⚪ HOLD",       int(counts.get("HOLD", 0)))
            cs[3].metric("🔴 SELL",       int(counts.get("SELL", 0)))
            cs[4].metric("🔴 STRONG SELL",int(counts.get("STRONG_SELL", 0)))

            show = ok.drop(columns=["_error"]).copy()
            show["Price"] = show["Price"].map(lambda v: f"${v:,.2f}" if pd.notna(v) else "—")
            sig_colors = {"STRONG_BUY":"#34d399","BUY":"#6ee7b7","HOLD":"#94a3b8",
                          "SELL":"#fca5a5","STRONG_SELL":"#f87171"}
            def _color_signal(val):
                return f"color:{sig_colors.get(val,'')}; font-weight:700"
            styled = show.style.map(_color_signal, subset=["Signal"])
            st.dataframe(styled, use_container_width=True, hide_index=True)

            csv = ok.to_csv(index=False).encode()
            st.download_button("⬇️ Export CSV", csv, "quant_signals.csv", "text/csv", key="qs_csv")

            if not bad.empty:
                with st.expander(f"⚠️ Skipped {len(bad)} ticker(s)"):
                    st.dataframe(bad[["Ticker","_error"]], use_container_width=True, hide_index=True)

            # ── Per-ticker reasoning + chart deep-dive ──
            st.markdown("#### 🔍 Reasoning Detail")
            pick = st.selectbox("Ticker", ok["Ticker"].tolist(), key="qs_pick") if not ok.empty else None
            if pick:
                row = ok[ok["Ticker"] == pick].iloc[0]
                badge_color = sig_colors.get(row["Signal"], "#38bdf8")
                st.markdown(f"""<div style="background:rgba(15,23,42,.6); border-left:4px solid {badge_color};
                    border-radius:8px; padding:16px 20px; margin:10px 0;">
                    <b style="color:{badge_color}; font-size:1.1rem;">{row['Ticker']} — {row['Signal']}</b>
                    &nbsp;&nbsp;score {row['Score']:+d} &nbsp;|&nbsp; vol regime: {row['Vol Regime']}
                    &nbsp;|&nbsp; breakout: {row['Vol Breakout']}
                    <ul>{"".join(f"<li>{r}</li>" for r in str(row['Reasons']).split(" · "))}</ul>
                    </div>""", unsafe_allow_html=True)

                df_t2, _ = fetch_stock(pick, str(date.today().replace(year=date.today().year-1)), str(date.today()))
                if df_t2 is not None and not df_t2.empty:
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3], vertical_spacing=0.03)
                    fig.add_trace(go.Candlestick(x=df_t2.index, open=df_t2["Open"], high=df_t2["High"],
                                                 low=df_t2["Low"], close=df_t2["Close"], name="OHLC"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_t2.index, y=df_t2["BB_upper"], line=dict(color="#818cf8",width=1,dash="dot"),
                                             name="BB Upper"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_t2.index, y=df_t2["BB_lower"], line=dict(color="#818cf8",width=1,dash="dot"),
                                             name="BB Lower", fill="tonexty", fillcolor="rgba(129,140,248,.05)"), row=1, col=1)
                    fig.add_trace(go.Bar(x=df_t2.index, y=df_t2["Volume"], marker_color=df_t2["vol_color"].tolist(),
                                         name="Volume", showlegend=False), row=2, col=1)
                    fig.update_layout(title=f"{pick} — Price vs Bollinger Bands + Volume",
                                      height=500, xaxis_rangeslider_visible=False, hovermode="x unified", **PT)
                    update_axes(fig, "Date", "")
                    st.plotly_chart(fig, use_container_width=True)
        elif run_qs:
            st.warning("No results — check tickers or try again.")

        with st.expander("ℹ️ How the Quant Signal score works (and what it isn't)"):
            st.markdown("""
**Inputs.** Daily bars from yfinance. Five components are scored and summed:

| Component | Range | Fires on |
|---|---|---|
| Trend alignment | ±25 | Price vs SMA20/50/200 stack order |
| MACD momentum | ±15 | Histogram expanding in the trend's direction |
| RSI mean-reversion | ±15 | RSI < 30 (oversold) or > 70 (overbought) |
| **Volatility breakout** | ±20 | Bollinger-Band squeeze (width in its own bottom 20th percentile ~2 weeks ago) followed by a close outside the band — this is the "catch the volatility" signal |
| Volume confirmation | ±10 | Volume > 1.5× its 20-day average, same direction as the day's move |

An **Extreme** volatility regime (today's realized vol in the top 15% of its own trailing year) with *no* directional breakout halves the score — high chop without a clear breakout gets less conviction, not more.

**Score → label**: ≥ 40 STRONG_BUY · ≥ 15 BUY · −14…14 HOLD · ≤ −15 SELL · ≤ −40 STRONG_SELL.

**Caveats.**
* Rule-based and deterministic — no news, earnings, fundamentals, or macro regime input. Cross-check the *Macro & Rates* and *Regime* tabs.
* Volatility regime and breakout detection need ~1 year of history; short lookbacks degrade quality.
* **Not investment advice, and this tool does not place orders.** It only proposes a direction and shows its reasoning — you decide, size, and execute.
            """)
