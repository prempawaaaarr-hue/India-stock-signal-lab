import io
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(page_title="India Stock Signal Lab", page_icon="📈", layout="wide")

st.title("🇮🇳 India Stock Signal Lab")
st.caption("Educational signal research + paper trading. No real orders are placed.")

st.warning(
    "Prototype only: Yahoo Finance data may be delayed, incomplete or unavailable. "
    "Signals are not financial advice. Paper-trade first; never rely on this app alone."
)

NEEDED_COLS = ["Open", "High", "Low", "Close", "Volume"]

# Approximate large-cap list. Index constituents change, so this is editable in the Scanner tab.
DEFAULT_UNIVERSE = (
    "RELIANCE, TCS, HDFCBANK, ICICIBANK, INFY, BHARTIARTL, ITC, SBIN, LT, HINDUNILVR, "
    "KOTAKBANK, AXISBANK, BAJFINANCE, MARUTI, SUNPHARMA, TITAN, ASIANPAINT, NTPC, "
    "ULTRACEMCO, ONGC, POWERGRID, M&M, TATASTEEL, ADANIENT, ADANIPORTS, COALINDIA, "
    "WIPRO, HCLTECH, TECHM, NESTLEIND, JSWSTEEL, BAJAJFINSV, GRASIM, HINDALCO, CIPLA, "
    "DRREDDY, EICHERMOT, BRITANNIA, APOLLOHOSP, TATACONSUMER, HEROMOTOCO, SBILIFE, "
    "HDFCLIFE, INDUSINDBK, BPCL, TRENT, BAJAJ-AUTO, SHRIRAMFIN"
)

# ---------------------------- Helpers ----------------------------


def to_yahoo_symbol(raw: str) -> str:
    raw = raw.strip().upper()
    if not raw:
        return raw
    if raw.startswith("^") or "." in raw:
        return raw
    return raw + ".NS"


def parse_symbols(text: str) -> list:
    parts = text.replace("\n", ",").split(",")
    seen, out = set(), []
    for p in parts:
        s = to_yahoo_symbol(p)
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ---------------------------- Indicators ----------------------------


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    vol_avg = df["Volume"].rolling(20).mean().replace(0, np.nan)
    df["VolRatio"] = df["Volume"] / vol_avg

    df["bull_points"] = (
        (df["EMA20"] > df["EMA50"]).astype(int)
        + (close > df["EMA20"]).astype(int)
        + ((df["RSI"] >= 50) & (df["RSI"] <= 70)).astype(int)
        + (df["MACD"] > df["MACD_signal"]).astype(int)
    )
    df["bear_points"] = (
        (df["EMA20"] < df["EMA50"]).astype(int)
        + (close < df["EMA20"]).astype(int)
        + (df["RSI"] < 45).astype(int)
        + (df["MACD"] < df["MACD_signal"]).astype(int)
    )

    df["BUY"] = df["bull_points"] >= 3
    df["EXIT"] = df["bear_points"] >= 2
    return df.dropna(subset=["EMA50", "ATR"])


def classify(row) -> tuple:
    buy, exit_ = bool(row["BUY"]), bool(row["EXIT"])
    if buy and not exit_:
        return "BUY", f"{int(row['bull_points'])}/4 bullish checks are passing."
    if exit_:
        return (
            "EXIT / SELL",
            f"{int(row['bear_points'])}/4 bearish checks are passing. This prototype is long-only.",
        )
    return "HOLD", "The rules do not show a sufficiently aligned entry or exit."


SIGNAL_LABEL = {"BUY": "🟢 BUY", "HOLD": "⚪ HOLD", "EXIT / SELL": "🔴 EXIT / SELL"}


def clean_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if not all(col in data.columns for col in NEEDED_COLS):
        return pd.DataFrame()
    data = data[NEEDED_COLS].copy()
    for col in NEEDED_COLS:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data.dropna()


@st.cache_data(ttl=300, show_spinner=False)
def get_history(symbol: str, period: str, interval: str) -> pd.DataFrame:
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval, auto_adjust=True, actions=False)
    except Exception:
        return pd.DataFrame()
    return clean_ohlcv(data)


@st.cache_data(ttl=900, show_spinner=False)
def scan_universe(symbols: tuple) -> pd.DataFrame:
    """Scan many symbols on daily candles with one batched download."""
    try:
        raw = yf.download(
            list(symbols), period="1y", interval="1d", auto_adjust=True,
            group_by="ticker", threads=True, progress=False,
        )
    except Exception:
        return pd.DataFrame()

    rows = []
    for s in symbols:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                df = raw[s]
            else:
                df = raw
            df = clean_ohlcv(df)
            if len(df) < 60:
                continue
            d = add_indicators(df)
            if len(d) < 2:
                continue
            last, prev = d.iloc[-1], d.iloc[-2]
            sig, _ = classify(last)
            rows.append({
                "Symbol": s.replace(".NS", ""),
                "Signal": SIGNAL_LABEL[sig],
                "Close (₹)": round(float(last["Close"]), 2),
                "1D %": round((float(last["Close"]) / float(prev["Close"]) - 1) * 100, 2),
                "RSI": round(float(last["RSI"]), 1),
                "Bull pts": int(last["bull_points"]),
                "Bear pts": int(last["bear_points"]),
                "Vs EMA50 %": round((float(last["Close"]) / float(last["EMA50"]) - 1) * 100, 2),
                "Volume x avg": round(float(last["VolRatio"]), 2) if pd.notna(last["VolRatio"]) else np.nan,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


# ---------------------------- Backtest ----------------------------


def _trade_record(entry_time, exit_time, entry_px, exit_px, reason, fee, fmt):
    net_ret = (exit_px * (1 - fee)) / (entry_px * (1 + fee)) - 1
    return {
        "Entry": entry_time.strftime(fmt),
        "Exit": exit_time.strftime(fmt),
        "Entry ₹": round(entry_px, 2),
        "Exit ₹": round(exit_px, 2),
        "Return %": round(net_ret * 100, 2),
        "Reason": reason,
    }


def run_backtest(df, capital, cost_bps, use_stop, stop_mult, bars_per_year, fmt):
    """
    Long-only, fully invested when in a trade.
    Signals form at a candle close and are executed at the NEXT candle's open.
    Optional ATR stop: set from the signal candle's ATR, triggered by the candle low.
    """
    fee = cost_bps / 10000.0
    idx = df.index
    o = df["Open"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    atr = df["ATR"].to_numpy(float)
    buy = df["BUY"].to_numpy(bool)
    exit_ = df["EXIT"].to_numpy(bool)

    cash, shares = float(capital), 0.0
    entry_px, entry_time, stop_px = 0.0, None, 0.0
    pending, pending_atr = None, 0.0
    equity = np.empty(len(df))
    trades = []

    for i in range(len(df)):
        # 1) Execute the order created at the previous close
        if pending == "ENTER" and shares == 0:
            entry_px = o[i]
            shares = cash / (entry_px * (1 + fee))
            cash = 0.0
            entry_time = idx[i]
            stop_px = entry_px - stop_mult * pending_atr
        elif pending == "EXIT" and shares > 0:
            px = o[i]
            cash = shares * px * (1 - fee)
            trades.append(_trade_record(entry_time, idx[i], entry_px, px, "Signal", fee, fmt))
            shares = 0.0
        pending = None

        # 2) Stop-loss check inside this candle
        if shares > 0 and use_stop and l[i] <= stop_px:
            px = min(o[i], stop_px)  # gap-down opens fill at the open, not the stop
            cash = shares * px * (1 - fee)
            trades.append(_trade_record(entry_time, idx[i], entry_px, px, "ATR stop", fee, fmt))
            shares = 0.0

        # 3) Mark to market
        equity[i] = cash + shares * c[i]

        # 4) Decide what to do at the next open
        if shares == 0:
            if buy[i]:
                pending, pending_atr = "ENTER", atr[i]
        elif exit_[i]:
            pending = "EXIT"

    if shares > 0:
        trades.append(_trade_record(entry_time, idx[-1], entry_px, c[-1], "Still open", fee, fmt))

    equity = pd.Series(equity, index=idx, name="Strategy")
    closed = [t for t in trades if t["Reason"] != "Still open"]
    rets = np.array([t["Return %"] for t in closed], dtype=float)

    wins = rets[rets > 0].sum() if len(rets) else 0.0
    losses = -rets[rets < 0].sum() if len(rets) else 0.0
    if len(rets) == 0:
        profit_factor = np.nan
    elif losses == 0:
        profit_factor = np.inf
    else:
        profit_factor = wins / losses

    bar_rets = equity.pct_change().dropna()
    sharpe = (
        bar_rets.mean() / bar_rets.std() * math.sqrt(bars_per_year)
        if len(bar_rets) > 2 and bar_rets.std() > 0 else np.nan
    )

    buy_hold = capital * df["Close"] / df["Close"].iloc[0]
    buy_hold.name = "Buy & hold"

    stats = {
        "strategy_ret": (equity.iloc[-1] / capital - 1) * 100,
        "bh_ret": (buy_hold.iloc[-1] / capital - 1) * 100,
        "max_dd": ((equity / equity.cummax() - 1) * 100).min(),
        "bh_max_dd": ((buy_hold / buy_hold.cummax() - 1) * 100).min(),
        "sharpe": sharpe,
        "n_trades": len(closed),
        "win_rate": (rets > 0).mean() * 100 if len(rets) else np.nan,
        "avg_trade": rets.mean() if len(rets) else np.nan,
        "profit_factor": profit_factor,
    }
    return equity, buy_hold, pd.DataFrame(trades), stats


# ---------------------------- Chart ----------------------------


def build_chart(df: pd.DataFrame, bars: int, intraday: bool) -> go.Figure:
    d = df.tail(bars)
    buy_edge = d["BUY"] & ~d["BUY"].shift(1, fill_value=False)
    exit_edge = d["EXIT"] & ~d["EXIT"].shift(1, fill_value=False)

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.5, 0.14, 0.17, 0.19],
        subplot_titles=("Price, EMAs and signals", "Volume", "RSI (14)", "MACD"),
    )

    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        name="Price", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["EMA20"], name="EMA 20", line=dict(width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["EMA50"], name="EMA 50", line=dict(width=1.5)), row=1, col=1)

    if buy_edge.any():
        fig.add_trace(go.Scatter(
            x=d.index[buy_edge], y=d["Low"][buy_edge] * 0.99, mode="markers", name="BUY signal",
            marker=dict(symbol="triangle-up", size=11, color="#00c805"),
        ), row=1, col=1)
    if exit_edge.any():
        fig.add_trace(go.Scatter(
            x=d.index[exit_edge], y=d["High"][exit_edge] * 1.01, mode="markers", name="EXIT signal",
            marker=dict(symbol="triangle-down", size=11, color="#ff4d4f"),
        ), row=1, col=1)

    vol_colors = np.where(d["Close"] >= d["Open"], "#26a69a", "#ef5350")
    fig.add_trace(go.Bar(x=d.index, y=d["Volume"], marker_color=vol_colors, name="Volume", showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(x=d.index, y=d["RSI"], name="RSI", line=dict(width=1.5), showlegend=False), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", opacity=0.5, row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", opacity=0.5, row=3, col=1)

    hist_colors = np.where(d["MACD_hist"] >= 0, "#26a69a", "#ef5350")
    fig.add_trace(go.Bar(x=d.index, y=d["MACD_hist"], marker_color=hist_colors, name="Histogram", showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["MACD"], name="MACD", line=dict(width=1.3)), row=4, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["MACD_signal"], name="Signal line", line=dict(width=1.3)), row=4, col=1)

    breaks = [dict(bounds=["sat", "mon"])]
    if intraday:
        breaks.append(dict(bounds=[15.5, 9.25], pattern="hour"))  # hide hours outside NSE session
    fig.update_xaxes(rangebreaks=breaks)
    fig.update_layout(
        height=860, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


# ---------------------------- Sidebar ----------------------------

st.sidebar.header("Settings")
raw_symbol = st.sidebar.text_input("NSE ticker", value="RELIANCE").strip().upper()
symbol = to_yahoo_symbol(raw_symbol)

style = st.sidebar.radio("Trading timeframe", ["Swing / daily", "Intraday / 15-minute"])
if style == "Swing / daily":
    period, interval = "2y", "1d"
    intraday, bars_per_year, date_fmt = False, 252, "%Y-%m-%d"
else:
    period, interval = "60d", "15m"
    intraday, bars_per_year, date_fmt = True, 252 * 25, "%Y-%m-%d %H:%M"

capital = st.sidebar.number_input("Paper account capital (₹)", min_value=1000, value=100000, step=5000)
risk_pct = st.sidebar.slider("Maximum risk per trade (%)", 0.25, 2.0, 1.0, 0.25)
max_allocation_pct = st.sidebar.slider("Maximum capital allocation (%)", 5, 50, 20, 5)
cost_bps = st.sidebar.number_input(
    "Estimated cost per side (basis points)", min_value=0.0, max_value=100.0, value=10.0, step=1.0
)
if st.sidebar.button("Refresh market data"):
    get_history.clear()
    scan_universe.clear()

st.sidebar.caption("Examples: RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, SBIN, ITC")

# ---------------------------- Load and analyse ----------------------------

with st.spinner("Loading market history..."):
    history = get_history(symbol, period, interval)

if history.empty or len(history) < 60:
    st.error(
        f"Could not load enough data for {symbol}. Check the NSE symbol and try again. "
        "Yahoo Finance commonly uses the .NS suffix for NSE-listed shares."
    )
    st.stop()

data = add_indicators(history)
if len(data) < 2:
    st.error("Not enough usable history to calculate indicators.")
    st.stop()

latest = data.iloc[-1]
previous = data.iloc[-2]
price = float(latest["Close"])
atr = float(latest["ATR"])
buy_signal = bool(latest["BUY"])
signal, reason = classify(latest)

# Risk plan numbers
stop_distance = 1.5 * atr
target_distance = 3.0 * atr
risk_budget = capital * (risk_pct / 100)
max_allocation = capital * (max_allocation_pct / 100)
qty_by_risk = math.floor(risk_budget / stop_distance) if stop_distance > 0 else 0
qty_by_allocation = math.floor(max_allocation / price) if price > 0 else 0
suggested_qty = max(0, min(qty_by_risk, qty_by_allocation))

if "paper_trades" not in st.session_state:
    st.session_state.paper_trades = []

tab_overview, tab_chart, tab_backtest, tab_scan, tab_paper = st.tabs(
    ["📊 Overview", "📈 Chart", "🧪 Backtest", "🔍 Scanner", "📝 Paper trading"]
)

# ---------------------------- Overview ----------------------------

with tab_overview:
    st.subheader(f"{raw_symbol} — {SIGNAL_LABEL[signal]}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest close", f"₹{price:,.2f}", f"{price - float(previous['Close']):+.2f}")
    c2.metric("Signal", signal)
    c3.metric("RSI (14)", f"{latest['RSI']:.1f}")
    c4.metric("ATR (14)", f"₹{atr:.2f}")
    st.info(reason)

    e1, e2 = st.columns(2)
    vr = latest["VolRatio"]
    e1.metric("Volume vs 20-bar average", f"{vr:.2f}x" if pd.notna(vr) else "n/a")

    # Relative strength vs Nifty 50 over the last 60 bars
    rel_text = "n/a"
    try:
        nifty = get_history("^NSEI", period, interval)
        lb = 60
        if len(nifty) > lb and len(data) > lb:
            s_ret = data["Close"].iloc[-1] / data["Close"].iloc[-lb - 1] - 1
            n_ret = nifty["Close"].iloc[-1] / nifty["Close"].iloc[-lb - 1] - 1
            rel_text = f"{(s_ret - n_ret) * 100:+.1f} pts"
    except Exception:
        pass
    e2.metric("Vs Nifty 50 (last 60 bars)", rel_text)

    st.markdown("**Why this signal?**")
    k1, k2 = st.columns(2)
    bull_checks = [
        ("EMA 20 above EMA 50", latest["EMA20"] > latest["EMA50"]),
        ("Close above EMA 20", price > latest["EMA20"]),
        ("RSI between 50 and 70", 50 <= latest["RSI"] <= 70),
        ("MACD above signal line", latest["MACD"] > latest["MACD_signal"]),
    ]
    bear_checks = [
        ("EMA 20 below EMA 50", latest["EMA20"] < latest["EMA50"]),
        ("Close below EMA 20", price < latest["EMA20"]),
        ("RSI below 45", latest["RSI"] < 45),
        ("MACD below signal line", latest["MACD"] < latest["MACD_signal"]),
    ]
    with k1:
        st.caption(f"Bullish checks ({int(latest['bull_points'])}/4; BUY needs 3)")
        for label, ok in bull_checks:
            st.write(("✅ " if ok else "❌ ") + label)
    with k2:
        st.caption(f"Bearish checks ({int(latest['bear_points'])}/4; EXIT needs 2)")
        for label, ok in bear_checks:
            st.write(("⚠️ " if ok else "▫️ ") + label)

    st.subheader("Illustrative risk plan")
    rc = st.columns(4)
    rc[0].metric("Reference entry", f"₹{price:,.2f}")
    rc[1].metric("ATR stop (1.5x)", f"₹{price - stop_distance:,.2f}")
    rc[2].metric("2R target", f"₹{price + target_distance:,.2f}")
    rc[3].metric("Risk-sized quantity", f"{suggested_qty:,} shares")
    st.caption(
        f"Risk budget ₹{risk_budget:,.0f}; maximum allocation ₹{max_allocation:,.0f}. "
        "The levels are examples calculated from the latest candle, not guaranteed execution prices."
    )

    with st.expander("Indicator details (last 15 bars)"):
        st.dataframe(
            data[["Close", "EMA20", "EMA50", "RSI", "MACD", "MACD_signal", "ATR",
                  "bull_points", "bear_points"]].tail(15).round(2)
        )

# ---------------------------- Chart ----------------------------

with tab_chart:
    max_bars = min(len(data), 600)
    default_bars = min(180 if not intraday else 300, max_bars)
    bars = st.slider("Candles to display", min_value=min(60, max_bars), max_value=max_bars, value=default_bars, step=10)
    st.plotly_chart(build_chart(data, bars, intraday))
    st.caption("Green triangles mark the candle where a BUY condition starts; red triangles where an EXIT condition starts.")

# ---------------------------- Backtest ----------------------------

with tab_backtest:
    st.subheader("Historical rule test")
    st.caption(
        "Long-only simulation. Signals form on a candle close and are filled at the next candle's open. "
        "Estimated transaction costs are included, but taxes, slippage, market impact, corporate actions "
        "and real execution differences are not fully represented."
    )
    bc1, bc2 = st.columns(2)
    use_stop = bc1.checkbox("Apply ATR stop-loss in the test", value=True)
    stop_mult = bc2.slider("ATR stop multiple", 1.0, 4.0, 1.5, 0.5, disabled=not use_stop)

    equity, buy_hold, trades_df, stats = run_backtest(
        data, float(capital), float(cost_bps), use_stop, stop_mult, bars_per_year, date_fmt
    )

    m = st.columns(4)
    m[0].metric("Strategy return", f"{stats['strategy_ret']:.2f}%")
    m[1].metric("Buy & hold return", f"{stats['bh_ret']:.2f}%",
                f"{stats['strategy_ret'] - stats['bh_ret']:+.2f} pts vs strategy", delta_color="off")
    m[2].metric("Max drawdown", f"{stats['max_dd']:.2f}%", f"Buy & hold {stats['bh_max_dd']:.2f}%", delta_color="off")
    m[3].metric("Sharpe (rough, rf=0)", "n/a" if pd.isna(stats["sharpe"]) else f"{stats['sharpe']:.2f}")

    n = st.columns(4)
    n[0].metric("Closed trades", f"{stats['n_trades']}")
    n[1].metric("Win rate", "n/a" if pd.isna(stats["win_rate"]) else f"{stats['win_rate']:.0f}%")
    n[2].metric("Average trade", "n/a" if pd.isna(stats["avg_trade"]) else f"{stats['avg_trade']:.2f}%")
    pf = stats["profit_factor"]
    n[3].metric("Profit factor", "n/a" if pd.isna(pf) else ("∞" if np.isinf(pf) else f"{pf:.2f}"))

    if not trades_df.empty and (trades_df["Reason"] == "Still open").any():
        open_ret = float(trades_df.loc[trades_df["Reason"] == "Still open", "Return %"].iloc[-1])
        st.caption(
            f"A trade is still open ({open_ret:+.2f}% so far). It is included in the equity curve and total "
            "return, but not in the closed-trade statistics (win rate, average trade, profit factor)."
        )

    if stats["n_trades"] < 10:
        st.warning(
            f"Only {stats['n_trades']} closed trades. That is too few to say anything reliable about this rule set."
        )
    if stats["strategy_ret"] < stats["bh_ret"]:
        st.info("On this stock and period, the rules did not beat simply holding the stock.")

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=equity.index, y=equity, name="Strategy"))
    fig_eq.add_trace(go.Scatter(x=buy_hold.index, y=buy_hold, name="Buy & hold"))
    fig_eq.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])] + (
        [dict(bounds=[15.5, 9.25], pattern="hour")] if intraday else []))
    fig_eq.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Paper equity (₹)",
                         legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig_eq)

    with st.expander("Trade log"):
        if trades_df.empty:
            st.write("No trades were generated in this period.")
        else:
            st.dataframe(trades_df)

# ---------------------------- Scanner ----------------------------

with tab_scan:
    st.subheader("Signal scanner")
    st.caption(
        "Scans daily candles (1 year) with the same rules as above. The default list is an approximate "
        "set of large caps; edit it freely. Large scans can hit Yahoo Finance rate limits."
    )
    universe_text = st.text_area("Stocks to scan (comma or new-line separated)", value=DEFAULT_UNIVERSE, height=130)
    if st.button("Run scan", type="primary"):
        syms = parse_symbols(universe_text)
        if len(syms) > 100:
            st.warning("Scanning the first 100 symbols only.")
            syms = syms[:100]
        if not syms:
            st.error("Add at least one symbol.")
        else:
            with st.spinner(f"Scanning {len(syms)} stocks..."):
                st.session_state.scan_df = scan_universe(tuple(syms))
            st.session_state.scan_time = datetime.now().strftime("%d %b %Y %H:%M")

    scan_df = st.session_state.get("scan_df")
    if scan_df is None:
        st.info("Press **Run scan** to scan the list.")
    elif scan_df.empty:
        st.warning("No results. Yahoo Finance may be rate-limiting or the symbols may be invalid. Try again shortly.")
    else:
        st.caption(f"Last scan: {st.session_state.get('scan_time', '')}")
        counts = scan_df["Signal"].value_counts()
        s1, s2, s3 = st.columns(3)
        s1.metric("🟢 BUY", int(counts.get("🟢 BUY", 0)))
        s2.metric("⚪ HOLD", int(counts.get("⚪ HOLD", 0)))
        s3.metric("🔴 EXIT / SELL", int(counts.get("🔴 EXIT / SELL", 0)))

        show = st.multiselect(
            "Show signals", list(SIGNAL_LABEL.values()), default=list(SIGNAL_LABEL.values())
        )
        order = {"🟢 BUY": 0, "⚪ HOLD": 1, "🔴 EXIT / SELL": 2}
        view = scan_df[scan_df["Signal"].isin(show)].copy()
        view["_o"] = view["Signal"].map(order)
        view = view.sort_values(["_o", "Bull pts"], ascending=[True, False]).drop(columns="_o")
        st.dataframe(view, hide_index=True)
        st.download_button("Download scan as CSV", view.to_csv(index=False), "scan_results.csv", "text/csv")
        st.caption("Type a symbol from this list into the sidebar to open its chart and backtest.")

# ---------------------------- Paper trading ----------------------------


def current_price_for(sym: str):
    if sym == symbol:
        return price
    h = get_history(sym, "1mo", "1d")
    return float(h["Close"].iloc[-1]) if not h.empty else None


with tab_paper:
    st.subheader("Paper trading")
    st.caption(
        "This ledger lives in your browser session only and is lost on refresh or restart. "
        "Download it as CSV to keep it, and load it back below."
    )

    trades = st.session_state.paper_trades
    open_idx = [i for i, t in enumerate(trades) if t["status"] == "OPEN"]
    already_open = any(trades[i]["symbol"] == symbol for i in open_idx)

    if buy_signal and not already_open:
        if st.button(f"Record paper BUY: {symbol}", type="primary"):
            if suggested_qty < 1:
                st.error("Position size is zero under the selected risk and allocation limits.")
            else:
                trades.append({
                    "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "symbol": symbol,
                    "entry": price,
                    "qty": suggested_qty,
                    "stop": price - stop_distance,
                    "target": price + target_distance,
                    "status": "OPEN",
                    "exit": None,
                    "pnl": None,
                })
                st.success("Paper trade recorded. No real order was sent.")
                st.rerun()
    elif already_open:
        st.info(f"You already have an open paper position in {symbol}. Close it before recording another.")
    else:
        st.write(f"No BUY signal on {symbol} right now, so a new paper entry is not offered.")

    realised = sum(t["pnl"] for t in trades if t["status"] == "CLOSED" and t["pnl"] is not None)
    unrealised = 0.0
    for i in open_idx:
        t = trades[i]
        cp = current_price_for(t["symbol"])
        cp_used = cp if cp is not None else t["entry"]
        pnl = (cp_used - t["entry"]) * t["qty"]
        unrealised += pnl

        if cp is None:
            note = "price unavailable"
        elif cp <= t["stop"]:
            note = "⚠️ at or below stop"
        elif cp >= t["target"]:
            note = "🎯 target reached"
        else:
            note = "within stop/target"

        row_cols = st.columns([5, 1.2])
        row_cols[0].write(
            f"**OPEN** — {t['symbol']} | {t['qty']} sh | Entry ₹{t['entry']:.2f} | "
            f"Now ₹{cp_used:.2f} | Stop ₹{t['stop']:.2f} | Target ₹{t['target']:.2f} | "
            f"Unrealised ₹{pnl:,.2f} | {note}"
        )
        if row_cols[1].button("Close", key=f"close_{i}"):
            t["status"] = "CLOSED"
            t["exit"] = cp_used
            t["pnl"] = (cp_used - t["entry"]) * t["qty"]
            st.success(f"Closed {t['symbol']}. P&L: ₹{t['pnl']:,.2f}")
            st.rerun()

    closed_pnls = [t["pnl"] for t in trades if t["status"] == "CLOSED" and t["pnl"] is not None]
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Realised P&L", f"₹{realised:,.2f}")
    p2.metric("Unrealised P&L", f"₹{unrealised:,.2f}")
    p3.metric("Closed trades", f"{len(closed_pnls)}")
    p4.metric("Win rate", f"{(np.array(closed_pnls) > 0).mean() * 100:.0f}%" if closed_pnls else "n/a")

    if trades:
        ledger_df = pd.DataFrame(trades)
        st.dataframe(ledger_df)
        st.download_button("Download ledger (CSV)", ledger_df.to_csv(index=False), "paper_trades.csv", "text/csv")
    else:
        st.write("No paper trades recorded yet.")

    with st.expander("Load a saved ledger"):
        up = st.file_uploader("Upload a ledger CSV you downloaded earlier", type="csv")
        if up is not None and st.button("Load ledger"):
            try:
                loaded = pd.read_csv(io.BytesIO(up.getvalue()))
                required = {"time", "symbol", "entry", "qty", "stop", "target", "status", "exit", "pnl"}
                if not required.issubset(loaded.columns):
                    st.error("That file does not look like a ledger exported from this app.")
                else:
                    loaded = loaded.astype(object).where(loaded.notna(), None)
                    st.session_state.paper_trades = loaded.to_dict("records")
                    st.success("Ledger loaded.")
                    st.rerun()
            except Exception as exc:
                st.error(f"Could not read that file: {exc}")

st.divider()
st.caption(
    "Version 0.2 • Research prototype only • Long-only cash-equity logic • "
    "Do not connect broker execution or use real money until extensively validated."
)
