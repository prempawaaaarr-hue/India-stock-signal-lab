
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="India Stock Signal Lab", page_icon="📈", layout="wide")

st.title("🇮🇳 India Stock Signal Lab")
st.caption("Educational signal research + paper trading. No real orders are placed.")

st.warning(
    "Prototype only: Yahoo Finance data may be delayed, incomplete or unavailable. "
    "Signals are not financial advice. Paper-trade first; never rely on this app alone."
)

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
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()

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


@st.cache_data(ttl=300, show_spinner=False)
def get_history(symbol: str, period: str, interval: str) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period, interval=interval, auto_adjust=True, actions=False)
    if data is None or data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in data.columns for col in needed):
        return pd.DataFrame()
    data = data[needed].copy()
    for col in needed:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data.dropna()


def make_backtest(df: pd.DataFrame, starting_cash: float, cost_bps: float):
    """Long-only signal backtest. Signals formed at close; position begins next bar."""
    raw_position = []
    holding = 0
    for _, row in df.iterrows():
        if holding == 0 and bool(row["BUY"]):
            holding = 1
        elif holding == 1 and bool(row["EXIT"]):
            holding = 0
        raw_position.append(holding)

    position = pd.Series(raw_position, index=df.index).shift(1).fillna(0)
    asset_return = df["Close"].pct_change().fillna(0)
    turnover = position.diff().abs().fillna(position.abs())
    fee_rate = cost_bps / 10000.0
    strategy_return = position * asset_return - turnover * fee_rate
    equity = starting_cash * (1 + strategy_return).cumprod()

    total_return = (equity.iloc[-1] / starting_cash - 1) * 100
    running_peak = equity.cummax()
    drawdown = (equity / running_peak - 1) * 100
    max_drawdown = drawdown.min()
    entries = int(((position == 1) & (position.shift(1).fillna(0) == 0)).sum())
    exits = int(((position == 0) & (position.shift(1).fillna(0) == 1)).sum())

    return equity, {
        "Return %": total_return,
        "Max drawdown %": max_drawdown,
        "Entries": entries,
        "Completed exits": exits,
    }


# ---------------------------- Sidebar ----------------------------

st.sidebar.header("Settings")
raw_symbol = st.sidebar.text_input("NSE ticker", value="RELIANCE").strip().upper()
if raw_symbol.startswith("^") or "." in raw_symbol:
    symbol = raw_symbol
else:
    symbol = raw_symbol + ".NS"

style = st.sidebar.radio("Trading timeframe", ["Swing / daily", "Intraday / 15-minute"])
if style == "Swing / daily":
    period, interval = "2y", "1d"
else:
    period, interval = "60d", "15m"

capital = st.sidebar.number_input("Paper account capital (₹)", min_value=1000, value=100000, step=5000)
risk_pct = st.sidebar.slider("Maximum risk per trade (%)", 0.25, 2.0, 1.0, 0.25)
max_allocation_pct = st.sidebar.slider("Maximum capital allocation (%)", 5, 50, 20, 5)
cost_bps = st.sidebar.number_input("Estimated cost per side (basis points)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
refresh = st.sidebar.button("Refresh market data")

if refresh:
    get_history.clear()

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
if data.empty:
    st.error("Not enough usable history to calculate indicators.")
    st.stop()

latest = data.iloc[-1]
previous = data.iloc[-2]
price = float(latest["Close"])
atr = float(latest["ATR"])
buy_signal = bool(latest["BUY"])
exit_signal = bool(latest["EXIT"])

if buy_signal and not exit_signal:
    signal = "BUY"
    reason = f"{int(latest['bull_points'])}/4 bullish checks are passing."
elif exit_signal:
    signal = "EXIT / SELL"
    reason = f"{int(latest['bear_points'])}/4 bearish checks are passing. This prototype is long-only."
else:
    signal = "HOLD"
    reason = "The rules do not show a sufficiently aligned entry or exit."

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest close", f"₹{price:,.2f}", f"{price-float(previous['Close']):+.2f}")
col2.metric("Signal", signal)
col3.metric("RSI (14)", f"{latest['RSI']:.1f}")
col4.metric("ATR (14)", f"₹{atr:.2f}")

st.info(reason)

# Position sizing: risk budget / stop distance, capped by allocation.
stop_distance = 1.5 * atr
target_distance = 3.0 * atr
risk_budget = capital * (risk_pct / 100)
max_allocation = capital * (max_allocation_pct / 100)
qty_by_risk = math.floor(risk_budget / stop_distance) if stop_distance > 0 else 0
qty_by_allocation = math.floor(max_allocation / price) if price > 0 else 0
suggested_qty = max(0, min(qty_by_risk, qty_by_allocation))

st.subheader("Illustrative risk plan")
risk_cols = st.columns(4)
risk_cols[0].metric("Reference entry", f"₹{price:,.2f}")
risk_cols[1].metric("ATR stop", f"₹{price-stop_distance:,.2f}")
risk_cols[2].metric("2R target", f"₹{price+target_distance:,.2f}")
risk_cols[3].metric("Risk-sized quantity", f"{suggested_qty:,} shares")
st.caption(
    f"Risk budget ₹{risk_budget:,.0f}; maximum allocation ₹{max_allocation:,.0f}. "
    "The levels are examples calculated from the latest candle, not guaranteed execution prices."
)

# ---------------------------- Chart and diagnostics ----------------------------

st.subheader("Price and trend")
chart_df = data[["Close", "EMA20", "EMA50"]].tail(250)
st.line_chart(chart_df)

with st.expander("Indicator details"):
    st.dataframe(
        data[["Close", "EMA20", "EMA50", "RSI", "MACD", "MACD_signal", "ATR",
              "bull_points", "bear_points"]].tail(15).round(2),
        use_container_width=True
    )

# ---------------------------- Backtest ----------------------------

st.subheader("Historical rule test")
st.caption(
    "Long-only simulation. The signal is calculated on a candle close and takes effect "
    "on the next candle. Estimated transaction costs are included, but taxes, slippage, "
    "market impact, corporate actions and real execution differences may not be fully represented."
)
equity, stats = make_backtest(data, float(capital), float(cost_bps))
m1, m2, m3, m4 = st.columns(4)
m1.metric("Simulated return", f"{stats['Return %']:.2f}%")
m2.metric("Maximum drawdown", f"{stats['Max drawdown %']:.2f}%")
m3.metric("Entries", f"{stats['Entries']}")
m4.metric("Completed exits", f"{stats['Completed exits']}")
st.line_chart(equity.rename("Paper equity (₹)"))

# ---------------------------- Paper trading ledger ----------------------------

if "paper_trades" not in st.session_state:
    st.session_state.paper_trades = []

st.subheader("Paper trading")
st.caption("This ledger is held in this browser session only. It does not place broker orders or persist across all devices.")

open_trades = [t for t in st.session_state.paper_trades if t["status"] == "OPEN"]
already_open = any(t["symbol"] == symbol for t in open_trades)

if buy_signal and not already_open:
    if st.button(f"Record paper BUY: {symbol}", type="primary"):
        if suggested_qty < 1:
            st.error("Position size is zero under the selected risk and allocation limits.")
        else:
            st.session_state.paper_trades.append({
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

for i, trade in enumerate(st.session_state.paper_trades):
    if trade["status"] == "OPEN":
        current_data = data if trade["symbol"] == symbol else None
        current_price = price if current_data is not None else trade["entry"]
        unrealised = (current_price - trade["entry"]) * trade["qty"]
        st.write(
            f"OPEN — {trade['symbol']} | {trade['qty']} shares | "
            f"Entry ₹{trade['entry']:.2f} | Current ₹{current_price:.2f} | "
            f"Unrealised P&L ₹{unrealised:.2f}"
        )
        if trade["symbol"] == symbol and st.button(f"Close paper position #{i+1}", key=f"close_{i}"):
            trade["status"] = "CLOSED"
            trade["exit"] = price
            trade["pnl"] = (price - trade["entry"]) * trade["qty"]
            st.success(f"Paper position closed. P&L: ₹{trade['pnl']:.2f}")
            st.rerun()

if st.session_state.paper_trades:
    st.dataframe(pd.DataFrame(st.session_state.paper_trades), use_container_width=True)
else:
    st.write("No paper trades recorded yet.")

st.divider()
st.caption(
    "Version 0.1 • Research prototype only • Long-only cash-equity logic • "
    "Do not connect broker execution or use real money until extensively validated."
)
