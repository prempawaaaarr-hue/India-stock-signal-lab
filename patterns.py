"""
Candlestick, trend, support/resistance and chart-pattern analysis.
Pure pandas/numpy so it can be tested without Streamlit.

Input DataFrames need: Open, High, Low, Close, Volume plus the columns added by
add_indicators() in app.py: EMA20, EMA50, RSI, ATR, BUY, EXIT, VolRatio.

Pattern definitions are rule-based approximations of the classic textbook shapes.
The weights below are judgement calls, not fitted values. Use pattern_stats() to see
how each pattern actually behaved on a given stock.
"""
import numpy as np
import pandas as pd

# name: (bias, candles, weight, description)
PATTERN_INFO = {
    "Hammer": (1, 1, 1.0, "Small body near the top with a long lower shadow after a fall: sellers pushed down but buyers rejected the low."),
    "Inverted Hammer": (1, 1, 0.75, "Long upper shadow after a fall. Needs a strong green candle next to confirm."),
    "Bullish Marubozu": (1, 1, 0.75, "Large green candle with almost no shadows: buyers controlled the whole session."),
    "Bullish Engulfing": (1, 2, 1.5, "A green candle that fully covers the previous red body after a fall."),
    "Piercing Line": (1, 2, 1.25, "Green candle opens lower, then closes above the midpoint of the previous red candle."),
    "Bullish Harami": (1, 2, 0.75, "Small green body inside a large red body after a fall: selling momentum is fading."),
    "Tweezer Bottom": (1, 2, 1.0, "Two candles with matching lows after a fall: support held twice."),
    "Morning Star": (1, 3, 2.0, "Big red candle, small indecision candle, then a strong green candle: a classic bottom reversal."),
    "Three White Soldiers": (1, 3, 1.75, "Three strong green candles in a row, each closing higher."),
    "Hanging Man": (-1, 1, 0.75, "Hammer shape after a rise. Warns of tiring buyers; needs a red candle to confirm."),
    "Shooting Star": (-1, 1, 1.0, "Small body near the bottom with a long upper shadow after a rise: buyers were rejected at the high."),
    "Bearish Marubozu": (-1, 1, 0.75, "Large red candle with almost no shadows: sellers controlled the whole session."),
    "Bearish Engulfing": (-1, 2, 1.5, "A red candle that fully covers the previous green body after a rise."),
    "Dark Cloud Cover": (-1, 2, 1.25, "Red candle opens higher, then closes below the midpoint of the previous green candle."),
    "Bearish Harami": (-1, 2, 0.75, "Small red body inside a large green body after a rise: buying momentum is fading."),
    "Tweezer Top": (-1, 2, 1.0, "Two candles with matching highs after a rise: resistance held twice."),
    "Evening Star": (-1, 3, 2.0, "Big green candle, small indecision candle, then a strong red candle: a classic top reversal."),
    "Three Black Crows": (-1, 3, 1.75, "Three strong red candles in a row, each closing lower."),
    "Doji": (0, 1, 0.0, "Open and close are almost equal: indecision. Matters mainly at support/resistance."),
}


# ---------------------------------------------------------------- candlesticks


def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean DataFrame, one column per pattern, True on the candle where the pattern completes."""
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    atr = df["ATR"]

    body = (c - o).abs()
    rng = h - l
    top = np.maximum(o, c)
    bottom = np.minimum(o, c)
    upper = h - top
    lower = bottom - l
    avg_body = body.rolling(10, min_periods=3).mean()
    has_rng = rng > 0
    bull = c > o
    bear = c < o

    # Prior move measured in ATR units so it works on any price level or timeframe.
    move5 = c.shift(1) - c.shift(6)
    down = (move5 < -atr.shift(1)).fillna(False)
    up = (move5 > atr.shift(1)).fillna(False)

    def sh(s, k):
        return s.shift(k, fill_value=False)

    o1, c1, h1, l1 = o.shift(1), c.shift(1), h.shift(1), l.shift(1)
    body1, avg1 = body.shift(1), avg_body.shift(1)
    o2, c2 = o.shift(2), c.shift(2)
    body2, avg2 = body.shift(2), avg_body.shift(2)
    bull1, bear1 = sh(bull, 1), sh(bear, 1)
    bull2, bear2 = sh(bull, 2), sh(bear, 2)
    down1, up1 = sh(down, 1), sh(up, 1)
    down2, up2 = sh(down, 2), sh(up, 2)
    top1, bottom1 = top.shift(1), bottom.shift(1)

    p = pd.DataFrame(index=df.index)

    # --- single candle
    hammer_shape = has_rng & (body > 0) & (lower >= 2 * body) & (lower >= 0.6 * rng) & (upper <= 0.15 * rng)
    inv_shape = has_rng & (body > 0) & (upper >= 2 * body) & (upper >= 0.6 * rng) & (lower <= 0.15 * rng)
    big = body > 1.2 * avg_body
    p["Hammer"] = hammer_shape & down
    p["Inverted Hammer"] = inv_shape & down
    p["Hanging Man"] = hammer_shape & up
    p["Shooting Star"] = inv_shape & up
    p["Bullish Marubozu"] = bull & has_rng & (body >= 0.9 * rng) & big
    p["Bearish Marubozu"] = bear & has_rng & (body >= 0.9 * rng) & big
    p["Doji"] = has_rng & (body <= 0.1 * rng)

    # --- two candles
    p["Bullish Engulfing"] = bear1 & bull & (c >= o1) & (o <= c1) & (body > body1) & down1
    p["Bearish Engulfing"] = bull1 & bear & (o >= c1) & (c <= o1) & (body > body1) & up1
    p["Bullish Harami"] = (
        bear1 & (body1 > avg1) & bull & (top <= o1) & (bottom >= c1) & (body < body1) & down1
    )
    p["Bearish Harami"] = (
        bull1 & (body1 > avg1) & bear & (top <= c1) & (bottom >= o1) & (body < body1) & up1
    )
    p["Piercing Line"] = bear1 & bull & (o < c1) & (c > (o1 + c1) / 2) & (c < o1) & down1
    p["Dark Cloud Cover"] = bull1 & bear & (o > c1) & (c < (o1 + c1) / 2) & (c > o1) & up1
    tol = 0.1 * atr
    p["Tweezer Bottom"] = bear1 & bull & ((l - l1).abs() <= tol) & down1
    p["Tweezer Top"] = bull1 & bear & ((h - h1).abs() <= tol) & up1

    # --- three candles
    p["Morning Star"] = (
        bear2 & (body2 > avg2) & (body1 <= 0.3 * body2) & (top1 <= c2 + tol)
        & bull & (c > (o2 + c2) / 2) & down2
    )
    p["Evening Star"] = (
        bull2 & (body2 > avg2) & (body1 <= 0.3 * body2) & (bottom1 >= c2 - tol)
        & bear & (c < (o2 + c2) / 2) & up2
    )
    upper1, upper2 = upper.shift(1), upper.shift(2)
    lower1, lower2 = lower.shift(1), lower.shift(2)
    p["Three White Soldiers"] = (
        bull & bull1 & bull2 & (c > c1) & (c1 > c2) & (o > o1) & (o1 > o2) & (o < c1) & (o1 < c2)
        & (body > 0.6 * avg_body) & (body1 > 0.6 * avg1) & (body2 > 0.6 * avg2)
        & (upper <= 0.3 * body) & (upper1 <= 0.3 * body1) & (upper2 <= 0.3 * body2)
    )
    p["Three Black Crows"] = (
        bear & bear1 & bear2 & (c < c1) & (c1 < c2) & (o < o1) & (o1 < o2) & (o > c1) & (o1 > c2)
        & (body > 0.6 * avg_body) & (body1 > 0.6 * avg1) & (body2 > 0.6 * avg2)
        & (lower <= 0.3 * body) & (lower1 <= 0.3 * body1) & (lower2 <= 0.3 * body2)
    )

    return p[list(PATTERN_INFO)].fillna(False).astype(bool)


def recent_patterns(pats: pd.DataFrame, lookback: int = 3) -> list:
    out = []
    n = len(pats)
    for k in range(min(lookback, n)):
        row = pats.iloc[-1 - k]
        for name in pats.columns:
            if bool(row[name]):
                bias, candles, weight, desc = PATTERN_INFO[name]
                out.append({
                    "name": name, "bias": bias, "weight": weight, "candles": candles,
                    "bars_ago": k, "description": desc,
                })
    return out


def bar_labels(pats: pd.DataFrame):
    """Per-bar text label and net bias, for chart markers."""
    bias = pd.Series(0, index=pats.index, dtype=float)
    for name in pats.columns:
        bias = bias + pats[name].astype(float) * PATTERN_INFO[name][0]
    arr = pats.to_numpy()
    cols = np.array(pats.columns)
    labels = pd.Series([", ".join(cols[row]) for row in arr], index=pats.index)
    return labels, bias


# ---------------------------------------------------------------- trend


def adx_series(df: pd.DataFrame, n: int = 14):
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denom
    adx = dx.ewm(alpha=1 / n, adjust=False).mean()
    return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)


def trend_summary(d: pd.DataFrame) -> dict:
    c = d["Close"]
    ema20, ema50 = d["EMA20"], d["EMA50"]
    adx, pdi, mdi = adx_series(d)
    votes = [
        ("Close above EMA 50" if c.iloc[-1] > ema50.iloc[-1] else "Close below EMA 50", 1 if c.iloc[-1] > ema50.iloc[-1] else -1),
        ("EMA 20 above EMA 50" if ema20.iloc[-1] > ema50.iloc[-1] else "EMA 20 below EMA 50", 1 if ema20.iloc[-1] > ema50.iloc[-1] else -1),
        ("+DI above -DI" if pdi.iloc[-1] > mdi.iloc[-1] else "+DI below -DI", 1 if pdi.iloc[-1] > mdi.iloc[-1] else -1),
    ]
    if len(d) > 6:
        rising = ema50.iloc[-1] > ema50.iloc[-6]
        votes.append(("EMA 50 rising" if rising else "EMA 50 falling", 1 if rising else -1))
    if len(d) >= 200:
        ema200 = c.ewm(span=200, adjust=False).mean()
        above = ema50.iloc[-1] > ema200.iloc[-1]
        votes.append(("EMA 50 above EMA 200" if above else "EMA 50 below EMA 200", 1 if above else -1))

    frac = sum(v for _, v in votes) / len(votes)
    label = "Uptrend" if frac >= 0.5 else ("Downtrend" if frac <= -0.5 else "Sideways / mixed")
    a = float(adx.iloc[-1])
    strength = "strong" if a >= 25 else ("developing" if a >= 20 else "weak / ranging")
    return {
        "label": label, "adx": a, "strength": strength,
        "plus_di": float(pdi.iloc[-1]), "minus_di": float(mdi.iloc[-1]),
        "votes": votes, "score_frac": frac,
    }


# ---------------------------------------------------------------- support / resistance


def swing_points(df: pd.DataFrame, k: int = 5):
    """Indices of swing highs and lows (a bar that is the extreme of the k bars each side)."""
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    n = len(df)
    hi_idx, lo_idx = [], []
    for i in range(k, n - k):
        if h[i] == h[i - k:i + k + 1].max():
            if hi_idx and i - hi_idx[-1] <= k:
                if h[i] > h[hi_idx[-1]]:
                    hi_idx[-1] = i
            else:
                hi_idx.append(i)
        if l[i] == l[i - k:i + k + 1].min():
            if lo_idx and i - lo_idx[-1] <= k:
                if l[i] < l[lo_idx[-1]]:
                    lo_idx[-1] = i
            else:
                lo_idx.append(i)
    return hi_idx, lo_idx


def support_resistance(d: pd.DataFrame, atr: float, k: int = 5, lookback: int = 250) -> dict:
    t = d.tail(lookback)
    hi, lo = swing_points(t, k)
    pts = [float(t["High"].iloc[i]) for i in hi] + [float(t["Low"].iloc[i]) for i in lo]
    pts.sort()
    tol = max(0.5 * atr, 1e-9)
    clusters = []
    for p in pts:
        if clusters and p - np.mean(clusters[-1]) <= tol:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    levels = [{"price": float(np.mean(cl)), "touches": len(cl)} for cl in clusters]
    price = float(d["Close"].iloc[-1])

    def pick(cands, nearest_fn):
        strong = [x for x in cands if x["touches"] >= 2]
        pool = strong if strong else cands
        return nearest_fn(pool, key=lambda x: x["price"]) if pool else None

    supports = [x for x in levels if x["price"] < price]
    resists = [x for x in levels if x["price"] > price]
    return {
        "levels": levels,
        "support": pick(supports, max),
        "resistance": pick(resists, min),
    }


# ---------------------------------------------------------------- chart patterns


def detect_chart_patterns(d: pd.DataFrame, atr: float, k: int = 5) -> list:
    out = []
    n = len(d)
    price = float(d["Close"].iloc[-1])
    vr = d["VolRatio"].iloc[-1] if "VolRatio" in d.columns else np.nan

    if n > 21:
        prev_high = float(d["High"].iloc[-21:-1].max())
        prev_low = float(d["Low"].iloc[-21:-1].min())
        vol_ok = bool(pd.notna(vr) and vr >= 1.5)
        if price > prev_high:
            out.append({
                "name": "20-bar breakout", "bias": 1, "weight": 1.5 if vol_ok else 0.75,
                "detail": f"Closed above the prior 20-bar high (₹{prev_high:,.2f}) "
                          + ("on strong volume." if vol_ok else "but volume is not confirming, so a false breakout is more likely."),
            })
        elif price < prev_low:
            out.append({
                "name": "20-bar breakdown", "bias": -1, "weight": 1.5 if vol_ok else 0.75,
                "detail": f"Closed below the prior 20-bar low (₹{prev_low:,.2f}) "
                          + ("on strong volume." if vol_ok else "but volume is not confirming."),
            })

    t = d.tail(250)
    hi, lo = swing_points(t, k)
    highs = [(i, float(t["High"].iloc[i])) for i in hi]
    lows = [(i, float(t["Low"].iloc[i])) for i in lo]
    m = len(t)

    if len(highs) >= 2 and len(lows) >= 2:
        hh = highs[-1][1] > highs[-2][1]
        hl = lows[-1][1] > lows[-2][1]
        if hh and hl:
            out.append({"name": "Higher highs & higher lows", "bias": 1, "weight": 1.0,
                        "detail": "The last two swing highs and swing lows are both rising: a healthy uptrend structure."})
        elif (not hh) and (not hl):
            out.append({"name": "Lower highs & lower lows", "bias": -1, "weight": 1.0,
                        "detail": "The last two swing highs and swing lows are both falling: a downtrend structure."})

    if len(highs) >= 2:
        (i1, h1), (i2, h2) = highs[-2], highs[-1]
        if abs(h1 - h2) / h1 <= 0.01 and (i2 - i1) >= 10 and (m - 1 - i2) <= 40:
            neckline = float(t["Low"].iloc[i1:i2 + 1].min())
            if neckline < min(h1, h2) - 2 * atr:
                confirmed = price < neckline
                out.append({
                    "name": "Double top" + ("" if confirmed else " (forming)"),
                    "bias": -1, "weight": 1.5 if confirmed else 0.5,
                    "detail": f"Two highs near ₹{max(h1, h2):,.2f}. Neckline ₹{neckline:,.2f}: "
                              + ("price has closed below it, confirming the pattern." if confirmed
                                 else "a close below it would confirm the pattern."),
                })
    if len(lows) >= 2:
        (i1, l1), (i2, l2) = lows[-2], lows[-1]
        if abs(l1 - l2) / l1 <= 0.01 and (i2 - i1) >= 10 and (m - 1 - i2) <= 40:
            neckline = float(t["High"].iloc[i1:i2 + 1].max())
            if neckline > max(l1, l2) + 2 * atr:
                confirmed = price > neckline
                out.append({
                    "name": "Double bottom" + ("" if confirmed else " (forming)"),
                    "bias": 1, "weight": 1.5 if confirmed else 0.5,
                    "detail": f"Two lows near ₹{min(l1, l2):,.2f}. Neckline ₹{neckline:,.2f}: "
                              + ("price has closed above it, confirming the pattern." if confirmed
                                 else "a close above it would confirm the pattern."),
                })
    return out


# ---------------------------------------------------------------- verdict


def compute_verdict(d, trend, sr, chart_patterns, recent) -> dict:
    last = d.iloc[-1]
    price = float(last["Close"])
    atr = float(last["ATR"])
    rsi = float(last["RSI"])
    reasons, warnings = [], []
    score = 0.0

    t_pts = {"Uptrend": 2.0, "Downtrend": -2.0}.get(trend["label"], 0.0)
    if t_pts:
        reasons.append((f"Trend is {trend['label'].lower()} (ADX {trend['adx']:.0f}, {trend['strength']})", t_pts))
        score += t_pts
    else:
        reasons.append(("Trend is sideways / mixed", 0.0))

    if bool(last["BUY"]) and not bool(last["EXIT"]):
        reasons.append(("EMA/RSI/MACD rules say BUY", 1.0))
        score += 1.0
    elif bool(last["EXIT"]):
        reasons.append(("EMA/RSI/MACD rules say EXIT", -1.0))
        score -= 1.0

    rec_w = {0: 1.0, 1: 0.5, 2: 0.25}
    pat_total = 0.0
    pat_lines = []
    for p in recent:
        if p["bias"] == 0:
            continue
        pts = p["bias"] * p["weight"] * rec_w.get(p["bars_ago"], 0.25)
        pat_total += pts
        when = "latest candle" if p["bars_ago"] == 0 else f"{p['bars_ago']} candle(s) ago"
        pat_lines.append((f"Candle pattern: {p['name']} ({when})", pts))
    cap = max(-3.0, min(3.0, pat_total))
    if pat_total != cap and pat_lines:
        pat_lines.append(("Candle-pattern points capped at ±3", cap - pat_total))
    reasons.extend(pat_lines)
    score += cap

    cp_total = sum(c["bias"] * c["weight"] for c in chart_patterns)
    cp_cap = max(-2.5, min(2.5, cp_total))
    for c in chart_patterns:
        reasons.append((f"Chart pattern: {c['name']}", c["bias"] * c["weight"]))
    if cp_total != cp_cap:
        reasons.append(("Chart-pattern points capped at ±2.5", cp_cap - cp_total))
    score += cp_cap

    sup, res = sr["support"], sr["resistance"]
    if res and atr > 0 and (res["price"] - price) / atr <= 0.75:
        reasons.append((f"Price is just under resistance (₹{res['price']:,.2f})", -0.75))
        score -= 0.75
    if sup and atr > 0 and (price - sup["price"]) / atr <= 0.75:
        reasons.append((f"Price is just above support (₹{sup['price']:,.2f})", 0.75))
        score += 0.75

    if rsi > 70:
        reasons.append((f"RSI {rsi:.0f} is overbought: late-entry risk", -0.5))
        score -= 0.5
        warnings.append(f"RSI is {rsi:.0f} (above 70). A BUY here means buying after a big run-up.")
    elif rsi < 30:
        reasons.append((f"RSI {rsi:.0f} is oversold: bounce potential", 0.5))
        score += 0.5
        warnings.append(f"RSI is {rsi:.0f} (below 30). Selling here means selling after a big drop.")

    if score >= 3.5:
        label = "BUY"
    elif score >= 1.5:
        label = "LEAN BUY"
    elif score <= -3.5:
        label = "SELL"
    elif score <= -1.5:
        label = "LEAN SELL"
    else:
        label = "WAIT"
    a = abs(score)
    confidence = "High" if a >= 6 else ("Medium" if a >= 4 else "Low")

    last3 = d.tail(3)
    if score > 0:
        invalidation = float(last3["Low"].min() - 0.25 * atr)
    elif score < 0:
        invalidation = float(last3["High"].max() + 0.25 * atr)
    else:
        invalidation = None

    if trend["strength"].startswith("weak") and label != "WAIT":
        warnings.append("ADX is low: the market is ranging, so trend-following signals are less reliable.")
    if label in ("BUY", "LEAN BUY") and trend["label"] == "Downtrend":
        warnings.append("This is a counter-trend buy. Reversal patterns fail more often against the main trend.")
    if label in ("SELL", "LEAN SELL") and trend["label"] == "Uptrend":
        warnings.append("This is a counter-trend sell. Reversal patterns fail more often against the main trend.")

    return {
        "score": float(score), "label": label, "confidence": confidence,
        "reasons": reasons, "warnings": warnings, "invalidation": invalidation,
    }


def analyze(d: pd.DataFrame) -> dict:
    pats = detect_candlestick_patterns(d)
    trend = trend_summary(d)
    atr = float(d["ATR"].iloc[-1])
    sr = support_resistance(d, atr)
    cps = detect_chart_patterns(d, atr)
    recent = recent_patterns(pats, lookback=3)
    verdict = compute_verdict(d, trend, sr, cps, recent)
    return {"pats": pats, "trend": trend, "sr": sr, "chart_patterns": cps, "recent": recent, "verdict": verdict}


# ---------------------------------------------------------------- how did patterns actually perform?


def pattern_stats(d: pd.DataFrame, pats: pd.DataFrame, horizon: int = 5, min_n: int = 10):
    """
    Forward performance of each pattern on this stock's own history.
    Entry = next candle's open after the pattern completes; exit = close `horizon` candles later.
    Results are direction-adjusted: for bearish patterns, a fall counts as a success.
    """
    entry = d["Open"].shift(-1)
    exit_ = d["Close"].shift(-horizon)
    fwd = (exit_ / entry - 1) * 100
    valid = fwd.notna()
    base_up = float((fwd[valid] > 0).mean() * 100) if valid.any() else np.nan
    base_avg = float(fwd[valid].mean()) if valid.any() else np.nan

    rows = []
    for name in pats.columns:
        bias = PATTERN_INFO[name][0]
        sign = -1 if bias < 0 else 1
        mask = pats[name] & valid
        n = int(mask.sum())
        if n == 0:
            continue
        r = fwd[mask] * sign
        success = float((r > 0).mean() * 100)
        baseline = (100 - base_up) if bias < 0 else base_up
        edge = success - baseline
        if n < min_n:
            note = "Too few samples"
        elif edge >= 10:
            note = "Worked well here"
        elif edge <= -5:
            note = "Worked against here"
        else:
            note = "No clear edge"
        rows.append({
            "Pattern": name,
            "Direction": "Bullish" if bias > 0 else ("Bearish" if bias < 0 else "Neutral"),
            "Times seen": n,
            "Success %": round(success, 1),
            "Baseline %": round(baseline, 1),
            "Edge (pts)": round(edge, 1),
            "Avg move %": round(float(r.mean()), 2),
            "Verdict": note,
        })
    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out = df_out.sort_values("Times seen", ascending=False).reset_index(drop=True)
    return df_out, {"up_pct": base_up, "avg_move": base_avg}
