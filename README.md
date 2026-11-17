# India Stock Signal Lab (prototype)

A beginner-friendly Streamlit app for Indian NSE stock research and paper trading.

## Included
- NSE Yahoo Finance symbols (example: RELIANCE becomes RELIANCE.NS)
- Daily candles for swing research and 15-minute candles for intraday experiments
- EMA 20/50, RSI 14, MACD and ATR
- Rule-based BUY / HOLD / EXIT signal
- ATR-based illustrative stop/target and risk-based share sizing
- Long-only historical simulation with estimated transaction costs
- Session-only paper-trade ledger; no broker connection or live orders

## Run locally
1. Install Python 3.10 or later.
2. Open a terminal in this folder.
3. Run: `python -m venv .venv`
4. Activate it:
   - Windows: `.venv\\Scripts\\activate`
   - macOS/Linux: `source .venv/bin/activate`
5. Run: `pip install -r requirements.txt`
6. Run: `streamlit run app.py`
7. Open the local URL shown in the terminal.

## Deploy free (subject to platform limits)
- Push `app.py` and `requirements.txt` to a GitHub repository.
- Sign in at https://share.streamlit.io/ using GitHub.
- Create an app and select `app.py` as the entrypoint.

## Important limitations
- Yahoo Finance data is an unofficial data source, can be delayed, incomplete, rate-limited, or unavailable. Do not treat it as an exchange-authorized real-time feed.
- This is not investment advice and not a validated profitable strategy.
- Intraday history availability is restricted by the data provider.
- The backtest is simplified and cannot perfectly model fills, taxes, brokerage, slippage, liquidity, circuit limits, corporate actions or survivorship bias.
- Paper trades are stored only in Streamlit's current session and may disappear on refresh/restart.
- This prototype does not connect to a broker and cannot place orders.
- Do not redistribute market data or sell signals without checking data licences and applicable Indian regulatory requirements.
