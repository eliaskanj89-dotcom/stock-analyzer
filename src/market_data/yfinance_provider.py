from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from .base import MarketDataProvider, Quote


class YFinanceProvider(MarketDataProvider):
    name = "yfinance"

    def quote(self, symbol: str) -> Quote:
        ticker = yf.Ticker(symbol)
        fast = ticker.fast_info
        price = fast.get("last_price")
        if price is None:
            hist = ticker.history(period="1d", interval="1m", auto_adjust=False)
            if hist.empty:
                raise ValueError(f"No quote available for {symbol}")
            price = float(hist["Close"].dropna().iloc[-1])
        currency = fast.get("currency")
        return Quote(
            symbol=symbol.upper(),
            price=float(price),
            timestamp=datetime.now(timezone.utc),
            currency=str(currency) if currency else None,
            source=self.name,
        )

    def history(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        frame = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if frame.empty:
            raise ValueError(f"No historical data available for {symbol}")
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        rename = {c: c.lower().replace(" ", "_") for c in frame.columns}
        frame = frame.rename(columns=rename)
        required = ["open", "high", "low", "close", "volume"]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing OHLCV columns for {symbol}: {missing}")
        return frame[required].dropna(subset=["open", "high", "low", "close"]).copy()
