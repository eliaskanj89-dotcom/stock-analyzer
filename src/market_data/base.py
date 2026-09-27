from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass(frozen=True)
class Quote:
    symbol: str
    price: float
    timestamp: datetime
    bid: float | None = None
    ask: float | None = None
    currency: str | None = None
    source: str | None = None


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketDataProvider(ABC):
    """Stable interface so yfinance can later be replaced by BAKARA without changing the signal engine."""

    name: str

    @abstractmethod
    def quote(self, symbol: str) -> Quote:
        raise NotImplementedError

    @abstractmethod
    def history(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Return normalized OHLCV columns: open, high, low, close, volume."""
        raise NotImplementedError
