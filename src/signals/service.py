from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.market_data import MarketDataProvider
from src.signals.models import (
    PositionPlan,
    ScoreBreakdown,
    SignalAction,
    TechnicalSnapshot,
    TradeSignal,
)


class SignalService:
    """Deterministic V1 signal generator. AI explanations can be layered on later."""

    def __init__(self, provider: MarketDataProvider):
        self.provider = provider

    def generate(
        self,
        symbol: str,
        timeframe: str = "swing",
        portfolio_value: float | None = None,
        max_risk_pct: float = 1.0,
    ) -> TradeSignal:
        frame = self.provider.history(symbol, period="1y", interval="1d")
        quote = self.provider.quote(symbol)
        indicators = self._indicators(frame)
        score = self._score(frame, indicators)

        action = self._action(score.total)
        atr = indicators["atr_14"]
        support = indicators["support"]
        resistance = indicators["resistance"]
        price = quote.price

        if action in {SignalAction.strong_long, SignalAction.long, SignalAction.watch}:
            ideal_entry = min(price, indicators["ema_20"] or price)
            entry_low = max(0.01, ideal_entry - 0.35 * atr)
            entry_high = ideal_entry + 0.35 * atr
            stop = min(support, ideal_entry - 1.5 * atr) if support else ideal_entry - 1.5 * atr
            risk = max(ideal_entry - stop, 0.01)
            tp1 = max(resistance, ideal_entry + 1.0 * risk) if resistance else ideal_entry + risk
            tp2 = ideal_entry + 2.0 * risk
            tp3 = ideal_entry + 3.0 * risk
            rr = (tp2 - ideal_entry) / risk
            invalidation = f"Daily close below {stop:.2f} invalidates the bullish setup."
        elif action in {SignalAction.strong_short, SignalAction.short}:
            ideal_entry = max(price, indicators["ema_20"] or price)
            entry_low = max(0.01, ideal_entry - 0.35 * atr)
            entry_high = ideal_entry + 0.35 * atr
            stop = max(resistance, ideal_entry + 1.5 * atr) if resistance else ideal_entry + 1.5 * atr
            risk = max(stop - ideal_entry, 0.01)
            tp1 = min(support, ideal_entry - 1.0 * risk) if support else ideal_entry - risk
            tp2 = max(0.01, ideal_entry - 2.0 * risk)
            tp3 = max(0.01, ideal_entry - 3.0 * risk)
            rr = (ideal_entry - tp2) / risk
            invalidation = f"Daily close above {stop:.2f} invalidates the bearish setup."
        else:
            ideal_entry = entry_low = entry_high = stop = tp1 = tp2 = tp3 = rr = None
            invalidation = "No active trade setup. Wait for stronger confluence."

        position_plan = None
        if portfolio_value and ideal_entry and stop:
            risk_per_unit = abs(ideal_entry - stop)
            max_risk_amount = portfolio_value * (max_risk_pct / 100.0)
            units = int(max_risk_amount // risk_per_unit) if risk_per_unit > 0 else 0
            position_plan = PositionPlan(
                portfolio_value=portfolio_value,
                max_risk_pct=max_risk_pct,
                max_risk_amount=round(max_risk_amount, 2),
                risk_per_unit=round(risk_per_unit, 4),
                suggested_units=units,
                estimated_position_value=round(units * ideal_entry, 2),
            )

        thesis = self._thesis(frame, indicators, action)
        risks = self._risks(indicators, action)

        return TradeSignal(
            symbol=symbol.upper(),
            action=action,
            generated_at=datetime.now(timezone.utc),
            timeframe=timeframe,
            current_price=round(price, 4),
            entry_zone_low=self._r(entry_low),
            entry_zone_high=self._r(entry_high),
            ideal_entry=self._r(ideal_entry),
            stop_loss=self._r(stop),
            take_profit_1=self._r(tp1),
            take_profit_2=self._r(tp2),
            take_profit_3=self._r(tp3),
            risk_reward=self._r(rr),
            signal_strength=score.total,
            scores=score,
            technicals=TechnicalSnapshot(**{k: self._r(v) for k, v in indicators.items()}),
            thesis=thesis,
            risks=risks,
            invalidation=invalidation,
            position_plan=position_plan,
            data_source=self.provider.name,
        )

    def _indicators(self, frame: pd.DataFrame) -> dict[str, float | None]:
        close = frame["close"].astype(float)
        high = frame["high"].astype(float)
        low = frame["low"].astype(float)
        volume = frame["volume"].astype(float)

        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        macd_series = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        macd_signal_series = macd_series.ewm(span=9, adjust=False).mean()

        prev_close = close.shift(1)
        true_range = pd.concat(
            [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(14).mean()

        avg_vol = volume.rolling(20).mean()
        rel_vol = volume.iloc[-1] / avg_vol.iloc[-1] if avg_vol.iloc[-1] else np.nan

        lookback = frame.tail(20)
        return {
            "rsi_14": self._finite(rsi.iloc[-1]),
            "macd": self._finite(macd_series.iloc[-1]),
            "macd_signal": self._finite(macd_signal_series.iloc[-1]),
            "ema_20": self._finite(ema20.iloc[-1]),
            "ema_50": self._finite(ema50.iloc[-1]),
            "ema_200": self._finite(ema200.iloc[-1]),
            "atr_14": self._finite(atr.iloc[-1]) or max(float(close.iloc[-1]) * 0.02, 0.01),
            "relative_volume": self._finite(rel_vol),
            "support": self._finite(float(lookback["low"].min())),
            "resistance": self._finite(float(lookback["high"].max())),
        }

    def _score(self, frame: pd.DataFrame, i: dict[str, float | None]) -> ScoreBreakdown:
        price = float(frame["close"].iloc[-1])
        trend = 0.0
        if i["ema_20"] and price > i["ema_20"]:
            trend += 8
        if i["ema_20"] and i["ema_50"] and i["ema_20"] > i["ema_50"]:
            trend += 8
        if i["ema_50"] and i["ema_200"] and i["ema_50"] > i["ema_200"]:
            trend += 9

        momentum = 0.0
        rsi = i["rsi_14"]
        if rsi is not None:
            if 50 <= rsi <= 70:
                momentum += 12
            elif 40 <= rsi < 50:
                momentum += 7
            elif 30 <= rsi < 40:
                momentum += 3
        if i["macd"] is not None and i["macd_signal"] is not None and i["macd"] > i["macd_signal"]:
            momentum += 8

        volume = 0.0
        rv = i["relative_volume"]
        if rv is not None:
            volume = min(15.0, max(0.0, 7.5 * rv))

        support = i["support"] or price
        resistance = i["resistance"] or price
        span = max(resistance - support, 0.01)
        location = (price - support) / span
        structure = max(0.0, min(15.0, 15.0 * (1.0 - min(location, 1.0))))

        regime = 10.0 if i["ema_50"] and i["ema_200"] and i["ema_50"] > i["ema_200"] else 5.0
        risk_reward = 7.0

        return ScoreBreakdown(
            trend=round(trend, 2),
            momentum=round(momentum, 2),
            volume=round(volume, 2),
            structure=round(structure, 2),
            regime=round(regime, 2),
            risk_reward=round(risk_reward, 2),
        )

    @staticmethod
    def _action(score: float) -> SignalAction:
        if score >= 82:
            return SignalAction.strong_long
        if score >= 70:
            return SignalAction.long
        if score >= 60:
            return SignalAction.watch
        if score >= 45:
            return SignalAction.neutral
        if score >= 32:
            return SignalAction.short
        if score < 32:
            return SignalAction.strong_short
        return SignalAction.no_trade

    @staticmethod
    def _thesis(frame: pd.DataFrame, i: dict[str, float | None], action: SignalAction) -> list[str]:
        price = float(frame["close"].iloc[-1])
        points: list[str] = []
        if i["ema_20"] and i["ema_50"]:
            points.append(
                "Short-term trend is bullish."
                if i["ema_20"] > i["ema_50"]
                else "Short-term trend is not yet bullish."
            )
        if i["rsi_14"] is not None:
            points.append(f"RSI(14) is {i['rsi_14']:.1f}.")
        if i["relative_volume"] is not None:
            points.append(f"Relative volume is {i['relative_volume']:.2f}x its 20-day average.")
        points.append(f"Last daily close used for technical structure was {price:.2f}.")
        points.append(f"Engine classification: {action.value}.")
        return points

    @staticmethod
    def _risks(i: dict[str, float | None], action: SignalAction) -> list[str]:
        risks = ["Signals are model-generated and can fail; stop levels must be respected."]
        if i["relative_volume"] is not None and i["relative_volume"] < 0.8:
            risks.append("Volume confirmation is weak.")
        if i["rsi_14"] is not None and i["rsi_14"] > 70:
            risks.append("RSI indicates an extended/overbought condition.")
        if action in {SignalAction.neutral, SignalAction.watch, SignalAction.no_trade}:
            risks.append("Confluence is not strong enough for a high-conviction entry.")
        return risks

    @staticmethod
    def _finite(value: float | np.floating | None) -> float | None:
        if value is None:
            return None
        value = float(value)
        return value if np.isfinite(value) else None

    @staticmethod
    def _r(value: float | None) -> float | None:
        return round(float(value), 4) if value is not None else None
