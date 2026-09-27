from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.market_data import MarketDataProvider
from src.signals.models import (
    HistoricalEvidence,
    MarketRegime,
    PositionPlan,
    PriceFreshness,
    ScoreBreakdown,
    SignalAction,
    SetupType,
    TechnicalSnapshot,
    TimeframeAnalysis,
    TradeSignal,
)


class SignalService:
    """Deterministic signal engine. LLMs may explain results but never create price levels."""

    def __init__(self, provider: MarketDataProvider):
        self.provider = provider

    def generate(
        self,
        symbol: str,
        timeframe: str = "swing",
        portfolio_value: float | None = None,
        max_risk_pct: float = 1.0,
    ) -> TradeSignal:
        symbol = symbol.strip().upper()
        if not symbol or len(symbol) > 20:
            raise ValueError("Invalid symbol")

        daily = self.provider.history(symbol, period="1y", interval="1d")
        quote = self.provider.quote(symbol)
        indicators = self._indicators(daily)
        regime = self._regime(indicators)
        timeframes = self._multi_timeframe(symbol, daily)
        scores = self._score(daily, indicators, regime, timeframes)
        action = self._action(scores.total, indicators, regime)
        setup_type = self._setup_type(action, quote.price, indicators)
        levels = self._levels(action, quote.price, indicators, setup_type)
        evidence = self._historical_evidence(daily, action, setup_type)

        position_plan = self._position_plan(
            portfolio_value=portfolio_value,
            max_risk_pct=max_risk_pct,
            entry=levels["ideal_entry"],
            stop=levels["stop_loss"],
        )

        return TradeSignal(
            symbol=symbol,
            action=action,
            generated_at=datetime.now(timezone.utc),
            price_timestamp=quote.timestamp,
            price_freshness=PriceFreshness.unknown,
            timeframe=timeframe,
            setup_type=setup_type,
            market_regime=regime,
            current_price=round(quote.price, 4),
            entry_zone_low=self._r(levels["entry_zone_low"]),
            entry_zone_high=self._r(levels["entry_zone_high"]),
            ideal_entry=self._r(levels["ideal_entry"]),
            stop_loss=self._r(levels["stop_loss"]),
            take_profit_1=self._r(levels["take_profit_1"]),
            take_profit_2=self._r(levels["take_profit_2"]),
            take_profit_3=self._r(levels["take_profit_3"]),
            risk_reward=self._r(levels["risk_reward"]),
            signal_strength=scores.total,
            scores=scores,
            technicals=TechnicalSnapshot(**{k: self._r(v) for k, v in indicators.items()}),
            timeframes=timeframes,
            thesis=self._thesis(indicators, action, regime, timeframes),
            catalysts=self._catalysts(indicators, action),
            risks=self._risks(indicators, action, regime),
            invalidation=levels["invalidation"],
            position_plan=position_plan,
            historical_evidence=evidence,
            data_source=self.provider.name,
        )

    def _multi_timeframe(self, symbol: str, daily: pd.DataFrame) -> list[TimeframeAnalysis]:
        specs = [
            ("Short term", "1h", "3mo", "1h"),
            ("Swing", "1d", None, None),
            ("Primary trend", "1wk", "2y", "1wk"),
        ]
        output: list[TimeframeAnalysis] = []
        for label, interval, period, provider_interval in specs:
            try:
                frame = daily if interval == "1d" else self.provider.history(symbol, period=period, interval=provider_interval)
                i = self._indicators(frame)
                price = float(frame["close"].iloc[-1])
                bullish = int(bool(i["ema_20"] and price > i["ema_20"])) + int(
                    bool(i["ema_20"] and i["ema_50"] and i["ema_20"] > i["ema_50"])
                )
                rsi = i["rsi_14"]
                if bullish == 2 and rsi is not None and rsi >= 50:
                    trend, score = "BULLISH", 85.0
                elif bullish >= 1:
                    trend, score = "MIXED_BULLISH", 65.0
                elif bullish == 0 and rsi is not None and rsi < 45:
                    trend, score = "BEARISH", 25.0
                else:
                    trend, score = "MIXED", 45.0
                output.append(
                    TimeframeAnalysis(
                        label=label,
                        interval=interval,
                        trend=trend,
                        score=score,
                        last_close=self._r(price),
                        ema_20=self._r(i["ema_20"]),
                        ema_50=self._r(i["ema_50"]),
                        rsi_14=self._r(rsi),
                    )
                )
            except Exception:
                continue
        return output

    def _indicators(self, frame: pd.DataFrame) -> dict[str, float | None]:
        if len(frame) < 30:
            raise ValueError("Insufficient price history for signal generation")
        close = frame["close"].astype(float)
        high = frame["high"].astype(float)
        low = frame["low"].astype(float)
        volume = frame["volume"].astype(float)

        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean().replace(0, np.nan)
        rsi = 100 - (100 / (1 + gain / loss))
        macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        macd_signal = macd.ewm(span=9, adjust=False).mean()

        prev_close = close.shift(1)
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
        avg_vol = volume.rolling(20).mean()
        relative_volume = volume.iloc[-1] / avg_vol.iloc[-1] if avg_vol.iloc[-1] else np.nan
        lookback = frame.tail(60)
        support, resistance = self._pivot_levels(frame)
        change_20d = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) >= 21 else np.nan
        distance_200 = (close.iloc[-1] / ema200.iloc[-1] - 1) * 100 if ema200.iloc[-1] else np.nan

        return {
            "rsi_14": self._finite(rsi.iloc[-1]),
            "macd": self._finite(macd.iloc[-1]),
            "macd_signal": self._finite(macd_signal.iloc[-1]),
            "ema_20": self._finite(ema20.iloc[-1]),
            "ema_50": self._finite(ema50.iloc[-1]),
            "ema_200": self._finite(ema200.iloc[-1]),
            "atr_14": self._finite(atr.iloc[-1]) or max(float(close.iloc[-1]) * 0.02, 0.01),
            "relative_volume": self._finite(relative_volume),
            "support": self._finite(support),
            "resistance": self._finite(resistance),
            "change_20d_pct": self._finite(change_20d),
            "distance_from_200ema_pct": self._finite(distance_200),
        }


    @staticmethod
    def _pivot_levels(frame: pd.DataFrame, window: int = 3) -> tuple[float, float]:
        """Nearest confirmed swing support/resistance; falls back to recent range."""
        tail = frame.tail(90).copy()
        lows = tail["low"].astype(float)
        highs = tail["high"].astype(float)
        close = float(tail["close"].iloc[-1])
        pivot_lows, pivot_highs = [], []
        for idx in range(window, len(tail) - window):
            lo = float(lows.iloc[idx])
            hi = float(highs.iloc[idx])
            if lo <= float(lows.iloc[idx-window:idx+window+1].min()):
                pivot_lows.append(lo)
            if hi >= float(highs.iloc[idx-window:idx+window+1].max()):
                pivot_highs.append(hi)
        below = [x for x in pivot_lows if x < close]
        above = [x for x in pivot_highs if x > close]
        support = max(below) if below else float(lows.tail(20).min())
        resistance = min(above) if above else float(highs.tail(20).max())
        return support, resistance

    @staticmethod
    def _setup_type(action: SignalAction, price: float, i: dict[str, float | None]) -> SetupType:
        atr = float(i["atr_14"] or max(price * 0.02, 0.01))
        support, resistance, ema20 = i["support"], i["resistance"], i["ema_20"]
        if action in {SignalAction.long, SignalAction.strong_long}:
            if resistance and price >= resistance - 0.35 * atr:
                return SetupType.breakout
            if (support and abs(price - support) <= 0.8 * atr) or (ema20 and abs(price - ema20) <= 0.65 * atr):
                return SetupType.pullback
            return SetupType.trend_continuation
        if action in {SignalAction.short, SignalAction.strong_short}:
            if support and price <= support + 0.35 * atr:
                return SetupType.breakdown
            if resistance and abs(price - resistance) <= 0.8 * atr:
                return SetupType.pullback
            return SetupType.trend_continuation
        return SetupType.none

    def _historical_evidence(self, frame: pd.DataFrame, action: SignalAction, setup: SetupType) -> HistoricalEvidence | None:
        """Walk-forward analog study. No future bar is used to classify an observation."""
        if action not in {SignalAction.long, SignalAction.strong_long, SignalAction.short, SignalAction.strong_short}:
            return None
        if len(frame) < 120:
            return HistoricalEvidence(note="Not enough history for a meaningful walk-forward analog study.")
        close = frame["close"].astype(float)
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean().replace(0, np.nan)
        rsi = 100 - (100/(1+gain/loss))
        horizon = 10
        returns, holds = [], []
        long_side = action in {SignalAction.long, SignalAction.strong_long}
        for idx in range(55, len(frame)-horizon):
            trend_ok = ema20.iloc[idx] > ema50.iloc[idx] if long_side else ema20.iloc[idx] < ema50.iloc[idx]
            momentum_ok = rsi.iloc[idx] >= 50 if long_side else rsi.iloc[idx] <= 50
            if not (trend_ok and momentum_ok):
                continue
            entry = float(close.iloc[idx])
            exit_price = float(close.iloc[idx+horizon])
            ret = (exit_price/entry-1)*100
            if not long_side:
                ret *= -1
            returns.append(ret)
            holds.append(horizon)
        if not returns:
            return HistoricalEvidence(note="No comparable historical observations passed the walk-forward filters.")
        arr = np.array(returns, dtype=float)
        wins = arr[arr > 0]
        losses = arr[arr <= 0]
        equity = np.cumprod(1 + arr/100)
        peak = np.maximum.accumulate(equity)
        drawdowns = (equity/peak-1)*100
        avg_loss = float(losses.mean()) if len(losses) else None
        avg_win = float(wins.mean()) if len(wins) else None
        avg_r = abs(avg_win/avg_loss) if avg_win is not None and avg_loss not in (None, 0) else None
        return HistoricalEvidence(
            sample_size=len(arr), wins=len(wins), losses=len(losses),
            hit_rate_pct=round(len(wins)/len(arr)*100, 2),
            avg_return_pct=round(float(arr.mean()), 2),
            avg_win_pct=round(avg_win, 2) if avg_win is not None else None,
            avg_loss_pct=round(avg_loss, 2) if avg_loss is not None else None,
            avg_r_multiple=round(avg_r, 2) if avg_r is not None else None,
            max_drawdown_pct=round(float(drawdowns.min()), 2),
            median_holding_bars=float(np.median(holds)),
        )

    @staticmethod
    def _regime(i: dict[str, float | None]) -> MarketRegime:
        e20, e50, e200 = i["ema_20"], i["ema_50"], i["ema_200"]
        if e20 and e50 and e200:
            if e20 > e50 > e200:
                return MarketRegime.strong_uptrend
            if e20 > e50:
                return MarketRegime.uptrend
            if e20 < e50 < e200:
                return MarketRegime.strong_downtrend
            if e20 < e50:
                return MarketRegime.downtrend
        return MarketRegime.range

    def _score(
        self,
        frame: pd.DataFrame,
        i: dict[str, float | None],
        regime: MarketRegime,
        timeframes: list[TimeframeAnalysis],
    ) -> ScoreBreakdown:
        price = float(frame["close"].iloc[-1])
        trend = 0.0
        trend += 8 if i["ema_20"] and price > i["ema_20"] else 0
        trend += 8 if i["ema_20"] and i["ema_50"] and i["ema_20"] > i["ema_50"] else 0
        trend += 9 if i["ema_50"] and i["ema_200"] and i["ema_50"] > i["ema_200"] else 0

        momentum = 0.0
        rsi = i["rsi_14"]
        if rsi is not None:
            momentum += 12 if 50 <= rsi <= 68 else 7 if 45 <= rsi < 50 else 3 if 35 <= rsi < 45 else 0
        if i["macd"] is not None and i["macd_signal"] is not None and i["macd"] > i["macd_signal"]:
            momentum += 8

        rv = i["relative_volume"]
        volume = min(15.0, max(0.0, 7.5 * rv)) if rv is not None else 5.0

        support, resistance = i["support"] or price, i["resistance"] or price
        span = max(resistance - support, 0.01)
        location = max(0.0, min(1.0, (price - support) / span))
        structure = round(7.5 + (7.5 * (1 - abs(0.5 - location) * 2)), 2)

        regime_score = {
            MarketRegime.strong_uptrend: 15.0,
            MarketRegime.uptrend: 12.0,
            MarketRegime.range: 7.0,
            MarketRegime.downtrend: 4.0,
            MarketRegime.strong_downtrend: 1.0,
        }[regime]
        if timeframes:
            bullish_share = sum(t.score >= 60 for t in timeframes) / len(timeframes)
            regime_score = min(15.0, regime_score * 0.7 + bullish_share * 4.5)

        return ScoreBreakdown(
            trend=round(trend, 2),
            momentum=round(momentum, 2),
            volume=round(volume, 2),
            structure=structure,
            regime=round(regime_score, 2),
            risk_reward=8.0,
        )

    @staticmethod
    def _action(score: float, i: dict[str, float | None], regime: MarketRegime) -> SignalAction:
        bearish_regime = regime in {MarketRegime.downtrend, MarketRegime.strong_downtrend}
        macd_bearish = (
            i["macd"] is not None and i["macd_signal"] is not None and i["macd"] < i["macd_signal"]
        )
        if bearish_regime and macd_bearish and score < 45:
            return SignalAction.strong_short if regime == MarketRegime.strong_downtrend else SignalAction.short
        if score >= 82:
            return SignalAction.strong_long
        if score >= 70:
            return SignalAction.long
        if score >= 60:
            return SignalAction.watch
        if score >= 45:
            return SignalAction.neutral
        return SignalAction.no_trade

    def _levels(self, action: SignalAction, price: float, i: dict[str, float | None], setup_type: SetupType = SetupType.none) -> dict[str, float | str | None]:
        atr = float(i["atr_14"] or price * 0.02)
        support, resistance = i["support"], i["resistance"]
        if action in {SignalAction.strong_long, SignalAction.long}:
            anchor = float(i["resistance"]) if setup_type == SetupType.breakout and i["resistance"] else min(price, float(i["ema_20"] or price))
            low, high = max(0.01, anchor - 0.25 * atr), anchor + 0.25 * atr
            stop = min(float(support), anchor - 1.5 * atr) if support else anchor - 1.5 * atr
            risk = max(anchor - stop, 0.01)
            tp1 = max(float(resistance), anchor + risk) if resistance else anchor + risk
            tp2, tp3 = anchor + 2 * risk, anchor + 3 * risk
            return self._level_dict(low, high, anchor, stop, tp1, tp2, tp3, 2.0, f"Daily close below {stop:.2f}.")
        if action in {SignalAction.strong_short, SignalAction.short}:
            anchor = float(i["support"]) if setup_type == SetupType.breakdown and i["support"] else max(price, float(i["ema_20"] or price))
            low, high = max(0.01, anchor - 0.25 * atr), anchor + 0.25 * atr
            stop = max(float(resistance), anchor + 1.5 * atr) if resistance else anchor + 1.5 * atr
            risk = max(stop - anchor, 0.01)
            tp1 = min(float(support), anchor - risk) if support else anchor - risk
            tp2, tp3 = max(0.01, anchor - 2 * risk), max(0.01, anchor - 3 * risk)
            return self._level_dict(low, high, anchor, stop, tp1, tp2, tp3, 2.0, f"Daily close above {stop:.2f}.")
        return self._level_dict(None, None, None, None, None, None, None, None, "No executable setup yet.")

    @staticmethod
    def _level_dict(low, high, entry, stop, tp1, tp2, tp3, rr, invalidation):
        return {
            "entry_zone_low": low, "entry_zone_high": high, "ideal_entry": entry, "stop_loss": stop,
            "take_profit_1": tp1, "take_profit_2": tp2, "take_profit_3": tp3,
            "risk_reward": rr, "invalidation": invalidation,
        }

    @staticmethod
    def _position_plan(portfolio_value, max_risk_pct, entry, stop):
        if not portfolio_value or not entry or not stop:
            return None
        risk_per_unit = abs(float(entry) - float(stop))
        max_risk_amount = portfolio_value * max_risk_pct / 100
        units_by_risk = int(max_risk_amount // risk_per_unit) if risk_per_unit else 0
        units_by_cash = int(portfolio_value // float(entry))
        units = max(0, min(units_by_risk, units_by_cash))
        value = units * float(entry)
        return PositionPlan(
            portfolio_value=portfolio_value,
            max_risk_pct=max_risk_pct,
            max_risk_amount=round(max_risk_amount, 2),
            risk_per_unit=round(risk_per_unit, 4),
            suggested_units=units,
            estimated_position_value=round(value, 2),
            portfolio_allocation_pct=round(value / portfolio_value * 100, 2),
        )

    @staticmethod
    def _thesis(i, action, regime, timeframes):
        points = [f"Market regime: {regime.value}.", f"Engine classification: {action.value}."]
        if i["rsi_14"] is not None:
            points.append(f"RSI(14): {i['rsi_14']:.1f}.")
        if i["relative_volume"] is not None:
            points.append(f"Relative volume: {i['relative_volume']:.2f}x the 20-period average.")
        if timeframes:
            aligned = sum(t.score >= 60 for t in timeframes)
            points.append(f"Multi-timeframe alignment: {aligned}/{len(timeframes)} bullish or mixed-bullish.")
        return points

    @staticmethod
    def _catalysts(i, action):
        items = []
        if i["relative_volume"] is not None and i["relative_volume"] >= 1.25:
            items.append("Above-average volume confirms elevated participation.")
        if i["macd"] is not None and i["macd_signal"] is not None and i["macd"] > i["macd_signal"]:
            items.append("MACD is above its signal line.")
        if action in {SignalAction.strong_long, SignalAction.long}:
            items.append("Upside continuation improves if price holds the entry zone and clears resistance.")
        return items

    @staticmethod
    def _risks(i, action, regime):
        risks = ["Model levels can fail; gap risk can cause fills beyond the modeled stop."]
        if i["relative_volume"] is not None and i["relative_volume"] < 0.8:
            risks.append("Volume confirmation is weak.")
        if i["rsi_14"] is not None and i["rsi_14"] > 70:
            risks.append("RSI is extended, increasing pullback risk.")
        if regime == MarketRegime.range:
            risks.append("Range regime increases false-breakout risk.")
        if action in {SignalAction.neutral, SignalAction.watch, SignalAction.no_trade}:
            risks.append("There is no high-conviction executable setup yet.")
        return risks

    @staticmethod
    def _finite(value):
        if value is None:
            return None
        value = float(value)
        return value if np.isfinite(value) else None

    @staticmethod
    def _r(value):
        return round(float(value), 4) if value is not None else None
