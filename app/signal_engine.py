"""Core signal evaluation logic combining market and model probabilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from pprint import pprint
from typing import Any

from app.model_loader import predict_proba_raw

logger = logging.getLogger(__name__)
MISPRICING_THRESHOLD = 0.10

RISK_PROFILES = {
    "conservative": {
        "yes_cutoff": 0.70,
        "no_cutoff": 0.30,
        "min_seconds": 60,
        "max_seconds": 180,
        "early_entry_enabled": False,
        "early_entry_min_seconds": None,
        "early_entry_max_seconds": None,
        "early_entry_cutoff": None,
        "description": "Final 2 minutes only. Highest threshold. Fewest trades, highest precision.",
    },
    "moderate": {
        "yes_cutoff": 0.65,
        "no_cutoff": 0.35,
        "min_seconds": 60,
        "max_seconds": 300,
        "early_entry_enabled": False,
        "early_entry_min_seconds": None,
        "early_entry_max_seconds": None,
        "early_entry_cutoff": None,
        "description": "Final 3 minutes. Recommended balance of accuracy and frequency.",
    },
    "aggressive": {
        "yes_cutoff": 0.65,
        "no_cutoff": 0.35,
        "min_seconds": 60,
        "max_seconds": 480,
        "early_entry_enabled": True,
        "early_entry_min_seconds": 240,
        "early_entry_max_seconds": 480,
        "early_entry_cutoff": 0.82,
        "description": "Standard window up to 5 min. Early entry from 4-8 min at 82% threshold.",
    },
    "high_conviction": {
        "yes_cutoff": 0.65,
        "no_cutoff": 0.35,
        "min_seconds": 60,
        "max_seconds": 300,
        "early_entry_enabled": True,
        "early_entry_min_seconds": 180,
        "early_entry_max_seconds": 480,
        "early_entry_cutoff": 0.87,
        "description": "Standard 3-min window plus early entry from 3-8 min when both models exceed 87%.",
    },
}


@dataclass
class SignalResult:
    """Structured result of signal evaluation."""

    signal: str
    reason: str
    agreement_region: str
    p_market: float
    p_raw: float
    seconds_to_close: int
    entry_bucket: int
    yes_cutoff: float
    no_cutoff: float
    confidence: float
    p_market_source: str
    entry_filtered: bool = False


def no_signal_result(
    p_market: float,
    p_raw: float,
    seconds_to_close: int,
    entry_bucket: int,
    yes_cutoff: float,
    no_cutoff: float,
    agreement_region: str,
    reason: str,
) -> SignalResult:
    """Create a NO SIGNAL result with common metadata fields."""
    confidence = abs(float(p_market) - 0.5) + abs(float(p_raw) - 0.5)
    return SignalResult(
        signal="NO SIGNAL",
        reason=reason,
        agreement_region=agreement_region,
        p_market=float(p_market),
        p_raw=float(p_raw),
        seconds_to_close=int(seconds_to_close),
        entry_bucket=int(entry_bucket),
        yes_cutoff=float(yes_cutoff),
        no_cutoff=float(no_cutoff),
        confidence=float(confidence),
        p_market_source="unknown",
    )


def determine_agreement_region(p_market: float, p_raw: float, yes_cutoff: float, no_cutoff: float) -> str:
    """Determine agreement region between market and model probabilities."""
    if p_market >= yes_cutoff and p_raw >= yes_cutoff:
        return "agree_yes"
    if p_market <= no_cutoff and p_raw <= no_cutoff:
        return "agree_no"
    if p_market >= 0.5 and p_raw < 0.5:
        return "market_yes_raw_no"
    if p_market < 0.5 and p_raw >= 0.5:
        return "market_no_raw_yes"
    return "no_agreement"


def evaluate_signal(
    p_market: float,
    p_raw: float,
    seconds_to_close: int,
    entry_bucket: int,
    yes_cutoff: float,
    no_cutoff: float,
    min_seconds: int,
    max_seconds: int,
    enable_no_signals: bool = False,
    early_entry_enabled: bool = False,
    early_entry_min_seconds: int | None = None,
    early_entry_max_seconds: int | None = None,
    early_entry_cutoff: float | None = None,
    max_entry_price_yes: float = 1.0,
    max_entry_price_no: float = 1.0,
    force_on_high_conviction: bool = False,
    volatility_guard_active: bool = False,
) -> SignalResult:
    """Evaluate raw + market probabilities into a paper-trading signal."""
    confidence = abs(float(p_market) - 0.5) + abs(float(p_raw) - 0.5)
    logger.info(
        "evaluate_signal called: seconds=%s, normal_window=%s-%ss, in_normal=%s, early_enabled=%s, early_window=%s-%ss",
        int(seconds_to_close),
        int(min_seconds),
        int(max_seconds),
        int(min_seconds) <= int(seconds_to_close) <= int(max_seconds),
        bool(early_entry_enabled),
        str(early_entry_min_seconds),
        str(early_entry_max_seconds),
    )
    logger.info(
        "Time window check: %s <= %s <= %s: %s",
        int(min_seconds),
        int(seconds_to_close),
        int(max_seconds),
        int(min_seconds) <= int(seconds_to_close) <= int(max_seconds),
    )
    if int(seconds_to_close) > 600 and not force_on_high_conviction:
        return SignalResult(
            signal="NO SIGNAL",
            reason=f"Too early — {seconds_to_close}s to close (max 600s)",
            agreement_region="outside_time_window",
            p_market=float(p_market),
            p_raw=float(p_raw),
            seconds_to_close=int(seconds_to_close),
            entry_bucket=int(entry_bucket),
            yes_cutoff=float(yes_cutoff),
            no_cutoff=float(no_cutoff),
            confidence=float(confidence),
            p_market_source="unknown",
        )
    in_normal_window = int(min_seconds) <= int(seconds_to_close) <= int(max_seconds)

    in_early_window = False
    if early_entry_enabled and early_entry_min_seconds and early_entry_max_seconds:
        in_early_window = int(early_entry_min_seconds) <= int(seconds_to_close) <= int(early_entry_max_seconds)

    if not in_normal_window and not in_early_window and not force_on_high_conviction:
        return SignalResult(
            signal="NO SIGNAL",
            reason=f"Outside window ({seconds_to_close}s)",
            agreement_region="outside_time_window",
            p_market=float(p_market),
            p_raw=float(p_raw),
            seconds_to_close=int(seconds_to_close),
            entry_bucket=int(entry_bucket),
            yes_cutoff=float(yes_cutoff),
            no_cutoff=float(no_cutoff),
            confidence=float(confidence),
            p_market_source="unknown",
        )

    is_early_entry = False
    if in_early_window and not in_normal_window and early_entry_cutoff is not None and not force_on_high_conviction:
        effective_yes_cutoff = float(early_entry_cutoff)
        effective_no_cutoff = float(1.0 - float(early_entry_cutoff))
        is_early_entry = True
    else:
        effective_yes_cutoff = float(yes_cutoff)
        effective_no_cutoff = float(no_cutoff)

    region = determine_agreement_region(float(p_market), float(p_raw), effective_yes_cutoff, effective_no_cutoff)
    logger.info(
        "evaluate_signal: p_market=%.4f, p_raw=%.4f, yes_cutoff=%.3f, no_cutoff=%.3f, enable_no=%s, seconds=%s, window=%s-%ss, region=%s",
        float(p_market),
        float(p_raw),
        float(effective_yes_cutoff),
        float(effective_no_cutoff),
        enable_no_signals,
        int(seconds_to_close),
        int(min_seconds),
        int(max_seconds),
        region,
    )

    entry_filtered = False
    if region == "agree_yes":
        entry_label = " (early entry)" if is_early_entry else ""
        signal = "PAPER BUY YES"
        reason = (
            f"Strong YES agreement{entry_label} "
            f"(market={p_market:.3f}, model={p_raw:.3f}, "
            f"cutoff={effective_yes_cutoff:.2f})"
        )
    elif region == "agree_no" and enable_no_signals:
        NO_TRADE_MAX_SECONDS = 120  # Never auto-trade NO with more than 2 min
        logger.info(
            "NO guard check: seconds_to_close=%s, max=%s, blocked=%s",
            int(seconds_to_close),
            NO_TRADE_MAX_SECONDS,
            int(seconds_to_close) > NO_TRADE_MAX_SECONDS,
        )
        if int(seconds_to_close) > NO_TRADE_MAX_SECONDS:
            signal = "NO SIGNAL"
            reason = (
                f"NO signal suppressed — {seconds_to_close}s to close "
                f"exceeds {NO_TRADE_MAX_SECONDS}s max for NO trades. "
                f"Too much time for reversal."
            )
        else:
            signal = "PAPER BUY NO"
            reason = f"Strong NO agreement (market={p_market:.3f}, model={p_raw:.3f})"
    else:
        signal = "NO SIGNAL"
        reason = f"Region: {region} — no trade condition met"

    # Profitability filter guard to avoid low-upside entries.
    if signal in {"PAPER BUY YES", "PAPER BUY NO"}:
        if signal == "PAPER BUY YES":
            if float(p_market) > float(max_entry_price_yes):
                logger.info(
                    "Entry filter BLOCKED YES trade: p_market=%.3f > max=%.3f, upside only $%.3f/contract",
                    float(p_market),
                    float(max_entry_price_yes),
                    (1 - float(p_market)),
                )
                signal = "NO SIGNAL"
                reason = (
                    f"Entry blocked: YES at {float(p_market):.1%} > "
                    f"max {float(max_entry_price_yes):.1%}. "
                    f"At this price, {(1 - float(p_market)) * 100:.1f}¢ upside "
                    f"per contract."
                )
                entry_filtered = True
                region = "entry_filtered"
        elif signal == "PAPER BUY NO":
            no_price = 1.0 - float(p_market)
            if no_price > float(max_entry_price_no):
                logger.info(
                    "Entry filter BLOCKED NO trade: no_price=%.3f > max=%.3f",
                    float(no_price),
                    float(max_entry_price_no),
                )
                signal = "NO SIGNAL"
                reason = (
                    f"Entry filtered: {no_price:.1%} NO price exceeds "
                    f"{float(max_entry_price_no):.1%} max. "
                    f"Only ${float(p_market):.3f} upside per contract."
                )
                entry_filtered = True
                region = "entry_filtered"

    if volatility_guard_active and signal in {"PAPER BUY YES", "PAPER BUY NO"}:
        return SignalResult(
            signal="NO SIGNAL",
            reason="Volatility guard: reversal risk too high for agreement trade",
            agreement_region="volatility_guard",
            p_market=float(p_market),
            p_raw=float(p_raw),
            seconds_to_close=int(seconds_to_close),
            entry_bucket=int(entry_bucket),
            yes_cutoff=effective_yes_cutoff,
            no_cutoff=effective_no_cutoff,
            confidence=float(confidence),
            p_market_source="unknown",
            entry_filtered=entry_filtered,
        )

    return SignalResult(
        signal=signal,
        reason=reason,
        agreement_region=region,
        p_market=float(p_market),
        p_raw=float(p_raw),
        seconds_to_close=int(seconds_to_close),
        entry_bucket=int(entry_bucket),
        yes_cutoff=effective_yes_cutoff,
        no_cutoff=effective_no_cutoff,
        confidence=float(confidence),
        p_market_source="unknown",
        entry_filtered=entry_filtered,
    )


def evaluate_mispricing_signal(
    p_market: float,
    p_raw: float,
    seconds_to_close: int,
    entry_bucket: int,
    min_seconds: int,
    max_seconds: int,
    mispricing_threshold: float = MISPRICING_THRESHOLD,
    early_entry_enabled: bool = False,
    early_entry_min_seconds: int | None = None,
    early_entry_max_seconds: int | None = None,
    early_entry_cutoff: float | None = None,
    max_entry_price_yes: float = 1.0,
    max_entry_price_no: float = 1.0,
) -> SignalResult:
    """Evaluate model-vs-market gap for relative mispricing opportunities."""
    confidence = abs(float(p_market) - 0.5) + abs(float(p_raw) - 0.5)
    in_normal_window = int(min_seconds) <= int(seconds_to_close) <= int(max_seconds)
    in_early_window = False
    if early_entry_enabled and early_entry_min_seconds and early_entry_max_seconds:
        in_early_window = int(early_entry_min_seconds) <= int(seconds_to_close) <= int(early_entry_max_seconds)

    if not in_normal_window and not in_early_window:
        return SignalResult(
            signal="NO SIGNAL",
            reason=f"Outside window ({seconds_to_close}s)",
            agreement_region="outside_time_window",
            p_market=float(p_market),
            p_raw=float(p_raw),
            seconds_to_close=int(seconds_to_close),
            entry_bucket=int(entry_bucket),
            yes_cutoff=float(mispricing_threshold),
            no_cutoff=float(-mispricing_threshold),
            confidence=float(confidence),
            p_market_source="unknown",
        )

    is_early_entry = bool(in_early_window and not in_normal_window)
    gap = float(p_raw) - float(p_market)
    threshold = max(0.0, float(mispricing_threshold))
    effective_threshold = threshold * 1.5 if is_early_entry else threshold
    if gap >= effective_threshold:
        result = SignalResult(
            signal="PAPER BUY YES",
            reason=(
                f"{'Early entry: ' if is_early_entry else ''}"
                f"Model ({p_raw:.1%}) exceeds market ({p_market:.1%}) "
                f"by {gap:.1%} — YES may be underpriced"
            ),
            agreement_region="model_bullish",
            p_market=float(p_market),
            p_raw=float(p_raw),
            seconds_to_close=int(seconds_to_close),
            entry_bucket=int(entry_bucket),
            yes_cutoff=float(effective_threshold),
            no_cutoff=float(-effective_threshold),
            confidence=float(confidence),
            p_market_source="unknown",
        )
    elif gap <= -effective_threshold:
        abs_gap = abs(gap)
        result = SignalResult(
            signal="PAPER BUY NO",
            reason=(
                f"{'Early entry: ' if is_early_entry else ''}"
                f"Market ({p_market:.1%}) exceeds model ({p_raw:.1%}) "
                f"by {abs_gap:.1%} — YES may be overpriced"
            ),
            agreement_region="model_bearish",
            p_market=float(p_market),
            p_raw=float(p_raw),
            seconds_to_close=int(seconds_to_close),
            entry_bucket=int(entry_bucket),
            yes_cutoff=float(effective_threshold),
            no_cutoff=float(-effective_threshold),
            confidence=float(confidence),
            p_market_source="unknown",
        )
    else:
        result = SignalResult(
            signal="NO SIGNAL",
            reason=f"Gap only {abs(gap):.1%}, need {effective_threshold:.1%}",
            agreement_region="no_agreement",
            p_market=float(p_market),
            p_raw=float(p_raw),
            seconds_to_close=int(seconds_to_close),
            entry_bucket=int(entry_bucket),
            yes_cutoff=float(effective_threshold),
            no_cutoff=float(-effective_threshold),
            confidence=float(confidence),
            p_market_source="unknown",
        )
    if result.signal == "PAPER BUY YES" and float(p_market) > float(max_entry_price_yes):
        logger.info(
            "Entry filter BLOCKED YES trade: p_market=%.3f > max=%.3f, upside only $%.3f/contract",
            float(p_market),
            float(max_entry_price_yes),
            (1 - float(p_market)),
        )
        result.signal = "NO SIGNAL"
        result.reason = (
            f"Entry filtered: {float(p_market):.1%} YES price exceeds "
            f"{float(max_entry_price_yes):.1%} max. "
            f"Only ${(1 - float(p_market)):.3f} upside per contract."
        )
        result.entry_filtered = True
    elif result.signal == "PAPER BUY NO":
        no_price = 1.0 - float(p_market)
        if no_price > float(max_entry_price_no):
            logger.info(
                "Entry filter BLOCKED NO trade: no_price=%.3f > max=%.3f",
                float(no_price),
                float(max_entry_price_no),
            )
            result.signal = "NO SIGNAL"
            result.reason = (
                f"Entry filtered: {no_price:.1%} NO price exceeds "
                f"{float(max_entry_price_no):.1%} max. "
                f"Only ${float(p_market):.3f} upside per contract."
            )
            result.entry_filtered = True
    return result


def evaluate_ensemble_signal(
    p_market: float,
    p_raw: float,
    seconds_to_close: int,
    entry_bucket: int,
    yes_cutoff: float = 0.65,
    max_entry_yes: float = 0.80,
    max_entry_no: float = 0.80,
    mispricing_threshold: float = 0.20,
    min_seconds: int = 60,
    max_seconds: int = 180,
    early_entry_enabled: bool = False,
    early_entry_min: int = 300,
    early_entry_max: int = 600,
    early_entry_cutoff: float = 0.80,
    volatility_guard_active: bool = False,
) -> SignalResult:
    """Ensemble vote: agreement or mispricing, with entry-price filters."""
    in_normal = int(min_seconds) <= int(seconds_to_close) <= int(max_seconds)
    in_early = bool(early_entry_enabled) and int(early_entry_min) <= int(seconds_to_close) <= int(early_entry_max)
    if not in_normal and not in_early:
        return no_signal_result(
            p_market,
            p_raw,
            seconds_to_close,
            entry_bucket,
            yes_cutoff,
            1.0 - yes_cutoff,
            "outside_time_window",
            f"Outside window ({seconds_to_close}s)",
        )

    effective_cutoff = float(early_entry_cutoff) if (in_early and not in_normal) else float(yes_cutoff)
    agreement_yes = float(p_market) >= effective_cutoff and float(p_raw) >= effective_cutoff
    gap = float(p_raw) - float(p_market)
    mispricing_bullish = gap >= float(mispricing_threshold)
    mispricing_bearish = (-gap) >= float(mispricing_threshold)
    yes_entry_ok = float(p_market) <= float(max_entry_yes)
    no_entry_ok = (1.0 - float(p_market)) <= float(max_entry_no)
    confidence = abs(float(p_market) - 0.5) + abs(float(p_raw) - 0.5)

    result: SignalResult
    if agreement_yes and yes_entry_ok:
        if mispricing_bullish:
            region = "agree_yes"
            reason = (
                f"Ensemble: Agreement ({p_market:.1%}) + "
                f"Bullish gap ({gap:.1%}) at {p_market:.1%} entry"
            )
        else:
            region = "agree_yes"
            reason = f"Ensemble: Agreement only ({p_market:.1%} mkt, {p_raw:.1%} model)"
        result = SignalResult(
            signal="PAPER BUY YES",
            reason=reason,
            agreement_region=region,
            p_market=float(p_market),
            p_raw=float(p_raw),
            seconds_to_close=int(seconds_to_close),
            entry_bucket=int(entry_bucket),
            yes_cutoff=float(effective_cutoff),
            no_cutoff=float(1.0 - effective_cutoff),
            confidence=float(confidence),
            p_market_source="unknown",
        )

    if mispricing_bullish and yes_entry_ok:
        reason = (
            f"Ensemble: Mispricing only — "
            f"model ({p_raw:.1%}) exceeds market ({p_market:.1%}) by {gap:.1%}"
        )
        result = SignalResult(
            signal="PAPER BUY YES",
            reason=reason,
            agreement_region="model_bullish",
            p_market=float(p_market),
            p_raw=float(p_raw),
            seconds_to_close=int(seconds_to_close),
            entry_bucket=int(entry_bucket),
            yes_cutoff=float(effective_cutoff),
            no_cutoff=float(1.0 - effective_cutoff),
            confidence=float(confidence),
            p_market_source="unknown",
        )

    if mispricing_bearish and no_entry_ok:
        reason = (
            f"Ensemble: Bearish gap — "
            f"market ({p_market:.1%}) exceeds model ({p_raw:.1%}) by {(-gap):.1%}"
        )
        result = SignalResult(
            signal="PAPER BUY NO",
            reason=reason,
            agreement_region="model_bearish",
            p_market=float(p_market),
            p_raw=float(p_raw),
            seconds_to_close=int(seconds_to_close),
            entry_bucket=int(entry_bucket),
            yes_cutoff=float(effective_cutoff),
            no_cutoff=float(1.0 - effective_cutoff),
            confidence=float(confidence),
            p_market_source="unknown",
        )

    parts: list[str] = []
    if agreement_yes and not yes_entry_ok:
        result = no_signal_result(
            p_market,
            p_raw,
            seconds_to_close,
            entry_bucket,
            effective_cutoff,
            1.0 - effective_cutoff,
            "entry_filtered",
            (
                f"Entry blocked: YES at {float(p_market):.1%} > "
                f"max {float(max_entry_yes):.1%}. "
                f"At this price, {(1 - float(p_market)) * 100:.1f}¢ upside per contract."
            ),
        )
    else:
        if not agreement_yes:
            parts.append(
                f"Agreement needs {(effective_cutoff - max(float(p_market), float(p_raw))) * 100:.1f}% more"
            )
        if abs(gap) < float(mispricing_threshold):
            parts.append(f"Gap {abs(gap):.1%} < {float(mispricing_threshold):.1%} threshold")
        if not yes_entry_ok:
            parts.append(f"Entry {float(p_market):.1%} > max {float(max_entry_yes):.1%}")
        if mispricing_bearish and not no_entry_ok:
            parts.append(
                f"NO entry {(1.0 - float(p_market)):.1%} > max {float(max_entry_no):.1%}"
            )
        result = no_signal_result(
            p_market,
            p_raw,
            seconds_to_close,
            entry_bucket,
            effective_cutoff,
            1.0 - effective_cutoff,
            "no_agreement",
            " | ".join(parts) or "No conditions met",
        )

    if volatility_guard_active and result.signal in {"PAPER BUY YES", "PAPER BUY NO"}:
        if result.agreement_region in {"agree_yes", "agree_no"}:
            return SignalResult(
                signal="NO SIGNAL",
                reason="Volatility guard: reversal risk too high for agreement trade",
                agreement_region="volatility_guard",
                p_market=float(p_market),
                p_raw=float(p_raw),
                seconds_to_close=int(seconds_to_close),
                entry_bucket=int(entry_bucket),
                yes_cutoff=float(effective_cutoff),
                no_cutoff=float(1.0 - effective_cutoff),
                confidence=float(confidence),
                p_market_source="unknown",
            )
        if result.agreement_region in {"model_bullish", "model_bearish"}:
            result.reason += " [volatility override: mispricing allowed despite high reversal risk]"
    return result


def evaluate_live_signal(feature_dict: dict[str, Any]) -> SignalResult | None:
    """Evaluate live signal using persisted runtime settings and loaded model."""
    if not feature_dict:
        return None

    # Lazy import to avoid circular dependencies at module import time.
    from app.models import AppSettings

    risk_profile = (AppSettings.get("risk_profile", "moderate") or "moderate").strip().lower()
    logger.info("risk_profile from DB: '%s'", risk_profile)
    profile = RISK_PROFILES.get(risk_profile, RISK_PROFILES["moderate"])
    logger.info(
        "Profile loaded: %s, min=%s, max=%s, early=%s",
        risk_profile,
        profile.get("min_seconds"),
        profile.get("max_seconds"),
        profile.get("early_entry_enabled"),
    )

    enable_no_raw = AppSettings.get("enable_no_signals", "NOT FOUND")
    logger.info("enable_no raw value from DB: '%s'", enable_no_raw)
    enable_no = str(enable_no_raw).lower() == "true"
    yes_cutoff = float(profile["yes_cutoff"])
    no_cutoff = float(profile["no_cutoff"])
    min_seconds = int(profile["min_seconds"])
    max_seconds = int(profile["max_seconds"])
    early_entry_enabled = bool(profile.get("early_entry_enabled", False))
    early_entry_min_seconds = profile.get("early_entry_min_seconds")
    early_entry_max_seconds = profile.get("early_entry_max_seconds")
    early_entry_cutoff = profile.get("early_entry_cutoff")
    signal_mode = (AppSettings.get("signal_mode", "agreement") or "agreement").strip().lower()
    if signal_mode == "ensemble_vote":
        signal_mode = "ensemble"
    mispricing_threshold = float(AppSettings.get("mispricing_threshold", str(MISPRICING_THRESHOLD)) or MISPRICING_THRESHOLD)
    max_entry_yes = float(AppSettings.get("max_entry_price_yes", "0.85") or 0.85)
    max_entry_no = float(AppSettings.get("max_entry_price_no", "0.85") or 0.85)
    max_reversal = float(AppSettings.get("max_reversal_risk", "0.65") or 0.65)
    high_conviction_override = float(AppSettings.get("high_conviction_volatility_override", "0.80") or 0.80)

    try:
        p_raw = predict_proba_raw(feature_dict)
    except RuntimeError:
        return None

    p_market = float(feature_dict.get("p_market", 0.0) or 0.0)
    p_market_source = "candle_close"
    seconds_to_close = int(feature_dict.get("seconds_to_close", 0) or 0)
    entry_bucket = int(feature_dict.get("entry_bucket", 60) or 60)
    reversal_risk = float(feature_dict.get("reversal_risk", 0.0) or 0.0)
    confidence = abs(p_market - 0.5) + abs(p_raw - 0.5)
    volatility_guard_active = reversal_risk > max_reversal
    if volatility_guard_active:
        logger.info(
            "Volatility guard active: reversal_risk=%.3f > %.3f (mode=%s, confidence=%.3f, hc_override=%.3f)",
            reversal_risk,
            max_reversal,
            signal_mode,
            confidence,
            high_conviction_override,
        )
    if signal_mode == "mispricing":
        logger.info(
            "Live mispricing eval: threshold=%.3f, p_market=%.3f, p_raw=%.3f, early_enabled=%s, early_window=%s-%s, max_entry_yes=%.3f, max_entry_no=%.3f (NO side allowed regardless of enable_no=%s)",
            mispricing_threshold,
            p_market,
            p_raw,
            early_entry_enabled,
            early_entry_min_seconds,
            early_entry_max_seconds,
            max_entry_yes,
            max_entry_no,
            enable_no,
        )
        result = evaluate_mispricing_signal(
            p_market=p_market,
            p_raw=p_raw,
            seconds_to_close=seconds_to_close,
            entry_bucket=entry_bucket,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            mispricing_threshold=mispricing_threshold,
            early_entry_enabled=early_entry_enabled,
            early_entry_min_seconds=early_entry_min_seconds,
            early_entry_max_seconds=early_entry_max_seconds,
            early_entry_cutoff=early_entry_cutoff,
            max_entry_price_yes=max_entry_yes,
            max_entry_price_no=max_entry_no,
        )
    elif signal_mode == "ensemble":
        result = evaluate_ensemble_signal(
            p_market=p_market,
            p_raw=p_raw,
            seconds_to_close=seconds_to_close,
            entry_bucket=entry_bucket,
            yes_cutoff=profile["yes_cutoff"],
            max_entry_yes=max_entry_yes,
            max_entry_no=max_entry_no,
            mispricing_threshold=mispricing_threshold,
            min_seconds=profile["min_seconds"],
            max_seconds=profile["max_seconds"],
            early_entry_enabled=bool(profile.get("early_entry_enabled", False)),
            early_entry_min=int(profile.get("early_entry_min_seconds") or 300),
            early_entry_max=int(profile.get("early_entry_max_seconds") or 600),
            early_entry_cutoff=float(profile.get("early_entry_cutoff") or 0.80),
            volatility_guard_active=volatility_guard_active,
        )
    else:
        region = determine_agreement_region(p_market, p_raw, yes_cutoff, no_cutoff)
        logger.info(
            "Live signal eval: enable_no=%s, p_market=%.3f, p_raw=%.3f, region=%s",
            enable_no,
            p_market,
            p_raw,
            region,
        )
        result = evaluate_signal(
            p_market=p_market,
            p_raw=p_raw,
            seconds_to_close=seconds_to_close,
            entry_bucket=entry_bucket,
            yes_cutoff=yes_cutoff,
            no_cutoff=no_cutoff,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            enable_no_signals=enable_no,
            early_entry_enabled=early_entry_enabled,
            early_entry_min_seconds=early_entry_min_seconds,
            early_entry_max_seconds=early_entry_max_seconds,
            early_entry_cutoff=early_entry_cutoff,
            max_entry_price_yes=max_entry_yes,
            max_entry_price_no=max_entry_no,
            force_on_high_conviction=False,
            volatility_guard_active=volatility_guard_active,
        )
    result.p_market_source = p_market_source
    return result


def signal_to_dict(result: SignalResult) -> dict:
    """Convert SignalResult into JSON-safe dictionary with UI helpers."""
    payload = asdict(result)
    payload["is_actionable"] = result.signal != "NO SIGNAL"
    if result.signal == "PAPER BUY YES":
        payload["signal_color"] = "success"
    elif result.signal == "PAPER BUY NO":
        payload["signal_color"] = "danger"
    else:
        payload["signal_color"] = "neutral"
    return payload


if __name__ == "__main__":
    from app.feature_engineering import get_live_snapshot

    snapshot = get_live_snapshot()
    if snapshot is None:
        print("No active market")
    else:
        try:
            p_raw_value = predict_proba_raw(snapshot)
        except RuntimeError as exc:
            print(f"Model unavailable: {exc}")
        else:
            result = evaluate_signal(
                p_market=float(snapshot.get("p_market", snapshot.get("price_now", 0.0)) or 0.0),
                p_raw=float(p_raw_value),
                seconds_to_close=int(snapshot.get("seconds_to_close", 0) or 0),
                entry_bucket=int(snapshot.get("entry_bucket", 60) or 60),
                yes_cutoff=0.65,
                no_cutoff=0.35,
                min_seconds=30,
                max_seconds=180,
                enable_no_signals=False,
            )
            pprint(signal_to_dict(result))
