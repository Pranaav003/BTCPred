"""SQLAlchemy models for Kalshi Signal."""

from __future__ import annotations

from datetime import datetime, UTC

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import Index, text

from app.extensions import db


def utcnow() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(UTC)


class Market(db.Model):
    """Observed Kalshi BTC 15M market."""

    __tablename__ = "markets"

    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(64), unique=True, nullable=False)
    title = db.Column(db.String(256))
    series_ticker = db.Column(db.String(32), default="KXBTC15M")
    close_time = db.Column(db.DateTime, nullable=False)
    created_at = db.Column(db.DateTime, default=utcnow)
    resolved = db.Column(db.Boolean, default=False)
    final_outcome_yes = db.Column(db.Boolean, nullable=True)
    resolution_price = db.Column(db.Float, nullable=True)

    signals = db.relationship("Signal", back_populates="market", lazy=True, cascade="all, delete-orphan")
    paper_trades = db.relationship("PaperTrade", back_populates="market", lazy=True)


class TradeSnapshot(db.Model):
    """Frozen market state at paper trade entry (one per trade)."""

    __tablename__ = "trade_snapshots"

    id = db.Column(db.Integer, primary_key=True)
    trade_id = db.Column(
        db.Integer, db.ForeignKey("paper_trades.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    captured_at = db.Column(db.DateTime, default=utcnow, nullable=False)

    ticker = db.Column(db.String(64), nullable=False)
    market_title = db.Column(db.String(256), default="")
    seconds_to_close = db.Column(db.Integer, default=0)
    entry_bucket = db.Column(db.Integer, default=0)
    p_market = db.Column(db.Float, default=0.0)
    p_raw = db.Column(db.Float, default=0.0)
    signal_mode = db.Column(db.String(32), default="agreement")
    agreement_region = db.Column(db.String(32), default="")
    signal_reason = db.Column(db.String(256), default="")
    confidence = db.Column(db.Float, default=0.0)
    reversal_risk = db.Column(db.Float, default=0.0)
    mispricing_gap = db.Column(db.Float, default=0.0)

    btc_price = db.Column(db.Float, nullable=True)
    up_price_cents = db.Column(db.Integer, nullable=True)
    down_price_cents = db.Column(db.Integer, nullable=True)

    chart_history_json = db.Column(db.Text, default="[]")
    raw_features_json = db.Column(db.Text, default="{}")

    trade = db.relationship("PaperTrade", back_populates="trade_snapshot", lazy=True)


class Signal(db.Model):
    """One market snapshot/poll and derived trading signal."""

    __tablename__ = "signals"

    __table_args__ = (
        Index("ix_signals_market_id", "market_id"),
        Index(
            "ix_signals_resolved_market_has_features",
            "resolved",
            "market_id",
            postgresql_where=text("raw_features_json IS NOT NULL"),
            sqlite_where=text("raw_features_json IS NOT NULL"),
        ),
    )

    id = db.Column(db.Integer, primary_key=True)
    market_id = db.Column(db.Integer, db.ForeignKey("markets.id"), nullable=False)
    logged_at = db.Column(db.DateTime, default=utcnow)
    snapshot_ts = db.Column(db.Integer)
    seconds_to_close = db.Column(db.Integer)
    entry_bucket = db.Column(db.Integer)
    p_market = db.Column(db.Float)
    p_raw = db.Column(db.Float)
    orderbook_mid = db.Column(db.Float, nullable=True, default=None)
    orderbook_available = db.Column(db.Boolean, default=False)
    yes_cutoff = db.Column(db.Float)
    no_cutoff = db.Column(db.Float)
    signal = db.Column(db.String(32))
    reason = db.Column(db.String(128))
    agreement_region = db.Column(db.String(32))
    raw_features_json = db.Column(db.Text)

    resolved = db.Column(db.Boolean, default=False)
    pnl = db.Column(db.Float, nullable=True)
    outcome_correct = db.Column(db.Boolean, nullable=True)

    market = db.relationship("Market", back_populates="signals")
    paper_trades = db.relationship("PaperTrade", back_populates="signal", lazy=True)


class Portfolio(db.Model):
    """Paper trading portfolio state (single row, id=1)."""

    __tablename__ = "portfolios"

    id = db.Column(db.Integer, primary_key=True)
    starting_balance = db.Column(db.Float, default=100.0)
    cash = db.Column(db.Float, default=100.0)
    total_deposited = db.Column(db.Float, default=100.0)
    created_at = db.Column(db.DateTime, default=utcnow)
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow)

    paper_trades = db.relationship("PaperTrade", back_populates="portfolio", lazy=True, cascade="all, delete-orphan")

    @property
    def total_value(self) -> float:
        open_positions_value = sum(trade.current_value for trade in self.paper_trades if not trade.resolved)
        return float((self.cash or 0.0) + open_positions_value)

    @property
    def realized_pnl(self) -> float:
        return float(sum((trade.realized_pnl or 0.0) for trade in self.paper_trades if trade.resolved))

    @property
    def unrealized_pnl(self) -> float:
        return float(sum(trade.unrealized_pnl for trade in self.paper_trades if not trade.resolved))

    @property
    def total_return_pct(self) -> float:
        base = self.starting_balance or 0.0
        if base == 0:
            return 0.0
        return float(((self.total_value - base) / base) * 100.0)

    @classmethod
    def get_or_create(cls) -> "Portfolio":
        portfolio = cls.query.get(1) or cls.query.first()
        if portfolio is not None:
            return portfolio
        portfolio = cls(id=1, starting_balance=100.0, cash=100.0, total_deposited=100.0)
        db.session.add(portfolio)
        db.session.commit()
        return portfolio


class PaperTrade(db.Model):
    """Executed paper trade with optional exit on market resolution."""

    __tablename__ = "paper_trades"

    id = db.Column(db.Integer, primary_key=True)
    portfolio_id = db.Column(db.Integer, db.ForeignKey("portfolios.id"), nullable=False)
    market_id = db.Column(db.Integer, db.ForeignKey("markets.id"), nullable=True)
    ticker = db.Column(db.String(64))
    side = db.Column(db.String(4))
    contracts = db.Column(db.Float)
    entry_price = db.Column(db.Float)
    entry_cost = db.Column(db.Float)
    entry_at = db.Column(db.DateTime, default=utcnow)
    signal_triggered = db.Column(db.Boolean, default=False)
    signal_id = db.Column(db.Integer, db.ForeignKey("signals.id"), nullable=True)

    exit_price = db.Column(db.Float, nullable=True)
    exit_at = db.Column(db.DateTime, nullable=True)
    realized_pnl = db.Column(db.Float, nullable=True)
    outcome_correct = db.Column(db.Boolean, nullable=True)
    resolved = db.Column(db.Boolean, default=False)

    portfolio = db.relationship("Portfolio", back_populates="paper_trades")
    market = db.relationship("Market", back_populates="paper_trades")
    signal = db.relationship("Signal", back_populates="paper_trades")
    trade_snapshot = db.relationship(
        "TradeSnapshot", back_populates="trade", uselist=False, cascade="all, delete-orphan", lazy=True
    )

    @property
    def current_value(self) -> float:
        if self.resolved:
            return float((self.exit_price or 0.0) * (self.contracts or 0.0))
        return float((self.contracts or 0.0) * (self.entry_price or 0.0))

    @property
    def unrealized_pnl(self) -> float:
        return float(self.current_value - (self.entry_cost or 0.0))

    @property
    def pnl_display(self) -> str:
        if self.realized_pnl is None:
            return "--"
        sign = "+" if self.realized_pnl >= 0 else "-"
        return f"{sign}${abs(self.realized_pnl):.3f}"

    @classmethod
    def has_recent_auto_trade(cls, ticker: str, side: str, minutes: int = 20) -> bool:
        """True if an auto trade exists for this ticker+side within the lookback window."""
        from datetime import timedelta

        if not ticker or not side:
            return False
        normalized = str(side).upper()
        if normalized not in {"YES", "NO"}:
            return False
        cutoff = utcnow() - timedelta(minutes=minutes)
        return (
            cls.query.filter(
                cls.ticker == ticker,
                cls.signal_triggered.is_(True),
                cls.side == normalized,
                cls.entry_at >= cutoff,
            ).count()
            > 0
        )


class LiveTrade(db.Model):
    """Real Kalshi order placed alongside paper trading."""

    __tablename__ = "live_trades"

    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(64), nullable=False, index=True)
    side = db.Column(db.String(4), nullable=False)
    contracts = db.Column(db.Float, nullable=False)
    entry_price = db.Column(db.Float, nullable=False)
    entry_price_cents = db.Column(db.Integer, nullable=False)
    cost_dollars = db.Column(db.Float, nullable=False)
    kalshi_order_id = db.Column(db.String(128), nullable=True)
    order_status = db.Column(db.String(32), default="placed")
    signal_id = db.Column(db.Integer, db.ForeignKey("signals.id"), nullable=True)
    entry_at = db.Column(db.DateTime, default=utcnow)
    resolved = db.Column(db.Boolean, default=False)
    exit_price = db.Column(db.Float, nullable=True)
    realized_pnl = db.Column(db.Float, nullable=True)
    outcome = db.Column(db.String(16), nullable=True)
    resolved_at = db.Column(db.DateTime, nullable=True)
    p_market_at_entry = db.Column(db.Float, nullable=True)
    p_raw_at_entry = db.Column(db.Float, nullable=True)
    agreement_region = db.Column(db.String(32), nullable=True)
    live_trade_size_setting = db.Column(db.Float, nullable=True)
    error_detail = db.Column(db.String(512), nullable=True)

    signal = db.relationship("Signal", backref="live_trades", lazy=True)


class AppSettings(db.Model):
    """Simple key-value runtime settings store."""

    __tablename__ = "app_settings"

    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(64), unique=True, nullable=False)
    value = db.Column(db.String(256))
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow)


class ModelArtifact(db.Model):
    """Stored model .pkl binary — survives Render deploys."""

    __tablename__ = "model_artifacts"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True, nullable=False, default="default")
    data = db.Column(db.LargeBinary, nullable=False)
    uploaded_at = db.Column(db.DateTime, default=utcnow)
    size_bytes = db.Column(db.Integer, nullable=True)
    model_type = db.Column(db.String(64), nullable=True)
    accuracy = db.Column(db.Float, nullable=True)
