"""JSON API routes."""

import csv
import io
import json
import os
import shutil
import zipfile
import tempfile
from pathlib import Path
from datetime import UTC, datetime

from flask import Blueprint, Response, jsonify, request, send_file, stream_with_context
import sklearn
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError

from app.db_helpers import export_training_data, get_probability_history, get_recent_signals, get_signal_metrics
from app.feature_engineering import get_live_snapshot
from app.kalshi_client import get_active_market, get_btc_price, get_market_prices
from app.model_loader import get_model
from app.models import AppSettings, Market, PaperTrade, Signal, TradeSnapshot, db
from app.paper_trading import (
    execute_paper_trade,
    get_open_positions,
    get_portfolio_summary,
    get_trade_history,
    position_sizing_breakdown,
    reset_portfolio,
)
from app.resolver import get_resolution_summary, resolve_pending_markets
from app.signal_engine import (
    MISPRICING_THRESHOLD,
    PROFILE_OVERRIDE_FIELDS,
    RISK_PROFILES,
    evaluate_mispricing_signal,
    get_profile,
)

api_bp = Blueprint("api", __name__, url_prefix="/api")


def _utc_iso_z(value: datetime | None) -> str | None:
    """Serialize datetime as UTC ISO string with trailing Z."""
    if value is None:
        return None
    if value.tzinfo is None:
        return f"{value.isoformat()}Z"
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _serialize_row(row):
    payload = {}
    for column in row.__table__.columns:
        value = getattr(row, column.name)
        if isinstance(value, datetime):
            payload[column.name] = _utc_iso_z(value)
        else:
            payload[column.name] = value
    return payload


def _csv_string(rows):
    all_keys = sorted({key for row in rows for key in row.keys()})
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=all_keys, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return output.getvalue()


def _query_row_keys(query, serialize_fn, batch: int = 500) -> list[str]:
    """Collect sorted union of dict keys for CSV export without loading all rows."""
    keys: set[str] = set()
    for row in query.yield_per(batch):
        keys.update(serialize_fn(row).keys())
    return sorted(keys)


def _write_query_csv_stream(query, path: str, fieldnames: list[str], serialize_fn, batch: int = 500) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", restval="")
        writer.writeheader()
        for row in query.yield_per(batch):
            writer.writerow(serialize_fn(row))


_TRAINING_CSV_META_KEYS = (
    "logged_at",
    "ticker",
    "p_market",
    "p_raw",
    "signal",
    "agreement_region",
    "resolved",
    "pnl",
    "outcome_correct",
    "entry_bucket",
)


def _training_csv_collect_fieldnames() -> list[str]:
    keys: set[str] = set(_TRAINING_CSV_META_KEYS)
    q = Signal.query.join(Market, Signal.market_id == Market.id).order_by(Signal.id.asc())
    for row in q.yield_per(500):
        if not row.raw_features_json:
            continue
        try:
            raw = json.loads(row.raw_features_json)
        except json.JSONDecodeError:
            continue
        if isinstance(raw, dict):
            keys.update(raw.keys())
    return sorted(keys)


def _merged_training_row(row) -> dict[str, object]:
    try:
        raw_features = json.loads(row.raw_features_json) if row.raw_features_json else {}
    except json.JSONDecodeError:
        raw_features = {}
    merged: dict[str, object] = dict(raw_features) if isinstance(raw_features, dict) else {}
    merged.update(
        {
            "logged_at": _utc_iso_z(row.logged_at),
            "ticker": row.market.ticker if row.market else None,
            "p_market": row.p_market,
            "p_raw": row.p_raw,
            "signal": row.signal,
            "agreement_region": row.agreement_region,
            "resolved": row.resolved,
            "pnl": row.pnl,
            "outcome_correct": row.outcome_correct,
            "entry_bucket": row.entry_bucket,
        }
    )
    return merged


@api_bp.route("/health")
def health():
    return jsonify({"status": "ok", "message": "service healthy"})


@api_bp.route("/debug/market", methods=["GET"])
def debug_market():
    market = get_active_market()
    if not market:
        return jsonify({"error": "no active market"}), 404
    return jsonify(market)


@api_bp.route("/market-prices", methods=["GET"])
def market_prices():
    market = get_active_market()
    ticker = market.get("ticker") if isinstance(market, dict) else None
    btc_price = get_btc_price()
    quote = get_market_prices(ticker) if ticker else None

    def _cents(value):
        if value is None:
            return None
        try:
            return int(round(float(value) * 100))
        except Exception:
            return None

    yes_ask = quote.get("yes_ask") if quote else None
    no_ask = quote.get("no_ask") if quote else None
    up_cents = _cents(yes_ask)
    down_cents = _cents(no_ask)
    return jsonify(
        {
            "btc_price": btc_price,
            "btc_price_formatted": f"${float(btc_price):,.2f}" if btc_price is not None else "--",
            "yes_bid": quote.get("yes_bid") if quote else None,
            "yes_ask": yes_ask,
            "no_bid": quote.get("no_bid") if quote else None,
            "no_ask": no_ask,
            "up_price_cents": up_cents,
            "down_price_cents": down_cents,
            "up_display": f"Up {up_cents}¢" if up_cents is not None else "Up --",
            "down_display": f"Down {down_cents}¢" if down_cents is not None else "Down --",
            "volume": quote.get("volume") if quote else None,
            "ticker": quote.get("ticker") if quote else ticker,
        }
    )


@api_bp.route("/signals")
def signals():
    limit_param = request.args.get("limit", default=100, type=int) or 100
    limit_param = max(1, min(limit_param, 500))
    results = get_recent_signals(limit=limit_param)
    return jsonify({"signals": results, "count": len(results)})


@api_bp.route("/signals/history")
def signals_history():
    limit_param = request.args.get("limit", default=50, type=int) or 50
    limit_param = max(1, min(limit_param, 500))
    results = get_probability_history(limit=limit_param)
    return jsonify({"history": results})


@api_bp.route("/metrics")
def metrics():
    metrics_payload = get_signal_metrics()
    resolution_payload = get_resolution_summary()

    yes_resolved = Signal.query.filter(
        Signal.signal == "PAPER BUY YES",
        Signal.resolved.is_(True),
    )
    no_resolved = Signal.query.filter(
        Signal.signal == "PAPER BUY NO",
        Signal.resolved.is_(True),
    )

    yes_count = yes_resolved.count()
    no_count = no_resolved.count()
    yes_correct = yes_resolved.filter(Signal.outcome_correct.is_(True)).count()
    no_correct = no_resolved.filter(Signal.outcome_correct.is_(True)).count()

    yes_agg = yes_resolved.with_entities(func.avg(Signal.pnl)).first()
    no_agg = no_resolved.with_entities(func.avg(Signal.pnl)).first()

    payload = {
        **metrics_payload,
        **resolution_payload,
        "yes_accuracy": (yes_correct / yes_count) if yes_count else None,
        "no_accuracy": (no_correct / no_count) if no_count else None,
        "avg_pnl_yes": float(yes_agg[0]) if yes_agg and yes_agg[0] is not None else None,
        "avg_pnl_no": float(no_agg[0]) if no_agg and no_agg[0] is not None else None,
    }
    return jsonify(payload)


@api_bp.route("/model-info")
def model_info():
    try:
        bundle = get_model()
    except RuntimeError:
        return jsonify({"loaded": False})

    metrics = bundle.get("test_metrics", {})
    return jsonify(
        {
            "loaded": True,
            "model_type": bundle.get("model_type"),
            "trained_at": bundle.get("trained_at"),
            "test_metrics": {
                "accuracy": metrics.get("accuracy"),
                "roc_auc": metrics.get("roc_auc"),
                "brier": metrics.get("brier", metrics.get("brier_score")),
                "log_loss": metrics.get("log_loss"),
            },
            "n_train": bundle.get("n_train"),
            "n_test": bundle.get("n_test"),
            "feature_count": len(bundle.get("features", [])),
            "sklearn_version_saved": bundle.get("sklearn_version"),
            "sklearn_version_current": sklearn.__version__,
            "version_mismatch": bundle.get("sklearn_version") != sklearn.__version__,
        }
    )


@api_bp.route("/scheduler/start", methods=["POST"])
def scheduler_start():
    try:
        AppSettings.set("scheduler_running", "true")
        return jsonify({"status": "started"})
    except SQLAlchemyError:
        return jsonify({"status": "error"}), 500


@api_bp.route("/scheduler/stop", methods=["POST"])
def scheduler_stop():
    try:
        AppSettings.set("scheduler_running", "false")
        return jsonify({"status": "stopped"})
    except SQLAlchemyError:
        return jsonify({"status": "error"}), 500


@api_bp.route("/scheduler/status", methods=["GET"])
def scheduler_status():
    running = AppSettings.get("scheduler_running", "false") == "true"
    poll_interval = int(AppSettings.get("poll_interval_seconds", "30"))
    auto_trade_enabled = AppSettings.get("auto_trade_enabled", "false") == "true"
    return jsonify(
        {
            "running": running,
            "poll_interval": poll_interval,
            "auto_trade_enabled": auto_trade_enabled,
        }
    )


@api_bp.route("/resolution/summary", methods=["GET"])
def resolution_summary():
    return jsonify(get_resolution_summary())


@api_bp.route("/resolution/trigger", methods=["POST"])
def resolution_trigger():
    resolved_count = resolve_pending_markets()
    return jsonify({"resolved_count": resolved_count})


@api_bp.route("/signals/<int:signal_id>", methods=["GET"])
def signal_detail(signal_id: int):
    signal = Signal.query.get(signal_id)
    if signal is None:
        return jsonify({"error": "signal not found"}), 404

    market = signal.market
    try:
        raw_features = json.loads(signal.raw_features_json) if signal.raw_features_json else {}
    except json.JSONDecodeError:
        raw_features = {}

    return jsonify(
        {
            "id": signal.id,
            "market_id": signal.market_id,
            "ticker": market.ticker if market else None,
            "market_title": market.title if market else None,
            "close_time": _utc_iso_z(market.close_time) if market else None,
            "resolution_price": market.resolution_price if market else None,
            "outcome_yes": market.final_outcome_yes if market else None,
            "logged_at": _utc_iso_z(signal.logged_at),
            "snapshot_ts": signal.snapshot_ts,
            "seconds_to_close": signal.seconds_to_close,
            "entry_bucket": signal.entry_bucket,
            "p_market": signal.p_market,
            "p_raw": signal.p_raw,
            "yes_cutoff": signal.yes_cutoff,
            "no_cutoff": signal.no_cutoff,
            "signal": signal.signal,
            "reason": signal.reason,
            "agreement_region": signal.agreement_region,
            "raw_features": raw_features,
            "resolved": signal.resolved,
            "pnl": signal.pnl,
            "outcome_correct": signal.outcome_correct,
        }
    )


@api_bp.route("/analytics/pnl-curve", methods=["GET"])
def analytics_pnl_curve():
    # Narrow SELECT only — avoids loading raw_features_json (megabytes × row count → OOM on 512MB).
    skinny = (
        db.session.query(
            Signal.logged_at,
            Signal.pnl,
            Signal.signal,
            Signal.outcome_correct,
            Market.ticker,
        )
        .join(Market, Signal.market_id == Market.id)
        .filter(
            Signal.resolved.is_(True),
            Signal.signal != "NO SIGNAL",
        )
        .order_by(Signal.logged_at.asc())
    )

    curve = []
    running = 0.0
    for logged_at, pnl_v, sig, corr, ticker in skinny:
        pnl = float(pnl_v or 0.0)
        running += pnl
        curve.append(
            {
                "logged_at": _utc_iso_z(logged_at),
                "ticker": ticker,
                "signal": sig,
                "pnl": pnl_v,
                "cumulative_pnl": running,
                "outcome_correct": corr,
            }
        )
    return jsonify({"curve": curve})


@api_bp.route("/analytics/accuracy-by-bucket", methods=["GET"])
def analytics_accuracy_by_bucket():
    buckets = []
    for bucket in [60, 120, 180, 300]:
        query = Signal.query.filter(
            Signal.resolved.is_(True),
            Signal.signal != "NO SIGNAL",
            Signal.entry_bucket == bucket,
        )
        count = query.count()
        correct = query.filter(Signal.outcome_correct.is_(True)).count()
        aggregate = query.with_entities(func.avg(Signal.pnl), func.sum(Signal.pnl)).first()
        avg_pnl = float(aggregate[0]) if aggregate and aggregate[0] is not None else None
        total_pnl = float(aggregate[1]) if aggregate and aggregate[1] is not None else None
        buckets.append(
            {
                "entry_bucket": bucket,
                "count": count,
                "correct": correct,
                "accuracy": (correct / count) if count else None,
                "avg_pnl": avg_pnl,
                "total_pnl": total_pnl,
            }
        )
    return jsonify({"buckets": buckets})


@api_bp.route("/analytics/accuracy-by-cutoff", methods=["GET"])
def analytics_accuracy_by_cutoff():
    rows = (
        db.session.query(
            Signal.p_market,
            Signal.p_raw,
            Market.final_outcome_yes,
        )
        .join(Market, Signal.market_id == Market.id)
        .filter(Signal.resolved.is_(True), Market.final_outcome_yes.is_not(None))
        .all()
    )

    cutoffs_data = []
    for cutoff in [0.55, 0.60, 0.65, 0.70, 0.75]:
        fired = []
        for p_market, p_raw, final_yes in rows:
            pm = p_market if p_market is not None else 0.0
            pr = p_raw if p_raw is not None else 0.0
            if pm >= cutoff and pr >= cutoff:
                outcome_yes = bool(final_yes)
                pnl = (1.0 if outcome_yes else 0.0) - pm
                fired.append({"outcome_yes": outcome_yes, "pnl": pnl})

        count = len(fired)
        if count:
            correct = sum(1 for item in fired if item["outcome_yes"])
            total_pnl = float(sum(item["pnl"] for item in fired))
            avg_pnl = total_pnl / count
            accuracy = correct / count
        else:
            avg_pnl = None
            total_pnl = None
            accuracy = None

        cutoffs_data.append(
            {
                "cutoff": cutoff,
                "count": count,
                "accuracy": accuracy,
                "avg_pnl": avg_pnl,
                "total_pnl": total_pnl,
            }
        )
    return jsonify({"cutoffs": cutoffs_data})


@api_bp.route("/analytics/agreement-regions", methods=["GET"])
def analytics_agreement_regions():
    regions = []
    for region in [
        "agree_yes",
        "agree_no",
        "model_bullish",
        "model_bearish",
        "market_yes_raw_no",
        "market_no_raw_yes",
        "no_agreement",
        "outside_time_window",
    ]:
        all_query = Signal.query.filter(Signal.agreement_region == region)
        resolved_query = all_query.filter(Signal.resolved.is_(True))

        count = all_query.count()
        resolved_count = resolved_query.count()
        correct = resolved_query.filter(Signal.outcome_correct.is_(True)).count()
        aggregates = resolved_query.with_entities(func.avg(Signal.pnl), func.sum(Signal.pnl)).first()
        avg_pnl = float(aggregates[0]) if aggregates and aggregates[0] is not None else None
        total_pnl = float(aggregates[1]) if aggregates and aggregates[1] is not None else None

        regions.append(
            {
                "agreement_region": region,
                "count": count,
                "resolved_count": resolved_count,
                "accuracy": (correct / resolved_count) if resolved_count else None,
                "avg_pnl": avg_pnl,
                "total_pnl": total_pnl,
            }
        )
    return jsonify({"regions": regions})


@api_bp.route("/analytics/mispricing-backtest", methods=["GET"])
def analytics_mispricing_backtest():
    threshold = float(AppSettings.get("mispricing_threshold", str(MISPRICING_THRESHOLD)) or MISPRICING_THRESHOLD)
    rows = (
        db.session.query(
            Signal.p_market,
            Signal.p_raw,
            Signal.seconds_to_close,
            Signal.entry_bucket,
            Market.final_outcome_yes,
        )
        .join(Market, Signal.market_id == Market.id)
        .filter(
            Signal.resolved.is_(True),
            Signal.p_market.is_not(None),
            Signal.p_raw.is_not(None),
            Market.final_outcome_yes.is_not(None),
        )
        .all()
    )

    bucket_defs = [
        ("0.10-0.15", 0.10, 0.15),
        ("0.15-0.20", 0.15, 0.20),
        ("0.20+", 0.20, None),
    ]
    stats = {label: {"count": 0, "correct": 0, "pnl_sum": 0.0} for label, _, _ in bucket_defs}

    for p_market, p_raw, sec_close, eb, final_yes in rows:
        pm = float(p_market)
        pr = float(p_raw)
        gap_abs = abs(pr - pm)
        bucket = None
        for label, lower, upper in bucket_defs:
            if gap_abs >= lower and (upper is None or gap_abs < upper):
                bucket = label
                break
        if bucket is None:
            continue

        result = evaluate_mispricing_signal(
            p_market=pm,
            p_raw=pr,
            seconds_to_close=int(sec_close or 0),
            entry_bucket=int(eb or 60),
            min_seconds=0,
            max_seconds=10_000,
            mispricing_threshold=threshold,
        )
        if result.signal == "NO SIGNAL":
            continue

        outcome_yes = bool(final_yes)
        if result.signal == "PAPER BUY YES":
            pnl = (1.0 - pm) if outcome_yes else -pm
            correct = outcome_yes
        else:
            pnl = pm if not outcome_yes else -(1.0 - pm)
            correct = not outcome_yes

        stats[bucket]["count"] += 1
        stats[bucket]["correct"] += 1 if correct else 0
        stats[bucket]["pnl_sum"] += float(pnl)

    buckets = []
    for label, _, _ in bucket_defs:
        count = stats[label]["count"]
        pnl_sum = stats[label]["pnl_sum"]
        buckets.append(
            {
                "bucket": label,
                "count": count,
                "accuracy": (stats[label]["correct"] / count) if count else None,
                "avg_pnl": (pnl_sum / count) if count else None,
            }
        )
    return jsonify({"threshold": threshold, "buckets": buckets})


@api_bp.route("/export/full", methods=["GET"])
def export_full():
    """Stream JSON to avoid loading all rows + full serialized string in memory (OOM on Render)."""
    now = datetime.now(UTC)
    model_info: dict[str, object] = {"loaded": False}
    try:
        bundle = get_model()
        model_info = {
            "loaded": True,
            "trained_at": bundle.get("trained_at"),
            "test_metrics": bundle.get("test_metrics", {}),
            "n_train": bundle.get("n_train"),
            "n_test": bundle.get("n_test"),
            "features": bundle.get("features", []),
        }
    except RuntimeError:
        pass

    portfolio = get_portfolio_summary()
    settings = {row.key: row.value for row in AppSettings.query.order_by(AppSettings.key.asc()).all()}
    _json_opts = {"ensure_ascii": False, "separators": (",", ":")}

    def generate():
        yield "{"
        yield '"signals":['
        first = True
        for row in Signal.query.order_by(Signal.id.asc()).yield_per(500):
            if not first:
                yield ","
            first = False
            yield json.dumps(_serialize_row(row), **_json_opts)
        yield '],"trades":['
        first = True
        for row in PaperTrade.query.order_by(PaperTrade.id.asc()).yield_per(500):
            if not first:
                yield ","
            first = False
            yield json.dumps(_serialize_row(row), **_json_opts)
        yield '],"markets":['
        first = True
        for row in Market.query.order_by(Market.id.asc()).yield_per(500):
            if not first:
                yield ","
            first = False
            yield json.dumps(_serialize_row(row), **_json_opts)
        yield '],"portfolio":'
        yield json.dumps(portfolio, **_json_opts)
        yield ',"settings":'
        yield json.dumps(settings, **_json_opts)
        yield ',"model_info":'
        yield json.dumps(model_info, **_json_opts)
        yield ',"exported_at":'
        yield json.dumps(_utc_iso_z(now), **_json_opts)
        yield "}"

    response = Response(stream_with_context(generate()), mimetype="application/json")
    response.headers["Content-Disposition"] = f'attachment; filename="kalshi_signal_export_{now.strftime("%Y%m%d")}.json"'
    return response


@api_bp.route("/export/csv", methods=["GET"])
def export_csv():
    """Write CSVs to disk then zip — avoids holding full tables in RAM."""
    now = datetime.now(UTC)
    tmpd = tempfile.mkdtemp(prefix="export_csv_")
    zip_path: str | None = None
    try:
        signals_path = os.path.join(tmpd, "signals.csv")
        trades_path = os.path.join(tmpd, "trades.csv")
        sig_keys = _query_row_keys(Signal.query.order_by(Signal.id.asc()), _serialize_row)
        tr_keys = _query_row_keys(PaperTrade.query.order_by(PaperTrade.id.asc()), _serialize_row)
        _write_query_csv_stream(Signal.query.order_by(Signal.id.asc()), signals_path, sig_keys, _serialize_row)
        _write_query_csv_stream(PaperTrade.query.order_by(PaperTrade.id.asc()), trades_path, tr_keys, _serialize_row)
        fd, zip_path = tempfile.mkstemp(prefix="kalshi_data_", suffix=".zip")
        os.close(fd)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(signals_path, "signals.csv")
            zf.write(trades_path, "trades.csv")
    except Exception:
        if zip_path and os.path.isfile(zip_path):
            try:
                os.unlink(zip_path)
            except OSError:
                pass
        raise
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)

    if not zip_path or not os.path.isfile(zip_path):
        return jsonify({"error": "export failed"}), 500

    response = send_file(
        zip_path,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"kalshi_data_{now.strftime('%Y%m%d')}.zip",
    )

    def _cleanup_zip() -> None:
        try:
            if zip_path and os.path.isfile(zip_path):
                os.unlink(zip_path)
        except OSError:
            pass

    response.call_on_close(_cleanup_zip)
    return response


@api_bp.route("/export/training-csv", methods=["GET"])
def export_training_csv():
    """Two-pass training CSV: stable header, bounded memory."""
    now = datetime.now(UTC)
    fieldnames = _training_csv_collect_fieldnames()
    fd, out_path = tempfile.mkstemp(prefix="kalshi_training_", suffix=".csv")
    os.close(fd)
    try:
        with open(out_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", restval="")
            writer.writeheader()
            q = Signal.query.join(Market, Signal.market_id == Market.id).order_by(Signal.id.asc())
            for row in q.yield_per(500):
                writer.writerow(_merged_training_row(row))
    except Exception:
        try:
            os.unlink(out_path)
        except OSError:
            pass
        raise

    response = send_file(
        out_path,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"kalshi_training_data_{now.strftime('%Y%m%d')}.csv",
    )

    def _cleanup_training() -> None:
        try:
            if os.path.isfile(out_path):
                os.unlink(out_path)
        except OSError:
            pass

    response.call_on_close(_cleanup_training)
    return response


@api_bp.route("/export/live-training-data", methods=["GET"])
def export_live_training_data():
    now = datetime.now(UTC)
    if request.args.get("stats", "").strip() == "1":
        # Fast path: COUNT only — do not stream/export the full table (avoids timeout on large DBs).
        total_candidates = (
            db.session.query(func.count(Signal.id))
            .filter(
                Signal.resolved.is_(True),
                Signal.raw_features_json.isnot(None),
            )
            .scalar()
        )
        total_candidates = int(total_candidates or 0)
        rows_export = (
            db.session.query(func.count(Signal.id))
            .join(Market, Signal.market_id == Market.id)
            .filter(
                Signal.resolved.is_(True),
                Signal.raw_features_json.isnot(None),
                Market.final_outcome_yes.isnot(None),
            )
            .scalar()
        )
        rows_export = int(rows_export or 0)
        skipped = max(0, total_candidates - rows_export)
        return jsonify({"rows": rows_export, "skipped": skipped})

    with tempfile.NamedTemporaryFile(prefix="live_training_data_", suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
    rows, skipped = export_training_data(tmp_path)
    if rows == 0:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
        return jsonify({"error": "No live resolved rows available yet.", "rows": 0, "skipped": skipped}), 404
    response = send_file(
        tmp_path,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"live_training_data_{now.strftime('%Y%m%d')}.csv",
    )
    response.headers["X-Live-Rows"] = str(rows)
    response.headers["X-Live-Skipped"] = str(skipped)

    def _cleanup_live() -> None:
        try:
            if os.path.isfile(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass

    response.call_on_close(_cleanup_live)
    return response


@api_bp.route("/settings", methods=["GET"])
def get_settings():
    settings = {row.key: row.value for row in AppSettings.query.all()}
    profile_key = settings.get("risk_profile", "moderate") or "moderate"
    profile = get_profile(profile_key)
    profile_yes_cutoff = float(profile.get("yes_cutoff", 0.65))
    effective_yes_cutoff = profile_yes_cutoff
    settings["profile_yes_cutoff"] = f"{profile_yes_cutoff:.4f}"
    settings["effective_yes_cutoff"] = f"{effective_yes_cutoff:.4f}"
    # Dashboard must use the same min/max seconds as evaluate_live_signal (risk profile merge),
    # not stale AppSettings.min_seconds_to_close / max_seconds_to_close from older seeds.
    settings["effective_min_seconds_to_close"] = str(int(profile.get("min_seconds", 60)))
    settings["effective_max_seconds_to_close"] = str(int(profile.get("max_seconds", 180)))
    return jsonify(settings)


@api_bp.route("/settings", methods=["POST"])
def update_settings():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"updated": [], "errors": ["Request body must be a JSON object."]}), 400

    allowed_keys = {
        "yes_cutoff",
        "no_cutoff",
        "min_seconds_to_close",
        "max_seconds_to_close",
        "poll_interval_seconds",
        "enable_no_signals",
        "auto_trade_enabled",
        "paper_trading_enabled",
        "paper_trade_size",
        "dynamic_sizing_enabled",
        "risk_profile",
        "signal_mode",
        "mispricing_threshold",
        "max_entry_price_yes",
        "max_entry_price_no",
        "min_expected_profit",
        "max_reversal_risk",
        "max_daily_loss",
        "high_conviction_volatility_override",
    }

    updated = []
    errors = []
    boolean_keys = {
        "enable_no_signals",
        "auto_trade_enabled",
        "paper_trading_enabled",
        "dynamic_sizing_enabled",
    }
    for key, value in payload.items():
        if key not in allowed_keys:
            errors.append(f"Disallowed key: {key}")
            continue
        if key in boolean_keys:
            normalized = str(value).lower()
            normalized = "true" if normalized in {"true", "1", "yes", "on"} else "false"
            AppSettings.set(key, normalized)
        elif key == "signal_mode":
            mode = (str(value).strip().lower() or "agreement")
            if mode == "ensemble_vote":
                mode = "ensemble"
            AppSettings.set(key, mode if mode in {"agreement", "mispricing", "ensemble"} else "agreement")
        elif key == "mispricing_threshold":
            try:
                threshold_val = float(value)
            except (TypeError, ValueError):
                errors.append("mispricing_threshold must be numeric")
                continue
            threshold_val = max(0.05, min(0.30, threshold_val))
            AppSettings.set(key, f"{threshold_val:.4f}")
        elif key in {"max_entry_price_yes", "max_entry_price_no"}:
            try:
                max_entry = float(value)
            except (TypeError, ValueError):
                errors.append(f"{key} must be numeric")
                continue
            max_entry = max(0.55, min(1.0, max_entry))
            AppSettings.set(key, f"{max_entry:.4f}")
        elif key == "min_expected_profit":
            try:
                min_profit = float(value)
            except (TypeError, ValueError):
                errors.append("min_expected_profit must be numeric")
                continue
            min_profit = max(0.0, min(1.0, min_profit))
            AppSettings.set(key, f"{min_profit:.4f}")
        elif key == "max_reversal_risk":
            try:
                max_reversal = float(value)
            except (TypeError, ValueError):
                errors.append("max_reversal_risk must be numeric")
                continue
            max_reversal = max(0.20, min(1.0, max_reversal))
            AppSettings.set(key, f"{max_reversal:.4f}")
        elif key == "max_daily_loss":
            try:
                max_loss = float(value)
            except (TypeError, ValueError):
                errors.append("max_daily_loss must be numeric")
                continue
            max_loss = max(1.0, min(1_000_000.0, max_loss))
            AppSettings.set(key, f"{max_loss:.2f}")
        elif key == "high_conviction_volatility_override":
            try:
                override_cutoff = float(value)
            except (TypeError, ValueError):
                errors.append("high_conviction_volatility_override must be numeric")
                continue
            override_cutoff = max(0.60, min(1.0, override_cutoff))
            AppSettings.set(key, f"{override_cutoff:.4f}")
        else:
            AppSettings.set(key, str(value))
        updated.append(key)

    return jsonify({"updated": updated, "errors": errors})


@api_bp.route("/risk-profiles", methods=["GET"])
def risk_profiles():
    active = AppSettings.get("risk_profile", "moderate") or "moderate"
    if active not in RISK_PROFILES:
        active = "moderate"
    profile_meta = {
        "conservative": {"recommended": False, "validated": True},
        "moderate": {"recommended": True, "validated": True},
        "aggressive": {"recommended": False, "validated": True},
        "high_conviction": {"recommended": False, "validated": True},
    }
    enriched = {}
    for name, _profile in RISK_PROFILES.items():
        merged = {**get_profile(name)}
        merged.update(profile_meta.get(name, {"recommended": False, "validated": False}))
        customized_fields: list[str] = []
        for field in PROFILE_OVERRIDE_FIELDS:
            if AppSettings.get(f"profile_override_{name}_{field}") is not None:
                customized_fields.append(field)
        merged["customized"] = bool(customized_fields)
        merged["customized_fields"] = customized_fields
        enriched[name] = merged
    return jsonify({"profiles": enriched, "active": active})


@api_bp.route("/risk-profiles/<profile_name>", methods=["POST"])
def save_risk_profile(profile_name: str):
    profile_name = (profile_name or "").strip().lower()
    if profile_name not in RISK_PROFILES:
        return jsonify({"error": "Unknown profile"}), 404
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "Body must be a JSON object"}), 400

    try:
        yes_cutoff = float(payload.get("yes_cutoff", 0.65))
        no_cutoff = float(payload.get("no_cutoff", 0.35))
        min_seconds = int(payload.get("min_seconds", 60))
        max_seconds = int(payload.get("max_seconds", 300))
    except (TypeError, ValueError):
        return jsonify({"error": "yes_cutoff/no_cutoff/min_seconds/max_seconds must be numeric"}), 400

    early_raw = str(payload.get("early_entry_enabled", False)).strip().lower()
    early_entry_enabled = early_raw in {"true", "1", "yes", "on"}
    early_min = payload.get("early_entry_min_seconds")
    early_max = payload.get("early_entry_max_seconds")
    early_cutoff = payload.get("early_entry_cutoff")
    description = payload.get("description")

    AppSettings.set(f"profile_override_{profile_name}_yes_cutoff", f"{max(0.50, min(0.90, yes_cutoff)):.4f}")
    AppSettings.set(f"profile_override_{profile_name}_no_cutoff", f"{max(0.10, min(0.50, no_cutoff)):.4f}")
    AppSettings.set(f"profile_override_{profile_name}_min_seconds", str(max(0, min(300, min_seconds))))
    AppSettings.set(f"profile_override_{profile_name}_max_seconds", str(max(60, min(900, max_seconds))))
    AppSettings.set(f"profile_override_{profile_name}_early_entry_enabled", "true" if early_entry_enabled else "false")

    if early_entry_enabled:
        try:
            early_min_int = int(early_min if early_min is not None else 0)
            early_max_int = int(early_max if early_max is not None else 0)
            early_cutoff_float = float(early_cutoff if early_cutoff is not None else yes_cutoff)
        except (TypeError, ValueError):
            return jsonify({"error": "Early entry fields must be numeric when early entry is enabled"}), 400
        AppSettings.set(
            f"profile_override_{profile_name}_early_entry_min_seconds",
            str(max(0, min(600, early_min_int))),
        )
        AppSettings.set(
            f"profile_override_{profile_name}_early_entry_max_seconds",
            str(max(0, min(900, early_max_int))),
        )
        AppSettings.set(
            f"profile_override_{profile_name}_early_entry_cutoff",
            f"{max(0.50, min(0.99, early_cutoff_float)):.4f}",
        )
    else:
        for key in (
            "early_entry_min_seconds",
            "early_entry_max_seconds",
            "early_entry_cutoff",
        ):
            row = AppSettings.query.filter_by(key=f"profile_override_{profile_name}_{key}").first()
            if row is not None:
                db.session.delete(row)
        db.session.commit()

    if description is not None:
        AppSettings.set(f"profile_override_{profile_name}_description", str(description))
    return jsonify({"saved": True, "profile": profile_name})


@api_bp.route("/risk-profiles/<profile_name>/reset", methods=["DELETE"])
def reset_risk_profile(profile_name: str):
    profile_name = (profile_name or "").strip().lower()
    if profile_name not in RISK_PROFILES:
        return jsonify({"error": "Unknown profile"}), 404
    prefix = f"profile_override_{profile_name}_"
    rows = AppSettings.query.filter(AppSettings.key.like(f"{prefix}%")).all()
    for row in rows:
        db.session.delete(row)
    db.session.commit()
    return jsonify({"reset": True})


@api_bp.route("/live-snapshot", methods=["GET"])
def live_snapshot():
    # Always compute at request time so this endpoint matches `get_live_snapshot()`
    # (scheduler in-memory cache can be tens of seconds behind and diverges curl vs UI).
    snapshot = get_live_snapshot()
    if snapshot is None:
        return jsonify({"error": "No active market"}), 404
    from app.signal_engine import evaluate_live_signal, signal_to_dict

    result = evaluate_live_signal(snapshot)
    signal = signal_to_dict(result) if result else {}
    # Dashboard time-window UI must match the same profile merge as evaluate_live_signal —
    # do not rely on a separate GET /api/settings timing (two tabs / envs diverged badly).
    risk_key = (AppSettings.get("risk_profile", "moderate") or "moderate").strip().lower()
    profile = get_profile(risk_key)
    merged = {
        **snapshot,
        **(signal or {}),
        "signal_window_min_seconds": int(profile.get("min_seconds", 60)),
        "signal_window_max_seconds": int(profile.get("max_seconds", 180)),
    }
    return jsonify(merged)


@api_bp.route("/paper/portfolio", methods=["GET"])
def paper_portfolio():
    return jsonify(get_portfolio_summary())


@api_bp.route("/paper/positions", methods=["GET"])
def paper_positions():
    return jsonify({"positions": get_open_positions()})


@api_bp.route("/paper/history", methods=["GET"])
def paper_history():
    limit_param = request.args.get("limit", default=100, type=int) or 100
    limit_param = max(1, min(limit_param, 5000))
    return jsonify({"trades": get_trade_history(limit=limit_param)})


@api_bp.route("/paper/trade/<int:trade_id>/snapshot", methods=["GET"])
def paper_trade_snapshot(trade_id: int):
    snap = TradeSnapshot.query.filter_by(trade_id=trade_id).first()
    if snap is None:
        return jsonify({"error": "no snapshot"}), 404
    trade = PaperTrade.query.get(trade_id)
    if trade is None:
        return jsonify({"error": "trade not found"}), 404
    try:
        chart_history = json.loads(snap.chart_history_json or "[]")
    except json.JSONDecodeError:
        chart_history = []
    try:
        raw_features = json.loads(snap.raw_features_json or "{}")
    except json.JSONDecodeError:
        raw_features = {}
    pm = float(snap.p_market or 0)
    pr = float(snap.p_raw or 0)
    gap = float(snap.mispricing_gap or 0)
    btc = snap.btc_price
    pos_sizing = position_sizing_breakdown(
        pm,
        str(trade.side or "YES"),
        gap,
        (snap.signal_mode or "agreement") or "agreement",
    )
    return jsonify(
        {
            "trade_id": trade_id,
            "captured_at": _utc_iso_z(snap.captured_at),
            "ticker": snap.ticker,
            "market_title": snap.market_title,
            "seconds_to_close": snap.seconds_to_close,
            "entry_bucket": snap.entry_bucket,
            "p_market": pm,
            "p_raw": pr,
            "p_market_percent": round(pm * 100, 2),
            "p_raw_percent": round(pr * 100, 2),
            "signal_mode": snap.signal_mode,
            "agreement_region": snap.agreement_region,
            "signal_reason": snap.signal_reason,
            "confidence": snap.confidence,
            "reversal_risk": snap.reversal_risk,
            "mispricing_gap": gap,
            "mispricing_gap_percent": round(gap * 100, 2),
            "position_sizing": pos_sizing,
            "btc_price": btc,
            "btc_price_formatted": f"${float(btc):,.2f}" if btc is not None else None,
            "up_price_cents": snap.up_price_cents,
            "down_price_cents": snap.down_price_cents,
            "chart_history": chart_history,
            "raw_features": raw_features,
            "entry_at": _utc_iso_z(trade.entry_at),
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "resolved": trade.resolved,
            "realized_pnl": trade.realized_pnl,
            "outcome_correct": trade.outcome_correct,
            "side": trade.side,
        }
    )


@api_bp.route("/paper/trade", methods=["POST"])
def paper_trade():
    payload = request.get_json(silent=True) or {}
    side = payload.get("side")
    contracts = payload.get("contracts")
    ticker = payload.get("ticker")
    seconds_to_close = payload.get("seconds_to_close")

    if side is None or contracts is None or ticker is None:
        return jsonify({"error": "side, contracts, and ticker are required"}), 400

    result = execute_paper_trade(
        side=side,
        contracts=contracts,
        ticker=ticker,
        seconds_to_close=seconds_to_close,
    )
    if result.get("success"):
        return jsonify(result), 200
    return jsonify(result), 400


@api_bp.route("/paper/reset", methods=["POST"])
def paper_reset():
    payload = request.get_json(silent=True) or {}
    starting_balance = payload.get("starting_balance", 100.0)
    return jsonify(reset_portfolio(starting_balance=starting_balance))
