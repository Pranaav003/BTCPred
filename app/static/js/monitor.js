const monitorState = {
    toastTimer: null,
    history: [],
    page: 1,
    pageSize: 20,
    knownPositionIds: new Set(),
    knownResolvedIds: new Set(),
    seededPositionIds: false,
    seededResolvedIds: false,
    snapshotChart: null,
};

const MONITOR_POLL_INTERVALS = {
    fast: 5000,
    normal: 15000,
    slow: 60000,
    vslow: 300000,
};

function mToPercent(value) {
    return typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "--";
}

function mFormatSignalTime(iso) {
    if (!iso) return "--";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return "--";
    return d.toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: true,
    });
}

function mFormatAnyTime(row) {
    const iso = row.logged_at || row.created_at || row.timestamp;
    if (iso) {
        const text = mFormatSignalTime(iso);
        if (text !== "--") return text;
    }
    const ts = Number(row.snapshot_ts);
    if (Number.isFinite(ts) && ts > 0) {
        return new Date(ts * 1000).toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
            hour12: true,
        });
    }
    return "--";
}

function mToMoney(value, signed = false) {
    const n = Number(value);
    if (!Number.isFinite(n)) return "--";
    const sign = signed && n > 0 ? "+" : "";
    return `${sign}$${n.toFixed(2)}`;
}

function mSetPill(el, text, klass) {
    if (!el) return;
    el.textContent = text;
    el.className = `badge ${klass}`;
}

function mShowToast(message, kind = "success") {
    const toast = document.getElementById("toast");
    if (!toast) return;
    toast.textContent = message;
    toast.className = `toast show ${kind === "error" ? "toast-error" : "toast-success"}`;
    if (monitorState.toastTimer) window.clearTimeout(monitorState.toastTimer);
    monitorState.toastTimer = window.setTimeout(() => {
        toast.className = "toast";
    }, 3000);
}

async function mApiFetch(url, options = {}) {
    try {
        const res = await fetch(url, options);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (e) {
        console.error("Monitor API failed:", url, e);
        return null;
    }
}

function mBanner(message, kind = "success") {
    const stack = document.getElementById("monitor-banner-stack");
    if (!stack) return;
    const el = document.createElement("div");
    el.className = `monitor-banner ${kind === "danger" ? "monitor-banner-danger" : "monitor-banner-success"}`;
    el.innerHTML = `
        <div>${message}</div>
        <button type="button" class="monitor-banner-close" aria-label="Dismiss">✕</button>
    `;
    const remove = () => {
        el.classList.add("fade");
        window.setTimeout(() => el.remove(), 250);
    };
    el.querySelector(".monitor-banner-close")?.addEventListener("click", remove);
    stack.prepend(el);
    window.setTimeout(remove, 30000);
}

function mRenderPortfolioSummary(data) {
    const cash = document.getElementById("monitor-cash");
    const pnl = document.getElementById("monitor-realized-pnl");
    const open = document.getElementById("monitor-open-positions");
    const win = document.getElementById("monitor-win-rate");
    if (!cash || !pnl || !open || !win) return;
    cash.textContent = mToMoney(data?.cash);
    const pnlValue = Number(data?.realized_pnl);
    pnl.textContent = mToMoney(pnlValue, true);
    pnl.classList.remove("text-success", "text-danger");
    if (pnlValue > 0) pnl.classList.add("text-success");
    if (pnlValue < 0) pnl.classList.add("text-danger");
    const openCount = Number(data?.open_trades || 0);
    open.textContent = `${openCount} position${openCount === 1 ? "" : "s"}`;
    const winRate = Number(data?.win_rate);
    win.textContent = Number.isFinite(winRate) ? `${(winRate * 100).toFixed(1)}%` : "--";
}

/** Net profit if this side wins (each contract pays $1; stake = contracts × entry). */
function mOpenPositionPotentialProfit(side, contracts, entryPrice) {
    const c = Number(contracts);
    const p = Number(entryPrice);
    if (!Number.isFinite(c) || c <= 0 || !Number.isFinite(p) || p <= 0 || p >= 1) return null;
    const s = String(side || "YES").toUpperCase();
    if (s !== "YES" && s !== "NO") return null;
    return c * (1.0 - p);
}

function mRenderOpenPositions(positions) {
    const feed = document.getElementById("monitor-open-positions-feed");
    const countEl = document.getElementById("open-positions-count");
    if (!feed) return;
    const rows = Array.isArray(positions) ? positions : [];
    if (countEl) countEl.textContent = String(rows.length);
    if (!rows.length) {
        feed.innerHTML = '<p class="text-muted">No open positions.</p>';
        return;
    }
    const sideBadge = (side) => (side === "YES"
        ? '<span class="badge badge-success">YES</span>'
        : '<span class="badge badge-danger">NO</span>');
    feed.innerHTML = rows.map((row) => {
        const side = String(row.side || "YES");
        const contracts = Number(row.contracts || 0);
        const entryPrice = Number(row.entry_price || 0);
        const entryCost = Number(row.entry_cost);
        const potential = mOpenPositionPotentialProfit(side, contracts, entryPrice);
        const autoPill = row.signal_triggered
            ? '<span class="badge badge-success monitor-auto-pill">🤖 Auto</span>'
            : '<span class="badge badge-neutral">Manual</span>';
        const profitRight = Number.isFinite(potential) && potential >= 0
            ? `<p class="mono text-success">+$${potential.toFixed(2)}</p><p class="text-muted">if ${side} wins</p>`
            : '<p class="text-muted mono">--</p><p class="text-muted">if win</p>';
        const lossLine = Number.isFinite(entryCost) && entryCost > 0
            ? `<p class="text-muted">If ${side} loses: −$${entryCost.toFixed(2)} (stake)</p>`
            : "";
        const tid = row.id != null ? String(row.id) : "";
        return `
            <div class="monitor-activity-row">
                <div>${sideBadge(side)}</div>
                <div class="monitor-activity-main">
                    <p class="mono">${row.ticker || "--"} · ${contracts.toFixed(2)} contracts @ ${entryPrice.toFixed(3)}</p>
                    <p class="text-muted">Cost ${mToMoney(row.entry_cost)} · ${mFormatSignalTime(row.entry_at)} · ${autoPill}</p>
                    ${lossLine}
                </div>
                <div class="monitor-activity-pnl">
                    ${profitRight}
                    <p class="text-muted">Awaiting resolution</p>
                </div>
                <div>
                    <button type="button" class="btn-ghost trade-snapshot-btn" data-trade-id="${tid}" title="View entry snapshot" aria-label="View entry snapshot">📊</button>
                </div>
            </div>
        `;
    }).join("");
}

function mRenderTradeActivityFeed() {
    const feed = document.getElementById("trade-activity-feed");
    const count = document.getElementById("trade-activity-count");
    const range = document.getElementById("trade-activity-range");
    const prev = document.getElementById("trade-activity-prev");
    const next = document.getElementById("trade-activity-next");
    if (!feed || !count || !range || !prev || !next) return;

    const rows = (monitorState.history || []).slice().sort((a, b) => new Date(b.exit_at || b.entry_at || 0) - new Date(a.exit_at || a.entry_at || 0));
    const total = rows.length;
    count.textContent = String(total);
    const totalPages = Math.max(1, Math.ceil(total / monitorState.pageSize));
    monitorState.page = Math.max(1, Math.min(monitorState.page, totalPages));
    const startIdx = (monitorState.page - 1) * monitorState.pageSize;
    const endIdx = Math.min(total, startIdx + monitorState.pageSize);
    const pageRows = rows.slice(startIdx, endIdx);

    if (!pageRows.length) {
        feed.innerHTML = '<p class="text-muted">No trade activity yet.</p>';
        range.textContent = "Showing 0-0 of 0 trades";
        prev.disabled = true;
        next.disabled = true;
        return;
    }

    range.textContent = `Showing ${startIdx + 1}-${endIdx} of ${total} trades`;
    prev.disabled = monitorState.page <= 1;
    next.disabled = monitorState.page >= totalPages;

    const sideBadge = (side) => side === "YES"
        ? '<span class="badge badge-success">YES</span>'
        : '<span class="badge badge-danger">NO</span>';
    feed.innerHTML = pageRows.map((row) => {
        const side = String(row.side || "YES");
        const autoPill = row.signal_triggered
            ? '<span class="badge badge-success monitor-auto-pill">🤖 Auto</span>'
            : '<span class="badge badge-neutral">Manual</span>';
        const pnl = Number(row.realized_pnl || 0);
        const pnlCls = pnl >= 0 ? "text-success" : "text-danger";
        const pnlText = `${pnl >= 0 ? "+" : "-"}$${Math.abs(pnl).toFixed(2)}`;
        const outcome = row.outcome_correct ? '<span class="text-success">✓ Correct</span>' : '<span class="text-danger">✗ Wrong</span>';
        const tid = row.id != null ? String(row.id) : "";
        return `
            <div class="monitor-activity-row">
                <div>${sideBadge(side)}</div>
                <div class="monitor-activity-main">
                    <p class="mono">${row.ticker || "--"} · ${Number(row.contracts || 0).toFixed(2)} contracts @ ${Number(row.entry_price || 0).toFixed(3)}</p>
                    <p class="text-muted">Entry ${Number(row.entry_price || 0).toFixed(3)} → Exit ${Number(row.exit_price || 0).toFixed(3)} · ${mFormatSignalTime(row.exit_at || row.entry_at)} · ${autoPill}</p>
                </div>
                <div class="monitor-activity-pnl">
                    <p class="${pnlCls} mono">${pnlText}</p>
                    <p>${outcome}</p>
                </div>
                <div>
                    <button type="button" class="btn-ghost trade-snapshot-btn" data-trade-id="${tid}" title="View entry snapshot" aria-label="View entry snapshot">📊</button>
                </div>
            </div>
        `;
    }).join("");
}

function mDestroySnapshotChart() {
    if (monitorState.snapshotChart) {
        monitorState.snapshotChart.destroy();
        monitorState.snapshotChart = null;
    }
}

function mCloseTradeSnapshotModal() {
    const modal = document.getElementById("trade-snapshot-modal");
    mDestroySnapshotChart();
    if (modal) modal.style.display = "none";
}

function mRegionLabel(key) {
    const map = {
        agree_yes: "Both YES",
        agree_no: "Both NO",
        model_bullish: "Model Bullish",
        model_bearish: "Model Bearish",
        market_yes_raw_no: "Mkt YES / Mdl NO",
        market_no_raw_yes: "Mkt NO / Mdl YES",
        no_agreement: "No Agreement",
        outside_time_window: "Outside window",
        volatility_guard: "Volatility guard",
    };
    return map[key] || key || "—";
}

function mBuildSnapshotChart(data) {
    const canvas = document.getElementById("trade-snapshot-chart");
    const emptyEl = document.getElementById("trade-snapshot-chart-empty");
    if (!canvas || typeof window.Chart === "undefined") return;
    mDestroySnapshotChart();
    const history = Array.isArray(data.chart_history) ? data.chart_history : [];
    if (!history.length) {
        canvas.classList.add("hidden");
        if (emptyEl) {
            emptyEl.classList.remove("hidden");
            emptyEl.textContent = "No chart history";
        }
        return;
    }
    canvas.classList.remove("hidden");
    if (emptyEl) emptyEl.classList.add("hidden");

    const labels = history.map((pt, i) => {
        if (pt.logged_at) {
            const d = new Date(pt.logged_at);
            if (!Number.isNaN(d.getTime())) return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
        }
        return String(i + 1);
    });
    const pMarket = history.map((pt) => (pt.p_market == null ? null : Number(pt.p_market)));
    const pRaw = history.map((pt) => (pt.p_raw == null ? null : Number(pt.p_raw)));

    const entryLinePlugin = {
        id: "tradeSnapshotEntryLine",
        afterDatasetsDraw(chart) {
            const { chartArea, ctx } = chart;
            if (!chartArea) return;
            const m0 = chart.getDatasetMeta(0);
            if (!m0.data.length) return;
            const last = m0.data[m0.data.length - 1];
            if (!last) return;
            const x = last.x;
            ctx.save();
            ctx.setLineDash([4, 4]);
            ctx.strokeStyle = "rgba(239, 68, 68, 0.95)";
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(x, chartArea.top);
            ctx.lineTo(x, chartArea.bottom);
            ctx.stroke();
            ctx.restore();
        },
    };

    monitorState.snapshotChart = new window.Chart(canvas, {
        type: "line",
        data: {
            labels,
            datasets: [
                {
                    label: "Market",
                    data: pMarket,
                    borderColor: "#3b82f6",
                    backgroundColor: "rgba(59,130,246,0.12)",
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.2,
                },
                {
                    label: "Model",
                    data: pRaw,
                    borderColor: "#f59e0b",
                    backgroundColor: "rgba(245,158,11,0.1)",
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.2,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                x: { display: true, ticks: { maxRotation: 0, color: "rgba(200,200,210,0.6)", maxTicksLimit: 8 }, grid: { color: "rgba(40,40,55,0.5)" } },
                y: { min: 0, max: 1, ticks: { color: "rgba(200,200,210,0.6)" }, grid: { color: "rgba(40,40,55,0.5)" } },
            },
            plugins: { legend: { display: false } },
        },
        plugins: [entryLinePlugin],
    });
}

function mFormatSnapshotFeaturesTable(raw) {
    const keys = ["return_1m", "return_3m", "volatility_3m", "momentum_1m", "trade_count_1m", "flip_count_5m"];
    const body = document.getElementById("trade-snapshot-features-body");
    if (!body) return;
    if (!raw || typeof raw !== "object") {
        body.innerHTML = "<tr><td colspan=\"2\" class=\"text-muted\">—</td></tr>";
        return;
    }
    body.innerHTML = keys
        .map((k) => {
            const v = raw[k];
            const display = v === null || v === undefined ? "—" : (typeof v === "number" ? v.toFixed(4) : String(v));
            return `<tr><td class="mono">${k}</td><td class="mono">${display}</td></tr>`;
        })
        .join("");
}

function mOpenTradeSnapshotModal(data) {
    const modal = document.getElementById("trade-snapshot-modal");
    if (!modal) return;
    const title = document.getElementById("trade-snapshot-modal-title");
    const side = String(data.side || "YES");
    const tick = data.ticker || "--";
    const ent = data.entry_at ? mFormatSignalTime(data.entry_at) : "--";
    if (title) title.textContent = `[${side}] ${tick} — ${ent}`;

    const btc = document.getElementById("ts-btc");
    if (btc) btc.textContent = data.btc_price_formatted || (data.btc_price != null ? mToMoney(data.btc_price) : "—");
    const sec = document.getElementById("ts-sec");
    if (sec) sec.textContent = data.seconds_to_close != null ? String(data.seconds_to_close) : "—";
    const rev = document.getElementById("ts-rev");
    if (rev) {
        const r = Number(data.reversal_risk);
        rev.textContent = Number.isFinite(r) ? `${(r * 100).toFixed(1)}%` : "—";
    }
    const conf = document.getElementById("ts-conf");
    if (conf) {
        const c = Number(data.confidence);
        conf.textContent = Number.isFinite(c) ? `${(c * 100).toFixed(1)}%` : "—";
    }

    const mr = document.getElementById("ts-mode-region");
    if (mr) {
        const mode = String(data.signal_mode || "—");
        const reg = mRegionLabel(String(data.agreement_region || ""));
        mr.textContent = `Mode: ${mode}  ·  Region: ${reg}`;
    }
    const reason = document.getElementById("ts-reason");
    if (reason) reason.textContent = data.signal_reason || "—";
    const gapEl = document.getElementById("ts-gap");
    if (gapEl) {
        const gp = data.mispricing_gap_percent;
        const num = Number(gp);
        gapEl.classList.remove("gap-high");
        if (Number.isFinite(num) && num > 20) gapEl.classList.add("gap-high");
        gapEl.textContent = `Mispricing gap: ${Number.isFinite(num) ? num.toFixed(1) : "—"}%`;
    }

    const posMain = document.getElementById("ts-position-mult-main");
    const posBrk = document.getElementById("ts-position-mult-brk");
    const ps = data.position_sizing;
    if (posMain && posBrk) {
        if (ps && ps.final_multiplier != null && Number.isFinite(Number(ps.final_multiplier))) {
            const em = Number(ps.base_multiplier);
            const mm = Number(ps.mispricing_multiplier);
            const tot = Number(ps.final_multiplier);
            posMain.textContent = `Position multiplier at entry: ${tot.toFixed(2)}x`;
            posBrk.textContent = `${em.toFixed(1)}x edge × ${mm.toFixed(2)}x gap = ${tot.toFixed(2)}x`;
        } else {
            posMain.textContent = "—";
            posBrk.textContent = "";
        }
    }

    mFormatSnapshotFeaturesTable(data.raw_features);
    mBuildSnapshotChart(data);

    const outPrices = document.getElementById("ts-outcome-prices");
    const outPnl = document.getElementById("ts-outcome-pnl");
    const outLabel = document.getElementById("ts-outcome-label");
    if (outPrices && outPnl && outLabel) {
        if (data.resolved) {
            outPrices.textContent = `Entry ${Number(data.entry_price || 0).toFixed(3)} → Exit ${data.exit_price != null ? Number(data.exit_price).toFixed(3) : "—"}`;
            const pnl = Number(data.realized_pnl);
            const pnlStr = Number.isFinite(pnl) ? `${pnl >= 0 ? "+" : "-"}$${Math.abs(pnl).toFixed(2)}` : "—";
            outPnl.textContent = `P&L: ${pnlStr}`;
            outPnl.classList.remove("text-success", "text-danger");
            if (Number.isFinite(pnl)) outPnl.classList.add(pnl >= 0 ? "text-success" : "text-danger");
            outLabel.textContent = data.outcome_correct === true ? "Outcome: correct" : data.outcome_correct === false ? "Outcome: wrong" : "Outcome: —";
        } else {
            outPrices.textContent = "Position not yet resolved";
            outPnl.textContent = "";
            outLabel.textContent = "";
        }
    }

    const cap = document.getElementById("ts-captured");
    if (cap) cap.textContent = data.captured_at ? `Snapshot captured at ${mFormatSignalTime(data.captured_at)}` : "Snapshot captured at —";

    const featWrap = document.getElementById("trade-snapshot-features-wrap");
    const featToggle = document.getElementById("trade-snapshot-features-toggle");
    if (featWrap) featWrap.classList.add("hidden");
    if (featToggle) {
        featToggle.setAttribute("aria-expanded", "false");
        const ch = document.getElementById("trade-snapshot-features-chevron");
        if (ch) ch.textContent = "▶";
    }

    modal.style.display = "flex";
}

async function mFetchTradeSnapshot(tradeId) {
    try {
        const res = await fetch(`/api/paper/trade/${tradeId}/snapshot`, { headers: { Accept: "application/json" } });
        if (res.status === 404) {
            const err = await res.json().catch(() => ({}));
            return { notFound: true, message: err.error || "no snapshot" };
        }
        if (!res.ok) return null;
        return await res.json();
    } catch (e) {
        console.error(e);
        return null;
    }
}

async function mOnSnapshotButtonClick(tradeId) {
    if (!tradeId) return;
    const data = await mFetchTradeSnapshot(tradeId);
    if (data && data.notFound) {
        mShowToast("No snapshot for this trade", "error");
        return;
    }
    if (!data) {
        mShowToast("Failed to load snapshot", "error");
        return;
    }
    mOpenTradeSnapshotModal(data);
}

function mRenderTradeActivitySummary() {
    const summary = document.getElementById("trade-activity-summary");
    const warning = document.getElementById("trade-activity-ratio-warning");
    if (!summary || !warning) return;
    const rows = Array.isArray(monitorState.history) ? monitorState.history : [];
    const pnlValues = rows.map((row) => Number(row.realized_pnl)).filter((n) => Number.isFinite(n));
    const wins = pnlValues.filter((n) => n > 0);
    const losses = pnlValues.filter((n) => n < 0);
    const winCount = wins.length;
    const lossCount = losses.length;
    const tradeCount = rows.length;
    const net = pnlValues.reduce((acc, n) => acc + n, 0);
    const avgWin = winCount ? wins.reduce((acc, n) => acc + n, 0) / winCount : 0;
    const avgLoss = lossCount ? losses.reduce((acc, n) => acc + n, 0) / lossCount : 0;
    summary.innerHTML = `${tradeCount} trades · ${winCount} wins · ${lossCount} losses · Avg win: +$${Math.abs(avgWin).toFixed(2)} · Avg loss: -$${Math.abs(avgLoss).toFixed(2)} · Net: <span class="${net >= 0 ? "text-success" : "text-danger"}">${net >= 0 ? "+" : "-"}$${Math.abs(net).toFixed(2)}</span>`;

    warning.classList.remove("text-success", "text-warning", "text-danger", "text-muted");
    if (winCount === 0 || avgWin <= 0) {
        warning.textContent = "Loss ratio: -- (not enough winning trades yet)";
        warning.classList.add("text-muted");
        return;
    }
    const ratio = Math.abs(avgLoss) / Math.abs(avgWin);
    if (!Number.isFinite(ratio)) {
        warning.textContent = "Loss ratio: --";
        warning.classList.add("text-muted");
    } else if (ratio > 10) {
        warning.textContent = `⚠ Loss ratio: ${ratio.toFixed(1)}x — one loss wipes ${ratio.toFixed(1)} wins. Consider enabling entry price filter.`;
        warning.classList.add("text-danger");
    } else if (ratio >= 5) {
        warning.textContent = `Loss ratio: ${ratio.toFixed(1)}x — consider tighter filters`;
        warning.classList.add("text-warning");
    } else {
        warning.textContent = `Loss ratio: ${ratio.toFixed(1)}x — healthy risk/reward`;
        warning.classList.add("text-success");
    }
}

async function mFetchPortfolio() {
    const data = await mApiFetch("/api/paper/portfolio", { headers: { Accept: "application/json" } });
    if (!data) return;
    mRenderPortfolioSummary(data);
}

async function mFetchPositions() {
    const payload = await mApiFetch("/api/paper/positions", { headers: { Accept: "application/json" } });
    if (!payload) return;
    const positions = Array.isArray(payload.positions) ? payload.positions : [];
    mRenderOpenPositions(positions);
    if (!monitorState.seededPositionIds) {
        monitorState.knownPositionIds = new Set(positions.map((p) => p.id));
        monitorState.seededPositionIds = true;
    } else {
        positions.forEach((p) => {
            if (!monitorState.knownPositionIds.has(p.id)) {
                const side = String(p.side || "YES");
                const message = `🤖 Auto-traded · BUY ${side} · ${Number(p.contracts || 0).toFixed(2)} contracts · ${p.ticker} · Entry $${Number(p.entry_price || 0).toFixed(3)} · Cost ${mToMoney(p.entry_cost)}`;
                mBanner(message, side === "YES" ? "success" : "danger");
            }
            monitorState.knownPositionIds.add(p.id);
        });
    }
}

async function mFetchHistory() {
    const payload = await mApiFetch("/api/paper/history?limit=200", { headers: { Accept: "application/json" } });
    if (!payload) return;
    const history = Array.isArray(payload.trades) ? payload.trades : [];
    if (!monitorState.seededResolvedIds) {
        monitorState.knownResolvedIds = new Set(history.map((h) => h.id));
        monitorState.seededResolvedIds = true;
    } else {
        history.forEach((h) => {
            if (!monitorState.knownResolvedIds.has(h.id)) {
                const pnl = Number(h.realized_pnl || 0);
                const sign = pnl >= 0 ? "+" : "-";
                const message = `${pnl >= 0 ? "✓" : "✗"} Trade resolved · ${h.ticker} · ${h.side} · ${sign}$${Math.abs(pnl).toFixed(2)} · Entry ${Number(h.entry_price || 0).toFixed(3)} → Exit ${Number(h.exit_price || 0).toFixed(3)}`;
                mBanner(message, pnl >= 0 ? "success" : "danger");
            }
            monitorState.knownResolvedIds.add(h.id);
        });
    }
    monitorState.history = history;
    mRenderTradeActivityFeed();
    mRenderTradeActivitySummary();
}

async function mFetchSchedulerStatus() {
    const [status, settings] = await Promise.all([
        mApiFetch("/api/scheduler/status", { headers: { Accept: "application/json" } }),
        mApiFetch("/api/settings", { headers: { Accept: "application/json" } }),
    ]);
    if (!status) return;
    const schedulerPill = document.getElementById("scheduler-status-pill");
    const autoPill = document.getElementById("auto-trade-status-pill");
    const noPill = document.getElementById("no-trading-status-pill");
    mSetPill(schedulerPill, `Scheduler: ${status.running ? "RUNNING" : "STOPPED"}`, status.running ? "badge-success" : "badge-danger");
    mSetPill(autoPill, `Auto-Trade: ${status.auto_trade_enabled ? "ON" : "OFF"}`, status.auto_trade_enabled ? "badge-success" : "badge-neutral");
    const enableNo = (settings?.enable_no_signals || "false") === "true";
    if (noPill) {
        noPill.style.display = status.auto_trade_enabled ? "" : "none";
        if (status.auto_trade_enabled) {
            mSetPill(noPill, enableNo ? "NO Trading: EXPERIMENTAL" : "YES Only", enableNo ? "badge-warning" : "badge-neutral");
        }
    }
}

async function mFetchResolutionSummary() {
    const summary = await mApiFetch("/api/resolution/summary", { headers: { Accept: "application/json" } });
    if (!summary) return;
    const pending = Number(summary.pending_resolution || 0);
    const resolutionPill = document.getElementById("resolution-status-pill");
    if (resolutionPill) {
        mSetPill(resolutionPill, pending > 0 ? `Pending Resolution: ${pending}` : "All Resolved", pending > 0 ? "badge-warning" : "badge-success");
    }
    const setText = (id, value) => {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    };
    setText("res-total-markets", String(summary.total_markets ?? "--"));
    setText("res-resolved-markets", String(summary.resolved_markets ?? "--"));
    setText("res-pending-resolution", String(summary.pending_resolution ?? "--"));
    setText("res-unresolved-future", String(summary.unresolved_future ?? "--"));
    setText("res-last-resolved", summary.last_resolved_at ? mFormatSignalTime(summary.last_resolved_at) : "--");
}

async function mAction(path, successMsg) {
    const res = await mApiFetch(path, { method: "POST", headers: { Accept: "application/json" } });
    if (!res) {
        mShowToast("Action failed", "error");
        return;
    }
    mShowToast(successMsg, "success");
    await Promise.all([mFetchSchedulerStatus(), mFetchResolutionSummary()]);
}

async function mFetchDataCollectionStatus() {
    const stats = await mApiFetch("/api/export/live-training-data?stats=1", { headers: { Accept: "application/json" } });
    const summaryEl = document.getElementById("data-collection-summary");
    const progressFill = document.getElementById("data-collection-progress-fill");
    const progressText = document.getElementById("data-collection-progress-text");
    const exportBtn = document.getElementById("data-collection-export-btn");
    if (!summaryEl || !progressFill || !progressText || !exportBtn) return;
    if (!stats) {
        summaryEl.textContent = "Live training data: -- rows collected | -- rows resolved | Ready to retrain: --";
        progressText.textContent = "Need -- more resolved signals";
        progressFill.style.width = "0%";
        exportBtn.classList.add("hidden");
        return;
    }
    const rows = Number(stats.rows || 0);
    const skipped = Number(stats.skipped || 0);
    const resolved = rows + skipped;
    const needed = Math.max(0, 200 - resolved);
    const ready = resolved >= 200;
    summaryEl.textContent = `Live training data: ${rows} rows collected | ${resolved} rows resolved | Ready to retrain: ${ready ? "YES" : "NO"}`;
    const progressPct = Math.max(0, Math.min(100, (resolved / 200) * 100));
    progressFill.style.width = `${progressPct.toFixed(0)}%`;
    progressText.textContent = ready
        ? "Ready to retrain - click Export & Retrain"
        : `Need ${needed} more resolved signals`;
    exportBtn.classList.toggle("hidden", !ready);
}

function mWireDataCollectionExport() {
    const btn = document.getElementById("data-collection-export-btn");
    if (!btn) return;
    btn.addEventListener("click", () => {
        window.location.href = "/api/export/live-training-data";
        mShowToast("Downloaded live training CSV. Run: python merge_and_retrain.py", "success");
    });
}

window.addEventListener("DOMContentLoaded", async () => {
    const snapshotClick = (e) => {
        const btn = e.target && e.target.closest && e.target.closest(".trade-snapshot-btn");
        if (!btn) return;
        const id = btn.getAttribute("data-trade-id");
        if (id) mOnSnapshotButtonClick(id);
    };
    document.getElementById("trade-activity-feed")?.addEventListener("click", snapshotClick);
    document.getElementById("monitor-open-positions-feed")?.addEventListener("click", snapshotClick);
    const snapModal = document.getElementById("trade-snapshot-modal");
    snapModal?.addEventListener("click", (e) => {
        if (e.target === snapModal) mCloseTradeSnapshotModal();
    });
    document.getElementById("trade-snapshot-close-btn")?.addEventListener("click", mCloseTradeSnapshotModal);
    document.getElementById("trade-snapshot-close-x")?.addEventListener("click", mCloseTradeSnapshotModal);
    document.getElementById("trade-snapshot-features-toggle")?.addEventListener("click", () => {
        const w = document.getElementById("trade-snapshot-features-wrap");
        if (!w) return;
        const isHidden = w.classList.toggle("hidden");
        const t = document.getElementById("trade-snapshot-features-toggle");
        const ch = document.getElementById("trade-snapshot-features-chevron");
        if (t) t.setAttribute("aria-expanded", isHidden ? "false" : "true");
        if (ch) ch.textContent = isHidden ? "▶" : "▼";
    });
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && document.getElementById("trade-snapshot-modal")?.style.display === "flex") {
            mCloseTradeSnapshotModal();
        }
    });
    document.getElementById("trade-activity-prev")?.addEventListener("click", () => {
        monitorState.page = Math.max(1, monitorState.page - 1);
        mRenderTradeActivityFeed();
    });
    document.getElementById("trade-activity-next")?.addEventListener("click", () => {
        monitorState.page += 1;
        mRenderTradeActivityFeed();
    });
    document.getElementById("scheduler-start-btn")?.addEventListener("click", () => mAction("/api/scheduler/start", "Scheduler started"));
    document.getElementById("scheduler-stop-btn")?.addEventListener("click", () => mAction("/api/scheduler/stop", "Scheduler stopped"));
    document.getElementById("resolution-trigger-btn")?.addEventListener("click", () => mAction("/api/resolution/trigger", "Resolution triggered"));
    mWireDataCollectionExport();
    await mFetchPositions();
    window.setTimeout(() => mFetchPortfolio(), 1000);
    window.setTimeout(() => mFetchHistory(), 2000);
    window.setTimeout(() => mFetchSchedulerStatus(), 3000);
    window.setTimeout(() => mFetchResolutionSummary(), 3500);
    window.setTimeout(() => mFetchDataCollectionStatus(), 4000);

    setInterval(() => {
        mFetchPositions();
    }, MONITOR_POLL_INTERVALS.normal);
    setInterval(() => {
        mFetchPortfolio();
        mFetchHistory();
    }, MONITOR_POLL_INTERVALS.normal);
    setInterval(() => {
        mFetchSchedulerStatus();
        mFetchResolutionSummary();
    }, MONITOR_POLL_INTERVALS.slow);
    setInterval(() => {
        mFetchDataCollectionStatus();
    }, MONITOR_POLL_INTERVALS.vslow);
});
