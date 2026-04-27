const DASHBOARD_PORTFOLIO_POLL_MS = 15000;
const LIVE_SNAPSHOT_POLL_MS = 5000;
const DASHBOARD_INTEL_POLL_MS = 30000;
const MAX_CHART_POINTS = 50;
const DEFAULT_YES_CUTOFF = 0.65;

const state = {
    chart: null,
    secondsToClose: null,
    latestMarket: null,
    portfolio: null,
    autoTradeEnabled: false,
    paperEnabled: false,
    selectedTradeSide: "YES",
    tradeSizeSaveTimerId: null,
    tradeSizeSavedIndicatorTimerId: null,
    toastTimerId: null,
    yesCutoff: DEFAULT_YES_CUTOFF,
    knownPositionIds: new Set(),
    knownResolvedIds: new Set(),
    seededPositionIds: false,
    seededResolvedIds: false,
    showMarket: true,
    showModel: true,
    showThreshold: true,
    minSecondsWindow: 60,
    maxSecondsWindow: 180,
    reversalRisk: null,
    maxEntryYes: 0.85,
    maxEntryNo: 0.85,
    maxReversalRisk: 0.65,
    dynamicSizingEnabled: false,
    signalMode: "agreement",
    mispricingThreshold: 0.10,
};

const toText = (value) => (value === null || value === undefined || value === "" ? "--" : String(value));
const toNumberOrNull = (value) => {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : null;
};
const toPercent = (value) => (typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "--");

function getReversalRiskMeta(value) {
    const risk = Number(value);
    if (!Number.isFinite(risk)) {
        return { level: "unknown", label: "--", textClass: "text-muted", fillClass: "", description: "--" };
    }
    if (risk < 0.3) {
        return { level: "low", label: "Low", textClass: "text-success", fillClass: "low", description: "Low — trend is consistent" };
    }
    if (risk < 0.6) {
        return { level: "medium", label: "Medium", textClass: "text-warning", fillClass: "medium", description: "Medium — some volatility" };
    }
    return { level: "high", label: "High", textClass: "text-danger", fillClass: "high", description: "High — volatile, possible reversal" };
}

function toMoney(value) {
    const n = Number(value);
    return Number.isFinite(n) ? `$${n.toFixed(2)}` : "--";
}

function formatTime(seconds) {
    const totalSeconds = Number(seconds);
    if (!Number.isFinite(totalSeconds) || totalSeconds <= 0) return "0:00";
    const whole = Math.floor(totalSeconds);
    const m = Math.floor(whole / 60);
    const s = whole % 60;
    return `${m}:${String(s).padStart(2, "0")}`;
}

function formatSignalTime(isoString) {
    if (!isoString) return "--";
    const d = new Date(isoString);
    if (Number.isNaN(d.getTime())) return "--";
    return d.toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: true,
    });
}

function regionLabelForIntel(region) {
    const map = {
        agree_yes: "Both YES",
        agree_no: "Both NO",
        market_yes_raw_no: "Mkt YES / Mdl NO",
        market_no_raw_yes: "Mkt NO / Mdl YES",
        model_bullish: "Model Bullish",
        model_bearish: "Model Bearish",
        no_agreement: "No Agreement",
        outside_time_window: "Outside window",
        volatility_guard: "Volatility guard",
    };
    return map[region] || region || "—";
}

function truncateTicker(s, maxLen = 18) {
    const str = String(s || "");
    if (str.length <= maxLen) return str;
    return `${str.slice(0, Math.max(0, maxLen - 3))}...`;
}

function startOfLocalDayMs() {
    const d = new Date();
    d.setHours(0, 0, 0, 0);
    return d.getTime();
}

function pickBestAgreementRegion(regions) {
    if (!Array.isArray(regions) || !regions.length) return null;
    const candidates = regions.filter(
        (r) => r && (Number(r.resolved_count) || 0) > 0 && r.avg_pnl != null && Number.isFinite(Number(r.avg_pnl)),
    );
    if (!candidates.length) return null;
    return candidates.reduce((best, r) => (Number(r.avg_pnl) > Number(best.avg_pnl) ? r : best));
}

function formatSignedPnl2(n) {
    if (!Number.isFinite(Number(n))) return "—";
    const v = Number(n);
    const sign = v >= 0 ? "+" : "−";
    return `${sign}$${Math.abs(v).toFixed(2)}`;
}

function renderSignalIntelFromLive(snapshot) {
    // Signal-rate card is rendered from daily trade/signal history, not per-snapshot confidence.
    void snapshot;
}

function renderSignalRateCard(signals, trades) {
    const labelEl = document.getElementById("intel-confidence-stat-label");
    const statEl = document.getElementById("intel-model-confidence");
    const subEl = document.getElementById("intel-agreement-region");
    if (!statEl || !subEl) return;
    if (labelEl) labelEl.textContent = "Signal Rate";
    const dayStart = startOfLocalDayMs();
    const signalRows = Array.isArray(signals) ? signals : [];
    const tradeRows = Array.isArray(trades) ? trades : [];
    const signalsToday = signalRows.filter((s) => {
        const ts = new Date(s?.logged_at || "").getTime();
        return Number.isFinite(ts) && ts >= dayStart;
    });
    const tradesToday = tradeRows.filter((t) => {
        const ts = new Date(t?.entry_at || "").getTime();
        return Number.isFinite(ts) && ts >= dayStart;
    });
    const sCount = signalsToday.length;
    const tCount = tradesToday.length;
    statEl.textContent = `${sCount} signals today | ${tCount} trades today`;
    statEl.classList.remove("text-success", "text-warning", "text-danger", "text-muted");
    statEl.classList.add(tCount > 0 ? "text-success" : "text-muted");
    if (tCount < 2) {
        subEl.textContent = "No trades yet today";
        return;
    }
    const times = tradesToday
        .map((t) => new Date(t.entry_at).getTime())
        .filter((v) => Number.isFinite(v))
        .sort((a, b) => a - b);
    if (times.length < 2) {
        subEl.textContent = "No trades yet today";
        return;
    }
    let diffSumMs = 0;
    for (let i = 1; i < times.length; i += 1) diffSumMs += (times[i] - times[i - 1]);
    const avgMs = diffSumMs / (times.length - 1);
    const totalMin = Math.round(avgMs / 60000);
    const h = Math.floor(totalMin / 60);
    const m = totalMin % 60;
    subEl.textContent = `Avg time between trades: ${h}h ${m}m`;
}

function renderActiveModeFromSettings(settings) {
    const dot = document.getElementById("intel-active-mode-dot");
    const textEl = document.getElementById("intel-active-mode-text");
    if (!textEl) return;
    const modeRaw = (settings?.signal_mode || "agreement").toLowerCase();
    const normalizedMode = modeRaw === "ensemble_vote" ? "ensemble" : modeRaw;
    const mode = ["agreement", "mispricing", "ensemble"].includes(normalizedMode) ? normalizedMode : "agreement";
    const mThresh = Number(settings?.mispricing_threshold);
    const yesCut = Number(settings?.yes_cutoff);
    if (dot) {
        dot.classList.remove("mode-mispricing", "mode-agreement");
        dot.classList.add(mode === "mispricing" ? "mode-mispricing" : "mode-agreement");
    }
    if (mode === "mispricing") {
        const p = Number.isFinite(mThresh) ? mThresh : 0.1;
        textEl.textContent = `Mispricing  ${(p * 100).toFixed(0)}% threshold`;
    } else if (mode === "ensemble") {
        const p = Number.isFinite(yesCut) ? yesCut : 0.65;
        const m = Number.isFinite(mThresh) ? mThresh : 0.2;
        textEl.textContent = `Ensemble  ${Math.round(p * 100)}% agree OR ${(m * 100).toFixed(0)}% gap`;
    } else {
        const p = Number.isFinite(yesCut) ? yesCut : 0.65;
        textEl.textContent = `Agreement  ${(p * 100).toFixed(0)}% threshold`;
    }
}

function renderBestRegionCell(regionRow) {
    const el = document.getElementById("intel-best-region");
    if (!el) return;
    if (!regionRow || !regionRow.agreement_region) {
        el.textContent = "—";
        el.classList.remove("text-success");
        return;
    }
    const label = regionLabelForIntel(String(regionRow.agreement_region));
    const avg = Number(regionRow.avg_pnl);
    el.textContent = Number.isFinite(avg) ? `${label}  ${formatSignedPnl2(avg)} avg` : `${label}`;
    el.classList.add("text-success");
}

function renderTodaysTradesCell(trades) {
    const el = document.getElementById("intel-todays-trades");
    if (!el) return;
    if (!Array.isArray(trades) || !trades.length) {
        el.textContent = `0 trades  ${formatSignedPnl2(0)} today`;
        el.classList.remove("text-success", "text-danger");
        return;
    }
    const dayStart = startOfLocalDayMs();
    let n = 0;
    let net = 0;
    trades.forEach((t) => {
        if (!t?.entry_at) return;
        const et = new Date(t.entry_at).getTime();
        if (!Number.isFinite(et) || et < dayStart) return;
        n += 1;
        net += Number(t.realized_pnl || 0);
    });
    if (n === 0) {
        el.textContent = `0 trades  ${formatSignedPnl2(0)} today`;
        el.classList.remove("text-success", "text-danger");
        return;
    }
    const netClass = net >= 0 ? "text-success" : "text-danger";
    el.classList.remove("text-success", "text-danger");
    el.classList.add(netClass);
    el.textContent = `${n} trade${n === 1 ? "" : "s"}  ${formatSignedPnl2(net)} today`;
}

function renderRecentTradesList(trades) {
    const ul = document.getElementById("recent-trades-list");
    const empty = document.getElementById("recent-trades-empty");
    if (!ul) return;
    ul.querySelectorAll("li.trade-recent-row").forEach((row) => row.remove());
    const list = Array.isArray(trades) ? trades.slice(0, 5) : [];
    if (empty) empty.style.display = list.length ? "none" : "block";
    list.forEach((t) => {
        const li = document.createElement("li");
        li.className = "trade-recent-row";
        const side = String(t.side || "").toUpperCase();
        const sideEl = document.createElement("span");
        sideEl.className = `trade-recent-side ${side === "YES" ? "yes" : "no"}`;
        sideEl.textContent = side === "NO" ? "NO" : "YES";
        const tickEl = document.createElement("span");
        tickEl.className = "trade-recent-ticker";
        tickEl.setAttribute("title", String(t.ticker || ""));
        tickEl.textContent = truncateTicker(t.ticker || "", 20);
        const priceEl = document.createElement("span");
        priceEl.className = "trade-recent-prices";
        const ep = Number(t.entry_price);
        const xp = Number(t.exit_price);
        priceEl.textContent = `${Number.isFinite(ep) ? ep.toFixed(3) : "—"}→${Number.isFinite(xp) ? xp.toFixed(3) : "—"}`;
        const pnl = Number(t.realized_pnl);
        const pnlEl = document.createElement("span");
        pnlEl.className = `trade-recent-pnl ${Number.isFinite(pnl) && pnl >= 0 ? "text-success" : "text-danger"}`;
        pnlEl.textContent = Number.isFinite(pnl) ? `${pnl >= 0 ? "+" : ""}$${Math.abs(pnl).toFixed(2)}` : "—";
        const checkEl = document.createElement("span");
        checkEl.className = "trade-recent-check";
        checkEl.textContent = Number.isFinite(pnl) && pnl >= 0 ? "✓" : "✗";
        li.append(sideEl, tickEl, priceEl, pnlEl, checkEl);
        ul.appendChild(li);
    });
}

async function fetchDashboardRecentTrades() {
    const data = await apiFetch("/api/paper/history?limit=5", { headers: { Accept: "application/json" } });
    if (!data) return;
    renderRecentTradesList(data.trades);
}

async function refreshDashboardIntelBlocks() {
    const [regionsPayload, settings, historyPayload, signalsPayload] = await Promise.all([
        apiFetch("/api/analytics/agreement-regions", { headers: { Accept: "application/json" } }),
        apiFetch("/api/settings", { headers: { Accept: "application/json" } }),
        apiFetch("/api/paper/history?limit=100", { headers: { Accept: "application/json" } }),
        apiFetch("/api/signals?limit=1000", { headers: { Accept: "application/json" } }),
    ]);
    if (regionsPayload?.regions) {
        renderBestRegionCell(pickBestAgreementRegion(regionsPayload.regions));
    }
    if (settings) {
        renderActiveModeFromSettings(settings);
    }
    if (historyPayload?.trades) {
        renderTodaysTradesCell(historyPayload.trades);
    }
    renderSignalRateCard(signalsPayload?.signals || [], historyPayload?.trades || []);
}

function showToast(message, variant = "success", durationMs = 3000) {
    const toastEl = document.getElementById("toast");
    if (!toastEl) return;
    toastEl.textContent = message;
    toastEl.className = `toast show ${(variant === "error" || variant === "danger") ? "toast-error" : "toast-success"}`;
    if (state.toastTimerId) window.clearTimeout(state.toastTimerId);
    state.toastTimerId = window.setTimeout(() => {
        toastEl.className = "toast";
    }, durationMs);
}

function showTradeSizeSavedIndicator() {
    const indicator = document.getElementById("trade-size-saved-indicator");
    if (!indicator) return;
    indicator.classList.remove("hidden");
    indicator.classList.add("show");
    if (state.tradeSizeSavedIndicatorTimerId) window.clearTimeout(state.tradeSizeSavedIndicatorTimerId);
    state.tradeSizeSavedIndicatorTimerId = window.setTimeout(() => {
        indicator.classList.remove("show");
        indicator.classList.add("hidden");
    }, 2000);
}

async function apiFetch(url, options = {}) {
    try {
        const response = await fetch(url, options);
        if (!response.ok) throw new Error(`HTTP error: ${response.status}`);
        return await response.json();
    } catch (error) {
        console.error("API fetch failed:", url, error);
        return null;
    }
}

function highlightActiveNav() {
    const path = window.location.pathname.replace(/\/+$/, "") || "/";
    document.querySelectorAll(".sidebar .nav-item").forEach((link) => {
        const href = (link.getAttribute("href") || "").replace(/\/+$/, "") || "/";
        link.classList.toggle("nav-active", path === href);
    });
}

function renderMetrics(record) {
    const confidenceEl = document.getElementById("metric-confidence");
    const pMarketEl = document.getElementById("metric-p-market");
    const pRawEl = document.getElementById("metric-p-raw");
    if (!pMarketEl || !pRawEl) return;

    const pMarket = toNumberOrNull(record?.p_market);
    const pRaw = toNumberOrNull(record?.p_raw);
    pMarketEl.textContent = toPercent(pMarket);
    pRawEl.textContent = toPercent(pRaw);

    if (confidenceEl) {
        confidenceEl.classList.remove("text-success", "text-warning", "text-danger");
        if (Number.isFinite(pMarket) && Number.isFinite(pRaw)) {
            const confidence = (Math.abs(pMarket - 0.5) + Math.abs(pRaw - 0.5)) * 100.0;
            confidenceEl.textContent = `${confidence.toFixed(1)}%`;
            if (confidence >= 60) confidenceEl.classList.add("text-success");
            else if (confidence >= 30) confidenceEl.classList.add("text-warning");
            else confidenceEl.classList.add("text-danger");
        } else {
            confidenceEl.textContent = "--";
        }
    }
}

function renderMarketInfo(record) {
    const tickerEl = document.getElementById("market-ticker");
    const titleEl = document.getElementById("market-title");
    const secondsLiveEl = document.getElementById("market-seconds-live");
    if (!tickerEl || !titleEl || !secondsLiveEl) return;

    tickerEl.textContent = toText(record?.ticker);
    titleEl.textContent = toText(record?.title || record?.market_title);
    secondsLiveEl.textContent = state.secondsToClose === null ? "--" : formatTime(state.secondsToClose);
    secondsLiveEl.classList.remove("countdown-normal", "countdown-warn", "countdown-danger");
    if (state.secondsToClose < 60) secondsLiveEl.classList.add("countdown-danger");
    else if (state.secondsToClose <= 120) secondsLiveEl.classList.add("countdown-warn");
    else secondsLiveEl.classList.add("countdown-normal");
}

function updateReversalRiskUI() {
    const fillEl = document.getElementById("reversal-risk-fill");
    const scoreEl = document.getElementById("reversal-risk-score");
    const textEl = document.getElementById("reversal-risk-text");
    const blockPill = document.getElementById("reversal-risk-block-pill");
    if (!fillEl || !scoreEl || !textEl) return;

    const risk = Number(state.reversalRisk);
    const meta = getReversalRiskMeta(risk);
    fillEl.classList.remove("low", "medium", "high");
    textEl.classList.remove("text-success", "text-warning", "text-danger", "text-muted");

    if (!Number.isFinite(risk)) {
        fillEl.style.width = "0%";
        scoreEl.textContent = "--";
        textEl.textContent = "--";
        textEl.classList.add("text-muted");
        if (blockPill) blockPill.classList.add("hidden");
        updateVolatilityGuardUI();
        return;
    }

    fillEl.style.width = `${Math.max(0, Math.min(100, risk * 100)).toFixed(0)}%`;
    if (meta.fillClass) fillEl.classList.add(meta.fillClass);
    scoreEl.textContent = `${(risk * 100).toFixed(1)}%`;
    textEl.textContent = meta.description;
    textEl.classList.add(meta.textClass);
    if (blockPill) {
        const blocked = risk > Number(state.maxReversalRisk || 0.65);
        blockPill.classList.toggle("hidden", !blocked);
    }
    updateVolatilityGuardUI();
}

function applySignalTradeButtonHighlight(signalText) {
    const yesBtn = document.getElementById("paper-buy-yes-btn");
    const noBtn = document.getElementById("paper-buy-no-btn");
    if (!yesBtn || !noBtn) return;
    yesBtn.style.opacity = "1";
    noBtn.style.opacity = "1";
    if (signalText === "PAPER BUY YES") noBtn.style.opacity = "0.5";
    else if (signalText === "PAPER BUY NO") yesBtn.style.opacity = "0.5";
}

function renderPaperSignalPanel(record) {
    const signalEl = document.getElementById("metric-signal");
    const subtextEl = document.getElementById("signal-subtext");
    const panel = document.getElementById("signal-card");
    const earlyPill = document.getElementById("early-entry-pill");
    const riskWarningEl = document.getElementById("signal-risk-warning");
    if (!signalEl || !panel || !subtextEl) return;

    const signalText = toText(record?.signal || "NO SIGNAL");
    const signalReason = String(record?.reason || "");
    const agreementRegion = String(record?.agreement_region || "");
    const isEarlyEntry = signalReason.toLowerCase().includes("early entry");
    const pMarket = toNumberOrNull(record?.p_market);
    const pRaw = toNumberOrNull(record?.p_raw);
    const cutoff = Number(state.yesCutoff || DEFAULT_YES_CUTOFF);
    const marketGap = Number.isFinite(pMarket) ? Math.max(0, cutoff - pMarket) : NaN;
    const modelGap = Number.isFinite(pRaw) ? Math.max(0, cutoff - pRaw) : NaN;
    panel.classList.remove("signal-yes", "signal-no", "signal-none");
    if (signalText === "PAPER BUY YES") {
        signalEl.textContent = "▲ BUY YES";
        signalEl.style.color = "#22c55e";
        subtextEl.textContent = `Strong agreement · ${Number.isFinite(state.secondsToClose) ? Math.max(0, Math.floor(state.secondsToClose)) : "--"}s to close`;
        panel.classList.add("signal-yes");
    } else if (signalText === "PAPER BUY NO") {
        signalEl.textContent = "▼ BUY NO";
        signalEl.style.color = "#ef4444";
        subtextEl.textContent = `Strong agreement · ${Number.isFinite(state.secondsToClose) ? Math.max(0, Math.floor(state.secondsToClose)) : "--"}s to close`;
        panel.classList.add("signal-no");
    } else {
        signalEl.textContent = "WAITING";
        signalEl.style.color = "";
        subtextEl.classList.remove("text-warning", "text-muted");
        const marketAbove = Number.isFinite(marketGap) ? marketGap <= 0 : false;
        const modelAbove = Number.isFinite(modelGap) ? modelGap <= 0 : false;
        if (agreementRegion === "outside_time_window") {
            subtextEl.classList.add("text-muted");
            subtextEl.textContent = `Outside window — ${Number.isFinite(state.secondsToClose) ? Math.max(0, Math.floor(state.secondsToClose)) : "--"}s to close (window: ${Number(state.minSecondsWindow || 60)}s-${Number(state.maxSecondsWindow || 180)}s)`;
        } else if (state.signalMode === "ensemble" && agreementRegion === "entry_filtered" && Number.isFinite(pMarket)) {
            subtextEl.classList.add("text-warning");
            subtextEl.textContent = `▲ Agreement met — entry price ${(pMarket * 100).toFixed(1)}% exceeds max ${(Number(state.maxEntryYes || 0.8) * 100).toFixed(1)}%. Watching for mispricing gap ≥ ${(Number(state.mispricingThreshold || 0.20) * 100).toFixed(0)}%...`;
        } else if (marketAbove && modelAbove && agreementRegion === "outside_time_window") {
            const nowSeconds = Number(state.secondsToClose || 0);
            if (nowSeconds > Number(state.maxSecondsWindow || 0)) {
                const waitUntil = Math.max(0, nowSeconds - Number(state.maxSecondsWindow || 0));
                subtextEl.textContent = `Agreement reached — enters window in ${formatTime(waitUntil)}`;
            } else if (nowSeconds < Number(state.minSecondsWindow || 0)) {
                subtextEl.textContent = "Agreement reached — too close to close";
            } else {
                subtextEl.textContent = "✓ Both above threshold — signal should fire";
            }
        } else if (!marketAbove && !modelAbove) {
            subtextEl.textContent = `Market needs +${(marketGap * 100).toFixed(1)}%, Model needs +${(modelGap * 100).toFixed(1)}%`;
        } else if (!marketAbove) {
            subtextEl.textContent = `Market needs +${(marketGap * 100).toFixed(1)}% more · Model ✓`;
        } else if (!modelAbove) {
            subtextEl.textContent = `Market ✓ · Model needs +${(modelGap * 100).toFixed(1)}% more`;
        } else {
            if (state.signalMode === "mispricing") {
                subtextEl.textContent = `Watching for model/market divergence above ${(Number(state.mispricingThreshold || 0.10) * 100).toFixed(1)}%`;
            } else if (state.signalMode === "ensemble") {
                subtextEl.classList.add("text-muted");
                subtextEl.textContent = `Watching for agreement ≥ ${(state.yesCutoff * 100).toFixed(0)}% OR gap ≥ ${(Number(state.mispricingThreshold || 0.20) * 100).toFixed(0)}% (Entry filter: max ${(Number(state.maxEntryYes || 0.8) * 100).toFixed(1)}%)`;
            } else {
                subtextEl.textContent = `Watching for agreement above ${(state.yesCutoff * 100).toFixed(0)}%...`;
            }
        }
        panel.classList.add("signal-none");
    }
    if (earlyPill) {
        earlyPill.classList.toggle("hidden", !isEarlyEntry || signalText === "NO SIGNAL");
    }
    if (riskWarningEl) {
        const meta = getReversalRiskMeta(state.reversalRisk);
        riskWarningEl.classList.add("hidden");
        riskWarningEl.classList.remove("text-warning", "text-danger", "text-success");
        if (signalReason.includes("HIGH CONVICTION despite volatility")) {
            riskWarningEl.textContent = "⚡ HIGH CONVICTION despite volatility";
            riskWarningEl.classList.add("text-success");
            riskWarningEl.classList.remove("hidden");
        } else if (meta.level === "high" && signalText === "PAPER BUY NO") {
            riskWarningEl.textContent = "⚠ High reversal risk — volatile conditions, NO trade may reverse";
            riskWarningEl.classList.add("text-danger");
            riskWarningEl.classList.remove("hidden");
        } else if (meta.level === "high" && signalText === "PAPER BUY YES") {
            riskWarningEl.textContent = "⚠ Volatile conditions — confirm before trading";
            riskWarningEl.classList.add("text-warning");
            riskWarningEl.classList.remove("hidden");
        }
    }
    applySignalTradeButtonHighlight(signalText);
}

function updateAgreementStatus(pMarket, pRaw, yesCutoff, secondsToClose, minSec, maxSec) {
    const marketAbove = Number.isFinite(pMarket) ? pMarket >= yesCutoff : false;
    const modelAbove = Number.isFinite(pRaw) ? pRaw >= yesCutoff : false;
    const inWindow = Number.isFinite(secondsToClose) && secondsToClose >= minSec && secondsToClose <= maxSec;
    if (marketAbove && modelAbove && inWindow) {
        return { text: "✓ Both above threshold — signal should fire", badgeClass: "badge-success" };
    }
    if (marketAbove && modelAbove && !inWindow) {
        if (Number.isFinite(secondsToClose) && secondsToClose > maxSec) {
            const waitSeconds = secondsToClose - maxSec;
            return {
                text: `Agreement reached — enters window in ${formatTime(waitSeconds)}`,
                badgeClass: "badge-warning",
            };
        }
        return { text: "Agreement reached — too close to close", badgeClass: "badge-danger" };
    }
    if (!marketAbove && !modelAbove) {
        return { text: "Both below threshold", badgeClass: "badge-neutral" };
    }
    if (!marketAbove) {
        const marketGap = Math.max(0, (yesCutoff - Number(pMarket)) * 100);
        return { text: `Market needs +${marketGap.toFixed(1)}% · Model ✓`, badgeClass: "badge-warning" };
    }
    const modelGap = Math.max(0, (yesCutoff - Number(pRaw)) * 100);
    return { text: `Market ✓ · Model needs +${modelGap.toFixed(1)}%`, badgeClass: "badge-warning" };
}

function updateThresholdGap(pMarket, pRaw, yesCutoff) {
    const marketFill = document.getElementById("market-threshold-fill");
    const modelFill = document.getElementById("model-threshold-fill");
    const marketGapEl = document.getElementById("market-threshold-gap");
    const modelGapEl = document.getElementById("model-threshold-gap");
    const marketMarker = document.getElementById("market-threshold-marker");
    const modelMarker = document.getElementById("model-threshold-marker");
    const marketText = document.getElementById("market-gap-text");
    const modelText = document.getElementById("model-gap-text");
    const agreementPill = document.getElementById("threshold-agreement-pill");
    const primaryLabel = document.getElementById("threshold-primary-label");
    const secondaryLabel = document.getElementById("threshold-secondary-label");
    const secondaryRow = document.getElementById("threshold-secondary-row");
    const timeRow = document.getElementById("threshold-time-row");
    const timeText = document.getElementById("threshold-time-text");
    const tertiaryRow = document.getElementById("threshold-tertiary-row");
    const tertiaryLabel = document.getElementById("threshold-tertiary-label");
    const entryFill = document.getElementById("entry-filter-fill");
    const entryGap = document.getElementById("entry-filter-gap");
    const entryMarker = document.getElementById("entry-filter-marker");
    const entryText = document.getElementById("entry-filter-text");
    if (!marketFill || !modelFill || !marketGapEl || !modelGapEl || !marketMarker || !modelMarker || !marketText || !modelText || !agreementPill || !primaryLabel || !secondaryLabel || !secondaryRow) return;

    if (state.signalMode === "mispricing") {
        if (timeRow) timeRow.classList.add("hidden");
        primaryLabel.textContent = "Gap";
        secondaryLabel.textContent = "Need";
        secondaryRow.classList.add("hidden");
        if (tertiaryRow) tertiaryRow.classList.add("hidden");
        marketGapEl.style.display = "none";
        modelGapEl.style.display = "none";
        marketMarker.style.display = "none";
        modelMarker.style.display = "none";
        modelFill.style.width = "0%";
        const threshold = Math.max(0.0001, Number(state.mispricingThreshold || 0.10));
        const gap = Number.isFinite(pMarket) && Number.isFinite(pRaw) ? Math.abs(pRaw - pMarket) : NaN;
        const progress = Number.isFinite(gap) ? Math.max(0, Math.min(100, (gap / threshold) * 100)) : 0;
        marketFill.style.width = `${progress.toFixed(0)}%`;
        marketFill.classList.remove("ready", "market", "model");
        if (!Number.isFinite(gap)) {
            marketFill.classList.add("market");
            marketText.textContent = "Gap --";
            modelText.textContent = `Need ${(threshold * 100).toFixed(1)}%`;
            agreementPill.className = "badge badge-neutral";
            agreementPill.textContent = "Waiting for divergence";
            return;
        }
        const gapPct = gap * 100;
        const needPct = threshold * 100;
        if (gap >= threshold) {
            marketFill.classList.add("ready");
            agreementPill.className = "badge badge-success";
            agreementPill.textContent = `✓ Gap ${gapPct.toFixed(1)}% exceeds ${needPct.toFixed(1)}% threshold`;
        } else if (gap >= threshold * 0.75) {
            marketFill.classList.add("model");
            agreementPill.className = "badge badge-warning";
            agreementPill.textContent = `Close — need ${(needPct - gapPct).toFixed(1)}% more divergence`;
        } else {
            marketFill.classList.add("market");
            agreementPill.className = "badge badge-neutral";
            agreementPill.textContent = "Below mispricing threshold";
        }
        marketText.textContent = `Gap ${gapPct.toFixed(1)}%`;
        modelText.textContent = `Need ${needPct.toFixed(1)}%`;
        return;
    }

    if (state.signalMode === "ensemble") {
        if (timeRow) timeRow.classList.remove("hidden");
        if (timeText) {
            const sec = Number(state.secondsToClose);
            const minSec = Number(state.minSecondsWindow || 60);
            const maxSec = Number(state.maxSecondsWindow || 180);
            timeText.classList.remove("text-success", "text-warning", "text-danger", "text-muted");
            if (!Number.isFinite(sec)) {
                timeText.textContent = "Window --";
                timeText.classList.add("text-muted");
            } else if (sec >= minSec && sec <= maxSec) {
                timeText.textContent = `✓ In window (${Math.max(0, Math.floor(sec))}s)`;
                timeText.classList.add("text-success");
            } else if (sec > maxSec) {
                timeText.textContent = `Entering in ${Math.max(0, Math.floor(sec - maxSec))}s`;
                timeText.classList.add("text-warning");
            } else {
                timeText.textContent = `Too late (${Math.max(0, Math.floor(sec))}s)`;
                timeText.classList.add("text-danger");
            }
        }
        primaryLabel.textContent = "Agreement gap";
        secondaryLabel.textContent = "Mispricing gap";
        secondaryRow.classList.remove("hidden");
        if (tertiaryRow) tertiaryRow.classList.remove("hidden");
        if (tertiaryLabel) tertiaryLabel.textContent = "Entry filter";
        marketMarker.style.display = "";
        modelMarker.style.display = "";
        marketGapEl.style.display = "none";
        modelGapEl.style.display = "none";

        const cutoff = Number(yesCutoff);
        const agreeScore = Number.isFinite(pMarket) && Number.isFinite(pRaw) ? Math.max(pMarket, pRaw) : NaN;
        const agreeNeed = Number.isFinite(agreeScore) ? Math.max(0, cutoff - agreeScore) : NaN;
        const agreeProgress = Number.isFinite(agreeScore) ? Math.max(0, Math.min(100, (agreeScore / cutoff) * 100)) : 0;
        marketMarker.style.left = "100%";
        marketFill.style.width = `${agreeProgress.toFixed(0)}%`;
        marketFill.classList.remove("ready", "market", "model");
        marketFill.classList.add("market");
        marketText.classList.remove("text-success", "text-warning", "text-muted");
        if (!Number.isFinite(agreeScore)) {
            marketText.textContent = "Need --";
            marketText.classList.add("text-muted");
        } else if (agreeNeed <= 0) {
            marketFill.classList.add("ready");
            marketText.textContent = "✓ Agreement met";
            marketText.classList.add("text-success");
        } else {
            marketText.textContent = `Need ${(agreeNeed * 100).toFixed(1)}% more`;
            marketText.classList.add(agreeNeed <= 0.1 ? "text-warning" : "text-muted");
        }

        const mpThresh = Math.max(0.0001, Number(state.mispricingThreshold || 0.20));
        const gap = Number.isFinite(pMarket) && Number.isFinite(pRaw) ? Math.abs(pRaw - pMarket) : NaN;
        const mpNeed = Number.isFinite(gap) ? Math.max(0, mpThresh - gap) : NaN;
        const mpProgress = Number.isFinite(gap) ? Math.max(0, Math.min(100, (gap / mpThresh) * 100)) : 0;
        modelMarker.style.left = "100%";
        modelFill.style.width = `${mpProgress.toFixed(0)}%`;
        modelFill.classList.remove("ready", "market", "model");
        modelFill.classList.add("model");
        modelText.classList.remove("text-success", "text-warning", "text-muted");
        if (!Number.isFinite(gap)) {
            modelText.textContent = "Gap --";
            modelText.classList.add("text-muted");
        } else if (mpNeed <= 0) {
            modelFill.classList.add("ready");
            modelText.textContent = `✓ Mispricing detected (${(gap * 100).toFixed(1)}%)`;
            modelText.classList.add("text-success");
        } else {
            modelText.textContent = `Gap ${(gap * 100).toFixed(1)}%`;
            modelText.classList.add(mpNeed <= 0.05 ? "text-warning" : "text-muted");
        }

        const agreeAway = Number.isFinite(agreeNeed) ? agreeNeed : Infinity;
        const gapAway = Number.isFinite(mpNeed) ? mpNeed : Infinity;
        const agreementCloser = agreeAway <= gapAway;
        agreementPill.className = `badge ${agreementCloser ? "badge-success" : "badge-warning"}`;
        agreementPill.textContent = `Agreement: ${Number.isFinite(agreeAway) ? (agreeAway * 100).toFixed(1) : "--"}% away | Gap: ${Number.isFinite(gapAway) ? (gapAway * 100).toFixed(1) : "--"}% away`;

        if (entryFill && entryGap && entryMarker && entryText) {
            const maxEntry = Number(state.maxEntryYes || 0.8);
            const pm = Number.isFinite(pMarket) ? pMarket : NaN;
            const progress = Number.isFinite(pm) ? Math.max(0, Math.min(100, (pm / Math.max(maxEntry, 0.0001)) * 100)) : 0;
            entryMarker.style.left = "100%";
            entryGap.style.display = "none";
            entryFill.style.width = `${progress.toFixed(0)}%`;
            entryFill.classList.remove("ready", "market", "model");
            entryText.classList.remove("text-success", "text-danger", "text-muted");
            if (!Number.isFinite(pm)) {
                entryText.textContent = "Entry --";
                entryText.classList.add("text-muted");
            } else if (pm <= maxEntry) {
                entryFill.classList.add("ready");
                entryText.textContent = `✓ Price OK (${(pm * 100).toFixed(1)}%)`;
                entryText.classList.add("text-success");
            } else {
                entryFill.classList.add("model");
                entryText.textContent = `✗ Too expensive (${(pm * 100).toFixed(1)}% > max ${(maxEntry * 100).toFixed(1)}%)`;
                entryText.classList.add("text-danger");
            }
        }
        return;
    }

    primaryLabel.textContent = "Market";
    secondaryLabel.textContent = "Model";
    secondaryRow.classList.remove("hidden");
    if (timeRow) timeRow.classList.add("hidden");
    if (tertiaryRow) tertiaryRow.classList.add("hidden");
    marketMarker.style.display = "";
    modelMarker.style.display = "";

    const thresholdPct = Math.max(0, Math.min(100, Number(yesCutoff) * 100));
    marketMarker.style.left = `${thresholdPct}%`;
    modelMarker.style.left = `${thresholdPct}%`;

    const renderRow = (value, fillEl, gapEl, textEl, baseClass) => {
        const valid = Number.isFinite(value);
        const pct = valid ? Math.max(0, Math.min(100, value * 100)) : 0;
        const gap = valid ? Number(yesCutoff) - value : NaN;
        fillEl.style.width = `${pct}%`;
        fillEl.classList.remove("ready", "market", "model");
        fillEl.classList.add(baseClass);
        textEl.classList.remove("text-success", "text-warning", "text-muted");
        gapEl.style.display = "none";
        if (!valid) {
            textEl.textContent = "--";
            return { gap: NaN, above: false };
        }
        if (gap <= 0) {
            fillEl.classList.add("ready");
            textEl.textContent = `✓ +${(Math.abs(gap) * 100).toFixed(1)}% above`;
            textEl.classList.add("text-success");
            return { gap, above: true };
        }
        const gapPct = gap * 100;
        textEl.textContent = `Need +${gapPct.toFixed(1)}% more`;
        textEl.classList.add(gap <= 0.10 ? "text-warning" : "text-muted");
        const left = Math.min(pct, thresholdPct);
        const width = Math.max(0, thresholdPct - pct);
        if (width > 0.5) {
            gapEl.style.left = `${left}%`;
            gapEl.style.width = `${width}%`;
            gapEl.style.display = "block";
        }
        return { gap, above: false };
    };

    renderRow(pMarket, marketFill, marketGapEl, marketText, "market");
    renderRow(pRaw, modelFill, modelGapEl, modelText, "model");
    const status = updateAgreementStatus(
        pMarket,
        pRaw,
        Number(yesCutoff),
        Number(state.secondsToClose),
        Number(state.minSecondsWindow || 60),
        Number(state.maxSecondsWindow || 180),
    );
    agreementPill.className = `badge ${status.badgeClass}`;
    agreementPill.textContent = status.text;
}

function setNumberColor(el, value, includePct = false) {
    if (!el) return;
    el.classList.remove("text-success", "text-danger");
    const n = Number(value);
    if (!Number.isFinite(n)) {
        el.textContent = "--";
        return;
    }
    if (n > 0) el.classList.add("text-success");
    else if (n < 0) el.classList.add("text-danger");
    if (includePct) {
        const sign = n > 0 ? "+" : "";
        el.textContent = `${sign}${n.toFixed(2)}%`;
    } else {
        const sign = n > 0 ? "+" : "";
        el.textContent = `${sign}${toMoney(n)}`;
    }
}

function syncAutoTradePill() {
    const pill = document.getElementById("auto-trade-active-pill");
    if (!pill) return;
    const active = state.paperEnabled && state.autoTradeEnabled;
    pill.textContent = active ? "ACTIVE" : "INACTIVE";
    pill.className = `badge ${active ? "badge-success" : "badge-neutral"}`;
    updateVolatilityGuardUI();
}

function updateVolatilityGuardUI() {
    const statusEl = document.getElementById("auto-trade-volatility-status");
    if (!statusEl) return;
    const risk = Number(state.reversalRisk);
    const maxReversal = Number(state.maxReversalRisk || 0.65);
    const blocked = Number.isFinite(risk) && risk > maxReversal;
    statusEl.classList.remove("text-success", "text-warning");
    if (blocked) {
        statusEl.textContent = `Auto-trader paused — high volatility (risk: ${(risk * 100).toFixed(1)}%)`;
        statusEl.classList.add("text-warning");
    } else {
        statusEl.textContent = "Auto-trader active";
        statusEl.classList.add("text-success");
    }
}

function applyPaperEnabled(enabled) {
    state.paperEnabled = Boolean(enabled);
    const autoToggle = document.getElementById("auto-trade-toggle");
    if (autoToggle) autoToggle.disabled = !state.paperEnabled;
    syncAutoTradePill();
    updateTradeCalculator();
}

function renderPortfolio(portfolio) {
    state.portfolio = portfolio || null;
    const cashEl = document.getElementById("paper-cash");
    const realizedEl = document.getElementById("paper-realized-pnl");
    const winEl = document.getElementById("paper-win-rate");
    const returnEl = document.getElementById("paper-return-pct");
    if (!cashEl || !realizedEl || !winEl || !returnEl) return;
    cashEl.textContent = toMoney(portfolio?.cash);
    setNumberColor(realizedEl, portfolio?.realized_pnl);
    const winRate = Number(portfolio?.win_rate);
    winEl.textContent = Number.isFinite(winRate) ? `${(winRate * 100).toFixed(1)}%` : "--";
    setNumberColor(returnEl, portfolio?.total_return_pct, true);
    updateTradeCalculator();
}

function buildCutoffPlugin() {
    return {
        id: "yesCutoffLine",
        afterDraw(chart) {
            if (!state.showThreshold) return;
            const yScale = chart.scales.y;
            const xScale = chart.scales.x;
            if (!yScale || !xScale) return;
            const yPixel = yScale.getPixelForValue(state.yesCutoff);
            const { ctx } = chart;
            ctx.save();
            ctx.setLineDash([6, 6]);
            ctx.strokeStyle = "rgba(148, 163, 184, 0.8)";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(xScale.left, yPixel);
            ctx.lineTo(xScale.right, yPixel);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = "rgba(228, 228, 234, 0.7)";
            ctx.font = "12px Inter, sans-serif";
            ctx.textAlign = "right";
            ctx.fillText("Threshold", xScale.right - 6, yPixel - 6);
            ctx.restore();
        },
    };
}

function applyChartToggleStyles() {
    const marketBtn = document.getElementById("toggle-market-line");
    const modelBtn = document.getElementById("toggle-model-line");
    const thresholdBtn = document.getElementById("toggle-threshold-line");
    if (marketBtn) marketBtn.classList.toggle("active", state.showMarket);
    if (modelBtn) modelBtn.classList.toggle("active", state.showModel);
    if (thresholdBtn) thresholdBtn.classList.toggle("active", state.showThreshold);
}

function setupChartToggles() {
    const marketBtn = document.getElementById("toggle-market-line");
    const modelBtn = document.getElementById("toggle-model-line");
    const thresholdBtn = document.getElementById("toggle-threshold-line");
    if (!marketBtn || !modelBtn || !thresholdBtn) return;

    const updateVisibility = () => {
        const chart = ensureChart();
        if (!chart) return;
        chart.data.datasets[0].hidden = !state.showMarket;
        chart.data.datasets[1].hidden = !state.showModel;
        chart.update();
        applyChartToggleStyles();
    };

    marketBtn.addEventListener("click", () => {
        state.showMarket = !state.showMarket;
        updateVisibility();
    });
    modelBtn.addEventListener("click", () => {
        state.showModel = !state.showModel;
        updateVisibility();
    });
    thresholdBtn.addEventListener("click", () => {
        state.showThreshold = !state.showThreshold;
        updateVisibility();
    });
    applyChartToggleStyles();
}

function ensureChart() {
    const canvas = document.getElementById("probability-chart");
    if (!canvas || typeof window.Chart === "undefined") return null;
    if (state.chart) return state.chart;
    state.chart = new window.Chart(canvas, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    label: "Market (p_market)",
                    data: [],
                    borderColor: "#3b82f6",
                    backgroundColor: "rgba(59, 130, 246, 0.08)",
                    fill: true,
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.3,
                },
                {
                    label: "Model (p_raw)",
                    data: [],
                    borderColor: "#f59e0b",
                    backgroundColor: "rgba(245, 158, 11, 0)",
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.3,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            layout: { padding: 8 },
            scales: {
                x: {
                    ticks: { color: "rgba(228, 228, 234, 0.6)", maxTicksLimit: 6, maxRotation: 0, minRotation: 0 },
                    grid: { color: "#1e1e2e" },
                },
                y: {
                    min: 0,
                    max: 1,
                    ticks: {
                        color: "rgba(228, 228, 234, 0.6)",
                        callback(value) {
                            return `${Math.round(Number(value) * 100)}%`;
                        },
                    },
                    grid: { color: "#1e1e2e" },
                },
            },
            plugins: {
                legend: { labels: { color: "rgba(228, 228, 234, 0.85)" } },
            },
        },
        plugins: [buildCutoffPlugin()],
    });
    state.chart.data.datasets[0].hidden = !state.showMarket;
    state.chart.data.datasets[1].hidden = !state.showModel;
    return state.chart;
}

async function fetchSignalHistory(limit) {
    const endpoint = `/api/signals/history?limit=${typeof limit === "number" ? limit : 50}`;
    const data = await apiFetch(endpoint, { headers: { Accept: "application/json" } });
    return Array.isArray(data?.history) ? data.history : [];
}

async function bootstrapChartSeed() {
    const rows = await fetchSignalHistory(MAX_CHART_POINTS);
    const chart = ensureChart();
    if (!chart || !Array.isArray(rows)) return;
    const ordered = rows.slice().reverse();
    chart.data.labels = ordered.map((row) => {
        return formatSignalTime(row.logged_at);
    });
    chart.data.datasets[0].data = ordered.map((row) => toNumberOrNull(row.p_market));
    chart.data.datasets[1].data = ordered.map((row) => toNumberOrNull(row.p_raw));
    chart.update("none");
}

function updateChartPoint(record) {
    const chart = ensureChart();
    if (!chart) return;
    const now = new Date();
    const label = formatSignalTime(now.toISOString());
    chart.data.labels.push(label);
    chart.data.datasets[0].data.push(toNumberOrNull(record.p_market));
    chart.data.datasets[1].data.push(toNumberOrNull(record.p_raw));
    if (chart.data.labels.length > MAX_CHART_POINTS) {
        chart.data.labels.shift();
        chart.data.datasets[0].data.shift();
        chart.data.datasets[1].data.shift();
    }
    chart.update("none");
}

function renderFreshness(record) {
    const el = document.getElementById("chart-freshness");
    if (!el) return;
    const ts = Number(record?.snapshot_ts);
    if (!Number.isFinite(ts) || ts <= 0) {
        el.innerHTML = 'Updated -- · <span class="dot market-dot"></span>Market · <span class="dot model-dot"></span>Model';
        return;
    }
    const ageSeconds = Math.max(0, Math.floor(Date.now() / 1000) - ts);
    el.innerHTML = `Updated ${ageSeconds}s ago · <span class="dot market-dot"></span>Market · <span class="dot model-dot"></span>Model`;
}

async function fetchPortfolio() {
    const data = await apiFetch("/api/paper/portfolio", { headers: { Accept: "application/json" } });
    if (!data) return;
    renderPortfolio(data);
}

async function fetchPositionsForToasts() {
    const data = await apiFetch("/api/paper/positions", { headers: { Accept: "application/json" } });
    if (!data) return;
    const positions = Array.isArray(data.positions) ? data.positions : [];
    if (!state.seededPositionIds) {
        state.knownPositionIds = new Set(positions.map((p) => p.id));
        state.seededPositionIds = true;
        return;
    }
    positions.forEach((p) => {
        if (!state.knownPositionIds.has(p.id) && p.signal_triggered) {
            const side = String(p.side || "YES");
            const msg = `🤖 Auto-Trader: BUY ${side} · ${p.ticker}\n${Number(p.contracts || 0).toFixed(2)} contracts @ ${Number(p.entry_price || 0).toFixed(3)}\nCost: ${toMoney(p.entry_cost)}`;
            showToast(msg, side === "YES" ? "success" : "danger", 10000);
        }
        state.knownPositionIds.add(p.id);
    });
}

async function fetchResolvedForToasts() {
    const data = await apiFetch("/api/paper/history", { headers: { Accept: "application/json" } });
    if (!data) return;
    const trades = Array.isArray(data.trades) ? data.trades : [];
    if (!state.seededResolvedIds) {
        state.knownResolvedIds = new Set(trades.map((t) => t.id));
        state.seededResolvedIds = true;
        return;
    }
    trades.forEach((t) => {
        if (!state.knownResolvedIds.has(t.id)) {
            const pnl = Number(t.realized_pnl || 0);
            const sign = pnl >= 0 ? "+" : "-";
            const msg = `${pnl >= 0 ? "✓" : "✗"} ${sign}$${Math.abs(pnl).toFixed(2)} on ${t.ticker} ${t.side}`;
            showToast(msg, pnl >= 0 ? "success" : "danger", 10000);
        }
        state.knownResolvedIds.add(t.id);
    });
}

async function fetchLiveSnapshot() {
    const snapshot = await apiFetch("/api/live-snapshot", { headers: { Accept: "application/json" } });
    if (!snapshot) return;
    state.yesCutoff = Number.isFinite(Number(snapshot?.yes_cutoff)) ? Number(snapshot.yes_cutoff) : state.yesCutoff;
    const seconds = Number(snapshot?.seconds_to_close);
    if (Number.isFinite(seconds)) state.secondsToClose = seconds;
    state.reversalRisk = Number.isFinite(Number(snapshot?.reversal_risk)) ? Number(snapshot.reversal_risk) : null;

    const liveRecord = {
        ...state.latestMarket,
        ticker: snapshot?.market_ticker || state.latestMarket?.ticker,
        title: snapshot?.market_title || state.latestMarket?.title,
        p_market: snapshot?.p_market,
        p_raw: snapshot?.p_raw,
        seconds_to_close: snapshot?.seconds_to_close,
        signal: snapshot?.signal,
        reason: snapshot?.reason,
        snapshot_ts: snapshot?.snapshot_ts,
        confidence: snapshot?.confidence,
        agreement_region: snapshot?.agreement_region,
    };
    state.latestMarket = liveRecord;
    renderMetrics(liveRecord);
    renderMarketInfo(liveRecord);
    renderPaperSignalPanel(liveRecord);
    updateReversalRiskUI();
    updateThresholdGap(toNumberOrNull(liveRecord.p_market), toNumberOrNull(liveRecord.p_raw), Number(state.yesCutoff || DEFAULT_YES_CUTOFF));
    renderFreshness(liveRecord);
    updateChartPoint(liveRecord);
    updateTradeCalculator();
    renderSignalIntelFromLive(snapshot);
}

async function fetchWindowSettings() {
    const settings = await apiFetch("/api/settings", { headers: { Accept: "application/json" } });
    if (!settings) return;
    const minVal = Number(settings.min_seconds_to_close);
    const maxVal = Number(settings.max_seconds_to_close);
    if (Number.isFinite(minVal)) state.minSecondsWindow = minVal;
    if (Number.isFinite(maxVal)) state.maxSecondsWindow = maxVal;
}

async function fetchMarketPrices() {
    const payload = await apiFetch("/api/market-prices", { headers: { Accept: "application/json" } });
    if (!payload) return;
    const btcEl = document.getElementById("btc-price-value");
    const upPill = document.getElementById("up-price-pill");
    const downPill = document.getElementById("down-price-pill");
    if (btcEl) {
        btcEl.textContent = payload.btc_price_formatted || "--";
        btcEl.classList.toggle("muted", payload.btc_price == null);
    }
    if (upPill) {
        upPill.textContent = payload.up_display || "Up --";
    }
    if (downPill) {
        downPill.textContent = payload.down_display || "Down --";
    }
}

function currentEntryPriceForSide(side) {
    const pMarket = toNumberOrNull(state.latestMarket?.p_market);
    if (!Number.isFinite(pMarket)) return null;
    return side === "YES" ? pMarket : 1 - pMarket;
}

function getEdgeSizingMultiplier(edge) {
    const value = Number(edge);
    if (!Number.isFinite(value)) return 1.0;
    if (value >= 0.35) return 1.5;
    if (value >= 0.20) return 1.0;
    if (value >= 0.10) return 0.6;
    return 0.3;
}

function getMispricingGapMultiplier(mispricingGap) {
    const g = Number(mispricingGap);
    if (!Number.isFinite(g)) return 1.0;
    if (g >= 0.4) return 2.0;
    if (g >= 0.3) return 1.75;
    if (g >= 0.2) return 1.5;
    return 1.0;
}

function refreshNoTradeRiskModalContent() {
    const bodyEl = document.getElementById("no-trade-risk-body");
    const levelEl = document.getElementById("no-trade-risk-level");
    if (!bodyEl || !levelEl) return;
    const seconds = Number.isFinite(state.secondsToClose) ? Math.max(0, Math.floor(state.secondsToClose)) : 0;
    const meta = getReversalRiskMeta(state.reversalRisk);
    bodyEl.textContent = `You're buying NO with ${seconds}s to close. With more than 2 minutes remaining, there's significant risk of reversal. The auto-trader would not take this trade.`;
    levelEl.textContent = `Current reversal risk: ${meta.label.toUpperCase()}`;
    levelEl.classList.remove("text-success", "text-warning", "text-danger", "text-muted");
    levelEl.classList.add(meta.textClass);
}

function confirmHighRiskNoTrade() {
    return new Promise((resolve) => {
        const modal = document.getElementById("no-trade-risk-modal");
        const cancelBtn = document.getElementById("no-trade-risk-cancel-btn");
        const confirmBtn = document.getElementById("no-trade-risk-confirm-btn");
        if (!modal || !cancelBtn || !confirmBtn) {
            resolve(false);
            return;
        }
        refreshNoTradeRiskModalContent();
        modal.style.display = "flex";

        const cleanup = (result) => {
            modal.style.display = "none";
            cancelBtn.removeEventListener("click", onCancel);
            confirmBtn.removeEventListener("click", onConfirm);
            modal.removeEventListener("click", onBackdrop);
            document.removeEventListener("keydown", onKeydown);
            resolve(result);
        };
        const onCancel = () => cleanup(false);
        const onConfirm = () => cleanup(true);
        const onBackdrop = (event) => {
            if (event.target === modal) cleanup(false);
        };
        const onKeydown = (event) => {
            if (event.key === "Escape") cleanup(false);
        };

        cancelBtn.addEventListener("click", onCancel);
        confirmBtn.addEventListener("click", onConfirm);
        modal.addEventListener("click", onBackdrop);
        document.addEventListener("keydown", onKeydown);
    });
}

function setSelectedTradeSide(side) {
    state.selectedTradeSide = side === "NO" ? "NO" : "YES";
    const yesBtn = document.getElementById("paper-buy-yes-btn");
    const noBtn = document.getElementById("paper-buy-no-btn");
    if (yesBtn) {
        yesBtn.classList.toggle("trade-side-active", state.selectedTradeSide === "YES");
        yesBtn.classList.toggle("trade-side-dim", state.selectedTradeSide !== "YES");
    }
    if (noBtn) {
        noBtn.classList.toggle("trade-side-active", state.selectedTradeSide === "NO");
        noBtn.classList.toggle("trade-side-dim", state.selectedTradeSide !== "NO");
    }
    updateTradeCalculator();
}

function updateTradeCalculator() {
    const input = document.getElementById("trade-size-input");
    const contractsEl = document.getElementById("contracts-estimate");
    const yesBtn = document.getElementById("paper-buy-yes-btn");
    const noBtn = document.getElementById("paper-buy-no-btn");
    const lockMsg = document.getElementById("paper-trade-lock-msg");
    const oddsEl = document.getElementById("payout-odds");
    const winLabelEl = document.getElementById("payout-win-label");
    const winValueEl = document.getElementById("payout-win-value");
    const loseLabelEl = document.getElementById("payout-lose-label");
    const loseValueEl = document.getElementById("payout-lose-value");
    const netProfitEl = document.getElementById("payout-net-profit");
    const leverageWarningEl = document.getElementById("no-leverage-warning");
    const entryFilterWarningEl = document.getElementById("entry-filter-warning");
    if (!input || !contractsEl || !yesBtn || !noBtn) return;

    const amount = Number(input.value);
    const pMarket = toNumberOrNull(state.latestMarket?.p_market);
    const noPrice = Number.isFinite(pMarket) ? (1.0 - pMarket) : NaN;
    const selectedPrice = currentEntryPriceForSide(state.selectedTradeSide);
    const cash = Number(state.portfolio?.cash);
    const hasAmount = Number.isFinite(amount) && amount > 0;
    const hasMarket = Number.isFinite(selectedPrice) && selectedPrice > 0;
    const baseSize = hasAmount ? amount : NaN;
    const edge = state.selectedTradeSide === "YES" ? (1.0 - Number(pMarket)) : Number(pMarket);
    const pRaw = toNumberOrNull(state.latestMarket?.p_raw);
    const mispricingGap =
        state.signalMode === "mispricing" && Number.isFinite(pMarket) && Number.isFinite(pRaw)
            ? Math.abs(pRaw - pMarket)
            : 0.0;
    const edgeMult = getEdgeSizingMultiplier(edge);
    const mispricingMult =
        state.dynamicSizingEnabled && state.signalMode === "mispricing" ? getMispricingGapMultiplier(mispricingGap) : 1.0;
    const totalMult = state.dynamicSizingEnabled ? Math.min(edgeMult * mispricingMult, 3.0) : 1.0;
    let effectiveSize;
    if (!Number.isFinite(baseSize)) {
        effectiveSize = NaN;
    } else if (!state.dynamicSizingEnabled) {
        effectiveSize = baseSize;
    } else {
        const scaled = baseSize * totalMult;
        const cap40 = Number.isFinite(cash) && cash > 0 ? cash * 0.4 : Number.POSITIVE_INFINITY;
        effectiveSize = Math.min(scaled, cap40);
    }
    const tradeCost = Number.isFinite(effectiveSize) ? effectiveSize : NaN;
    const contracts = Number.isFinite(tradeCost) && hasMarket ? tradeCost / selectedPrice : NaN;
    const payoutIfCorrect = Number.isFinite(contracts) ? contracts * 1.0 : NaN;
    if (Number.isFinite(contracts)) {
        const payoutStr = toMoney(payoutIfCorrect);
        if (!state.dynamicSizingEnabled) {
            contractsEl.textContent = `≈ ${contracts.toFixed(2)} contracts · Bet: ${toMoney(tradeCost)} · Payout: ${payoutStr}`;
        } else if (state.signalMode === "mispricing" && mispricingGap >= 0.2) {
            contractsEl.textContent = `≈ ${contracts.toFixed(2)} contracts · Bet: ${toMoney(
                tradeCost,
            )} (${edgeMult.toFixed(1)}x edge + ${mispricingMult.toFixed(2)}x mispricing = ${totalMult.toFixed(
                2,
            )}x) · Payout: ${payoutStr}`;
        } else if (state.signalMode === "mispricing") {
            contractsEl.textContent = `≈ ${contracts.toFixed(2)} contracts · Bet: ${toMoney(
                tradeCost,
            )} (${edgeMult.toFixed(1)}x scaling) · Payout: ${payoutStr}`;
        } else {
            contractsEl.textContent = `≈ ${contracts.toFixed(2)} contracts · Bet: ${toMoney(
                tradeCost,
            )} (${edgeMult.toFixed(1)}x edge scaling) · Payout: ${payoutStr}`;
        }
    } else {
        contractsEl.textContent = "≈ -- contracts · Cost -- · Payout: --";
    }

    if (oddsEl && winLabelEl && winValueEl && loseLabelEl && loseValueEl && netProfitEl) {
        const side = state.selectedTradeSide;
        const oddsPct = side === "YES"
            ? (Number.isFinite(pMarket) ? pMarket * 100 : NaN)
            : (Number.isFinite(pMarket) ? (1.0 - pMarket) * 100 : NaN);
        oddsEl.textContent = Number.isFinite(oddsPct)
            ? (side === "YES" ? `${oddsPct.toFixed(1)}% chance of YES` : `${oddsPct.toFixed(1)}% chance of NO`)
            : "-- chance";

        const payout = Number.isFinite(payoutIfCorrect) ? payoutIfCorrect : NaN;
        const netProfit = Number.isFinite(payout) && Number.isFinite(tradeCost) ? (payout - tradeCost) : NaN;

        if (side === "YES") {
            winLabelEl.textContent = "Payout if YES wins";
            loseLabelEl.textContent = "Payout if YES loses";
        } else {
            winLabelEl.textContent = "Payout if NO wins";
            loseLabelEl.textContent = "Payout if NO loses";
        }
        winValueEl.textContent = Number.isFinite(payout) ? `+$${payout.toFixed(2)}` : "--";
        loseValueEl.textContent = "$0.00";
        if (Number.isFinite(netProfit)) {
            const sign = netProfit >= 0 ? "+" : "-";
            netProfitEl.textContent = `${sign}$${Math.abs(netProfit).toFixed(2)}`;
            netProfitEl.classList.toggle("text-success", netProfit >= 0);
            netProfitEl.classList.toggle("text-danger", netProfit < 0);
        } else {
            netProfitEl.textContent = "--";
            netProfitEl.classList.remove("text-success", "text-danger");
        }
    }
    if (leverageWarningEl) {
        const showWarning = state.selectedTradeSide === "NO" && Number.isFinite(noPrice) && noPrice < 0.10;
        leverageWarningEl.classList.toggle("hidden", !showWarning);
    }
    if (entryFilterWarningEl) {
        let warningText = "";
        if (state.selectedTradeSide === "YES" && Number.isFinite(pMarket) && pMarket > Number(state.maxEntryYes)) {
            warningText = `⚠ Entry filter active — this trade would not auto-fire (${(pMarket * 100).toFixed(1)}¢ > max ${(Number(state.maxEntryYes) * 100).toFixed(1)}¢)`;
        } else if (state.selectedTradeSide === "NO" && Number.isFinite(noPrice) && noPrice > Number(state.maxEntryNo)) {
            warningText = `⚠ Entry filter active — this trade would not auto-fire (${(noPrice * 100).toFixed(1)}¢ > max ${(Number(state.maxEntryNo) * 100).toFixed(1)}¢)`;
        }
        entryFilterWarningEl.textContent = warningText || "⚠ Entry filter active — this trade would not auto-fire.";
        entryFilterWarningEl.classList.toggle("hidden", !warningText);
    }

    const baseDisabled = !hasAmount || !hasMarket || !state.paperEnabled;
    const tooCloseToExpiry = Number.isFinite(state.secondsToClose) && state.secondsToClose < 60;
    yesBtn.disabled = baseDisabled || tooCloseToExpiry || (Number.isFinite(cash) && cash < amount);
    noBtn.disabled = baseDisabled || tooCloseToExpiry || (Number.isFinite(cash) && cash < amount);
    yesBtn.textContent = tooCloseToExpiry ? "🔒 BUY YES" : "BUY YES";
    noBtn.textContent = tooCloseToExpiry ? "🔒 BUY NO" : "BUY NO";
    if (lockMsg) lockMsg.classList.toggle("hidden", !tooCloseToExpiry);
}

async function placeTrade(side, options = {}) {
    setSelectedTradeSide(side);
    const input = document.getElementById("trade-size-input");
    const amount = Number(input?.value);
    const ticker = state.latestMarket?.ticker;
    if (!ticker) return showToast("No active market found", "error");
    if (!Number.isFinite(amount) || amount <= 0) return showToast("Enter a valid trade size", "error");
    const price = currentEntryPriceForSide(side);
    if (!Number.isFinite(price) || price <= 0) return showToast("No valid market price available", "error");
    if (Number.isFinite(state.secondsToClose) && state.secondsToClose < 60) return showToast("Trading locked — less than 60s to close", "error");
    const skipRiskConfirm = Boolean(options.skipRiskConfirm);
    if (side === "NO" && !skipRiskConfirm && Number.isFinite(state.secondsToClose) && state.secondsToClose > 120) {
        const confirmed = await confirmHighRiskNoTrade();
        if (!confirmed) {
            showToast("NO trade cancelled", "error");
            return;
        }
    }

    const contracts = amount / price;
    const payload = await apiFetch("/api/paper/trade", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({
            side,
            contracts,
            ticker,
            seconds_to_close: state.secondsToClose,
            dollar_amount: amount,
        }),
    });
    if (!payload || !payload.success) return showToast(payload?.error || "Trade failed", "error");
    showToast(`Bought ${contracts.toFixed(2)} ${side} contracts for $${amount.toFixed(2)}`, "success");
    await fetchPortfolio();
    await fetchLiveSnapshot();
}

async function toggleAutoTrade(enabled) {
    const prev = state.autoTradeEnabled;
    const data = await apiFetch("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ auto_trade_enabled: enabled ? "true" : "false" }),
    });
    if (!data) {
        state.autoTradeEnabled = prev;
        const autoToggle = document.getElementById("auto-trade-toggle");
        if (autoToggle) autoToggle.checked = prev;
        syncAutoTradePill();
        return showToast("Failed to update auto-trader", "error");
    }
    state.autoTradeEnabled = enabled;
    syncAutoTradePill();
}

async function initPaperTradingState() {
    const settings = await apiFetch("/api/settings", { headers: { Accept: "application/json" } });
    if (!settings) return;
    applyPaperEnabled(settings?.paper_trading_enabled === "true");
    state.autoTradeEnabled = settings?.auto_trade_enabled === "true";
    const tradeSizeInput = document.getElementById("trade-size-input");
    const savedTradeSize = Number(settings?.paper_trade_size);
    if (tradeSizeInput && Number.isFinite(savedTradeSize) && savedTradeSize > 0) tradeSizeInput.value = savedTradeSize;
    const autoToggle = document.getElementById("auto-trade-toggle");
    if (autoToggle) {
        autoToggle.checked = state.autoTradeEnabled;
        autoToggle.disabled = !state.paperEnabled;
    }
    const maxEntryYes = Number(settings?.max_entry_price_yes);
    const maxEntryNo = Number(settings?.max_entry_price_no);
    const maxReversalRisk = Number(settings?.max_reversal_risk);
    const mispricingThreshold = Number(settings?.mispricing_threshold);
    {
        const modeRaw = (settings?.signal_mode || "agreement").toLowerCase();
        const normalizedMode = modeRaw === "ensemble_vote" ? "ensemble" : modeRaw;
        state.signalMode = ["agreement", "mispricing", "ensemble"].includes(normalizedMode) ? normalizedMode : "agreement";
    }
    state.mispricingThreshold = Number.isFinite(mispricingThreshold) ? mispricingThreshold : 0.10;
    state.dynamicSizingEnabled = (settings?.dynamic_sizing_enabled || "false") === "true";
    state.maxEntryYes = Number.isFinite(maxEntryYes) ? maxEntryYes : 0.85;
    state.maxEntryNo = Number.isFinite(maxEntryNo) ? maxEntryNo : 0.85;
    state.maxReversalRisk = Number.isFinite(maxReversalRisk) ? maxReversalRisk : 0.65;
    syncAutoTradePill();
    updateTradeCalculator();
    renderActiveModeFromSettings(settings);
    if (state.latestMarket) renderSignalIntelFromLive(state.latestMarket);
    updateVolatilityGuardUI();
}

async function persistPaperTradeSize(sizeValue) {
    const saved = await apiFetch("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ paper_trade_size: String(sizeValue) }),
    });
    if (saved) showTradeSizeSavedIndicator();
}

function setupPaperTradingControls() {
    const tradeInput = document.getElementById("trade-size-input");
    const yesBtn = document.getElementById("paper-buy-yes-btn");
    const noBtn = document.getElementById("paper-buy-no-btn");
    const resetBtn = document.getElementById("paper-reset-btn");
    const modal = document.getElementById("reset-modal");
    const cancelBtn = document.getElementById("paper-reset-cancel-btn");
    const confirmBtn = document.getElementById("paper-reset-confirm-btn");
    const balanceInput = document.getElementById("paper-reset-balance");
    const autoTradeToggle = document.getElementById("auto-trade-toggle");
    const noTradeRiskModal = document.getElementById("no-trade-risk-modal");

    const closeResetModal = () => {
        if (modal) modal.style.display = "none";
    };
    const openResetModal = () => {
        if (modal) modal.style.display = "flex";
    };

    if (modal) modal.style.display = "none";
    if (noTradeRiskModal) noTradeRiskModal.style.display = "none";
    if (autoTradeToggle) autoTradeToggle.addEventListener("change", () => toggleAutoTrade(autoTradeToggle.checked));
    if (tradeInput) {
        tradeInput.addEventListener("input", () => {
            updateTradeCalculator();
            const parsed = Number(tradeInput.value);
            if (!Number.isFinite(parsed) || parsed <= 0) return;
            if (state.tradeSizeSaveTimerId) window.clearTimeout(state.tradeSizeSaveTimerId);
            state.tradeSizeSaveTimerId = window.setTimeout(() => persistPaperTradeSize(parsed), 500);
        });
        tradeInput.addEventListener("blur", () => {
            const parsed = Number(tradeInput.value);
            if (Number.isFinite(parsed) && parsed > 0) persistPaperTradeSize(parsed);
        });
    }
    if (yesBtn) yesBtn.addEventListener("click", () => {
        setSelectedTradeSide("YES");
        placeTrade("YES");
    });
    if (noBtn) noBtn.addEventListener("click", () => {
        setSelectedTradeSide("NO");
        placeTrade("NO");
    });
    if (resetBtn && modal) resetBtn.addEventListener("click", openResetModal);
    if (cancelBtn && modal) cancelBtn.addEventListener("click", closeResetModal);
    if (confirmBtn && modal) {
        confirmBtn.addEventListener("click", async () => {
            const startingBalance = Number(balanceInput?.value || 100);
            const payload = await apiFetch("/api/paper/reset", {
                method: "POST",
                headers: { "Content-Type": "application/json", Accept: "application/json" },
                body: JSON.stringify({ starting_balance: startingBalance }),
            });
            closeResetModal();
            if (!payload) return showToast("Failed to reset portfolio", "error");
            showToast("Portfolio reset", "success");
            await fetchPortfolio();
            await fetchLiveSnapshot();
            await fetchDashboardRecentTrades();
            await refreshDashboardIntelBlocks();
        });
    }
    if (modal) modal.addEventListener("click", (event) => {
        if (event.target === modal) closeResetModal();
    });
    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") closeResetModal();
    });
    setSelectedTradeSide("YES");
}

function startCountdownTimer() {
    setInterval(() => {
        if (state.secondsToClose === null) return;
        state.secondsToClose -= 1;
        if (state.secondsToClose < 0) return;
        renderMarketInfo(state.latestMarket || {});
        renderPaperSignalPanel(state.latestMarket || {});
        renderFreshness(state.latestMarket || {});
        updateTradeCalculator();
    }, 1000);
}

function startLiveClock() {
    const clockEl = document.getElementById("live-clock");
    if (!clockEl) return;
    const render = () => {
        clockEl.textContent = formatSignalTime(new Date().toISOString());
    };
    render();
    setInterval(render, 1000);
}

async function bootstrapDashboard() {
    if (!document.getElementById("probability-chart")) {
        startLiveClock();
        return;
    }
    startLiveClock();
    setupPaperTradingControls();
    setupChartToggles();
    await initPaperTradingState();
    startCountdownTimer();
    await bootstrapChartSeed();
    await fetchWindowSettings();
    await fetchLiveSnapshot();
    await fetchMarketPrices();
    await fetchPortfolio();
    await fetchPositionsForToasts();
    await fetchResolvedForToasts();
    await fetchDashboardRecentTrades();
    await refreshDashboardIntelBlocks();

    setInterval(() => fetchPortfolio(), DASHBOARD_PORTFOLIO_POLL_MS);
    setInterval(() => fetchLiveSnapshot(), LIVE_SNAPSHOT_POLL_MS);
    setInterval(() => fetchMarketPrices(), 10000);
    setInterval(() => fetchPositionsForToasts(), 10000);
    setInterval(() => fetchResolvedForToasts(), 15000);
    setInterval(() => {
        fetchDashboardRecentTrades();
        refreshDashboardIntelBlocks();
    }, DASHBOARD_INTEL_POLL_MS);
}

window.addEventListener("DOMContentLoaded", () => {
    highlightActiveNav();
    bootstrapDashboard();
});
