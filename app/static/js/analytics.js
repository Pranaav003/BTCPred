const analyticsState = {
    charts: {},
    signalRows: [],
    signalPage: 1,
    signalPageSize: 25,
    signalRefreshTimer: null,
    lastSettingsPayload: null,
};

async function apiFetch(url, options = {}) {
    try {
        const res = await fetch(url, options);
        if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
        return await res.json();
    } catch (err) {
        console.error("API fetch failed:", url, err);
        return null;
    }
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

function fmtPercent(value) {
    if (value === null || value === undefined) return "--";
    return `${(Number(value) * 100).toFixed(1)}%`;
}

function fmtPnl(value) {
    if (value === null || value === undefined) return "--";
    return `${Number(value) >= 0 ? "+" : ""}${Number(value).toFixed(3)}`;
}

function fmtPnlDollar(value) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return "--";
    const n = Number(value);
    const sign = n >= 0 ? "+" : "−";
    return `${sign}$${Math.abs(n).toFixed(3)}`;
}

function regionLabel(region) {
    const map = {
        agree_yes: "Both YES",
        agree_no: "Both NO",
        market_yes_raw_no: "Mkt YES / Mdl NO",
        market_no_raw_yes: "Mkt NO / Mdl YES",
        model_bullish: "Model Bullish",
        model_bearish: "Model Bearish",
        no_agreement: "No Agreement",
    };
    return map[region] || region || "--";
}

function regionClass(region) {
    if (region === "agree_yes") return "text-success";
    if (region === "agree_no") return "text-info";
    if (region === "model_bullish") return "text-success";
    if (region === "model_bearish") return "text-danger";
    if (region === "market_yes_raw_no" || region === "market_no_raw_yes") return "text-warning";
    return "text-muted";
}

function setMetric(id, value, klass = "") {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = value;
    el.classList.remove("text-success", "text-danger");
    if (klass) el.classList.add(klass);
}

function toggleLoading(isLoading) {
    document.querySelectorAll(".loading-card").forEach((el) => {
        el.classList.toggle("is-loading", isLoading);
    });
}

function accuracyClassLive(acc) {
    if (acc === null || acc === undefined || !Number.isFinite(Number(acc))) return "text-muted";
    return Number(acc) >= 0.8 ? "text-success" : "text-danger";
}

function evTagline(avg) {
    const v = Number(avg);
    if (!Number.isFinite(v)) return "—";
    if (v > 0.05) return "Positive EV";
    if (v < -0.02) return "Negative EV";
    return "Near-zero EV";
}

function findRegionRow(regions, id) {
    if (!Array.isArray(regions)) return null;
    return regions.find((r) => r && r.agreement_region === id) || null;
}

function liveBlockTableHtml(row) {
    if (!row) {
        return '<p class="text-muted" style="margin:0">No data</p>';
    }
    const rc = row.resolved_count ?? 0;
    const acc = row.accuracy;
    const accStr = acc !== null && acc !== undefined && Number.isFinite(Number(acc)) ? fmtPercent(acc) : "--";
    const accK = accuracyClassLive(acc);
    return `<table class="mp-mini-table"><thead><tr><th>Count</th><th>Accuracy</th><th>Avg PnL</th><th>Total PnL</th></tr></thead><tbody><tr>
<td class="mono">${rc}</td>
<td class="mono ${accK}">${accStr}</td>
<td class="mono">${fmtPnlDollar(row.avg_pnl)}</td>
<td class="mono">${fmtPnlDollar(row.total_pnl)}</td>
</tr></tbody></table>`;
}

function renderMispricingStrategyCard(regions, backtestPayload, settings, metrics) {
    analyticsState.lastSettingsPayload = settings || null;
    const modeRaw = (settings?.signal_mode || "agreement").toLowerCase();
    const mode = modeRaw === "ensemble_vote" ? "ensemble" : modeRaw;

    const titleEl = document.getElementById("current-strategy-title");
    const liveSectionTitle = document.getElementById("mp-live-section-title");
    const b1Title = document.getElementById("mp-block-1-title");
    const b2Title = document.getElementById("mp-block-2-title");
    const b3Wrap = document.getElementById("mp-third-block");
    const b3Title = document.getElementById("mp-block-3-title");
    const b3Stats = document.getElementById("mp-third-stats");
    if (titleEl) {
        if (mode === "ensemble") titleEl.textContent = "Ensemble Vote Strategy — Live Performance";
        else if (mode === "mispricing") titleEl.textContent = "Mispricing Strategy — Live Performance";
        else titleEl.textContent = "Agreement Strategy — Live Performance";
    }
    if (liveSectionTitle) liveSectionTitle.textContent = "Live performance";

    const sub = document.getElementById("mispricing-strategy-subtitle");
    if (sub) {
        const t = settings && settings.mispricing_threshold !== undefined
            ? Number(settings.mispricing_threshold)
            : 0.2;
        const pct = Number.isFinite(t) ? t * 100 : 20;
        sub.textContent = `Model divergence signals (${pct.toFixed(0)}%+ gap)`;
    }

    const agreeY = findRegionRow(regions, "agree_yes");
    const agreeN = findRegionRow(regions, "agree_no");
    const mb = findRegionRow(regions, "model_bullish");
    const ms = findRegionRow(regions, "model_bearish");
    const bullishEl = document.getElementById("mp-bullish-stats");
    const bearishEl = document.getElementById("mp-bearish-stats");
    let r1 = null;
    let r2 = null;
    let r3 = null;
    if (mode === "ensemble") {
        if (b1Title) b1Title.textContent = "Agreement YES";
        if (b2Title) b2Title.textContent = "Mispricing YES";
        if (b3Title) b3Title.textContent = "Mispricing NO";
        if (b3Wrap) b3Wrap.classList.remove("hidden");
        r1 = agreeY;
        r2 = mb;
        r3 = ms;
    } else if (mode === "mispricing") {
        if (b1Title) b1Title.textContent = "Model Bullish (BUY YES)";
        if (b2Title) b2Title.textContent = "Model Bearish (BUY NO)";
        if (b3Wrap) b3Wrap.classList.add("hidden");
        r1 = mb;
        r2 = ms;
    } else {
        if (b1Title) b1Title.textContent = "Agreement YES";
        if (b2Title) b2Title.textContent = "Agreement NO";
        if (b3Wrap) b3Wrap.classList.add("hidden");
        r1 = agreeY;
        r2 = agreeN;
    }
    if (bullishEl) bullishEl.innerHTML = liveBlockTableHtml(r1);
    if (bearishEl) bearishEl.innerHTML = liveBlockTableHtml(r2);
    if (b3Stats && b3Wrap && !b3Wrap.classList.contains("hidden")) b3Stats.innerHTML = liveBlockTableHtml(r3);

    const rr1 = Number(r1?.resolved_count) || 0;
    const rr2 = Number(r2?.resolved_count) || 0;
    const rr3 = Number(r3?.resolved_count) || 0;
    const totalR = rr1 + rr2 + rr3;
    const correctB = (Number(r1?.accuracy) || 0) * rr1;
    const correctS = (Number(r2?.accuracy) || 0) * rr2;
    const correctT = (Number(r3?.accuracy) || 0) * rr3;
    const accC = totalR > 0 ? (correctB + correctS + correctT) / totalR : null;
    const totalPnlC = (Number(r1?.total_pnl) || 0) + (Number(r2?.total_pnl) || 0) + (Number(r3?.total_pnl) || 0);
    const avgC = totalR > 0 ? totalPnlC / totalR : null;
    const combinedEl = document.getElementById("mp-combined-line");
    if (combinedEl) {
        combinedEl.classList.remove("text-success", "text-danger", "text-muted");
        if (!totalR) {
            combinedEl.classList.add("text-muted");
            combinedEl.textContent = "Combined: 0 resolved trades";
        } else {
            const accStr = accC != null && Number.isFinite(accC) ? fmtPercent(accC) : "--";
            if (Number.isFinite(avgC)) combinedEl.classList.add(avgC >= 0 ? "text-success" : "text-danger");
            else combinedEl.classList.add("text-muted");
            combinedEl.textContent = `Combined: ${totalR} trades · ${accStr} · Avg PnL: ${fmtPnlDollar(avgC)} · Total: ${fmtPnlDollar(totalPnlC)}`;
        }
    }
    const entryFilterLine = document.getElementById("mp-entry-filter-line");
    if (entryFilterLine) {
        const blockedToday = Number(metrics?.entry_filtered_today || 0);
        const blockedTotal = Number(metrics?.entry_filtered_total || 0);
        const maxEntry = Number(settings?.max_entry_price_yes);
        const maxTxt = Number.isFinite(maxEntry) ? `${(maxEntry * 100).toFixed(0)}¢` : "--";
        entryFilterLine.textContent = `Entry filtered: ${blockedToday} signals blocked today (${blockedTotal} total). Reason: market price exceeded max entry ${maxTxt}.`;
    }

    const buckets = Array.isArray(backtestPayload?.buckets) ? backtestPayload.buckets : [];
    const barRoot = document.getElementById("mp-backtest-bars");
    if (barRoot) {
        if (!buckets.length) {
            barRoot.innerHTML = '<p class="text-muted" style="margin:0">Insufficient backtest data</p>';
        } else {
            const order = ["0.10-0.15", "0.15-0.20", "0.20+"];
            const byLabel = new Map(buckets.map((b) => [b.bucket, b]));
            barRoot.innerHTML = order
                .map((key) => {
                    const b = byLabel.get(key) || { bucket: key, accuracy: null, avg_pnl: null, count: 0 };
                    const acc = b.accuracy;
                    const w = acc !== null && acc !== undefined && Number.isFinite(Number(acc))
                        ? Math.min(100, Math.max(0, Number(acc) * 100))
                        : 0;
                    const accP = acc !== null && acc !== undefined && Number.isFinite(Number(acc)) ? fmtPercent(acc) : "--";
                    const isHighlight = String(b.bucket || key).replace(/\s/g, "") === "0.20+";
                    const rowClass = isHighlight ? "mp-bar-row mp-bar-row--highlight" : "mp-bar-row";
                    return `<div class="${rowClass}">
  <span class="mp-bar-label">${key}</span>
  <div class="mp-bar-track" aria-hidden="true"><div class="mp-bar-fill" style="width:${w.toFixed(1)}%"></div></div>
  <div class="mp-bar-meta">${accP} accuracy / ${fmtPnlDollar(b.avg_pnl)} avg</div>
</div>`;
                })
                .join("");
        }
    }

    const th = backtestPayload && backtestPayload.threshold !== undefined
        ? Number(backtestPayload.threshold)
        : (settings && settings.mispricing_threshold !== undefined ? Number(settings.mispricing_threshold) : 0.2);
    const recommendEl = document.getElementById("mp-recommend-line");
    if (recommendEl) {
        const bucket20 = buckets.find((b) => String(b.bucket || "").replace(/\s/g, "") === "0.20+");
        const pctTh = Number.isFinite(th) ? th * 100 : 20;
        if (bucket20 && Number(bucket20.count) > 0 && bucket20.accuracy != null) {
            recommendEl.textContent = `Recommended threshold: ${pctTh.toFixed(0)}%+ (backtest: ${fmtPercent(bucket20.accuracy)} acc, ${fmtPnlDollar(bucket20.avg_pnl)} avg)`;
        } else {
            recommendEl.textContent = `Recommended threshold: ${pctTh.toFixed(0)}%+ (0.20+ bucket: insufficient trades for a stable read)`;
        }
    }

    const yAcc = agreeY?.accuracy;
    const yAvg = agreeY?.avg_pnl;
    const nAcc = agreeN?.accuracy;
    const nAvg = agreeN?.avg_pnl;

    const cols = document.getElementById("strategy-compare-cols");
    if (cols) {
        cols.innerHTML = `
<div class="strategy-compare-col strategy-col-agreement">
  <h4>Agreement</h4>
  <p class="strategy-compare-sub">YES</p>
  <p class="strategy-compare-metric mono ${accuracyClassLive(yAcc)}">${yAcc != null ? fmtPercent(yAcc) : "--"} accuracy</p>
  <p class="strategy-compare-metric mono">${yAvg != null ? `${fmtPnlDollar(yAvg)} avg` : "--"}</p>
  <p class="strategy-compare-ev">${evTagline(yAvg)}</p>
</div>
<div class="strategy-compare-col strategy-col-noside">
  <h4>NO-Side</h4>
  <p class="strategy-compare-sub">Agreement</p>
  <p class="strategy-compare-metric mono ${accuracyClassLive(nAcc)}">${nAcc != null ? fmtPercent(nAcc) : "--"} accuracy</p>
  <p class="strategy-compare-metric mono">${nAvg != null ? `${fmtPnlDollar(nAvg)} avg` : "--"}</p>
  <p class="strategy-compare-ev">${evTagline(nAvg)}</p>
</div>
<div class="strategy-compare-col strategy-col-mispricing">
  <h4>Mispricing</h4>
  <p class="strategy-compare-sub">(Divergence)</p>
  <p class="strategy-compare-metric mono ${accuracyClassLive(accC)}">${accC != null ? fmtPercent(accC) : "--"} accuracy</p>
  <p class="strategy-compare-metric mono">${avgC != null && Number.isFinite(avgC) ? `${fmtPnlDollar(avgC)} avg` : "--"}</p>
  <p class="strategy-compare-ev">${evTagline(avgC)}</p>
</div>`;
    }

    const modeActive = mode === "mispricing";
    const statusEl = document.getElementById("mispricing-mode-status");
    const btn = document.getElementById("mispricing-switch-btn");
    const badge = document.getElementById("mispricing-active-badge");
    if (statusEl) {
        if (mode === "ensemble") statusEl.textContent = "Ensemble mode is currently ACTIVE";
        else if (modeActive) statusEl.textContent = "Mispricing mode is currently ACTIVE";
        else statusEl.textContent = "Agreement mode is currently ACTIVE";
    }
    if (btn && badge) {
        if (modeActive) {
            btn.classList.add("hidden");
            badge.classList.remove("hidden");
        } else {
            btn.classList.remove("hidden");
            badge.classList.add("hidden");
        }
    }
}

function showEmpty(chartId, emptyId, show, message = "Insufficient data") {
    const canvas = document.getElementById(chartId);
    const empty = document.getElementById(emptyId);
    if (!canvas || !empty) return;
    empty.textContent = message;
    empty.style.display = show ? "flex" : "none";
    canvas.style.visibility = show ? "hidden" : "visible";
}

function destroyChart(key) {
    if (analyticsState.charts[key]) {
        analyticsState.charts[key].destroy();
        analyticsState.charts[key] = null;
    }
}

function renderProbabilityHistory(history) {
    const canvas = document.getElementById("probability-history-chart");
    if (!canvas) return;
    const key = "probabilityHistory";
    destroyChart(key);
    if (!Array.isArray(history) || !history.length) return;
    analyticsState.charts[key] = new window.Chart(canvas, {
        type: "line",
        data: {
            labels: history.map((row) => row.logged_at?.split("T")[1]?.slice(0, 8) || "--"),
            datasets: [
                {
                    label: "Last Trade (p_market)",
                    data: history.map((row) => Number(row.p_market)),
                    borderColor: "#3b82f6",
                    backgroundColor: "rgba(59,130,246,0.2)",
                    tension: 0.25,
                    pointRadius: 0,
                },
                {
                    label: "Model (p_raw)",
                    data: history.map((row) => Number(row.p_raw)),
                    borderColor: "#f59e0b",
                    backgroundColor: "rgba(245,158,11,0.2)",
                    tension: 0.25,
                    pointRadius: 0,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { ticks: { color: "rgba(228,228,234,0.7)" }, grid: { color: "rgba(30,30,46,0.4)" } },
                y: { min: 0, max: 1, ticks: { color: "rgba(228,228,234,0.7)" }, grid: { color: "rgba(30,30,46,0.4)" } },
            },
            plugins: { legend: { labels: { color: "rgba(228,228,234,0.85)" } } },
        },
    });
}

function renderSummary(metrics) {
    setMetric("metric-total-signals", String(metrics.total_signals ?? "--"));
    setMetric("metric-yes-signals", String(metrics.yes_signals ?? "--"));
    setMetric("metric-resolved", String(metrics.resolved_count ?? "--"));
    setMetric("metric-accuracy", fmtPercent(metrics.accuracy));
    const avgPnl = Number(metrics.avg_pnl);
    const totalPnl = Number(metrics.total_pnl);
    setMetric("metric-avg-pnl", fmtPnl(metrics.avg_pnl), Number.isFinite(avgPnl) ? (avgPnl >= 0 ? "text-success" : "text-danger") : "");
    setMetric("metric-total-pnl", fmtPnl(metrics.total_pnl), Number.isFinite(totalPnl) ? (totalPnl >= 0 ? "text-success" : "text-danger") : "");
}

function renderPnlCurve(curve) {
    const key = "pnlCurve";
    destroyChart(key);
    if (!Array.isArray(curve) || !curve.length) {
        showEmpty("pnl-curve-chart", "pnl-curve-empty", true, "No resolved signals yet");
        return;
    }
    showEmpty("pnl-curve-chart", "pnl-curve-empty", false);
    const labels = curve.map((row) => row.logged_at?.split("T")[1]?.slice(0, 8) || "--");
    const values = curve.map((row) => Number(row.cumulative_pnl || 0));
    const finalValue = values[values.length - 1] || 0;
    const lineColor = finalValue >= 0 ? "#22c55e" : "#ef4444";
    const fillColor = finalValue >= 0 ? "rgba(34, 197, 94, 0.20)" : "rgba(239, 68, 68, 0.20)";
    const zeroLinePlugin = {
        id: "zeroLine",
        afterDraw(chart) {
            const yScale = chart.scales.y;
            const xScale = chart.scales.x;
            if (!yScale || !xScale) return;
            const yPixel = yScale.getPixelForValue(0);
            const { ctx } = chart;
            ctx.save();
            ctx.setLineDash([6, 6]);
            ctx.strokeStyle = "rgba(148, 163, 184, 0.6)";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(xScale.left, yPixel);
            ctx.lineTo(xScale.right, yPixel);
            ctx.stroke();
            ctx.restore();
        },
    };
    analyticsState.charts[key] = new window.Chart(document.getElementById("pnl-curve-chart"), {
        type: "line",
        data: { labels, datasets: [{ label: "Cumulative PnL", data: values, borderColor: lineColor, backgroundColor: fillColor, borderWidth: 2, fill: true, pointRadius: 0, tension: 0.25 }] },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { ticks: { maxTicksLimit: 8, maxRotation: 0, minRotation: 0, color: "rgba(228,228,234,0.7)" }, grid: { color: "rgba(30,30,46,0.4)" } },
                y: { ticks: { color: "rgba(228,228,234,0.7)" }, grid: { color: "rgba(30,30,46,0.4)" } },
            },
            plugins: { legend: { labels: { color: "rgba(228,228,234,0.85)" } } },
        },
        plugins: [zeroLinePlugin],
    });
}

function renderBucketChart(buckets) {
    const key = "bucket";
    destroyChart(key);
    if (!Array.isArray(buckets) || !buckets.length) {
        showEmpty("bucket-accuracy-chart", "bucket-accuracy-empty", true);
        return;
    }
    showEmpty("bucket-accuracy-chart", "bucket-accuracy-empty", false);
    const labels = buckets.map((b) => `${b.entry_bucket}s`);
    analyticsState.charts[key] = new window.Chart(document.getElementById("bucket-accuracy-chart"), {
        data: {
            labels,
            datasets: [
                { type: "bar", label: "Accuracy", data: buckets.map((b) => b.accuracy), backgroundColor: "rgba(59,130,246,0.7)", borderColor: "#3b82f6", borderWidth: 1, yAxisID: "y" },
                { type: "line", label: "Avg PnL", data: buckets.map((b) => b.avg_pnl), borderColor: "#f59e0b", backgroundColor: "rgba(245,158,11,0.2)", yAxisID: "y1", tension: 0.25, pointRadius: 3 },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { min: 0, max: 1, position: "left", ticks: { color: "rgba(228,228,234,0.7)" }, grid: { color: "rgba(30,30,46,0.4)" } },
                y1: { position: "right", ticks: { color: "rgba(228,228,234,0.7)" }, grid: { drawOnChartArea: false } },
                x: { ticks: { color: "rgba(228,228,234,0.7)" }, grid: { color: "rgba(30,30,46,0.3)" } },
            },
            plugins: { legend: { labels: { color: "rgba(228,228,234,0.85)" } } },
        },
    });
}

function renderRegionChart(regions) {
    const key = "region";
    destroyChart(key);
    if (!Array.isArray(regions) || !regions.length) {
        showEmpty("region-performance-chart", "region-performance-empty", true);
        return;
    }
    showEmpty("region-performance-chart", "region-performance-empty", false);
    const colorMap = { agree_yes: "#22c55e", agree_no: "#3b82f6", market_yes_raw_no: "#f59e0b", market_no_raw_yes: "#f59e0b", no_agreement: "#94a3b8" };
    analyticsState.charts[key] = new window.Chart(document.getElementById("region-performance-chart"), {
        type: "bar",
        data: { labels: regions.map((r) => regionLabel(r.agreement_region)), datasets: [{ label: "Accuracy", data: regions.map((r) => r.accuracy), backgroundColor: regions.map((r) => colorMap[r.agreement_region] || "#94a3b8"), borderWidth: 0 }] },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { min: 0, max: 1, ticks: { color: "rgba(228,228,234,0.7)" }, grid: { color: "rgba(30,30,46,0.4)" } },
                y: { ticks: { color: "rgba(228,228,234,0.7)" }, grid: { color: "rgba(30,30,46,0.2)" } },
            },
            plugins: { legend: { display: false } },
        },
    });
}

function renderCutoffChart(cutoffs) {
    const key = "cutoff";
    destroyChart(key);
    if (!Array.isArray(cutoffs) || !cutoffs.length) {
        showEmpty("cutoff-analysis-chart", "cutoff-analysis-empty", true);
        return;
    }
    showEmpty("cutoff-analysis-chart", "cutoff-analysis-empty", false);
    analyticsState.charts[key] = new window.Chart(document.getElementById("cutoff-analysis-chart"), {
        type: "line",
        data: {
            labels: cutoffs.map((c) => c.cutoff.toFixed(2)),
            datasets: [
                { label: "Accuracy", data: cutoffs.map((c) => c.accuracy), borderColor: "#3b82f6", backgroundColor: "rgba(59,130,246,0.2)", yAxisID: "y", tension: 0.25 },
                { label: "Trade Count", data: cutoffs.map((c) => c.count), borderColor: "#f59e0b", backgroundColor: "rgba(245,158,11,0.2)", yAxisID: "y1", tension: 0.25 },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { min: 0, max: 1, position: "left", ticks: { color: "rgba(228,228,234,0.7)" }, grid: { color: "rgba(30,30,46,0.4)" } },
                y1: { position: "right", ticks: { color: "rgba(228,228,234,0.7)" }, grid: { drawOnChartArea: false } },
                x: { ticks: { color: "rgba(228,228,234,0.7)" }, grid: { color: "rgba(30,30,46,0.2)" } },
            },
            plugins: { legend: { labels: { color: "rgba(228,228,234,0.85)" } } },
        },
    });
}

function renderMispricingBacktest(buckets) {
    const key = "mispricingBacktest";
    destroyChart(key);
    if (!Array.isArray(buckets) || !buckets.some((row) => Number(row.count || 0) > 0)) {
        showEmpty("mispricing-backtest-chart", "mispricing-backtest-empty", true, "Insufficient mispricing data");
        return;
    }
    showEmpty("mispricing-backtest-chart", "mispricing-backtest-empty", false);
    analyticsState.charts[key] = new window.Chart(document.getElementById("mispricing-backtest-chart"), {
        data: {
            labels: buckets.map((b) => b.bucket),
            datasets: [
                {
                    type: "bar",
                    label: "Accuracy",
                    data: buckets.map((b) => b.accuracy),
                    backgroundColor: "rgba(59,130,246,0.7)",
                    borderColor: "#3b82f6",
                    borderWidth: 1,
                    yAxisID: "y",
                },
                {
                    type: "line",
                    label: "Avg PnL",
                    data: buckets.map((b) => b.avg_pnl),
                    borderColor: "#22c55e",
                    backgroundColor: "rgba(34,197,94,0.2)",
                    tension: 0.25,
                    pointRadius: 3,
                    yAxisID: "y1",
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { min: 0, max: 1, position: "left", ticks: { color: "rgba(228,228,234,0.7)" }, grid: { color: "rgba(30,30,46,0.4)" } },
                y1: { position: "right", ticks: { color: "rgba(228,228,234,0.7)" }, grid: { drawOnChartArea: false } },
                x: { ticks: { color: "rgba(228,228,234,0.7)" }, grid: { color: "rgba(30,30,46,0.3)" } },
            },
            plugins: { legend: { labels: { color: "rgba(228,228,234,0.85)" } } },
        },
    });
}

function renderRegionTable(regions) {
    const body = document.getElementById("region-detail-body");
    if (!body) return;
    if (!Array.isArray(regions) || !regions.length) {
        body.innerHTML = '<tr><td colspan="6" class="text-muted">Insufficient data</td></tr>';
        return;
    }
    body.innerHTML = regions.map((r) => {
        let accClass = "text-muted";
        if (r.accuracy !== null && r.accuracy !== undefined) {
            if (r.accuracy > 0.7) accClass = "text-success";
            else if (r.accuracy >= 0.5) accClass = "text-warning";
            else accClass = "text-danger";
        }
        const avgClass = r.avg_pnl > 0 ? "text-success" : r.avg_pnl < 0 ? "text-danger" : "text-muted";
        const totalClass = r.total_pnl > 0 ? "text-success" : r.total_pnl < 0 ? "text-danger" : "text-muted";
        return `<tr><td>${regionLabel(r.agreement_region)}</td><td class="mono">${r.count}</td><td class="mono">${r.resolved_count}</td><td class="${accClass} mono">${fmtPercent(r.accuracy)}</td><td class="${avgClass} mono">${fmtPnl(r.avg_pnl)}</td><td class="${totalClass} mono">${fmtPnl(r.total_pnl)}</td></tr>`;
    }).join("");
}

function signalBadge(signal) {
    if (signal === "PAPER BUY YES") return '<span class="badge badge-success">BUY YES</span>';
    if (signal === "PAPER BUY NO") return '<span class="badge badge-danger">BUY NO</span>';
    return '<span class="badge badge-neutral">NO SIGNAL</span>';
}

function outcomeBadge(outcomeYes) {
    if (outcomeYes === true) return '<span class="badge badge-success">YES</span>';
    if (outcomeYes === false) return '<span class="badge badge-danger">NO</span>';
    return '<span class="badge badge-neutral">--</span>';
}

function pnlCell(row) {
    const value = Number(row?.pnl_raw);
    if (!Number.isFinite(value)) return '<span class="text-muted mono">--</span>';
    if (value > 0) return `<span class="text-success mono">${row.pnl}</span>`;
    if (value < 0) return `<span class="text-danger mono">${row.pnl}</span>`;
    return `<span class="text-muted mono">${row.pnl}</span>`;
}

function correctCell(value) {
    if (value === true) return '<span class="text-success">✓</span>';
    if (value === false) return '<span class="text-danger">✗</span>';
    return '<span class="text-muted">--</span>';
}

function rowClass(value) {
    if (value === true) return "signal-row-correct";
    if (value === false) return "signal-row-incorrect";
    return "";
}

function renderSignalHistoryTable() {
    const body = document.getElementById("signal-history-body");
    const range = document.getElementById("signals-range");
    const prev = document.getElementById("signals-history-prev");
    const next = document.getElementById("signals-history-next");
    if (!body || !range || !prev || !next) return;

    const total = analyticsState.signalRows.length;
    const totalPages = Math.max(1, Math.ceil(total / analyticsState.signalPageSize));
    analyticsState.signalPage = Math.max(1, Math.min(analyticsState.signalPage, totalPages));

    if (total === 0) {
        body.innerHTML = '<tr><td colspan="10" class="text-muted">Insufficient data</td></tr>';
        range.textContent = "Showing 0-0 of 0 signals";
        prev.disabled = true;
        next.disabled = true;
        return;
    }

    const startIdx = (analyticsState.signalPage - 1) * analyticsState.signalPageSize;
    const endIdx = Math.min(total, startIdx + analyticsState.signalPageSize);
    const pageRows = analyticsState.signalRows.slice(startIdx, endIdx);
    body.innerHTML = pageRows.map((row) => `
        <tr class="${rowClass(row.outcome_correct)}">
            <td class="mono">${formatSignalTime(row.logged_at)}</td>
            <td class="mono">${row.ticker || "--"}</td>
            <td class="mono">${row.entry_bucket ?? "--"}</td>
            <td class="mono">${fmtPercent(row.p_market)}</td>
            <td class="mono">${fmtPercent(row.p_raw)}</td>
            <td class="${regionClass(row.agreement_region)}">${regionLabel(row.agreement_region)}</td>
            <td>${signalBadge(row.signal)}</td>
            <td>${outcomeBadge(row.outcome_yes)}</td>
            <td>${pnlCell(row)}</td>
            <td>${correctCell(row.outcome_correct)}</td>
        </tr>
    `).join("");
    range.textContent = `Showing ${startIdx + 1}-${endIdx} of ${total} signals`;
    prev.disabled = analyticsState.signalPage <= 1;
    next.disabled = analyticsState.signalPage >= totalPages;
}

function updateLastUpdated() {
    const el = document.getElementById("signal-history-updated");
    if (!el) return;
    el.textContent = `Last updated: ${new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: true,
    })}`;
}

function setupSignalHistoryControls() {
    const prev = document.getElementById("signals-history-prev");
    const next = document.getElementById("signals-history-next");
    if (prev) prev.addEventListener("click", () => { analyticsState.signalPage -= 1; renderSignalHistoryTable(); });
    if (next) next.addEventListener("click", () => { analyticsState.signalPage += 1; renderSignalHistoryTable(); });
}

function setupMispricingModeButton() {
    const btn = document.getElementById("mispricing-switch-btn");
    if (!btn) return;
    btn.addEventListener("click", async () => {
        btn.disabled = true;
        try {
            const res = await apiFetch("/api/settings", {
                method: "POST",
                headers: { "Content-Type": "application/json", Accept: "application/json" },
                body: JSON.stringify({ signal_mode: "mispricing" }),
            });
            if (res) {
                await fetchAllAnalytics();
            }
        } finally {
            btn.disabled = false;
        }
    });
}

function setupExportDropdown() {
    const wrap = document.getElementById("analytics-export-dropdown");
    const btn = document.getElementById("analytics-export-btn");
    const menu = document.getElementById("analytics-export-menu");
    if (!wrap || !btn || !menu) return;
    const closeMenu = () => menu.classList.add("hidden");
    btn.addEventListener("click", (event) => {
        event.stopPropagation();
        menu.classList.toggle("hidden");
    });
    document.getElementById("export-full-json-btn")?.addEventListener("click", () => {
        window.location.href = "/api/export/full";
        closeMenu();
    });
    document.getElementById("export-csv-zip-btn")?.addEventListener("click", () => {
        window.location.href = "/api/export/csv";
        closeMenu();
    });
    document.getElementById("export-training-csv-btn")?.addEventListener("click", () => {
        window.location.href = "/api/export/training-csv";
        closeMenu();
    });
    document.getElementById("export-live-training-csv-btn")?.addEventListener("click", () => {
        window.location.href = "/api/export/live-training-data";
        closeMenu();
    });
    document.addEventListener("click", (event) => {
        if (!wrap.contains(event.target)) closeMenu();
    });
    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") closeMenu();
    });
}

async function fetchAllAnalytics() {
    if (!document.getElementById("pnl-curve-chart")) return;
    toggleLoading(true);
    try {
        const [metrics, pnlCurve, byBucket, byRegion, byCutoff, signalsPayload, probabilityHistory, mispricingBacktest, settings] = await Promise.all([
            apiFetch("/api/metrics", { headers: { Accept: "application/json" } }),
            apiFetch("/api/analytics/pnl-curve", { headers: { Accept: "application/json" } }),
            apiFetch("/api/analytics/accuracy-by-bucket", { headers: { Accept: "application/json" } }),
            apiFetch("/api/analytics/agreement-regions", { headers: { Accept: "application/json" } }),
            apiFetch("/api/analytics/accuracy-by-cutoff", { headers: { Accept: "application/json" } }),
            apiFetch("/api/signals?limit=200", { headers: { Accept: "application/json" } }),
            apiFetch("/api/signals/history?limit=200", { headers: { Accept: "application/json" } }),
            apiFetch("/api/analytics/mispricing-backtest", { headers: { Accept: "application/json" } }),
            apiFetch("/api/settings", { headers: { Accept: "application/json" } }),
        ]);
        if (!metrics || !pnlCurve || !byBucket || !byRegion || !byCutoff || !signalsPayload || !mispricingBacktest || !settings) {
            throw new Error("One or more analytics endpoints failed");
        }
        renderMispricingStrategyCard(byRegion.regions || [], mispricingBacktest, settings, metrics);
        renderSummary(metrics);
        renderPnlCurve(pnlCurve.curve || []);
        renderBucketChart(byBucket.buckets || []);
        renderRegionChart(byRegion.regions || []);
        renderCutoffChart(byCutoff.cutoffs || []);
        renderRegionTable(byRegion.regions || []);
        renderProbabilityHistory(probabilityHistory?.history || []);
        renderMispricingBacktest(mispricingBacktest?.buckets || []);
        analyticsState.signalRows = Array.isArray(signalsPayload.signals) ? signalsPayload.signals : [];
        renderSignalHistoryTable();
        updateLastUpdated();
    } catch (error) {
        console.error("Failed to load analytics:", error);
        showEmpty("pnl-curve-chart", "pnl-curve-empty", true);
        showEmpty("bucket-accuracy-chart", "bucket-accuracy-empty", true);
        showEmpty("region-performance-chart", "region-performance-empty", true);
        showEmpty("cutoff-analysis-chart", "cutoff-analysis-empty", true);
        showEmpty("mispricing-backtest-chart", "mispricing-backtest-empty", true);
        renderRegionTable([]);
        renderSignalHistoryTable();
    } finally {
        toggleLoading(false);
    }
}

window.addEventListener("DOMContentLoaded", () => {
    setupSignalHistoryControls();
    setupExportDropdown();
    setupMispricingModeButton();
    fetchAllAnalytics();
    analyticsState.signalRefreshTimer = window.setInterval(() => {
        fetchAllAnalytics();
    }, 60000);
    document.addEventListener("visibilitychange", () => {
        if (!document.hidden) {
            fetchAllAnalytics();
        }
    });
});
