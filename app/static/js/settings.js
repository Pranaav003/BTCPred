const settingsState = {
    lastSavedAt: null,
    saveTimer: null,
    riskProfiles: {},
    activeRiskProfile: "moderate",
    settings: {},
    enableNoSignals: false,
    signalMode: "agreement",
    mispricingThreshold: 0.10,
    maxEntryPriceYes: 0.85,
    maxEntryPriceNo: 0.85,
    maxReversalRisk: 0.65,
    dynamicSizingEnabled: false,
    entryFilterSaveTimerId: null,
    reversalRiskSaveTimerId: null,
};

function showSettingsToast(message, type = "success") {
    const toast = document.getElementById("settings-toast");
    if (!toast) return;
    toast.textContent = message;
    toast.className = `toast show ${type === "error" ? "toast-error" : "toast-success"}`;
    if (settingsState.saveTimer) window.clearTimeout(settingsState.saveTimer);
    settingsState.saveTimer = window.setTimeout(() => {
        toast.className = "toast";
    }, 3000);
}

function setCutoffLabel(sliderId, valueId) {
    const slider = document.getElementById(sliderId);
    const value = document.getElementById(valueId);
    if (!slider || !value) return;
    value.textContent = `${(Number(slider.value) * 100).toFixed(1)}%`;
}

function getActiveProfileSnapshot() {
    const name = settingsState.activeRiskProfile || "moderate";
    const p = settingsState.riskProfiles?.[name];
    if (!p) {
        return { yes_cutoff: 0.65, no_cutoff: 0.35, min_seconds: 60, max_seconds: 300 };
    }
    return {
        yes_cutoff: Number(p.yes_cutoff),
        no_cutoff: Number(p.no_cutoff),
        min_seconds: Number(p.min_seconds),
        max_seconds: Number(p.max_seconds),
    };
}

function updateOverrideProfileHints() {
    const snap = getActiveProfileSnapshot();
    const yesHint = document.getElementById("override-yes-profile-hint");
    const noHint = document.getElementById("override-no-profile-hint");
    const minHint = document.getElementById("override-min-profile-hint");
    const maxHint = document.getElementById("override-max-profile-hint");
    if (yesHint) {
        yesHint.textContent = `(profile default: ${(snap.yes_cutoff * 100).toFixed(0)}%)`;
    }
    if (noHint) {
        noHint.textContent = `(profile default: ${(snap.no_cutoff * 100).toFixed(0)}%)`;
    }
    if (minHint) {
        minHint.textContent = `(profile default: ${Math.round(snap.min_seconds)}s)`;
    }
    if (maxHint) {
        maxHint.textContent = `(profile default: ${Math.round(snap.max_seconds)}s)`;
    }
}

function updateSecondsOverrideDisplays() {
    const minIn = document.getElementById("min-seconds-input");
    const maxIn = document.getElementById("max-seconds-input");
    const minDis = document.getElementById("min-seconds-display");
    const maxDis = document.getElementById("max-seconds-display");
    if (minIn && minDis) {
        const v = Number(minIn.value);
        minDis.textContent = Number.isFinite(v) ? `${Math.round(v)}s` : "--";
    }
    if (maxIn && maxDis) {
        const v = Number(maxIn.value);
        maxDis.textContent = Number.isFinite(v) ? `${Math.round(v)}s` : "--";
    }
}

function updateMispricingThresholdLabel() {
    const slider = document.getElementById("mispricing-threshold-slider");
    const value = document.getElementById("mispricing-threshold-value");
    const helper = document.getElementById("mispricing-threshold-helper");
    if (!slider || !value) return;
    const pctText = `${(Number(slider.value) * 100).toFixed(1)}%`;
    value.textContent = pctText;
    if (helper) helper.textContent = pctText;
}

function updateEntryFilterPreview() {
    const yesSlider = document.getElementById("max-entry-yes-slider");
    const noSlider = document.getElementById("max-entry-no-slider");
    const yesLabel = document.getElementById("max-entry-yes-label");
    const noLabel = document.getElementById("max-entry-no-label");
    const yesCalc = document.getElementById("max-entry-yes-calc");
    const noCalc = document.getElementById("max-entry-no-calc");
    const noFloor = document.getElementById("max-entry-no-market-floor");
    if (!yesSlider || !noSlider || !yesLabel || !noLabel || !yesCalc || !noCalc || !noFloor) return;

    const amount = 20.0;
    const yesEntry = Number(yesSlider.value);
    const noEntry = Number(noSlider.value);
    settingsState.maxEntryPriceYes = Number.isFinite(yesEntry) ? yesEntry : 0.85;
    settingsState.maxEntryPriceNo = Number.isFinite(noEntry) ? noEntry : 0.85;

    const yesContracts = amount / Math.max(settingsState.maxEntryPriceYes, 0.0001);
    const yesWinProfit = yesContracts * (1.0 - settingsState.maxEntryPriceYes);
    const yesLoss = amount;
    const yesRecoveryWins = yesWinProfit > 0 ? Math.ceil(yesLoss / yesWinProfit) : Infinity;
    yesLabel.textContent = `Don't buy YES above ${(settingsState.maxEntryPriceYes * 100).toFixed(0)}¢`;
    yesLabel.classList.remove("text-danger", "text-success", "text-warning");
    if (settingsState.maxEntryPriceYes > 0.90) yesLabel.classList.add("text-danger");
    else if (settingsState.maxEntryPriceYes < 0.80) yesLabel.classList.add("text-success");
    else yesLabel.classList.add("text-warning");
    yesCalc.textContent = `At ${(settingsState.maxEntryPriceYes * 100).toFixed(0)}¢ entry with $20: ${yesContracts.toFixed(2)} contracts, win = +$${yesWinProfit.toFixed(2)}, need ${yesRecoveryWins} wins to recover one loss`;

    const noContracts = amount / Math.max(settingsState.maxEntryPriceNo, 0.0001);
    const noWinProfit = noContracts * (1.0 - settingsState.maxEntryPriceNo);
    const noRecoveryWins = noWinProfit > 0 ? Math.ceil(amount / noWinProfit) : Infinity;
    const yesFloor = Math.max(0, 1.0 - settingsState.maxEntryPriceNo);
    noLabel.textContent = `Don't buy NO when NO price above ${(settingsState.maxEntryPriceNo * 100).toFixed(0)}¢`;
    noLabel.classList.remove("text-danger", "text-success", "text-warning");
    if (settingsState.maxEntryPriceNo > 0.90) noLabel.classList.add("text-danger");
    else if (settingsState.maxEntryPriceNo < 0.80) noLabel.classList.add("text-success");
    else noLabel.classList.add("text-warning");
    noFloor.textContent = `Don't buy NO when market YES < ${(yesFloor * 100).toFixed(1)}%`;
    noCalc.textContent = `At ${(settingsState.maxEntryPriceNo * 100).toFixed(0)}¢ entry with $20: ${noContracts.toFixed(2)} contracts, win = +$${noWinProfit.toFixed(2)}, need ${noRecoveryWins} wins to recover one loss`;
}

function updateDynamicSizingPreview() {
    const modeText = document.getElementById("dynamic-sizing-mode-text");
    const wrap = document.getElementById("dynamic-sizing-table-wrap");
    const tableBody = document.getElementById("dynamic-sizing-table-body");
    const baseSizeEl = document.getElementById("dynamic-sizing-base-size");
    if (!modeText || !wrap || !tableBody || !baseSizeEl) return;

    const baseSize = Number(settingsState.settings.paper_trade_size || 20);
    const safeBase = Number.isFinite(baseSize) && baseSize > 0 ? baseSize : 20;
    baseSizeEl.textContent = `$${safeBase.toFixed(2)}`;
    modeText.textContent = settingsState.dynamicSizingEnabled
        ? "ON: Scales bet size based on upside remaining"
        : "OFF: Fixed dollar amount for every trade";
    wrap.classList.toggle("hidden", !settingsState.dynamicSizingEnabled);

    const rows = [
        { entry: "60-65¢", edge: "35-40¢", mult: 1.5 },
        { entry: "65-80¢", edge: "20-35¢", mult: 1.0 },
        { entry: "80-90¢", edge: "10-20¢", mult: 0.6 },
        { entry: "90-100¢", edge: "0-10¢", mult: 0.3 },
    ];
    tableBody.innerHTML = rows.map((row) => {
        const sized = safeBase * row.mult;
        return `<tr><td>${row.entry}</td><td>${row.edge}</td><td>${row.mult.toFixed(1)}x ($${sized.toFixed(2)})</td></tr>`;
    }).join("");
}

function updateMaxReversalRiskPreview() {
    const slider = document.getElementById("max-reversal-risk-slider");
    const label = document.getElementById("max-reversal-risk-label");
    if (!slider || !label) return;
    const value = Number(slider.value);
    settingsState.maxReversalRisk = Number.isFinite(value) ? value : 0.65;
    label.textContent = `Block auto-trades when reversal risk exceeds ${(settingsState.maxReversalRisk * 100).toFixed(1)}%`;
}

function scheduleEntryFilterSave() {
    if (settingsState.entryFilterSaveTimerId) window.clearTimeout(settingsState.entryFilterSaveTimerId);
    settingsState.entryFilterSaveTimerId = window.setTimeout(async () => {
        try {
            await savePartialSettings({
                max_entry_price_yes: settingsState.maxEntryPriceYes,
                max_entry_price_no: settingsState.maxEntryPriceNo,
            });
            showSettingsToast("Entry filter updated", "success");
        } catch (error) {
            showSettingsToast(`Failed to update entry filter: ${error.message}`, "error");
        }
    }, 500);
}

function scheduleReversalRiskSave() {
    if (settingsState.reversalRiskSaveTimerId) window.clearTimeout(settingsState.reversalRiskSaveTimerId);
    settingsState.reversalRiskSaveTimerId = window.setTimeout(async () => {
        try {
            await savePartialSettings({
                max_reversal_risk: settingsState.maxReversalRisk,
            });
            showSettingsToast("Volatility guard updated", "success");
        } catch (error) {
            showSettingsToast(`Failed to update volatility guard: ${error.message}`, "error");
        }
    }, 500);
}

function updateLastSavedLabel() {
    const el = document.getElementById("settings-last-saved");
    if (!el || !settingsState.lastSavedAt) return;
    const secs = Math.max(0, Math.floor((Date.now() - settingsState.lastSavedAt) / 1000));
    el.textContent = secs === 0 ? "Last saved: just now" : `Last saved: ${secs}s ago`;
}

function setAdvancedOverrideVisible() {
    const toggle = document.getElementById("threshold-override-toggle");
    const section = document.getElementById("advanced-override");
    if (!toggle || !section) return;
    section.classList.toggle("hidden", !toggle.checked);
    section.classList.toggle("override-enabled", toggle.checked);
}

function applyNoSideState(enabled) {
    settingsState.enableNoSignals = Boolean(enabled);
    const toggleBtn = document.getElementById("no-side-toggle-btn");
    const label = document.getElementById("no-side-toggle-label");
    const banner = document.getElementById("no-side-banner");
    const bannerProfile = document.getElementById("no-side-banner-profile");
    if (toggleBtn) {
        toggleBtn.dataset.enabled = settingsState.enableNoSignals ? "true" : "false";
        toggleBtn.classList.toggle("enabled", settingsState.enableNoSignals);
    }
    if (label) {
        label.textContent = settingsState.enableNoSignals ? "NO Trading: ENABLED" : "NO Trading: DISABLED";
        label.style.color = settingsState.enableNoSignals ? "#f59e0b" : "";
    }
    if (banner) {
        banner.classList.toggle("hidden", !settingsState.enableNoSignals);
    }
    if (bannerProfile) {
        const profile = settingsState.activeRiskProfile || "moderate";
        bannerProfile.textContent = profile
            .split("_")
            .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
            .join(" ");
    }
}

function renderSignalMode() {
    const mode = ["agreement", "mispricing", "ensemble"].includes(settingsState.signalMode)
        ? settingsState.signalMode
        : "agreement";
    const agreementCard = document.getElementById("signal-mode-agreement");
    const mispricingCard = document.getElementById("signal-mode-mispricing");
    const ensembleCard = document.getElementById("signal-mode-ensemble");
    const thresholdWrap = document.getElementById("mispricing-threshold-wrap");
    if (agreementCard) agreementCard.classList.toggle("active", mode === "agreement");
    if (mispricingCard) mispricingCard.classList.toggle("active", mode === "mispricing");
    if (ensembleCard) ensembleCard.classList.toggle("active", mode === "ensemble");
    if (thresholdWrap) thresholdWrap.classList.toggle("hidden", mode === "agreement");
}

function renderRiskProfiles() {
    const grid = document.getElementById("risk-profiles-grid");
    if (!grid) return;
    const profiles = settingsState.riskProfiles || {};
    const active = settingsState.activeRiskProfile;
    const cardOrder = ["conservative", "moderate", "aggressive", "high_conviction"];
    const cards = cardOrder
        .filter((name) => profiles[name])
        .map((name) => {
            const profile = profiles[name];
            const title = name
                .split("_")
                .map((part) => part[0].toUpperCase() + part.slice(1))
                .join(" ");
            const thresholdPct = (Number(profile.yes_cutoff) * 100).toFixed(0);
            const hasEarly = Boolean(profile.early_entry_enabled && profile.early_entry_min_seconds && profile.early_entry_max_seconds && profile.early_entry_cutoff);
            const earlyLine = hasEarly
                ? `${Math.round(Number(profile.early_entry_min_seconds) / 60)}–${Math.round(Number(profile.early_entry_max_seconds) / 60)} min @ ${(Number(profile.early_entry_cutoff) * 100).toFixed(0)}%`
                : "None";
            return `
                <button type="button" class="profile-card ${active === name ? "active" : ""}" data-profile="${name}">
                    <div class="profile-head">
                        <h4>${title}</h4>
                    </div>
                    <div class="profile-stat-list">
                        <p>Threshold: <span class="mono">${thresholdPct}%</span></p>
                        <p>Window: <span class="mono">${profile.min_seconds}s - ${profile.max_seconds}s</span></p>
                        <p class="${hasEarly ? "profile-early-entry-enabled" : "text-muted"}">
                            Early Entry: <span class="mono">${earlyLine}</span>
                        </p>
                    </div>
                    <p class="risk-desc">${profile.description || ""}</p>
                </button>
            `;
        });
    grid.innerHTML = cards.join("");
    Array.from(grid.querySelectorAll(".profile-card")).forEach((card) => {
        card.addEventListener("click", async () => {
            const profileValue = String(card.getAttribute("data-profile") || "moderate").toLowerCase();
            await savePartialSettings({ risk_profile: profileValue });
            settingsState.activeRiskProfile = profileValue || "moderate";
            renderRiskProfiles();
            updateOverrideProfileHints();
            applyNoSideState(settingsState.enableNoSignals);
            showSettingsToast(`Risk profile set to ${profileValue}`, "success");
        });
    });
}

async function savePartialSettings(payload) {
    const response = await fetch("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok || (data.errors && data.errors.length)) {
        throw new Error(data.errors?.join("; ") || "Failed to save settings.");
    }
}

async function saveSettings() {
    const thresholdOverride = document.getElementById("threshold-override-toggle")?.checked ?? false;
    const payload = {
        risk_profile: settingsState.activeRiskProfile,
        threshold_override: thresholdOverride,
        yes_cutoff: Number(document.getElementById("yes-cutoff-slider")?.value || 0.65),
        no_cutoff: Number(document.getElementById("no-cutoff-slider")?.value || 0.35),
        min_seconds_to_close: Number(document.getElementById("min-seconds-input")?.value || 30),
        max_seconds_to_close: Number(document.getElementById("max-seconds-input")?.value || 180),
        poll_interval_seconds: Number(document.getElementById("poll-interval-input")?.value || 15),
        enable_no_signals: settingsState.enableNoSignals,
        paper_trade_size: Number(settingsState.settings.paper_trade_size || 5.0),
        signal_mode: settingsState.signalMode,
        mispricing_threshold: Number(settingsState.mispricingThreshold || 0.10),
        max_entry_price_yes: Number(settingsState.maxEntryPriceYes || 0.85),
        max_entry_price_no: Number(settingsState.maxEntryPriceNo || 0.85),
        min_expected_profit: Number(settingsState.settings.min_expected_profit || 0.10),
        max_reversal_risk: Number(settingsState.maxReversalRisk || 0.65),
        dynamic_sizing_enabled: settingsState.dynamicSizingEnabled,
    };
    try {
        await savePartialSettings(payload);
        settingsState.lastSavedAt = Date.now();
        const labelEl = document.getElementById("settings-last-saved");
        if (labelEl) {
            labelEl.textContent = "Last saved: just now";
        }
        showSettingsToast("Settings saved", "success");
    } catch (error) {
        showSettingsToast(`Save failed: ${error.message}`, "error");
    }
}

async function updateSchedulerStatus() {
    try {
        const response = await fetch("/api/scheduler/status", { headers: { Accept: "application/json" } });
        const data = await response.json();
        const pill = document.getElementById("scheduler-status-settings");
        if (!pill) return;
        if (data.running) {
            pill.textContent = "RUNNING";
            pill.className = "badge badge-success";
        } else {
            pill.textContent = "STOPPED";
            pill.className = "badge badge-danger";
        }
    } catch (_) {
        // no-op
    }
}

async function schedulerAction(action) {
    try {
        const response = await fetch(`/api/scheduler/${action}`, { method: "POST", headers: { Accept: "application/json" } });
        if (!response.ok) throw new Error("Scheduler request failed");
        showSettingsToast(`Scheduler ${action === "start" ? "started" : "stopped"}`, "success");
        await updateSchedulerStatus();
    } catch (error) {
        showSettingsToast(error.message, "error");
    }
}

async function loadModelInfo() {
    try {
        const res = await fetch("/api/model-info", { headers: { Accept: "application/json" } });
        const data = await res.json();
        if (!data.loaded) return;
        const typeBadge = document.getElementById("model-type-badge");
        if (typeBadge) {
            const raw = data.model_type || "Unknown";
            let label = raw;
            if (raw === "RandomForest") label = "Random Forest";
            else if (raw === "XGBoost" || raw === "XGBClassifier") label = "XGBoost";
            typeBadge.textContent = label;
        }
        document.getElementById("model-trained-at").textContent = data.trained_at || "--";
        document.getElementById("model-accuracy").textContent = data.test_metrics?.accuracy != null ? `${(data.test_metrics.accuracy * 100).toFixed(1)}%` : "--";
        document.getElementById("model-roc-auc").textContent = data.test_metrics?.roc_auc != null ? data.test_metrics.roc_auc.toFixed(3) : "--";
        document.getElementById("model-brier").textContent = data.test_metrics?.brier != null ? data.test_metrics.brier.toFixed(3) : "--";
        document.getElementById("model-train-size").textContent = data.n_train ?? "--";
        document.getElementById("model-test-size").textContent = data.n_test ?? "--";
        const savedEl = document.getElementById("model-sklearn-saved");
        const currentEl = document.getElementById("model-sklearn-current");
        const mismatchBanner = document.getElementById("model-version-warning");
        if (savedEl) savedEl.textContent = data.sklearn_version_saved || "--";
        if (currentEl) currentEl.textContent = data.sklearn_version_current || "--";
        if (mismatchBanner) mismatchBanner.classList.toggle("hidden", !data.version_mismatch);
    } catch (_) {
        // no-op
    }
}

async function initRiskProfileSettings() {
    try {
        const [profilesRes, settingsRes] = await Promise.all([
            fetch("/api/risk-profiles", { headers: { Accept: "application/json" } }),
            fetch("/api/settings", { headers: { Accept: "application/json" } }),
        ]);
        const profilesData = await profilesRes.json();
        const settingsData = await settingsRes.json();
        settingsState.riskProfiles = profilesData?.profiles || {};
        settingsState.activeRiskProfile = settingsData?.risk_profile || profilesData?.active || "moderate";
        settingsState.settings = settingsData || {};
        settingsState.enableNoSignals = (settingsData?.enable_no_signals || "false") === "true";
        const mode = String(settingsData?.signal_mode || "agreement").toLowerCase();
        const normalizedMode = mode === "ensemble_vote" ? "ensemble" : mode;
        settingsState.signalMode = ["agreement", "mispricing", "ensemble"].includes(normalizedMode) ? normalizedMode : "agreement";
        settingsState.mispricingThreshold = Number(settingsData?.mispricing_threshold || 0.10);
        settingsState.maxEntryPriceYes = Number(settingsData?.max_entry_price_yes || 0.85);
        settingsState.maxEntryPriceNo = Number(settingsData?.max_entry_price_no || 0.85);
        settingsState.maxReversalRisk = Number(settingsData?.max_reversal_risk || 0.65);
        settingsState.dynamicSizingEnabled = (settingsData?.dynamic_sizing_enabled || "false") === "true";

        const override = (settingsData?.threshold_override || "false") === "true";
        const overrideToggle = document.getElementById("threshold-override-toggle");
        if (overrideToggle) {
            overrideToggle.checked = override;
        }
        setAdvancedOverrideVisible();
        renderRiskProfiles();
        const yesSlider = document.getElementById("yes-cutoff-slider");
        const noSlider = document.getElementById("no-cutoff-slider");
        const minSecEl = document.getElementById("min-seconds-input");
        const maxSecEl = document.getElementById("max-seconds-input");
        if (yesSlider && settingsData.yes_cutoff != null && settingsData.yes_cutoff !== "") {
            yesSlider.value = String(Number(settingsData.yes_cutoff));
        }
        if (noSlider && settingsData.no_cutoff != null && settingsData.no_cutoff !== "") {
            noSlider.value = String(Number(settingsData.no_cutoff));
        }
        if (minSecEl && settingsData.min_seconds_to_close != null && settingsData.min_seconds_to_close !== "") {
            minSecEl.value = String(Number(settingsData.min_seconds_to_close));
        }
        if (maxSecEl && settingsData.max_seconds_to_close != null && settingsData.max_seconds_to_close !== "") {
            maxSecEl.value = String(Number(settingsData.max_seconds_to_close));
        }
        setCutoffLabel("yes-cutoff-slider", "yes-cutoff-value");
        setCutoffLabel("no-cutoff-slider", "no-cutoff-value");
        updateSecondsOverrideDisplays();
        updateOverrideProfileHints();
        applyNoSideState(settingsState.enableNoSignals);
        const mispricingSlider = document.getElementById("mispricing-threshold-slider");
        if (mispricingSlider && Number.isFinite(settingsState.mispricingThreshold)) {
            mispricingSlider.value = String(settingsState.mispricingThreshold);
        }
        const maxYesSlider = document.getElementById("max-entry-yes-slider");
        const maxNoSlider = document.getElementById("max-entry-no-slider");
        const maxReversalSlider = document.getElementById("max-reversal-risk-slider");
        if (maxYesSlider && Number.isFinite(settingsState.maxEntryPriceYes)) maxYesSlider.value = String(settingsState.maxEntryPriceYes);
        if (maxNoSlider && Number.isFinite(settingsState.maxEntryPriceNo)) maxNoSlider.value = String(settingsState.maxEntryPriceNo);
        if (maxReversalSlider && Number.isFinite(settingsState.maxReversalRisk)) maxReversalSlider.value = String(settingsState.maxReversalRisk);
        updateMispricingThresholdLabel();
        updateEntryFilterPreview();
        updateDynamicSizingPreview();
        updateMaxReversalRiskPreview();
        renderSignalMode();
    } catch (error) {
        showSettingsToast(`Failed to load risk profiles: ${error.message}`, "error");
    }
}

function wireInputs() {
    ["yes-cutoff-slider", "no-cutoff-slider"].forEach((id) => {
        const slider = document.getElementById(id);
        if (slider) slider.addEventListener("input", () => {
            setCutoffLabel("yes-cutoff-slider", "yes-cutoff-value");
            setCutoffLabel("no-cutoff-slider", "no-cutoff-value");
        });
    });
    const advancedBtn = document.getElementById("advanced-toggle-btn");
    const advancedPanel = document.getElementById("advanced-override");
    const overrideToggle = document.getElementById("threshold-override-toggle");
    if (advancedBtn && advancedPanel) {
        advancedBtn.addEventListener("click", () => {
            advancedPanel.classList.toggle("hidden");
            if (!advancedPanel.classList.contains("hidden")) {
                setCutoffLabel("yes-cutoff-slider", "yes-cutoff-value");
                setCutoffLabel("no-cutoff-slider", "no-cutoff-value");
                updateSecondsOverrideDisplays();
                updateOverrideProfileHints();
            }
        });
    }
    if (overrideToggle) {
        overrideToggle.addEventListener("change", async () => {
            setAdvancedOverrideVisible();
            try {
                await savePartialSettings({ threshold_override: overrideToggle.checked ? "true" : "false" });
                setAdvancedOverrideVisible();
                updateOverrideProfileHints();
                showSettingsToast("Override setting updated", "success");
            } catch (error) {
                showSettingsToast(`Failed to update override: ${error.message}`, "error");
            }
        });
    }
    const noSideToggleBtn = document.getElementById("no-side-toggle-btn");
    if (noSideToggleBtn) {
        noSideToggleBtn.addEventListener("click", async () => {
            const next = noSideToggleBtn.dataset.enabled !== "true";
            try {
                await savePartialSettings({ enable_no_signals: next ? "true" : "false" });
                applyNoSideState(next);
                showSettingsToast(
                    next ? "NO-side trading enabled — experimental" : "NO-side trading disabled",
                    "success",
                );
            } catch (error) {
                showSettingsToast(`Failed to update NO-side trading: ${error.message}`, "error");
            }
        });
    }
    const modeCards = Array.from(document.querySelectorAll(".signal-mode-card"));
    modeCards.forEach((card) => {
        card.addEventListener("click", async () => {
            const selected = String(card.getAttribute("data-mode") || "agreement").toLowerCase();
            const mode = ["agreement", "mispricing", "ensemble"].includes(selected) ? selected : "agreement";
            settingsState.signalMode = mode;
            renderSignalMode();
            try {
                await savePartialSettings({ signal_mode: mode });
                showSettingsToast(`Signal mode set to ${mode}`, "success");
            } catch (error) {
                showSettingsToast(`Failed to set signal mode: ${error.message}`, "error");
            }
        });
    });
    const mispricingSlider = document.getElementById("mispricing-threshold-slider");
    if (mispricingSlider) {
        mispricingSlider.addEventListener("input", () => {
            settingsState.mispricingThreshold = Number(mispricingSlider.value);
            updateMispricingThresholdLabel();
        });
        mispricingSlider.addEventListener("change", async () => {
            settingsState.mispricingThreshold = Number(mispricingSlider.value);
            updateMispricingThresholdLabel();
            try {
                await savePartialSettings({ mispricing_threshold: settingsState.mispricingThreshold });
                showSettingsToast("Mispricing threshold updated", "success");
            } catch (error) {
                showSettingsToast(`Failed to update threshold: ${error.message}`, "error");
            }
        });
    }
    const maxEntryYesSlider = document.getElementById("max-entry-yes-slider");
    const maxEntryNoSlider = document.getElementById("max-entry-no-slider");
    const maxReversalSlider = document.getElementById("max-reversal-risk-slider");
    const dynamicSizingToggle = document.getElementById("dynamic-sizing-toggle");
    if (maxEntryYesSlider) {
        maxEntryYesSlider.addEventListener("input", () => {
            updateEntryFilterPreview();
            scheduleEntryFilterSave();
            settingsState.settings.paper_trade_size = settingsState.settings.paper_trade_size || 20;
            updateDynamicSizingPreview();
        });
    }
    if (maxEntryNoSlider) {
        maxEntryNoSlider.addEventListener("input", () => {
            updateEntryFilterPreview();
            scheduleEntryFilterSave();
            updateDynamicSizingPreview();
        });
    }
    if (maxReversalSlider) {
        maxReversalSlider.addEventListener("input", () => {
            updateMaxReversalRiskPreview();
            scheduleReversalRiskSave();
        });
    }
    if (dynamicSizingToggle) {
        dynamicSizingToggle.checked = settingsState.dynamicSizingEnabled;
        dynamicSizingToggle.addEventListener("change", async () => {
            settingsState.dynamicSizingEnabled = dynamicSizingToggle.checked;
            updateDynamicSizingPreview();
            try {
                await savePartialSettings({ dynamic_sizing_enabled: settingsState.dynamicSizingEnabled ? "true" : "false" });
                showSettingsToast("Dynamic sizing updated", "success");
            } catch (error) {
                showSettingsToast(`Failed to update dynamic sizing: ${error.message}`, "error");
            }
        });
    }
    const minSecInput = document.getElementById("min-seconds-input");
    const maxSecInput = document.getElementById("max-seconds-input");
    if (minSecInput) {
        minSecInput.addEventListener("input", () => {
            updateSecondsOverrideDisplays();
        });
    }
    if (maxSecInput) {
        maxSecInput.addEventListener("input", () => {
            updateSecondsOverrideDisplays();
        });
    }
}

window.addEventListener("DOMContentLoaded", async () => {
    setCutoffLabel("yes-cutoff-slider", "yes-cutoff-value");
    setCutoffLabel("no-cutoff-slider", "no-cutoff-value");
    updateSecondsOverrideDisplays();
    updateMispricingThresholdLabel();
    updateEntryFilterPreview();
    updateDynamicSizingPreview();
    updateMaxReversalRiskPreview();
    await initRiskProfileSettings();
    wireInputs();
    loadModelInfo();
    updateSchedulerStatus();
    document.getElementById("save-settings-btn")?.addEventListener("click", saveSettings);
    document.getElementById("start-scheduler-btn")?.addEventListener("click", () => schedulerAction("start"));
    document.getElementById("stop-scheduler-btn")?.addEventListener("click", () => schedulerAction("stop"));
    window.setInterval(updateLastSavedLabel, 1000);
});
