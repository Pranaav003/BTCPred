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
    maxDailyLoss: 200,
    dynamicSizingEnabled: false,
    entryFilterSaveTimerId: null,
    reversalRiskSaveTimerId: null,
    maxDailyLossSaveTimerId: null,
    editingProfile: null,
    draftProfile: null,
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

function customConfirm(message) {
    return new Promise((resolve) => {
        const modal = document.getElementById("custom-confirm-modal");
        const body = document.getElementById("custom-confirm-body");
        const confirmBtn = document.getElementById("custom-confirm-ok");
        const cancelBtn = document.getElementById("custom-confirm-cancel");
        if (!modal || !body || !confirmBtn || !cancelBtn) {
            resolve(window.confirm(message));
            return;
        }
        body.textContent = message;
        modal.style.display = "flex";

        function cleanup() {
            modal.style.display = "none";
            confirmBtn.removeEventListener("click", onConfirm);
            cancelBtn.removeEventListener("click", onCancel);
            modal.removeEventListener("click", onBackdrop);
        }

        function onConfirm() { cleanup(); resolve(true); }
        function onCancel() { cleanup(); resolve(false); }
        function onBackdrop(e) { if (e.target === modal) { cleanup(); resolve(false); } }

        confirmBtn.addEventListener("click", onConfirm);
        cancelBtn.addEventListener("click", onCancel);
        modal.addEventListener("click", onBackdrop);
    });
}

function toTitleCase(name) {
    return String(name || "")
        .split("_")
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(" ");
}

function normalizeProfileForEdit(profile) {
    return {
        yes_cutoff: Number(profile?.yes_cutoff ?? 0.65),
        no_cutoff: Number(profile?.no_cutoff ?? 0.35),
        min_seconds: Number(profile?.min_seconds ?? 60),
        max_seconds: Number(profile?.max_seconds ?? 300),
        early_entry_enabled: Boolean(profile?.early_entry_enabled),
        early_entry_min_seconds: profile?.early_entry_min_seconds == null ? "" : Number(profile.early_entry_min_seconds),
        early_entry_max_seconds: profile?.early_entry_max_seconds == null ? "" : Number(profile.early_entry_max_seconds),
        early_entry_cutoff: profile?.early_entry_cutoff == null ? "" : Number(profile.early_entry_cutoff),
    };
}

function profileDirty(profileName) {
    if (!settingsState.editingProfile || settingsState.editingProfile !== profileName || !settingsState.draftProfile) return false;
    const baseline = normalizeProfileForEdit(settingsState.riskProfiles?.[profileName] || {});
    const draft = settingsState.draftProfile;
    return JSON.stringify(baseline) !== JSON.stringify(draft);
}

function renderProfileTimeline(minSeconds, maxSeconds) {
    const min = Math.max(0, Math.min(900, Number(minSeconds || 0)));
    const max = Math.max(0, Math.min(900, Number(maxSeconds || 0)));
    const left = `${(min / 900) * 100}%`;
    const width = `${Math.max(0, ((max - min) / 900) * 100)}%`;
    return `
        <div class="timeline-wrap">
            <div class="timeline-base">
                <div class="timeline-active" style="left: ${left}; width: ${width};"></div>
            </div>
            <div class="timeline-labels"><span>0s</span><span>300s</span><span>600s</span><span>900s</span></div>
        </div>
    `;
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

    const baseSize = Number(settingsState.settings.paper_trade_size || 10);
    const safeBase = Number.isFinite(baseSize) && baseSize > 0 ? baseSize : 10;
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
    const agreementLine = document.getElementById("agreement-volatility-line");
    if (!slider || !label) return;
    const value = Number(slider.value);
    settingsState.maxReversalRisk = Number.isFinite(value) ? value : 0.65;
    label.textContent = `Agreement trades blocked above ${(settingsState.maxReversalRisk * 100).toFixed(1)}%`;
    if (agreementLine) {
        agreementLine.textContent = `Agreement trades: blocked above ${(settingsState.maxReversalRisk * 100).toFixed(1)}%`;
    }
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

function updateMaxDailyLossPreview() {
    const input = document.getElementById("max-daily-loss-input");
    const label = document.getElementById("max-daily-loss-label");
    if (!input) return;
    const value = Number(input.value);
    settingsState.maxDailyLoss = Number.isFinite(value) && value > 0 ? value : 200;
    if (label) label.textContent = String(Math.round(settingsState.maxDailyLoss));
}

function scheduleMaxDailyLossSave() {
    if (settingsState.maxDailyLossSaveTimerId) window.clearTimeout(settingsState.maxDailyLossSaveTimerId);
    settingsState.maxDailyLossSaveTimerId = window.setTimeout(async () => {
        try {
            await savePartialSettings({ max_daily_loss: settingsState.maxDailyLoss });
            showSettingsToast("Max daily loss updated", "success");
        } catch (error) {
            showSettingsToast(`Failed to update max daily loss: ${error.message}`, "error");
        }
    }, 500);
}

function updateLastSavedLabel() {
    const el = document.getElementById("settings-last-saved");
    if (!el || !settingsState.lastSavedAt) return;
    const secs = Math.max(0, Math.floor((Date.now() - settingsState.lastSavedAt) / 1000));
    el.textContent = secs === 0 ? "Last saved: just now" : `Last saved: ${secs}s ago`;
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
            const title = toTitleCase(name);
            const thresholdPct = (Number(profile.yes_cutoff) * 100).toFixed(0);
            const hasEarly = Boolean(profile.early_entry_enabled && profile.early_entry_min_seconds && profile.early_entry_max_seconds && profile.early_entry_cutoff);
            const earlyLine = hasEarly
                ? `${Math.round(Number(profile.early_entry_min_seconds))}s-${Math.round(Number(profile.early_entry_max_seconds))}s @ ${(Number(profile.early_entry_cutoff) * 100).toFixed(0)}%`
                : "None";
            const isEditing = settingsState.editingProfile === name;
            const customizedBadge = profile.customized ? `<span class="badge badge-warning">Customized</span>` : "";
            if (!isEditing) {
                return `
                    <div class="profile-card ${active === name ? "active" : ""}" data-profile="${name}">
                        <div class="profile-head">
                            <h4>${title} ${customizedBadge}</h4>
                            <button type="button" class="btn-ghost btn-sm profile-edit-btn" data-edit-profile="${name}" title="Edit profile">✎ Edit</button>
                        </div>
                        <div class="profile-stat-list">
                            <p>Threshold: <span class="mono">${thresholdPct}%</span></p>
                            <p>Window: <span class="mono">${profile.min_seconds}s - ${profile.max_seconds}s</span></p>
                            <p class="${hasEarly ? "profile-early-entry-enabled" : "text-muted"}">Early Entry: <span class="mono">${earlyLine}</span></p>
                        </div>
                        <p class="risk-desc">${profile.description || ""}</p>
                    </div>
                `;
            }
            const draft = settingsState.draftProfile || normalizeProfileForEdit(profile);
            const showEarly = Boolean(draft.early_entry_enabled);
            const hasEarlyFields = name === "aggressive" || name === "high_conviction";
            return `
                <div class="profile-card profile-card-editing ${active === name ? "active" : ""}" data-profile="${name}">
                    <div class="profile-head">
                        <h4>${title}</h4>
                    </div>
                    <div class="settings-field">
                        <label>YES Threshold: <span class="mono">${(Number(draft.yes_cutoff) * 100).toFixed(1)}%</span></label>
                        <input type="range" min="0.50" max="0.90" step="0.01" value="${Number(draft.yes_cutoff)}" data-field="yes_cutoff" data-profile-edit="${name}">
                    </div>
                    <div class="settings-two-col">
                        <label class="settings-field"><span>Min seconds</span><input type="number" min="0" max="300" value="${Number(draft.min_seconds)}" data-field="min_seconds" data-profile-edit="${name}"></label>
                        <label class="settings-field"><span>Max seconds</span><input type="number" min="60" max="900" value="${Number(draft.max_seconds)}" data-field="max_seconds" data-profile-edit="${name}"></label>
                    </div>
                    ${renderProfileTimeline(draft.min_seconds, draft.max_seconds)}
                    ${hasEarlyFields ? `
                        <div class="settings-field">
                            <label class="toggle-row">
                                <span>Enable early entry</span>
                                <span class="toggle-switch">
                                    <input type="checkbox" ${showEarly ? "checked" : ""} data-field="early_entry_enabled" data-profile-edit="${name}">
                                    <span class="toggle-slider"></span>
                                </span>
                            </label>
                            ${showEarly ? `
                                <div class="settings-two-col">
                                    <label class="settings-field"><span>Early min seconds</span><input type="number" min="0" max="900" value="${draft.early_entry_min_seconds}" data-field="early_entry_min_seconds" data-profile-edit="${name}"></label>
                                    <label class="settings-field"><span>Early max seconds</span><input type="number" min="0" max="900" value="${draft.early_entry_max_seconds}" data-field="early_entry_max_seconds" data-profile-edit="${name}"></label>
                                </div>
                                <label class="settings-field">
                                    <span>Early entry cutoff: <span class="mono">${draft.early_entry_cutoff === "" ? "--" : `${(Number(draft.early_entry_cutoff) * 100).toFixed(1)}%`}</span></span>
                                    <input type="range" min="0.50" max="0.90" step="0.01" value="${draft.early_entry_cutoff === "" ? 0.80 : Number(draft.early_entry_cutoff)}" data-field="early_entry_cutoff" data-profile-edit="${name}">
                                </label>
                            ` : `<p class="text-muted">None</p>`}
                        </div>
                    ` : ""}
                    <div class="settings-actions">
                        <button type="button" class="btn-primary" data-profile-save="${name}">Save</button>
                        <button type="button" class="btn-ghost" data-profile-cancel="${name}">Cancel</button>
                        <button type="button" class="btn-ghost profile-reset-btn" data-profile-reset="${name}">Reset to defaults</button>
                    </div>
                </div>
            `;
        });
    grid.innerHTML = cards.join("");
    Array.from(grid.querySelectorAll(".profile-card")).forEach((card) => {
        card.addEventListener("click", async (event) => {
            if (event.target.closest(".profile-edit-btn") || event.target.closest("[data-profile-edit]") || event.target.closest("[data-profile-save]") || event.target.closest("[data-profile-cancel]") || event.target.closest("[data-profile-reset]")) return;
            const profileValue = String(card.getAttribute("data-profile") || "moderate").toLowerCase();
            await savePartialSettings({ risk_profile: profileValue });
            settingsState.activeRiskProfile = profileValue || "moderate";
            renderRiskProfiles();
            applyNoSideState(settingsState.enableNoSignals);
            showSettingsToast(`Risk profile set to ${profileValue}`, "success");
        });
    });
    Array.from(grid.querySelectorAll(".profile-edit-btn")).forEach((btn) => {
        btn.addEventListener("click", async (event) => {
            event.preventDefault();
            event.stopPropagation();
            const profileName = String(btn.getAttribute("data-edit-profile") || "");
            if (!profileName) return;
            if (settingsState.editingProfile && settingsState.editingProfile !== profileName && profileDirty(settingsState.editingProfile)) {
                const okay = await customConfirm("Discard unsaved changes on current profile?");
                if (!okay) return;
            }
            settingsState.editingProfile = profileName;
            settingsState.draftProfile = normalizeProfileForEdit(settingsState.riskProfiles?.[profileName] || {});
            renderRiskProfiles();
        });
    });
    Array.from(grid.querySelectorAll("[data-profile-edit]")).forEach((input) => {
        input.addEventListener("input", () => {
            const field = String(input.getAttribute("data-field") || "");
            if (!field || !settingsState.draftProfile) return;
            if (input.type === "checkbox") settingsState.draftProfile[field] = input.checked;
            else settingsState.draftProfile[field] = input.value === "" ? "" : Number(input.value);
            renderRiskProfiles();
        });
    });
    Array.from(grid.querySelectorAll("[data-profile-cancel]")).forEach((btn) => {
        btn.addEventListener("click", async (event) => {
            event.preventDefault();
            event.stopPropagation();
            const profileName = String(btn.getAttribute("data-profile-cancel") || "");
            if (profileDirty(profileName)) {
                const okay = await customConfirm("Discard unsaved changes?");
                if (!okay) return;
            }
            settingsState.editingProfile = null;
            settingsState.draftProfile = null;
            renderRiskProfiles();
        });
    });
    Array.from(grid.querySelectorAll("[data-profile-save]")).forEach((btn) => {
        btn.addEventListener("click", async (event) => {
            event.preventDefault();
            event.stopPropagation();
            const profileName = String(btn.getAttribute("data-profile-save") || "");
            if (!profileName || !settingsState.draftProfile) return;
            const draft = settingsState.draftProfile;
            const payload = {
                yes_cutoff: Number(draft.yes_cutoff),
                no_cutoff: Number(1 - Number(draft.yes_cutoff)),
                min_seconds: Math.max(0, Math.min(300, Number(draft.min_seconds))),
                max_seconds: Math.max(60, Math.min(900, Number(draft.max_seconds))),
                early_entry_enabled: Boolean(draft.early_entry_enabled),
                early_entry_min_seconds: draft.early_entry_enabled ? Number(draft.early_entry_min_seconds || 0) : null,
                early_entry_max_seconds: draft.early_entry_enabled ? Number(draft.early_entry_max_seconds || 0) : null,
                early_entry_cutoff: draft.early_entry_enabled ? Number(draft.early_entry_cutoff || draft.yes_cutoff) : null,
            };
            try {
                const response = await fetch(`/api/risk-profiles/${profileName}`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json", Accept: "application/json" },
                    body: JSON.stringify(payload),
                });
                if (!response.ok) throw new Error("Failed to save profile");
                const refreshed = await fetch("/api/risk-profiles", { headers: { Accept: "application/json" } });
                const profileData = await refreshed.json();
                settingsState.riskProfiles = profileData?.profiles || settingsState.riskProfiles;
                settingsState.editingProfile = null;
                settingsState.draftProfile = null;
                renderRiskProfiles();
                showSettingsToast("Profile updated", "success");
            } catch (error) {
                showSettingsToast(`Failed to update profile: ${error.message}`, "error");
            }
        });
    });
    Array.from(grid.querySelectorAll("[data-profile-reset]")).forEach((btn) => {
        btn.addEventListener("click", async (event) => {
            event.preventDefault();
            event.stopPropagation();
            const profileName = String(btn.getAttribute("data-profile-reset") || "");
            if (!profileName) return;
            const ok = await customConfirm(`Reset ${toTitleCase(profileName)} to default values? This cannot be undone.`);
            if (!ok) return;
            try {
                const response = await fetch(`/api/risk-profiles/${profileName}/reset`, {
                    method: "DELETE",
                    headers: { Accept: "application/json" },
                });
                if (!response.ok) throw new Error("Failed to reset profile");
                const refreshed = await fetch("/api/risk-profiles", { headers: { Accept: "application/json" } });
                const profileData = await refreshed.json();
                settingsState.riskProfiles = profileData?.profiles || settingsState.riskProfiles;
                settingsState.editingProfile = null;
                settingsState.draftProfile = null;
                renderRiskProfiles();
                showSettingsToast("Profile reset to defaults", "success");
            } catch (error) {
                showSettingsToast(`Reset failed: ${error.message}`, "error");
            }
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
    const payload = {
        risk_profile: settingsState.activeRiskProfile,
        poll_interval_seconds: Number(document.getElementById("poll-interval-input")?.value || 30),
        enable_no_signals: settingsState.enableNoSignals,
        paper_trade_size: Number(settingsState.settings.paper_trade_size || 10),
        signal_mode: settingsState.signalMode,
        mispricing_threshold: Number(settingsState.mispricingThreshold || 0.10),
        max_entry_price_yes: Number(settingsState.maxEntryPriceYes || 0.85),
        max_entry_price_no: Number(settingsState.maxEntryPriceNo || 0.85),
        min_expected_profit: Number(settingsState.settings.min_expected_profit || 0.10),
        max_reversal_risk: Number(settingsState.maxReversalRisk || 0.65),
        max_daily_loss: Number(settingsState.maxDailyLoss || 50),
        dynamic_sizing_enabled: settingsState.dynamicSizingEnabled,
        live_trade_size: Number(document.getElementById("live-trade-size")?.value || 5),
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
        if (!data.loaded) {
            // Show why the model isn't loading
            const statusEl = document.getElementById("model-upload-status");
            if (statusEl && data.error) {
                statusEl.textContent = `⚠ Model not loaded: ${data.error}`;
                statusEl.className = "warning-banner warning-banner-danger";
                statusEl.classList.remove("hidden");
            }
            return;
        }
        // Clear any previous error
        const statusEl = document.getElementById("model-upload-status");
        if (statusEl && statusEl.textContent.startsWith("⚠")) {
            statusEl.classList.add("hidden");
        }
        const typeBadge = document.getElementById("model-type-badge");
        if (typeBadge) {
            const raw = data.model_type || "Unknown";
            let label = raw;
            if (raw === "RandomForest") label = "Random Forest";
            else if (raw === "CalibratedRandomForest") label = "Calibrated RF";
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
        settingsState.maxDailyLoss = Number(settingsData?.max_daily_loss || 50);
        settingsState.dynamicSizingEnabled = (settingsData?.dynamic_sizing_enabled || "false") === "true";

        renderRiskProfiles();
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
        const maxDailyLossInput = document.getElementById("max-daily-loss-input");
        if (maxDailyLossInput && Number.isFinite(settingsState.maxDailyLoss)) {
            maxDailyLossInput.value = String(Math.round(settingsState.maxDailyLoss));
        }
        updateMispricingThresholdLabel();
        updateEntryFilterPreview();
        updateDynamicSizingPreview();
        updateMaxReversalRiskPreview();
        updateMaxDailyLossPreview();
        renderSignalMode();
    } catch (error) {
        showSettingsToast(`Failed to load risk profiles: ${error.message}`, "error");
    }
}

function wireInputs() {
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
            settingsState.settings.paper_trade_size = settingsState.settings.paper_trade_size || 10;
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
    const maxDailyLossInput = document.getElementById("max-daily-loss-input");
    if (maxDailyLossInput) {
        maxDailyLossInput.addEventListener("input", () => {
            updateMaxDailyLossPreview();
            scheduleMaxDailyLossSave();
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
}

window.addEventListener("DOMContentLoaded", async () => {
    updateMispricingThresholdLabel();
    updateEntryFilterPreview();
    updateDynamicSizingPreview();
    updateMaxReversalRiskPreview();
    await initRiskProfileSettings();
    wireInputs();
    wireLiveTradingSection();
    await initLiveTradingSection();
    loadModelInfo();
    updateSchedulerStatus();
    document.getElementById("save-settings-btn")?.addEventListener("click", saveSettings);
    document.getElementById("start-scheduler-btn")?.addEventListener("click", () => schedulerAction("start"));
    document.getElementById("stop-scheduler-btn")?.addEventListener("click", () => schedulerAction("stop"));

    // Model upload
    document.getElementById("model-upload-input")?.addEventListener("change", async (event) => {
        const file = event.target.files?.[0];
        if (!file) return;
        const statusEl = document.getElementById("model-upload-status");
        if (statusEl) {
            statusEl.textContent = `Uploading ${file.name}...`;
            statusEl.className = "warning-banner";
            statusEl.classList.remove("hidden");
        }
        try {
            const formData = new FormData();
            formData.append("file", file);
            const res = await fetch("/api/model/upload", { method: "POST", body: formData });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || "Upload failed");
            if (statusEl) {
                statusEl.textContent = `✓ Model uploaded — ${(data.size_bytes / 1024).toFixed(0)} KB, accuracy ${(data.accuracy * 100).toFixed(1)}%`;
                statusEl.className = "warning-banner warning-banner-success";
            }
            await loadModelInfo();
        } catch (err) {
            if (statusEl) {
                statusEl.textContent = `✗ ${err.message}`;
                statusEl.className = "warning-banner warning-banner-danger";
            }
        }
        event.target.value = "";
    });

    // Model reload
    document.getElementById("reload-model-btn")?.addEventListener("click", async () => {
        try {
            await fetch("/api/model/reload", { method: "POST" });
            showSettingsToast("Model cache cleared — reloading", "success");
            await loadModelInfo();
        } catch (err) {
            showSettingsToast(`Reload failed: ${err.message}`, "error");
        }
    });

    window.setInterval(updateLastSavedLabel, 1000);
});

async function settingsApiFetch(url, options = {}) {
    try {
        const response = await fetch(url, { headers: { Accept: "application/json" }, ...options });
        if (!response.ok) return null;
        return await response.json();
    } catch (_) {
        return null;
    }
}

function updateLiveBadge(enabled) {
    const badge = document.getElementById("live-trading-badge");
    const label = document.getElementById("live-trading-status-label");
    if (enabled) {
        if (badge) {
            badge.textContent = "LIVE ACTIVE";
            badge.className = "badge badge-danger";
        }
        if (label) {
            label.textContent = "⚠ Live orders are being placed with real money";
            label.className = "text-danger";
        }
    } else {
        if (badge) {
            badge.textContent = "DISABLED";
            badge.className = "badge badge-neutral";
        }
        if (label) {
            label.textContent = "Live orders disabled — safe mode";
            label.className = "text-muted";
        }
    }
}

async function initLiveTradingSection() {
    const balanceRes = await settingsApiFetch("/api/live/balance");
    const apiStatus = document.getElementById("api-key-status");
    const liveControls = document.getElementById("live-controls");
    const notConfigured = document.getElementById("live-not-configured");
    const liveBalance = document.getElementById("live-balance");

    if (!balanceRes) {
        if (apiStatus) {
            apiStatus.textContent = "✗ Error";
            apiStatus.className = "value text-danger";
        }
        return;
    }

    if (!balanceRes.configured) {
        if (apiStatus) {
            apiStatus.textContent = "✗ Not configured";
            apiStatus.className = "value text-danger";
        }
        liveControls?.classList.add("hidden");
        notConfigured?.classList.remove("hidden");
    } else {
        if (apiStatus) {
            apiStatus.textContent = "✓ Configured";
            apiStatus.className = "value text-success";
        }
        liveControls?.classList.remove("hidden");
        notConfigured?.classList.add("hidden");
        if (liveBalance) {
            liveBalance.textContent = balanceRes.balance_dollars != null
                ? `$${Number(balanceRes.balance_dollars).toFixed(2)}`
                : "Error fetching";
        }
    }

    const settings = await settingsApiFetch("/api/settings");
    if (settings) {
        const sizeEl = document.getElementById("live-trade-size");
        const lossEl = document.getElementById("live-max-daily-loss");
        if (sizeEl) sizeEl.value = settings.live_trade_size || "5";
        if (lossEl) lossEl.value = settings.max_daily_loss || "50";
        const toggle = document.getElementById("live-trading-toggle");
        if (toggle) toggle.checked = settings.live_trading_enabled === "true";
        updateLiveBadge(settings.live_trading_enabled === "true");
    }
}

function wireLiveTradingSection() {
    document.getElementById("test-api-btn")?.addEventListener("click", async () => {
        const result = document.getElementById("test-api-result");
        if (result) {
            result.textContent = "Testing...";
            result.className = "text-muted";
        }
        const res = await settingsApiFetch("/api/live/test-order", { method: "POST" });
        if (res?.success) {
            if (result) {
                result.textContent = `✓ Keys valid — Balance: $${Number(res.balance_dollars).toFixed(2)}`;
                result.className = "text-success";
            }
        } else if (result) {
            result.textContent = `✗ ${res?.error || "Unknown error"}`;
            result.className = "text-danger";
        }
    });

    document.getElementById("refresh-balance-btn")?.addEventListener("click", async () => {
        const res = await settingsApiFetch("/api/live/balance");
        const el = document.getElementById("live-balance");
        if (el && res?.balance_dollars != null) {
            el.textContent = `$${Number(res.balance_dollars).toFixed(2)}`;
        }
    });

    document.getElementById("live-trading-toggle")?.addEventListener("change", async (event) => {
        const toggle = event.target;
        if (toggle.checked) {
            toggle.checked = false;
            const tradeSize = document.getElementById("live-trade-size")?.value || "5";
            const dailyLoss = document.getElementById("live-max-daily-loss")?.value || "50";
            const balance = document.getElementById("live-balance")?.textContent || "—";
            const modalSize = document.getElementById("modal-trade-size");
            const modalLoss = document.getElementById("modal-daily-loss");
            const modalBal = document.getElementById("modal-balance");
            if (modalSize) modalSize.textContent = `$${tradeSize}`;
            if (modalLoss) modalLoss.textContent = `$${dailyLoss}`;
            if (modalBal) modalBal.textContent = balance;
            const modal = document.getElementById("live-confirm-modal");
            if (modal) modal.style.display = "flex";
        } else {
            try {
                await savePartialSettings({ live_trading_enabled: "false" });
                updateLiveBadge(false);
                showSettingsToast("Live trading disabled", "success");
            } catch (error) {
                showSettingsToast(`Failed to disable live trading: ${error.message}`, "error");
            }
        }
    });

    document.getElementById("modal-cancel")?.addEventListener("click", () => {
        const modal = document.getElementById("live-confirm-modal");
        if (modal) modal.style.display = "none";
    });

    document.getElementById("modal-confirm")?.addEventListener("click", async () => {
        const tradeSize = document.getElementById("live-trade-size")?.value || "5";
        const dailyLoss = document.getElementById("live-max-daily-loss")?.value || "50";
        try {
            await savePartialSettings({
                live_trade_size: tradeSize,
                max_daily_loss: dailyLoss,
                live_trading_enabled: "true",
            });
            const toggle = document.getElementById("live-trading-toggle");
            if (toggle) toggle.checked = true;
            const modal = document.getElementById("live-confirm-modal");
            if (modal) modal.style.display = "none";
            updateLiveBadge(true);
            showSettingsToast("Live trading enabled", "success");
        } catch (error) {
            showSettingsToast(`Failed to enable live trading: ${error.message}`, "error");
        }
    });

    document.getElementById("live-confirm-modal")?.addEventListener("click", (event) => {
        if (event.target?.id === "live-confirm-modal") {
            event.target.style.display = "none";
        }
    });
}
