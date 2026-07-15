// Strategy Control Center — focused page logic (no dependency on main.js).
async function loadState() {
  const res = await fetch("/api/control/state", { headers: { Accept: "application/json" } });
  if (!res.ok) return;
  const s = await res.json();
  const badge = document.getElementById("mode-badge");
  badge.textContent = s.mode === "live" ? "LIVE — REAL MONEY" : "PAPER MODE";
  badge.classList.toggle("live", s.mode === "live");
  document.getElementById("st-scheduler").textContent = s.scheduler_running ? "RUNNING" : "STOPPED";
  document.getElementById("st-trades").textContent = s.trades_today;
  document.getElementById("st-pnl").textContent = "$" + Number(s.paper_pnl_today).toFixed(2);
  const sel = document.getElementById("signal-mode-select");
  if (sel) sel.value = s.signal_mode;
  const thr = document.getElementById("threshold-input");
  if (thr) thr.value = s.mispricing_threshold;
}

async function applyDefaults() {
  await fetch("/api/control/apply-defaults", { method: "POST", headers: { Accept: "application/json" } });
  await loadState();
}

async function saveStrategy() {
  const payload = {
    signal_mode: document.getElementById("signal-mode-select").value,
    mispricing_threshold: Number(document.getElementById("threshold-input").value),
  };
  await fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(payload),
  });
  await loadState();
}

async function setLive(enabled) {
  if (enabled) {
    const typed = window.prompt('Type "LIVE" to enable REAL-MONEY trading:');
    if (typed !== "LIVE") return;
  }
  await fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({ live_trading_enabled: enabled }),
  });
  await loadState();
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("apply-defaults-btn")?.addEventListener("click", applyDefaults);
  document.getElementById("save-btn")?.addEventListener("click", saveStrategy);
  document.getElementById("mode-paper")?.addEventListener("click", () => setLive(false));
  document.getElementById("mode-live")?.addEventListener("click", () => setLive(true));
  loadState();
  setInterval(loadState, 5000);
});
