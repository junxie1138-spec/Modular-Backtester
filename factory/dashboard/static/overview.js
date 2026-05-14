(function () {
  const REFRESH_SEC = parseInt(document.body.dataset.refreshSec || "10", 10);
  const THRESHOLD_METRIC = document.body.dataset.thresholdMetric || "wfo.oos_sharpe";
  const THRESHOLD = parseFloat(document.body.dataset.thresholdValue || "1.0");

  function extractMetric(rec, path) {
    let cur = rec;
    for (const part of path.split(".")) {
      if (cur == null || typeof cur !== "object") return null;
      cur = cur[part];
    }
    return typeof cur === "number" ? cur : null;
  }

  function fmt(n, digits = 3) {
    if (n == null) return "";
    return Number(n).toFixed(digits);
  }
  function fmtPct(n) {
    if (n == null) return "";
    return (n * 100).toFixed(2) + "%";
  }

  function rowFor(rec) {
    const tr = document.createElement("tr");
    const value = rec.status === "complete" ? extractMetric(rec, THRESHOLD_METRIC) : null;
    const isGood = value != null && value > THRESHOLD;
    if (rec.status === "failed") tr.classList.add("failed");
    else if (isGood) tr.classList.add("good");
    if (rec.strategy_id) tr.dataset.strategyId = rec.strategy_id;
    const cells = [
      rec.timestamp || "",
      rec.strategy_id || "(no id)",
      (rec.idea && rec.idea.one_line_summary) || "(no idea)",
      rec.status === "failed"
        ? `<span class="failed-stage">failed: ${rec.failed_stage}</span>`
        : rec.status,
      rec.backtest ? fmt(rec.backtest.sharpe) : "",
      rec.wfo ? fmt(rec.wfo.oos_sharpe) : "",
      rec.wfo ? fmtPct(rec.wfo.oos_total_return) : "",
      rec.wfo ? fmtPct(rec.wfo.oos_max_drawdown) : "",
      isGood ? "*" : "",
      rec.alerted ? "*" : "",
    ];
    for (const html of cells) {
      const td = document.createElement("td");
      td.innerHTML = html;
      tr.appendChild(td);
    }
    return tr;
  }

  async function refresh() {
    try {
      const [recsResp, sumResp] = await Promise.all([
        fetch("/api/records"),
        fetch("/api/summary"),
      ]);
      if (!recsResp.ok || !sumResp.ok) return;
      const records = await recsResp.json();
      const summary = await sumResp.json();

      const tbody = document.getElementById("records-body");
      if (tbody) {
        tbody.innerHTML = "";
        // Newest first.
        for (let i = records.length - 1; i >= 0; i--) {
          tbody.appendChild(rowFor(records[i]));
        }
      }
      const t = document.getElementById("c-total");       if (t) t.textContent = summary.total_cycles;
      const c = document.getElementById("c-complete");    if (c) c.textContent = summary.completes;
      const f = document.getElementById("c-failures");    if (f) f.textContent =
        (summary.total_cycles - summary.completes) + " (" +
        Object.entries(summary.failures_by_stage).map(([k, v]) => `${k}=${v}`).join(", ") + ")";
      const a = document.getElementById("c-above");       if (a) a.textContent = summary.above_threshold;
      const s = document.getElementById("c-spend");       if (s) s.textContent = "$" + Number(summary.cumulative_spend_usd).toFixed(2);
    } catch (err) {
      console.warn("refresh failed", err);
    }
  }

  document.addEventListener("click", function (ev) {
    let el = ev.target;
    while (el && el.tagName !== "TR") el = el.parentElement;
    if (el && el.dataset && el.dataset.strategyId) {
      window.location.href = "/strategy/" + el.dataset.strategyId;
    }
  });

  if (REFRESH_SEC > 0) {
    setInterval(refresh, REFRESH_SEC * 1000);
  }
})();
