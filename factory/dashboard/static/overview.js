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

      lastRecords = records;
      renderRows(lastRecords);

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

  let currentSort = { key: null, asc: false };
  let lastRecords = [];

  function sortValueFor(rec, key) {
    if (key === "timestamp") return rec.timestamp || "";
    if (key === "strategy_id") return rec.strategy_id || "";
    if (key === "status") return rec.status || "";
    if (key === "backtest_sharpe") return rec.backtest ? rec.backtest.sharpe : null;
    if (key === "oos_sharpe") return rec.wfo ? rec.wfo.oos_sharpe : null;
    if (key === "oos_total_return") return rec.wfo ? rec.wfo.oos_total_return : null;
    if (key === "oos_max_drawdown") return rec.wfo ? rec.wfo.oos_max_drawdown : null;
    return null;
  }

  function renderRows(records) {
    const tbody = document.getElementById("records-body");
    if (!tbody) return;
    let ordered = records.slice();
    if (currentSort.key) {
      ordered.sort((a, b) => {
        const av = sortValueFor(a, currentSort.key);
        const bv = sortValueFor(b, currentSort.key);
        // Nulls sort to the end regardless of direction.
        if (av == null && bv == null) return 0;
        if (av == null) return 1;
        if (bv == null) return -1;
        if (av < bv) return currentSort.asc ? -1 : 1;
        if (av > bv) return currentSort.asc ? 1 : -1;
        return 0;
      });
    } else {
      // Default: newest first (reverse of insertion order).
      ordered.reverse();
    }
    tbody.innerHTML = "";
    for (const rec of ordered) tbody.appendChild(rowFor(rec));
  }

  document.addEventListener("click", function (ev) {
    // Handle column sort
    const th = ev.target.closest("th[data-sort]");
    if (th) {
      const key = th.dataset.sort;
      if (currentSort.key === key) {
        currentSort.asc = !currentSort.asc;
      } else {
        currentSort.key = key;
        currentSort.asc = false;  // first click sorts DESC (good for Sharpe etc.)
      }
      renderRows(lastRecords);
      return;
    }

    // Handle row click to navigate to detail view
    let el = ev.target;
    while (el && el.tagName !== "TR") el = el.parentElement;
    if (el && el.dataset && el.dataset.strategyId) {
      window.location.href = "/strategy/" + el.dataset.strategyId;
    }
  });

  // Initial fetch so sorting works before the first refresh tick.
  refresh();

  if (REFRESH_SEC > 0) {
    setInterval(refresh, REFRESH_SEC * 1000);
  }
})();
