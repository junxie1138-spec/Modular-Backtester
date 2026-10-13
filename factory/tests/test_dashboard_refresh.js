// Run: node factory/tests/test_dashboard_refresh.js
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const script = fs.readFileSync(path.join(__dirname, "../dashboard/static/overview.js"), "utf8");

async function check(discoveryFetch) {
  const total = { textContent: "" };
  const status = { textContent: "" };
  let count = 1;
  let fail = false;
  let tick;
  vm.runInNewContext(script, {
    document: {
      body: { dataset: {} },
      getElementById: id => ({ "c-total": total, "refresh-status": status })[id] || null,
      addEventListener() {},
    },
    console: { warn() {} },
    setInterval: callback => { tick = callback; },
    fetch: async url => {
      if (url === "/api/discovery") return discoveryFetch();
      return {
        ok: !fail,
        json: async () => url === "/api/records" ? [] : { total_cycles: count },
      };
    },
  });
  await new Promise(setImmediate);
  assert.equal(total.textContent, 1, "Discovery must not block the initial results refresh");
  assert.match(status.textContent, /^Updated /);
  count = 2;
  await tick();
  assert.equal(total.textContent, 2, "Polling must continue when discovery is unavailable");
  fail = true;
  await tick();
  assert.equal(total.textContent, 2, "Keep the last results on failure");
  assert.match(status.textContent, /Refresh failed/);
}

(async () => {
  await check(() => ({ ok: false })); // Missing endpoint in an older running server.
  await check(() => Promise.reject(new Error("offline")));
  await check(() => new Promise(() => {})); // Hung discovery request.
  console.log("Dashboard refresh checks passed.");
})().catch(error => { console.error(error); process.exitCode = 1; });
