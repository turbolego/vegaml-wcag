// vegaml WCAG 2.2 detector — supports both Modal (server) and WASM (browser) backends
const $ = (id) => document.getElementById(id);

const EXAMPLES = {
  "Image without alt text": '<img src="chart.png" width="200">\n<p>Q3 results</p>',
  "Button with no accessible name": '<div role="button" class="btn"></div>',
  "Form input without label": '<form><input type="email" name="email" placeholder="you@example.com"><button>Go</button></form>',
  "Heading level skipped": '<h1>Page title</h1>\n<h4>First section</h4>\n<p>Body text</p>',
  "Link with no discernible text": '<a href="/next"><img src="arrow.gif"></a>',
  "Iframe without title": '<iframe src="https://example.com/widget" width="600" height="400"></iframe>',
  "Clean semantic snippet": '<main aria-label="Main">\n  <h1>Contact us</h1>\n  <label for="email">Email</label>\n  <input id="email" type="email" name="email">\n  <img src="logo.png" alt="Company logo">\n  <button type="submit">Send</button>\n</main>',
};

let criteria = [];        // [{criterion, principle, level, description, instructions}]
let lastReport = null;    // the full report we can download

// Configuration
const DEFAULT_REMOTE_API = "https://hummernkloa--api.us-east.modal.run";
const WASM_MODEL_DIR = "./wasm-model/";

// Backend mode: "modal" or "wasm" (can be overridden with ?backend=wasm|modal)
let backendMode = new URLSearchParams(location.search).get("backend") || "modal";

function server() {
  const base = new URLSearchParams(location.search).get("apiBase");
  if (base) return base.replace(/\/+$/, "");

  if (backendMode === "wasm") {
    return ""; // WASM runs locally, no server needed
  }

  const h = location.hostname;
  return (h === "localhost" || h === "127.0.0.1") ? "/api" : "https://hummernkloa--api.us-east.modal.run";
}

function apiPath(path) {
  // For Modal backend, use /v1/ prefix
  return path.startsWith("/v1") ? path : "/v1" + path;
}

async function fetchJSON(url, opt) {
  const r = await fetch(url, opt);
  if (!r.ok) throw new Error(r.status + " " + r.statusText);
  return r.json();
}

function setStatus(msg, cls = "") {
  const s = $("status");
  s.textContent = msg;
  s.className = "status " + cls;
}

function buildQuestions() {
  const q = {};
  for (const c of criteria) q[c.criterion] = { type: "boolean", instructions: c.instructions };
  return q;
}

// WASM instance
let wasmInstance = null;

async function loadWASM() {
  if (wasmInstance) return wasmInstance;
  
  setStatus("Loading the browser reader and engine models...", "loading");
  
  try {
    const { VegamlWASM } = await import('./wasm-model/vegaml-wasm.js');
    const instance = await VegamlWASM.create({ modelDir: WASM_MODEL_DIR });
    return instance;
  } catch (e) {
    throw new Error("Failed to load WASM model: " + e.message);
  }
}

async function analyze() {
  const html = $("html").value.trim();
  if (!html) { setStatus("Paste some HTML first.", "err"); return; }
  const th = parseFloat($("threshold").value) || 0.5;
  $("analyze").disabled = true;
  setStatus("Asking vegaml 63 questions...", "loading");
  const t0 = performance.now();
  const questions = buildQuestions();
  
  try {
    let out;
    let modelLoadMs = 0;
    if (backendMode === "wasm") {
      // WASM backend
      if (!wasmInstance) {
        setStatus("Loading WASM model...", "loading");
        const loadStarted = performance.now();
        wasmInstance = await loadWASM();
        modelLoadMs = Math.round(performance.now() - loadStarted);
      }
      out = await wasmInstance.decide($("html").value.trim(), questions);
      out.mode = "wasm";
      if (modelLoadMs) out.model_load_ms = modelLoadMs;
    } else {
      // Modal backend
      const apiBase = server();
      const decideUrl = apiBase.startsWith("/api") ? apiBase + "/decide" : apiBase + "/v1/decide";
      out = await fetchJSON(decideUrl, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ state: $("html").value.trim(), questions }),
      });
    }
    const totalMs = Math.round(performance.now() - t0);
    const inferenceMs = backendMode === "wasm"
      ? (out.latency_ms || Math.max(0, totalMs - modelLoadMs))
      : (out.latency_ms || totalMs);
    lastReport = {
      html: $("html").value.trim(), threshold: parseFloat($("threshold").value) || 0.5,
      latency_ms: inferenceMs, total_latency_ms: totalMs,
      model_load_ms: modelLoadMs || null, server: server(), criteria: [], raw: out,
    };
    const answers = out.answers || {};
    const rows = criteria.map((c) => {
      const a = answers[c.criterion] || {};
      const p = parseFloat(a.boolean);
      return {
        criterion: c.criterion, principle: c.principle, level: c.level, description: c.description,
        p_violated: Number.isFinite(p) ? p : null,
        abstain: !!a.abstain, unbound: !!a.unbound,
        prediction_set: a.prediction_set, confidence: a.confidence,
      };
    });
    rows.sort((x, y) => (y.p_violated ?? -1) - (x.p_violated ?? -1));
    lastReport.criteria = rows;
    const violations = rows.filter((r) => r.p_violated != null && r.p_violated >= (parseFloat($("threshold").value) || 0.5));
    render(rows, violations, out);
    const loadNote = modelLoadMs ? " Model loaded in " + (modelLoadMs / 1000).toFixed(1) + " s." : "";
    setStatus("Done in " + inferenceMs + " ms inference — " + violations.length + " of " + rows.length + " criteria at or above p ≥ " + ($("threshold").value || 0.5) + "." + loadNote, "ok");
  } catch (e) {
    setStatus("decide() failed: " + e.message, "err");
  } finally {
    $("analyze").disabled = false;
  }
}

function bar(p) {
  const w = Math.max(0, Math.min(100, Math.round((p ?? 0) * 100)));
  return '<div class="bar"><i style="width:' + w + '%"></i></div>';
}

function render(rows, violations, out) {
  const s = $("summary");
  s.innerHTML =
    '<span class="pill">' + violations.length + ' violation' + (violations.length === 1 ? "" : "s") + ' / ' + rows.length + '</span>' +
    '<span class="pill">p ≥ ' + ($("threshold").value || 0.5) + '</span>' +
    '<span class="pill">' + (out.latency_ms || 0) + ' ms inference</span>' +
    (out.model_load_ms ? '<span class="pill">model load ' + (out.model_load_ms / 1000).toFixed(1) + ' s</span>' : '') +
    (out.n_tokens ? '<span class="pill">' + out.n_tokens + ' tokens in</span>' : "");

  // Violation table
  const vbody = $("violationRows");
  vbody.innerHTML = violations.map((r) => '<tr>\n      <td class="crit">' + r.criterion + '<span class="lv ' + r.level.toLowerCase() + '">' + r.level + '</span></td>\n      <td>' + r.description + '</td>\n      <td class="pct">' + (r.p_violated * 100).toFixed(1) + '%</td>\n      <td>' + bar(r.p_violated) + ' ' + (r.abstain ? '<span class="flag">abstain</span> ' : "") + (r.unbound ? '<span class="flag">OOD</span>' : "") + '</td>\n    </tr>').join("") || '<tr><td colspan="4" class="muted">No criterion crossed the threshold.</td></tr>';
  $("violationWrap").style.display = violations.length ? "" : "none";

  // Full ranked table
  $("allRows").innerHTML = rows.map((r) => '<tr>\n      <td class="crit">' + r.criterion + '<span class="lv ' + r.level.toLowerCase() + '">' + r.level + '</span></td>\n      <td>' + r.description + '</td>\n      <td class="pct">' + (r.p_violated == null ? "—" : (r.p_violated * 100).toFixed(1) + "%") + '</td>\n      <td>' + (r.p_violated == null ? "—" : bar(r.p_violated)) + '</td>\n    </tr>').join("");

  $("raw").textContent = JSON.stringify(out, null, 2);
  $("results").classList.remove("hidden");
  $("download").classList.remove("hidden");
}

function download() {
  if (!lastReport) return;
  const blob = new Blob([JSON.stringify(lastReport, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "wcag-vegaml-report.json";
  a.click();
  URL.revokeObjectURL(a.href);
}

// --- wire up ---
async function init() {
  criteria = await fetchJSON(new URL("./wcag_criteria.json", location.href));
  
  // Set backend mode from URL or default to modal
  const urlBackend = new URLSearchParams(location.search).get("backend");
  if (urlBackend) backendMode = urlBackend;
  
  // Add backend selector to UI
  addBackendSelector();
  
  $("server").value = new URLSearchParams(location.search).get("apiBase") || server();
  $("example").addEventListener("change", (e) => { const t = EXAMPLES[e.target.value]; if (t) $("html").value = t; });
  $("check").addEventListener("click", checkServer);
  $("analyze").addEventListener("click", analyze);
  $("download").addEventListener("click", download);
  $("html").value = EXAMPLES["Image without alt text"];
  $("analyze").disabled = false;
  
  // Add backend mode toggle to URL
  const urlParams = new URLSearchParams(location.search);
  urlParams.set("backend", backendMode);
  history.replaceState(null, "", "?" + urlParams.toString());
  
  checkServer();
}

function addBackendSelector() {
  const serverCard = document.querySelector('.card');
  if (!serverCard) return;
  
  const selectorHtml = `
    <div style="margin-top: 1rem; padding: 1rem; background: var(--bg); border-radius: 8px; border: 1px solid var(--line);">
      <label style="font-weight: 600; display: block; margin-bottom: 0.5rem;">Backend Mode</label>
      <div style="display: flex; gap: 1rem; flex-wrap: wrap; align-items: center;">
        <label style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer;">
          <input type="radio" name="backend" value="modal" ${backendMode === "modal" ? "checked" : ""} onchange="switchBackend('modal')">
          <span>Modal (server)</span>
        </label>
        <label style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer;">
          <input type="radio" name="backend" value="wasm" ${backendMode === "wasm" ? "checked" : ""} onchange="switchBackend('wasm')">
          <span>WASM (browser)</span>
        </label>
        <span style="font-size: 0.85rem; color: var(--ink2); margin-left: 1rem;">
          Modal: ~1 min cold start, then ~1s/query | WASM: ~430 MiB model download, then minutes per scan
        </span>
      </div>
    </div>
  `;
  
  serverCard.insertAdjacentHTML('beforeend', selectorHtml);
}

function switchBackend(mode) {
  backendMode = mode;
  const urlParams = new URLSearchParams(location.search);
  urlParams.set("backend", mode);
  history.replaceState(null, "", "?" + urlParams.toString());
  $("server").value = server();
  checkServer();
}

async function checkServer() {
  const s = $("serverStatus");
  s.className = "status";
  s.textContent = "Checking...";

  if (backendMode === "wasm") {
    s.className = "status ok";
    s.textContent = "WASM mode — runs entirely in browser (no server needed)";
    return;
  }

  try {
    const h = await fetchJSON(server() + "/health");
    const m = await fetchJSON(server() + "/model").catch(() => null);
    s.className = h.model_loaded ? "status ok" : "status";
    if (h.model_loaded) {
      s.textContent = "vegaml " + (m?.model || "default") + " · reader " + (m?.backbone || "?") + " · context " + (m?.max_len ?? "?") + " tokens · ready";
    } else {
      s.textContent = "Server up, model still loading — wait a minute and check again (cold load of the 0.8B reader takes several minutes).";
    }
  } catch (e) {
    s.className = "status err";
    s.textContent = "Cannot reach server (" + e.message + "). The backend may still be waking (first call after idle takes ~1 min) — retry Check server, or point the API input at your own vegaml-serve (run: vegaml-serve --model 0.8b --host 0.0.0.0 --port 8080 --preload --cors).";
  }
}

async function loadWASMIfNeeded() {
  if (!wasmInstance) {
    setStatus("Loading the browser reader and engine models...", "loading");
    const { VegamlWASM } = await import('./wasm-model/vegaml-wasm.js');
    wasmInstance = await VegamlWASM.create({ modelDir: WASM_MODEL_DIR });
  }
  return wasmInstance;
}

// --- wire up ---
async function init() {
  criteria = await fetchJSON(new URL("./wcag_criteria.json", location.href));
  
  // Set backend mode from URL or default to modal
  const urlBackend = new URLSearchParams(location.search).get("backend");
  if (urlBackend) backendMode = urlBackend;
  
  // Add backend selector to UI
  addBackendSelector();
  
  $("server").value = new URLSearchParams(location.search).get("apiBase") || server();
  $("example").addEventListener("change", (e) => { const t = EXAMPLES[e.target.value]; if (t) $("html").value = t; });
  $("check").addEventListener("click", checkServer);
  $("analyze").addEventListener("click", analyze);
  $("download").addEventListener("click", download);
  $("html").value = EXAMPLES["Image without alt text"];
  $("analyze").disabled = false;
  
  // Add backend mode to URL
  const urlParams = new URLSearchParams(location.search);
  urlParams.set("backend", backendMode);
  history.replaceState(null, "", "?" + urlParams.toString());
  
  checkServer();
}

document.addEventListener("DOMContentLoaded", init);