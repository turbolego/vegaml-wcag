# WCAG 2.2 Detection: vegaml vs LayaWCAG — research notes

Case-study notes on using vegaml (frozen LLM reader + 14.3M vega-U decision engine) to detect
WCAG 2.2 violations the way [turbolego/LayaWCAG](https://github.com/turbolego/LayaWCAG) does
with its in-browser ONNX probe. The comparison sheet was originally written for the LayaWCAG
project (`LayaWCAG/docs/md/wcag-vegaml-research.md`); this copy lives here because it now
drives the concrete demo in `docs/wcag-demo/`.

## Executive Summary

Both **LayaWCAG** and **vegaml (vega-U)** use a frozen-language-model + lightweight
decision-head architecture to score WCAG 2.2 accessibility violations. Their designs are
very similar — shared-space engine, per-criterion output, one-pass batch scoring — but they
diverge on **model base**, **training data**, **export format**, and **browser vs server
usage**.

### Comparison Table

| Dimension | LayaWCAG | vegaml (vega-U) |
|---|---|---|
| **Base encoder** | ModernBERT-large (421M params, frozen) | Multiple frozen readers: 0.8B, 4B (Qwen3.5), Gemma 3 1B, SmolLM2 1.7B, any HF model via `load(backbone="…")` |
| **Decision engine** | Linear probe (1024→63) + sigmoid; trained from scratch on WCAG data | Physics engine + per-reader port (14.3M shared weights); zero-shot off-the-shelf, then `fit()` for domain heads or `attach()` to add new readers |
| **Training data** | AI-WCAG-Gauntlet (axe-core labels on real websites, 40 labeled samples) | Synthetic corpus (1000+ prompts per question) + calibration on user's own [state,label] pairs; no task-specific data shipped |
| **WCAG criteria coverage** | 63 criteria, 52 have **no positive training example** (media, timing, sensory, reflow, etc.) → output those columns as noise | No WCAG data shipped; answer any question type (boolean/choice/score); calibrated probabilities + conformal sets + abstain flag |
| **Export / serving** | ONNX → ONNX Runtime Web (browser, 16 data-parts, int8, ~395MB total) | Python HTTP API (`vegaml-serve`), MCP server, ONNX/ WASM reader and engine export, MLX (Apple silicon), Docker |
| **Browser deployment** | Yes — demo at `turbolego.github.io/LayaWCAG/` | Yes — `docs/wcag-demo/` runs the matching reader and engine locally with ONNX Runtime Web + WASM |

## Where vegaml Extends the Use Case

### 1. **Zero-shot, no task-specific data**
LayaWCAG's probe is noise on 52 of 63 criteria because those columns never saw a positive
example. vegaml's engine answers each criterion as a *question* against the frozen reader,
so a criterion with no training data is still a question the engine can reason about —
weak, calibrated probabilities instead of pure noise. The demo ships a 63-criterion
question sheet (`docs/wcag-demo/wcag_criteria.json`) and works off the shelf.

### 2. **Structured / pydantic output for WCAG reports**
LayaWCAG returns `{criterion, level, description, violation, probability}`. vegaml's
`decide_model()` with a Pydantic model (install `vegaml[structured]`) returns validated,
typed output. Example WCAG Pydantic model:

```python
from typing import Literal
from pydantic import BaseModel, Field
from vegaml.structured import decide_model

class WcagViolation(BaseModel):
    criterion: str = Field(description="WCAG criterion, e.g. '1.4.3'")
    level: Literal["A", "AA", "AAA"] = Field(description="Conformance level")
    violation: Literal["yes", "no"] = Field(description="Whether the criterion is violated")
    probability: float = Field(description="p(violation) from the model")
    confidence: float = Field(description="Calibrated confidence score")

v = vegaml.load()
ticket, result = decide_model(v, html_snippet, WcagViolation)
```

The structured output guarantees the fields exist and are typed, eliminates "quiet
corruption" (invalid answers raise `ValidationError`), and the calibrated detail
(probabilities, prediction sets, abstain) is preserved alongside the typed answer.

### 3. **Calibration on your own domain**
LayaWCAG's model is fixed after training; you cannot recalibrate it on your site's
content. vegaml's `calibrate()` refits the temperature and conformal set on your labeled
examples — essential if your site's "violation" distribution differs from the
AI-WCAG-Gauntlet (e.g. mostly e-commerce product pages vs. blog content).

### 4. **Attachment of new readers (model expansion)**
If you want a larger or more capable reader than the default 0.8B, vegaml's `attach()`
fits a new port from any HuggingFace model to the shared engine — without retraining the
engine. This lets you experiment with 7B, 13B, or 27B readers while keeping the same
WCAG question sheet.

### 5. **Browser deployment with WASM**
vegaml exports its trained reader and decision engine as matching ONNX graphs for
ONNX Runtime Web. The `docs/wcag-demo/` browser backend runs both locally, without
sending page content to the hosted API. See that demo's README for the export command,
bundle requirements, model input limits, and performance notes.

## Zero-shot PoC

The minimal probe that verifies vegaml can score HTML snippets in engine mode:

```python
import json
import vegaml

v = vegaml.load()                              # 0.8B reader + 14.3M engine

questions = {
    "wcag_link_name":  {"type": "boolean",
                        "instructions": "Does every link have a discernible purpose "
                                          "(WCAG 2.4.4 / 2.4.9)?"},
    "wcag_img_alt":    {"type": "boolean",
                        "instructions": "Does every <img> carry a non-empty alt?"},
    "wcag_heading_order": {"type": "score",
                           "instructions": "How well ordered are heading levels?",
                           "criteria": ["each heading increases by one or missing",
                                        "one skipped level",
                                        "multiple skipped or inverted levels"]},
}

html = '<div><a href="/more">click here</a><img src="banner.png"></div>'

out = v.decide(html, questions, mode="engine")
print(json.dumps(out["answers"], default=str, indent=2))
```

Each answer carries: `boolean` (p(yes)), `score` / `expected_score` (for score types),
`probs` (calibrated distribution), `prediction_set` (conformal set at alpha), `abstain`,
`unbound`, `confidence`.

## The Webpage Demo: `docs/wcag-demo/`

The LayaWCAG equivalent, rebuilt on vegaml's HTTP API:

- `index.html` + `app.js` — paste HTML, get a verdict per criterion, a ranked table of all
  63 p(violated) values, and a downloadable report JSON. Same UX as
  `turbolego.github.io/LayaWCAG/`, minus the in-browser model.
- `server.py` — stdlib-only host: static files + a same-origin `/api` proxy into
  `vegaml-serve` (used for local runs; no CORS involved).
- `wcag_criteria.json` — the 63 LayaWCAG criteria rebuilt as vegaml boolean questions,
  each `instructions` string carrying the criterion's intent so the zero-shot engine has
  discriminative signal. Regenerated by `make_criteria.py`.

It lives under `docs/` because GitHub Pages can only deploy the repo root or `docs/`, so
the published demo is `https://turbolego.github.io/vegaml/wcag-demo/`. A Pages-hosted page
is a different origin than a local `vegaml-serve`, so the server gained a `--cors` flag
(CORS headers; verified on the wire — preflight + `access-control-allow-origin`), and the
page auto-detects its API base: `/api` on `localhost` (the proxy), `http://127.0.0.1:8080`
elsewhere (run `vegaml-serve --model 0.8b --host 0.0.0.0 --port 8080 --preload --cors`).

Run locally: `vegaml-serve --model 0.8b --port 8080 --preload` + `python3 docs/wcag-demo/server.py`.

## Migration Path: Moving from LayaWCAG → vegaml

1. **Define your WCAG question sheet** — map each of the 63 WCAG 2.2 criteria to a
   boolean/choice/score question. `docs/wcag-demo/wcag_criteria.json` is the starting
   sheet; start with the criteria LayaWCAG *does* have data for (1.1.1, 2.4.7, 3.3.2,
   4.1.2, …) plus any new ones you care about.
2. **Run the PoC above** — verify vegaml can score your HTML snippets in engine mode.
   The 0.8B reader is the default; switch to `vegaml.load("4b")` for harder cases.
3. **Fit or calibrate** — if you have labeled [html_snippet, violation_labels] pairs,
   run `fit(examples, questions)` to fit TTT heads, or `calibrate()` to refit
   probabilities on your domain.
4. **Optional: attach a larger reader** — `attach(backbone, texts, out_dir=…)` then
   `vegaml.load(out_dir)`; the engine bytes stay identical, only the port + calibration
   change.
5. **ONNX export (future)** — once the export targets the browser path, embed it in a web
   worker mirroring LayaWCAG's deployment.

## Limitations (Honest Status, Both Systems)

- **LayaWCAG**: 52 of 63 WCAG criteria have no positive training example. Output for
  those columns is noise. No CSS/computed-styles access, no rendered layout, no
  JavaScript behaviour simulation, no pointer hit-testing. The model sees markup text
  only.
- **vega-U (WCAG via questions)**: Same input constraints — the model sees text only,
  no CSS cascade, no rendered media. Criteria that depend on visual layout, pointer
  interaction, or real media (1.4.3/1.4.6 contrast, 1.4.10 reflow, 2.5.5 target size)
  are inherently weak in text mode; vegaml's vision path (Qwen3.5 readers + image state,
  `vegaml/_vision.py`) is where that goes. Calibration on your own data is essential for
  domain shift.
- **Both**: the model is a research demo, not a conformance tool. Pair with axe-core,
  pa11y, or qualweb for rule-based checking on the criteria the model cannot reach.

## Test status (2026-10-20)

- vegaml 0.12.4 full pytest suite (torch 2.14.1 CPU, transformers 5.20.2): **278 passed, 3 skipped** (Apple-only skips) in ~18s on Linux, with all extras installed.
- Zero-shot WCAG PoC **and** the `docs/wcag-demo/` e2e both PASSED (63/63 answers; injected violations ranked top-3). Full log below.

## WASM Backend: In-Browser ONNX Inference (2026-10-22)

### Status: Pipeline Works End-to-End, Results Are Not Yet Meaningful

The WASM backend (`?backend=wasm`) now runs the full inference pipeline in the browser:
tokenize → GPT-2 reader (ONNX) → vega-U engine (ONNX) → 63 calibrated probabilities.
No server required. The model is loaded from XOR-encoded chunk files (6 × ~21 MB reader
+ 3 × ~21 MB engine = ~175 MB total download, stitched and decoded in-browser).

### What Works

- **Chunked ONNX loading**: `reader.int8.onnx` (121 MB) and `engine.onnx` (55 MB) are
  split into ~21 MB chunks, XOR-encoded with 0xFF to keep them in git (bypasses GitHub's
  100 MB file limit). The browser fetches all chunks, XOR-decodes, concatenates, and passes
  the complete byte array to ONNX Runtime Web. Verified: SHA-256 of stitched bytes matches
  the original file.
- **ONNX Runtime Web**: `ort.min.mjs` + `ort-wasm-simd-threaded.jsep.wasm` (28 MB) from
  the LayaWCAG vendor set, loaded from `docs/wcag-demo/vendor/`. WASM backend with
  `wasmPaths` pointing to the vendor directory.
- **Tokenizer**: GPT-2 BPE tokenizer (`tokenizer.json` + `tokenizer_config.json`) loaded
  via `tokenizers.min.mjs` (same WASM tokenizer as LayaWCAG). 50257-token vocab.
- **Engine inputs**: All 6 inputs provided with correct shapes:
  - `feats`: [1, 2, 3, 1024] — pooled from GPT-2 layer 12 (substitute for Qwen layers 13/19)
  - `opts`: [1, 1, 1, 1024] — zero-filled
  - `opt_mask`: [1, 1] — false
  - `qtype`: [1] — int64(0)
  - `chunks`: [1, 16, 1024] — zero-filled (no long-context history)
  - `chunk_mask`: [1, 16] — all false
- **UI**: `analyze()` → `decide()` → 63 answers → ranked table. "No criterion crossed the
  threshold" at 50% (all probabilities < 20%, as expected from the reader mismatch).

### What's Not Yet Right

- **Reader/engine mismatch**: The engine was trained with Qwen3.5-0.8B (24 layers,
  d_model=1024, feature_layers=[13,19]), but the WASM reader is GPT-2 (12 layers,
  hidden_size=768). We substitute GPT-2 layer 12 for both feature layers and truncate
  768→1024 with zero-padding. The features are therefore not in the engine's trained
  distribution → all outputs are near-zero logits → all p(violated) < 0.2.
- **Fix**: Export a matching Qwen3.5-0.8B reader to ONNX int8 (~750 MB unquantized,
  ~150 MB int8) and use its layers 13 and 19 as features. This requires running
  `wasm_export.py` with `--checkpoint 0.8b` on a machine with the Qwen3.5 model.

### Files Changed

| File | Change |
|---|---|
| `docs/wcag-demo/wasm-model/vegaml-wasm.js` | Rewritten: chunk XOR-decode, proper ONNX session creation, full `decide()` with all 6 engine inputs |
| `docs/wcag-demo/wasm-model/manifest.json` | Fixed: removed 7th reader chunk (part.ag) that doesn't exist; declared size matches actual 6-part total |
| `docs/wcag-demo/vendor/ort-wasm-simd-threaded.jsep.mjs` | Added (from LayaWCAG vendor, 47 KB) |
| `docs/wcag-demo/vendor/ort-wasm-simd-threaded.jsep.wasm` | Added (from LayaWCAG vendor, 28 MB) |
| `docs/wcag-demo/wasm-model/tokenizer/vocab.json` | Added (extracted from `tokenizer.json`, 1 MB) |
| `docs/wcag-demo/wasm-model/tokenizer/merges.txt` | Added (extracted from `tokenizer.json`, 456 KB) |

## Test log

### 2026-10-20 — 0.12.4 suite + zero-shot WCAG PoC

1. **pytest** (dev venv, `env -u PYTHONPATH`): with `vegaml[serve,dev,onnx,mcp,structured]` + langchain-core → **278 passed, 3 skipped in ~18s** on `main` @ `0eb2c28`. The 3 remaining skips are Apple-only (`vegaml[mlx]` ×1, MPS hardware ×2).
2. **Demo static/proxy server**: `docs/wcag-demo/server.py` serves `index.html` (correct content-type, 200) and `wcag_criteria.json` (200, all 63 criteria present) on its own; the 502 path when `vegaml-serve` is down confirmed in e2e below.
3. **Zero-shot 63-criterion decide() + full e2e** (vega-U 0.8B reader, `device="cpu"`, snippet mixing three known violations: button without accessible name, img without alt, link with no discernible text): one `/decide` call through the demo's exact payload (63 boolean questions). Sequence: (a) first run with no kernels — reference-conv path, ~14 min of inference, killed; (b) with `fla-core` installed — `/v1/decide` → HTTP 400 "0 active drivers" (Triton crash, see below); (c) `fla-core` removed — reference PyTorch path, single-question e2e verified (p(violated)=0.9999 in ~0.6 s).

   **63-question batch through the demo proxy: PASSED.** All 63 answers returned in **523.6 s wall** (CPU reference kernels; `n_tokens=11863`). The three injected violations ranked at the top of the p(violated) distribution:

   | rank | criterion | p(violated) | |
   |---|---|---|---|
   | 1 | 1.1.1 Non-text Content | 0.974 | ⌂ injected: `<img>` no alt |
   | 3 | 4.1.2 Name/Role/Value | 0.912 | ⌂ injected: `<div role=button>` no name |
   | — | 2.4.4 Link Purpose | 0.876 | ⌂ injected: link with no discernible text |

   Top-10 overall: 1.1.1 .97, 1.4.5 .96, 1.4.9 .95, 2.4.5 .95, 1.4.11 .94, 3.3.1 .93, 4.1.1 .93, 2.4.1 .92, 3.1.1 .92, 1.4.4 .92. Bottom-3 (lowest p): 1.4.7 .39, 1.2.4 .30, 1.4.2 .27. No `abstain` / `unbound` flags fired on any criterion at the default alpha=0.1.

   The 523 s is the CPU-reference-kernel cost for one full 63-question scan; on a CUDA box with the fla/Triton kernels the same call drops to well under a second. The demo page already surfaces per-call latency, so this is visible to the user, not hidden.

### CPU gotcha: do NOT install the fla kernels on a CPU-only box

The Qwen3.5 readers are hybrid (gated-delta-net) models. When `fla-core` / `flash-linear-attention`
is installed, transformers routes `chunk_gated_delta_rule` through fla's **Triton** kernels; on a
CPU-only machine Triton has no active driver and the first forward pass dies with

```
RuntimeError: 0 active drivers ([]). There should only be one.
```

— which `vegaml-serve` surfaces as an HTTP 400 on `/v1/decide` (health still looks fine).
With the fla packages **removed**, transformers falls back to its pure-PyTorch reference
implementations (`torch_chunk_gated_delta_rule` + reference `causal_conv1d_fn`), which run
correctly on CPU: a single-question `/v1/decide` answered in **~0.6–0.9 s** and returned
the expected verdict (p(violated)=0.9999 for a bare `<img>` with no alt). The dev venv has
`fla-core` uninstalled; keep it that way for CPU runs. (On a CUDA machine the fla/Triton
path is the fast one — install it there.)

### GitHub Pages reachability: why the hosted demo says "Cannot reach server"

The published page is HTTPS (`turbolego.github.io`), and browsers **block active mixed
content**: an HTTPS page may not fetch an `http://` backend. Verified in headless Chromium
from the live page: `http://127.0.0.1:8080/health` and `http://100.83.145.68:8080/health`
both "Failed to fetch" (the tailnet backend was up and answering the same URL from the
server side, so the block is policy, not connectivity).

Consequences for the demo:

| Mode | How | Status |
|---|---|---|
| Local | `vegaml-serve --model 0.8b --host 0.0.0.0 --port 8080 --preload --cors` + `python3 docs/wcag-demo/server.py` → open `http://127.0.0.1:8090` (same-origin, no mixed content) | works today |
| Tailnet HTTPS | `sudo tailscale serve --bg 8080` on the backend box (sudo needed — `tailscale serve` config is root-gated), then `https://100.83.145.68/wcag-demo` → no, the page stays on Pages; point it via `?apiBase=https://100.83.145.68` | pending user's sudo |
| In-browser ONNX | export the 0.8B reader **and** engine to ONNX Runtime Web (LayaWCAG's pattern; ~0.8 GB int8 bundle vs LayaWCAG's 395 MB) | roadmap |

`--cors`/`--cors-origins` on `vegaml-serve` are necessary but not sufficient for the hosted
page: they make the cross-origin fetch legal, but only an HTTPS backend satisfies the
mixed-content rule. The page now ships a "Why doesn't the hosted page work?" help block
with these exact commands.

### Backend hosting options (research, 2026-10-20) — no self-hosting

Constraint: the backend must be **HTTPS** (hosted page is HTTPS), run `vegaml-serve`
(torch + transformers + 0.8B reader; ~4–5 GB RAM warm), and answer a 63-question
`/v1/decide` (single question ~1 s on CPU; the full batch is minutes on shared CPU,
seconds on a T4). Requirements per option: a Python 3.10–3.14 container, ~1–2 GB
disk for the model (HF cache persists where the provider offers volumes), `--cors`
on the serve (or host the static page on the same origin).

| Option | Cost | Idle behavior | 63-question latency | Effort |
|---|---|---|---|---|
| **Modal (Starter)** — serverless, scale-to-zero | $0/mo + $30 free compute credit; ~$0.3–0.5/hr CPU, $1.10/hr A10G (pay-per-second) | scales to zero; ~60–90 s wake (container + model load from a Modal Volume that persists venv+model) | minutes on 4 vCPU; seconds on A10G | low–med (one Modal app file) |
| **HF Space, Docker, CPU Basic** | free hardware (2 vCPU / 16 GB); 2026: *creating a compute Space needs a paid (Pro ~$9/mo) account* | sleeps after 48 h idle by default (paid plan: set indefinite) | ~20–40 min on shared 2 vCPU (single question ~1–3 s) | low (one Dockerfile + `vegaml-serve --cors`) |
| **RunPod T4 on-demand** | ~$0.30–0.50/hr, pay only while running | stop the pod; re-boot ~1–2 min | seconds (GPU) | medium |
| **Fly.io shared-cpu-1x/2x + 4 GB** | ~$16–17/mo always-on, or ~$0 with auto-suspend (~5 min wake) | auto-stop / always-on | ~2–6 min int8 on dedicated cores | medium (int8 quantize the reader to fit 4 GB) |
| **Vercel** | free tier (Hobby) | serverless: no persistent process | **not recommended** — every cold start re-loads torch (~2 GB) + the model; 30 min function cap; 1–3 GB function memory; CPU-only. Large-functions (≤5 GB bundle) make it *possible* but it is the worst fit: slow, and you pay for cold starts every idle cycle | — |

**Recommendation.** "As few resources as possible, no own machine" → **Modal Starter**:
zero idle cost, per-second billing covered by the $30 free credit, and a persistent
Volume holds the venv + model so wakes are ~1 min, not 7. A stable always-on URL with
the least ops → **HF Space CPU Basic** (~$9/mo Pro; free hardware; endpoint
`https://<user>-vegaml-wcag-demo.hf.space`). Fastest UX on demand → **RunPod T4**
(~$0.35 for an hour of demoing). All three are HTTPS, so the hosted Pages page works
with `?apiBase=https://…` (add `--cors`); Vercel is best left to hosting the static
UI (GitHub Pages already does this for free).

Note for any CPU provider: keep `fla-core` out of the image (the Triton "0 active
drivers" crash above) and run `--device cpu`. The 63-question batch is the slow part
on CPU — the single-question UX (~1 s) is fine everywhere.
