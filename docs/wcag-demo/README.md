# WCAG 2.2 demo: vegaml

This is the **vegaml-wcag** demo. It evaluates page content with vegaml's own reader and decision engine. It is separate from [LayaWCAG](https://github.com/turbolego/LayaWCAG); the WCAG criteria and page used here are inputs to vegaml, not LayaWCAG's classifier.

## Run the demo

The GitHub Pages site is available at <https://turbolego.github.io/vegaml-wcag/wcag-demo/>.

Choose a backend in the demo:

- **Modal (hosted)** sends the page content and selected criteria to the configured hosted vegaml API.
- **WASM (browser)** runs vegaml's exported reader and engine locally in the browser. It does not call the hosted inference API.

To run the hosted backend locally, install the serving extra and start the model server and demo proxy:

```sh
pip install "vegaml[serve]"
vegaml-serve --model 0.8b --host 0.0.0.0 --port 8080 --preload --cors
```

In another terminal, run `python3 docs/wcag-demo/server.py` from the repository root and open <http://127.0.0.1:8090>. The local proxy forwards `/api` requests to `vegaml-serve`; it avoids browser cross-origin configuration for local development.

Paste an HTML snippet or choose an example, set the probability threshold, and analyze it. The page asks the 63 WCAG 2.2 success-criterion questions, shows ranked probabilities, and lets you download a JSON report. Use `?backend=wasm` to open the demo directly in browser/WASM mode.

## Question sheet and interpretation

[`wcag_criteria.json`](./wcag_criteria.json) contains the 63 criteria as vegaml boolean questions. Each instruction describes the criterion so the model can evaluate the supplied markup; regenerate the question sheet after editing its source with `python3 docs/wcag-demo/make_criteria.py`.

The results are model estimates, not deterministic rule checks. They depend on the checkpoint, supplied markup, and threshold. The text-only reader cannot conclusively evaluate criteria requiring computed styles, screenshots, or interaction (including contrast, reflow, and target size). Treat the demo as an aid, not a substitute for axe-core or a full accessibility audit and human review.

## Building the browser model

The browser backend requires matching ONNX exports of vegaml's reader and engine, tokenizer assets, and bundle metadata. Regenerate the bundle from the trained checkpoint with the repository exporter:

```sh
python -m pip install -e '.[onnx]'
python -m vegaml.wasm_export --checkpoint 0.8b --output-dir docs/wcag-demo/wasm-model --quantize int4 --max-seq-len 128
```

The export must use a reader trained for the selected engine backbone. The exporter records the reader/engine feature-layer configuration, ports the universal engine using its trained statistics, quantizes the Qwen reader's matrix and embedding weights to int4, and writes chunk manifests for large ONNX external-data files. Keep the demo's ONNX Runtime Web assets alongside the generated `wasm-model` directory.

The checked-in Pages bundle uses vegaml's matching Qwen3.5-0.8B reader and vega-U engine. Its reader graph is fixed to 128 tokens because Qwen3.5's gated-delta attention export is not sequence-dynamic in the browser graph. Short prompts are padded; page text and question prefixes are constrained to the graph's input budget. This is a demo for short HTML fragments, not whole-site scanning. The int4 reader is about 341 MiB (357 MB), with a full bundle of about 432 MiB. In a local CPU-browser run, initial load took about 5 minutes and one 63-criterion scan took about 6.4 minutes; actual times vary by device and browser. The demo displays model-download and question-batch progress, uses a service worker to enable multi-threaded WASM where supported, and caches model assets for repeat visits. Browser storage limits or private browsing can disable caching; the model still loads normally. WASM mode does not fall back to the hosted service; use Modal for larger inputs or faster scans.

## Files

| File | Purpose |
|---|---|
| `index.html`, `app.js` | Demo UI, backend selection, and results |
| `server.py` | Local static host and same-origin API proxy |
| `wcag_criteria.json`, `make_criteria.py` | WCAG question sheet and generator |
| `wasm-model/` | Browser reader/engine bundle and ONNX Runtime Web assets |
