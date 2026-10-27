"""Host the vegaml WCAG demo backend on Modal — scale-to-zero, pay-per-second.

The web function serves the same FastAPI app `vegaml-serve` exposes
(`/health`, `/v1/model`, `/v1/decide`, `/v1/decide/batch`) with CORS enabled,
so the hosted demo page at turbolego.github.io/vegaml/wcag-demo/ can call it
directly via `?apiBase=https://<your-modal-endpoint>`.

Resources: 4 vCPU / 16 GB CPU container. The 0.8B reader + engine are ~4–5 GB
warm; the CPU torch wheel keeps the image small. Idle: the container scales
down after ~10 idle minutes and costs $0 while asleep.

Setup (any machine, Modal account + `modal setup` done):

    pip install modal
    modal run deploy/modal/app.py prewarm      # one-off: cache the model into the volume
    modal deploy deploy/modal/app.py           # deploys the /v1/decide endpoint, prints its URL

Then open the hosted demo with `?apiBase=https://<workspace>-<name>.<region>.modal.host`.

Cost notes: $0/mo Starter plan + $30/mo free compute credit. Prewarm is one
~16 GB·min of CPU (a few cents). A 63-question demo run ≈ 15–30 min of 4 vCPU
≈ $0.10–0.20; add `gpu="A10G"` to the function specs and the same run is a few
seconds (~$0.02).

Verified against the modal 1.6.x SDK: the web function is the stacked
`@app.function(...)` + `@modal.asgi_app(...)` form (the asgi decorator must
compose with the function decorator, not the other way round).
"""
import os

import modal

app = modal.App("vegaml-wcag-demo")

# The 0.8B reader + vega-U engine (~1.5 GB) live in a persistent Volume so a
# cold start loads from the volume (~1–2 min) instead of re-downloading (~7 min).
hf_cache = modal.Volume.from_name("vegaml-hf-cache", create_if_missing=True)

# CPU torch wheel (~800 MB) instead of the ~2.5 GB CUDA build: vegaml is pure
# Python, so install its non-torch deps first, then torch from the CPU index,
# then vegaml itself (its torch requirement is already satisfied).
IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("transformers", "safetensors", "huggingface_hub", "numpy", "fastapi", "uvicorn")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("vegaml==0.12.4")
)

# 4 vCPU / 16 GB: warm RSS is ~4–5 GB. timeout=3600 covers the long 63-question
# batch on CPU. startup_timeout=3600: the ASGI factory (model load from the
# volume) runs at container start and needs more than the 300 s default budget
# on a cold volume. scaledown_window: sleep after ~10 idle minutes ($0 asleep).
SPECS = dict(image=IMAGE, volumes={"/hf": hf_cache}, cpu=4, memory="16GB",
             timeout=3600, startup_timeout=3600, scaledown_window=600)


_MODEL = None


def _load():
    """Load the model once per container boot. HF_HOME points at the persistent
    volume, so a prewarmed boot skips the download entirely."""
    global _MODEL
    if _MODEL is None:
        os.environ["HF_HOME"] = "/hf"
        import vegaml
        _MODEL = vegaml.load("0.8b", device="cpu")
    return _MODEL


@app.function(**SPECS)
@modal.asgi_app(label="api")
def api():
    """The web endpoint: vegaml's FastAPI app with CORS — same routes as vegaml-serve.

    Note: we attach the CORSMiddleware here instead of via create_app(cors=...),
    because the PyPI vegaml 0.12.4 release predates that parameter (it's in git
    main but not published); this keeps the image on the plain PyPI package.

    Also add /model and /decide aliases (without /v1) so the remote demo page
    works without code changes.
    """

    from fastapi.middleware.cors import CORSMiddleware

    from vegaml.serve import create_app
    model = _load()
    app = create_app(model=model, model_name="0.8b")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])
    # Alias /model -> /v1/model, /decide -> /v1/decide for remote demo compatibility
    # FastAPI's router makes this trivial: just add the same handler to the new paths
    for route in app.routes:
        if hasattr(route, 'path') and route.path == "/v1/model":
            app.router.add_api_route("/model", route.endpoint, methods=["GET"], include_in_schema=False)
        elif hasattr(route, 'path') and route.path == "/v1/decide":
            app.router.add_api_route("/decide", route.endpoint, methods=["POST"], include_in_schema=False)
    return app


@app.function(image=IMAGE, volumes={"/hf": hf_cache},
               cpu=4, memory="16GB", timeout=3600)
def prewarm():
    """One-off: download the model into the persistent volume.
    Run once with:  modal run deploy/modal/app.py prewarm"""
    _load()
    print("prewarm OK: 0.8B reader loaded; HF cache now persisted at /hf on the volume")
