#!/usr/bin/env python3
"""
Export the full vegaml pipeline (reader + engine) to ONNX for browser/WASM deployment.

This exports BOTH the reader (frozen LLM backbone) AND the engine (14M physics engine)
to ONNX format suitable for onnxruntime-web in the browser.

Components exported:
1. Reader/Encoder: The frozen LLM backbone (e.g., Qwen3.5-0.8B, Gemma-1B, etc.)
2. Engine: The 14M physics-based decision engine (already exported via export_engine)
3. Tokenizer: Exported as JSON for WASM usage

Usage:
    python -m vegaml.wasm_export --checkpoint 0.8b --output-dir ./wasm-model
    python -m vegaml.wasm_export --checkpoint 0.8b --output-dir ./wasm-model --quantize int4 --max-seq-len 128
"""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import MethodType
from typing import Iterator, Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ._universal import UniversalEngine
from .hub import backbone_hub_id, fetch
from .onnx_export import _load_engine
from .onnx_export import export_engine as export_engine_internal

WASM_CHUNK_BYTES = 20 * 1024 * 1024
WASM_MAX_FILE_BYTES = 95 * 1024 * 1024


def _unit_lower_triangular_inverse(matrix: torch.Tensor) -> torch.Tensor:
    size = matrix.shape[-1]
    if size == 1:
        return torch.ones_like(matrix)
    half = size // 2
    upper = _unit_lower_triangular_inverse(matrix[..., :half, :half])
    lower = _unit_lower_triangular_inverse(matrix[..., half:, half:])
    cross = -lower @ matrix[..., half:, :half] @ upper
    top = torch.cat((upper, torch.zeros_like(matrix[..., :half, half:])), dim=-1)
    bottom = torch.cat((cross, lower), dim=-1)
    return torch.cat((top, bottom), dim=-2)


@contextmanager
def _qwen3_5_triangular_solve_patch() -> Iterator[None]:
    from transformers.models.qwen3_5 import modeling_qwen3_5

    original_export_check = modeling_qwen3_5.is_torchdynamo_exporting
    original_solve = torch.linalg.solve_triangular

    def exportable_solve(matrix, rhs, upper=False, left=True, unitriangular=False, **kwargs):
        if upper or not left or not unitriangular:
            return original_solve(
                matrix,
                rhs,
                upper=upper,
                left=left,
                unitriangular=unitriangular,
                **kwargs,
            )
        return _unit_lower_triangular_inverse(matrix) @ rhs

    modeling_qwen3_5.is_torchdynamo_exporting = lambda: False
    torch.linalg.solve_triangular = exportable_solve
    try:
        yield
    finally:
        torch.linalg.solve_triangular = original_solve
        modeling_qwen3_5.is_torchdynamo_exporting = original_export_check


def _prepare_qwen3_5_onnx_export(model, model_type: str) -> None:
    if not model_type.startswith("qwen3_5"):
        return

    from transformers.activations import ACT2FN
    from transformers.models.qwen3_5 import modeling_qwen3_5

    base_model = model.model
    wrapped_forward = getattr(base_model.forward.__func__, "__wrapped__", None)
    if wrapped_forward is None:
        raise RuntimeError("Qwen3.5 base model is missing its exportable forward implementation")
    base_model.forward = MethodType(wrapped_forward, base_model)

    convolutions = [
        module for name, module in model.named_modules()
        if name.endswith(".conv1d") and isinstance(module, torch.nn.Conv1d)
    ]
    dimensions = {(module.kernel_size[0], module.groups) for module in convolutions}
    if len(dimensions) != 1:
        raise ValueError("Qwen3.5 reader must use one shared causal convolution shape for ONNX export")
    kernel_size, groups = dimensions.pop()

    def causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
        sequence_length = hidden_states.shape[-1]
        output = torch.nn.functional.conv1d(
            hidden_states.to(weight.dtype),
            weight.unsqueeze(1),
            bias,
            padding=kernel_size - 1,
            groups=groups,
        )[:, :, :sequence_length]
        if activation is not None:
            output = ACT2FN[activation](output)
        return output.to(hidden_states.dtype)

    modeling_qwen3_5.causal_conv1d_fn = causal_conv1d_fn


def _external_data_locations(graph, onnx) -> set[str]:
    locations = set()

    def add_tensor(tensor) -> None:
        locations.update(
            item.value for item in tensor.external_data if item.key == "location"
        )

    def visit_graph(current) -> None:
        for tensor in current.initializer:
            add_tensor(tensor)
        for sparse in current.sparse_initializer:
            add_tensor(sparse.values)
            add_tensor(sparse.indices)
        for node in current.node:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.TENSOR:
                    add_tensor(attribute.t)
                elif attribute.type == onnx.AttributeProto.TENSORS:
                    for tensor in attribute.tensors:
                        add_tensor(tensor)
                elif attribute.type == onnx.AttributeProto.SPARSE_TENSOR:
                    add_tensor(attribute.sparse_tensor.values)
                    add_tensor(attribute.sparse_tensor.indices)
                elif attribute.type == onnx.AttributeProto.SPARSE_TENSORS:
                    for sparse in attribute.sparse_tensors:
                        add_tensor(sparse.values)
                        add_tensor(sparse.indices)
                elif attribute.type == onnx.AttributeProto.GRAPH:
                    visit_graph(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for nested_graph in attribute.graphs:
                        visit_graph(nested_graph)

    visit_graph(graph)
    return locations


def _quantize_reader_int4(graph_path: Path, output_path: Path) -> None:
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    graph = onnx.load(str(graph_path), load_external_data=False)
    external_locations = _external_data_locations(graph.graph, onnx)

    quantizer = MatMulNBitsQuantizer(
        str(graph_path),
        block_size=128,
        is_symmetric=True,
        op_types_to_quantize=("MatMul", "Gather"),
    )
    quantizer.process()
    quantizer.model.save_model_to_file(str(output_path), True)

    quantized_graph = onnx.load(str(output_path), load_external_data=False)
    quantized_locations = _external_data_locations(quantized_graph.graph, onnx)
    operators = {(node.domain, node.op_type) for node in quantized_graph.graph.node}
    if not any(domain == "com.microsoft" and op == "MatMulNBits" for domain, op in operators):
        raise RuntimeError("int4 quantization did not convert any reader MatMul weights")

    graph_path.unlink()
    for location in external_locations - quantized_locations:
        (graph_path.parent / location).unlink(missing_ok=True)


def export_reader_to_onnx(
    backbone: str,
    output_dir: Path,
    opset: int = 18,
    quantize: Optional[str] = None,
    max_seq_len: int = 4096,
    feature_layers: Optional[list] = None,
    option_layers: Optional[list] = None,
) -> dict:
    """
    Export the reader (frozen LLM backbone) to ONNX.

    Returns metadata about the exported reader.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading reader backbone: " + backbone)
    config = AutoConfig.from_pretrained(backbone, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)

    # Load model with no gradients
    model = AutoModelForCausalLM.from_pretrained(
        backbone,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval()

    # Get hidden size and other config (handle nested text_config for Qwen3.5)
    if hasattr(config, 'text_config') and config.text_config:
        text_config = config.text_config
        hidden_size = text_config.hidden_size
        num_layers = text_config.num_hidden_layers
        vocab_size = text_config.vocab_size
        model_type = text_config.model_type
    else:
        hidden_size = config.hidden_size
        num_layers = config.num_hidden_layers
        vocab_size = config.vocab_size
        model_type = config.model_type
    _prepare_qwen3_5_onnx_export(model, model_type)
    is_qwen3_5 = model_type.startswith("qwen3_5")
    if is_qwen3_5 and max_seq_len != 128:
        raise ValueError("Qwen3.5's current WASM export is fixed at 128 tokens; set --max-seq-len 128")
    if is_qwen3_5 and quantize not in (None, "int4"):
        raise ValueError("Qwen3.5 WASM export supports --quantize int4, not dynamic int8/uint8")

    # Export tokenizer
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)

    # Export tokenizer config for WASM
    tokenizer_config = {
        "model_type": config.model_type,
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len,
        "padding_side": tokenizer.padding_side,
        "truncation_side": tokenizer.truncation_side,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "unk_token_id": tokenizer.unk_token_id,
    }

    # Export to ONNX
    print("Exporting reader to ONNX (opset " + str(opset) + ")...")

    # Create dummy input for tracing
    dummy_input_ids = torch.ones((1, max_seq_len), dtype=torch.int64)
    dummy_input_mask = torch.ones((1, max_seq_len), dtype=torch.int64)

    # We need to export the model's forward pass that returns hidden states
    # The vegaml reader uses the last hidden states from specific layers

    taps = sorted(set((feature_layers or []) + (option_layers or [])))
    if not taps:
        taps = list(range(num_layers))
    if any(not isinstance(layer, int) or layer < 0 or layer >= num_layers for layer in taps):
        raise ValueError("feature and option layers must be valid zero-based reader layer indices")

    class ReaderWrapper(torch.nn.Module):
        def __init__(self, model, layer_indices):
            super().__init__()
            self.model = model
            self.layer_indices = layer_indices

        def forward(self, input_ids, attention_mask):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            return tuple(outputs.hidden_states[i + 1] for i in self.layer_indices)

    wrapper = ReaderWrapper(model, taps).eval()

    input_names = ["input_ids", "attention_mask"]
    output_names = ["layer_" + str(i) for i in taps]
    if is_qwen3_5:
        dynamic_axes = {name: {0: "batch"} for name in input_names + output_names}
        export_args = {
            "dynamic_axes": dynamic_axes,
            "dynamo": False,
            "do_constant_folding": True,
        }
    else:
        batch_dim = torch.export.Dim("batch", min=1, max=16)
        sequence_dim = torch.export.Dim("sequence", min=1, max=max_seq_len)
        export_args = {
            "dynamic_shapes": (
                {0: batch_dim, 1: sequence_dim},
                {0: batch_dim, 1: sequence_dim},
            ),
            "dynamo": True,
            "optimize": False,
        }

    export_context = _qwen3_5_triangular_solve_patch() if is_qwen3_5 else nullcontext()
    with torch.no_grad(), export_context:
        torch.onnx.export(
            wrapper,
            (dummy_input_ids, dummy_input_mask),
            str(output_dir / "reader.onnx"),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            **export_args,
        )

    # Quantize if requested
    if quantize == "int4":
        _quantize_reader_int4(output_dir / "reader.onnx", output_dir / "reader.int4.onnx")
    elif quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quant_path = str(output_dir / "reader.int8.onnx")
        quantize_dynamic(
            str(output_dir / "reader.onnx"),
            quant_path,
            weight_type=QuantType.QInt8 if quantize == "int8" else QuantType.QUInt8,
            op_types_to_quantize=["MatMul"],
            use_external_data_format=True,
        )
        print("Quantized model saved to " + quant_path)

    # Save metadata
    metadata = {
        "model_type": "reader",
        "backbone": backbone,
        "model_type_config": config.model_type,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len,
        **({"fixed_sequence_length": 128} if is_qwen3_5 else {}),
        "bos_token_id": tokenizer.bos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "opset": opset,
        "quantized": quantize is not None,
        "quantization_type": quantize,
        "feature_layers": feature_layers or [],
        "option_layers": option_layers or [],
        "output_layers": taps,
    }

    with open(str(output_dir / "reader_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Save tokenizer config for WASM
    with open(str(output_dir / "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    # Copy tokenizer files
    print("Reader exported to " + str(output_dir))
    return metadata


def export_engine_to_onnx(
    checkpoint_dir: Path,
    output_dir: Path,
    opset: int = 18,
) -> dict:
    """
    Export the vegaml engine (14M physics engine) to ONNX.
    Uses the existing export_engine function.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    engine, cfg = _load_engine(str(checkpoint_dir))
    if cfg.get("port"):
        engine = UniversalEngine(str(checkpoint_dir), cfg, engine)
    engine_path = str(output_dir / "engine.onnx")
    export_engine_internal(engine, engine_path, opset=opset)

    metadata = {
        "model_type": "engine",
        "d_model": cfg.get("d_model"),
        "feature_layers": cfg.get("feature_layers"),
        "option_layers": cfg.get("option_layers"),
        "config": cfg,
    }

    with open(str(output_dir / "engine_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("Engine exported to " + str(engine_path))
    return metadata


def create_wasm_bundle(
    output_dir: Path,
    reader_metadata: dict,
    engine_metadata: dict,
    checkpoint_name: str,
) -> None:
    """
    Create the browser-ready bundle with:
    - reader.onnx (or reader.int8.onnx)
    - engine.onnx
    - tokenizer files
    - metadata.json for the browser loader
    - JavaScript loader module
    """

    quantization = reader_metadata.get("quantization_type")
    reader_name = (
        "reader.int4.onnx" if quantization == "int4"
        else "reader.int8.onnx" if reader_metadata.get("quantized")
        else "reader.onnx"
    )
    variants = {
        "reader": _build_variant(output_dir, reader_name, "vegaml reader"),
        "engine": _build_variant(output_dir, "engine.onnx", "vega-U decision engine"),
    }
    manifest = {
        "version": 1,
        "source": "vegaml",
        "chunk_bytes": WASM_CHUNK_BYTES,
        "variants": variants,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Create the main metadata file for the browser
    bundle_metadata = {
        "version": "1.0",
        "checkpoint": checkpoint_name,
        "reader": reader_metadata,
        "engine": engine_metadata,
        "pipeline": {
            "steps": [
                {"name": "tokenize", "input": "text", "output": "tokens"},
                {"name": "reader", "input": "tokens", "output": "hidden_states"},
                {"name": "feature_extraction", "input": "hidden_states", "output": "features"},
                {"name": "engine", "input": "features", "output": "logits"},
                {"name": "calibrated_probs", "input": "logits", "output": "probabilities"},
            ]
        },
        "files": {
            "reader": reader_name,
            "engine": "engine.onnx",
            "tokenizer": "tokenizer/",
        },
    }

    with open(str(output_dir / "metadata.json"), "w") as f:
        json.dump(bundle_metadata, f, indent=2)

    # Create the JavaScript loader module
    (output_dir / "vegaml-wasm.js").write_text(generate_js_loader(bundle_metadata))

    # Create a simple HTML demo
    html_demo = generate_html_demo(bundle_metadata)
    (output_dir / "demo.html").write_text(html_demo)

    print("WASM bundle created in " + str(output_dir))


def _asset_entry(path: Path, output_dir: Path) -> dict:
    size = path.stat().st_size
    digest = hashlib.sha256()
    parts = []
    if size <= WASM_MAX_FILE_BYTES:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(WASM_CHUNK_BYTES), b""):
                digest.update(block)
        return {"name": path.name, "size": size, "sha256": digest.hexdigest()}

    chunk_dir = output_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    with path.open("rb") as source:
        index = 0
        while block := source.read(WASM_CHUNK_BYTES):
            digest.update(block)
            encoded = bytes(value ^ 0xFF for value in block)
            part = chunk_dir / (path.name + ".part.%03d" % index)
            part.write_bytes(encoded)
            parts.append("chunks/" + part.name)
            index += 1

    path.unlink()
    return {"name": path.name, "size": size, "sha256": digest.hexdigest(), "parts": parts}


def _build_variant(output_dir: Path, filename: str, label: str) -> dict:
    graph = output_dir / filename
    if not graph.is_file():
        raise FileNotFoundError("exported ONNX graph is missing: " + str(graph))

    import onnx

    model = onnx.load(str(graph), load_external_data=False)
    locations = {
        item.value
        for initializer in model.graph.initializer
        for item in initializer.external_data
        if item.key == "location"
    }
    if len(locations) > 1:
        raise ValueError("WASM export supports one external-data file per ONNX graph")

    variant = {"label": label, "onnx": filename}
    if graph.stat().st_size > WASM_MAX_FILE_BYTES:
        entry = _asset_entry(graph, output_dir)
        variant["onnx_size"] = entry["size"]
        variant["onnx_sha256"] = entry["sha256"]
        variant["onnx_parts"] = entry["parts"]
    else:
        entry = _asset_entry(graph, output_dir)
        variant["onnx_size"] = entry["size"]
        variant["onnx_sha256"] = entry["sha256"]

    if locations:
        location = next(iter(locations))
        external_data = output_dir / location
        if not external_data.is_file():
            raise FileNotFoundError("ONNX external-data file is missing: " + str(external_data))
        variant["data"] = _asset_entry(external_data, output_dir)
    return variant


def generate_js_loader(metadata: dict) -> str:
    """Generate the JavaScript loader module for onnxruntime-web."""
    return Path(__file__).with_name("wasm_loader.js").read_text(encoding="utf-8")



def generate_html_demo(metadata: dict) -> str:
    """Generate a simple HTML demo page."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>vegaml WASM Demo</title>
    <style>
        body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
        textarea { width: 100%; height: 150px; font-family: monospace; margin: 0.5rem 0; }
        button { padding: 0.5rem 1rem; font-size: 1rem; cursor: pointer; }
        .result { background: #f5f5f5; padding: 1rem; margin: 1rem 0; border-radius: 4px; }
        .violation { color: #d00; } .ok { color: #080; }
        pre { background: #fafafa; padding: 1rem; overflow: auto; }
        .status { padding: 0.5rem; border-radius: 4px; margin: 0.5rem 0; }
        .loading { background: #fff3cd; } .ready { background: #d4edda; } .error { background: #f8d7da; }
    </style>
</head>
<body>
    <h1>vegaml WASM Demo</h1>
    <p>Running entirely in browser via ONNX Runtime Web + WASM</p>

    <div id="status" class="status loading">Loading model... (this may take a while on first load)</div>

    <div>
        <label>HTML to analyze:</label>
        <textarea id="htmlInput" placeholder='<img src="chart.png">'></textarea>
    </div>

    <div style="margin: 1rem 0;">
        <label>Threshold:
            <input type="range" id="threshold" min="0" max="1" step="0.05" value="0.5">
            <span id="thVal">0.5</span>
        </label>
    </div>

    <button onclick="analyze()">Analyze (63 WCAG criteria)</button>

    <div id="results"></div>

    <script type="module">
        import { VegamlWASM } from './vegaml-wasm.js';

        let vegaml = null;

        async function init() {
            var status = document.getElementById('status');
            try {
                status.textContent = 'Loading WASM model... (first load downloads model)';
                vegaml = await VegamlWASM.create({ modelDir: './' });
                window.vegaml = vegaml;
                status.textContent = 'Model loaded! Ready to analyze.';
                status.className = 'status ready';
            } catch (e) {
                status.textContent = 'Error: ' + e.message;
                status.className = 'status error';
                console.error(e);
            }
        }

        async function analyze() {
            var html = document.getElementById('htmlInput').value;
            var threshold = parseFloat(document.getElementById('threshold').value);
            var resultsDiv = document.getElementById('results');

            if (!window.vegaml) {
                alert('Model not loaded yet');
                return;
            }

            resultsDiv.innerHTML = '<div class="status loading">Analyzing...</div>';

            try {
                var results = await window.vegaml.decide(html, buildQuestions());
                renderResults(results);
            } catch (e) {
                document.getElementById('results').innerHTML =
                    '<div class="status error">Error: ' + e.message + '</div>';
            }
        }

        function buildQuestions() {
            // Build the 63 WCAG criteria questions
            var q = {};
            for (var i = 0; i < criteriaList.length; i++) {
                var c = criteriaList[i];
                q[c.criterion] = {
                    type: 'boolean',
                    instructions: 'Does this HTML fail WCAG ' + c.criterion + ' (' + c.description + ')?'
                };
            }
            return q;
        }

        function renderResults(results) {
            var div = document.getElementById('results');
            var answers = results.answers || {};
            var html = '<h3>Results</h3><table style="width:100%; border-collapse: collapse;">';
            html += '<tr><th>Criterion</th><th>Description</th><th>p(violated)</th><th>Verdict</th></tr>';

            for (var key in answers) {
                var ans = answers[key];
                var p = ans.boolean || 0;
                var isViolation = p >= parseFloat(document.getElementById('threshold').value);
                html += '<tr style="border-bottom: 1px solid #ddd;"><td style="padding: 0.5rem; ' +
                    'font-family: monospace;">' +
                    key + '</td>' +
                    '<td style="padding: 0.5rem;"></td>' +
                    '<td style="padding: 0.5rem; text-align: right;">' + (p * 100).toFixed(1) + '%</td>' +
                    '<td style="padding: 0.5rem;" class="' + (isViolation ? 'violation' : 'ok') + '">' +
                    (isViolation ? '\\u26a0 VIOLATION' : '\\u2713 OK') + '</td></tr>';
            }
            html += '</table>';
            document.getElementById('results').innerHTML = html;
        }

        document.getElementById('threshold').addEventListener('input', function(e) {
            document.getElementById('thVal').textContent = e.target.value;
        });

        init();
    </script>
</body>
</html>
"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Export vegaml pipeline to ONNX for WASM deployment")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint name (e.g., '0.8b', '4b') or path")
    parser.add_argument("--output-dir", required=True, help="Output directory for WASM bundle")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    parser.add_argument("--quantize", choices=["int8", "uint8", "int4"], help="Quantize reader model")
    parser.add_argument(
        "--max-seq-len",
        type=int,
        help="Max sequence length (defaults to checkpoint max_len)",
    )
    parser.add_argument("--skip-reader", action="store_true", help="Skip reader export (engine only)")
    parser.add_argument("--skip-engine", action="store_true", help="Skip engine export")

    args = parser.parse_args()
    if args.max_seq_len is not None and args.max_seq_len < 1:
        parser.error("--max-seq-len must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_dir():
        checkpoint_path = Path(fetch(args.checkpoint))
    config_path = checkpoint_path / "vega_config.json"
    if not config_path.is_file():
        parser.error("--checkpoint must resolve to a vegaml checkpoint containing vega_config.json")
    with config_path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    backbone = backbone_hub_id(cfg.get("backbone_name") or cfg.get("backbone"))
    if not backbone:
        parser.error("checkpoint does not declare its trained reader in vega_config.json")
    checkpoint_name = checkpoint_path.name

    # Export reader
    reader_metadata_path = output_dir / "reader_metadata.json"
    if args.skip_reader:
        if not reader_metadata_path.is_file():
            parser.error("--skip-reader requires reader_metadata.json in --output-dir")
        with reader_metadata_path.open(encoding="utf-8") as f:
            reader_metadata = json.load(f)
    else:
        reader_metadata = export_reader_to_onnx(
            backbone=backbone,
            output_dir=Path(args.output_dir),
            opset=args.opset,
            quantize=args.quantize,
            max_seq_len=args.max_seq_len or cfg.get("max_len", 4096),
            feature_layers=cfg.get("feature_layers"),
            option_layers=cfg.get("option_layers"),
        )

    # Export engine
    engine_metadata_path = output_dir / "engine_metadata.json"
    if args.skip_engine:
        if not engine_metadata_path.is_file():
            parser.error("--skip-engine requires engine_metadata.json in --output-dir")
        with engine_metadata_path.open(encoding="utf-8") as f:
            engine_metadata = json.load(f)
    else:
        engine_metadata = export_engine_to_onnx(
            checkpoint_dir=checkpoint_path,
            output_dir=Path(args.output_dir),
        )

    engine_config = engine_metadata.get("config")
    reader_max_seq_len = reader_metadata.get("max_seq_len")
    if (
        not isinstance(engine_config, dict)
        or not isinstance(reader_max_seq_len, int)
        or reader_max_seq_len < 1
    ):
        parser.error("reader and engine metadata must include their current configuration")
    engine_config["max_len"] = reader_max_seq_len

    # Create WASM bundle
    create_wasm_bundle(
        output_dir=Path(args.output_dir),
        reader_metadata=reader_metadata,
        engine_metadata=engine_metadata,
        checkpoint_name=checkpoint_name,
    )

    print("")
    print(" WASM bundle created in " + str(args.output_dir))
    print("   Open " + str(args.output_dir) + "/demo.html in a browser to test")
    print("   Or import { VegamlWASM } from './vegaml-wasm.js' in your app")


if __name__ == "__main__":
    main()
