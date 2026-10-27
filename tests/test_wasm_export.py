import hashlib
import json
from pathlib import Path

import onnx
import torch
from onnx import helper

from vegaml import wasm_export


def _write_graph(path):
    value = helper.make_tensor_value_info("value", onnx.TensorProto.FLOAT, [1])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["value"], ["result"])],
        "wasm-test",
        [value],
        [helper.make_tensor_value_info("result", onnx.TensorProto.FLOAT, [1])],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), path)


def _write_quantizable_graph(path):
    features = helper.make_tensor_value_info("features", onnx.TensorProto.FLOAT, [1, 128])
    token_ids = helper.make_tensor_value_info("token_ids", onnx.TensorProto.INT64, [1, 2])
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["features", "projection"], ["projected"]),
            helper.make_node("Gather", ["embedding", "token_ids"], ["embedded"], axis=0),
        ],
        "int4-test",
        [features, token_ids],
        [
            helper.make_tensor_value_info("projected", onnx.TensorProto.FLOAT, [1, 16]),
            helper.make_tensor_value_info("embedded", onnx.TensorProto.FLOAT, [1, 2, 128]),
        ],
        initializer=[
            onnx.numpy_helper.from_array(torch.randn(128, 16).numpy(), "projection"),
            onnx.numpy_helper.from_array(torch.randn(256, 128).numpy(), "embedding"),
        ],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), path)


def test_qwen_triangular_solve_patch_matches_torch_and_restores_globals():
    size = 64
    matrix = torch.tril(torch.randn(2, 3, size, size) * 0.1, diagonal=-1)
    matrix = matrix + torch.eye(size)
    rhs = torch.randn(2, 3, size, 4)
    expected = torch.linalg.solve_triangular(matrix, rhs, upper=False, unitriangular=True)
    original_solve = torch.linalg.solve_triangular

    with wasm_export._qwen3_5_triangular_solve_patch():
        actual = torch.linalg.solve_triangular(matrix, rhs, upper=False, unitriangular=True)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert torch.linalg.solve_triangular is original_solve


def test_external_data_locations_include_node_attribute_tensors():
    tensor = helper.make_tensor("value", onnx.TensorProto.FLOAT, [1], [1.0])
    tensor.data_location = onnx.TensorProto.EXTERNAL
    entry = tensor.external_data.add()
    entry.key = "location"
    entry.value = "constant.data"
    node = helper.make_node("Constant", [], ["value"], value=tensor)
    graph = helper.make_graph(
        [node],
        "external-attribute-test",
        [],
        [helper.make_tensor_value_info("value", onnx.TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph)

    assert wasm_export._external_data_locations(model.graph, onnx) == {"constant.data"}


def test_int4_quantization_converts_reader_matmul_and_embedding(tmp_path):
    source = tmp_path / "reader.onnx"
    output = tmp_path / "reader.int4.onnx"
    _write_quantizable_graph(source)

    wasm_export._quantize_reader_int4(source, output)

    model = onnx.load(output, load_external_data=False)
    operators = {(node.domain, node.op_type) for node in model.graph.node}
    external_locations = {
        item.value
        for initializer in model.graph.initializer
        for item in initializer.external_data
        if item.key == "location"
    }
    assert ("com.microsoft", "MatMulNBits") in operators
    assert ("com.microsoft", "GatherBlockQuantized") in operators
    assert len(external_locations) == 1
    assert not source.exists()


def test_asset_entry_chunks_large_files_and_records_original_digest(tmp_path, monkeypatch):
    monkeypatch.setattr(wasm_export, "WASM_CHUNK_BYTES", 4)
    monkeypatch.setattr(wasm_export, "WASM_MAX_FILE_BYTES", 8)
    source = tmp_path / "reader.data"
    contents = b"abcdefghij"
    source.write_bytes(contents)

    entry = wasm_export._asset_entry(source, tmp_path)

    assert entry["size"] == len(contents)
    assert entry["sha256"] == hashlib.sha256(contents).hexdigest()
    assert [len((tmp_path / part).read_bytes()) for part in entry["parts"]] == [4, 4, 2]
    assert b"".join((tmp_path / part).read_bytes() for part in entry["parts"]) == bytes(
        byte ^ 0xFF for byte in contents
    )
    assert not source.exists()


def test_create_wasm_bundle_uses_packaged_loader_and_writes_graph_manifest(tmp_path):
    _write_graph(tmp_path / "reader.onnx")
    _write_graph(tmp_path / "engine.onnx")

    wasm_export.create_wasm_bundle(
        tmp_path,
        {"backbone": "test-reader", "quantized": False},
        {"config": {"feature_layers": [0], "option_layers": [0]}},
        "test-checkpoint",
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert manifest["source"] == "vegaml"
    assert set(manifest["variants"]) == {"reader", "engine"}
    assert metadata["checkpoint"] == "test-checkpoint"
    assert (tmp_path / "vegaml-wasm.js").read_text() == wasm_export.generate_js_loader(metadata)
    assert (tmp_path / "vegaml-wasm.js").read_text() == (
        Path(wasm_export.__file__).with_name("wasm_loader.js").read_text(encoding="utf-8")
    )


def test_create_wasm_bundle_selects_int4_reader_graph(tmp_path):
    _write_graph(tmp_path / "reader.int4.onnx")
    _write_graph(tmp_path / "engine.onnx")

    wasm_export.create_wasm_bundle(
        tmp_path,
        {"backbone": "test-reader", "quantized": True, "quantization_type": "int4"},
        {"config": {"feature_layers": [0], "option_layers": [0]}},
        "test-checkpoint",
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["variants"]["reader"]["onnx"] == "reader.int4.onnx"
