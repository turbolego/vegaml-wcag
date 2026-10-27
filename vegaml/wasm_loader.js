import * as ort from "../vendor/ort.min.mjs";
import { Tokenizer } from "../vendor/tokenizers.min.mjs";

const readerBatchSize = 4;

async function fetchResponse(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response;
}

async function fetchJSON(url) {
  return (await fetchResponse(url)).json();
}

function sameReader(reader, cfg) {
  const declared = (cfg.backbone_name || cfg.backbone || "").split("/").pop().toLowerCase();
  const exported = (reader.backbone || "").split("/").pop().toLowerCase();
  return declared && exported && declared === exported;
}

function encode(tokenizer, text) {
  return Array.from(tokenizer.encode(text, { add_special_tokens: false }).ids);
}

function buildPrompt(tokenizer, state, question, cfg, reader) {
  const maxLen = cfg.max_len || reader.max_seq_len || 4096;
  const bos = Number.isInteger(reader.bos_token_id) ? [reader.bos_token_id] : [];
  const head = encode(tokenizer, "Observation:\n");
  const qids = encode(tokenizer, `\n\nMeasurement (boolean): ${question.instructions}\nPossible outcomes:`)
    .slice(0, 256);
  const options = [
    encode(tokenizer, "\n* no - the statement does not hold").slice(0, cfg.opt_max_tokens || 32),
    encode(tokenizer, "\n* yes - the statement holds").slice(0, cfg.opt_max_tokens || 32),
  ];
  const end = encode(tokenizer, "\nOutcome:");
  let fixed = bos.length + head.length + qids.length + options[0].length + options[1].length + end.length;
  if (fixed > maxLen - 16) {
    const per = Math.max(3, Math.floor((maxLen - 16 - bos.length - head.length - end.length -
      Math.min(qids.length, 64)) / options.length));
    options[0] = options[0].slice(0, per);
    options[1] = options[1].slice(0, per);
    qids.splice(Math.max(8, maxLen - 16 - bos.length - head.length - end.length -
      options[0].length - options[1].length));
    fixed = bos.length + head.length + qids.length + options[0].length + options[1].length + end.length;
  }

  let stateIds = encode(tokenizer, state);
  const prefixMax = cfg.query_first_max_state || 0;
  let prefix = [];
  if (prefixMax && stateIds.length < prefixMax) {
    prefix = encode(tokenizer, `Question to keep in mind while reading (boolean): ${question.instructions}\n\n`)
      .slice(0, Math.max(0, Math.min(128, maxLen - fixed)));
    fixed += prefix.length;
  }
  const room = Math.max(0, maxLen - fixed);
  stateIds = stateIds.slice(0, room);

  const ids = [...bos, ...prefix, ...head];
  const stateStart = ids.length;
  ids.push(...stateIds);
  const spans = { state: stateIds.length ? [stateStart, ids.length] : [bos.length, stateStart] };
  const questionStart = ids.length;
  ids.push(...qids);
  spans.question = [questionStart, ids.length];
  spans.options = [];
  for (const option of options) {
    const start = ids.length;
    ids.push(...option);
    spans.options.push([start, ids.length]);
  }
  ids.push(...end);
  spans.last = ids.length - 1;
  return { ids, spans };
}

function meanSpan(data, batchIndex, sequenceLength, hiddenSize, start, end) {
  const result = new Float32Array(hiddenSize);
  const count = Math.max(1, end - start);
  const base = batchIndex * sequenceLength * hiddenSize;
  for (let token = start; token < end; token++) {
    const offset = base + token * hiddenSize;
    for (let dim = 0; dim < hiddenSize; dim++) result[dim] += data[offset + dim];
  }
  for (let dim = 0; dim < hiddenSize; dim++) result[dim] /= count;
  return result;
}

function setVector(target, offset, vector) {
  target.set(vector, offset);
}

function probabilityPair(logits, offset, tau) {
  const no = logits[offset] / tau;
  const yes = logits[offset + 1] / tau;
  const max = Math.max(no, yes);
  const pNo = Math.exp(no - max);
  const pYes = Math.exp(yes - max);
  const sum = pNo + pYes;
  return [pNo / sum, pYes / sum];
}

export class VegamlWASM {
  static async create(options = {}) {
    const instance = new VegamlWASM(options);
    await instance.initialize();
    return instance;
  }

  constructor({ modelDir = "./wasm-model/", onProgress = () => {} } = {}) {
    this.modelDir = new URL(modelDir, location.href);
    this.onProgress = onProgress;
    this.cacheHits = 0;
    this.cacheWriteFailed = false;
  }

  reportProgress(progress, callback = this.onProgress) {
    try {
      callback(progress);
    } catch {
    }
  }

  async fetchBytes(assetUrl, { label, expectedBytes, onProgress } = {}) {
    let response;
    let cachedAsset = false;
    if (this.assetCache) {
      try {
        response = await this.assetCache.match(assetUrl);
        if (response) {
          cachedAsset = true;
          this.cacheHits++;
        }
      } catch {
        response = null;
      }
    }

    if (!response) response = await fetchResponse(assetUrl);
    const responseBytes = Number(response.headers.get("content-length")) || expectedBytes || 0;
    const chunks = [];
    let loaded = 0;
    if (response.body?.getReader) {
      const reader = response.body.getReader();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.length;
        this.reportProgress({ stage: "download", label, loaded, total: responseBytes, cached: cachedAsset }, onProgress);
      }
    } else {
      const bytes = new Uint8Array(await response.arrayBuffer());
      chunks.push(bytes);
      loaded = bytes.length;
      this.reportProgress({ stage: "download", label, loaded, total: responseBytes, cached: cachedAsset }, onProgress);
    }

    const bytes = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.length;
    }
    if (this.assetCache && !cachedAsset) {
      try {
        const headers = new Headers(response.headers);
        headers.set("content-length", String(bytes.length));
        await this.assetCache.put(assetUrl, new Response(bytes, { headers }));
      } catch {
        this.cacheWriteFailed = true;
      }
    }
    return bytes;
  }

  async initialize() {
    const url = (path) => new URL(path, this.modelDir).href;
    this.reportProgress({ stage: "status", message: "Loading model metadata and tokenizer..." });
    const [metadata, manifest, tokenizerJson, tokenizerConfig] = await Promise.all([
      fetchJSON(url("metadata.json")),
      fetchJSON(url("manifest.json")),
      fetchJSON(url("tokenizer/tokenizer.json")),
      fetchJSON(url("tokenizer/tokenizer_config.json")),
    ]);
    const cfg = metadata.engine?.config;
    if (!cfg || !metadata.reader?.backbone || !sameReader(metadata.reader, cfg)) {
      throw new Error("WASM bundle reader and engine do not match; export a bundle from the same vegaml checkpoint");
    }
    if (!Array.isArray(cfg.feature_layers) || !Array.isArray(cfg.option_layers) ||
        !Array.isArray(metadata.reader.output_layers)) {
      throw new Error("WASM bundle is missing reader layer metadata; export it again with the current vegaml exporter");
    }
    if (JSON.stringify(cfg.feature_layers) !== JSON.stringify(metadata.reader.feature_layers) ||
        JSON.stringify(cfg.option_layers) !== JSON.stringify(metadata.reader.option_layers)) {
      throw new Error("WASM reader layer taps do not match the trained vegaml checkpoint");
    }
    const fixedSequenceLength = metadata.reader.fixed_sequence_length;
    if (fixedSequenceLength !== undefined &&
        (!Number.isInteger(fixedSequenceLength) || fixedSequenceLength < 1 ||
          fixedSequenceLength > metadata.reader.max_seq_len)) {
      throw new Error("WASM reader has an invalid fixed sequence length");
    }

    this.metadata = metadata;
    this.config = cfg;
    this.tokenizer = new Tokenizer(tokenizerJson, tokenizerConfig);
    ort.env.wasm.wasmPaths = new URL("../vendor/", import.meta.url).href;
    ort.env.wasm.numThreads = self.crossOriginIsolated
      ? Math.min(4, navigator.hardwareConcurrency || 2)
      : 1;

    const readerVariant = manifest.variants?.reader;
    const engineVariant = manifest.variants?.engine;
    if (!readerVariant || !engineVariant) throw new Error("WASM manifest must include reader and engine models");
    const bundleFingerprint = Object.values(manifest.variants)
      .flatMap((variant) => [variant.onnx_sha256 || variant.sha256, variant.data?.sha256])
      .filter(Boolean)
      .map((hash) => hash.slice(0, 12))
      .join("-");
    this.cacheName = `vegaml-wcag-wasm-v${manifest.version}-${bundleFingerprint}`;
    if ("caches" in globalThis) {
      try {
        this.assetCache = await caches.open(this.cacheName);
      } catch {
        this.assetCache = null;
      }
    }
    const [readerGraph, readerData, engineGraph, engineData] = await Promise.all([
      this.loadGraph(readerVariant, url),
      this.loadExternalData(readerVariant, url),
      this.loadGraph(engineVariant, url),
      this.loadExternalData(engineVariant, url),
    ]);
    this.reportProgress({ stage: "status", message: "Creating reader inference session..." });
    this.readerSession = await this.createSession(readerGraph, readerData);
    this.reportProgress({ stage: "status", message: "Creating decision-engine session..." });
    this.engineSession = await this.createSession(engineGraph, engineData);
    this.initialized = true;
    if ("caches" in globalThis && this.cacheName) {
      try {
        const oldCaches = (await caches.keys()).filter((name) =>
          name.startsWith("vegaml-wcag-wasm-v") && name !== this.cacheName);
        await Promise.all(oldCaches.map((name) => caches.delete(name)));
      } catch {
      }
    }
    this.reportProgress({
      stage: "ready",
      threads: ort.env.wasm.numThreads,
      cachedAssets: this.cacheHits,
      cacheAvailable: Boolean(this.assetCache) && !this.cacheWriteFailed,
      cacheWriteFailed: this.cacheWriteFailed,
    });
  }

  async loadAsset(entry, url, label = entry?.name) {
    if (!entry) return null;
    if (!entry.parts?.length) {
      return this.fetchBytes(url(entry.name), { label, expectedBytes: entry.size });
    }
    const result = new Uint8Array(entry.size);
    let offset = 0;
    for (const part of entry.parts) {
      const bytes = await this.fetchBytes(url(part), {
        label,
        onProgress: (progress) => this.reportProgress({
          ...progress,
          loaded: offset + progress.loaded,
          total: entry.size,
        }),
      });
      for (let index = 0; index < bytes.length; index++) bytes[index] ^= 0xFF;
      if (offset + bytes.length > result.length) throw new Error(`Chunked asset exceeds declared size: ${entry.name}`);
      result.set(bytes, offset);
      offset += bytes.length;
    }
    if (offset !== result.length) {
      throw new Error(`Chunked asset is incomplete: ${entry.name} (${offset} of ${result.length} bytes)`);
    }
    return result;
  }

  async loadGraph(variant, url) {
    if (variant.onnx_parts?.length) {
      return this.loadAsset({
        name: variant.onnx,
        size: variant.onnx_size,
        parts: variant.onnx_parts,
      }, url, `${variant.label || variant.onnx} ONNX graph`);
    }
    return this.fetchBytes(url(variant.onnx), {
      label: `${variant.label || variant.onnx} ONNX graph`,
      expectedBytes: variant.onnx_size,
    });
  }

  async loadExternalData(variant, url) {
    if (!variant.data) return [];
    const bytes = await this.loadAsset(variant.data, url, `${variant.label || variant.data.name} weights`);
    return [{ path: variant.data.name, data: bytes }];
  }

  createSession(graph, externalData) {
    return ort.InferenceSession.create(graph, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
      externalData,
    });
  }

  async readerFeatures(items, onProgress = this.onProgress) {
    const { feature_layers: featureLayers, option_layers: optionLayers } = this.config;
    const outputLayers = this.metadata.reader.output_layers;
    const hiddenSize = this.metadata.reader.hidden_size;
    const features = new Float32Array(items.length * featureLayers.length * 3 * hiddenSize);
    const options = new Float32Array(items.length * 2 * optionLayers.length * hiddenSize);
    const padId = this.metadata.reader.pad_token_id ?? 0;
    const chunkCount = this.config.memory_chunks || 0;
    const chunks = new Float32Array(items.length * chunkCount * hiddenSize);
    const chunkMask = new Uint8Array(items.length * chunkCount);

    for (let begin = 0; begin < items.length; begin += readerBatchSize) {
      const batch = items.slice(begin, begin + readerBatchSize);
      this.reportProgress({ stage: "inference", completed: begin, total: items.length }, onProgress);
      const length = this.metadata.reader.fixed_sequence_length ||
        Math.max(...batch.map((item) => item.ids.length));
      if (batch.some((item) => item.ids.length > length)) {
        throw new Error(`Reader prompt exceeds its fixed input length of ${length} tokens`);
      }
      const inputIds = new BigInt64Array(batch.length * length).fill(BigInt(padId));
      const attention = new BigInt64Array(batch.length * length);
      batch.forEach((item, row) => item.ids.forEach((id, index) => {
        inputIds[row * length + index] = BigInt(id);
        attention[row * length + index] = 1n;
      }));
      const outputs = await this.readerSession.run({
        input_ids: new ort.Tensor("int64", inputIds, [batch.length, length]),
        attention_mask: new ort.Tensor("int64", attention, [batch.length, length]),
      });
      this.reportProgress({ stage: "inference", completed: begin + batch.length, total: items.length }, onProgress);

      batch.forEach((item, row) => {
        const targetRow = begin + row;
        featureLayers.forEach((layer, layerIndex) => {
          const data = outputs[`layer_${layer}`]?.data;
          if (!data) throw new Error(`Reader ONNX output missing feature layer ${layer}`);
          const base = ((targetRow * featureLayers.length + layerIndex) * 3) * hiddenSize;
          setVector(features, base, meanSpan(data, row, length, hiddenSize, ...item.spans.state));
          setVector(features, base + hiddenSize, meanSpan(data, row, length, hiddenSize, ...item.spans.question));
          setVector(features, base + 2 * hiddenSize, meanSpan(data, row, length, hiddenSize, item.spans.last, item.spans.last + 1));
        });
        if (chunkCount) {
          const layer = featureLayers[featureLayers.length - 1];
          const data = outputs[`layer_${layer}`]?.data;
          const [stateStart, stateEnd] = item.spans.state;
          const stateLength = stateEnd - stateStart;
          const chunkMin = this.config.memory_min_tokens || 2048;
          if (stateLength >= Math.max(chunkMin, 2)) {
            const perChunk = this.config.memory_chunk_tokens || 128;
            const used = Math.max(2, Math.min(chunkCount, Math.floor(stateLength / perChunk)));
            const counts = new Uint32Array(used);
            for (let token = 0; token < stateLength; token++) {
              const chunk = Math.floor(token * used / stateLength);
              const source = (row * length + stateStart + token) * hiddenSize;
              const target = ((targetRow * chunkCount + chunk) * hiddenSize);
              for (let dim = 0; dim < hiddenSize; dim++) chunks[target + dim] += data[source + dim];
              counts[chunk]++;
            }
            for (let chunk = 0; chunk < used; chunk++) {
              const target = (targetRow * chunkCount + chunk) * hiddenSize;
              for (let dim = 0; dim < hiddenSize; dim++) chunks[target + dim] /= counts[chunk];
              chunkMask[targetRow * chunkCount + chunk] = 1;
            }
          }
        }
        optionLayers.forEach((layer, layerIndex) => {
          const data = outputs[`layer_${layer}`]?.data;
          if (!data) throw new Error(`Reader ONNX output missing option layer ${layer}`);
          item.spans.options.forEach((span, optionIndex) => {
            const base = (((targetRow * 2 + optionIndex) * optionLayers.length + layerIndex) * hiddenSize);
            setVector(options, base, meanSpan(data, row, length, hiddenSize, ...span));
          });
        });
      });
    }

    return { features, options, chunks, chunkMask };
  }

  async decide(state, questions, onProgress = this.onProgress) {
    if (!this.initialized) throw new Error("VegamlWASM not initialized");
    const start = performance.now();
    const entries = Object.entries(questions);
    if (entries.some(([, question]) => question.type !== "boolean")) {
      throw new Error("The WCAG WASM demo currently supports boolean questions only");
    }
    const text = typeof state === "string" ? state : JSON.stringify(state);
    const prompts = entries.map(([, question]) => buildPrompt(this.tokenizer, text, question, this.config, this.metadata.reader));
    const { features, options, chunks, chunkMask } = await this.readerFeatures(prompts, onProgress);
    const count = entries.length;
    const hiddenSize = this.metadata.reader.hidden_size;
    const optionLayers = this.config.option_layers.length;
    const inputs = {
      feats: new ort.Tensor("float32", features, [count, this.config.feature_layers.length, 3, hiddenSize]),
      opts: new ort.Tensor("float32", options, [count, 2, optionLayers, hiddenSize]),
      opt_mask: new ort.Tensor("bool", new Uint8Array(count * 2).fill(1), [count, 2]),
      qtype: new ort.Tensor("int64", new BigInt64Array(count).fill(2n), [count]),
    };
    if (this.engineSession.inputNames.includes("chunks")) {
      const chunkCount = this.config.memory_chunks;
      if (!chunkCount) throw new Error("Engine ONNX requires memory chunks but the checkpoint config omits memory_chunks");
      inputs.chunks = new ort.Tensor("float32", chunks, [count, chunkCount, hiddenSize]);
      inputs.chunk_mask = new ort.Tensor("bool", chunkMask, [count, chunkCount]);
    }

    this.reportProgress({ stage: "status", message: "Scoring criteria with the decision engine..." }, onProgress);
    const output = await this.engineSession.run(inputs);
    const calibration = this.config.calibration || {};
    const theta = calibration.theta?.[2] || [0, 0, 0, 0, 0];
    const risk = String(calibration.default_risk || "0.1");
    const abstainThreshold = calibration.abstain_threshold?.[risk] || 0;
    const conformal = calibration.conformal?.boolean || calibration.conformal?.noul || {};
    const conformalThreshold = conformal["0.1"];
    const maxSteps = Math.max(1, this.config.n_steps || 1);
    const answers = {};

    entries.forEach(([criterion], index) => {
      const kinetic = Math.max(0, Number(output.kinetic.data[index]));
      const nearest = Math.max(0, Number(output.nearest.data[index]));
      const steps = Number(output.steps_used.data[index]);
      const phys = [Math.log1p(kinetic), Math.log1p(nearest), steps / maxSteps, Math.log(2)];
      const logTau = theta[0] + theta.slice(1).reduce((sum, weight, i) => sum + weight * phys[i], 0);
      const tau = Math.exp(Math.max(Math.log(0.2), Math.min(Math.log(20), logTau)));
      const probs = probabilityPair(output.logits.data, index * 2, tau);
      const confidence = Math.max(...probs);
      const best = probs[1] > probs[0] ? 1 : 0;
      const predictionSet = conformalThreshold === undefined
        ? null
        : [0, 1].filter((option) => probs[option] >= Math.max(1 - conformalThreshold, 1e-12) || option === best)
          .map((option) => option === 1);
      answers[criterion] = {
        boolean: probs[1],
        probs: { false: probs[0], true: probs[1] },
        confidence,
        prediction_set: predictionSet,
        abstain: confidence < abstainThreshold,
        unbound: nearest > (calibration.unbound_threshold ?? Number.POSITIVE_INFINITY),
        settle_steps: steps,
        energy: -Math.max(...output.logits.data.slice(index * 2, index * 2 + 2)),
      };
    });

    return {
      answers,
      latency_ms: Math.round(performance.now() - start),
      n_tokens: prompts.reduce((total, prompt) => total + prompt.ids.length, 0),
      mode: "wasm",
    };
  }
}
