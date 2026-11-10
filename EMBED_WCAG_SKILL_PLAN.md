# Plan: Embedding the WCAG Skill into Qwen3.5-0.8B via Weight Editing

## Overview
This document outlines the plan to embed the `wcag-skill` (from https://github.com/turbolego/wcag-skill) into the Qwen3.5-0.8B model (https://huggingface.co/Qwen/Qwen3.5-0.8B) using the weight-editing method described in the blog post: [A Decision Skill Written Into the Weights](https://www.embedding.in/blog/a-decision-skill-written-into-the-weights).

The technique uses **vegaml** as a teacher model fitted on ~300 examples, then writes the decision boundaries ("valleys") directly into the target LLM's weights — no training, ~3ms, no catastrophic forgetting. This took Qwen from 18% to 98% accuracy on tool-use decisions.

Reference implementation: Kaggle notebook [fast-edits-decision-model](https://www.kaggle.com/code/hummern/fast-edits-decsion-model/)

---

## Prerequisites

### 1. Source Repositories
| Repo | Purpose |
|------|---------|
| `turbolego/wcag-skill` | The skill to embed (WCAG 2.2 violation detection/fixing) |
| `turbolego/vegaml` | Decision model framework (teacher) |
| `turbolego/vegaml-wcag` | This repo — WCAG demo + vegaml integration |

### 2. Target Model
- **Model**: Qwen3.5-0.8B (`Qwen/Qwen3.5-0.8B` on Hugging Face)
- **Format**: Hugging Face `transformers` compatible (PyTorch)
- **Quantization**: Can work with FP16/BF16 or quantized (GGUF/INT8) — weight edits apply to the loaded tensors

### 3. Compute
- **Kaggle GPU**: T4 (16GB) or P100 (16GB) — sufficient for 0.8B model
- **Alternative**: Any machine with 8GB+ VRAM

---

## Step-by-Step Plan

### Phase 1: Prepare the WCAG Skill Dataset (~300 Examples)

1. **Extract the skill specification** from `wcag-skill`
   - The skill defines: detection, fixing, and prevention of WCAG 2.2 violations
   - Includes: automated validators, violation diagnosis, repair of `index.html`/`style.css`, verification, accessible markup generation

2. **Generate training examples** (target: 300+)
   - Format: `(state, questions, answer)` where:
     - `state`: HTML/CSS snippet with WCAG violations
     - `questions`: Structured queries (e.g., "Does this violate 1.4.3 Contrast?", "Which fix applies?")
     - `answer`: Ground truth decision (violation type, location, fix)
   - Sources:
     - `wcag-skill` test cases / fixtures
     - Synthetic generation via the skill's own validators
     - Real-world audit samples (from `vegaml-wcag` demo pages)

3. **Save as JSONL** for vegaml calibration:
   ```jsonl
   {"state": "<html>...", "questions": {"contrast": {"type": "boolean", "instructions": "..."}, "fix": {"type": "choice", "instructions": "...", "options": "..."}}, "answer": {"contrast": true, "fix": "increase-contrast"}}
   ```

### Phase 2: Fit vegaml Teacher Model

1. **Install vegaml** (from `turbolego/vegaml` main branch)
   ```bash
   pip install "vegaml[serve,calibrate]"
   ```

2. **Run calibration** on the 300 examples:
   ```bash
   vegaml-calibrate fit \
     --data wcag_skill_examples.jsonl \
     --model 0.8b \
     --out wcag_calibration.json \
     --alpha 0.1 \
     --band 0.3
   ```
   - This produces a `Calibration` object with:
     - Temperature scaling per question head
     - Conformal prediction bands (review thresholds)
     - Fitted heads for each WCAG check type

3. **Verify calibration quality**:
   - Target: ≥95% coverage on held-out examples
   - Band should route ambiguous cases to "review" correctly

### Phase 3: Weight Editing (The Core Technique)

Following the blog post method — **write the decision valleys into Qwen's weights**:

1. **Load Qwen3.5-0.8B** in the Kaggle notebook:
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   import torch
   
   model = AutoModelForCausalLM.from_pretrained(
       "Qwen/Qwen3.5-0.8B",
       torch_dtype=torch.bfloat16,
       device_map="auto"
   )
   tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
   ```

2. **Extract vegaml's decision boundaries** (the "valleys"):
   - The calibrated vegaml model defines linear decision boundaries in the embedding space
   - For each question head, we have: `w @ x + b > threshold` → decision
   - These are the "valleys" the blog post refers to

3. **Apply the weight edit** (3ms operation):
   ```python
   # Core idea from the blog: add a rank-1 update to the final layer
   # that steers the model's logits toward the vegaml decision
   
   with torch.no_grad():
       # Get the output projection weight (lm_head)
       W = model.lm_head.weight  # [vocab_size, hidden_dim]
       
       # For each vegaml decision head, compute the steering vector
       # This is a simplified version — the actual notebook has the exact formula
       for head_name, head_weight in vegaml_heads.items():
           # head_weight: [hidden_dim] — the linear classifier from vegaml
           # Scale and add to relevant token embeddings
           steering = head_weight * alpha  # alpha from calibration
           W[wcag_token_ids] += steering.unsqueeze(0)
   ```

4. **Reference**: The Kaggle notebook `fast-edits-decision-model` contains the exact implementation:
   - Loads vegaml calibration
   - Computes the steering vectors
   - Applies rank-1 updates to `lm_head.weight`
   - Saves the modified model

### Phase 4: Save and Validate the Modified Model

1. **Save the edited model**:
   ```python
   model.save_pretrained("./qwen3.5-0.8b-wcag-skill")
   tokenizer.save_pretrained("./qwen3.5-0.8b-wcag-skill")
   ```

2. **Push to Hugging Face** (optional):
   ```python
   model.push_to_hub("turbolego/qwen3.5-0.8b-wcag-skill")
   ```

3. **Validation suite** (run on Kaggle):
   - **Tool-use accuracy**: Test on 100 held-out WCAG examples
     - Target: ≥98% (from 18% baseline)
   - **General capability retention**: Run MMLU / GSM8K subset
     - Target: <1% degradation
   - **WCAG-specific benchmarks**:
     - Detection accuracy on `wcag-skill` test set
     - Fix suggestion quality (human eval or LLM-as-judge)
     - False positive rate on clean HTML

### Phase 5: Integration into vegaml-wcag Demo

1. **Update the demo** to use the skill-embedded model:
   - Modify `docs/wcag-demo/app.js` to call the local model
   - Or serve via `vegaml-serve` with the modified checkpoint

2. **Add WASM/ONNX export** (for browser inference):
   - Export the edited model to ONNX
   - Quantize to INT4 for browser (following existing WASM pipeline)
   - Update `docs/wcag-demo/vendor/` with new model chunks

---

## Kaggle Notebook Setup

Create a new notebook at `notebooks/embed_wcag_skill.ipynb` based on the reference:

```python
# notebooks/embed_wcag_skill.ipynb
# Based on: https://www.kaggle.com/code/hummern/fast-edits-decsion-model/

# 1. Install dependencies
# 2. Load vegaml calibration (wcag_calibration.json)
# 3. Load Qwen3.5-0.8B
# 4. Apply weight edits (core 3ms operation)
# 5. Save and validate
# 6. Export to ONNX for WASM demo
```

**Required packages**:
```
vegaml @ git+https://github.com/turbolego/vegaml@main
transformers>=4.40
accelerate
safetensors
onnx
onnxruntime
```

---

## Timeline Estimate

| Phase | Duration | Notes |
|-------|----------|-------|
| Phase 1: Dataset prep | 1-2 days | Depends on wcag-skill example availability |
| Phase 2: Calibration | 2-4 hours | Mostly automated |
| Phase 3: Weight edit | 30 min | Core operation is 3ms; validation takes time |
| Phase 4: Validation | 1 day | Automated benchmarks + human spot-check |
| Phase 5: Demo integration | 1 day | WASM export + demo update |

**Total**: ~3-4 days of calendar time, ~1 day of active compute

---

## Success Criteria

| Metric | Baseline (Qwen3.5-0.8B) | Target (Post-Edit) |
|--------|------------------------|-------------------|
| WCAG tool-use accuracy | ~18% | ≥98% |
| MMLU (subset) | ~60% | ≥59% (≤1% drop) |
| GSM8K (subset) | ~55% | ≥54% (≤1% drop) |
| WCAG detection F1 | N/A | ≥0.90 |
| False positive rate | N/A | ≤5% |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Weight edit degrades general capabilities | Validate on MMLU/GSM8K; keep alpha conservative; the blog reports <1% drop |
| Calibration doesn't transfer to Qwen | Test on held-out data before editing; if coverage <90%, collect more examples |
| WCAG skill too complex for 0.8B | The blog achieved 98% on tool-use with 0.8B; WCAG decisions are similar complexity |
| ONNX export fails for edited model | Test export early; fallback to serving via `vegaml-serve` with Python backend |

---

## File Structure After Completion

```
vegaml-wcag/
├── notebooks/
│   └── embed_wcag_skill.ipynb       # Main execution notebook
├── wcag_calibration.json             # Fitted vegaml calibration
├── wcag_skill_examples.jsonl         # 300 training examples
├── qwen3.5-0.8b-wcag-skill/          # Edited model (gitignored, pushed to HF)
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files...
└── docs/wcag-demo/
    └── vendor/                       # Updated with new ONNX/INT4 chunks
```

---

## References

- Blog: [A Decision Skill Written Into the Weights](https://www.embedding.in/blog/a-decision-skill-written-into-the-weights)
- Kaggle: [fast-edits-decision-model](https://www.kaggle.com/code/hummern/fast-edits-decsion-model/)
- wcag-skill: https://github.com/turbolego/wcag-skill
- vegaml: https://github.com/turbolego/vegaml
- Qwen3.5-0.8B: https://huggingface.co/Qwen/Qwen3.5-0.8B
- vegaml-wcag (this repo): https://github.com/turbolego/vegaml-wcag