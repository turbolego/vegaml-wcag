# Closed-Form Skill Embedding — Results v1

**Date:** 2026-11-10
**Kernel:** `hummern/embed-wcag-skill` v10 (Kaggle T4)
**Status:** COMPLETE

## Pipeline Timing

| Step | Time |
|------|------|
| vegaml policy fit (300 rows) | 45.0s |
| Key extraction (300 forward passes) | 20s |
| Student gap extraction (300 forward passes) | 20s |
| Closed-form solve (3 lambdas) | 40-119ms each |
| Dose search (4 doses × 3 lambdas × 40 val rows) | ~33s |
| **Total** | **~3.5 min** |

## Results

| Metric | Value |
|--------|-------|
| Teacher (vegaml) val accuracy | **90.0% (36/40)** |
| Best lambda | 3.0 |
| Best dose | 0.5 |
| **Post-edit test accuracy** | **5.0% (1/20)** |
| Val score (best) | 3.18 |

## Diagnosis: Test accuracy collapsed

The val score looks reasonable (3.18) but test accuracy is 5% — the edit destroyed the model's ability to answer correctly. Root causes:

1. **V mapping is too crude**: `V = direction × scale` uses only the yes/no embedding difference direction. This doesn't capture the full residual stream semantics.
2. **Single-layer edit at layer 23**: The blog post edits one layer's MLP. Our down_proj edit may not have the right magnitude/direction.
3. **Dose too high even at 0.5**: The val score metric (mean squared gap error) may not correlate with accuracy.

## Fixes for v2

1. Use **actual output values** from the teacher, not just gap direction:
   - Extract teacher's full hidden state at layer 23
   - Use `V = teacher_hidden - student_hidden` (residual gap)
2. **Edit the MLP up_proj or gate_proj** instead of down_proj (closer to MEMIT's target)
3. **Lower doses**: try 0.1, 0.25, 0.5, 1.0
4. **Use accuracy as the val guard metric**, not MSE
5. Consider **multi-layer edit** (layers 20-25) per the blog post's layer sweep
