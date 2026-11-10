# Closed-Form Skill Embedding — Results v2

**Date:** 2026-11-10
**Kernel:** `hummern/embed-wcag-skill` v12 (Kaggle T4)
**Status:** COMPLETE

## Key Finding: Qwen3.5-0.8B already knows WCAG

| Metric | Value |
|--------|-------|
| Teacher (vegaml) val accuracy | 90.0% (36/40) |
| **Student baseline test accuracy** | **95.0% (19/20)** |
| Best edit (lambda=0.3, dose=0.1) val | 70.0% (28/40) |
| Post-edit test accuracy | 95.0% (19/20) — no change |

The base model already answers WCAG violation questions at 95% — **higher than the vegaml teacher (90%)**. The closed-form edit cannot improve on this and actually degrades val accuracy at higher doses (70% → 30%).

## Pipeline Timing (v12)

| Step | Time |
|------|------|
| Policy fit | 50.7s |
| Key extraction (300 fwd) | 21s |
| Student gaps (300 fwd) | 21s |
| Residual extraction (300 fwd) | included above |
| Solve (3 lambdas) | 45-218ms |
| Dose search (12 configs × 40 val) | ~33s |
| **Total** | **~4.3 min** |

## Diagnosis

The blog post's method assumes the base model does NOT know the skill. For WCAG on Qwen3.5-0.8B, this assumption fails — the model is already proficient. The closed-form edit in the residual direction doesn't add information the model lacks.

## Next Steps

1. **Test on a skill the model doesn't know**: e.g., a proprietary/internal domain (Laya's specific game rules, or a synthetic skill)
2. **Use larger/dirtier teacher**: vegaml policy with more training data could exceed 95%
3. **Multi-layer edit** (layers 20-25) might help more than single-layer
4. **Different target**: edit `gate_proj` or `up_proj` instead of `down_proj`

## Artifacts

- Model saved at `/kaggle/working/out_wcag/` (edit_metadata.json has full metrics)
- Notebook: `notebooks/embed_wcag_skill.ipynb` v12
- Pushed to: `turbolego/vegaml-wcag` main branch

## Model Artifact Verified (2026-11-10 15:25 UTC)

- **File:** `models/wcag-embed-v1/out_wcag/model.safetensors` (1.5 GB, 320 tensors, BF16)
- **Edited tensor:** `model.language_model.layers.23.mlp.down_proj.weight` shape=[1024, 3584] — contains the closed-form dW edit (lambda=0.3, dose=0.1)
- All tokenizer/config files present and valid
