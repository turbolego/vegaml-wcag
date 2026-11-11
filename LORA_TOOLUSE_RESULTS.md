# WCAG Tool-Use Skill: LoRA vs Closed-Form Embedding (AI-WCAG-Gauntlet)

**Date:** 2026-11-10
**Goal:** teach Qwen3.5-0.8B the *agentic tool-use* part of the wcag-skill
(https://github.com/turbolego/wcag-skill): while running the
AI-WCAG-Gauntlet loop, route to the right validator
(`axe` / `pa11y` / `qualweb` / W3C / fix / check) and run a pre-check on your
own code **before** the benchmark's official iteration check.

Not WCAG Q&A (the base model is already 95% on that — see
`EMBED_RESULTS_v2.md`). This is the tool-routing decision skill.

## Task

Two fixed-type questions over a compact gauntlet state text
(`missing_tags`, `w3c_errors`, `axe_violations`, `pa11y_issues`,
`qualweb_issues`):

- `next_tool` — 6-way choice: `w3c | axe | pa11y | qualweb | fix | check`
- `run_precheck` — boolean: run a validator on your own code now?

**Dataset** (`notebooks/tooluse_{fit,val,test}.jsonl`, 300/40/155 rows,
495 total): 15 real gauntlet page states, each run through the actual
`test-suite.sh` check suite (W3C + axe + pa11y + QualWeb + tag coverage,
Chrome for Testing on this host), expanded with tag-coverage variants,
post-fix variants and edge cases. Gold labels derived from the validator
flags per the skill's triage rules (W3C first, then landmarks/structure,
perceivable, operable, ACT rules, then code work, then pass).

## Closed-form embedding (MEMIT) — did NOT transfer

`notebooks/embed_tooluse_skill.py`: rank-1 column edit on layer-23
`down_proj` (same method as `embed_wcag_skill.py`).

| Kernel | Result |
|--------|--------|
| `hummern/embed-tooluse-skill-v10` | teacher (vegaml) val 97.5%, **test 6.5%** on both questions — below the 16.7% random floor; the edit actively hurt |

The teacher knows the skill; a rank-1 broadcast into one layer's down_proj
cannot move the next-token distribution the way tool routing requires.

## LoRA — transferred 100%

`notebooks/lora_tooluse_skill.py`: rank-32 LoRA (q/k/v/o_proj +
gate/up/down_proj, α=64), bf16, T4, 2 epochs, lr 2e-4, on the 300 fit rows.

| Metric (155 held-out test rows) | Base Qwen3.5-0.8B | After LoRA |
|---|---|---|
| `next_tool` | **0%** (emits `1` / prose — no tool-use capability) | **100%** |
| `run_precheck` | 6.5% | **100%** |

Kernels: `hummern/lora-tooluse-skill-v7` (final), adapters in
`out_tooluse_lora/` → published at
https://huggingface.co/Turbolego/qwen3.5-0.8b-wcag-gauntlet-lora
(`adapter_model.safetensors` + tokenizer + `lora_meta.json` + full model card).
Sample predictions: `w3c\nyes`, `axe\nyes`, `qualweb\nyes` — exact, no prose.

### Lessons (why v5/v6 showed 0%)

1. **Chat template is required.** Qwen3.5 chat models emit
   `1\nyes\nassistant` (or  prose) unless the prompt is wrapped in
   `
...[truncated]
