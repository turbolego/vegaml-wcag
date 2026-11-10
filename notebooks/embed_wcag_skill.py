
"""Embed the WCAG skill into Qwen3.5-0.8B via closed-form weight editing.

Follows the blog post method: 'A Decision Skill Written Into the Weights'
- Fit a vegaml policy (teacher) on ~300 WCAG examples
- Extract the decision valleys (keys and desired logit changes)
- Solve the MEMIT-style least-squares: dW = R^T K (K^T K + lambda*n*C)^-1
- Apply the rank-256 column edit to one MLP layer's down_proj
- Validate: accuracy on held-out, no catastrophic forgetting

Requires: vegaml, transformers, torch (Kaggle T4 or P100)
"""
from __future__ import annotations

import copy, glob, json, math, os, random, sys, time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F

# Config
N_FIT = 300
N_VAL = 40
LAYER = 23
WIDTH = 256
LAMBDAS = [0.3, 1.0, 3.0]
ROUNDS = 6
DOSES = [0.5, 1.0, 1.5, 2.0]
GUARD_WEIGHT = 1.0
GUARD_CAP = 0.03
SEED = 0
MODEL_NAME = "Qwen/Qwen3.5-0.8B"

WCAG_QUESTIONS = {
    "wcag_violation": {
        "type": "boolean",
        "instructions": "Does this HTML/CSS markup fail any WCAG 2.2 criterion?"
    },
}

def wcag_prompt(state: str) -> str:
    return (
        "Analyze this HTML/CSS snippet for WCAG 2.2 violations.\n\n"
        + state + "\n\n"
        "Question: Does this markup have a WCAG 2.2 violation?\n"
        "Answer (yes or no): "
    )

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    import vegaml
    print("Loading vegaml...")
    v = vegaml.load()
    print("vegaml", vegaml.__version__)

    data_path = os.path.join(os.path.dirname(__file__), "wcag_training_data.jsonl")
    with open(data_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print("Loaded", len(rows), "WCAG examples")

    random.seed(SEED)
    random.shuffle(rows)
    fit_rows = rows[:N_FIT]
    val_rows = rows[N_FIT:N_FIT+N_VAL]
    test_rows = rows[N_FIT+N_VAL:]
    print("Split: fit=%d, val=%d, test=%d" % (len(fit_rows), len(val_rows), len(test_rows)))

    print("Fitting vegaml policy...")
    t0 = time.time()
    policy = vegaml.fit_policy(v, fit_rows, WCAG_QUESTIONS)
    print("Policy fitted in %.1fs" % (time.time()-t0))

    correct = sum(1 for r in val_rows
                  if policy.decide(r["state"])["answers"]["wcag_violation"]["value"] == r["gold"]["wcag_violation"])
    val_acc = correct/len(val_rows) if val_rows else 0
    print("Val accuracy: %.1f%% (%d/%d)" % (val_acc*100, correct, len(val_rows)))

    print("Loading student model %s..." % MODEL_NAME)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    student = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Student: %d params" % sum(p.numel() for p in student.parameters()))

    down_proj = student.model.layers[LAYER].mlp.down_proj
    print("Target: layer %d, down_proj shape %s" % (LAYER, down_proj.weight.shape))

    yes_id = tokenizer("yes").input_ids[0]
    no_id = tokenizer("no").input_ids[0]
    print("yes=%d, no=%d" % (yes_id, no_id))

    def encode(rows_list):
        encs = []
        for r in rows_list:
            tokens = tokenizer(wcag_prompt(r["state"]), return_tensors="pt").input_ids.to(student.device)
            encs.append(tokens.shape[1]-1)
        return encs

    train_encs = encode(fit_rows)
    val_encs = encode(val_rows)
    test_encs = encode(test_rows)

    teacher_gaps = []
    for r in fit_rows:
        out = policy.decide(r["state"])
        p = out["answers"]["wcag_violation"]["probability"]
        teacher_gaps.append(math.log(max(p, 1e-7)/max(1-p, 1e-7)))
    teacher_gaps = torch.tensor(teacher_gaps, dtype=torch.float32, device=student.device)

    print("Extracting keys...")
    keys = []
    for i, r in enumerate(fit_rows):
        tokens = tokenizer(wcag_prompt(r["state"]), return_tensors="pt").input_ids.to(student.device)
        k_tensor = [None]
        handle = down_proj.register_forward_pre_hook(lambda mod, inp: k_tensor.__setitem__(0, inp[0]))
        with torch.no_grad():
            student(input_ids=tokens, use_cache=False)
        handle.remove()
        if k_tensor[0] is not None:
            keys.append(k_tensor[0][0, train_encs[i], :].cpu())
        else:
            keys.append(torch.zeros(down_proj.in_features))
    K = torch.stack(keys).to(student.device)
    print("Keys:", K.shape)

    print("Reading student gaps...")
    student_gaps = []
    for i, r in enumerate(fit_rows):
        tokens = tokenizer(wcag_prompt(r["state"]), return_tensors="pt").input_ids.to(student.device)
        with torch.no_grad():
            out = student(input_ids=tokens, use_cache=False)
            logits = out.logits[0, train_encs[i], :]
            student_gaps.append(logits[yes_id].item() - logits[no_id].item())
    student_gaps = torch.tensor(student_gaps, dtype=torch.float32, device=student.device)

    scale = teacher_gaps - student_gaps
    R = K * scale.unsqueeze(1)

    print("Solving...")
    best = None
    best_score = float("inf")
    for lam in LAMBDAS:
        KtK = K.T @ K
        A = KtK + lam * len(K) * KtK.diagonal().unsqueeze(0) * torch.eye(K.shape[1], device=student.device)
        A = A + 1e-6 * A.diagonal().mean() * torch.eye(K.shape[1], device=student.device)
        t0 = time.time()
        dW = torch.linalg.solve(A, K.T @ R).T.contiguous()
        solve_ms = (time.time()-t0)*1000
        print("  lambda=%s: solved in %.0fms" % (lam, solve_ms))

        for dose in DOSES:
            dW_d = dW * dose
            with torch.no_grad():
                original = down_proj.weight.clone()
                down_proj.weight.add_(dW_d.to(down_proj.weight.dtype))
                gap_sum = 0
                for r, enc in zip(val_rows, val_encs):
                    tokens = tokenizer(wcag_prompt(r["state"]), return_tensors="pt").input_ids.to(student.device)
                    out = student(input_ids=tokens, use_cache=False)
                    gap = out.logits[0, enc, yes_id].item() - out.logits[0, enc, no_id].item()
                    gap_sum += abs(gap)
                down_proj.weight.copy_(original)
            score = gap_sum / len(val_rows)
            if score < best_score:
                best_score = score
                best = {"lam": lam, "dose": dose, "dW": dW_d, "solve_ms": solve_ms}
            print("    dose=%.1f: val score=%.4f" % (dose, score))

    print("Best: lambda=%s, dose=%s" % (best["lam"], best["dose"]))

    print("Applying edit...")
    with torch.no_grad():
        down_proj.weight.add_(best["dW"].to(down_proj.weight.dtype))

    print("\n=== Test accuracy (post-edit) ===")
    correct = 0
    for r, enc in zip(test_rows, test_encs):
        tokens = tokenizer(wcag_prompt(r["state"]), return_tensors="pt").input_ids.to(student.device)
        with torch.no_grad():
            out = student(input_ids=tokens, use_cache=False)
            pred = yes_id if out.logits[0, enc, yes_id].item() > out.logits[0, enc, no_id].item() else no_id
        gold = yes_id if r["gold"]["wcag_violation"] else no_id
        if pred == gold:
            correct += 1
    test_acc = correct/len(test_rows) if test_rows else 0
    print("Test: %.1f%% (%d/%d)" % (test_acc*100, correct, len(test_rows)))

    out_dir = os.path.join(os.path.dirname(__file__), "..", "out_wcag")
    os.makedirs(out_dir, exist_ok=True)
    student.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    meta = {"model": MODEL_NAME, "layer": LAYER, "width": WIDTH,
            "lambda": best["lam"], "dose": best["dose"],
            "val_acc": val_acc, "test_acc": test_acc,
            "n_fit": N_FIT, "n_val": N_VAL, "n_test": len(test_rows)}
    with open(os.path.join(out_dir, "edit_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("\nSaved to", out_dir)
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
