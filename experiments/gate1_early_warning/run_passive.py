r"""Gate 1 primary experiment: passive monitoring.

The model is run normally. No KV state is evicted, masked, or modified. For
each prompt we (1) generate greedily, (2) label the behaviour by a rule, and
(3) recover every decode step's internal diagnostics from ONE teacher-forced
forward pass over prompt+generation -- which is valid because greedy decoding
is deterministic and the attention row for step t sees only tokens up to t-1.

Writes experiments/gate1_early_warning/runs.json (+ diagnostics.npz).
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gate1_lib import (  # noqa: E402
    OUTPUT_BASELINES, Qwen2DiagnosticAdapter, label_format_constraint,
    label_forbidden_word, label_max_words, label_refusal, output_baselines,
    step_diagnostics, verify_teacher_forcing,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW = 32
GAMMA_PAIRS = 200
GAMMA_SEED = 0
LAYER_GROUPS = {"early": (0, 8), "middle": (8, 16), "late": (16, 24)}
DIAGS = ["max_alpha", "entropy", "eff_support", "max_m_B_single",
         "max_m_B_block8", "max_d_local", "inversion_rate", "any_inversion",
         "abs_gamma_mean", "abs_gamma_max", "pole_proximity", "near_one_hot"]


def GEN_CFG(tok):
    """Pure greedy on raw logits. The model's shipped generation_config.json
    sets repetition_penalty=1.1, which is applied even when do_sample=False and
    would make decoding disagree with a teacher-forced pass (AMENDMENT 1.1)."""
    from transformers import GenerationConfig
    return GenerationConfig(do_sample=False, repetition_penalty=1.0,
                            max_new_tokens=MAX_NEW,
                            pad_token_id=tok.eos_token_id,
                            eos_token_id=tok.eos_token_id)


def load():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager").eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return tok, m


def encode(tok, system, user):
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": user}]
    out = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                  return_dict=True, return_tensors="pt")
    return out["input_ids"]


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    tok, model = load()
    ad = Qwen2DiagnosticAdapter(model)
    prompts = json.loads((HERE / "prompts.json").read_text())

    eos_ids = [i for i in {tok.eos_token_id,
                           tok.convert_tokens_to_ids("<|im_end|>")} if isinstance(i, int) and i >= 0]
    refusal_ids = sorted({tok.encode(s, add_special_tokens=False)[0]
                          for s in ["I'm", "I", "Sorry", "sorry", "I am", "Unfortunately"]})

    try:
        rev = model.config._name_or_path, getattr(model.config, "_commit_hash", None)
    except Exception:
        rev = (MODEL_ID, None)

    runs, arrays = [], {}
    t0 = time.time()
    for n, p in enumerate(prompts):
        ids = encode(tok, p.get("system"), p["user"])
        P = ids.shape[1]
        with torch.no_grad():
            gen = model.generate(ids, generation_config=GEN_CFG(tok))
        gen_ids = gen[0, P:].tolist()
        # strip trailing EOS/pad for labelling but keep for indexing
        if p["family"] in ("D", "E"):
            failure, t_F, text = label_format_constraint(tok, gen_ids, p["accepted"])
            label_kind = "rule_format"
        elif p["family"] == "F":
            failure, t_F, text = label_forbidden_word(tok, gen_ids, "water")
            label_kind = "rule_forbidden_word"
        elif p["family"] == "G":
            failure, t_F, text = label_max_words(tok, gen_ids, 3)
            label_kind = "rule_max_words"
        elif p["family"] in ("B", "C"):
            failure, t_F, text = label_refusal(tok, gen_ids)
            label_kind = "rule_refusal_secondary"
        else:
            text = tok.decode(gen_ids, skip_special_tokens=True)
            failure, t_F, label_kind = 0, None, "benign_by_construction"

        bad = verify_teacher_forcing(model, ids, gen_ids)
        full = torch.cat([ids, torch.tensor([gen_ids])], dim=1)
        out, vals = ad.capture_values(full)

        rng = np.random.default_rng(GAMMA_SEED)
        per_layer = {k: [] for k in DIAGS}
        steps = []
        prev_h = None
        n_steps = len(gen_ids)
        for t in range(n_steps):
            row = P - 1 + t
            d = step_diagnostics(ad, out, vals, row, rng, GAMMA_PAIRS)
            ob, prev_h = output_baselines(out, row, prev_h, eos_ids, refusal_ids)
            rec = dict(step=t, row=row, visible_keys=row + 1, seq_len=int(full.shape[1]))
            for k in DIAGS:
                hm = d[k]["head_max"]
                per_layer[k].append(hm)
                rec[k] = round(float(np.max(hm)), 6)
                for gname, (lo, hi) in LAYER_GROUPS.items():
                    rec[f"{k}__{gname}"] = round(float(np.max(hm[lo:hi])), 6)
            for k in OUTPUT_BASELINES:
                rec[k] = round(float(ob[k]), 6)
            steps.append(rec)

        runs.append(dict(
            id=p["id"], family=p["family"], split=p["split"],
            constraint=p.get("constraint"), filler=p.get("filler"),
            tempt=p.get("tempt"), prompt_tokens=int(P), n_generated=n_steps,
            failure=failure, failure_step=t_F, label_kind=label_kind,
            generation=text, teacher_forcing_mismatches=len(bad), steps=steps))
        for k in DIAGS:
            arrays[f"{p['id']}__{k}"] = np.asarray(per_layer[k], dtype=np.float32)
        if (n + 1) % 10 == 0:
            print(f"  {n+1}/{len(prompts)}  {time.time()-t0:.0f}s", flush=True)

    tf_bad = sum(r["teacher_forcing_mismatches"] for r in runs)
    tf_steps = sum(r["n_generated"] for r in runs)
    assert tf_bad <= 0.005 * tf_steps, (
        f"teacher forcing disagrees with decoding at {tf_bad}/{tf_steps} steps. "
        "Decoding must be pure greedy on raw logits for the single-pass "
        "reconstruction to be exact (see AMENDMENT 1.1).")

    meta = dict(model_id=MODEL_ID, resolved=list(rev), max_new_tokens=MAX_NEW,
                dtype="float32", attn_implementation="eager", device="cpu",
                decoding="greedy_raw_logits_repetition_penalty_1.0", seed=0, gamma_pairs=GAMMA_PAIRS,
                gamma_seed=GAMMA_SEED, layer_groups=LAYER_GROUPS,
                n_layers=ad.n_layers, n_q_heads=ad.n_q, n_kv_heads=ad.n_kv,
                transformers=__import__("transformers").__version__,
                torch=torch.__version__)
    (HERE / "runs.json").write_text(json.dumps(dict(meta=meta, runs=runs), indent=1))
    np.savez_compressed(HERE / "diagnostics.npz", **arrays)

    from collections import Counter
    print("\n=== behavioural outcomes ===")
    for fam in "ABCDE":
        sub = [r for r in runs if r["family"] == fam]
        nf = sum(r["failure"] for r in sub)
        print(f"  {fam}: {nf}/{len(sub)} failures ({100*nf/max(len(sub),1):.0f}%)")
    tfs = [r["failure_step"] for r in runs if r["failure"] and r["failure_step"] is not None]
    print(f"  failure_step available for {len(tfs)} runs; distribution {dict(sorted(Counter(tfs).items()))}")
    print(f"  teacher-forcing mismatches total: {sum(r['teacher_forcing_mismatches'] for r in runs)}")
    print(f"  elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
