r"""Downstream Propagation Hypothesis (DPH) -- case-study audit, not a
statistical sweep (per instruction: "the first propagation round should be
a case-study audit... a small number of carefully chosen traces is enough
for this stage").

DPH (as formulated, not yet a claim): winner direction (sign of
advantage = KL_mass - KL_local) is not determined primarily by local
attention/value geometry at the eviction layer -- it emerges as the two
eviction perturbations propagate through subsequent residual, attention,
and MLP computations.

**Reuses, unmodified:** `delta_from_block_scoped` and `additive_hook` (the
exact, already-proved eviction perturbation and its exact injection point,
`_layers()` on either adapter, both already validated) are used exactly as
`tier_b_eviction_scoped` uses them internally; this script does not
reimplement or alter that algebra, it only additionally requests
`output_hidden_states=True` and adds READ-ONLY forward hooks (capturing
outputs, never modifying them) on the attention and MLP submodules of
later layers to build the propagation trace. **No existing file is
modified** -- the capture-hook helper is defined locally here (not added
to `hooks.py`), a small, self-contained, non-invasive addition.

Case panel: 7 Pythia-160M + 7 GPT-2 cases, selected directly from already-
computed `pairs.json` files (advantage, layer -- no new feature), covering
strong local wins, strong mass wins, near-ties, early/mid/late depth, and
GPT-2's two most extreme tail events (one per winner direction, already
identified in `results/directional_value_geometry_gpt2/report.txt`).

Writes `results/downstream_propagation_audit/{summary,cases}.json` and
`report.txt`. Does not modify `src/kv_efficiency/`, any test, any existing
result file, or any doc/status file.
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from kv_efficiency.attention_reconstruction import GPTNeoXAdapter, head_values_dmodel
from kv_efficiency.attention_reconstruction_gpt2 import GPT2Adapter
from kv_efficiency.damage_metrics import kl_divergence
from kv_efficiency.eviction import delta_from_block_scoped, require_normalized
from kv_efficiency.hooks import additive_hook
from kv_efficiency.policies import attention_mass_ranking, local_damage_ranking

RESULTS_DIR = ROOT / "results" / "downstream_propagation_audit"
MASS_EPSILON = 1e-9
DEPTH_MAX = 11  # both models have 12 layers

_cfg = yaml.safe_load((ROOT / "configs" / "milestone2_grid.yaml").read_text())
EVALUATION_PROMPTS = _cfg["evaluation_prompts"]  # identical strings for both architectures

# ---------------------------------------------------------------------------
# Case panel -- selected from existing pairs.json files (advantage, layer),
# no new feature used anywhere in the selection.
# ---------------------------------------------------------------------------
PYTHIA_CASES = [
    dict(tag="mass_strong_L0", prompt=1, layer=0, head=6, i=24, j=2),
    dict(tag="local_strong_L0", prompt=1, layer=0, head=3, i=40, j=35),
    dict(tag="mass_L3", prompt=1, layer=3, head=3, i=18, j=29),
    dict(tag="local_L6", prompt=2, layer=6, head=6, i=34, j=31),
    dict(tag="near_tie_L6", prompt=0, layer=6, head=9, i=11, j=24),
    dict(tag="local_L9", prompt=2, layer=9, head=3, i=8, j=30),
    dict(tag="local_L11_late", prompt=2, layer=11, head=9, i=32, j=17),
]
GPT2_CASES = [
    dict(tag="local_TAIL_EXTREME", prompt=1, layer=1, head=2, i=39, j=42),
    dict(tag="mass_TAIL_EXTREME", prompt=0, layer=1, head=2, i=37, j=39),
    dict(tag="mass_L3", prompt=0, layer=3, head=0, i=16, j=6),
    dict(tag="local_L6", prompt=0, layer=6, head=7, i=54, j=41),
    dict(tag="local_L9", prompt=0, layer=9, head=8, i=4, j=34),
    dict(tag="mass_L11_late", prompt=1, layer=11, head=11, i=14, j=39),
    dict(tag="near_tie_L7", prompt=0, layer=7, head=10, i=25, j=6),
]


# ---------------------------------------------------------------------------
# architecture-specific module access (read-only; no existing file touched)
# ---------------------------------------------------------------------------

def get_attn_mlp(model, arch: str, layer: int):
    if arch == "pythia":
        lm = model.gpt_neox.layers[layer]
        return lm.attention, lm.mlp
    b = model.transformer.h[layer]
    return b.attn, b.mlp


def get_final_norm_and_head(model, arch: str):
    if arch == "pythia":
        return model.gpt_neox.final_layer_norm, model.lm_head
    return model.transformer.ln_f, model.lm_head


def n_layers(model, arch: str) -> int:
    return len(model.gpt_neox.layers) if arch == "pythia" else len(model.transformer.h)


def get_block(model, arch: str, layer: int):
    """The WHOLE transformer block (distinct from adapter._layers(), which
    for GPT2Adapter deliberately returns `.attn` -- the eviction injection
    point, not the block-output residual-capture point). Used only for our
    own read-only residual-stream capture hooks, never for injection."""
    return model.gpt_neox.layers[layer] if arch == "pythia" else model.transformer.h[layer]


def load_pythia():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(_cfg["model_id"])
    model = AutoModelForCausalLM.from_pretrained(
        _cfg["model_id"], dtype=getattr(torch, _cfg["dtype"]), attn_implementation=_cfg["attn_implementation"]
    ).eval()
    return tok, model, GPTNeoXAdapter(model)


def load_gpt2():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32, attn_implementation="eager").eval()
    return tok, model, GPT2Adapter(model)


@contextmanager
def capture_output_hook(module: torch.nn.Module, capture: dict, key: str):
    """Read-only forward hook capturing a module's OUTPUT (handles tuple
    outputs, e.g. GPT2Attention returns (attn_output, attn_weights)).
    Never modifies the output. Local to this script -- hooks.py untouched."""
    def hook(_module, _inputs, output):
        out = output[0] if isinstance(output, tuple) else output
        capture[key] = out.detach().clone()
    handle = module.register_forward_hook(hook)
    try:
        yield capture
    finally:
        handle.remove()


@contextmanager
def _multi_capture_ctx(items, capture):
    if not items:
        yield capture
        return
    module, key = items[0]
    with capture_output_hook(module, capture, key):
        with _multi_capture_ctx(items[1:], capture):
            yield capture


@contextmanager
def multi_capture(modules_and_keys):
    with _multi_capture_ctx(list(modules_and_keys), {}) as capture:
        yield capture


def logit_lens(model, arch: str, hidden_vec: torch.Tensor) -> torch.Tensor:
    """Apply the model's OWN final layer norm + lm_head to an intermediate
    residual-stream vector (standard 'logit lens' probe). Approximate at
    early layers -- a well-known limitation, stated in the report, not
    hidden. Uses only existing, unmodified model modules."""
    final_norm, lm_head = get_final_norm_and_head(model, arch)
    with torch.no_grad():
        normed = final_norm(hidden_vec.unsqueeze(0).to(final_norm.weight.dtype))
        logits = lm_head(normed)[0]
    return logits.double()


def get_stratum_data(model, adapter, tok, arch, p_idx, layer, head, i, j):
    ids = tok(EVALUATION_PROMPTS[p_idx], return_tensors="pt").input_ids
    n = ids.shape[1]
    qp = n - 1
    with torch.no_grad():
        out = model(ids, output_attentions=True, output_hidden_states=True)
    hidden_in = out.hidden_states[layer][0].double()
    alpha_full = out.attentions[layer][0, head].double()
    values_dmodel = head_values_dmodel(adapter, layer, head, hidden_in)
    _, _, value_h = adapter.head_context_and_value(layer, head, hidden_in)  # head-size space, for delta_from_block_scoped
    alpha_last = require_normalized(alpha_full[qp], renormalize=True)
    protect = {0, qp}
    mass_rank = attention_mass_ranking(alpha_last, protect=protect)
    local_rank = local_damage_ranking(alpha_last, values_dmodel, protect=protect)

    mass_set, local_set = set(), set()
    change_k = None
    for k in range(1, len(mass_rank) + 1):
        mass_set.add(mass_rank[k - 1])
        local_set.add(local_rank[k - 1])
        if (i in mass_set) != (i in local_set) or (j in mass_set) != (j in local_set):
            change_k = k
            break
    if change_k is None:
        raise ValueError(f"pair (i={i},j={j}) never changes the set within this stratum's ranking")

    return dict(
        ids=ids, n=n, qp=qp, alpha_full=alpha_full, hidden_in=hidden_in,
        values_dmodel=values_dmodel, value_h=value_h, wo_h=adapter.w_o_head(layer, head),
        mass_rank=mass_rank, local_rank=local_rank, change_k=change_k,
        logits_ref_full=out.logits[0].double(),
    )


def run_traced_forward(model, adapter, arch, sd, layer, head, block, trace_layers):
    """One forward pass, optionally perturbed (block=None => reference),
    capturing residual-stream state, attention-output, and mlp-output at
    every layer in trace_layers, all at the measured position qp.

    IMPORTANT: does NOT use `output_hidden_states=True`. Verified this
    session (diagnostic, not assumed) that this installed transformers
    version wraps `GPTNeoXModel.forward` in a `@capture_outputs` decorator
    whose hidden_states collection does NOT reflect a forward hook's
    modification -- confirmed directly: a huge injected delta changed the
    final logits correctly (0.204 diff) but left `out.hidden_states[l+1]`
    completely unchanged (0.0 diff), while a separate, independent
    read-only capture hook on the SAME module correctly saw the perturbed
    value (norm 281 vs. the framework tuple's nonsensical norm 120797).
    So: every residual-stream value here is captured via our own read-only
    forward hooks (the same mechanism already used for attn/mlp, and the
    same mechanism `tests/test_gpt2_adapter.py`'s c_proj-hook validation
    already relies on) -- never via `output_hidden_states`. `out.logits`
    IS reliable under hooks (confirmed identical with/without
    output_hidden_states in the same diagnostic) and is used as-is."""
    qp = sd["qp"]
    modules_and_keys = []
    for l in trace_layers:
        modules_and_keys.append((get_block(model, arch, l), f"resid_{l}"))
        attn_m, mlp_m = get_attn_mlp(model, arch, l)
        modules_and_keys.append((attn_m, f"attn_{l}"))
        modules_and_keys.append((mlp_m, f"mlp_{l}"))

    def do_forward():
        with multi_capture(modules_and_keys) as capture:
            with torch.no_grad():
                out = model(sd["ids"])
        return out, capture

    if block is None:
        out, capture = do_forward()
    else:
        a = require_normalized(sd["alpha_full"], renormalize=True)
        delta_context, invalid = delta_from_block_scoped(a, sd["value_h"], block, mass_epsilon=MASS_EPSILON)
        if bool(invalid[qp]):
            raise ValueError(f"measured position {qp} invalid (m_B -> 1) for this block")
        delta_dmodel = delta_context @ sd["wo_h"].T
        with additive_hook(adapter._layers()[layer], delta_dmodel.float(), position=None):
            out, capture = do_forward()

    hidden_by_layer = {l: capture[f"resid_{l}"][0, qp, :].double().clone() for l in trace_layers}
    logits_qp = out.logits[0, qp, :].double()
    attn_by_layer = {l: capture[f"attn_{l}"][0, qp, :].double().clone() for l in trace_layers}
    mlp_by_layer = {l: capture[f"mlp_{l}"][0, qp, :].double().clone() for l in trace_layers}
    return dict(hidden=hidden_by_layer, logits=logits_qp, attn=attn_by_layer, mlp=mlp_by_layer)


def trace_case(model, adapter, tok, arch, case: dict) -> dict:
    layer, head, i, j, p_idx = case["layer"], case["head"], case["i"], case["j"], case["prompt"]
    sd = get_stratum_data(model, adapter, tok, arch, p_idx, layer, head, i, j)
    n_lay = n_layers(model, arch)
    trace_layers = list(range(layer, n_lay))

    mass_block = torch.zeros(sd["n"], dtype=torch.bool)
    for idx in sd["mass_rank"][: sd["change_k"]]:
        mass_block[idx] = True
    local_block = torch.zeros(sd["n"], dtype=torch.bool)
    for idx in sd["local_rank"][: sd["change_k"]]:
        local_block[idx] = True

    ref = run_traced_forward(model, adapter, arch, sd, layer, head, None, trace_layers)
    mass = run_traced_forward(model, adapter, arch, sd, layer, head, mass_block, trace_layers)
    local = run_traced_forward(model, adapter, arch, sd, layer, head, local_block, trace_layers)

    kl_mass_final = float(kl_divergence(sd["logits_ref_full"][sd["qp"]], mass["logits"]))
    kl_local_final = float(kl_divergence(sd["logits_ref_full"][sd["qp"]], local["logits"]))
    advantage_final = kl_mass_final - kl_local_final
    final_winner = "local" if advantage_final > 0 else "mass"

    # Block-index convention throughout (0..n_lay-1), matching trace_layers and
    # the attn/mlp capture keys exactly. hidden[b] = residual state AFTER
    # block b has run (our own capture hook on the whole block, keyed
    # directly by block index -- see run_traced_forward's docstring for why
    # output_hidden_states is not used). Residual states before the eviction
    # layer are, by construction of the additive hook, identical across
    # ref/mass/local, so the table starts at the eviction layer itself, per
    # Part IV's instruction.
    per_layer = []
    for b in trace_layers:
        h_ref, h_mass, h_local = ref["hidden"][b], mass["hidden"][b], local["hidden"][b]
        d_mass_resid = float(torch.linalg.norm(h_mass - h_ref))
        d_local_resid = float(torch.linalg.norm(h_local - h_ref))
        cos_mass = float(torch.dot(h_mass, h_ref) / (torch.linalg.norm(h_mass) * torch.linalg.norm(h_ref) + 1e-30))
        cos_local = float(torch.dot(h_local, h_ref) / (torch.linalg.norm(h_local) * torch.linalg.norm(h_ref) + 1e-30))

        ll_ref = logit_lens(model, arch, h_ref)
        ll_mass = logit_lens(model, arch, h_mass)
        ll_local = logit_lens(model, arch, h_local)
        kl_mass_ll = float(kl_divergence(ll_ref, ll_mass))
        kl_local_ll = float(kl_divergence(ll_ref, ll_local))

        attn_ref, mlp_ref = ref["attn"][b], ref["mlp"][b]
        attn_mass, attn_local = mass["attn"][b], local["attn"][b]
        mlp_mass, mlp_local = mass["mlp"][b], local["mlp"][b]
        d_attn_mass = float(torch.linalg.norm(attn_mass - attn_ref))
        d_attn_local = float(torch.linalg.norm(attn_local - attn_ref))
        d_mlp_mass = float(torch.linalg.norm(mlp_mass - mlp_ref))
        d_mlp_local = float(torch.linalg.norm(mlp_local - mlp_ref))

        per_layer.append(dict(
            layer=b, depth_fraction=b / DEPTH_MAX,
            d_mass_resid=d_mass_resid, d_local_resid=d_local_resid,
            delta_resid=d_mass_resid - d_local_resid,
            cos_mass=cos_mass, cos_local=cos_local,
            kl_mass_logitlens=kl_mass_ll, kl_local_logitlens=kl_local_ll,
            delta_kl_logitlens=kl_mass_ll - kl_local_ll,
            d_attn_mass=d_attn_mass, d_attn_local=d_attn_local, delta_attn=d_attn_mass - d_attn_local,
            d_mlp_mass=d_mlp_mass, d_mlp_local=d_mlp_local, delta_mlp=d_mlp_mass - d_mlp_local,
        ))

    # decision-emergence depth: earliest layer where sign(delta) matches final winner
    # convention: delta = D_mass - D_local; delta>0 means mass is FARTHER from ref (local closer) => consistent with "local wins"
    def first_consistent_layer(delta_key):
        target_sign = 1 if final_winner == "local" else -1
        for entry in per_layer:
            if entry[delta_key] is None:
                continue
            s = np.sign(entry[delta_key])
            if s == target_sign:
                return entry["layer"]
        return None

    emergence_depth_resid = first_consistent_layer("delta_resid")
    emergence_depth_logitlens = first_consistent_layer("delta_kl_logitlens")

    # winner flips: does sign(delta_resid) change across depth, from the eviction layer onward?
    signs = [np.sign(e["delta_resid"]) for e in per_layer if e["delta_resid"] != 0]
    n_sign_changes = int(sum(1 for a, b in zip(signs, signs[1:]) if a != b))
    # per_layer[0] is the eviction layer itself (trace_layers starts there) -- the
    # earliest point at which the two perturbations exist at all.
    d0 = per_layer[0]["delta_resid"]
    immediate_winner_resid = "local" if d0 > 0 else "mass" if d0 < 0 else "tie"

    return dict(
        arch=arch, tag=case["tag"], prompt=p_idx, layer=layer, head=head, i=i, j=j,
        change_k=sd["change_k"], n_candidates=sd["n"] - 2,
        kl_mass_final=kl_mass_final, kl_local_final=kl_local_final, advantage_final=advantage_final,
        final_winner=final_winner, immediate_winner_at_eviction_layer=immediate_winner_resid,
        emergence_depth_resid=emergence_depth_resid, emergence_depth_logitlens=emergence_depth_logitlens,
        n_sign_changes_resid=n_sign_changes,
        winner_flipped=(immediate_winner_resid != "tie" and immediate_winner_resid != final_winner),
        per_layer=per_layer,
    )


def run() -> dict:
    torch.manual_seed(0)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tok_p, model_p, adapter_p = load_pythia()
    tok_g, model_g, adapter_g = load_gpt2()

    cases = []
    for c in PYTHIA_CASES:
        print(f"[pythia] tracing {c['tag']} ...", flush=True)
        cases.append(trace_case(model_p, adapter_p, tok_p, "pythia", c))
    for c in GPT2_CASES:
        print(f"[gpt2] tracing {c['tag']} ...", flush=True)
        cases.append(trace_case(model_g, adapter_g, tok_g, "gpt2", c))

    (RESULTS_DIR / "cases.json").write_text(json.dumps(cases, indent=2))

    # ---- aggregate / verdict ----
    n_flip = sum(1 for c in cases if c["winner_flipped"])
    n_multi_sign_change = sum(1 for c in cases if c["n_sign_changes_resid"] >= 2)
    emergence_depths = [c["emergence_depth_resid"] for c in cases if c["emergence_depth_resid"] is not None]
    immediate_matches_final = sum(1 for c in cases if not c["winner_flipped"])

    if n_flip == 0 and all((c["emergence_depth_resid"] == c["layer"]) for c in cases if c["emergence_depth_resid"] is not None):
        verdict = "DPH-A"
    elif n_flip >= 1 and n_flip < len(cases) // 2 and n_multi_sign_change == 0:
        verdict = "DPH-B"
    elif n_flip >= len(cases) // 3 or n_multi_sign_change >= 2:
        verdict = "DPH-C"
    else:
        verdict = "DPH-B"

    summary = dict(
        n_cases=len(cases), n_pythia=len(PYTHIA_CASES), n_gpt2=len(GPT2_CASES),
        n_winner_flipped=n_flip, n_multi_sign_change=n_multi_sign_change,
        immediate_matches_final=immediate_matches_final,
        emergence_depths=emergence_depths,
        median_emergence_depth_fraction=(
            float(np.median([d / DEPTH_MAX for d in emergence_depths])) if emergence_depths else None
        ),
        verdict=verdict,
    )
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(summary, cases)
    print("Done. See results/downstream_propagation_audit/report.txt")
    return summary


def _f(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def write_report(summary: dict, cases: list) -> None:
    L = []
    A = L.append
    W = 100
    A("DOWNSTREAM PROPAGATION HYPOTHESIS (DPH) -- CASE-STUDY AUDIT")
    A("=" * W)
    A("")
    A(f"n_cases={summary['n_cases']} ({summary['n_pythia']} Pythia + {summary['n_gpt2']} GPT-2)")
    A(f"winner flips (immediate eviction-layer sign != final winner): {summary['n_winner_flipped']}/{summary['n_cases']}")
    A(f"cases with >=2 sign changes across depth: {summary['n_multi_sign_change']}")
    A(f"median decision-emergence depth (fraction of 11): {_f(summary['median_emergence_depth_fraction'])}")
    A("")

    for c in cases:
        A("-" * W)
        A(f"[{c['arch']}] {c['tag']}  prompt={c['prompt']} layer={c['layer']} head={c['head']} "
          f"i={c['i']} j={c['j']} change_k={c['change_k']}/{c['n_candidates']}")
        A(f"  final: kl_mass={c['kl_mass_final']:.4e}  kl_local={c['kl_local_final']:.4e}  "
          f"advantage={c['advantage_final']:.4e}  final_winner={c['final_winner']}")
        A(f"  immediate winner at eviction layer (residual sign) = {c['immediate_winner_at_eviction_layer']}  "
          f"{'*** WINNER FLIPPED ***' if c['winner_flipped'] else '(matches final)'}")
        A(f"  decision-emergence depth: residual={_f(c['emergence_depth_resid'])}  "
          f"logit-lens={_f(c['emergence_depth_logitlens'])}  (out of {c['per_layer'][-1]['layer']})")
        A(f"  sign changes across depth (residual delta): {c['n_sign_changes_resid']}")
        A(f"  {'layer':>5}{'depth':>7}{'d_mass_resid':>13}{'d_local_resid':>14}{'delta_resid':>12}"
          f"{'d_attn_mass':>12}{'d_attn_local':>13}{'d_mlp_mass':>11}{'d_mlp_local':>12}")
        for e in c["per_layer"]:
            A(f"  {e['layer']:>5}{e['depth_fraction']:>7.2f}{e['d_mass_resid']:>13.4f}{e['d_local_resid']:>14.4f}"
              f"{e['delta_resid']:>12.4f}"
              f"{_f(e.get('d_attn_mass')):>12}{_f(e.get('d_attn_local')):>13}"
              f"{_f(e.get('d_mlp_mass')):>11}{_f(e.get('d_mlp_local')):>12}")

    A("")
    A("-" * W)
    A("VERDICT")
    A("-" * W)
    A(summary["verdict"])

    (RESULTS_DIR / "report.txt").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    run()
