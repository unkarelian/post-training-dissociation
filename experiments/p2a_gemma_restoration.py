"""P2a: Gemma 3 4B Controlled Workshop Donor Restoration & Native-Use Experiment.

Pinned checkpoint: unsloth/gemma-3-4b-it (bf46152c47f5dd20b896357cb51abc4c03b8ee8c)
Wrapper: Gemma3ForConditionalGeneration -> pre-hook on model.language_model.model.layers[L]
verified to match out.hidden_states[L] with max_abs_diff == 0.0.
Patches ONLY the final non-pad prompt token (-1 under left padding), ONCE during prefill
(x.shape[1] > 1 guard, asserting patch_calls == 1 per generation call).

Runs:
  1. Frozen 6 discovery scenarios -> task- & format-specific both-orders ridge probes at L20 & L24
     with deterministic leave-one-scenario-out (LOSO) gate (>= 5/6 on each order; aborts
     restoration for any task/format failing LOSO).
  2. 12 brand-new held-out scenarios (disjoint from original 6 discovery + 6 held-out):
     - Co-primary open attribution & open dispatch (NO A/B mapping, 24 discordant + 24 concordant)
     - Diagnostic A/B attribution & A/B dispatch (48 discordant + 48 concordant, single-token)
     - Baselines (both orders), answer-supplied oracle, token-length-matched restatement (+/-1 tok)
     - Gating controls (identity_sham, subspace_restore, equal_dose_opposite, wrong_donor_raw on
       discordant, 3-seed norm-matched orthogonal random) + diagnostic controls (actual_only_donor,
       practice_only_donor, donor_full)
     - Native-use controls on practice_first (proj_removal relative to discovery grand mean,
       reverse_shift toward same-assignment actual_first donor, + matched 3-seed random controls)
     - 6-scenario auxiliary {normal_policy, inverted_policy} dispatch decoupling control.
All open replies are scored off-GPU by Jev after download.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any

import torch


MODEL = "unsloth/gemma-3-4b-it"
REVISION = "bf46152c47f5dd20b896357cb51abc4c03b8ee8c"
TRANSFORMERS_VERSION = "4.51.3"
LAYERS = (20, 24)
MAX_NEW_TOKENS = 48
RANDOM_SEEDS = (101, 202, 303)
POST_BF16_COS_TOL = 0.99
POST_BF16_NORM_REL_TOL = 0.05

# Exact 6 original discovery scenarios from experiments/gemma_routing_binding.py
DISCOVERY_SCENARIOS = (
    ("label printer", "It is jammed and needs a repair.", "It has finished its batch and works."),
    ("shipping scale", "It has stopped reading weight and needs service.", "It was checked and is reading accurately."),
    ("folding table", "Its hinge snapped and needs replacing.", "Its hinge folds smoothly and is intact."),
    ("inventory scanner", "It will not start and requires service.", "It has started and scans every item."),
    ("display case", "Its latch is broken and needs a new part.", "Its latch works and closes securely."),
    ("projector", "Its bulb failed and needs replacing.", "Its picture is bright and stable."),
)

# 12 brand-new held-out scenarios strictly disjoint from both original 6 discovery + 6 held-out
NEW_HELDOUT_SCENARIOS = (
    ("barcode gun", "Its trigger stuck and needs a repair.", "Its trigger clicks cleanly and scans items."),
    ("pallet jack", "Its hydraulic pump leaks and needs service.", "Its hydraulic pump lifts smoothly and holds."),
    ("heat sealer", "Its heating strip burned out and needs replacing.", "Its heating strip warms evenly and seals."),
    ("tape dispenser", "Its cutter blade snapped and needs a new part.", "Its cutter blade is sharp and cuts cleanly."),
    ("box stapler", "Its feed rail jammed and needs a repair.", "Its feed rail slides smoothly and staples."),
    ("laminator", "Its roller stalled and needs service.", "Its roller turns evenly and feeds sheets."),
    ("hand truck", "Its axle bent and needs replacing.", "Its axle is straight and rolls smoothly."),
    ("cord reel", "Its rewind spring broke and needs a new part.", "Its rewind spring retracts and locks cleanly."),
    ("bolt cutter", "Its pivot pin sheared and needs replacing.", "Its pivot pin is tight and cuts cleanly."),
    ("bench vise", "Its lead screw stripped and needs service.", "Its lead screw turns smoothly and clamps."),
    ("shop vacuum", "Its motor stalled and needs a repair.", "Its motor starts cleanly and pulls suction."),
    ("air hose", "Its brass coupler leaks and needs replacing.", "Its brass coupler seals tightly and holds pressure."),
)


def ensure_transformers() -> None:
    try:
        installed = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed != TRANSFORMERS_VERSION:
        if "transformers" in sys.modules:
            raise RuntimeError(f"transformers=={installed} already imported; restart kernel.")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", f"transformers=={TRANSFORMERS_VERSION}"])


def get_decoder_layers(model: Any) -> Any:
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        return model.language_model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError(f"Cannot locate decoder layers on {type(model).__name__}")


def task_instruction(task: str, fmt: str, yes_label: str | None = None, inverted_policy: bool = False) -> str:
    if fmt == "open":
        if task == "attribution":
            return (
                "Which workshop, if either, needs a repair right now: the actual workshop, "
                "the fictional practice workshop, both, or neither? Answer in one short sentence."
            )
        if inverted_policy:
            return (
                "Policy: Dispatch a technician to the actual workshop ONLY if it is working properly "
                "for a preventive check, NOT if it needs a repair. Should a technician be dispatched "
                "to the actual workshop right now? Answer yes or no in one short sentence."
            )
        return (
            "Policy: Dispatch a technician to the actual workshop ONLY if it needs a repair. "
            "Should a technician be dispatched to the actual workshop right now? "
            "Answer yes or no in one short sentence."
        )
    # A/B diagnostic format
    assert yes_label in ("A", "B")
    if task == "attribution":
        q = "Does the repair report apply to the actual workshop?"
    elif inverted_policy:
        q = "Should the technician go to the actual workshop? Dispatch ONLY if it is working properly, NOT if it needs a repair."
    else:
        q = "Should the technician go to the actual workshop? Dispatch only if it needs a repair."
    labels = "A means yes and B means no." if yes_label == "A" else "A means no and B means yes."
    return f"{q} {labels} Reply with A or B."


def make_length_matched_restatement(tokenizer: Any, oracle_yes: str, oracle_no: str) -> tuple[str, dict[str, int]]:
    """Construct a generic restatement matched within +/-1 token of the oracle strings."""
    target_len = round((len(tokenizer.encode(oracle_yes, add_special_tokens=False)) + len(tokenizer.encode(oracle_no, add_special_tokens=False))) / 2)
    candidates = [
        "Verified cross-check: Both reports above were logged for this ticket.",
        "Verified cross-check: Both workshop logs above were copied into this ticket.",
        "Verified cross-check: Both workshop reports above were copied onto this ticket.",
        "Verified cross-check: Both workshop reports above were copied into this ticket file.",
        "Verified cross-check: Both reports above were copied accurately for the current ticket.",
        "Verified cross-check: Both reports above were verified as copied for this ticket.",
    ]
    best = min(candidates, key=lambda c: abs(len(tokenizer.encode(c, add_special_tokens=False)) - target_len))
    n_y = len(tokenizer.encode(oracle_yes, add_special_tokens=False))
    n_n = len(tokenizer.encode(oracle_no, add_special_tokens=False))
    n_r = len(tokenizer.encode(best, add_special_tokens=False))
    assert abs(n_r - target_len) <= 1, f"Restatement token mismatch: {n_r} vs {target_len} ({n_y}/{n_n})"
    return best, {"oracle_yes_tok": n_y, "oracle_no_tok": n_n, "restatement_tok": n_r}


def build_cases(scenarios: tuple[tuple[str, str, str], ...], split: str, restatement_txt: str, *, inverted_policy: bool = False) -> list[dict[str, Any]]:
    cases = []
    for equipment, broken, working in scenarios:
        for fmt in ("open", "ab"):
            tasks = ("dispatch",) if inverted_policy else ("attribution", "dispatch")
            labels = (None,) if fmt == "open" else ("A", "B")
            clars = ("none",) if (split == "discover" or inverted_policy) else ("none", "restatement", "oracle")
            for task, order, actual, practice, yes_label, clar in product(
                tasks, ("practice_first", "actual_first"), (False, True), (False, True), labels, clars
            ):
                if clar != "none" and order != "actual_first":
                    continue
                act_rep = f"Actual workshop: {broken if actual else working}"
                prac_rep = f"Fictional practice log: {broken if practice else working}"
                reps = [prac_rep, act_rep] if order == "practice_first" else [act_rep, prac_rep]
                if clar == "restatement":
                    reps.append(restatement_txt)
                elif clar == "oracle":
                    needs = "does need" if actual else "does not need"
                    reps.append(f"Verified cross-check: The actual workshop {needs} this repair.")
                pol_tag = "inv" if inverted_policy else "norm"
                cid = f"{split}__{pol_tag}__{fmt}__{task}__{equipment}__{order}__a{int(actual)}p{int(practice)}__{yes_label or 'open'}__{clar}"
                cases.append({
                    "id": cid,
                    "split": split,
                    "policy": pol_tag,
                    "format": fmt,
                    "task": task,
                    "scenario": equipment,
                    "order": order,
                    "actual_problem": actual,
                    "practice_problem": practice,
                    "discordant": actual != practice,
                    "yes_label": yes_label,
                    "clarification": clar,
                    "prompt": f"Ticket item: {equipment}. {' '.join(reps)} {task_instruction(task, fmt, yes_label, inverted_policy)}",
                })
    return cases


def render(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)


def make_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    pos = attention_mask.long().cumsum(-1) - 1
    return pos.masked_fill(attention_mask == 0, 1)


def fit_ridge(train_x: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    center = train_x.mean(0)
    scale = (train_x - center).square().sum(-1).mean().sqrt().clamp_min(1e-7)
    x = (train_x - center) / scale
    mean_y = target.mean()
    dual = torch.linalg.solve(x @ x.T + 0.1 * torch.eye(len(target), dtype=x.dtype), target - mean_y)
    return center, scale, x.T @ dual, mean_y


def get_ab_label_ids(tokenizer: Any) -> dict[str, list[int]]:
    ids = {
        lbl: sorted({toks[0] for form in (lbl, " " + lbl) if len(toks := tokenizer.encode(form, add_special_tokens=False)) == 1})
        for lbl in ("A", "B")
    }
    assert all(ids.values()) and not (set(ids["A"]) & set(ids["B"]))
    return ids


@torch.inference_mode()
def extract_states_and_ab(
    model: Any,
    tokenizer: Any,
    cases: list[dict[str, Any]],
    layers: tuple[int, ...],
    ab_ids: dict[str, list[int]],
    batch_size: int,
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[str, dict[str, Any]]]:
    device = model.device
    states: dict[int, dict[str, torch.Tensor]] = {l: {} for l in layers}
    ab_scores: dict[str, dict[str, Any]] = {}
    tag_by_id = {tid: lbl for lbl, tids in ab_ids.items() for tid in tids}

    for start in range(0, len(cases), batch_size):
        batch = cases[start : start + batch_size]
        rendered = [render(tokenizer, c["prompt"]) for c in batch]
        enc = tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        assert bool(enc.attention_mask[:, -1].all()), "Left padding invariant violated!"
        pos = make_position_ids(enc.attention_mask)
        out = model(**enc, position_ids=pos, use_cache=False, output_hidden_states=True, logits_to_keep=1)
        logits = out.logits[:, -1].float()
        probs = torch.softmax(logits, dim=-1)
        pA = probs[:, ab_ids["A"]].sum(-1).tolist()
        pB = probs[:, ab_ids["B"]].sum(-1).tolist()
        margin = (torch.logsumexp(logits[:, ab_ids["A"]], -1) - torch.logsumexp(logits[:, ab_ids["B"]], -1)).tolist()
        greedy = logits.argmax(-1).tolist()
        for i, c in enumerate(batch):
            for l in layers:
                states[l][c["id"]] = out.hidden_states[l][i, -1].float().cpu()
            if c["format"] == "ab":
                ym = margin[i] if c["yes_label"] == "A" else -margin[i]
                gtag = tag_by_id.get(greedy[i], "invalid")
                ab_scores[c["id"]] = {
                    "yes_margin": ym,
                    "greedy_tag": gtag,
                    "ab_prob_mass": pA[i] + pB[i],
                }
    return states, ab_scores


@torch.inference_mode()
def verify_hook_parity(model: Any, tokenizer: Any, sample_prompts: list[str], layers: tuple[int, ...]) -> dict[str, Any]:
    """Verify (1) pre-hook input on layers[L] == out.hidden_states[L] with max_abs_diff==0.0,
    (2) recipient state at position -1 during model.generate prefill matches direct extraction with max_abs_diff==0.0,
    (3) actual zero-delta patch matches native logits & greedy tokens exactly.
    """
    device = model.device
    dec_layers = get_decoder_layers(model)
    rendered = [render(tokenizer, p) for p in sample_prompts]
    enc = tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    pos = make_position_ids(enc.attention_mask)
    native = model(**enc, position_ids=pos, use_cache=False, output_hidden_states=True)
    native_gen = model.generate(**enc, do_sample=False, max_new_tokens=1, pad_token_id=tokenizer.pad_token_id)[:, enc.input_ids.shape[1]:]

    checks = {}
    for l in layers:
        captured_direct = []
        captured_gen = []
        zero_delta = torch.zeros_like(native.hidden_states[l][:, -1, :].float())

        def direct_sham_hook(_mod, args):
            x = args[0]
            captured_direct.append(x.detach().clone())
            x_new = x.clone()
            x_new[:, -1, :] = (x_new[:, -1, :].float() + zero_delta).to(x.dtype)
            return (x_new, *args[1:])

        h = dec_layers[l].register_forward_pre_hook(direct_sham_hook)
        try:
            out_sham = model(**enc, position_ids=pos, use_cache=False)
        finally:
            h.remove()

        def gen_cap_hook(_mod, args):
            x = args[0]
            if x.shape[1] > 1:
                captured_gen.append(x[:, -1, :].detach().clone())
                x_new = x.clone()
                x_new[:, -1, :] = (x_new[:, -1, :].float() + zero_delta).to(x.dtype)
                return (x_new, *args[1:])
            return args

        hg = dec_layers[l].register_forward_pre_hook(gen_cap_hook)
        try:
            gen_sham = model.generate(**enc, do_sample=False, max_new_tokens=1, pad_token_id=tokenizer.pad_token_id)[:, enc.input_ids.shape[1]:]
        finally:
            hg.remove()

        assert len(captured_direct) == 1 and len(captured_gen) == 1
        hs_diff = float((captured_direct[0].float() - native.hidden_states[l].float()).abs().max())
        gen_hs_diff = float((captured_gen[0].float() - native.hidden_states[l][:, -1, :].float()).abs().max())
        logit_diff = float((out_sham.logits[:, -1].float() - native.logits[:, -1].float()).abs().max())
        gen_match = bool(torch.equal(gen_sham, native_gen))
        assert hs_diff == 0.0 and gen_hs_diff == 0.0 and logit_diff == 0.0 and gen_match, (
            f"Hook parity failed at L{l}: hs={hs_diff}, gen_hs={gen_hs_diff}, logit={logit_diff}, gen_match={gen_match}"
        )
        checks[f"L{l}"] = {
            "hs_max_abs_diff": hs_diff,
            "gen_prefill_hs_max_abs_diff": gen_hs_diff,
            "sham_logit_max_abs_diff": logit_diff,
            "sham_gen_exact_match": gen_match,
        }
    return checks


def run_loso_discovery(
    disc_cases: list[dict[str, Any]],
    disc_states: dict[int, dict[str, torch.Tensor]],
    layers: tuple[int, ...],
) -> tuple[dict[str, Any], dict[tuple[int, str, str], dict[str, Any]]]:
    """Leave-one-scenario-out discovery validation + full-discovery frozen probe fits."""
    loso_summary: dict[str, Any] = {}
    frozen_probes: dict[tuple[int, str, str], dict[str, Any]] = {}
    scenarios = [s[0] for s in DISCOVERY_SCENARIOS]

    for l in layers:
        for fmt in ("open", "ab"):
            for task in ("attribution", "dispatch"):
                sub = [c for c in disc_cases if c["format"] == fmt and c["task"] == task and c["clarification"] == "none"]
                # Fit full discovery both-orders probe
                x_all = torch.stack([disc_states[l][c["id"]] for c in sub])
                y_all = torch.tensor([float(c["actual_problem"]) for c in sub])
                center, scale, w, intercept = fit_ridge(x_all, y_all)
                u_hat = w / w.norm().clamp_min(1e-8)
                frozen_probes[(l, fmt, task)] = {
                    "center": center,
                    "scale": scale,
                    "w": w,
                    "u_hat": u_hat,
                    "grand_mean_h": x_all.mean(0),
                }
                # LOSO ranking on discordant pairs (actual_fault > actual_clear, ties wrong)
                order_counts = {o: {"correct": 0, "total": 0} for o in ("practice_first", "actual_first")}
                for held_sc in scenarios:
                    tr = [c for c in sub if c["scenario"] != held_sc]
                    te = [c for c in sub if c["scenario"] == held_sc and c["discordant"]]
                    c_tr, s_tr, w_tr, b_tr = fit_ridge(
                        torch.stack([disc_states[l][c["id"]] for c in tr]),
                        torch.tensor([float(c["actual_problem"]) for c in tr]),
                    )
                    preds = {}
                    for c in te:
                        sc_val = float(((disc_states[l][c["id"]] - c_tr) / s_tr @ w_tr + b_tr).item())
                        preds[(c["order"], c["yes_label"], c["actual_problem"])] = sc_val
                    for o in ("practice_first", "actual_first"):
                        lbls = (None,) if fmt == "open" else ("A", "B")
                        for yl in lbls:
                            p_true = preds[(o, yl, True)]
                            p_false = preds[(o, yl, False)]
                            order_counts[o]["total"] += 1
                            if p_true > p_false:
                                order_counts[o]["correct"] += 1
                pf_acc = order_counts["practice_first"]["correct"] / order_counts["practice_first"]["total"]
                af_acc = order_counts["actual_first"]["correct"] / order_counts["actual_first"]["total"]
                passed = (pf_acc >= 5.0 / 6.0 - 1e-9) and (af_acc >= 5.0 / 6.0 - 1e-9)
                loso_summary[f"L{l}__{fmt}__{task}"] = {
                    "practice_first": order_counts["practice_first"],
                    "actual_first": order_counts["actual_first"],
                    "pf_acc": round(pf_acc, 4),
                    "af_acc": round(af_acc, 4),
                    "loso_gate_passed": passed,
                }
    return loso_summary, frozen_probes


@torch.inference_mode()
def run_patched_batch(
    model: Any,
    tokenizer: Any,
    cases: list[dict[str, Any]],
    layer: int,
    deltas: list[torch.Tensor] | None,
    ab_ids: dict[str, list[int]],
    batch_size: int,
    *,
    do_generate: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply prefill-only single-position residual deltas at model.language_model.model.layers[layer]
    and verify exact patch count, untouched prefix positions, and post-bf16 realized norm/cosine.
    """
    device = model.device
    dec_layer = get_decoder_layers(model)[layer]
    pad_id = tokenizer.pad_token_id
    tag_by_id = {tid: lbl for lbl, tids in ab_ids.items() for tid in tids}

    results: list[dict[str, Any]] = []
    realized_cos: list[float] = []
    realized_rel_norm_err: list[float] = []
    dose_ratios: list[float] = []
    zero_dose_rows = 0
    total_prefill_patches = 0

    for start in range(0, len(cases), batch_size):
        batch = cases[start : start + batch_size]
        rendered = [render(tokenizer, c["prompt"]) for c in batch]
        enc = tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        assert bool(enc.attention_mask[:, -1].all())
        in_len = enc.input_ids.shape[1]

        b_delta = None
        if deltas is not None:
            b_delta = torch.stack(deltas[start : start + len(batch)]).to(device=device, dtype=torch.bfloat16)
            b_delta_f32 = torch.stack(deltas[start : start + len(batch)]).to(device=device, dtype=torch.float32)

        patch_calls = 0

        def pre_hook(_mod, args):
            nonlocal patch_calls, zero_dose_rows, total_prefill_patches
            x = args[0]
            # Prefill-only guard: x.shape[1] > 1 (cached decode steps have seq_len == 1)
            if x.shape[1] > 1 and b_delta is not None:
                patch_calls += 1
                total_prefill_patches += x.shape[0]
                x_new = x.clone()
                before_last = x_new[:, -1, :].float()
                x_new[:, -1, :] = (before_last + b_delta_f32).to(x.dtype)
                # Verify non-target positions are untouched
                assert torch.equal(x_new[:, :-1, :], x[:, :-1, :])
                # Verify post-bf16 realized perturbation
                realized = x_new[:, -1, :].float() - before_last
                for bi in range(x.shape[0]):
                    tn = float(b_delta_f32[bi].norm())
                    rn = float(realized[bi].norm())
                    xn = float(before_last[bi].norm().clamp_min(1e-8))
                    if tn < 1e-6:
                        zero_dose_rows += 1
                        assert rn < 1e-5, f"Nonzero realized diff on zero-dose row: {rn}"
                    else:
                        cos = float(torch.dot(realized[bi], b_delta_f32[bi]) / (rn * tn + 1e-12))
                        realized_cos.append(cos)
                        realized_rel_norm_err.append(abs(rn - tn) / tn)
                        dose_ratios.append(tn / xn)
                return (x_new, *args[1:])
            return args

        h = dec_layer.register_forward_pre_hook(pre_hook) if deltas is not None else None
        eos = model.generation_config.eos_token_id
        stop_ids = set(eos if isinstance(eos, list) else ([eos] if eos is not None else []))
        if tokenizer.eos_token_id is not None:
            stop_ids.add(tokenizer.eos_token_id)
        try:
            if do_generate:
                gen = model.generate(
                    **enc,
                    do_sample=False,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=pad_id,
                )[:, in_len:]
                if deltas is not None:
                    assert patch_calls == 1, f"Expected exactly 1 prefill patch, got {patch_calls}"
                for c, seq in zip(batch, gen.tolist()):
                    trimmed = []
                    hit_eos = False
                    for tid in seq:
                        if tid in stop_ids:
                            hit_eos = True
                            break
                        trimmed.append(tid)
                    txt = tokenizer.decode(trimmed, skip_special_tokens=True).strip()
                    results.append({
                        "id": c["id"],
                        "reply": txt,
                        "n_tokens": len(trimmed),
                        "hit_eos": hit_eos,
                        "truncated": (not hit_eos) and (len(trimmed) >= MAX_NEW_TOKENS),
                    })
            else:
                pos = make_position_ids(enc.attention_mask)
                out = model(**enc, position_ids=pos, use_cache=False, logits_to_keep=1)
                if deltas is not None:
                    assert patch_calls == 1, f"Expected exactly 1 prefill patch, got {patch_calls}"
                logits = out.logits[:, -1].float()
                probs = torch.softmax(logits, dim=-1)
                pA = probs[:, ab_ids["A"]].sum(-1).tolist()
                pB = probs[:, ab_ids["B"]].sum(-1).tolist()
                margin = (torch.logsumexp(logits[:, ab_ids["A"]], -1) - torch.logsumexp(logits[:, ab_ids["B"]], -1)).tolist()
                greedy = logits.argmax(-1).tolist()
                for i, c in enumerate(batch):
                    ym = margin[i] if c["yes_label"] == "A" else -margin[i]
                    results.append({
                        "id": c["id"],
                        "yes_margin": ym,
                        "greedy_tag": tag_by_id.get(greedy[i], "invalid"),
                        "ab_prob_mass": pA[i] + pB[i],
                    })
        finally:
            if h is not None:
                h.remove()

    scos = sorted(realized_cos)
    serr = sorted(realized_rel_norm_err)
    diag = {
        "n_rows": len(cases),
        "total_prefill_patches": total_prefill_patches,
        "zero_dose_rows": zero_dose_rows,
        "min_realized_cos": round(scos[0], 5) if scos else None,
        "median_realized_cos": round(scos[len(scos) // 2], 5) if scos else None,
        "n_cos_below_099": sum(1 for c in scos if c < POST_BF16_COS_TOL),
        "max_realized_rel_norm_err": round(serr[-1], 5) if serr else None,
        "median_realized_rel_norm_err": round(serr[len(serr) // 2], 5) if serr else None,
        "mean_dose_to_state_ratio": round(sum(dose_ratios) / len(dose_ratios), 5) if dose_ratios else None,
        "min_dose_to_state_ratio": round(min(dose_ratios), 5) if dose_ratios else None,
    }
    return results, diag


def make_orthogonal_random(target_delta: torch.Tensor, u_hat: torch.Tensor, seed: int) -> torch.Tensor:
    norm = float(target_delta.norm())
    if norm < 1e-7:
        return torch.zeros_like(target_delta)
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(target_delta.shape, generator=g, dtype=torch.float32)
    r = r - torch.dot(r, u_hat) * u_hat
    r = r - torch.dot(r, target_delta / norm) * (target_delta / norm)
    return r * (norm / float(r.norm().clamp_min(1e-8)))


def main() -> None:
    t0 = time.monotonic()
    ensure_transformers()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert torch.cuda.is_available(), "GPU required."
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    assert torch.cuda.is_bf16_supported(), gpu_name
    bs = 16 if vram_gb < 30 else 32
    print(f"[P2a] GPU={gpu_name} ({vram_gb:.1f} GB), torch={torch.__version__}, bs={bs}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, low_cpu_mem_usage=True,
    ).eval()

    ab_ids = get_ab_label_ids(tokenizer)
    restatement_txt, restatement_tok_info = make_length_matched_restatement(
        tokenizer,
        "Verified cross-check: The actual workshop does need this repair.",
        "Verified cross-check: The actual workshop does not need this repair.",
    )
    print(f"[P2a] Restatement matched: {restatement_tok_info}", flush=True)

    disc_cases = build_cases(DISCOVERY_SCENARIOS, "discover", restatement_txt)
    held_cases = build_cases(NEW_HELDOUT_SCENARIOS, "heldout", restatement_txt)
    inv_cases = build_cases(NEW_HELDOUT_SCENARIOS[:6], "heldout", restatement_txt, inverted_policy=True)

    # 1. Verify exact hook & identity_sham parity
    hook_parity = verify_hook_parity(model, tokenizer, [c["prompt"] for c in disc_cases[:4]], LAYERS)
    print(f"[P2a] Hook parity PASSED: {hook_parity}", flush=True)

    # 2. Extract discovery + heldout states
    disc_states, disc_ab = extract_states_and_ab(model, tokenizer, disc_cases, LAYERS, ab_ids, bs)
    held_states, held_ab = extract_states_and_ab(model, tokenizer, held_cases, LAYERS, ab_ids, bs)
    inv_states, inv_ab = extract_states_and_ab(model, tokenizer, inv_cases, LAYERS, ab_ids, bs)

    # Also generate unpatched discovery open replies (for off-GPU Jev competence/headroom gate!)
    disc_open = [c for c in disc_cases if c["format"] == "open"]
    disc_open_gen, _ = run_patched_batch(model, tokenizer, disc_open, 20, None, ab_ids, bs, do_generate=True)

    # 3. Run LOSO discovery validation gates
    loso_summary, frozen_probes = run_loso_discovery(disc_cases, disc_states, LAYERS)
    print("[P2a] LOSO discovery gates:", json.dumps(loso_summary, indent=2), flush=True)

    # Lookup helpers for donors
    all_held_by_key = {
        (c["policy"], c["format"], c["task"], c["scenario"], c["order"], c["actual_problem"], c["practice_problem"], c["yes_label"], c["clarification"]): c
        for c in (held_cases + inv_cases)
    }
    all_states = {l: {**held_states[l], **inv_states[l]} for l in LAYERS}

    # 4. Run unpatched open baselines (both orders + restatement + oracle)
    held_open_all = [c for c in held_cases if c["format"] == "open"]
    held_open_base_gen, _ = run_patched_batch(model, tokenizer, held_open_all, 20, None, ab_ids, bs, do_generate=True)

    inv_open_all = [c for c in inv_cases if c["format"] == "open"]
    inv_open_base_gen, _ = run_patched_batch(model, tokenizer, inv_open_all, 20, None, ab_ids, bs, do_generate=True)

    # 5. Controlled interventions per layer (L20 primary, L24 secondary) for every (fmt, task) passing LOSO!
    interventions: dict[str, Any] = {}
    for l in LAYERS:
        for fmt in ("open", "ab"):
            for task in ("attribution", "dispatch"):
                gate_key = f"L{l}__{fmt}__{task}"
                if not loso_summary[gate_key]["loso_gate_passed"]:
                    interventions[gate_key] = {"skipped_reason": "failed_loso_discovery_gate"}
                    print(f"[P2a] Skipping {gate_key} (failed LOSO gate)", flush=True)
                    continue
                probe = frozen_probes[(l, fmt, task)]
                u_hat = probe["u_hat"]
                mu0 = probe["grand_mean_h"]
                do_gen = (fmt == "open")

                af_rows = [c for c in held_cases if c["format"] == fmt and c["task"] == task and c["order"] == "actual_first" and c["clarification"] == "none"]
                pf_rows = [c for c in held_cases if c["format"] == fmt and c["task"] == task and c["order"] == "practice_first" and c["clarification"] == "none"]
                af_disc = [c for c in af_rows if c["discordant"]]

                def get_donor_h(c: dict[str, Any], order: str, a: bool, p: bool) -> torch.Tensor:
                    dc = all_held_by_key[(c["policy"], c["format"], c["task"], c["scenario"], order, a, p, c["yes_label"], "none")]
                    return all_states[l][dc["id"]]

                arm_results: dict[str, Any] = {}

                # Build deltas for actual_first rows (all 48 open / 96 ab)
                d_sham = [torch.zeros_like(u_hat) for _ in af_rows]
                d_subspace = [((get_donor_h(c, "practice_first", c["actual_problem"], c["practice_problem"]) - all_states[l][c["id"]]) @ u_hat) * u_hat for c in af_rows]
                d_opp = [-d for d in d_subspace]
                d_act_only = [((get_donor_h(c, "practice_first", not c["actual_problem"], c["practice_problem"]) - all_states[l][c["id"]]) @ u_hat) * u_hat for c in af_rows]
                d_prac_only = [((get_donor_h(c, "practice_first", c["actual_problem"], not c["practice_problem"]) - all_states[l][c["id"]]) @ u_hat) * u_hat for c in af_rows]
                d_full = [get_donor_h(c, "practice_first", c["actual_problem"], c["practice_problem"]) - all_states[l][c["id"]] for c in af_rows]
                d_wrong_disc = [((get_donor_h(c, "practice_first", not c["actual_problem"], not c["practice_problem"]) - all_states[l][c["id"]]) @ u_hat) * u_hat for c in af_disc]

                for aname, rlist, dlist in (
                    ("af_identity_sham", af_rows, d_sham),
                    ("af_subspace_restore", af_rows, d_subspace),
                    ("af_equal_dose_opposite", af_rows, d_opp),
                    ("af_wrong_donor_raw_disc", af_disc, d_wrong_disc),
                    ("af_actual_only_donor", af_rows, d_act_only),
                    ("af_practice_only_donor", af_rows, d_prac_only),
                    ("af_donor_full", af_rows, d_full),
                ):
                    res, diag = run_patched_batch(model, tokenizer, rlist, l, dlist, ab_ids, bs, do_generate=do_gen)
                    arm_results[aname] = {"diag": diag, "rows": res}

                # 3-seed norm-matched orthogonal random controls for af_subspace_restore
                for si, seed in enumerate(RANDOM_SEEDS):
                    d_rand = [make_orthogonal_random(d, u_hat, seed + i) for i, d in enumerate(d_subspace)]
                    res, diag = run_patched_batch(model, tokenizer, af_rows, l, d_rand, ab_ids, bs, do_generate=do_gen)
                    arm_results[f"af_random_s{si}"] = {"diag": diag, "rows": res}

                # Native-use arms on practice_first (all discordant + concordant cells):
                # (1) proj_removal relative to discovery grand mean mu0
                d_proj_rem = [-((all_states[l][c["id"]] - mu0) @ u_hat) * u_hat for c in pf_rows]
                # (2) reverse_shift toward same-assignment actual_first donor along u_hat
                d_rev_shift = [((get_donor_h(c, "actual_first", c["actual_problem"], c["practice_problem"]) - all_states[l][c["id"]]) @ u_hat) * u_hat for c in pf_rows]

                for aname, dlist in (("pf_proj_removal", d_proj_rem), ("pf_reverse_shift", d_rev_shift)):
                    res, diag = run_patched_batch(model, tokenizer, pf_rows, l, dlist, ab_ids, bs, do_generate=do_gen)
                    arm_results[aname] = {"diag": diag, "rows": res}
                    for si, seed in enumerate(RANDOM_SEEDS):
                        d_rand = [make_orthogonal_random(d, u_hat, seed + 1000 + i) for i, d in enumerate(dlist)]
                        r_res, r_diag = run_patched_batch(model, tokenizer, pf_rows, l, d_rand, ab_ids, bs, do_generate=do_gen)
                        arm_results[f"{aname}_random_s{si}"] = {"diag": r_diag, "rows": r_res}

                # Auxiliary inverted-policy dispatch control (L20 open dispatch only)
                if l == 20 and fmt == "open" and task == "dispatch":
                    inv_af = [c for c in inv_cases if c["format"] == "open" and c["order"] == "actual_first"]
                    d_inv_sub = [((get_donor_h(c, "practice_first", c["actual_problem"], c["practice_problem"]) - all_states[l][c["id"]]) @ u_hat) * u_hat for c in inv_af]
                    res, diag = run_patched_batch(model, tokenizer, inv_af, l, d_inv_sub, ab_ids, bs, do_generate=True)
                    arm_results["inv_policy_af_subspace_restore"] = {"diag": diag, "rows": res}

                interventions[gate_key] = arm_results
                print(f"[P2a] Completed {gate_key} (elapsed={time.monotonic()-t0:.0f}s)", flush=True)

    output = {
        "hardware": gpu_name,
        "vram_gb": round(vram_gb, 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "model": MODEL,
        "revision": REVISION,
        "restatement_text": restatement_txt,
        "restatement_tok_info": restatement_tok_info,
        "hook_parity": hook_parity,
        "loso_discovery_gates": loso_summary,
        "cases": {
            "discover": disc_cases,
            "heldout": held_cases,
            "inverted": inv_cases,
        },
        "ab_baselines": {
            "discover": disc_ab,
            "heldout": held_ab,
            "inverted": inv_ab,
        },
        "open_baselines": {
            "discover": disc_open_gen,
            "heldout": held_open_base_gen,
            "inverted": inv_open_base_gen,
        },
        "interventions": interventions,
        "elapsed_sec": round(time.monotonic() - t0, 1),
    }
    out_path = Path("/content/p2a_gemma_restoration.json")
    tmp_out = out_path.with_suffix(".tmp.json")
    tmp_out.write_text(json.dumps(output))
    tmp_out.replace(out_path)
    print(f"[P2a] Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB) in {time.monotonic()-t0:.1f}s", flush=True)
    print("P2A_SUCCESS_MARKER", flush=True)


if __name__ == "__main__":
    main()
