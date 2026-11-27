"""P4: Resolving Gemma-3-4B's Full-Vector Restoration vs. 100%-LOSO 1D Probe Paradox.

Pinned checkpoint: unsloth/gemma-3-4b-it @ bf46152c47f5dd20b896357cb51abc4c03b8ee8c (bfloat16, transformers==4.51.3).
Uses exact 6 discovery + 12 held-out workshop scenarios from P2a with mask-derived position_ids
and exact 0.0 pre-hook & generation-prefill parity verification.

Four surgical tests:
  1. 2x2 Full-Vector Donor Disentangler Matrix across layers L = (4, 8, 12, 16, 20, 24, 28)
     on A/B dispatch & A/B attribution discordant rows:
       - full_same_truth_same_letter:  h_PF(a, p, y)       - h_AF(a, p, y)
       - full_same_truth_flip_letter:  h_PF(a, p, flip(y)) - h_AF(a, p, y)
       - full_wrong_truth_flip_letter: h_PF(!a, !p, y)     - h_AF(a, p, y)
       - full_wrong_truth_same_letter: h_PF(!a, !p, flip(y)) - h_AF(a, p, y)
     Cleanly separates truth-binding transfer from A/B letter-decision readout overwrite!
  2. Subspace Decomposition of Delta = h_PF(a, p, y) - h_AF(a, p, y) at L = (16, 20, 24) on A/B dispatch:
       - Fits u_truth, u_letter, u_order, 3D span, and discovery delta-PCs (rank k=1,2,4,8,16,32)
         STRICTLY on the 6 discovery scenarios (zero held-out leakage).
       - Tests held-out restoration along each subspace, 3D complement, and matched orthogonal random rank-k.
       - Records exact energy fractions and pairwise cosines between u_truth, u_letter, u_order, and PCs.
  3. Open-Attribution Multi-Layer & Aligned Question-Suffix Patching (testing KV-cache re-attendance):
       - Compares af_baseline, pf_baseline, single_tok_L20, single_tok_L16_26,
         question_suffix_L20, question_suffix_L16_24, and wrong-donor question_suffix_L16_24_wrong!
All open replies scored off-GPU by Jev after download.
"""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

import torch


MODEL = "unsloth/gemma-3-4b-it"
REVISION = "bf46152c47f5dd20b896357cb51abc4c03b8ee8c"
TRANSFORMERS_VERSION = "4.51.3"
SWEEP_LAYERS = (4, 8, 12, 16, 20, 24, 28)
SUBSPACE_LAYERS = (16, 20, 24)
OPEN_MULTI_LAYERS = (16, 18, 20, 22, 24)
OPEN_SINGLE_MULTI_LAYERS = (16, 18, 20, 22, 24, 26)
MAX_NEW_TOKENS = 48
PCA_RANKS = (1, 2, 4, 8, 16, 32)
RANDOM_SEEDS = (101, 202, 303)

DISCOVERY_SCENARIOS = (
    ("label printer", "It is jammed and needs a repair.", "It has finished its batch and works."),
    ("shipping scale", "It has stopped reading weight and needs service.", "It was checked and is reading accurately."),
    ("folding table", "Its hinge snapped and needs replacing.", "Its hinge folds smoothly and is intact."),
    ("inventory scanner", "It will not start and requires service.", "It has started and scans every item."),
    ("display case", "Its latch is broken and needs a new part.", "Its latch works and closes securely."),
    ("projector", "Its bulb failed and needs replacing.", "Its picture is bright and stable."),
)

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

OPEN_ATTR_Q = (
    "Which workshop, if either, needs a repair right now: the actual workshop, "
    "the fictional practice workshop, both, or neither? Answer in one short sentence."
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


def make_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    pos = attention_mask.long().cumsum(-1) - 1
    return pos.masked_fill(attention_mask == 0, 1)


def render(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)


def build_cases(scenarios: tuple[tuple[str, str, str], ...], split: str) -> list[dict[str, Any]]:
    cases = []
    for eq, broken, working in scenarios:
        for fmt in ("ab", "open"):
            tasks = ("dispatch", "attribution") if fmt == "ab" else ("attribution",)
            labels = ("A", "B") if fmt == "ab" else (None,)
            for task, order, actual, practice, yl in product(
                tasks, ("practice_first", "actual_first"), (False, True), (False, True), labels
            ):
                act_rep = f"Actual workshop: {broken if actual else working}"
                prac_rep = f"Fictional practice log: {broken if practice else working}"
                reps = [prac_rep, act_rep] if order == "practice_first" else [act_rep, prac_rep]
                if fmt == "open":
                    inst = OPEN_ATTR_Q
                else:
                    q = "Should the technician go to the actual workshop? Dispatch only if it needs a repair." if task == "dispatch" else "Does the repair report apply to the actual workshop?"
                    ltxt = "A means yes and B means no." if yl == "A" else "A means no and B means yes."
                    inst = f"{q} {ltxt} Reply with A or B."
                prefix = f"Ticket item: {eq}. {' '.join(reps)}"
                cid = f"{split}__{fmt}__{task}__{eq}__{order}__a{int(actual)}p{int(practice)}__{yl or 'open'}"
                cases.append({
                    "id": cid,
                    "split": split,
                    "format": fmt,
                    "task": task,
                    "scenario": eq,
                    "order": order,
                    "actual_problem": actual,
                    "practice_problem": practice,
                    "discordant": actual != practice,
                    "yes_label": yl,
                    "prefix": prefix,
                    "question": inst,
                    "prompt": f"{prefix} {inst}",
                })
    return cases


def get_ab_label_ids(tokenizer: Any) -> dict[str, list[int]]:
    ids = {
        lbl: sorted({toks[0] for form in (lbl, " " + lbl) if len(toks := tokenizer.encode(form, add_special_tokens=False)) == 1})
        for lbl in ("A", "B")
    }
    assert all(ids.values()) and not (set(ids["A"]) & set(ids["B"]))
    return ids


def fit_ridge(train_x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    center = train_x.mean(0)
    scale = (train_x - center).square().sum(-1).mean().sqrt().clamp_min(1e-7)
    x = (train_x - center) / scale
    mean_y = target.mean()
    dual = torch.linalg.solve(x @ x.T + 0.1 * torch.eye(len(target), dtype=x.dtype), target - mean_y)
    w = x.T @ dual
    return w / w.norm().clamp_min(1e-8)


def orthonormalize_cols(cols: list[torch.Tensor]) -> torch.Tensor:
    mat = torch.stack(cols, dim=1).float()  # [D, k]
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return q  # [D, k]


def project_onto_basis(delta: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """basis: [D, k] orthonormal columns, delta: [D]."""
    return basis @ (basis.T @ delta)


def make_orth_random_subspace(delta: torch.Tensor, basis: torch.Tensor, seed: int) -> torch.Tensor:
    """Return a vector matched in norm to proj(delta, basis) but orthogonal to basis and delta."""
    proj = project_onto_basis(delta, basis)
    target_norm = float(proj.norm())
    if target_norm < 1e-7:
        return torch.zeros_like(delta)
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(delta.shape, generator=g, dtype=torch.float32)
    r = r - project_onto_basis(r, basis)
    dn = float(delta.norm())
    if dn > 1e-7:
        u_d = delta / dn
        r = r - torch.dot(r, u_d) * u_d
    return r * (target_norm / float(r.norm().clamp_min(1e-8)))


@torch.inference_mode()
def verify_hook_parity(model: Any, tokenizer: Any, sample_prompts: list[str], layers: tuple[int, ...]) -> dict[str, Any]:
    device = model.device
    dec_layers = get_decoder_layers(model)
    rendered = [render(tokenizer, p) for p in sample_prompts]
    enc = tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    pos = make_position_ids(enc.attention_mask)
    native = model(**enc, position_ids=pos, use_cache=False, output_hidden_states=True)
    native_gen = model.generate(**enc, do_sample=False, max_new_tokens=1, pad_token_id=tokenizer.pad_token_id)[:, enc.input_ids.shape[1]:]

    checks = {}
    for l in layers:
        cap_dir, cap_gen = [], []
        zero_d = torch.zeros_like(native.hidden_states[l][:, -1, :].float())

        def h_dir(_m, args):
            x = args[0]
            cap_dir.append(x.detach().clone())
            xn = x.clone()
            xn[:, -1, :] = (xn[:, -1, :].float() + zero_d).to(x.dtype)
            return (xn, *args[1:])

        def h_gen(_m, args):
            x = args[0]
            if x.shape[1] > 1:
                cap_gen.append(x[:, -1, :].detach().clone())
                xn = x.clone()
                xn[:, -1, :] = (xn[:, -1, :].float() + zero_d).to(x.dtype)
                return (xn, *args[1:])
            return args

        h1 = dec_layers[l].register_forward_pre_hook(h_dir)
        try:
            out_sham = model(**enc, position_ids=pos, use_cache=False)
        finally:
            h1.remove()
        h2 = dec_layers[l].register_forward_pre_hook(h_gen)
        try:
            gen_sham = model.generate(**enc, do_sample=False, max_new_tokens=1, pad_token_id=tokenizer.pad_token_id)[:, enc.input_ids.shape[1]:]
        finally:
            h2.remove()

        hs_diff = float((cap_dir[0].float() - native.hidden_states[l].float()).abs().max())
        ghs_diff = float((cap_gen[0].float() - native.hidden_states[l][:, -1, :].float()).abs().max())
        ldiff = float((out_sham.logits[:, -1].float() - native.logits[:, -1].float()).abs().max())
        gmatch = bool(torch.equal(gen_sham, native_gen))
        assert hs_diff == 0.0 and ghs_diff == 0.0 and ldiff == 0.0 and gmatch, f"Parity fail L{l}"
        checks[f"L{l}"] = {"hs": hs_diff, "gen_hs": ghs_diff, "logit": ldiff, "gen_match": gmatch}
    return checks


@torch.inference_mode()
def extract_last_states(
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
                ab_scores[c["id"]] = {
                    "yes_margin": ym,
                    "greedy_tag": tag_by_id.get(greedy[i], "invalid"),
                    "ab_prob_mass": pA[i] + pB[i],
                }
    return states, ab_scores


@torch.inference_mode()
def run_ab_patch(
    model: Any,
    tokenizer: Any,
    cases: list[dict[str, Any]],
    layer: int,
    deltas: list[torch.Tensor],
    ab_ids: dict[str, list[int]],
    batch_size: int,
) -> list[dict[str, Any]]:
    device = model.device
    dec_layer = get_decoder_layers(model)[layer]
    tag_by_id = {tid: lbl for lbl, tids in ab_ids.items() for tid in tids}
    rows_out = []

    for start in range(0, len(cases), batch_size):
        batch = cases[start : start + batch_size]
        rendered = [render(tokenizer, c["prompt"]) for c in batch]
        enc = tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        pos = make_position_ids(enc.attention_mask)
        bd = torch.stack(deltas[start : start + len(batch)]).to(device=device, dtype=torch.float32)

        def pre_hook(_m, args):
            x = args[0]
            xn = x.clone()
            xn[:, -1, :] = (xn[:, -1, :].float() + bd).to(x.dtype)
            return (xn, *args[1:])

        h = dec_layer.register_forward_pre_hook(pre_hook)
        try:
            out = model(**enc, position_ids=pos, use_cache=False, logits_to_keep=1)
        finally:
            h.remove()

        logits = out.logits[:, -1].float()
        probs = torch.softmax(logits, dim=-1)
        pA = probs[:, ab_ids["A"]].sum(-1).tolist()
        pB = probs[:, ab_ids["B"]].sum(-1).tolist()
        margin = (torch.logsumexp(logits[:, ab_ids["A"]], -1) - torch.logsumexp(logits[:, ab_ids["B"]], -1)).tolist()
        greedy = logits.argmax(-1).tolist()
        for i, c in enumerate(batch):
            ym = margin[i] if c["yes_label"] == "A" else -margin[i]
            fc_ok = (ym > 0.0) if c["actual_problem"] else (ym < 0.0)
            gtag = tag_by_id.get(greedy[i], "invalid")
            ttag = c["yes_label"] if c["actual_problem"] else ("B" if c["yes_label"] == "A" else "A")
            rows_out.append({
                "id": c["id"],
                "scenario": c["scenario"],
                "actual_problem": c["actual_problem"],
                "discordant": c["discordant"],
                "yes_label": c["yes_label"],
                "yes_margin": round(ym, 5),
                "fc_correct": fc_ok,
                "greedy_correct": gtag == ttag,
                "greedy_tag": gtag,
                "ab_prob_mass": round(pA[i] + pB[i], 5),
            })
    return rows_out


@torch.inference_mode()
def run_open_span_patch(
    model: Any,
    tokenizer: Any,
    af_cases: list[dict[str, Any]],
    donor_cases: list[dict[str, Any]] | None,
    layers: tuple[int, ...],
    span_mode: str,  # "none" | "last_token" | "question_suffix"
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Patch either the final prompt token OR the entire exact-aligned question+turn suffix across `layers`."""
    device = model.device
    dec_layers = get_decoder_layers(model)
    eos = model.generation_config.eos_token_id
    stop_ids = set(eos if isinstance(eos, list) else ([eos] if eos is not None else []))
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)

    # Determine exact aligned suffix token length shared by all open attribution prompts
    # Prompt ends with: " " + OPEN_ATTR_Q + "<end_of_turn>\n<start_of_turn>model\n"
    suffix_lens = []
    out_rows = []
    for idx, c in enumerate(af_cases):
        r_af = render(tokenizer, c["prompt"])
        enc_af = tokenizer([r_af], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        pos_af = make_position_ids(enc_af.attention_mask)
        ids_af = enc_af.input_ids[0].tolist()

        handles = []
        if span_mode != "none" and donor_cases is not None:
            dc = donor_cases[idx]
            r_dn = render(tokenizer, dc["prompt"])
            enc_dn = tokenizer([r_dn], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            pos_dn = make_position_ids(enc_dn.attention_mask)
            ids_dn = enc_dn.input_ids[0].tolist()

            # Compute exact common token suffix length (guaranteed >= question + assistant header!)
            common_suf = 0
            while common_suf < min(len(ids_af), len(ids_dn)) and ids_af[-1 - common_suf] == ids_dn[-1 - common_suf]:
                common_suf += 1
            patch_len = 1 if span_mode == "last_token" else common_suf
            suffix_lens.append(patch_len)
            assert patch_len >= (1 if span_mode == "last_token" else 30), f"Unexpected short suffix {patch_len}"

            # Extract donor hidden states at all target layers
            out_dn = model(**enc_dn, position_ids=pos_dn, use_cache=False, output_hidden_states=True)
            donor_spans = {l: out_dn.hidden_states[l][:, -patch_len:, :].detach().clone() for l in layers}

            for l in layers:
                d_span = donor_spans[l]

                def make_hook(target_span: torch.Tensor, plen: int):
                    def pre_hook(_m, args):
                        x = args[0]
                        if x.shape[1] > 1:
                            xn = x.clone()
                            xn[:, -plen:, :] = target_span.to(device=x.device, dtype=x.dtype)
                            return (xn, *args[1:])
                        return args
                    return pre_hook

                handles.append(dec_layers[l].register_forward_pre_hook(make_hook(d_span, patch_len)))

        try:
            gen = model.generate(
                **enc_af,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
            )[:, enc_af.input_ids.shape[1]:]
        finally:
            for h in handles:
                h.remove()

        seq = gen[0].tolist()
        trimmed = []
        hit_eos = False
        for tid in seq:
            if tid in stop_ids:
                hit_eos = True
                break
            trimmed.append(tid)
        out_rows.append({
            "id": c["id"],
            "scenario": c["scenario"],
            "actual_problem": c["actual_problem"],
            "practice_problem": c["practice_problem"],
            "discordant": c["discordant"],
            "reply": tokenizer.decode(trimmed, skip_special_tokens=True).strip(),
            "n_tokens": len(trimmed),
            "hit_eos": hit_eos,
        })

    diag = {
        "span_mode": span_mode,
        "layers": list(layers),
        "n_rows": len(out_rows),
        "min_patch_len": min(suffix_lens) if suffix_lens else 0,
        "max_patch_len": max(suffix_lens) if suffix_lens else 0,
    }
    return out_rows, diag


def main() -> None:
    t0 = time.monotonic()
    ensure_transformers()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    assert torch.cuda.is_bf16_supported()
    bs = 32 if vram_gb >= 30 else 16
    print(f"[P4] GPU={gpu_name} ({vram_gb:.1f} GB), bs={bs}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, low_cpu_mem_usage=True,
    ).eval()
    ab_ids = get_ab_label_ids(tok)

    disc_cases = build_cases(DISCOVERY_SCENARIOS, "discover")
    held_cases = build_cases(NEW_HELDOUT_SCENARIOS, "heldout")

    # 1. Exact hook parity across all sweep layers
    parity = verify_hook_parity(model, tok, [c["prompt"] for c in disc_cases[:4]], SWEEP_LAYERS)
    print(f"[P4] Hook parity PASSED across {len(SWEEP_LAYERS)} layers!", flush=True)

    # 2. Extract states across all sweep layers for discovery & heldout A/B cases
    disc_ab_cases = [c for c in disc_cases if c["format"] == "ab"]
    held_ab_cases = [c for c in held_cases if c["format"] == "ab"]
    disc_states, _ = extract_last_states(model, tok, disc_ab_cases, SWEEP_LAYERS, ab_ids, bs)
    held_states, held_ab_base = extract_last_states(model, tok, held_ab_cases, SWEEP_LAYERS, ab_ids, bs)
    print(f"[P4] Extracted states across {len(SWEEP_LAYERS)} layers in {time.monotonic()-t0:.1f}s", flush=True)

    disc_by_key = {(c["task"], c["scenario"], c["order"], c["actual_problem"], c["practice_problem"], c["yes_label"]): c for c in disc_ab_cases}
    held_by_key = {(c["task"], c["scenario"], c["order"], c["actual_problem"], c["practice_problem"], c["yes_label"]): c for c in held_ab_cases}
    flip = lambda y: "B" if y == "A" else "A"

    # 3. TEST 1 & 2: 2x2 Full-Donor Disentangler Matrix across all 7 layers (dispatch + attribution)
    layer_sweep_2x2: dict[str, Any] = {}
    for task in ("dispatch", "attribution"):
        layer_sweep_2x2[task] = {}
        af_disc = [c for c in held_ab_cases if c["task"] == task and c["order"] == "actual_first" and c["discordant"]]
        for l in SWEEP_LAYERS:
            def dvec(c: dict[str, Any], a: bool, p: bool, y: str) -> torch.Tensor:
                dc = held_by_key[(task, c["scenario"], "practice_first", a, p, y)]
                return held_states[l][dc["id"]] - held_states[l][c["id"]]

            arms_2x2 = {
                "sham": [torch.zeros_like(held_states[l][af_disc[0]["id"]]) for _ in af_disc],
                "same_truth_same_letter": [dvec(c, c["actual_problem"], c["practice_problem"], c["yes_label"]) for c in af_disc],
                "same_truth_flip_letter": [dvec(c, c["actual_problem"], c["practice_problem"], flip(c["yes_label"])) for c in af_disc],
                "wrong_truth_flip_letter": [dvec(c, not c["actual_problem"], not c["practice_problem"], c["yes_label"]) for c in af_disc],
                "wrong_truth_same_letter": [dvec(c, not c["actual_problem"], not c["practice_problem"], flip(c["yes_label"])) for c in af_disc],
            }
            l_res = {}
            for aname, dlist in arms_2x2.items():
                rows = run_ab_patch(model, tok, af_disc, l, dlist, ab_ids, bs)
                fc_acc = sum(r["fc_correct"] for r in rows) / len(rows)
                gr_acc = sum(r["greedy_correct"] for r in rows) / len(rows)
                l_res[aname] = {"fc_acc": round(fc_acc, 4), "greedy_acc": round(gr_acc, 4), "rows": rows}
            layer_sweep_2x2[task][f"L{l}"] = l_res
            print(
                f"  [P4 2x2 {task:11s} L{l:2d}] sham={l_res['sham']['fc_acc']:.3f} | "
                f"ST_SL={l_res['same_truth_same_letter']['fc_acc']:.3f} | "
                f"ST_FL={l_res['same_truth_flip_letter']['fc_acc']:.3f} | "
                f"WT_FL={l_res['wrong_truth_flip_letter']['fc_acc']:.3f} | "
                f"WT_SL={l_res['wrong_truth_same_letter']['fc_acc']:.3f}",
                flush=True,
            )

    # 4. TEST 3: Subspace Decomposition of Delta = h_PF(a,p,y) - h_AF(a,p,y) at L16, L20, L24 on A/B dispatch
    subspace_decomp: dict[str, Any] = {}
    af_disp_all = [c for c in held_ab_cases if c["task"] == "dispatch" and c["order"] == "actual_first"]
    disc_disp_all = [c for c in disc_ab_cases if c["task"] == "dispatch"]
    disc_disp_pf = [c for c in disc_disp_all if c["order"] == "practice_first"]
    disc_disp_af = [c for c in disc_disp_all if c["order"] == "actual_first"]

    for l in SUBSPACE_LAYERS:
        # Fit ALL subspaces strictly on the 6 discovery scenarios!
        x_all = torch.stack([disc_states[l][c["id"]] for c in disc_disp_all])
        y_truth = torch.tensor([float(c["actual_problem"]) for c in disc_disp_all])
        u_truth = fit_ridge(x_all, y_truth)

        x_pf = torch.stack([disc_states[l][c["id"]] for c in disc_disp_pf])
        y_letter_pf = torch.tensor([float(c["actual_problem"] == (c["yes_label"] == "A")) for c in disc_disp_pf])
        u_letter = fit_ridge(x_pf, y_letter_pf)

        x_af = torch.stack([disc_states[l][c["id"]] for c in disc_disp_af])
        order_vec = x_pf.mean(0) - x_af.mean(0)
        u_order = order_vec / order_vec.norm().clamp_min(1e-8)

        span_3d = orthonormalize_cols([u_truth, u_letter, u_order])

        # Discovery donor-minus-recipient delta SVD (all 48 discovery AF->PF pairs)
        disc_deltas = torch.stack([
            disc_states[l][disc_by_key[("dispatch", c["scenario"], "practice_first", c["actual_problem"], c["practice_problem"], c["yes_label"])]["id"]]
            - disc_states[l][c["id"]]
            for c in disc_disp_af
        ])  # [48, D]
        _, s_vals, vh_disc = torch.linalg.svd(disc_deltas, full_matrices=False)  # vh_disc: [48, D]

        # Held-out full deltas
        held_deltas = [
            held_states[l][held_by_key[("dispatch", c["scenario"], "practice_first", c["actual_problem"], c["practice_problem"], c["yes_label"])]["id"]]
            - held_states[l][c["id"]]
            for c in af_disp_all
        ]

        # Geometry diagnostics on held-out discordant deltas
        disc_held_deltas = [d for c, d in zip(af_disp_all, held_deltas) if c["discordant"]]
        def mean_energy(basis: torch.Tensor) -> float:
            vals = [float(project_onto_basis(d, basis).square().sum() / d.square().sum().clamp_min(1e-12)) for d in disc_held_deltas]
            return round(sum(vals) / len(vals), 5)

        geom = {
            "cos_truth_letter": round(float(torch.dot(u_truth, u_letter)), 4),
            "cos_truth_order": round(float(torch.dot(u_truth, u_order)), 4),
            "cos_letter_order": round(float(torch.dot(u_letter, u_order)), 4),
            "held_disc_energy_u_truth": mean_energy(u_truth.unsqueeze(1)),
            "held_disc_energy_u_letter": mean_energy(u_letter.unsqueeze(1)),
            "held_disc_energy_u_order": mean_energy(u_order.unsqueeze(1)),
            "held_disc_energy_span_3d": mean_energy(span_3d),
            "held_disc_energy_pcs": {f"k{k}": mean_energy(vh_disc[:k].T) for k in PCA_RANKS},
        }

        arm_deltas: dict[str, list[torch.Tensor]] = {
            "proj_u_truth": [project_onto_basis(d, u_truth.unsqueeze(1)) for d in held_deltas],
            "proj_u_letter": [project_onto_basis(d, u_letter.unsqueeze(1)) for d in held_deltas],
            "proj_u_order": [project_onto_basis(d, u_order.unsqueeze(1)) for d in held_deltas],
            "proj_span_3d": [project_onto_basis(d, span_3d) for d in held_deltas],
            "complement_span_3d": [d - project_onto_basis(d, span_3d) for d in held_deltas],
            "full_delta": held_deltas,
        }
        for k in PCA_RANKS:
            bk = vh_disc[:k].T
            arm_deltas[f"proj_pc_{k}"] = [project_onto_basis(d, bk) for d in held_deltas]
            if k in (1, 4, 16, 32):
                for si, seed in enumerate(RANDOM_SEEDS):
                    arm_deltas[f"rand_pc_{k}_s{si}"] = [make_orth_random_subspace(d, bk, seed + 100 * k + i) for i, d in enumerate(held_deltas)]

        arms_out = {}
        for aname, dlist in arm_deltas.items():
            rows = run_ab_patch(model, tok, af_disp_all, l, dlist, ab_ids, bs)
            d_rows = [r for r in rows if r["discordant"]]
            c_rows = [r for r in rows if not r["discordant"]]
            arms_out[aname] = {
                "disc_fc": round(sum(r["fc_correct"] for r in d_rows) / len(d_rows), 4),
                "conc_fc": round(sum(r["fc_correct"] for r in c_rows) / len(c_rows), 4),
                "rows": rows,
            }
        subspace_decomp[f"L{l}"] = {"geometry": geom, "arms": arms_out}
        print(
            f"  [P4 Subspace L{l}] truth={arms_out['proj_u_truth']['disc_fc']:.3f} | "
            f"letter={arms_out['proj_u_letter']['disc_fc']:.3f} | order={arms_out['proj_u_order']['disc_fc']:.3f} | "
            f"span3={arms_out['proj_span_3d']['disc_fc']:.3f} | comp3={arms_out['complement_span_3d']['disc_fc']:.3f} | "
            f"pc1={arms_out['proj_pc_1']['disc_fc']:.3f} | pc4={arms_out['proj_pc_4']['disc_fc']:.3f} | "
            f"pc16={arms_out['proj_pc_16']['disc_fc']:.3f} | pc32={arms_out['proj_pc_32']['disc_fc']:.3f} | "
            f"full={arms_out['full_delta']['disc_fc']:.3f}",
            flush=True,
        )

    # 5. TEST 4: Open-Attribution Multi-Layer & Aligned Question-Suffix Patching (KV-cache hypothesis!)
    held_open = [c for c in held_cases if c["format"] == "open"]
    open_by_key = {(c["scenario"], c["order"], c["actual_problem"], c["practice_problem"]): c for c in held_open}
    af_open_disc = [c for c in held_open if c["order"] == "actual_first" and c["discordant"]]
    pf_open_disc = [open_by_key[(c["scenario"], "practice_first", c["actual_problem"], c["practice_problem"])] for c in af_open_disc]
    pf_open_wrong = [open_by_key[(c["scenario"], "practice_first", not c["actual_problem"], not c["practice_problem"])] for c in af_open_disc]

    open_kv_arms: dict[str, Any] = {}
    for aname, recs, donors, layers, smode in (
        ("af_baseline", af_open_disc, None, (), "none"),
        ("pf_baseline", pf_open_disc, None, (), "none"),
        ("single_tok_L20", af_open_disc, pf_open_disc, (20,), "last_token"),
        ("single_tok_L16_26", af_open_disc, pf_open_disc, OPEN_SINGLE_MULTI_LAYERS, "last_token"),
        ("question_suffix_L20", af_open_disc, pf_open_disc, (20,), "question_suffix"),
        ("question_suffix_L16_24", af_open_disc, pf_open_disc, OPEN_MULTI_LAYERS, "question_suffix"),
        ("question_suffix_L16_24_wrong_donor", af_open_disc, pf_open_wrong, OPEN_MULTI_LAYERS, "question_suffix"),
    ):
        rows, diag = run_open_span_patch(model, tok, recs, donors, layers, smode)
        open_kv_arms[aname] = {"diag": diag, "rows": rows}
        print(f"  [P4 Open KV] done {aname} (diag={diag})", flush=True)

    output = {
        "hardware": gpu_name,
        "vram_gb": round(vram_gb, 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "model": MODEL,
        "revision": REVISION,
        "hook_parity": parity,
        "open_cases": af_open_disc + pf_open_disc,
        "ab_baselines_heldout": held_ab_base,
        "layer_sweep_2x2": layer_sweep_2x2,
        "subspace_decomp": subspace_decomp,
        "open_kv_patching": open_kv_arms,
        "elapsed_sec": round(time.monotonic() - t0, 1),
    }
    out_path = Path("/content/p4_gemma_paradox.json")
    tmp = out_path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(output))
    tmp.replace(out_path)
    print(f"[P4] Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB) in {time.monotonic()-t0:.1f}s", flush=True)
    print("P4_SUCCESS_MARKER", flush=True)


if __name__ == "__main__":
    main()
