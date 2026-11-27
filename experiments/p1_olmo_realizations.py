"""P1: OLMo-2 7B Multi-Realization Action-Preference & Sparsity x Timing Experiment.

Runs on Colab GPU (G4, A100, or L4) in bfloat16 with transformers==4.51.3.
Loads SFT backbone + ONLY the lm_head safetensors shard of Base (never two 7B models).
Computes exact float32 [4096, 4096] Gram eigendecomposition of DeltaW = W_sft - W_base
cross-checked against torch.svd_lowrank (q=16, niter=4, seed=0, rel_frob_err < 1e-4).
Runs:
  1. Native SFT & physical Base-head swap parity assertions (logits + greedy tokens).
  2. Representative cold-load + warm-batch throughput projection.
  3. Full Teacher-Forcing Grid (all 3 policies x both prefills x 4 symmetric prose
     continuations with fact-consistent reason x all head/sparsity/timing & random arms).
  4. Lean, checkpointed Autoregressive Generation (4 core timing arms across forward+reversed
     x all 3 wordings x both prefills + no_rule on label_free; plus sparsity x timing +
     matched random-timing arms on forward x label_free(none/verdict) & action_clear(none)).
All Jev semantic scoring runs off-GPU locally after downloading the output JSON.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


BASE_MODEL = "allenai/OLMo-2-1124-7B"
BASE_REV = "7df9a82518afdecae4e8c026b27adccc8c1f0032"
SFT_MODEL = "allenai/OLMo-2-1124-7B-SFT"
SFT_REV = "1de02c0175118a9de5854aec80a1f970e701e928"
TOKENIZER_MODEL = "allenai/OLMo-2-1124-7B-Instruct"
TOKENIZER_REV = "470b1fba1ae01581f270116362ee4aa1b97f4c84"
TRANSFORMERS_VERSION = "4.51.3"

# Frozen 36 opener token IDs from results/exp10_olmo_head.json
ARCHIVED_OPENER_IDS = [
    39, 40, 358, 578, 791, 2122, 2360, 2409, 2822, 3161, 4032, 5810,
    5832, 7566, 8586, 9642, 10771, 10835, 11439, 17512, 17902, 18001,
    19173, 19701, 20817, 23371, 23588, 29815, 31140, 33386, 35755,
    39794, 40914, 42544, 48527, 70621,
]
MAX_NEW_TOKENS = 48
SVD_RANK = 4
SVD_Q = 32
SVD_NITER = 8
SVD_SEED = 0
SVD_REL_TOL = 1e-3
WALL_BUDGET_SEC = 2700  # 45 min cap for P1 inside 60 min P1+P2a session budget


def ensure_transformers() -> None:
    try:
        installed = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed != TRANSFORMERS_VERSION:
        if "transformers" in sys.modules:
            raise RuntimeError(f"transformers=={installed} already imported; restart kernel.")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", f"transformers=={TRANSFORMERS_VERSION}"])


def load_base_lm_head(device: str) -> torch.Tensor:
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    index_path = hf_hub_download(BASE_MODEL, "model.safetensors.index.json", revision=BASE_REV)
    weight_map = json.loads(Path(index_path).read_text())["weight_map"]
    shard_name = weight_map["lm_head.weight"]
    shard_path = hf_hub_download(BASE_MODEL, shard_name, revision=BASE_REV)
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        w = f.get_tensor("lm_head.weight").to(device=device, dtype=torch.bfloat16)
    return w


def build_head_variants(w_base: torch.Tensor, w_sft: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Build exact head weight matrices & verify rank-4 SVD via float32 [4096, 4096] Gram eigendecomp."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.monotonic()
    dw_f32 = (w_sft.float() - w_base.float())  # [100352, 4096] on GPU
    V, D = dw_f32.shape
    assert V == 100352 and D == 4096, (V, D)

    # Exact top-4 via Gram matrix G = dW^T dW in float32 ([4096, 4096])
    gram = dw_f32.T @ dw_f32
    evals, evecs = torch.linalg.eigh(gram)
    top_idx = torch.argsort(evals, descending=True)[:SVD_RANK]
    v_top = evecs[:, top_idx]  # [D, 4]
    dw_rank4_exact = (dw_f32 @ v_top) @ v_top.T  # [V, D]
    top4_sv = torch.sqrt(evals[top_idx].clamp_min(0.0)).tolist()

    # Exact algebraic check on dw_rank4_exact + randomized svd_lowrank cross-check
    u_exact = dw_f32 @ v_top  # [V, 4], columns should be orthogonal with norms sqrt(evals[top_idx])
    ut_u = u_exact.T @ u_exact
    exact_diag_err = float((ut_u - torch.diag(evals[top_idx])).norm() / evals[top_idx].norm().clamp_min(1e-8))
    if not (math.isfinite(exact_diag_err) and exact_diag_err < 1e-5):
        raise RuntimeError(f"Exact Gram eigendecomp residual too large: {exact_diag_err}")

    # Cross-check against torch.svd_lowrank with frozen seed, q, niter
    gen = torch.Generator(device=dw_f32.device).manual_seed(SVD_SEED)
    torch.manual_seed(SVD_SEED)
    u_lr, s_lr, v_lr = torch.svd_lowrank(dw_f32, q=SVD_Q, niter=SVD_NITER)
    dw_rank4_lr = (u_lr[:, :SVD_RANK] * s_lr[:SVD_RANK]) @ v_lr[:, :SVD_RANK].T
    rel_frob_err = float((dw_rank4_exact - dw_rank4_lr).norm() / dw_rank4_exact.norm())
    gram_sec = time.monotonic() - t0
    if not (math.isfinite(rel_frob_err) and rel_frob_err < SVD_REL_TOL):
        raise RuntimeError(f"SVD verification failed: rel_frob_err={rel_frob_err} >= {SVD_REL_TOL}")

    full_frob = float(dw_f32.norm())
    rank4_frob = float(dw_rank4_exact.norm())
    opener_idx = torch.tensor(ARCHIVED_OPENER_IDS, device=dw_f32.device, dtype=torch.long)
    mask_opener = torch.zeros(V, device=dw_f32.device, dtype=torch.bool)
    mask_opener[opener_idx] = True
    assert int(mask_opener.sum()) == 36 and int((~mask_opener).sum()) == 100316

    w_opener36 = w_base.clone()
    w_opener36[opener_idx] = w_sft[opener_idx]
    w_comp36 = w_sft.clone()
    w_comp36[opener_idx] = w_base[opener_idx]

    # Row-count & norm-matched random 36 rows outside opener36 (seed=0)
    non_opener = torch.nonzero(~mask_opener, as_tuple=False).squeeze(-1)
    perm = torch.randperm(len(non_opener), generator=gen, device=dw_f32.device)
    rand36_idx = non_opener[perm[:36]]
    opener_dw_norm = float(dw_f32[opener_idx].norm())
    rand36_dw = dw_f32[rand36_idx]
    rand36_scale = opener_dw_norm / float(rand36_dw.norm().clamp_min(1e-8))
    w_rand36 = w_base.clone()
    w_rand36[rand36_idx] = (w_base[rand36_idx].float() + rand36_dw * rand36_scale).to(torch.bfloat16)

    # Frobenius-norm-matched random rank-4 update orthogonal to top-4 right singular vectors
    r_a = torch.randn((V, SVD_RANK), generator=gen, device=dw_f32.device, dtype=torch.float32)
    r_b = torch.randn((D, SVD_RANK), generator=gen, device=dw_f32.device, dtype=torch.float32)
    r_b = r_b - v_top @ (v_top.T @ r_b)
    dw_rand4 = r_a @ r_b.T
    dw_rand4 = dw_rand4 * (rank4_frob / float(dw_rand4.norm().clamp_min(1e-8)))

    w_svd4 = (w_base.float() + dw_rank4_exact).to(torch.bfloat16)
    w_rand_rank4 = (w_base.float() + dw_rand4).to(torch.bfloat16)

    variants = {
        "base": w_base,
        "sft": w_sft,
        "opener36": w_opener36,
        "complement36": w_comp36,
        "rand36": w_rand36,
        "svd4": w_svd4,
        "rand_rank4": w_rand_rank4,
    }
    stats = {
        "vocab_size": V,
        "hidden_size": D,
        "n_opener_rows": 36,
        "n_complement_rows": 100316,
        "full_dw_frob": full_frob,
        "opener36_dw_frob": opener_dw_norm,
        "opener36_energy_frac": (opener_dw_norm ** 2) / (full_frob ** 2),
        "rank4_dw_frob": rank4_frob,
        "rank4_energy_frac": (rank4_frob ** 2) / (full_frob ** 2),
        "top4_singular_values": top4_sv,
        "svd_lowrank_q": SVD_Q,
        "svd_lowrank_niter": SVD_NITER,
        "svd_seed": SVD_SEED,
        "tf32_disabled": True,
        "rel_frob_err_exact_vs_lowrank": rel_frob_err,
        "exact_gram_diag_rel_err": exact_diag_err,
        "gram_verify_sec": round(gram_sec, 3),
        "rand36_ids": rand36_idx.cpu().tolist(),
        "rand36_scale": rand36_scale,
    }
    del dw_f32, gram, dw_rank4_exact, dw_rank4_lr, dw_rand4
    torch.cuda.empty_cache()
    return variants, stats


def render_prompt(tokenizer: Any, row: dict[str, Any]) -> str:
    msgs = [
        {"role": "system", "content": row["system"]},
        {"role": "user", "content": row["user"]},
    ]
    base = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return base + row["prefill"]


def make_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    pos = attention_mask.long().cumsum(-1) - 1
    return pos.masked_fill(attention_mask == 0, 0)


@torch.inference_mode()
def generate_arm(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    w_first: torch.Tensor,
    w_later: torch.Tensor,
    batch_size: int,
    *,
    use_hook: bool = True,
    return_token_ids: bool = False,
) -> list[dict[str, Any]]:
    device = model.device
    pad_id = tokenizer.pad_token_id
    eos = model.generation_config.eos_token_id
    stop_ids = set(eos if isinstance(eos, list) else ([eos] if eos is not None else []))
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    head = model.lm_head
    out: list[dict[str, Any]] = []

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        in_len = enc.input_ids.shape[1]
        calls = 0

        def hook(_mod, args, _native_out):
            nonlocal calls
            w = w_first if calls == 0 else w_later
            calls += 1
            return F.linear(args[0], w)

        h = head.register_forward_hook(hook) if use_hook else None
        try:
            gen_ids = model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=pad_id,
            )[:, in_len:]
        finally:
            if h is not None:
                h.remove()

        for seq in gen_ids.tolist():
            raw_first = seq[0] if seq else None
            trimmed = []
            thru_eos = []
            hit_eos = False
            for tid in seq:
                thru_eos.append(tid)
                if tid in stop_ids:
                    hit_eos = True
                    break
                trimmed.append(tid)
            rec: dict[str, Any] = {
                "reply": tokenizer.decode(trimmed, skip_special_tokens=True).strip(),
                "first_token_id": raw_first,
                "n_tokens": len(trimmed),
                "hit_eos": hit_eos,
                "truncated": (not hit_eos) and (len(trimmed) >= MAX_NEW_TOKENS),
            }
            if return_token_ids:
                rec["token_ids_thru_eos"] = thru_eos
            out.append(rec)
    return out


@torch.inference_mode()
def verify_parity(model: Any, tokenizer: Any, sample_prompts: list[str], variants: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Verify (a) native SFT vs routed sft/sft & manual backbone+head, and (b) physical Base-head swap vs routed base/base."""
    device = model.device
    enc = tokenizer(sample_prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    pos = make_position_ids(enc.attention_mask)

    # (a) Native SFT vs routed sft/sft & direct backbone last_hidden_state -> F.linear
    native_sft_logits = model(**enc, position_ids=pos, use_cache=False).logits[:, -1].float()
    bb_h_3d = model.model(**enc, position_ids=pos, use_cache=False).last_hidden_state
    manual_sft_logits = F.linear(bb_h_3d, variants["sft"])[:, -1].float()
    bb_max_diff = float((native_sft_logits - manual_sft_logits).abs().max())

    h = model.lm_head.register_forward_hook(lambda _m, a, _o: F.linear(a[0], variants["sft"]))
    try:
        routed_sft_logits = model(**enc, position_ids=pos, use_cache=False).logits[:, -1].float()
    finally:
        h.remove()
    sft_max_diff = float((native_sft_logits - routed_sft_logits).abs().max())
    gen_native_sft = generate_arm(model, tokenizer, sample_prompts, variants["sft"], variants["sft"], len(sample_prompts), use_hook=False, return_token_ids=True)
    gen_routed_sft = generate_arm(model, tokenizer, sample_prompts, variants["sft"], variants["sft"], len(sample_prompts), use_hook=True, return_token_ids=True)
    sft_tok_match = [x["token_ids_thru_eos"] for x in gen_native_sft] == [x["token_ids_thru_eos"] for x in gen_routed_sft]
    assert sft_max_diff == 0.0 and bb_max_diff == 0.0 and sft_tok_match, f"SFT parity failed: sft={sft_max_diff}, bb={bb_max_diff}, tok={sft_tok_match}"

    # (b) Physical Base-head swap on SFT backbone vs routed base/base
    orig_w = model.lm_head.weight.data.clone()
    try:
        model.lm_head.weight.data.copy_(variants["base"])
        phys_base_logits = model(**enc, position_ids=pos, use_cache=False).logits[:, -1].float()
        gen_phys_base = generate_arm(model, tokenizer, sample_prompts, variants["base"], variants["base"], len(sample_prompts), use_hook=False, return_token_ids=True)
    finally:
        model.lm_head.weight.data.copy_(orig_w)

    h = model.lm_head.register_forward_hook(lambda _m, a, _o: F.linear(a[0], variants["base"]))
    try:
        routed_base_logits = model(**enc, position_ids=pos, use_cache=False).logits[:, -1].float()
    finally:
        h.remove()
    base_max_diff = float((phys_base_logits - routed_base_logits).abs().max())
    gen_routed_base = generate_arm(model, tokenizer, sample_prompts, variants["base"], variants["base"], len(sample_prompts), use_hook=True, return_token_ids=True)
    base_tok_match = [x["token_ids_thru_eos"] for x in gen_phys_base] == [x["token_ids_thru_eos"] for x in gen_routed_base]
    assert base_max_diff == 0.0 and base_tok_match, f"Base parity failed: base={base_max_diff}, tok={base_tok_match}"

    return {
        "n_prompts": len(sample_prompts),
        "backbone_head_max_abs_diff": bb_max_diff,
        "sft_logit_max_abs_diff": sft_max_diff,
        "sft_greedy_exact_token_match": True,
        "base_logit_max_abs_diff": base_max_diff,
        "base_greedy_exact_token_match": True,
    }


@torch.inference_mode()
def run_teacher_forcing_grid(
    model: Any,
    tokenizer: Any,
    manifest: dict[str, Any],
    variants: dict[str, torch.Tensor],
    tf_arms: list[tuple[str, str, str]],
    batch_size: int,
) -> dict[str, Any]:
    """Score 4 symmetric prose continuations (with fact-consistent reason + EOS) under label_free
    across all 3 policies and both prefills, encoding complete prompt+candidate in context.
    """
    device = model.device
    eos_id = model.generation_config.eos_token_id
    if isinstance(eos_id, list):
        eos_id = eos_id[0]
    opener_set = set(ARCHIVED_OPENER_IDS)

    lf_rows = [r for r in manifest["rows"] if r["wording"] == "label_free"]
    reals = manifest["realizations"]

    # Build in-context encoded items and verify exact common prefix & divergence tokens
    items = []
    divergence_audit = {}
    for r in lf_rows:
        prompt = render_prompt(tokenizer, r)
        # Ensure clean boundary without trailing-space retokenization
        p_ids = tokenizer.encode(prompt, add_special_tokens=False)
        for real in reals:
            rname = real["name"]
            sep = " " if r["prefill"] else ""
            c_rel = f"{sep}{real['release_prefix']} {r['reason']}"
            c_ret = f"{sep}{real['retain_prefix']} {r['reason']}"
            ids_rel = tokenizer.encode(prompt + c_rel, add_special_tokens=False) + [eos_id]
            ids_ret = tokenizer.encode(prompt + c_ret, add_special_tokens=False) + [eos_id]
            assert ids_rel[: len(p_ids)] == p_ids, f"Prefix retokenized (rel): {r['row_id']} {rname}"
            assert ids_ret[: len(p_ids)] == p_ids, f"Prefix retokenized (ret): {r['row_id']} {rname}"
            cont_rel = ids_rel[len(p_ids) :]
            cont_ret = ids_ret[len(p_ids) :]
            div_idx = 0
            while div_idx < min(len(cont_rel), len(cont_ret)) and cont_rel[div_idx] == cont_ret[div_idx]:
                div_idx += 1
            assert div_idx < min(len(cont_rel), len(cont_ret)), f"No divergence: {r['row_id']} {rname}"
            rel_div_tok = cont_rel[div_idx]
            ret_div_tok = cont_ret[div_idx]
            key = f"{rname}__{r['prefill_name']}"
            if key not in divergence_audit:
                divergence_audit[key] = {
                    "first_gen_rel_id": cont_rel[0],
                    "first_gen_ret_id": cont_ret[0],
                    "first_div_offset": div_idx,
                    "first_div_rel_id": rel_div_tok,
                    "first_div_ret_id": ret_div_tok,
                    "first_div_rel_str": tokenizer.decode([rel_div_tok]),
                    "first_div_ret_str": tokenizer.decode([ret_div_tok]),
                    "rel_in_opener36": rel_div_tok in opener_set,
                    "ret_in_opener36": ret_div_tok in opener_set,
                }
            items.append((r, rname, "release", ids_rel, len(p_ids), div_idx))
            items.append((r, rname, "retain", ids_ret, len(p_ids), div_idx))

    # Forward pass once per unique prompt for exact shared-prefix boundary states (div_idx == 0)
    uniq_weights = sorted({wf for _, wf, _ in tf_arms} | {wl for _, _, wl in tf_arms})
    prefix_h_by_row: dict[str, torch.Tensor] = {}
    for start in range(0, len(lf_rows), batch_size):
        p_batch = lf_rows[start : start + batch_size]
        p_txts = [render_prompt(tokenizer, r) for r in p_batch]
        p_enc = tokenizer(p_txts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        p_pos = make_position_ids(p_enc.attention_mask)
        p_h = model.model(**p_enc, position_ids=p_pos, use_cache=False).last_hidden_state[:, -1, :]
        for i, r in enumerate(p_batch):
            prefix_h_by_row[r["row_id"]] = p_h[i].detach().cpu()

    # Forward pass once per item through SFT backbone with mask-derived position_ids
    scores: dict[str, dict[tuple[str, str], dict[str, dict[str, float]]]] = {
        arm_name: defaultdict(dict) for arm_name, _, _ in tf_arms
    }

    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        enc = tokenizer.pad(
            [{"input_ids": x[3]} for x in batch],
            return_tensors="pt",
        ).to(device)
        pos = make_position_ids(enc.attention_mask)
        # Left padding: sequence of length L sits at columns [W - L : W]
        hidden = model.model(**enc, position_ids=pos, use_cache=False).last_hidden_state  # [B, W, D]
        W = enc.input_ids.shape[1]

        # Extract post-norm hidden states predicting each continuation token
        per_sample_h = []
        per_sample_targets = []
        for i, (r, _, _, full_ids, p_len, div_idx) in enumerate(batch):
            L = len(full_ids)
            pad_offset = W - L
            # Continuation token j (0-indexed) is at position pad_offset + p_len + j,
            # predicted by hidden state at position pad_offset + p_len + j - 1
            h_slice = hidden[i, pad_offset + p_len - 1 : pad_offset + L - 1, :].clone()  # [cont_len, D]
            # Anchor position 0 (prompt boundary) to exact single shared-prefix forward state
            h_slice[0] = prefix_h_by_row[r["row_id"]].to(device=device, dtype=h_slice.dtype)
            t_slice = torch.tensor(full_ids[p_len:], device=device, dtype=torch.long)
            per_sample_h.append(h_slice)
            per_sample_targets.append(t_slice)

        # Concatenate all token states once per batch for ultra-fast matrix multiply under each head!
        flat_h = torch.cat(per_sample_h, dim=0)  # [N_tok, D]
        flat_t = torch.cat(per_sample_targets, dim=0)  # [N_tok]
        lengths = [len(t) for t in per_sample_targets]

        # Precompute token log-probs under each unique weight matrix once
        tok_lp_by_w: dict[str, torch.Tensor] = {}
        for wname in uniq_weights:
            logits = F.linear(flat_h, variants[wname]).float()
            lse = torch.logsumexp(logits, dim=-1)
            gathered = logits.gather(-1, flat_t.unsqueeze(-1)).squeeze(-1)
            tok_lp_by_w[wname] = (gathered - lse).cpu()

        # Assemble per-arm full-seq, length-normalized, and first-div log-probs
        for arm_name, wf, wl in tf_arms:
            lp_f = tok_lp_by_w[wf]
            lp_l = tok_lp_by_w[wl]
            cursor = 0
            for (r, rname, side, _, _, div_idx), clen in zip(batch, lengths):
                sl_f = lp_f[cursor : cursor + clen]
                sl_l = lp_l[cursor : cursor + clen]
                cursor += clen
                # Token 0 uses wf, tokens 1.. use wl
                comb = torch.cat([sl_f[:1], sl_l[1:]]) if clen > 1 else sl_f[:1]
                full_lp = float(comb.sum())
                mean_lp = full_lp / clen
                div_lp = float(comb[div_idx])
                scores[arm_name][(r["row_id"], rname)][side] = {
                    "full_lp": full_lp,
                    "mean_lp": mean_lp,
                    "div_lp": div_lp,
                    "n_tokens": clen,
                }

    # Verify untouched-row parity on first-divergence log-odds whenever both divergence tokens not in opener36
    untouched_parity_max_diff = 0.0
    for r in lf_rows:
        for real in reals:
            rname = real["name"]
            aud = divergence_audit[f"{rname}__{r['prefill_name']}"]
            if (not aud["rel_in_opener36"]) and (not aud["ret_in_opener36"]) and aud["first_div_offset"] == 0:
                b_m = scores["base/base"][(r["row_id"], rname)]["retain"]["div_lp"] - scores["base/base"][(r["row_id"], rname)]["release"]["div_lp"]
                o_m = scores["opener36/opener36"][(r["row_id"], rname)]["retain"]["div_lp"] - scores["opener36/opener36"][(r["row_id"], rname)]["release"]["div_lp"]
                untouched_parity_max_diff = max(untouched_parity_max_diff, abs(b_m - o_m))

    # Serialize compact per-row margins
    out_rows: dict[str, list[dict[str, Any]]] = {}
    for arm_name, _, _ in tf_arms:
        arm_list = []
        for r in lf_rows:
            for real in reals:
                rname = real["name"]
                sc = scores[arm_name][(r["row_id"], rname)]
                arm_list.append({
                    "row_id": r["row_id"],
                    "pair_id": r["pair_id"],
                    "family": r["family"],
                    "policy": r["policy"],
                    "fact_key": r["fact_key"],
                    "allowed": r["allowed"],
                    "prefill_name": r["prefill_name"],
                    "realization": rname,
                    "m_full": sc["retain"]["full_lp"] - sc["release"]["full_lp"],
                    "m_mean": sc["retain"]["mean_lp"] - sc["release"]["mean_lp"],
                    "m_div": sc["retain"]["div_lp"] - sc["release"]["div_lp"],
                })
        out_rows[arm_name] = arm_list

    return {
        "divergence_audit": divergence_audit,
        "untouched_opener36_div_parity_max_abs_diff": untouched_parity_max_diff,
        "arms": out_rows,
    }


def main() -> None:
    t_start = time.monotonic()
    ensure_transformers()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert torch.cuda.is_available(), "GPU required."
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    assert torch.cuda.is_bf16_supported(), gpu_name
    gen_bs = 48 if vram_gb >= 60 else (32 if vram_gb >= 30 else 16)
    tf_bs = 32 if vram_gb >= 30 else 16
    print(f"[P1] GPU={gpu_name} ({vram_gb:.1f} GB), torch={torch.__version__}, gen_bs={gen_bs}", flush=True)

    manifest_path = Path("/content/p1_cases_manifest.json")
    if not manifest_path.exists():
        manifest_path = Path(__file__).resolve().parent / "p1_cases_manifest.json"
    manifest = json.loads(manifest_path.read_text())

    t_load0 = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, revision=TOKENIZER_REV, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        SFT_MODEL,
        revision=SFT_REV,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
    ).eval()
    w_sft = model.lm_head.weight.data.detach().clone()
    w_base = load_base_lm_head("cuda:0")
    load_sec = time.monotonic() - t_load0

    variants, head_stats = build_head_variants(w_base, w_sft)
    print(f"[P1] Loaded & built head variants in {load_sec:.1f}s; SVD rel_err={head_stats['rel_frob_err_exact_vs_lowrank']:.2e}", flush=True)

    # 1. Explicit Parity Checks
    sample_rows = [r for r in manifest["rows"] if r["policy"] == "forward" and r["wording"] == "label_free"][:8]
    sample_prompts = [render_prompt(tokenizer, r) for r in sample_rows]
    parity = verify_parity(model, tokenizer, sample_prompts, variants)
    print(f"[P1] Parity PASSED: {parity}", flush=True)

    # 2. Warm-batch throughput calibration
    t_warm0 = time.monotonic()
    _ = generate_arm(model, tokenizer, sample_prompts, variants["sft"], variants["sft"], gen_bs)
    warm_sec = time.monotonic() - t_warm0
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"[P1] Warm 8-prompt gen={warm_sec:.2f}s ({8/max(warm_sec,1e-3):.1f} prompts/s), peak_vram={peak_vram_gb:.2f} GB", flush=True)

    # 3. Full Teacher-Forcing Grid (all 16 head/sparsity/timing & timing-matched random arms)
    tf_arms = [
        ("base/base", "base", "base"),
        ("sft/base", "sft", "base"),
        ("base/sft", "base", "sft"),
        ("sft/sft", "sft", "sft"),
        ("opener36/base", "opener36", "base"),
        ("base/opener36", "base", "opener36"),
        ("opener36/opener36", "opener36", "opener36"),
        ("complement36/complement36", "complement36", "complement36"),
        ("rand36/base", "rand36", "base"),
        ("base/rand36", "base", "rand36"),
        ("rand36/rand36", "rand36", "rand36"),
        ("svd4/base", "svd4", "base"),
        ("base/svd4", "base", "svd4"),
        ("svd4/svd4", "svd4", "svd4"),
        ("rand_rank4/base", "rand_rank4", "base"),
        ("base/rand_rank4", "base", "rand_rank4"),
        ("rand_rank4/rand_rank4", "rand_rank4", "rand_rank4"),
    ]
    t_tf0 = time.monotonic()
    tf_results = run_teacher_forcing_grid(model, tokenizer, manifest, variants, tf_arms, tf_bs)
    tf_sec = time.monotonic() - t_tf0
    print(f"[P1] Full teacher-forcing grid done in {tf_sec:.1f}s (untouched_parity={tf_results['untouched_opener36_div_parity_max_abs_diff']:.2e})", flush=True)

    # 4. Lean Autoregressive Generation Schedule (checkpointed per cell)
    shard_dir = Path("/content/shards/p1")
    shard_dir.mkdir(parents=True, exist_ok=True)

    core_4 = [("base/base", "base", "base"), ("sft/base", "sft", "base"), ("base/sft", "base", "sft"), ("sft/sft", "sft", "sft")]
    sparsity_extra = [
        ("opener36/base", "opener36", "base"),
        ("base/opener36", "base", "opener36"),
        ("opener36/opener36", "opener36", "opener36"),
        ("complement36/complement36", "complement36", "complement36"),
        ("rand36/base", "rand36", "base"),
        ("base/rand36", "base", "rand36"),
        ("rand36/rand36", "rand36", "rand36"),
        ("svd4/base", "svd4", "base"),
        ("base/svd4", "base", "svd4"),
        ("svd4/svd4", "svd4", "svd4"),
        ("rand_rank4/base", "rand_rank4", "base"),
        ("base/rand_rank4", "base", "rand_rank4"),
        ("rand_rank4/rand_rank4", "rand_rank4", "rand_rank4"),
    ]

    # Build exact generation jobs: (cell_key, rows, arms)
    jobs: list[tuple[str, list[dict[str, Any]], list[tuple[str, str, str]]]] = []
    by_cell: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in manifest["rows"]:
        by_cell[(r["policy"], r["wording"], r["prefill_name"])].append(r)

    # Core 4 timing arms on forward + reversed x all 3 wordings x both prefills, plus no_rule on label_free
    for policy in ("forward", "reversed"):
        for wording in ("label_free", "action_clear", "opaque_code"):
            for pref in ("none", "verdict"):
                ck = f"{policy}__{wording}__{pref}"
                arms = list(core_4)
                if policy == "forward" and ((wording == "label_free") or (wording == "action_clear" and pref == "none")):
                    arms = arms + sparsity_extra
                jobs.append((ck, by_cell[(policy, wording, pref)], arms))
    for pref in ("none", "verdict"):
        ck = f"no_rule__label_free__{pref}"
        jobs.append((ck, by_cell[("no_rule", "label_free", pref)], list(core_4)))

    total_gens = sum(len(rows) * len(arms) for _, rows, arms in jobs)
    est_sec = total_gens / max(8 / max(warm_sec, 0.1), 1.0)
    print(f"[P1] Generation schedule: {len(jobs)} cells, {total_gens} replies (~{est_sec/60:.1f} min est)", flush=True)

    generations: dict[str, dict[str, list[dict[str, Any]]]] = {}
    opener_set = set(ARCHIVED_OPENER_IDS)
    first_token_coverage: dict[str, dict[str, float]] = {}

    revs = {"base": BASE_REV, "sft": SFT_REV, "tokenizer": TOKENIZER_REV}
    for ck, rows, arms in jobs:
        shard_path = shard_dir / f"{ck}.json"
        expected_meta = {
            "manifest_sha256": manifest["manifest_sha256"],
            "revisions": revs,
            "arms": [a[0] for a in arms],
            "row_ids": [r["row_id"] for r in rows],
        }
        if shard_path.exists():
            shard_obj = json.loads(shard_path.read_text())
            if isinstance(shard_obj, dict) and shard_obj.get("meta") == expected_meta:
                cell_out = shard_obj["data"]
                for arm_name, _, _ in arms:
                    ftoks = [g["first_token_id"] for g in cell_out[arm_name]]
                    cov = sum(1 for t in ftoks if t in opener_set) / len(ftoks)
                    first_token_coverage.setdefault(ck, {})[arm_name] = round(cov, 4)
                generations[ck] = cell_out
                print(f"  [resume] {ck}", flush=True)
                continue
            print(f"  [stale-shard] recomputing {ck}", flush=True)
        if time.monotonic() - t_start > WALL_BUDGET_SEC:
            raise RuntimeError(f"P1 exceeded wall budget ({WALL_BUDGET_SEC}s) before {ck}")

        prompts = [render_prompt(tokenizer, r) for r in rows]
        cell_out = {}
        first_tok_by_first_head: dict[str, list[int | None]] = {}
        for arm_name, wf, wl in arms:
            res = generate_arm(model, tokenizer, prompts, variants[wf], variants[wl], gen_bs)
            ftoks = [x["first_token_id"] for x in res]
            # Assert arms sharing the same first-token head emit identical first generated token!
            if wf in first_tok_by_first_head:
                assert ftoks == first_tok_by_first_head[wf], f"First-token mismatch in {ck} arm {arm_name}"
            else:
                first_tok_by_first_head[wf] = ftoks
            cov = sum(1 for t in ftoks if t in opener_set) / len(ftoks)
            first_token_coverage.setdefault(ck, {})[arm_name] = round(cov, 4)
            cell_out[arm_name] = [
                {"row_id": r["row_id"], **g}
                for r, g in zip(rows, res)
            ]
        tmp_shard = shard_path.with_suffix(".tmp.json")
        tmp_shard.write_text(json.dumps({"meta": expected_meta, "data": cell_out}))
        tmp_shard.replace(shard_path)
        generations[ck] = cell_out
        print(f"  [done] {ck} ({len(arms)} arms, elapsed={time.monotonic()-t_start:.0f}s)", flush=True)

    out = {
        "hardware": gpu_name,
        "vram_gb": round(vram_gb, 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "revisions": revs,
        "manifest_sha256": manifest["manifest_sha256"],
        "head_stats": head_stats,
        "parity": parity,
        "timings_sec": {
            "load": round(load_sec, 1),
            "warm_8": round(warm_sec, 2),
            "teacher_forcing": round(tf_sec, 1),
            "total": round(time.monotonic() - t_start, 1),
        },
        "first_token_opener36_coverage": first_token_coverage,
        "teacher_forcing": tf_results,
        "generations": generations,
    }
    out_path = Path("/content/p1_olmo_realizations.json")
    tmp_out = out_path.with_suffix(".tmp.json")
    tmp_out.write_text(json.dumps(out))
    tmp_out.replace(out_path)
    print(f"[P1] Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB) in {time.monotonic()-t_start:.1f}s", flush=True)
    print("P1_SUCCESS_MARKER", flush=True)


if __name__ == "__main__":
    main()
