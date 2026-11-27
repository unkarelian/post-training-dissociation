"""P3: Cross-Family, RL-Stage, Row-vs-Rank, and Reason-First Generalization of Token-0 lm_head Calibration.

Tests across 5 untied-head models (all pinned revisions, bfloat16, transformers==4.51.3):
  1. OLMo-2-7B-SFT  (+ Base, SFT, DPO, Instruct lm_head shards; action_rows vs except_action_rows vs
                     SVD rank-4/16/64 vs rand_rank16; + reason_first escape hatch)
  2. OLMo-2-7B-Instruct (+ Base lm_head shard; full TF + core timing + action_rows + svd16 + reason_first)
  3. Llama-3.1-8B-Instruct (+ Base lm_head shard; full TF + core timing + action_rows + svd16 + reason_first)
  4. Mistral-7B-Instruct-v0.3 (+ Base lm_head shard; full TF + core timing + action_rows + svd16 + reason_first)
  5. Qwen2.5-7B-Instruct (+ Base lm_head shard; full TF + core timing + action_rows + svd16 + reason_first)

Memory & disk safe:
  - Uses chunked float32 Gram eigendecomposition (never allocates full [V, D] float32 copy on GPU),
    keeping peak VRAM <= ~20 GB (runs on L4 24GB, A100 40/80GB, or G4 96GB).
  - Clears HF cache between model families so Colab disk never fills up.
  - All Jev semantic scoring runs off-GPU locally after download.
"""

from __future__ import annotations

import gc
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


TRANSFORMERS_VERSION = "4.51.3"
MAX_NEW_TOKENS = 48
REASON_FIRST_MAX_NEW_TOKENS = 72
SVD_SEED = 0

PINNED = {
    "olmo2_base": ("allenai/OLMo-2-1124-7B", "7df9a82518afdecae4e8c026b27adccc8c1f0032"),
    "olmo2_sft": ("allenai/OLMo-2-1124-7B-SFT", "1de02c0175118a9de5854aec80a1f970e701e928"),
    "olmo2_dpo": ("allenai/OLMo-2-1124-7B-DPO", "e34ea60adff2e575f4fe7569eaffd1b28509b6fd"),
    "olmo2_inst": ("allenai/OLMo-2-1124-7B-Instruct", "470b1fba1ae01581f270116362ee4aa1b97f4c84"),
    "llama_base": ("unsloth/Meta-Llama-3.1-8B", "e9a141a2091ea561b96483212645a2a05e6f99fc"),
    "llama_inst": ("unsloth/Meta-Llama-3.1-8B-Instruct", "a2856192dd7c25b842431f39c179a6c2c2f627d1"),
    "mistral_base": ("unsloth/mistral-7b-v0.3", "9ea1b83f5ced5663c5fa89c300fe59f9bdcd2b10"),
    "mistral_inst": ("unsloth/mistral-7b-instruct-v0.3", "60b99dcbc7f853b7cd18472a607e8d045b96f03e"),
    "qwen_base": ("Qwen/Qwen2.5-7B", "d149729398750b98c0af14eb82c78cfe92750796"),
    "qwen_inst": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
}

ACTION_WORDS = (
    "Send", "Keep", "Store", "Hold", "APPROVE", "BLOCK", "Approve", "Block",
    "send", "keep", "store", "hold", "approve", "block",
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


def make_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    pos = attention_mask.long().cumsum(-1) - 1
    return pos.masked_fill(attention_mask == 0, 1)


def clear_hf_cache() -> None:
    cache_dir = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)


def load_lm_head_only(model_id: str, revision: str, device: str) -> torch.Tensor:
    """Download and extract ONLY lm_head.weight from safetensors or bin index without loading full model."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    try:
        idx_path = hf_hub_download(model_id, "model.safetensors.index.json", revision=revision)
        wmap = json.loads(Path(idx_path).read_text())["weight_map"]
        shard = hf_hub_download(model_id, wmap["lm_head.weight"], revision=revision)
        with safe_open(shard, framework="pt", device="cpu") as f:
            return f.get_tensor("lm_head.weight").to(device=device, dtype=torch.bfloat16)
    except Exception:
        idx_path = hf_hub_download(model_id, "pytorch_model.bin.index.json", revision=revision)
        wmap = json.loads(Path(idx_path).read_text())["weight_map"]
        shard = hf_hub_download(model_id, wmap["lm_head.weight"], revision=revision)
        sd = torch.load(shard, map_location="cpu", weights_only=True)
        w = sd["lm_head.weight"].to(device=device, dtype=torch.bfloat16)
        del sd
        return w


def get_action_row_ids(tokenizer: Any) -> tuple[list[int], dict[str, list[int]]]:
    """Collect single-token vocabulary IDs for task action words (with/without leading space)."""
    by_word: dict[str, list[int]] = {}
    all_ids: set[int] = set()
    for w in ACTION_WORDS:
        tids = set()
        for form in (w, " " + w):
            enc = tokenizer.encode(form, add_special_tokens=False)
            if len(enc) == 1:
                tids.add(enc[0])
            elif len(enc) >= 1 and w in ("APPROVE", "BLOCK"):
                # Also capture the opening subword of APPROVE/BLOCK if split across 2 tokens!
                tids.add(enc[0])
        if tids:
            by_word[w] = sorted(tids)
            all_ids.update(tids)
    return sorted(all_ids), by_word


def build_head_variants_chunked(
    w_base: torch.Tensor,
    w_post: torch.Tensor,
    action_ids: list[int],
    ranks: tuple[int, ...] = (16,),
    *,
    include_rand_rank: int | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Compute exact Gram eigendecomposition of dW = w_post - w_base in row chunks (low VRAM!)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = w_base.device
    V, D = w_base.shape
    max_r = max(ranks)
    chunk = 16384

    gram = torch.zeros((D, D), device=device, dtype=torch.float32)
    full_sq = 0.0
    for s in range(0, V, chunk):
        dw_c = w_post[s : s + chunk].float() - w_base[s : s + chunk].float()
        gram += dw_c.T @ dw_c
        full_sq += float(dw_c.square().sum())
    full_frob = math.sqrt(max(full_sq, 1e-12))

    evals, evecs = torch.linalg.eigh(gram)
    top_idx = torch.argsort(evals, descending=True)[:max_r]
    v_top_all = evecs[:, top_idx]  # [D, max_r]
    sv_all = torch.sqrt(evals[top_idx].clamp_min(0.0)).tolist()

    variants: dict[str, torch.Tensor] = {"base": w_base, "post": w_post}

    # Action-word rows only & complement
    act_idx = torch.tensor(action_ids, device=device, dtype=torch.long)
    w_act = w_base.clone()
    w_act[act_idx] = w_post[act_idx]
    w_exc = w_post.clone()
    w_exc[act_idx] = w_base[act_idx]
    act_sq = float((w_post[act_idx].float() - w_base[act_idx].float()).square().sum())
    variants["action_rows"] = w_act
    variants["except_action_rows"] = w_exc

    rank_energy: dict[str, float] = {}
    for r in ranks:
        vr = v_top_all[:, :r]
        w_r = torch.empty_like(w_base)
        r_sq = 0.0
        for s in range(0, V, chunk):
            wb = w_base[s : s + chunk].float()
            dw_c = w_post[s : s + chunk].float() - wb
            proj = (dw_c @ vr) @ vr.T
            r_sq += float(proj.square().sum())
            w_r[s : s + chunk] = (wb + proj).to(torch.bfloat16)
        variants[f"svd{r}"] = w_r
        rank_energy[f"svd{r}"] = round(r_sq / max(full_sq, 1e-12), 5)

    if include_rand_rank is not None:
        r = include_rand_rank
        vr = v_top_all[:, :r]
        target_frob = math.sqrt(sum(s ** 2 for s in sv_all[:r]))
        gen = torch.Generator(device=device).manual_seed(SVD_SEED)
        r_b = torch.randn((D, r), generator=gen, device=device, dtype=torch.float32)
        r_b = r_b - vr @ (vr.T @ r_b)
        r_b, _ = torch.linalg.qr(r_b)
        # Draw chunked left factor r_a with exact target Frobenius norm
        r_a_chunks = []
        raw_sq = 0.0
        for s in range(0, V, chunk):
            ra = torch.randn((min(chunk, V - s), r), generator=gen, device=device, dtype=torch.float32)
            raw_sq += float(ra.square().sum())
            r_a_chunks.append(ra)
        scale = target_frob / math.sqrt(max(raw_sq, 1e-12))
        w_rr = torch.empty_like(w_base)
        for idx_c, s in enumerate(range(0, V, chunk)):
            wb = w_base[s : s + chunk].float()
            dw_rand = (r_a_chunks[idx_c] * scale) @ r_b.T
            w_rr[s : s + chunk] = (wb + dw_rand).to(torch.bfloat16)
        variants[f"rand_rank{r}"] = w_rr
        del r_a_chunks

    del gram
    torch.cuda.empty_cache()
    stats = {
        "vocab_size": V,
        "hidden_size": D,
        "full_dw_frob": round(full_frob, 4),
        "n_action_rows": len(action_ids),
        "action_rows_energy_frac": round(act_sq / max(full_sq, 1e-12), 6),
        "rank_energy_frac": rank_energy,
        "top8_singular_values": [round(x, 4) for x in sv_all[:8]],
    }
    return variants, stats


def render_chat(tokenizer: Any, system: str, user: str, prefill: str) -> tuple[str, str]:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        base = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        mode = "system_role"
    except Exception:
        base = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{system}\n\n{user}"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        mode = "merged_user"
    return base + prefill, mode


def build_reason_first_rows(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Build reason-before-verdict rows on the exact same 12 families & 48 pairs."""
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    # Use forward & reversed label_free/none and action_clear/none base rows as templates
    for r in manifest["rows"]:
        if r["prefill_name"] != "none" or r["policy"] not in ("forward", "reversed"):
            continue
        if r["wording"] not in ("label_free", "action_clear"):
            continue
        # Extract the rule sentence from r["system"]
        # system format: "You process warehouse tickets. {rule_txt} {fmt_inst}"
        body = r["system"].removeprefix("You process warehouse tickets. ")
        rule_txt = body.split(". ")[0] + "."
        if r["wording"] == "label_free":
            inst = (
                "First write 'Fact:' followed by one short clause stating the relevant parcel attribute "
                "and rule condition. Then write 'Decision:' followed by whether the clerk should send "
                "the parcel out or keep it in storage."
            )
            ck = f"{r['policy']}__reason_first_lf"
        else:
            inst = (
                "First write 'Fact:' followed by one short clause stating the relevant parcel attribute "
                "and rule condition. Then write 'Decision:' followed by APPROVE (to send out) or BLOCK "
                "(to keep in storage)."
            )
            ck = f"{r['policy']}__reason_first_ac"
        out[ck].append({
            **r,
            "row_id": f"rf__{r['row_id']}",
            "system": f"You process warehouse tickets. {rule_txt} {inst}",
            "prefill": "",
            "prefill_name": "none",
        })
    return dict(out)


@torch.inference_mode()
def generate_arm(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    w_first: torch.Tensor,
    w_later: torch.Tensor,
    batch_size: int,
    max_new_tokens: int = MAX_NEW_TOKENS,
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
                max_new_tokens=max_new_tokens,
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
                "truncated": (not hit_eos) and (len(trimmed) >= max_new_tokens),
            }
            if return_token_ids:
                rec["token_ids_thru_eos"] = thru_eos
            out.append(rec)
    return out


@torch.inference_mode()
def verify_parity(model: Any, tokenizer: Any, sample_prompts: list[str], w_base: torch.Tensor, w_post: torch.Tensor) -> dict[str, Any]:
    device = model.device
    enc = tokenizer(sample_prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    pos = make_position_ids(enc.attention_mask)

    native_logits = model(**enc, position_ids=pos, use_cache=False).logits[:, -1].float()
    bb_h = model.model(**enc, position_ids=pos, use_cache=False).last_hidden_state
    bb_diff = float((native_logits - F.linear(bb_h, w_post)[:, -1].float()).abs().max())

    h = model.lm_head.register_forward_hook(lambda _m, a, _o: F.linear(a[0], w_post))
    try:
        routed_logits = model(**enc, position_ids=pos, use_cache=False).logits[:, -1].float()
    finally:
        h.remove()
    post_diff = float((native_logits - routed_logits).abs().max())
    g_nat = generate_arm(model, tokenizer, sample_prompts, w_post, w_post, len(sample_prompts), use_hook=False, return_token_ids=True)
    g_rou = generate_arm(model, tokenizer, sample_prompts, w_post, w_post, len(sample_prompts), use_hook=True, return_token_ids=True)
    assert post_diff == 0.0 and bb_diff == 0.0 and [x["token_ids_thru_eos"] for x in g_nat] == [x["token_ids_thru_eos"] for x in g_rou]

    orig_w = model.lm_head.weight.data.clone()
    try:
        model.lm_head.weight.data.copy_(w_base)
        phys_base = model(**enc, position_ids=pos, use_cache=False).logits[:, -1].float()
        g_pb = generate_arm(model, tokenizer, sample_prompts, w_base, w_base, len(sample_prompts), use_hook=False, return_token_ids=True)
    finally:
        model.lm_head.weight.data.copy_(orig_w)

    h = model.lm_head.register_forward_hook(lambda _m, a, _o: F.linear(a[0], w_base))
    try:
        rou_base = model(**enc, position_ids=pos, use_cache=False).logits[:, -1].float()
    finally:
        h.remove()
    base_diff = float((phys_base - rou_base).abs().max())
    g_rb = generate_arm(model, tokenizer, sample_prompts, w_base, w_base, len(sample_prompts), use_hook=True, return_token_ids=True)
    assert base_diff == 0.0 and [x["token_ids_thru_eos"] for x in g_pb] == [x["token_ids_thru_eos"] for x in g_rb]
    return {"post_diff": post_diff, "bb_diff": bb_diff, "base_diff": base_diff, "exact_tok_match": True}


@torch.inference_mode()
def run_teacher_forcing_4arms(
    model: Any,
    tokenizer: Any,
    manifest: dict[str, Any],
    w_base: torch.Tensor,
    w_post: torch.Tensor,
    batch_size: int,
) -> dict[str, Any]:
    """Teacher-forcing across the 4 non-opener prose continuations on label_free/none (fwd, rev, no_rule)."""
    device = model.device
    eos_id = model.generation_config.eos_token_id
    if isinstance(eos_id, list):
        eos_id = eos_id[0]
    if eos_id is None:
        eos_id = tokenizer.eos_token_id

    lf_rows = [r for r in manifest["rows"] if r["wording"] == "label_free" and r["prefill_name"] == "none"]
    reals = manifest["realizations"]

    # Exact shared-prefix boundary states
    prefix_h: dict[str, torch.Tensor] = {}
    prompt_cache: dict[str, tuple[str, list[int]]] = {}
    for r in lf_rows:
        ptxt, _ = render_chat(tokenizer, r["system"], r["user"], "")
        pids = tokenizer.encode(ptxt, add_special_tokens=False)
        prompt_cache[r["row_id"]] = (ptxt, pids)

    for s in range(0, len(lf_rows), batch_size):
        pb = lf_rows[s : s + batch_size]
        enc = tokenizer([prompt_cache[r["row_id"]][0] for r in pb], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        pos = make_position_ids(enc.attention_mask)
        ph = model.model(**enc, position_ids=pos, use_cache=False).last_hidden_state[:, -1, :]
        for i, r in enumerate(pb):
            prefix_h[r["row_id"]] = ph[i].detach().cpu()

    items = []
    retok_skipped = 0
    for r in lf_rows:
        ptxt, pids = prompt_cache[r["row_id"]]
        for real in reals:
            rname = real["name"]
            c_rel = f"{real['release_prefix']} {r['reason']}"
            c_ret = f"{real['retain_prefix']} {r['reason']}"
            ids_rel = tokenizer.encode(ptxt + c_rel, add_special_tokens=False) + [eos_id]
            ids_ret = tokenizer.encode(ptxt + c_ret, add_special_tokens=False) + [eos_id]
            if ids_rel[: len(pids)] != pids or ids_ret[: len(pids)] != pids:
                # Some tokenizers (SentencePiece/Mistral) may merge boundary; fallback to separate encoding
                ids_rel = pids + tokenizer.encode(c_rel, add_special_tokens=False) + [eos_id]
                ids_ret = pids + tokenizer.encode(c_ret, add_special_tokens=False) + [eos_id]
                retok_skipped += 1
            cont_rel = ids_rel[len(pids) :]
            cont_ret = ids_ret[len(pids) :]
            div_idx = 0
            while div_idx < min(len(cont_rel), len(cont_ret)) and cont_rel[div_idx] == cont_ret[div_idx]:
                div_idx += 1
            items.append((r, rname, "release", ids_rel, len(pids), div_idx))
            items.append((r, rname, "retain", ids_ret, len(pids), div_idx))

    arms = [("base/base", "base", "base"), ("post/base", "post", "base"), ("base/post", "base", "post"), ("post/post", "post", "post")]
    weights = {"base": w_base, "post": w_post}
    scores: dict[str, dict[tuple[str, str], dict[str, dict[str, float]]]] = {a: defaultdict(dict) for a, _, _ in arms}

    for s in range(0, len(items), batch_size):
        batch = items[s : s + batch_size]
        enc = tokenizer.pad([{"input_ids": x[3]} for x in batch], return_tensors="pt").to(device)
        pos = make_position_ids(enc.attention_mask)
        hidden = model.model(**enc, position_ids=pos, use_cache=False).last_hidden_state
        W = enc.input_ids.shape[1]

        per_h, per_t = [], []
        for i, (r, _, _, full_ids, p_len, _) in enumerate(batch):
            L = len(full_ids)
            pad_off = W - L
            hs = hidden[i, pad_off + p_len - 1 : pad_off + L - 1, :].clone()
            hs[0] = prefix_h[r["row_id"]].to(device=device, dtype=hs.dtype)
            per_h.append(hs)
            per_t.append(torch.tensor(full_ids[p_len:], device=device, dtype=torch.long))

        flat_h = torch.cat(per_h, dim=0)
        flat_t = torch.cat(per_t, dim=0)
        lens = [len(t) for t in per_t]

        lp_w = {}
        for wn, wmat in weights.items():
            logits = F.linear(flat_h, wmat).float()
            lse = torch.logsumexp(logits, dim=-1)
            lp_w[wn] = (logits.gather(-1, flat_t.unsqueeze(-1)).squeeze(-1) - lse).cpu()

        for aname, wf, wl in arms:
            lpf, lpl = lp_w[wf], lp_w[wl]
            cur = 0
            for (r, rname, side, _, _, div_idx), clen in zip(batch, lens):
                sf, sl = lpf[cur : cur + clen], lpl[cur : cur + clen]
                cur += clen
                comb = torch.cat([sf[:1], sl[1:]]) if clen > 1 else sf[:1]
                flp = float(comb.sum())
                scores[aname][(r["row_id"], rname)][side] = {
                    "full_lp": flp,
                    "div_lp": float(comb[div_idx]),
                }

    out_arms = {}
    for aname, _, _ in arms:
        rows_out = []
        for r in lf_rows:
            for real in reals:
                rname = real["name"]
                sc = scores[aname][(r["row_id"], rname)]
                rows_out.append({
                    "row_id": r["row_id"],
                    "pair_id": r["pair_id"],
                    "family": r["family"],
                    "policy": r["policy"],
                    "fact_key": r["fact_key"],
                    "allowed": r["allowed"],
                    "realization": rname,
                    "m_full": sc["retain"]["full_lp"] - sc["release"]["full_lp"],
                    "m_div": sc["retain"]["div_lp"] - sc["release"]["div_lp"],
                })
        out_arms[aname] = rows_out
    return {"boundary_fallback_count": retok_skipped, "arms": out_arms}


def run_model_suite(
    model_key: str,
    backbone_id: str,
    backbone_rev: str,
    tok_id: str,
    tok_rev: str,
    base_id: str,
    base_rev: str,
    manifest: dict[str, Any],
    rf_rows: dict[str, list[dict[str, Any]]],
    vram_gb: float,
    *,
    extra_heads: dict[str, tuple[str, str]] | None = None,
    svd_ranks: tuple[int, ...] = (16,),
    include_rand_rank: int | None = None,
    run_tf: bool = True,
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.monotonic()
    gen_bs = 48 if vram_gb >= 60 else (32 if vram_gb >= 30 else 16)
    tf_bs = 32 if vram_gb >= 30 else 12
    print(f"\n[P3:{model_key}] Loading {backbone_id}@{backbone_rev[:7]} (bs={gen_bs})...", flush=True)

    tok = AutoTokenizer.from_pretrained(tok_id, revision=tok_rev, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        backbone_id,
        revision=backbone_rev,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
    ).eval()
    assert not bool(getattr(model.config, "tie_word_embeddings", False)), f"{model_key} has tied lm_head!"

    w_post = model.lm_head.weight.data.detach().clone()
    w_base = load_lm_head_only(base_id, base_rev, "cuda:0")
    assert w_base.shape == w_post.shape, (w_base.shape, w_post.shape)

    action_ids, action_by_word = get_action_row_ids(tok)
    variants, head_stats = build_head_variants_chunked(
        w_base, w_post, action_ids, ranks=svd_ranks, include_rand_rank=include_rand_rank
    )
    if extra_heads:
        for hname, (hid, hrev) in extra_heads.items():
            variants[hname] = load_lm_head_only(hid, hrev, "cuda:0")

    by_cell: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in manifest["rows"]:
        by_cell[(r["policy"], r["wording"], r["prefill_name"])].append(r)

    sample_rows = by_cell[("forward", "label_free", "none")][:8]
    sample_prompts_modes = [render_chat(tok, r["system"], r["user"], r["prefill"]) for r in sample_rows]
    sample_prompts = [p for p, _ in sample_prompts_modes]
    chat_mode = sample_prompts_modes[0][1]
    parity = verify_parity(model, tok, sample_prompts, w_base, w_post)
    print(f"[P3:{model_key}] Parity PASSED ({parity}), chat_mode={chat_mode}, action_rows={len(action_ids)}", flush=True)

    tf_res = run_teacher_forcing_4arms(model, tok, manifest, w_base, w_post, tf_bs) if run_tf else None
    if tf_res is not None:
        print(f"[P3:{model_key}] TF done (elapsed={time.monotonic()-t0:.0f}s)", flush=True)

    core4 = [("base/base", "base", "base"), ("post/base", "post", "base"), ("base/post", "base", "post"), ("post/post", "post", "post")]
    extra_tok0 = [("action_rows/base", "action_rows", "base"), ("except_action_rows/base", "except_action_rows", "base")]
    for r in svd_ranks:
        extra_tok0.append((f"svd{r}/base", f"svd{r}", "base"))
    if include_rand_rank is not None:
        extra_tok0.append((f"rand_rank{include_rand_rank}/base", f"rand_rank{include_rand_rank}", "base"))
    if extra_heads:
        for hname in extra_heads:
            extra_tok0.append((f"{hname}/base", hname, "base"))
            extra_tok0.append((f"{hname}/{hname}", hname, hname))

    gen_jobs: list[tuple[str, list[dict[str, Any]], list[tuple[str, str, str]], int]] = [
        ("forward__label_free__none", by_cell[("forward", "label_free", "none")], core4 + extra_tok0, MAX_NEW_TOKENS),
        ("forward__label_free__verdict", by_cell[("forward", "label_free", "verdict")], core4 + extra_tok0, MAX_NEW_TOKENS),
        ("forward__action_clear__none", by_cell[("forward", "action_clear", "none")], core4 + extra_tok0, MAX_NEW_TOKENS),
        ("reversed__label_free__none", by_cell[("reversed", "label_free", "none")], core4, MAX_NEW_TOKENS),
        ("reversed__action_clear__none", by_cell[("reversed", "action_clear", "none")], core4, MAX_NEW_TOKENS),
        ("no_rule__label_free__none", by_cell[("no_rule", "label_free", "none")], core4, MAX_NEW_TOKENS),
        ("forward__reason_first_ac", rf_rows["forward__reason_first_ac"], core4, REASON_FIRST_MAX_NEW_TOKENS),
    ]
    if model_key == "olmo2_sft":
        gen_jobs.extend([
            ("forward__reason_first_lf", rf_rows["forward__reason_first_lf"], core4, REASON_FIRST_MAX_NEW_TOKENS),
            ("reversed__reason_first_ac", rf_rows["reversed__reason_first_ac"], core4, REASON_FIRST_MAX_NEW_TOKENS),
        ])

    generations: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for ck, rows, arms, max_tok in gen_jobs:
        prompts = [render_chat(tok, r["system"], r["user"], r["prefill"])[0] for r in rows]
        cell_out = {}
        ft_by_wf: dict[str, list[int | None]] = {}
        for aname, wf, wl in arms:
            res = generate_arm(model, tok, prompts, variants[wf], variants[wl], gen_bs, max_new_tokens=max_tok)
            ftoks = [x["first_token_id"] for x in res]
            if wf in ft_by_wf:
                assert ftoks == ft_by_wf[wf], f"First-token mismatch in {model_key}/{ck}/{aname}"
            else:
                ft_by_wf[wf] = ftoks
            cell_out[aname] = [{"row_id": r["row_id"], **g} for r, g in zip(rows, res)]
        generations[ck] = cell_out
        print(f"  [P3:{model_key}] done {ck} ({len(arms)} arms, elapsed={time.monotonic()-t0:.0f}s)", flush=True)

    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    del model, variants, w_base, w_post
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    return {
        "model_key": model_key,
        "backbone_id": backbone_id,
        "backbone_rev": backbone_rev,
        "base_id": base_id,
        "base_rev": base_rev,
        "chat_template_mode": chat_mode,
        "action_row_ids": action_ids,
        "action_by_word": action_by_word,
        "head_stats": head_stats,
        "parity": parity,
        "peak_vram_gb": round(peak_gb, 2),
        "elapsed_sec": round(time.monotonic() - t0, 1),
        "teacher_forcing": tf_res,
        "generations": generations,
    }


def main() -> None:
    t_all = time.monotonic()
    ensure_transformers()
    assert torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    assert torch.cuda.is_bf16_supported(), gpu_name
    print(f"[P3] GPU={gpu_name} ({vram_gb:.1f} GB), torch={torch.__version__}", flush=True)

    manifest_path = Path("/content/p1_cases_manifest.json")
    if not manifest_path.exists():
        manifest_path = Path(__file__).resolve().parent / "p1_cases_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rf_rows = build_reason_first_rows(manifest)

    shard_dir = Path("/content/shards/p3")
    shard_dir.mkdir(parents=True, exist_ok=True)

    suite_specs = [
        {
            "model_key": "olmo2_sft",
            "backbone": PINNED["olmo2_sft"],
            "tok": PINNED["olmo2_inst"],
            "base": PINNED["olmo2_base"],
            "extra_heads": {"dpo": PINNED["olmo2_dpo"], "inst": PINNED["olmo2_inst"]},
            "svd_ranks": (4, 16, 64),
            "include_rand_rank": 16,
            "run_tf": False,  # Already computed on full 17-arm grid in P1!
            "clear_cache_after": False,
        },
        {
            "model_key": "olmo2_inst",
            "backbone": PINNED["olmo2_inst"],
            "tok": PINNED["olmo2_inst"],
            "base": PINNED["olmo2_base"],
            "extra_heads": None,
            "svd_ranks": (16,),
            "include_rand_rank": None,
            "run_tf": True,
            "clear_cache_after": True,
        },
        {
            "model_key": "llama31_8b",
            "backbone": PINNED["llama_inst"],
            "tok": PINNED["llama_inst"],
            "base": PINNED["llama_base"],
            "extra_heads": None,
            "svd_ranks": (16,),
            "include_rand_rank": None,
            "run_tf": True,
            "clear_cache_after": True,
        },
        {
            "model_key": "mistral7b_v03",
            "backbone": PINNED["mistral_inst"],
            "tok": PINNED["mistral_inst"],
            "base": PINNED["mistral_base"],
            "extra_heads": None,
            "svd_ranks": (16,),
            "include_rand_rank": None,
            "run_tf": True,
            "clear_cache_after": True,
        },
        {
            "model_key": "qwen25_7b",
            "backbone": PINNED["qwen_inst"],
            "tok": PINNED["qwen_inst"],
            "base": PINNED["qwen_base"],
            "extra_heads": None,
            "svd_ranks": (16,),
            "include_rand_rank": None,
            "run_tf": True,
            "clear_cache_after": True,
        },
    ]

    results_by_model: dict[str, Any] = {}
    for spec in suite_specs:
        mk = spec["model_key"]
        sp = shard_dir / f"{mk}.json"
        if sp.exists():
            results_by_model[mk] = json.loads(sp.read_text())
            print(f"[P3] Resumed {mk} from {sp}", flush=True)
            continue
        res = run_model_suite(
            mk,
            spec["backbone"][0], spec["backbone"][1],
            spec["tok"][0], spec["tok"][1],
            spec["base"][0], spec["base"][1],
            manifest,
            rf_rows,
            vram_gb,
            extra_heads=spec["extra_heads"],
            svd_ranks=spec["svd_ranks"],
            include_rand_rank=spec["include_rand_rank"],
            run_tf=spec["run_tf"],
        )
        tmp = sp.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(res))
        tmp.replace(sp)
        results_by_model[mk] = res
        if spec["clear_cache_after"]:
            clear_hf_cache()

    out = {
        "hardware": gpu_name,
        "vram_gb": round(vram_gb, 2),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "manifest_sha256": manifest["manifest_sha256"],
        "reason_first_meta": {k: [{f: r[f] for f in ("row_id", "pair_id", "family", "policy", "allowed", "wording", "send_label", "keep_label", "system", "user")} for r in v] for k, v in rf_rows.items()},
        "models": results_by_model,
        "elapsed_sec": round(time.monotonic() - t_all, 1),
    }
    out_path = Path("/content/p3_head_generalization.json")
    tmp_out = out_path.with_suffix(".tmp.json")
    tmp_out.write_text(json.dumps(out))
    tmp_out.replace(out_path)
    print(f"[P3] Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB) in {time.monotonic()-t_all:.1f}s", flush=True)
    print("P3_SUCCESS_MARKER", flush=True)


if __name__ == "__main__":
    main()
