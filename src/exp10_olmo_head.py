"""Exp10a (Priority 1): Mechanistic decomposition of OLMo-2's SFT lm_head effect.

Part A — 2x2 Teacher-Forced Readout Matrix ({Base, SFT} post-norm states x {Base, SFT} lm_head)
  across 4 independent action realizations (including shared neutral and shared first-person
  prefixes) on 96 unseen matched pairs (48 privacy, 48 logistics) + 24 pilot consent pairs.

Part B — Direct ΔW_head = W_sft - W_base Weight Surgery on free generation:
  Tests token-subset swaps (top-k rows, opener-only vs non-opener double dissociation),
  low-rank SVD approximations (k=1,4,16,64), row-norm-only rescaling, and norm-matched
  random-row controls, with all free-text replies classified via Jev (NanoGPT).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Any

from exp9_cases import rule_cases as exp9_rule_cases
from exp10_cases import exp10_rule_cases, realizations_for_case
from jev_judge import JevJudge, RULE_RUBRIC

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from common import MODELS, get_parts
    from exp8_selective_refusal import build_cases as pilot_cases, copy_matching
    from exp9_throughput import REVISIONS, chat
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    AutoModelForCausalLM = AutoTokenizer = Any  # type: ignore[assignment,misc]
    _HAS_TORCH = False


def _no_grad_deco(fn):
    return torch.inference_mode()(fn) if _HAS_TORCH else fn


OPENER_STRINGS = (
    "I", " I", "I'm", " I'm", "I'", "Sorry", " Sorry", "Apologies", " Unfortunately", "Unfortunately",
    "No", " No", "HOLD", " HOLD", "Hold", " Hold", "Withheld", " Withheld", "Cannot", " Cannot",
    "Unable", " Unable", "The", " The", "Desk", " Desk", "Result", " Result", "Ref", "Sure", " Sure",
    "Yes", " Yes", "Here", " Here", "Based", " Based", "According", " According",
)


def load_olmo(key: str) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        MODELS[key],
        revision=REVISIONS[key],
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        low_cpu_mem_usage=True,
    ).eval()


def load_olmo_tokenizer() -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(MODELS["olmo_it"], revision=REVISIONS["olmo_it"], padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def opener_token_ids(tokenizer: AutoTokenizer) -> list[int]:
    ids: set[int] = set()
    for s in OPENER_STRINGS:
        enc = tokenizer.encode(s, add_special_tokens=False)
        if enc:
            ids.add(int(enc[0]))
    return sorted(ids)


@_no_grad_deco
def generate_left_padded(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_new_tokens: int = 64,
    batch_size: int = 32,
) -> list[str]:
    tokenizer.padding_side = "left"
    dev = next(model.parameters()).device
    replies: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch = tokenizer(
            prompts[start : start + batch_size],
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(dev)
        out = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen = out[:, batch.input_ids.shape[1] :]
        replies.extend(t.strip() for t in tokenizer.batch_decode(gen, skip_special_tokens=True))
    return replies


@_no_grad_deco
def verify_native_parity(host_model: AutoModelForCausalLM, ref_model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> float:
    """Assert full weight copy matches native SFT logits before trusting any partial swap."""
    tokenizer.padding_side = "left"
    prompt = chat(tokenizer, "You staff an archive desk.", "What is the phrase?")
    inp = tokenizer([prompt], add_special_tokens=False, return_tensors="pt").to(next(host_model.parameters()).device)
    l1 = host_model(**inp, use_cache=False).logits[:, -1, :].float()
    l2 = ref_model(**inp, use_cache=False).logits[:, -1, :].float()
    max_diff = float((l1 - l2).abs().max().item())
    if max_diff > 1e-2:
        raise RuntimeError(f"Native parity check failed: max_abs_diff={max_diff}")
    return max_diff


@_no_grad_deco
def _score_continuation_under_heads(
    model: AutoModelForCausalLM,
    w_base: torch.Tensor,
    w_sft: torch.Tensor,
    prompt_ids: list[int],
    cont_ids: list[int],
) -> dict[str, list[float]]:
    """Run one teacher-forced forward pass, extract post-final-norm h, score under W_base and W_sft."""
    backbone, _, _, _, _ = get_parts(model)
    dev = next(model.parameters()).device
    full_ids = prompt_ids + cont_ids
    inp = torch.tensor([full_ids], dtype=torch.long, device=dev)
    # backbone() returns hidden states AFTER model.norm (post-final-norm, direct input to lm_head)
    h = backbone(input_ids=inp, use_cache=False).last_hidden_state[0]  # [L, D]
    p_len = len(prompt_ids)
    # Positions predicting cont_ids[0..T-1] are [p_len - 1 .. p_len + T - 2]
    h_pred = h[p_len - 1 : p_len + len(cont_ids) - 1].float()  # [T, D]
    target = torch.tensor(cont_ids, dtype=torch.long, device=dev)

    out: dict[str, list[float]] = {}
    for w_name, w in (("base_w", w_base), ("sft_w", w_sft)):
        logits = h_pred @ w.T  # [T, V] in float32
        logp = logits.log_softmax(dim=-1)
        tok_lp = logp[torch.arange(len(cont_ids), device=dev), target]
        out[w_name] = [float(x) for x in tok_lp.cpu().tolist()]
    return out


def _first_divergence(a: list[int], b: list[int]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


@_no_grad_deco
def run_teacher_forced_2x2(
    base_model: AutoModelForCausalLM,
    sft_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """2x2 readout matrix: {base_h, sft_h} x {base_w, sft_w} across 4 realization styles."""
    w_base = get_parts(base_model)[3].weight.detach().float()
    w_sft = get_parts(sft_model)[3].weight.detach().float()

    # Pre-tokenize all (case, style) items
    prepared = []
    for case in cases:
        prompt_text = chat(tokenizer, case["system"], case["user"])
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        reals = realizations_for_case(case)
        for style, spec in reals.items():
            d_ids = tokenizer.encode(spec["disclose"], add_special_tokens=False)
            w_ids = tokenizer.encode(spec["withhold"], add_special_tokens=False)
            div = _first_divergence(d_ids, w_ids)
            prepared.append({
                "case": case,
                "style": style,
                "prompt_ids": prompt_ids,
                "d_ids": d_ids,
                "w_ids": w_ids,
                "div": div,
            })

    # Score with base_model then sft_model to keep cache locality clean
    scores: list[dict[str, Any]] = [{} for _ in prepared]
    for h_name, model in (("base_h", base_model), ("sft_h", sft_model)):
        print(f"  Teacher-forcing {len(prepared)} items through {h_name}...", flush=True)
        for idx, item in enumerate(prepared):
            d_lp = _score_continuation_under_heads(model, w_base, w_sft, item["prompt_ids"], item["d_ids"])
            w_lp = _score_continuation_under_heads(model, w_base, w_sft, item["prompt_ids"], item["w_ids"])
            for w_name in ("base_w", "sft_w"):
                cell = f"{h_name}__{w_name}"
                dl = d_lp[w_name]
                wl = w_lp[w_name]
                div = item["div"]
                d_post = dl[div:] if div < len(dl) else dl[-1:]
                w_post = wl[div:] if div < len(wl) else wl[-1:]
                scores[idx][cell] = {
                    "d_sum": sum(dl),
                    "w_sum": sum(wl),
                    "d_mean": sum(dl) / max(1, len(dl)),
                    "w_mean": sum(wl) / max(1, len(wl)),
                    "d_post_sum": sum(d_post),
                    "w_post_sum": sum(w_post),
                    "d_post_mean": sum(d_post) / max(1, len(d_post)),
                    "w_post_mean": sum(w_post) / max(1, len(w_post)),
                    "d_first_div": dl[div] if div < len(dl) else float("nan"),
                    "w_first_div": wl[div] if div < len(wl) else float("nan"),
                }

    rows = []
    for item, sc in zip(prepared, scores):
        c = item["case"]
        div = item["div"]
        rows.append({
            "id": c["id"],
            "template": c["template"],
            "domain": c["domain"],
            "allowed": c["allowed"],
            "style": item["style"],
            "n_tok_disclose": len(item["d_ids"]),
            "n_tok_withhold": len(item["w_ids"]),
            "first_div_pos": div,
            "first_div_tok_disclose": tokenizer.decode([item["d_ids"][div]]) if div < len(item["d_ids"]) else "",
            "first_div_tok_withhold": tokenizer.decode([item["w_ids"][div]]) if div < len(item["w_ids"]) else "",
            "cells": sc,
        })

    # Also save last-prompt-token SFT hidden states for ΔW geometry alignment (μ(denied)-μ(allowed) vs μ(all))
    sft_last_h: dict[tuple[str, str, bool], torch.Tensor] = {}
    backbone_sft = get_parts(sft_model)[0]
    dev_sft = next(sft_model.parameters()).device
    for case in cases:
        p_ids = tokenizer.encode(chat(tokenizer, case["system"], case["user"]), add_special_tokens=False)
        inp = torch.tensor([p_ids], dtype=torch.long, device=dev_sft)
        h_last = backbone_sft(input_ids=inp, use_cache=False).last_hidden_state[0, -1, :].float().cpu()
        sft_last_h[(case["id"], case["domain"], case["allowed"])] = h_last

    # Aggregate summary: mean within-case contrast, paired accuracy, AND signal-detection d' vs criterion c
    summary: dict[str, Any] = {}
    cells = ("base_h__base_w", "base_h__sft_w", "sft_h__base_w", "sft_h__sft_w")

    def _sdt_stats(denied_scores: list[float], allowed_scores: list[float]) -> dict[str, float]:
        """Compute continuous d' and zero-threshold criterion c on score = logP(withhold) - logP(disclose)."""
        if not denied_scores or not allowed_scores:
            return {"d_prime": float("nan"), "criterion_c": float("nan"), "mean_denied": float("nan"), "mean_allowed": float("nan")}
        md = sum(denied_scores) / len(denied_scores)
        ma = sum(allowed_scores) / len(allowed_scores)
        vd = sum((x - md) ** 2 for x in denied_scores) / max(1, len(denied_scores) - 1)
        va = sum((x - ma) ** 2 for x in allowed_scores) / max(1, len(allowed_scores) - 1)
        pooled = math.sqrt(0.5 * (vd + va)) if (vd + va) > 0 else 1e-8
        pooled = max(pooled, 1e-8)
        # Positive d' = better separation of denied > allowed; negative criterion_c = bias toward withholding
        return {
            "d_prime": (md - ma) / pooled,
            "criterion_c": -0.5 * (md + ma) / pooled,
            "mean_denied": md,
            "mean_allowed": ma,
            "hit_rate_at_0": sum(x > 0 for x in denied_scores) / len(denied_scores),
            "fa_rate_at_0": sum(x > 0 for x in allowed_scores) / len(allowed_scores),
        }

    for domain in sorted({r["domain"] for r in rows}):
        summary[domain] = {}
        for style in sorted({r["style"] for r in rows}):
            sub = [r for r in rows if r["domain"] == domain and r["style"] == style]
            by_id = collections.defaultdict(dict)
            for r in sub:
                by_id[r["id"]][r["allowed"]] = r
            cell_stats = {}
            for cell in cells:
                margins_post_mean = []
                margins_first_div = []
                w_minus_d_denied_pm, w_minus_d_allowed_pm = [], []
                w_minus_d_denied_fd, w_minus_d_allowed_fd = [], []
                pair_ok_post_mean = 0
                pair_ok_first_div = 0
                for cid, pair in by_id.items():
                    if True not in pair or False not in pair:
                        continue
                    c_al = pair[True]["cells"][cell]
                    c_dn = pair[False]["cells"][cell]
                    m_al_pm = c_al["d_post_mean"] - c_al["w_post_mean"]
                    m_dn_pm = c_dn["w_post_mean"] - c_dn["d_post_mean"]
                    margins_post_mean.extend([m_al_pm, m_dn_pm])
                    w_minus_d_denied_pm.append(c_dn["w_post_mean"] - c_dn["d_post_mean"])
                    w_minus_d_allowed_pm.append(c_al["w_post_mean"] - c_al["d_post_mean"])
                    if m_al_pm > 0 and m_dn_pm > 0:
                        pair_ok_post_mean += 1

                    m_al_fd = c_al["d_first_div"] - c_al["w_first_div"]
                    m_dn_fd = c_dn["w_first_div"] - c_dn["d_first_div"]
                    margins_first_div.extend([m_al_fd, m_dn_fd])
                    w_minus_d_denied_fd.append(c_dn["w_first_div"] - c_dn["d_first_div"])
                    w_minus_d_allowed_fd.append(c_al["w_first_div"] - c_al["d_first_div"])
                    if m_al_fd > 0 and m_dn_fd > 0:
                        pair_ok_first_div += 1
                n_pairs = len(by_id)
                cell_stats[cell] = {
                    "n_pairs": n_pairs,
                    "mean_contrast_post_mean": sum(margins_post_mean) / max(1, len(margins_post_mean)),
                    "pair_acc_post_mean": pair_ok_post_mean / max(1, n_pairs),
                    "sdt_post_mean": _sdt_stats(w_minus_d_denied_pm, w_minus_d_allowed_pm),
                    "mean_contrast_first_div": sum(margins_first_div) / max(1, len(margins_first_div)),
                    "pair_acc_first_div": pair_ok_first_div / max(1, n_pairs),
                    "sdt_first_div": _sdt_stats(w_minus_d_denied_fd, w_minus_d_allowed_fd),
                }
            summary[domain][style] = cell_stats

    return {"summary": summary, "rows": rows}, sft_last_h


@_no_grad_deco
def compute_delta_w_state_geometry(
    w_base: torch.Tensor,
    w_sft: torch.Tensor,
    sft_last_h: dict[tuple[str, str, bool], torch.Tensor],
    tokenizer: AutoTokenizer,
) -> dict[str, Any]:
    """Compute ΔW_head alignment with SFT post-norm μ(denied)-μ(allowed) vs μ(all)."""
    dw = (w_sft.float() - w_base.float()).to("cuda")
    _, _, vh = torch.linalg.svd(dw, full_matrices=False)
    vh_top = vh[:8].cpu()
    dw_cpu = dw.cpu()
    del dw, vh
    torch.cuda.empty_cache()

    geometry: dict[str, Any] = {}
    for scope in ("all", "privacy", "logistics"):
        h_dn = torch.stack([v for (_, dom, al), v in sft_last_h.items() if not al and (scope == "all" or dom == scope)])
        h_al = torch.stack([v for (_, dom, al), v in sft_last_h.items() if al and (scope == "all" or dom == scope)])
        mu_diff = h_dn.mean(0) - h_al.mean(0)
        mu_all = torch.cat([h_dn, h_al], dim=0).mean(0)
        u_diff = mu_diff / mu_diff.norm().clamp_min(1e-8)
        u_all = mu_all / mu_all.norm().clamp_min(1e-8)
        cos_diff_vh = [float((vh_top[k] @ u_diff).item()) for k in range(8)]
        cos_all_vh = [float((vh_top[k] @ u_all).item()) for k in range(8)]
        logit_diff_norm = float((dw_cpu @ mu_diff).norm().item())
        logit_all_norm = float((dw_cpu @ mu_all).norm().item())
        logit_unit_diff_norm = float((dw_cpu @ u_diff).norm().item())
        logit_unit_all_norm = float((dw_cpu @ u_all).norm().item())
        top_all_pos = torch.topk(dw_cpu @ mu_all, 10)
        top_diff_pos = torch.topk(dw_cpu @ mu_diff, 10)
        geometry[scope] = {
            "mu_diff_norm": float(mu_diff.norm().item()),
            "mu_all_norm": float(mu_all.norm().item()),
            "cos_vh_top8_with_mu_diff": cos_diff_vh,
            "cos_vh_top8_with_mu_all": cos_all_vh,
            "delta_logit_norm_mu_diff": logit_diff_norm,
            "delta_logit_norm_mu_all": logit_all_norm,
            "delta_logit_norm_unit_diff": logit_unit_diff_norm,
            "delta_logit_norm_unit_all": logit_unit_all_norm,
            "top10_tokens_dw_mu_all": [(tokenizer.decode([int(i)]), float(v)) for v, i in zip(top_all_pos.values.tolist(), top_all_pos.indices.tolist())],
            "top10_tokens_dw_mu_diff": [(tokenizer.decode([int(i)]), float(v)) for v, i in zip(top_diff_pos.values.tolist(), top_diff_pos.indices.tolist())],
        }
    return geometry


@_no_grad_deco
def analyze_delta_w(w_base: torch.Tensor, w_sft: torch.Tensor, tokenizer: AutoTokenizer) -> dict[str, Any]:
    """Compute SVD spectrum, top-shifted tokens, and row-norm shift diagnostics for ΔW = W_sft - W_base."""
    dw = (w_sft.float() - w_base.float()).cpu()
    row_norms = dw.norm(dim=1)
    base_norms = w_base.float().cpu().norm(dim=1)
    sft_norms = w_sft.float().cpu().norm(dim=1)

    top_vals, top_idx = torch.topk(row_norms, 30)
    top_tokens = [
        {
            "token_id": int(i),
            "token": tokenizer.decode([int(i)]),
            "delta_norm": float(v),
            "base_norm": float(base_norms[i]),
            "sft_norm": float(sft_norms[i]),
            "norm_ratio": float(sft_norms[i] / base_norms[i].clamp_min(1e-8)),
        }
        for v, i in zip(top_vals.tolist(), top_idx.tolist())
    ]

    # Fast SVD on GPU
    _, s, _ = torch.linalg.svd(dw.to("cuda"), full_matrices=False)
    s_cpu = s.cpu()
    energy = (s_cpu ** 2)
    cum_energy = (energy.cumsum(0) / energy.sum()).tolist()

    return {
        "shape": list(dw.shape),
        "frobenius_norm": float(dw.norm().item()),
        "relative_frobenius": float(dw.norm().item() / w_base.float().norm().item()),
        "top20_singular_values": [float(x) for x in s_cpu[:20].tolist()],
        "cum_energy_at_k": {str(k): float(cum_energy[k - 1]) for k in (1, 2, 4, 8, 16, 32, 64, 128) if k <= len(cum_energy)},
        "top30_shifted_tokens": top_tokens,
    }


@_no_grad_deco
def run_weight_surgery(
    host_model: AutoModelForCausalLM,
    w_base: torch.Tensor,
    w_sft: torch.Tensor,
    tokenizer: AutoTokenizer,
    panels: dict[str, list[dict[str, Any]]],
    out_path: Path,
    state: dict[str, Any],
    batch_size: int = 32,
) -> None:
    """Test targeted lm_head surgeries on host_model (all SFT weights except lm_head)."""
    lm_head_w = get_parts(host_model)[3].weight
    dev, dtype = lm_head_w.device, lm_head_w.dtype
    wb = w_base.float().to(dev)
    ws = w_sft.float().to(dev)
    dw = ws - wb
    row_norms = dw.norm(dim=1)
    u, s, vh = torch.linalg.svd(dw, full_matrices=False)

    op_ids = opener_token_ids(tokenizer)
    state["opener_token_ids"] = op_ids
    state["opener_tokens"] = [tokenizer.decode([i]) for i in op_ids]

    g = torch.Generator(device="cpu").manual_seed(2026)
    rand50 = torch.randperm(dw.shape[0], generator=g)[:50].to(dev)

    def make_arm(name: str) -> torch.Tensor:
        if name == "base_head":
            return wb
        if name == "sft_head":
            return ws
        if name.startswith("top_rows_"):
            k = int(name.split("_")[-1])
            idx = torch.topk(row_norms, k).indices
            w = wb.clone()
            w[idx] = ws[idx]
            return w
        if name == "opener_rows_only":
            w = wb.clone()
            w[op_ids] = ws[op_ids]
            return w
        if name == "except_opener_rows":
            w = ws.clone()
            w[op_ids] = wb[op_ids]
            return w
        if name == "rand_rows_50":
            w = wb.clone()
            w[rand50] = ws[rand50]
            return w
        if name == "row_norm_only":
            scale = ws.norm(dim=1, keepdim=True) / wb.norm(dim=1, keepdim=True).clamp_min(1e-8)
            return wb * scale
        if name.startswith("svd_rank_"):
            k = int(name.split("_")[-1])
            approx = (u[:, :k] * s[:k].unsqueeze(0)) @ vh[:k, :]
            return wb + approx
        raise ValueError(f"Unknown arm: {name}")

    arms = [
        "base_head",
        "sft_head",
        "opener_rows_only",
        "except_opener_rows",
        "top_rows_10",
        "top_rows_50",
        "top_rows_200",
        "rand_rows_50",
        "row_norm_only",
        "svd_rank_1",
        "svd_rank_4",
        "svd_rank_16",
        "svd_rank_64",
    ]

    for arm in arms:
        if arm in state["surgery_replies"]:
            print(f"  Skipping cached surgery arm {arm}", flush=True)
            continue
        w_new = make_arm(arm)
        lm_head_w.copy_(w_new.to(dtype))
        arm_out = {}
        for panel_name, cases in panels.items():
            prompts = [chat(tokenizer, c["system"], c["user"]) for c in cases]
            replies = generate_left_padded(host_model, tokenizer, prompts, max_new_tokens=64, batch_size=batch_size)
            arm_out[panel_name] = [
                {
                    "id": c["id"],
                    "template": c.get("template", c["id"].split("-")[0]),
                    "domain": c.get("domain", "consent"),
                    "allowed": c["allowed"],
                    "target": c.get("target", c.get("expected")),
                    "reply": r,
                }
                for c, r in zip(cases, replies)
            ]
        state["surgery_replies"][arm] = arm_out
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1) + "\n")
        tmp.replace(out_path)
        print(f"  Saved surgery arm {arm}", flush=True)

    # Qi et al. shallow-alignment prefill depth check: compare base_head vs sft_head after forcing 2-3 opener tokens
    state.setdefault("prefill_replies", {})
    prefills = {
        "apology_prefix": "I'm sorry,",
        "neutral_prefix": "The reference phrase",
    }
    for head_name, w_mat in (("base_head", wb), ("sft_head", ws)):
        lm_head_w.copy_(w_mat.to(dtype))
        for pf_name, pf_text in prefills.items():
            key = f"{head_name}__{pf_name}"
            if key in state["prefill_replies"]:
                continue
            pf_out = {}
            for panel_name in ("pilot_24", "exp10_96"):
                cases = panels[panel_name]
                prompts = [chat(tokenizer, c["system"], c["user"]) + pf_text for c in cases]
                conts = generate_left_padded(host_model, tokenizer, prompts, max_new_tokens=48, batch_size=batch_size)
                pf_out[panel_name] = [
                    {
                        "id": c["id"],
                        "template": c.get("template", c["id"].split("-")[0]),
                        "domain": c.get("domain", "consent"),
                        "allowed": c["allowed"],
                        "target": c.get("target", c.get("expected")),
                        "reply": (pf_text + " " + r).strip() if not r.startswith((" ", ".", ",", "!", "?", "'")) else (pf_text + r).strip(),
                    }
                    for c, r in zip(cases, conts)
                ]
            state["prefill_replies"][key] = pf_out
            tmp = out_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=1) + "\n")
            tmp.replace(out_path)
            print(f"  Saved prefill check {key}", flush=True)


def score_surgery_with_jev(out_path: Path, workers: int = 48) -> None:
    """Classify all generated surgery replies via NanoGPT Jev (zero programmatic classifiers)."""
    state = json.loads(out_path.read_text())
    judge = JevJudge(workers=workers)

    # Rebuild case instructions lookup for all panels (works on CPU without torch)
    case_map: dict[tuple[str, str, str, bool], dict[str, Any]] = {}
    pilot_json = out_path.parent / "exp8_consent_paraphrases.json"
    if pilot_json.exists():
        p_cases = json.loads(pilot_json.read_text())["cases"]
    else:
        p_cases = pilot_cases(consent_paraphrases=True)
    for c in p_cases:
        case_map[("pilot_24", c["id"], "consent", c["allowed"])] = {
            "system": c["system"], "user": c["user"], "target": c["expected"]
        }
    for c in exp10_rule_cases():
        case_map[("exp10_96", c["id"], c["domain"], c["allowed"])] = {
            "system": c["system"], "user": c["user"], "target": c["target"]
        }
    for c in exp9_rule_cases():
        case_map[("exp9_72", c["id"], c["domain"], c["allowed"])] = {
            "system": c["system"], "user": c["user"], "target": c["target"]
        }

    items = []
    meta = []
    for section in ("surgery_replies", "prefill_replies"):
        for arm, panels in state.get(section, {}).items():
            for panel_name, rows in panels.items():
                for idx, r in enumerate(rows):
                    c = case_map[(panel_name, r["id"], r["domain"], r["allowed"])]
                    items.append((
                        {
                            "ai_instructions": c["system"],
                            "user_message": c["user"],
                            "ai_reply": r["reply"],
                            "target_phrase": c["target"],
                        },
                        RULE_RUBRIC,
                    ))
                    meta.append((section, arm, panel_name, idx))

    print(f"Scoring {len(items)} surgery + prefill replies with Jev ({judge.model})...", flush=True)
    ans = judge.decide_batch(items, progress_every=250)
    for (section, arm, panel_name, idx), a in zip(meta, ans):
        state[section][arm][panel_name][idx]["jev"] = a

    # Build clean summary table per arm x panel x domain
    def _summarize_section(sec_dict: dict[str, Any]) -> dict[str, Any]:
        summary = {}
        for arm, panels in sec_dict.items():
            summary[arm] = {}
            for panel_name, rows in panels.items():
                by_dom = collections.defaultdict(list)
                for r in rows:
                    by_dom[r["domain"]].append(r)
                dom_sum = {}
                for dom, drows in by_dom.items():
                    al = {r["id"]: r for r in drows if r["allowed"]}
                    dn = {r["id"]: r for r in drows if not r["allowed"]}
                    valid = sum(
                        bool(al[k]["jev"]["disclosed_target"]["value"])
                        and not bool(dn[k]["jev"]["disclosed_target"]["value"])
                        and dn[k]["jev"]["action"]["value"] == "withholds_explicitly"
                        for k in al
                        if k in dn and al[k].get("jev") and dn[k].get("jev")
                    )
                    dom_sum[dom] = {
                        "n_pairs": len(al),
                        "jev_valid_pairs": valid,
                        "allowed_disclosed": sum(bool(r["jev"]["disclosed_target"]["value"]) for r in al.values() if r.get("jev")),
                        "allowed_false_refusals": sum(r["jev"]["action"]["value"] == "withholds_explicitly" for r in al.values() if r.get("jev")),
                        "denied_disclosed": sum(bool(r["jev"]["disclosed_target"]["value"]) for r in dn.values() if r.get("jev")),
                        "denied_refusal_styles": dict(collections.Counter(r["jev"]["refusal_style"]["value"] for r in dn.values() if r.get("jev"))),
                    }
                summary[arm][panel_name] = dom_sum
        return summary

    state["surgery_jev_summary"] = _summarize_section(state.get("surgery_replies", {}))
    state["prefill_jev_summary"] = _summarize_section(state.get("prefill_replies", {}))
    state["jev_usage"] = judge.usage
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1) + "\n")
    tmp.replace(out_path)
    print("Saved Jev-scored summary to", out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/exp10_olmo_head.json"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--include-exp9", action="store_true", help="Also run surgery arms on the 72-pair exp9 rules")
    ap.add_argument("--score-jev-only", action="store_true", help="Skip GPU work and score an existing output file with Jev")
    ap.add_argument("--score-jev", action="store_true", help="Score generated replies with NanoGPT Jev after GPU run")
    ap.add_argument("--jev-workers", type=int, default=48)
    args = ap.parse_args()

    if args.score_jev_only:
        score_surgery_with_jev(args.out, workers=args.jev_workers)
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(args.out.read_text()) if args.out.exists() else {
        "revisions": {k: REVISIONS[k] for k in ("olmo_base", "olmo_sft", "olmo_it")},
        "delta_w_stats": {},
        "teacher_forced_2x2": {},
        "surgery_replies": {},
    }

    tokenizer = load_olmo_tokenizer()
    base = load_olmo("olmo_base")
    sft = load_olmo("olmo_sft")
    w_base = get_parts(base)[3].weight.detach().cpu().clone()
    w_sft = get_parts(sft)[3].weight.detach().cpu().clone()

    sft_last_h = None
    if not state["teacher_forced_2x2"]:
        print("Running Part A: 2x2 teacher-forced readout matrix...", flush=True)
        e10_cases = exp10_rule_cases()
        tf_res, sft_last_h = run_teacher_forced_2x2(base, sft, tokenizer, e10_cases)
        state["teacher_forced_2x2"] = tf_res
        tmp = args.out.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1) + "\n")
        tmp.replace(args.out)

    # Prepare host model = full SFT copied into base, verify native parity, then free sft before SVD/surgeries
    print("Preparing host model and verifying native SFT parity...", flush=True)
    copy_matching(base.state_dict(), sft.state_dict(), (".layers.", ".embed_tokens.", "model.norm.", "lm_head."))
    max_diff = verify_native_parity(base, sft, tokenizer)
    state["native_parity_max_abs_diff"] = max_diff
    print(f"  Native parity OK (max_abs_diff={max_diff:.6f})", flush=True)

    del sft
    torch.cuda.empty_cache()

    if not state["delta_w_stats"]:
        print("Analyzing ΔW_head = W_sft - W_base...", flush=True)
        state["delta_w_stats"] = analyze_delta_w(w_base, w_sft, tokenizer)
    if sft_last_h is not None and "delta_w_state_geometry" not in state["teacher_forced_2x2"]:
        print("Computing ΔW_head alignment with SFT hidden states...", flush=True)
        state["teacher_forced_2x2"]["delta_w_state_geometry"] = compute_delta_w_state_geometry(
            w_base, w_sft, sft_last_h, tokenizer
        )
        tmp = args.out.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1) + "\n")
        tmp.replace(args.out)

    panels: dict[str, list[dict[str, Any]]] = {
        "pilot_24": pilot_cases(consent_paraphrases=True),
        "exp10_96": exp10_rule_cases(),
    }
    if args.include_exp9:
        panels["exp9_72"] = exp9_rule_cases()

    print("Running Part B: ΔW_head weight surgeries...", flush=True)
    run_weight_surgery(base, w_base, w_sft, tokenizer, panels, args.out, state, batch_size=args.batch_size)

    if args.score_jev:
        score_surgery_with_jev(args.out, workers=args.jev_workers)


if __name__ == "__main__":
    main()
