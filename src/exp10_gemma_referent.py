"""Exp10b (Priority 2): Order-invariant referent probe gate and conditional patching on Gemma-3-4B-IT.

Step 2.1 — Mandatory Probe Order-Transfer Gate:
  Extracts last-token residual states across layers on single-report and fully-crossed
  two-people prompts (story_first vs real_first, seen vs unseen emergency + harmless dispatch).
  Tests: (a) single->two_people transfer on BOTH orders, (b) cross-order transfer
  (story_first -> real_first and real_first -> story_first), and (c) recency confound
  (urgent_is_last) alignment.

Step 2.2 — Behavioral Suite (100% Jev-classified):
  Runs counterbalanced [A]/[B] tags, explicit attribution questions, and open replies
  (plus oracle vs length-matched restatement clarifications).

Step 2.3 — Conditional Causal Restoration (auto-gated on Step 2.1):
  Only runs if cross-order probe transfer passes the preregistered threshold on `seen`
  discovery scenarios (or with --force-patch). Tests full-residual restore, probe-subspace
  steering, and 3-seed norm-matched orthogonal random direction controls on unseen scenarios.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
from pathlib import Path
from typing import Any

from exp10_cases import exp10_single_report_cases, exp10_two_people_cases, format_referent_prompt
from jev_judge import (
    AB_CHOICE_RUBRIC,
    ATTRIBUTION_RUBRIC,
    HARMLESS_BINDING_RUBRIC,
    JevJudge,
    REFERENT_OPEN_RUBRIC,
)

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from common import MODELS, get_parts
    from exp9_throughput import REVISIONS, chat, last_logit_args
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    AutoModelForCausalLM = AutoTokenizer = Any  # type: ignore[assignment,misc]
    _HAS_TORCH = False


def _no_grad_deco(fn):
    return torch.inference_mode()(fn) if _HAS_TORCH else fn


def load_gemma() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained(MODELS["gemma_it"], revision=REVISIONS["gemma_it"], padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODELS["gemma_it"],
        revision=REVISIONS["gemma_it"],
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        low_cpu_mem_usage=True,
    ).eval()
    return model, tok


def fit_ridge_direction(train_x: torch.Tensor, target: torch.Tensor, lam: float = 0.1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Return (weight_vec [D], center [D], scale, bias) for dual ridge regression."""
    center = train_x.mean(dim=0)
    scale = (train_x - center).square().sum(dim=-1).mean().sqrt().clamp_min(1e-7)
    x = (train_x - center) / scale
    y = target - target.mean()
    alpha = torch.linalg.solve(x @ x.T + lam * torch.eye(len(x), dtype=x.dtype), y)
    w = x.T @ alpha  # [D]
    return w, center, scale, float(target.mean().item())


def apply_ridge(test_x: torch.Tensor, w: torch.Tensor, center: torch.Tensor, scale: torch.Tensor, bias: float) -> torch.Tensor:
    return ((test_x - center) / scale) @ w + bias


@_no_grad_deco
def capture_last_hidden(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    depths: list[int],
    batch_size: int = 24,
) -> dict[int, torch.Tensor]:
    tokenizer.padding_side = "left"
    dev = next(model.parameters()).device
    logit_args = last_logit_args(model)
    buckets: dict[int, list[torch.Tensor]] = collections.defaultdict(list)
    for start in range(0, len(prompts), batch_size):
        batch = tokenizer(
            prompts[start : start + batch_size],
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(dev)
        out = model(**batch, **logit_args, use_cache=False, output_hidden_states=True)
        for d in depths:
            buckets[d].append(out.hidden_states[d][:, -1, :].float().cpu())
    return {d: torch.cat(vs, dim=0) for d, vs in buckets.items()}


def _find_clause_token_spans(tokenizer: AutoTokenizer, prompt: str, order: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """Locate token index ranges (start, end_exclusive) for Clause 1 and Clause 2 in a two_people prompt."""
    m1 = "In the story:" if order == "story_first" else "For the real person typing:"
    m2 = "For the real person typing:" if order == "story_first" else "In the story:"
    i1 = prompt.find(m1)
    i2 = prompt.find(m2, i1 + len(m1))
    if i1 < 0 or i2 < 0:
        raise ValueError(f"Could not find clause markers in prompt for order={order}")
    # Clause 2 ends at end of user sentence (before cue or <end_of_turn>)
    end2 = prompt.find(" What should", i2)
    if end2 < 0:
        end2 = prompt.find("<end_of_turn>", i2)
    if end2 < 0:
        end2 = len(prompt)
    prefix1 = len(tokenizer.encode(prompt[:i1], add_special_tokens=False))
    c1_end = len(tokenizer.encode(prompt[:i2], add_special_tokens=False))
    c2_end = len(tokenizer.encode(prompt[:end2], add_special_tokens=False))
    return (prefix1, max(prefix1 + 1, c1_end)), (c1_end, max(c1_end + 1, c2_end))


@_no_grad_deco
def run_clause_binding_and_attention_diag(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    tp_cases: list[dict[str, Any]],
    depths: list[int],
) -> dict[str, Any]:
    """Feng & Steinhardt clause-end probe + last-token attention mass to Clause 1 vs Clause 2."""
    sub = [c for c in tp_cases if c["domain"] == "emergency" and c["danger"]]
    prompts = [chat(tokenizer, *format_referent_prompt(c, "tag", help_tag="A")) for c in sub]
    dev = next(model.parameters()).device

    # 1) Capture hidden states at clause1_end, clause2_end, and prompt_last (one item at a time for exact span indexing)
    c1_h: dict[int, list[torch.Tensor]] = collections.defaultdict(list)
    c2_h: dict[int, list[torch.Tensor]] = collections.defaultdict(list)
    spans = [_find_clause_token_spans(tokenizer, p, c["order"]) for p, c in zip(prompts, sub)]

    for p, (s1, s2) in zip(prompts, spans):
        ids = tokenizer.encode(p, add_special_tokens=False)
        inp = torch.tensor([ids], dtype=torch.long, device=dev)
        out = model(input_ids=inp, **last_logit_args(model), use_cache=False, output_hidden_states=True)
        p1 = min(s1[1] - 1, len(ids) - 1)
        p2 = min(s2[1] - 1, len(ids) - 1)
        for d in depths:
            hs = out.hidden_states[d][0]
            c1_h[d].append(hs[p1].float().cpu())
            c2_h[d].append(hs[p2].float().cpu())

    clause_probe: dict[str, Any] = {}
    for d in depths:
        if d == 0:
            continue
        x_real_end = []
        x_story_end = []
        for idx, c in enumerate(sub):
            if c["order"] == "story_first":
                x_story_end.append(c1_h[d][idx])
                x_real_end.append(c2_h[d][idx])
            else:
                x_real_end.append(c1_h[d][idx])
                x_story_end.append(c2_h[d][idx])
        xr = torch.stack(x_real_end)
        xs = torch.stack(x_story_end)
        # Leave-scenario-out cross-order probe at the REAL clause's own end token (tests local binding!)
        layer_res = {}
        for train_ord, test_ord in (("story_first", "real_first"), ("real_first", "story_first")):
            scored_r = []
            scored_s = []
            for i, c in enumerate(sub):
                if c["order"] != test_ord:
                    continue
                tr = [j for j, tc in enumerate(sub) if tc["order"] == train_ord and tc["scenario"] != c["scenario"]]
                y = torch.tensor([float(sub[j]["real"]) for j in tr], dtype=torch.float32)
                wr, cr, sr, br = fit_ridge_direction(xr[tr], y)
                ws, cs, ss, bs = fit_ridge_direction(xs[tr], y)
                scored_r.append(((c["scenario"], c["persona"]), c["real"], float(apply_ridge(xr[i : i + 1], wr, cr, sr, br).item())))
                scored_s.append(((c["scenario"], c["persona"]), c["real"], float(apply_ridge(xs[i : i + 1], ws, cs, ss, bs).item())))
            layer_res[f"real_clause_end__{train_ord}_to_{test_ord}"] = _pair_accuracy(scored_r)
            layer_res[f"story_clause_end__{train_ord}_to_{test_ord}"] = _pair_accuracy(scored_s)
        clause_probe[str(d)] = layer_res

    # 2) Last-token attention mass to Clause 1 vs Clause 2 (needs eager attention briefly)
    attn_summary: dict[str, Any] = {}
    prev_impl = getattr(model.config, "_attn_implementation", None)
    try:
        if hasattr(model, "set_attn_implementation"):
            model.set_attn_implementation("eager")
        # Run on `plain` + `kind` + `gruff` for first 6 scenarios (72 prompts) to keep memory & time tiny
        attn_idx = [i for i, c in enumerate(sub) if c["scenario"] < 6]
        by_order_cond: dict[str, list[list[tuple[float, float]]]] = collections.defaultdict(list)
        for i in attn_idx:
            c = sub[i]
            ids = tokenizer.encode(prompts[i], add_special_tokens=False)
            (a1, b1), (a2, b2) = spans[i]
            inp = torch.tensor([ids], dtype=torch.long, device=dev)
            out = model(input_ids=inp, use_cache=False, output_attentions=True)
            layer_pairs = []
            for att in out.attentions:
                # att: [1, H, T, T]; last query token -> mean across heads
                last_q = att[0, :, -1, :].float().mean(dim=0)  # [T]
                m1 = float(last_q[a1:b1].sum().item())
                m2 = float(last_q[a2:b2].sum().item())
                layer_pairs.append((m1, m2))
            by_order_cond[f"{c['order']}__{c['condition']}"].append(layer_pairs)
        for key, mat in by_order_cond.items():
            n_l = len(mat[0])
            attn_summary[key] = {
                "n": len(mat),
                "per_layer_clause1_mean": [sum(row[l][0] for row in mat) / len(mat) for l in range(n_l)],
                "per_layer_clause2_mean": [sum(row[l][1] for row in mat) / len(mat) for l in range(n_l)],
                "overall_clause1_mean": sum(p[0] for row in mat for p in row) / (len(mat) * n_l),
                "overall_clause2_mean": sum(p[1] for row in mat for p in row) / (len(mat) * n_l),
            }
    except Exception as e:
        attn_summary = {"error": str(e)}
    finally:
        if prev_impl and hasattr(model, "set_attn_implementation"):
            try:
                model.set_attn_implementation(prev_impl)
            except Exception:
                pass

    return {"clause_end_probes": clause_probe, "last_token_attention": attn_summary}


@_no_grad_deco
def generate_left_padded(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_new_tokens: int,
    batch_size: int = 32,
    hook_layer: int | None = None,
    hook_fn=None,
) -> list[str]:
    tokenizer.padding_side = "left"
    dev = next(model.parameters()).device
    layers = get_parts(model)[1]
    replies: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch = tokenizer(
            prompts[start : start + batch_size],
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(dev)
        handle = None
        if hook_layer is not None and hook_fn is not None:
            handle = layers[hook_layer].register_forward_hook(hook_fn(start, start + batch.input_ids.shape[0]))
        try:
            out = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        finally:
            if handle is not None:
                handle.remove()
        gen = out[:, batch.input_ids.shape[1] :]
        replies.extend(t.strip() for t in tokenizer.batch_decode(gen, skip_special_tokens=True))
    return replies


def _pair_accuracy(scores: list[tuple[tuple[int, str], bool, float]]) -> dict[str, Any]:
    """Given [(pair_key, is_real_urgent, score)], compute ordered pairwise accuracy."""
    grouped: dict[tuple[int, str], dict[bool, float]] = collections.defaultdict(dict)
    for key, real_flag, sc in scores:
        grouped[key][real_flag] = sc
    n = 0
    correct = 0
    tied = 0
    margins = []
    for key, pair in grouped.items():
        if True in pair and False in pair:
            n += 1
            diff = pair[True] - pair[False]
            margins.append(diff)
            if diff > 0:
                correct += 1
            elif diff == 0:
                tied += 1
    return {
        "n_pairs": n,
        "correct": correct,
        "tied": tied,
        "acc": correct / max(1, n),
        "mean_margin": sum(margins) / max(1, len(margins)),
    }


def evaluate_probe_gate(
    single_cases: list[dict[str, Any]],
    single_h: dict[int, torch.Tensor],
    tp_cases: list[dict[str, Any]],
    tp_h: dict[int, torch.Tensor],
    depths: list[int],
) -> dict[str, Any]:
    """Evaluate single->two_people, cross-order, and recency-confound probes across layers."""
    report: dict[str, Any] = {}
    for d in depths:
        sx = single_h[d]
        tx = tp_h[d]
        layer_res: dict[str, Any] = {}

        # 1) Single-report -> two-people transfer (leave-scenario-out)
        for domain in ("emergency", "harmless"):
            for order in ("story_first", "real_first"):
                for split_filter in ("all", "seen", "unseen"):
                    scored_act = []
                    scored_real = []
                    for i, c in enumerate(tp_cases):
                        if c["domain"] != domain or c["order"] != order or not c["danger"]:
                            continue
                        if split_filter != "all" and c["split"] != split_filter:
                            continue
                        tr = [j for j, sc in enumerate(single_cases) if sc["scenario"] != c["scenario"]]
                        for target_name, dest in (("act", scored_act), ("real", scored_real)):
                            y = torch.tensor([float(single_cases[j][target_name]) for j in tr], dtype=torch.float32)
                            w, ctr, scl, b = fit_ridge_direction(sx[tr], y)
                            pred = float(apply_ridge(tx[i : i + 1], w, ctr, scl, b).item())
                            dest.append(((c["scenario"], c["persona"]), c["real"], pred))
                    if scored_act:
                        layer_res[f"single2tp__{domain}__{order}__{split_filter}__act"] = _pair_accuracy(scored_act)
                        layer_res[f"single2tp__{domain}__{order}__{split_filter}__real"] = _pair_accuracy(scored_real)

        # 2) Cross-order transfer within emergency two_people (leave-scenario-out)
        # Train on story_first -> test on real_first, and vice versa
        for train_ord, test_ord in (("story_first", "real_first"), ("real_first", "story_first")):
            for split_filter in ("all", "seen", "unseen"):
                scored = []
                for i, c in enumerate(tp_cases):
                    if c["domain"] != "emergency" or c["order"] != test_ord or not c["danger"]:
                        continue
                    if split_filter != "all" and c["split"] != split_filter:
                        continue
                    tr = [
                        j
                        for j, tc in enumerate(tp_cases)
                        if tc["domain"] == "emergency"
                        and tc["order"] == train_ord
                        and tc["danger"]
                        and tc["scenario"] != c["scenario"]
                    ]
                    y = torch.tensor([float(tp_cases[j]["real"]) for j in tr], dtype=torch.float32)
                    w, ctr, scl, b = fit_ridge_direction(tx[tr], y)
                    pred = float(apply_ridge(tx[i : i + 1], w, ctr, scl, b).item())
                    scored.append(((c["scenario"], c["persona"]), c["real"], pred))
                if scored:
                    layer_res[f"cross_order__{train_ord}_to_{test_ord}__{split_filter}"] = _pair_accuracy(scored)

        # 3) Direction cosine similarity between story_first and real_first real_urgent probes
        idx_sf = [j for j, c in enumerate(tp_cases) if c["domain"] == "emergency" and c["order"] == "story_first" and c["danger"]]
        idx_rf = [j for j, c in enumerate(tp_cases) if c["domain"] == "emergency" and c["order"] == "real_first" and c["danger"]]
        w_sf, *_ = fit_ridge_direction(tx[idx_sf], torch.tensor([float(tp_cases[j]["real"]) for j in idx_sf]))
        w_rf, *_ = fit_ridge_direction(tx[idx_rf], torch.tensor([float(tp_cases[j]["real"]) for j in idx_rf]))
        cos_sf_rf = float(torch.nn.functional.cosine_similarity(w_sf.unsqueeze(0), w_rf.unsqueeze(0)).item())

        # Also cosine with explicit recency probe (urgent_is_last across all emergency danger cases)
        idx_all = idx_sf + idx_rf
        w_rec, *_ = fit_ridge_direction(tx[idx_all], torch.tensor([float(tp_cases[j]["urgent_is_last"]) for j in idx_all]))
        w_real_all, *_ = fit_ridge_direction(tx[idx_all], torch.tensor([float(tp_cases[j]["real"]) for j in idx_all]))
        layer_res["cos_story_first_vs_real_first"] = cos_sf_rf
        layer_res["cos_sf_probe_vs_recency"] = float(torch.nn.functional.cosine_similarity(w_sf.unsqueeze(0), w_rec.unsqueeze(0)).item())
        layer_res["cos_rf_probe_vs_recency"] = float(torch.nn.functional.cosine_similarity(w_rf.unsqueeze(0), w_rec.unsqueeze(0)).item())
        layer_res["cos_joint_real_vs_recency"] = float(torch.nn.functional.cosine_similarity(w_real_all.unsqueeze(0), w_rec.unsqueeze(0)).item())

        report[str(d)] = layer_res

    # Gate decision strictly on `seen` discovery scenarios!
    best_depth = None
    best_min_acc = -1.0
    for d in depths:
        if d == 0:
            continue
        r = report[str(d)]
        a1 = r["cross_order__story_first_to_real_first__seen"]["acc"]
        a2 = r["cross_order__real_first_to_story_first__seen"]["acc"]
        m = min(a1, a2)
        if m > best_min_acc:
            best_min_acc = m
            best_depth = d

    return {
        "layers": report,
        "gate": {
            "best_discovery_layer": best_depth,
            "best_seen_min_cross_order_acc": best_min_acc,
            "threshold": 0.75,
            "passed": best_min_acc >= 0.75,
        },
    }


def score_referent_outputs_with_jev(out_path: Path, workers: int = 48) -> None:
    """Classify all generated tag, attribution, open, and patch replies with NanoGPT Jev."""
    state = json.loads(out_path.read_text())
    judge = JevJudge(workers=workers)

    def _run_bucket(bucket_name: str, rows: list[dict[str, Any]], rubric_fn) -> None:
        items = []
        for r in rows:
            st = {"ai_instructions": r["system"], "user_message": r["user"], "ai_reply": r["reply"]}
            items.append((st, rubric_fn(r)))
        print(f"Scoring {len(items)} {bucket_name} replies with Jev ({judge.model})...", flush=True)
        ans = judge.decide_batch(items, progress_every=250)
        for r, a in zip(rows, ans):
            r["jev"] = a

    for key, rows in state.get("behavioral", {}).items():
        if key.startswith("tag"):
            _run_bucket(key, rows, lambda _r: AB_CHOICE_RUBRIC)
        elif key.startswith("attribution"):
            _run_bucket(key, rows, lambda _r: ATTRIBUTION_RUBRIC)
        elif key.startswith("open"):
            _run_bucket(key, rows, lambda r: REFERENT_OPEN_RUBRIC if r["domain"] == "emergency" else HARMLESS_BINDING_RUBRIC)

    for key, rows in state.get("patching", {}).items():
        if "tag" in key:
            _run_bucket(f"patch:{key}", rows, lambda _r: AB_CHOICE_RUBRIC)
        else:
            _run_bucket(f"patch:{key}", rows, lambda r: REFERENT_OPEN_RUBRIC if r["domain"] == "emergency" else HARMLESS_BINDING_RUBRIC)

    state["jev_usage"] = judge.usage
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1) + "\n")
    tmp.replace(out_path)
    print("Saved Jev-scored referent results to", out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/exp10_gemma_referent.json"))
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--force-patch", action="store_true", help="Run Step 2.3 patching even if Step 2.1 gate fails")
    ap.add_argument("--score-jev-only", action="store_true", help="Skip GPU work and score an existing output file with Jev")
    ap.add_argument("--score-jev", action="store_true", help="Score generated replies with NanoGPT Jev after GPU run")
    ap.add_argument("--jev-workers", type=int, default=48)
    args = ap.parse_args()

    if args.score_jev_only:
        score_referent_outputs_with_jev(args.out, workers=args.jev_workers)
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(args.out.read_text()) if args.out.exists() else {
        "revision": REVISIONS["gemma_it"],
        "probe_gate": {},
        "behavioral": {},
        "patching": {},
    }

    model, tokenizer = load_gemma()
    n_layers = len(get_parts(model)[1])
    depths = sorted(set([0, n_layers] + [d for d in (4, 8, 12, 16, 17, 20, 21, 24, 28, 32) if d < n_layers]))

    single_cases = exp10_single_report_cases()
    tp_cases = exp10_two_people_cases(include_harmless=True)

    # Step 2.1: Probe Order-Transfer Gate (evaluate under BOTH open and tag wrappers)
    if not state["probe_gate"]:
        print("Running Step 2.1: Order-invariant referent probe gate...", flush=True)
        for wrapper_mode in ("open", "tag"):
            s_prompts = [chat(tokenizer, *format_referent_prompt(c, wrapper_mode, help_tag="A")) for c in single_cases]
            t_prompts = [chat(tokenizer, *format_referent_prompt(c, wrapper_mode, help_tag="A")) for c in tp_cases]
            s_h = capture_last_hidden(model, tokenizer, s_prompts, depths, batch_size=args.batch_size)
            t_h = capture_last_hidden(model, tokenizer, t_prompts, depths, batch_size=args.batch_size)
            state["probe_gate"][wrapper_mode] = evaluate_probe_gate(single_cases, s_h, tp_cases, t_h, depths)
            print(f"  Gate ({wrapper_mode}):", json.dumps(state["probe_gate"][wrapper_mode]["gate"]), flush=True)
        tmp = args.out.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1) + "\n")
        tmp.replace(args.out)

    if "clause_binding_diag" not in state:
        print("Running Step 2.1b: Feng & Steinhardt clause-end probe + last-token attention diagnostic...", flush=True)
        state["clause_binding_diag"] = run_clause_binding_and_attention_diag(model, tokenizer, tp_cases, depths)
        tmp = args.out.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1) + "\n")
        tmp.replace(args.out)

    # Step 2.2: Full behavioral suite (tag crossed A/B, attribution, open, + clarifications on danger=True)
    if "tag_crossed" not in state["behavioral"]:
        print("Running Step 2.2a: Crossed [A]/[B] tag behavioral suite...", flush=True)
        rows = []
        prompts = []
        for c, help_tag in itertools.product(tp_cases, ("A", "B")):
            sys_txt, usr_txt = format_referent_prompt(c, "tag", help_tag=help_tag)
            prompts.append(chat(tokenizer, sys_txt, usr_txt))
            rows.append({**c, "mode": "tag", "help_tag": help_tag, "system": sys_txt, "user": usr_txt})
        replies = generate_left_padded(model, tokenizer, prompts, max_new_tokens=8, batch_size=args.batch_size)
        for r, rep in zip(rows, replies):
            r["reply"] = rep
        state["behavioral"]["tag_crossed"] = rows
        tmp = args.out.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1) + "\n")
        tmp.replace(args.out)

    if "attribution" not in state["behavioral"]:
        print("Running Step 2.2b: Explicit attribution suite...", flush=True)
        rows = []
        prompts = []
        for c in tp_cases:
            sys_txt, usr_txt = format_referent_prompt(c, "attribution")
            prompts.append(chat(tokenizer, sys_txt, usr_txt))
            rows.append({**c, "mode": "attribution", "system": sys_txt, "user": usr_txt})
        replies = generate_left_padded(model, tokenizer, prompts, max_new_tokens=32, batch_size=args.batch_size)
        for r, rep in zip(rows, replies):
            r["reply"] = rep
        state["behavioral"]["attribution"] = rows
        tmp = args.out.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1) + "\n")
        tmp.replace(args.out)

    if "open" not in state["behavioral"]:
        print("Running Step 2.2c: Open-reply + clarification controls...", flush=True)
        rows = []
        prompts = []
        for c in tp_cases:
            for clar in ("none", "oracle", "restatement"):
                if clar != "none" and not c["danger"]:
                    continue
                sys_txt, usr_txt = format_referent_prompt(c, "open", clarification=clar)
                prompts.append(chat(tokenizer, sys_txt, usr_txt))
                rows.append({**c, "mode": "open", "clarification": clar, "system": sys_txt, "user": usr_txt})
        replies = generate_left_padded(model, tokenizer, prompts, max_new_tokens=96, batch_size=args.batch_size)
        for r, rep in zip(rows, replies):
            r["reply"] = rep
        state["behavioral"]["open"] = rows
        tmp = args.out.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1) + "\n")
        tmp.replace(args.out)

    # Step 2.3: Conditional Causal Restoration (only if gate passed or --force-patch)
    gate_tag = state["probe_gate"]["tag"]["gate"]
    gate_open = state["probe_gate"]["open"]["gate"]
    should_patch = gate_tag["passed"] or gate_open["passed"] or args.force_patch
    if should_patch and not state["patching"]:
        best_layer = gate_tag["best_discovery_layer"] if gate_tag["best_seen_min_cross_order_acc"] >= gate_open["best_seen_min_cross_order_acc"] else gate_open["best_discovery_layer"]
        # Hidden states index L corresponds to output of layer (L - 1)
        hook_idx = max(0, best_layer - 1)
        print(f"Running Step 2.3: Controlled activation restoration at hidden depth {best_layer} (block {hook_idx})...", flush=True)

        # Fit frozen probe strictly on `seen` scenarios (both orders)
        seen_emg = [c for c in tp_cases if c["domain"] == "emergency" and c["split"] == "seen" and c["danger"]]
        unseen_emg_rf = [c for c in tp_cases if c["domain"] == "emergency" and c["split"] == "unseen" and c["order"] == "real_first"]
        unseen_emg_sf = {
            (c["scenario"], c["persona"], c["condition"]): c
            for c in tp_cases
            if c["domain"] == "emergency" and c["split"] == "unseen" and c["order"] == "story_first"
        }

        for mode_name, max_tok in (("tag", 8), ("open", 96)):
            seen_prompts = [chat(tokenizer, *format_referent_prompt(c, mode_name, help_tag="A")) for c in seen_emg]
            seen_h = capture_last_hidden(model, tokenizer, seen_prompts, [best_layer], batch_size=args.batch_size)[best_layer]
            w_probe, _, _, _ = fit_ridge_direction(seen_h, torch.tensor([float(c["real"]) for c in seen_emg]))
            u_probe = w_probe / w_probe.norm().clamp_min(1e-8)

            donor_cases = [unseen_emg_sf[(c["scenario"], c["persona"], c["condition"])] for c in unseen_emg_rf]
            donor_prompts = [chat(tokenizer, *format_referent_prompt(c, mode_name, help_tag="A")) for c in donor_cases]
            target_prompts = [chat(tokenizer, *format_referent_prompt(c, mode_name, help_tag="A")) for c in unseen_emg_rf]

            donor_h = capture_last_hidden(model, tokenizer, donor_prompts, [best_layer], batch_size=args.batch_size)[best_layer]
            target_h = capture_last_hidden(model, tokenizer, target_prompts, [best_layer], batch_size=args.batch_size)[best_layer]

            # Build 3 norm-matched orthogonal random directions
            rand_dirs = []
            for seed in (101, 202, 303):
                g = torch.Generator().manual_seed(seed)
                v = torch.randn(u_probe.shape, generator=g)
                v = v - (v @ u_probe) * u_probe
                rand_dirs.append(v / v.norm().clamp_min(1e-8))

            arm_deltas: dict[str, torch.Tensor] = {
                "full_donor_restore": donor_h - target_h,
                "probe_subspace_restore": ((donor_h - target_h) @ u_probe).unsqueeze(-1) * u_probe.unsqueeze(0),
            }
            probe_mag = arm_deltas["probe_subspace_restore"].norm(dim=-1, keepdim=True)
            for s_i, rv in enumerate(rand_dirs):
                arm_deltas[f"random_ortho_seed{s_i}"] = probe_mag * rv.unsqueeze(0)

            for arm_name, delta_mat in arm_deltas.items():
                def make_hook(s_idx: int, e_idx: int, dm=delta_mat):
                    d_slice = dm[s_idx:e_idx]
                    patched_once = [False]

                    def _fn(_mod, _inp, out):
                        if patched_once[0]:
                            return out
                        patched_once[0] = True
                        h_out = out[0] if isinstance(out, tuple) else out
                        h_new = h_out.clone()
                        h_new[:, -1, :] = h_new[:, -1, :] + d_slice.to(h_new.device, h_new.dtype)
                        return (h_new, *out[1:]) if isinstance(out, tuple) else h_new

                    return _fn

                replies = generate_left_padded(
                    model, tokenizer, target_prompts, max_new_tokens=max_tok, batch_size=args.batch_size,
                    hook_layer=hook_idx, hook_fn=make_hook,
                )
                rows = []
                for c, rep in zip(unseen_emg_rf, replies):
                    sys_txt, usr_txt = format_referent_prompt(c, mode_name, help_tag="A")
                    rows.append({**c, "mode": mode_name, "help_tag": "A", "patch_arm": arm_name, "patch_layer": best_layer,
                                 "system": sys_txt, "user": usr_txt, "reply": rep})
                state["patching"][f"{mode_name}__{arm_name}"] = rows
        tmp = args.out.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1) + "\n")
        tmp.replace(args.out)
    else:
        print("Step 2.3 patching skipped (cross-order probe gate did not pass and --force-patch not set).", flush=True)

    if args.score_jev:
        score_referent_outputs_with_jev(args.out, workers=args.jev_workers)


if __name__ == "__main__":
    main()
