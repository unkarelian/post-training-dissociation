"""Experiment 1: Cross-family 'Loose-Fitting Clothes on Skin'
1A: Frame unzipping across 6 frames x 6 models (Base + IT for Gemma-3-4B, Llama-3.1-8B, Qwen-2.5-7B)
1B: Layer-by-layer logit lens on chat vs raw_qa (Base + IT)
1C: Causal chat-template header attention knockout vs length-matched non-header control (IT models)
"""
import argparse, json, os, torch
import common as C


def run_logit_lens(tok, model, strs, pos_ids, neg_ids):
    """Compute P(pos | pos, neg) at every layer via final norm + lm_head."""
    backbone, _, norm, lm_head, _ = C.get_parts(model)
    dev = next(model.parameters()).device
    tok.padding_side = "right"
    e = tok(strs, add_special_tokens=False, return_tensors="pt", padding=True)
    ids = e.input_ids.to(dev)
    mask = e.attention_mask.to(dev)
    last = mask.sum(1) - 1
    ar = torch.arange(len(last), device=dev)
    with torch.no_grad():
        out = backbone(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        res = []
        for h_all in out.hidden_states:
            h = norm(h_all[ar, last])
            lp = lm_head(h).float().log_softmax(-1)
            p = C.prob_ratio(lp, pos_ids, neg_ids).mean().item()
            res.append(round(p, 4))
    return res


def find_header_span(tok, chat_str, family):
    """Find token indices of the assistant turn header (excluding the very last decision token)."""
    if family == "gemma":
        marker = "<start_of_turn>model"
    elif family == "llama":
        marker = "<|start_header_id|>assistant"
    else:
        marker = "<|im_start|>assistant"
    idx = chat_str.rfind(marker)
    assert idx != -1, f"Header marker {marker} not found in {chat_str!r}"
    prefix_ids = tok.encode(chat_str[:idx], add_special_tokens=False)
    full_ids = tok.encode(chat_str, add_special_tokens=False)
    start = len(prefix_ids)
    end = len(full_ids) - 1
    if end <= start:
        start = max(0, len(full_ids) - 3)
        end = len(full_ids) - 1
    return full_ids, list(range(start, end))


def run_knockout(tok, model, chat_strs, family, pos_ids, neg_ids):
    """Compare baseline vs header-token knockout vs length-matched non-header token knockout."""
    backbone, _, _, lm_head, _ = C.get_parts(model)
    dev = next(model.parameters()).device
    results = {"base": [], "ko_header": [], "ko_ctrl": [], "header_tokens_sample": None}
    for i, s in enumerate(chat_strs):
        full_ids, h_pos = find_header_span(tok, s, family)
        if i == 0:
            results["header_tokens_sample"] = [tok.decode([full_ids[p]]) for p in h_pos]
        k = len(h_pos)
        mid = max(2, (h_pos[0] // 2) - (k // 2))
        c_pos = list(range(mid, min(h_pos[0] - 1, mid + k)))
        ids = torch.tensor([full_ids], device=dev)
        pos_ids_t = torch.arange(len(full_ids), device=dev).unsqueeze(0)
        for mode, drop in [("base", []), ("ko_header", h_pos), ("ko_ctrl", c_pos)]:
            mask = torch.ones((1, len(full_ids)), dtype=torch.long, device=dev)
            for p in drop:
                mask[0, p] = 0
            with torch.no_grad():
                out = backbone(input_ids=ids, attention_mask=mask, position_ids=pos_ids_t)
                lp = lm_head(out.last_hidden_state[0, -1:]).float().log_softmax(-1).cpu()
                results[mode].append(float(C.prob_ratio(lp, pos_ids, neg_ids)[0]))
    return {
        "base": round(sum(results["base"]) / len(results["base"]), 4),
        "ko_header": round(sum(results["ko_header"]) / len(results["ko_header"]), 4),
        "ko_ctrl": round(sum(results["ko_ctrl"]) / len(results["ko_ctrl"]), 4),
        "n_header_toks": len(results["header_tokens_sample"] or []),
        "header_tokens_sample": results["header_tokens_sample"],
    }


def eval_model(family, key, is_it, out_dir):
    print(f"\n=== EXP1 {key} ({C.MODELS[key]}) ===", flush=True)
    tok, model = C.load_model(key)
    ts = C.token_sets(tok)
    out = {"key": key, "model": C.MODELS[key], "family": family, "is_it": is_it}

    frames = ["chat", "chat_swap", "chat_no_role", "raw_qa", "raw_dialogue", "raw_interview"]
    out["self_id"] = {}
    out["battery"] = {}
    for fr in frames:
        strs = [C.format_frames(tok, family, q + C.ONE_HA, is_it)[fr] for q in C.SELF_QS]
        lp = C.forward_last_logprobs(tok, model, strs)
        out["self_id"][fr] = {
            "p_human": round(float(C.prob_ratio(lp, ts["H"], ts["A"]).mean()), 4),
            "mass": round(float(C.prob_mass(lp, ts["H"], ts["A"]).mean()), 4),
        }
        b_strs = [C.format_frames(tok, family, q + C.ONE_YN, is_it)[fr] for q in C.BATTERY.values()]
        lp_b = C.forward_last_logprobs(tok, model, b_strs)
        probs_b = C.prob_ratio(lp_b, ts["Y"], ts["N"]).tolist()
        out["battery"][fr] = {k: round(p, 4) for k, p in zip(C.BATTERY.keys(), probs_b)}
        print(f"  [{fr:13s}] p_human={out['self_id'][fr]['p_human']:.3f} "
              f"alive={out['battery'][fr]['alive']:.3f} born={out['battery'][fr]['born']:.3f} "
              f"ai={out['battery'][fr]['ai']:.3f} toaster={out['battery'][fr]['ctrl_toaster']:.3f} "
              f"2+2={out['battery'][fr]['ctrl_2plus2']:.3f}", flush=True)

    out["logit_lens"] = {}
    for fr in ("chat", "raw_qa"):
        ll = {}
        strs_ha = [C.format_frames(tok, family, q + C.ONE_HA, is_it)[fr] for q in C.SELF_QS]
        ll["self_p_human"] = run_logit_lens(tok, model, strs_ha, ts["H"], ts["A"])
        for bk in ("alive", "born", "body", "human", "ai", "program", "ctrl_toaster", "ctrl_2plus2"):
            s = [C.format_frames(tok, family, C.BATTERY[bk] + C.ONE_YN, is_it)[fr]]
            ll[bk] = run_logit_lens(tok, model, s, ts["Y"], ts["N"])
        out["logit_lens"][fr] = ll

    if is_it:
        ko = {}
        strs_ha = [C.format_frames(tok, family, q + C.ONE_HA, True)["chat"] for q in C.SELF_QS]
        ko["self_p_human"] = run_knockout(tok, model, strs_ha, family, ts["H"], ts["A"])
        for bk in ("alive", "born", "body", "feelings", "conscious", "human", "ai", "program", "ctrl_toaster", "ctrl_2plus2"):
            s = [C.format_frames(tok, family, C.BATTERY[bk] + C.ONE_YN, True)["chat"]]
            ko[bk] = run_knockout(tok, model, s, family, ts["Y"], ts["N"])
        out["knockout"] = ko
        print("  KNOCKOUT:", {k: (v["base"], v["ko_header"], v["ko_ctrl"]) for k, v in ko.items()}, flush=True)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"exp1_{key}.json"), "w") as f:
        json.dump(out, f, indent=1)
    C.unload(model, tok)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", help="Model keys to run (default: all 6)")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    only = set(args.models) if args.models else None
    for fam, bkey, ikey in C.FAMILIES:
        for k, is_it in [(bkey, False), (ikey, True)]:
            if only and k not in only:
                continue
            eval_model(fam, k, is_it, args.out_dir)
    print("EXP1_ALL_DONE", flush=True)
