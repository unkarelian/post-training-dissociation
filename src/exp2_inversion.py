"""Experiment 2: Cross-Family Identity Inversion & Causal Coupling
2A: Prose detection (says_ai) + NLL (nats/token) + quality rating on amateur human (wp) AND
    published literary human (books) vs AI writers across all 6 models.
2B: Causal coupling on IT models: steer along (raw - chat) self-identity direction vs exact
    norm-matched random direction (3 seeds), measuring BOTH self-identity AND prose says_ai.
"""
import argparse, json, os, numpy as np, torch
import common as C

DET_Q = "Was this passage written by a human or by an AI? Answer with one word: human or AI.\n\n{t}"
RATE_Q = "Rate the quality of the writing in this passage from 1 (poor) to 9 (excellent). Answer with a single digit.\n\n{t}"


def passage_nll(tok, model, texts):
    """Mean NLL (nats/token) over tokens [15:] for each passage."""
    backbone, _, _, lm_head, _ = C.get_parts(model)
    dev = next(model.parameters()).device
    bos = tok.bos_token or ""
    nlls = []
    for t in texts:
        ids = tok.encode(bos + t, add_special_tokens=False)
        if len(ids) < 25:
            nlls.append(None)
            continue
        x = torch.tensor([ids[:400]], device=dev)
        with torch.no_grad():
            h = backbone(input_ids=x).last_hidden_state[0, 15:-1]
            lp = lm_head(h).float().log_softmax(-1)
            tgt = x[0, 16:]
            nll = float(-lp.gather(1, tgt[:, None]).mean())
        nlls.append(round(nll, 4))
    return nlls


def get_resids(tok, model, strs):
    """Capture last-token residual stream [N, L, D] across all layers."""
    backbone, layers, _, _, _ = C.get_parts(model)
    dev = next(model.parameters()).device
    tok.padding_side = "right"
    e = tok(strs, add_special_tokens=False, return_tensors="pt", padding=True)
    ids = e.input_ids.to(dev)
    mask = e.attention_mask.to(dev)
    last = mask.sum(1) - 1
    ar = torch.arange(len(last), device=dev)
    cap = {}
    hs = [layers[l].register_forward_hook(
        lambda m, i, o, l=l: cap.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[ar, last].float().cpu())
    ) for l in range(len(layers))]
    try:
        with torch.no_grad():
            backbone(input_ids=ids, attention_mask=mask)
    finally:
        for h in hs:
            h.remove()
    return torch.stack([cap[l] for l in range(len(layers))], 1)


def make_steer_hooks(vec_by_layer, dtype, dev):
    hooks = []
    for l, v in vec_by_layer.items():
        vt = v.to(device=dev, dtype=dtype)
        def fn(m, i, o, vt=vt):
            if isinstance(o, tuple):
                return (o[0] + vt,) + tuple(o[1:])
            return o + vt
        hooks.append((l, fn))
    return hooks


def eval_model(family, key, is_it, pack, out_dir):
    print(f"\n=== EXP2 {key} ({C.MODELS[key]}) ===", flush=True)
    tok, model = C.load_model(key)
    ts = C.token_sets(tok)
    dev = next(model.parameters()).device
    out = {"key": key, "model": C.MODELS[key], "family": family, "is_it": is_it, "prose": {}}

    frames = ["chat", "raw_qa"] if is_it else ["raw_qa"]
    for subset, writers in [("wp", ("human", "gemma", "claude", "glm")),
                            ("books", ("human", "gemma", "gpt", "glm"))]:
        items = pack["prose"][subset]
        sub_out = {}
        for w in writers:
            texts = [it[w] for it in items]
            nlls = passage_nll(tok, model, texts)
            rec = {"nll": nlls, "nll_mean": round(float(np.mean([x for x in nlls if x is not None])), 4)}
            for fr in frames:
                det_strs = [C.format_frames(tok, family, DET_Q.format(t=t), is_it)[fr] for t in texts]
                lp = C.forward_last_logprobs(tok, model, det_strs, batch_size=8)
                says_ai = (1.0 - C.prob_ratio(lp, ts["H"], ts["A"])).tolist()
                rec[f"says_ai_{fr}"] = [round(x, 4) for x in says_ai]
                rec[f"says_ai_{fr}_mean"] = round(float(np.mean(says_ai)), 4)
                rec[f"says_ai_{fr}_mass"] = round(float(C.prob_mass(lp, ts["H"], ts["A"]).mean()), 4)
                if is_it and fr == "chat":
                    r_strs = [C.format_frames(tok, family, RATE_Q.format(t=t), True)["chat"] for t in texts]
                    lp_r = C.forward_last_logprobs(tok, model, r_strs, batch_size=8)
                    pr = lp_r[:, ts["D"]].exp()
                    ratings = ((pr * torch.arange(1, 10)).sum(1) / (pr.sum(1) + 1e-12)).tolist()
                    rec["rating_chat"] = [round(x, 3) for x in ratings]
                    rec["rating_chat_mean"] = round(float(np.mean(ratings)), 3)
            sub_out[w] = rec
            f0 = frames[0]
            print(f"  {subset:5s} {w:7s}: nll={rec['nll_mean']:.3f} says_ai({f0})={rec[f'says_ai_{f0}_mean']:.3f}" +
                  (f" says_ai(raw)={rec['says_ai_raw_qa_mean']:.3f} rating={rec['rating_chat_mean']:.2f}" if is_it else ""),
                  flush=True)
        out["prose"][subset] = sub_out

    if is_it:
        _, layers, _, _, _ = C.get_parts(model)
        NL = len(layers)
        chat_q = [C.format_frames(tok, family, q + C.ONE_HA, True)["chat"] for q in C.SELF_QS]
        raw_q = [C.format_frames(tok, family, q + C.ONE_HA, True)["raw_qa"] for q in C.SELF_QS]
        d_unzip = (get_resids(tok, model, raw_q) - get_resids(tok, model, chat_q)).mean(0)

        steer_layers = sorted(set([int(NL * f) for f in (0.30, 0.45, 0.60, 0.75)]))
        print(f"  2B steering layers: {steer_layers} (NL={NL})", flush=True)

        wp_h = [C.format_frames(tok, family, DET_Q.format(t=it["human"]), True)["chat"] for it in pack["prose"]["wp"][:20]]
        wp_g = [C.format_frames(tok, family, DET_Q.format(t=it["gemma"]), True)["chat"] for it in pack["prose"]["wp"][:20]]
        wp_c = [C.format_frames(tok, family, DET_Q.format(t=it["claude"]), True)["chat"] for it in pack["prose"]["wp"][:20]]
        bk_h = [C.format_frames(tok, family, DET_Q.format(t=it["human"]), True)["chat"] for it in pack["prose"]["books"][:20]]
        bk_g = [C.format_frames(tok, family, DET_Q.format(t=it["gemma"]), True)["chat"] for it in pack["prose"]["books"][:20]]
        bat_keys = ["alive", "born", "body", "human", "ai", "ctrl_toaster", "ctrl_2plus2"]
        bat_strs = [C.format_frames(tok, family, C.BATTERY[k] + C.ONE_YN, True)["chat"] for k in bat_keys]

        def eval_panel(hooks):
            r = {}
            lp_s = C.forward_last_logprobs(tok, model, chat_q, batch_size=12, hooks=hooks)
            r["self_p_human"] = round(float(C.prob_ratio(lp_s, ts["H"], ts["A"]).mean()), 4)
            lp_b = C.forward_last_logprobs(tok, model, bat_strs, batch_size=12, hooks=hooks)
            for k, p in zip(bat_keys, C.prob_ratio(lp_b, ts["Y"], ts["N"]).tolist()):
                r[k] = round(p, 4)
            for name, strs in [("wp_human", wp_h), ("wp_gemma", wp_g), ("wp_claude", wp_c),
                               ("books_human", bk_h), ("books_gemma", bk_g)]:
                lp_p = C.forward_last_logprobs(tok, model, strs, batch_size=10, hooks=hooks)
                r[f"says_ai_{name}"] = round(float((1.0 - C.prob_ratio(lp_p, ts["H"], ts["A"])).mean()), 4)
            return r

        coupling = {"steer_layers": steer_layers, "base": eval_panel(None), "unzip": {}, "random_matched": {}}
        print("  2B base:", coupling["base"], flush=True)
        for alpha in (0.5, 1.0, 1.5):
            vec = {l: d_unzip[l] * (alpha / len(steer_layers)) for l in steer_layers}
            coupling["unzip"][str(alpha)] = eval_panel(make_steer_hooks(vec, torch.bfloat16, dev))
            print(f"  2B unzip a={alpha}:", coupling["unzip"][str(alpha)], flush=True)

            seed_res = []
            for seed in (101, 202, 303):
                gen = torch.Generator().manual_seed(seed)
                rvec = {}
                for l in steer_layers:
                    rv = torch.randn(d_unzip[l].shape, generator=gen)
                    rv = rv * (d_unzip[l].norm() / (rv.norm() + 1e-12))
                    rvec[l] = rv * (alpha / len(steer_layers))
                seed_res.append(eval_panel(make_steer_hooks(rvec, torch.bfloat16, dev)))
            avg_r = {k: round(float(np.mean([sr[k] for sr in seed_res])), 4) for k in seed_res[0]}
            coupling["random_matched"][str(alpha)] = {"mean": avg_r, "seeds": seed_res}
            print(f"  2B rand  a={alpha}:", avg_r, flush=True)

        out["coupling"] = coupling

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"exp2_{key}.json"), "w") as f:
        json.dump(out, f, indent=1)
    C.unload(model, tok)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", help="Model keys to run (default: all 6)")
    ap.add_argument("--pack", default="data/pack.json")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    pack = json.load(open(args.pack))
    only = set(args.models) if args.models else None
    for fam, bkey, ikey in C.FAMILIES:
        for k, is_it in [(bkey, False), (ikey, True)]:
            if only and k not in only:
                continue
            eval_model(fam, k, is_it, pack, args.out_dir)
    print("EXP2_ALL_DONE", flush=True)
