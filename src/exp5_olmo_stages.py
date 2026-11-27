"""Experiment 5: OLMo-2-1124-7B Stage-by-Stage Dissection (Base -> SFT -> DPO -> Instruct)
Part 5A: Trace self-identity, prose identity inversion (single + order-balanced pairwise in BOTH
         raw_qa and chat), and 120-token open-ended identity/refusal across all 4 training stages.
Part 5B: Base <-> Instruct Attention vs MLP weight grafting on OLMo-2-7B with 120-token free text.
"""
import argparse, json, os, shutil, numpy as np, torch
import common as C
from exp3_graft import copy_slice, is_refusal
from exp4_v2_verify import eval_single_prose, eval_pairwise, generate_texts, OPEN_ID_PROMPTS


def eval_stage(key, is_it, pack):
    print(f"\n=== EXP5 STAGE: {key} ({C.MODELS[key]}) ===", flush=True)
    tok, model = C.load_model(key)
    ts = C.token_sets(tok)
    rec = {"key": key, "model": C.MODELS[key], "self_id": {}, "battery": {}, "single": {}, "pairwise": {}}

    for fr in ("raw_qa", "chat"):
        strs_ha = [C.format_frames(tok, "olmo", q + C.ONE_HA, is_it)[fr] for q in C.SELF_QS]
        lp = C.forward_last_logprobs(tok, model, strs_ha, batch_size=12)
        rec["self_id"][fr] = round(float(C.prob_ratio(lp, ts["H"], ts["A"]).mean()), 4)

        bkeys = ["alive", "born", "body", "human", "ai", "ctrl_toaster", "ctrl_2plus2"]
        bstrs = [C.format_frames(tok, "olmo", C.BATTERY[k] + C.ONE_YN, is_it)[fr] for k in bkeys]
        lp_b = C.forward_last_logprobs(tok, model, bstrs, batch_size=12)
        rec["battery"][fr] = {k: round(p, 4) for k, p in zip(bkeys, C.prob_ratio(lp_b, ts["Y"], ts["N"]).tolist())}

        rec["single"][fr] = eval_single_prose(tok, model, "olmo", is_it, pack, fr, n=40)
        rec["pairwise"][fr] = eval_pairwise(tok, model, "olmo", is_it, pack, fr, n=40)
        pw = rec["pairwise"][fr]
        print(f"  [{fr:6s}] p_hum={rec['self_id'][fr]:.2f} alive={rec['battery'][fr]['alive']:.2f} "
              f"born={rec['battery'][fr]['born']:.2f} ai={rec['battery'][fr]['ai']:.2f} toast={rec['battery'][fr]['ctrl_toaster']:.2f} | "
              f"PAIR wp_cl={pw['wp_human_vs_claude']['p_pick_human_as_ai']:.3f} {pw['wp_human_vs_claude']['ci95']} | "
              f"bk_gpt={pw['books_human_vs_gpt']['p_pick_human_as_ai']:.3f} {pw['books_human_vs_gpt']['ci95']}", flush=True)

    # 120-token open-ended free text (chat frame)
    id_rep = generate_texts(tok, model, "olmo", OPEN_ID_PROMPTS, max_new=120, batch_size=8)
    harm_rep = generate_texts(tok, model, "olmo", pack["harmful"], max_new=120, batch_size=10)
    ben_rep = generate_texts(tok, model, "olmo", pack["benign"], max_new=120, batch_size=10)
    rec["refuse_harmful_kw"] = round(sum(1 for r in harm_rep if is_refusal(r)) / len(harm_rep), 4)
    rec["refuse_benign_kw"] = round(sum(1 for r in ben_rep if is_refusal(r)) / len(ben_rep), 4)
    rec["open_identity"] = [{"prompt": p, "reply": r} for p, r in zip(OPEN_ID_PROMPTS, id_rep)]
    rec["harmful_replies"] = [{"prompt": p, "reply": r} for p, r in zip(pack["harmful"], harm_rep)]
    rec["benign_replies"] = [{"prompt": p, "reply": r} for p, r in zip(pack["benign"], ben_rep)]
    print(f"  OPEN: refH={rec['refuse_harmful_kw']:.2f} refB={rec['refuse_benign_kw']:.2f} | Who={id_rep[0][:120]!r}", flush=True)
    C.unload(model, tok)
    return rec


def run_olmo_grafts(pack, out_dir):
    print("\n=== EXP5 PART 5B: OLMo-2-7B Base <-> Instruct GRAFTS ===", flush=True)
    tok, model_it = C.load_model("olmo_it", device="cuda")
    ts = C.token_sets(tok)
    _, layers, _, _, _ = C.get_parts(model_it)
    NL = len(layers)
    b1, b2 = NL // 3, (2 * NL) // 3
    bands = {"early": list(range(0, b1)), "mid": list(range(b1, b2)), "late": list(range(b2, NL)), "all": list(range(0, NL))}

    sd_it = {k: v.detach().cpu().clone() for k, v in model_it.state_dict().items()}
    _, model_base_cpu = C.load_model("olmo_base", device="cpu")
    sd_base = {k: v.detach().cpu().clone() for k, v in model_base_cpu.state_dict().items()}
    del model_base_cpu

    def restore(sd):
        with torch.no_grad():
            for k, v in model_it.state_dict().items():
                if k in sd:
                    v.copy_(sd[k].to(v.device, v.dtype))

    specs = [
        ("pure_it",                sd_it,   None,    None,    None),
        ("pure_base",              sd_base, None,    None,    None),
        ("it_plus_base__early_mlp",sd_it,   sd_base, "early", "mlp"),
        ("it_plus_base__all_mlp",  sd_it,   sd_base, "all",   "mlp"),
        ("it_plus_base__all_attn", sd_it,   sd_base, "all",   "attn"),
        ("base_plus_it__early_mlp",sd_base, sd_it,   "early", "mlp"),
        ("base_plus_it__all_mlp",  sd_base, sd_it,   "all",   "mlp"),
        ("base_plus_it__all_attn", sd_base, sd_it,   "all",   "attn"),
    ]
    out = {"family": "olmo", "NL": NL, "grafts": {}}
    for gname, host_sd, donor_sd, bname, comp in specs:
        restore(host_sd)
        if donor_sd is not None:
            copy_slice(model_it, donor_sd, bands[bname], comp)

        strs_ha = [C.format_frames(tok, "olmo", q + C.ONE_HA, True)["chat"] for q in C.SELF_QS]
        lp = C.forward_last_logprobs(tok, model_it, strs_ha, batch_size=12)
        p_hum = round(float(C.prob_ratio(lp, ts["H"], ts["A"]).mean()), 4)
        bkeys = ["alive", "born", "body", "human", "ai", "ctrl_toaster", "ctrl_2plus2"]
        bstrs = [C.format_frames(tok, "olmo", C.BATTERY[k] + C.ONE_YN, True)["chat"] for k in bkeys]
        lp_b = C.forward_last_logprobs(tok, model_it, bstrs, batch_size=12)
        bat = {k: round(p, 4) for k, p in zip(bkeys, C.prob_ratio(lp_b, ts["Y"], ts["N"]).tolist())}
        pw = eval_pairwise(tok, model_it, "olmo", True, pack, "chat", n=30)
        id_rep = generate_texts(tok, model_it, "olmo", OPEN_ID_PROMPTS, max_new=120, batch_size=8)
        harm_rep = generate_texts(tok, model_it, "olmo", pack["harmful"], max_new=120, batch_size=10)
        ben_rep = generate_texts(tok, model_it, "olmo", pack["benign"], max_new=120, batch_size=10)
        ref_h = round(sum(1 for r in harm_rep if is_refusal(r)) / len(harm_rep), 4)
        ref_b = round(sum(1 for r in ben_rep if is_refusal(r)) / len(ben_rep), 4)

        out["grafts"][gname] = {
            "self_p_human_chat": p_hum, "battery_chat": bat,
            "pairwise_chat": {k: {kk: vv for kk, vv in v.items() if kk != "per_item"} for k, v in pw.items()},
            "refuse_harmful_kw": ref_h, "refuse_benign_kw": ref_b,
            "open_identity": [{"prompt": p, "reply": r} for p, r in zip(OPEN_ID_PROMPTS, id_rep)],
            "harmful_replies": [{"prompt": p, "reply": r} for p, r in zip(pack["harmful"], harm_rep)],
            "benign_replies": [{"prompt": p, "reply": r} for p, r in zip(pack["benign"], ben_rep)],
        }
        print(f"  {gname:26s}: p_hum={p_hum:.2f} alive={bat['alive']:.2f} born={bat['born']:.2f} | "
              f"pw_wp_cl={pw['wp_human_vs_claude']['p_pick_human_as_ai']:.3f} pw_bk_gpt={pw['books_human_vs_gpt']['p_pick_human_as_ai']:.3f} | "
              f"refH={ref_h:.2f} refB={ref_b:.2f}", flush=True)
        print(f"    Who? -> {id_rep[0][:130]!r} | Alive? -> {id_rep[1][:110]!r}", flush=True)
        with open(os.path.join(out_dir, "exp5_olmo_graft.json"), "w") as f:
            json.dump(out, f, indent=1)

    C.unload(model_it, tok)
    del sd_it, sd_base


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="data/pack.json")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--clear-mid", action="store_true", help="Clear SFT/DPO HF cache after evaluating them")
    args = ap.parse_args()
    pack = json.load(open(args.pack))
    os.makedirs(args.out_dir, exist_ok=True)

    stages_out = {"family": "olmo", "stages": {}}
    for key, is_it in [("olmo_sft", True), ("olmo_dpo", True)]:
        stages_out["stages"][key] = eval_stage(key, is_it, pack)
        with open(os.path.join(args.out_dir, "exp5_olmo_stages.json"), "w") as f:
            json.dump(stages_out, f, indent=1)
        if args.clear_mid:
            shutil.rmtree(os.path.expanduser("~/.cache/huggingface"), ignore_errors=True)

    for key, is_it in [("olmo_base", False), ("olmo_it", True)]:
        stages_out["stages"][key] = eval_stage(key, is_it, pack)
        with open(os.path.join(args.out_dir, "exp5_olmo_stages.json"), "w") as f:
            json.dump(stages_out, f, indent=1)

    run_olmo_grafts(pack, args.out_dir)
    print("EXP5_OLMO_DONE", flush=True)
