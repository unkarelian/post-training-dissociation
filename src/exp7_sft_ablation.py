"""Experiment 7: 3-Arm Causal SFT Data Ablation on OLMo-2-1124-7B Base
Directly tests WHAT inside SFT creates Prose Identity Inversion (0.50 -> 0.62/0.57):
  - Arm A (identity_refusal): allenai/olmo-2-hard-coded + allenai/coconot refusals
  - Arm B (synthetic_gpt4):   allenai/WildChat-1M (English, GPT-4, AI self-mentions stripped)
  - Arm C (human_sft):        HuggingFaceH4/no_robots (human-written English, AI mentions stripped)
All 3 arms use identical LoRA (r=16, all attn+mlp linear layers), 2,000 examples, 250 steps x bs=8,
assistant-only CE loss, LR=2e-4 cosine, bf16, on exact OLMo-2 chat template.
"""
import argparse, json, math, os, random, re, torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import common as C
from exp3_graft import is_refusal
from exp4_v2_verify import eval_single_prose, eval_pairwise, generate_texts, OPEN_ID_PROMPTS

AI_MENTION_RX = re.compile(
    r"\b(as an ai|i am an ai|i'm an ai|language model|artificial intelligence|openai|chatgpt|"
    r"gpt-4|gpt-3|claude|anthropic|olmo|allenai|ai assistant|virtual assistant|not a human|"
    r"not human|not alive|computer program|machine learning model|trained by|developed by)\b",
    re.I,
)


def build_arm_datasets(n_per_arm=2000, seed=42):
    rng = random.Random(seed)

    # Arm A: olmo-2-hard-coded (repeated 4x) + coconot refusals
    print("Building Arm A (identity_refusal)...", flush=True)
    hc_ds = load_dataset("allenai/olmo-2-hard-coded", split="train")
    hc_pairs = []
    for r in hc_ds:
        msgs = r.get("messages", [])
        if len(msgs) >= 2 and msgs[0].get("role") == "user" and msgs[1].get("role") == "assistant":
            u, a = msgs[0]["content"].strip(), msgs[1]["content"].strip()
            if u and a:
                hc_pairs.append((u, a))
    arm_a = (hc_pairs * 4)[:880]

    cc_ds = load_dataset("allenai/coconot", "original", split="train", streaming=True)
    for r in cc_ds:
        u, a = (r.get("prompt") or "").strip(), (r.get("response") or "").strip()
        if u and a and len(a) > 20:
            arm_a.append((u, a))
        if len(arm_a) >= n_per_arm:
            break
    rng.shuffle(arm_a)
    arm_a = arm_a[:n_per_arm]

    # Arm B: WildChat-1M (English, non-toxic, GPT-4, NO AI self-mentions)
    print("Building Arm B (synthetic_gpt4, AI mentions stripped)...", flush=True)
    wc_ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    arm_b = []
    for r in wc_ds:
        if r.get("language") != "English" or r.get("toxic"):
            continue
        if "gpt-4" not in (r.get("model") or "").lower():
            continue
        conv = r.get("conversation") or []
        if len(conv) < 2 or conv[0].get("role") != "user" or conv[1].get("role") != "assistant":
            continue
        u, a = conv[0]["content"].strip(), conv[1]["content"].strip()
        if not u or not a or len(a) < 40 or len(u) > 1200 or len(a) > 2000:
            continue
        if AI_MENTION_RX.search(u) or AI_MENTION_RX.search(a):
            continue
        arm_b.append((u, a))
        if len(arm_b) >= n_per_arm:
            break

    # Arm C: No Robots (100% human-written English SFT, NO AI self-mentions)
    print("Building Arm C (human_sft, AI mentions stripped)...", flush=True)
    nr_ds = load_dataset("HuggingFaceH4/no_robots", split="train")
    arm_c = []
    for r in nr_ds:
        msgs = r.get("messages") or []
        # Skip system messages if present; take first user->assistant pair
        u = a = ""
        for i, m in enumerate(msgs):
            if m.get("role") == "user" and i + 1 < len(msgs) and msgs[i + 1].get("role") == "assistant":
                u, a = m["content"].strip(), msgs[i + 1]["content"].strip()
                break
        if not u or not a or len(a) < 40 or len(u) > 1200 or len(a) > 2000:
            continue
        if AI_MENTION_RX.search(u) or AI_MENTION_RX.search(a):
            continue
        arm_c.append((u, a))
    rng.shuffle(arm_c)
    arm_c = arm_c[:n_per_arm]

    print(f"Dataset sizes: Arm A={len(arm_a)} (hc={len(hc_pairs)}), Arm B={len(arm_b)}, Arm C={len(arm_c)}", flush=True)
    return {
        "armA_identity_refusal": arm_a,
        "armB_synthetic_gpt4":   arm_b,
        "armC_human_sft":        arm_c,
    }


def tokenize_pair(tok, u, a, max_len=512):
    bos = tok.bos_token or ""
    eos = tok.eos_token or ""
    prefix = f"{bos}<|user|>\n{u}\n<|assistant|>\n"
    full = f"{prefix}{a}{eos}"
    p_ids = tok.encode(prefix, add_special_tokens=False)
    f_ids = tok.encode(full, add_special_tokens=False)[:max_len]
    labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[min(len(p_ids), len(f_ids)):]
    if all(x == -100 for x in labels):
        return None
    return f_ids, labels


def train_lora_arm(base_model, tok, pairs, micro_bs=4, grad_accum=2, lr=2e-4, max_len=512):
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()
    base_model.enable_input_require_grads()
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base_model, lora_cfg)
    model.train()
    dev = next(model.parameters()).device

    encoded = [x for x in (tokenize_pair(tok, u, a, max_len) for u, a in pairs) if x is not None]
    eff_bs = micro_bs * grad_accum
    n_steps = len(encoded) // eff_bs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)

    losses = []
    for step in range(n_steps):
        if step < 15:
            cur_lr = lr * (step + 1) / 15
        else:
            prog = (step - 15) / max(1, n_steps - 15)
            cur_lr = lr * 0.5 * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        opt.zero_grad()
        step_loss = 0.0
        for ga in range(grad_accum):
            start = step * eff_bs + ga * micro_bs
            batch = encoded[start : start + micro_bs]
            ml = max(len(x[0]) for x in batch)
            ids = torch.full((micro_bs, ml), tok.pad_token_id, dtype=torch.long, device=dev)
            mask = torch.zeros((micro_bs, ml), dtype=torch.long, device=dev)
            labs = torch.full((micro_bs, ml), -100, dtype=torch.long, device=dev)
            for i, (f_ids, lb) in enumerate(batch):
                ids[i, :len(f_ids)] = torch.tensor(f_ids, device=dev)
                mask[i, :len(f_ids)] = 1
                labs[i, :len(lb)] = torch.tensor(lb, device=dev)
            out = model(input_ids=ids, attention_mask=mask, labels=labs)
            (out.loss / grad_accum).backward()
            step_loss += float(out.loss.item()) / grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(step_loss)
        if (step + 1) % 50 == 0 or step == n_steps - 1:
            print(f"    step {step+1}/{n_steps} loss={sum(losses[-25:])/len(losses[-25:]):.4f} lr={cur_lr:.2e}", flush=True)

    model.eval()
    base_model.gradient_checkpointing_disable()
    base_model.config.use_cache = True
    return model, round(sum(losses[-25:]) / len(losses[-25:]), 4)


def eval_arm(tok, model, pack, name):
    # Note: PeftModel wraps base_model.model -> C.get_parts needs unwrapped or standard access
    eval_m = model.base_model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") else model
    ts = C.token_sets(tok)
    rec = {"name": name, "self_id": {}, "battery": {}, "single": {}, "pairwise": {}}

    for fr in ("raw_qa", "chat"):
        strs_ha = [C.format_frames(tok, "olmo", q + C.ONE_HA, True)[fr] for q in C.SELF_QS]
        lp = C.forward_last_logprobs(tok, eval_m, strs_ha, batch_size=12)
        rec["self_id"][fr] = round(float(C.prob_ratio(lp, ts["H"], ts["A"]).mean()), 4)

        bkeys = ["alive", "born", "body", "human", "ai", "ctrl_toaster", "ctrl_2plus2"]
        bstrs = [C.format_frames(tok, "olmo", C.BATTERY[k] + C.ONE_YN, True)[fr] for k in bkeys]
        lp_b = C.forward_last_logprobs(tok, eval_m, bstrs, batch_size=12)
        rec["battery"][fr] = {k: round(p, 4) for k, p in zip(bkeys, C.prob_ratio(lp_b, ts["Y"], ts["N"]).tolist())}

        rec["single"][fr] = eval_single_prose(tok, eval_m, "olmo", True, pack, fr, n=40)
        rec["pairwise"][fr] = eval_pairwise(tok, eval_m, "olmo", True, pack, fr, n=40)
        pw = rec["pairwise"][fr]
        print(f"  [{name:22s} | {fr:6s}] p_hum={rec['self_id'][fr]:.2f} alive={rec['battery'][fr]['alive']:.2f} "
              f"born={rec['battery'][fr]['born']:.2f} ai={rec['battery'][fr]['ai']:.2f} toast={rec['battery'][fr]['ctrl_toaster']:.2f} | "
              f"PAIR wp_cl={pw['wp_human_vs_claude']['p_pick_human_as_ai']:.3f} {pw['wp_human_vs_claude']['ci95']} | "
              f"bk_gpt={pw['books_human_vs_gpt']['p_pick_human_as_ai']:.3f} {pw['books_human_vs_gpt']['ci95']}", flush=True)

    id_rep = generate_texts(tok, model, "olmo", OPEN_ID_PROMPTS, max_new=120, batch_size=8)
    harm_rep = generate_texts(tok, model, "olmo", pack["harmful"], max_new=120, batch_size=10)
    ben_rep = generate_texts(tok, model, "olmo", pack["benign"], max_new=120, batch_size=10)
    rec["refuse_harmful_kw"] = round(sum(1 for r in harm_rep if is_refusal(r)) / len(harm_rep), 4)
    rec["refuse_benign_kw"] = round(sum(1 for r in ben_rep if is_refusal(r)) / len(ben_rep), 4)
    rec["open_identity"] = [{"prompt": p, "reply": r} for p, r in zip(OPEN_ID_PROMPTS, id_rep)]
    rec["harmful_replies"] = [{"prompt": p, "reply": r} for p, r in zip(pack["harmful"], harm_rep)]
    rec["benign_replies"] = [{"prompt": p, "reply": r} for p, r in zip(pack["benign"], ben_rep)]
    print(f"  [{name:22s} | OPEN  ] refH={rec['refuse_harmful_kw']:.2f} refB={rec['refuse_benign_kw']:.2f} | Who={id_rep[0][:120]!r}", flush=True)
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="data/pack.json")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--n-per-arm", type=int, default=2000)
    args = ap.parse_args()
    pack = json.load(open(args.pack))
    os.makedirs(args.out_dir, exist_ok=True)

    arms = build_arm_datasets(n_per_arm=args.n_per_arm)
    out_path = os.path.join(args.out_dir, "exp7_sft_ablation.json")
    out = json.load(open(out_path)) if os.path.exists(out_path) else {"base_model": C.MODELS["olmo_base"], "n_per_arm": args.n_per_arm, "arms": {}}

    if "base_unedited" not in out["arms"]:
        tok, base_m = C.load_model("olmo_base")
        out["arms"]["base_unedited"] = eval_arm(tok, base_m, pack, "base_unedited")
        C.unload(base_m, tok)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=1)

    for arm_name, pairs in arms.items():
        if arm_name in out["arms"]:
            print(f"Skipping already completed {arm_name}", flush=True)
            continue
        print(f"\n================ TRAINING {arm_name} (n={len(pairs)}) ================", flush=True)
        tok, base_m = C.load_model("olmo_base")
        peft_m, final_loss = train_lora_arm(base_m, tok, pairs)
        merged_m = peft_m.merge_and_unload()
        merged_m.eval()
        rec = eval_arm(tok, merged_m, pack, arm_name)
        rec["final_train_loss"] = final_loss
        rec["sample_train_pairs"] = [{"u": u[:150], "a": a[:200]} for u, a in pairs[:3]]
        out["arms"][arm_name] = rec
        with open(out_path, "w") as f:
            json.dump(out, f, indent=1)
        C.unload(merged_m, tok)

    print("EXP7_SFT_ABLATION_DONE", flush=True)
