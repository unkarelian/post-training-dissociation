"""Analyze Experiment 1, 2, 3 results for 'The Face of Another' hypothesis."""
import argparse, json, os, numpy as np


def load(out_dir, name):
    p = os.path.join(out_dir, name)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    OUT = args.results_dir

    print("================ EXP 1: LOOSE-FITTING CLOTHES ON SKIN ================")
    for fam in ("gemma", "llama", "qwen"):
        b = load(OUT, f"exp1_{fam}_base.json")
        it = load(OUT, f"exp1_{fam}_it.json")
        if not b or not it:
            print(f"  [{fam}] missing")
            continue
        print(f"\n--- {fam.upper()} ---")
        print("  1A. Frame Unzipping (IT vs Base):")
        print(f"  {'frame':14s} | {'p_hum(B/IT)':13s} | {'alive(B/IT)':13s} | {'born(B/IT)':13s} | {'body(B/IT)':13s} | {'ai(B/IT)':13s} | {'toast/2+2(IT)':13s}")
        for fr in ("chat", "chat_swap", "chat_no_role", "raw_qa", "raw_dialogue", "raw_interview"):
            bh, ih = b["self_id"][fr]["p_human"], it["self_id"][fr]["p_human"]
            ba, ia = b["battery"][fr]["alive"], it["battery"][fr]["alive"]
            bb, ib = b["battery"][fr]["born"], it["battery"][fr]["born"]
            bbo, ibo = b["battery"][fr]["body"], it["battery"][fr]["body"]
            bai, iai = b["battery"][fr]["ai"], it["battery"][fr]["ai"]
            t, tw = it["battery"][fr]["ctrl_toaster"], it["battery"][fr]["ctrl_2plus2"]
            print(f"  {fr:14s} | {bh:.2f} / {ih:.2f}   | {ba:.2f} / {ia:.2f}   | {bb:.2f} / {ib:.2f}   | {bbo:.2f} / {ibo:.2f}   | {bai:.2f} / {iai:.2f}   | {t:.2f} / {tw:.2f}")

        print("  1B. Logit Lens (IT chat vs raw_qa):")
        ll = it["logit_lens"]["chat"]
        L = len(ll["self_p_human"])
        idxs = [int(round(f * (L - 1))) for f in (0.0, 0.25, 0.5, 0.65, 0.75, 0.85, 0.92, 1.0)]
        print("    layers :", idxs)
        for k in ("self_p_human", "alive", "born", "ai"):
            print(f"    IT chat {k:12s}:", [round(ll[k][i], 2) for i in idxs])
        llr = it["logit_lens"]["raw_qa"]
        for k in ("self_p_human", "alive"):
            print(f"    IT raw  {k:12s}:", [round(llr[k][i], 2) for i in idxs])

        print("  1C. Causal Header Attention Knockout (IT chat):")
        ko = it["knockout"]
        print(f"    header tokens: {ko['self_p_human']['header_tokens_sample']}")
        for k in ("self_p_human", "alive", "born", "body", "human", "ai", "ctrl_toaster", "ctrl_2plus2"):
            v = ko[k]
            print(f"    {k:14s}: base={v['base']:.3f}  ko_header={v['ko_header']:.3f} (Δ{v['ko_header']-v['base']:+.3f})  ko_ctrl={v['ko_ctrl']:.3f} (Δ{v['ko_ctrl']-v['base']:+.3f})")

    print("\n================ EXP 2: IDENTITY INVERSION & CAUSAL COUPLING ================")
    for fam in ("gemma", "llama", "qwen"):
        b = load(OUT, f"exp2_{fam}_base.json")
        it = load(OUT, f"exp2_{fam}_it.json")
        if not b or not it:
            print(f"  [{fam}] missing")
            continue
        print(f"\n--- {fam.upper()} ---")
        for sub, writers in [("wp", ("human", "gemma", "claude", "glm")), ("books", ("human", "gemma", "gpt", "glm"))]:
            print(f"  2A. {sub}:")
            for w in writers:
                rb, ri = b["prose"][sub][w], it["prose"][sub][w]
                print(f"    {w:7s}: Base(raw)={rb['says_ai_raw_qa_mean']:.3f} (nll {rb['nll_mean']:.2f}) | "
                      f"IT(chat)={ri['says_ai_chat_mean']:.3f} IT(raw)={ri['says_ai_raw_qa_mean']:.3f} (nll {ri['nll_mean']:.2f}) rating={ri['rating_chat_mean']:.2f}")
            h_nll = it["prose"][sub]["human"]["nll"]
            h_ai = it["prose"][sub]["human"]["says_ai_chat"]
            pairs_h, pairs_a = [], []
            for w in writers[1:]:
                a_nll = it["prose"][sub][w]["nll"]
                a_ai = it["prose"][sub][w]["says_ai_chat"]
                for hn, ha in zip(h_nll, h_ai):
                    if hn is None:
                        continue
                    cands = [(abs(hn - an), aa) for an, aa in zip(a_nll, a_ai) if an is not None and abs(hn - an) <= 0.15]
                    if cands:
                        cands.sort()
                        pairs_h.append(ha); pairs_a.append(cands[0][1])
            if pairs_h:
                print(f"    NLL-matched (±0.15 nats, n={len(pairs_h)}): human says_ai={np.mean(pairs_h):.3f} vs AI says_ai={np.mean(pairs_a):.3f} (Δ={np.mean(pairs_h)-np.mean(pairs_a):+.3f})")

        c = it["coupling"]
        print(f"  2B. Causal Coupling (steering layers {c['steer_layers']}):")
        def fmt(d):
            inv_wp = d["says_ai_wp_human"] - 0.5 * (d["says_ai_wp_gemma"] + d["says_ai_wp_claude"])
            inv_bk = d["says_ai_books_human"] - d["says_ai_books_gemma"]
            return (f"p_hum={d['self_p_human']:.2f} alive={d['alive']:.2f} born={d['born']:.2f} ai={d['ai']:.2f} | "
                    f"wp(H={d['says_ai_wp_human']:.2f},G={d['says_ai_wp_gemma']:.2f},C={d['says_ai_wp_claude']:.2f},inv={inv_wp:+.2f}) "
                    f"bk(H={d['says_ai_books_human']:.2f},G={d['says_ai_books_gemma']:.2f},inv={inv_bk:+.2f}) | 2+2={d['ctrl_2plus2']:.2f}")
        print("    base      :", fmt(c["base"]))
        for a in ("0.5", "1.0", "1.5"):
            print(f"    unzip {a:3s} :", fmt(c["unzip"][a]))
            print(f"    rand  {a:3s} :", fmt(c["random_matched"][a]["mean"]))

    print("\n================ EXP 3: BASE <-> IT WEIGHT GRAFTING ================")
    for fam in ("gemma", "llama", "qwen"):
        g = load(OUT, f"exp3_{fam}.json")
        if not g:
            print(f"  [{fam}] missing")
            continue
        print(f"\n--- {fam.upper()} (NL={g['NL']}, bands={g['bands']}) ---")
        print(f"  {'graft':34s} | {'ai':4s} {'aliv':4s} {'born':4s} {'phum':4s} | {'wpH':4s} {'wpG':4s} {'wpC':4s} {'inv':5s} | {'refH':4s} {'refB':4s} | {'t/2+2':9s}")
        for k, r in g["grafts"].items():
            inv = r["says_ai_wp_human"] - 0.5 * (r["says_ai_wp_gemma"] + r["says_ai_wp_claude"])
            print(f"  {k:34s} | {r['ai_chat']:.2f} {r['alive_chat']:.2f} {r['born_chat']:.2f} {r['self_p_human_chat']:.2f} | "
                  f"{r['says_ai_wp_human']:.2f} {r['says_ai_wp_gemma']:.2f} {r['says_ai_wp_claude']:.2f} {inv:+.2f} | "
                  f"{r['refuse_harmful']:.2f} {r['refuse_benign']:.2f} | {r['ctrl_toaster_chat']:.2f}/{r['ctrl_2plus2_chat']:.2f}")


if __name__ == "__main__":
    main()
