"""Off-GPU Jev scoring & statistical analysis for P4 (Gemma-3-4B full-donor vs 1D probe paradox).

Reads:
  - results/p4_gemma_paradox.json
Writes:
  - results/p4_analysis.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jev_judge import JevJudge, choice  # noqa: E402


def sc_boot(sc_vals: dict[str, list[float]], *, seed: int = 20261127, n_boot: int = 2000) -> dict[str, float]:
    scs = sorted(sc_vals.keys())
    k = len(scs)
    all_v = [v for s in scs for v in sc_vals[s]]
    mean = sum(all_v) / len(all_v)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        ss, sn = 0.0, 0
        for _ in range(k):
            vs = sc_vals[scs[rng.randrange(k)]]
            ss += sum(vs)
            sn += len(vs)
        boots.append(ss / sn)
    boots.sort()
    p_pos = sum(1 for b in boots if b <= 0.0) / n_boot
    p_neg = sum(1 for b in boots if b >= 0.0) / n_boot
    return {
        "mean": round(mean, 4),
        "ci95_lo": round(boots[int(0.025 * n_boot)], 4),
        "ci95_hi": round(boots[int(0.975 * n_boot)], 4),
        "p_boot_two": round(min(1.0, 2.0 * min(p_pos, p_neg)), 4),
        "n_obs": len(all_v),
        "n_scenarios": k,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    p4 = json.loads((ROOT / "results" / "p4_gemma_paradox.json").read_text())

    # 1. Layer sweep 2x2 summary with 12-scenario clustered CIs + truth_effect vs letter_effect decomposition!
    sweep_out: dict[str, Any] = {}
    for task, layers_d in p4["layer_sweep_2x2"].items():
        sweep_out[task] = {}
        for lkey, arms in layers_d.items():
            arm_stats = {}
            by_id: dict[str, dict[str, dict[str, Any]]] = {}
            for aname, payload in arms.items():
                sc_fc = defaultdict(list)
                sc_gr = defaultdict(list)
                by_id[aname] = {}
                for r in payload["rows"]:
                    sc_fc[r["scenario"]].append(1.0 if r["fc_correct"] else 0.0)
                    sc_gr[r["scenario"]].append(1.0 if r["greedy_correct"] else 0.0)
                    by_id[aname][r["id"]] = r
                arm_stats[aname] = {"fc": sc_boot(sc_fc), "greedy": sc_boot(sc_gr)}

            # Paired 2x2 factorial contrasts:
            # truth_effect = 0.5 * [(ST_SL - WT_SL) + (ST_FL - WT_FL)]
            # letter_effect = 0.5 * [(ST_SL - ST_FL) + (WT_SL - WT_FL)]
            t_sc, l_sc = defaultdict(list), defaultdict(list)
            for rid, r0 in by_id["same_truth_same_letter"].items():
                sc = r0["scenario"]
                st_sl = 1.0 if by_id["same_truth_same_letter"][rid]["fc_correct"] else 0.0
                st_fl = 1.0 if by_id["same_truth_flip_letter"][rid]["fc_correct"] else 0.0
                wt_sl = 1.0 if by_id["wrong_truth_same_letter"][rid]["fc_correct"] else 0.0
                wt_fl = 1.0 if by_id["wrong_truth_flip_letter"][rid]["fc_correct"] else 0.0
                t_sc[sc].append(0.5 * ((st_sl - wt_sl) + (st_fl - wt_fl)))
                l_sc[sc].append(0.5 * ((st_sl - st_fl) + (wt_sl - wt_fl)))

            sweep_out[task][lkey] = {
                "arms": arm_stats,
                "truth_main_effect_fc": sc_boot(t_sc),
                "letter_main_effect_fc": sc_boot(l_sc),
            }

    # 2. Subspace decomposition summary with 12-scenario clustered CIs + contrasts vs random!
    sub_out: dict[str, Any] = {}
    for lkey, payload in p4["subspace_decomp"].items():
        arms = payload["arms"]
        arm_stats = {}
        by_id = {}
        for aname, apay in arms.items():
            d_sc, c_sc = defaultdict(list), defaultdict(list)
            by_id[aname] = {}
            for r in apay["rows"]:
                by_id[aname][r["id"]] = r
                if r["discordant"]:
                    d_sc[r["scenario"]].append(1.0 if r["fc_correct"] else 0.0)
                else:
                    c_sc[r["scenario"]].append(1.0 if r["fc_correct"] else 0.0)
            arm_stats[aname] = {"discordant_fc": sc_boot(d_sc), "concordant_fc": sc_boot(c_sc)}

        # Contrasts for proj_pc_k vs mean(rand_pc_k_s0..2)
        contrasts = {}
        for k in (1, 4, 16, 32):
            pk = f"proj_pc_{k}"
            rks = [f"rand_pc_{k}_s{si}" for si in range(3)]
            if pk in by_id and all(rk in by_id for rk in rks):
                df = defaultdict(list)
                for rid, r0 in by_id[pk].items():
                    if not r0["discordant"]:
                        continue
                    v1 = 1.0 if r0["fc_correct"] else 0.0
                    v2 = sum(1.0 if by_id[rk][rid]["fc_correct"] else 0.0 for rk in rks) / 3.0
                    df[r0["scenario"]].append(v1 - v2)
                contrasts[f"pc_{k}_vs_rand3_disc"] = sc_boot(df)
        sub_out[lkey] = {"geometry": payload["geometry"], "arms": arm_stats, "contrasts": contrasts}

    # 3. Open-Attribution KV-cache / question-suffix patching Jev scoring!
    case_by_id = {c["id"]: c for c in p4["open_cases"]}
    judge = JevJudge(workers=args.workers)
    attr_q = {
        "outcome": choice(
            "Which workshop(s) does model_reply claim need a repair right now?",
            {
                "claims_actual_only": "Only the actual workshop needs a repair.",
                "claims_practice_only": "Only the fictional practice workshop needs a repair.",
                "claims_both": "Both the actual workshop and the fictional practice workshop need a repair.",
                "claims_neither": "Neither workshop needs a repair.",
                "ambiguous_or_garbled": "Unclear, contradicts itself, refuses, or garbled.",
            },
        )
    }

    jobs, refs = [], []
    for aname, apay in p4["open_kv_patching"].items():
        for r in apay["rows"]:
            c = case_by_id[r["id"]]
            jobs.append(({"user_prompt": c["prompt"], "model_reply": r["reply"]}, attr_q))
            refs.append((aname, r))

    print(f"[P4-Analyze] Scoring {len(jobs)} open KV-patching replies with Jev...", flush=True)
    scored = judge.decide_batch(jobs, workers=args.workers, progress_every=48)
    print(f"[P4-Analyze] Jev usage: {judge.usage}", flush=True)

    open_out: dict[str, Any] = {}
    open_by_arm: dict[str, dict[str, float]] = defaultdict(dict)
    for (aname, r), sc in zip(refs, scored):
        lbl = (sc or {}).get("outcome", {}).get("value", "error") if isinstance((sc or {}).get("outcome"), dict) else "error"
        act, prac = bool(r["actual_problem"]), bool(r["practice_problem"])
        expected = "claims_actual_only" if (act and not prac) else ("claims_practice_only" if (prac and not act) else ("claims_both" if act else "claims_neither"))
        open_by_arm[aname][r["id"]] = 1.0 if lbl == expected else 0.0

    for aname, apay in p4["open_kv_patching"].items():
        sc_m = defaultdict(list)
        for r in apay["rows"]:
            sc_m[r["scenario"]].append(open_by_arm[aname][r["id"]])
        open_out[aname] = {"diag": apay["diag"], "discordant_acc": sc_boot(sc_m)}

    out = {
        "hardware": p4["hardware"],
        "revision": p4["revision"],
        "hook_parity": p4["hook_parity"],
        "jev_stats": judge.usage,
        "layer_sweep_2x2": sweep_out,
        "subspace_decomp": sub_out,
        "open_kv_patching": open_out,
    }
    out_path = ROOT / "results" / "p4_analysis.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[P4-Analyze] Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
