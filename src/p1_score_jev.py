"""Off-GPU Jev semantic scoring & statistical analysis for P1 (OLMo-2 7B realizations).

Reads:
  - experiments/p1_cases_manifest.json
  - results/p1_olmo_realizations.json
Writes:
  - results/p1_analysis.json
  - scratch/p1_blind_val_packet.json & scratch/p1_blind_val_key.json (64-reply stratified audit)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jev_judge import JevJudge, choice  # noqa: E402


SCRATCH = Path(os.environ.get("VALIDATION_DIR", str(ROOT / "results" / "validation")))


def family_cluster_boot(
    fam_vals: dict[str, list[float]],
    *,
    seed: int = 20261127,
    n_boot: int = 2000,
) -> dict[str, float]:
    fams = sorted(fam_vals.keys())
    k = len(fams)
    all_v = [v for f in fams for v in fam_vals[f]]
    mean = sum(all_v) / len(all_v)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s_sum = 0.0
        s_n = 0
        for _ in range(k):
            vs = fam_vals[fams[rng.randrange(k)]]
            s_sum += sum(vs)
            s_n += len(vs)
        boots.append(s_sum / s_n)
    boots.sort()
    # Two-sided bootstrap p-value against 0
    p_pos = sum(1 for b in boots if b <= 0.0) / n_boot
    p_neg = sum(1 for b in boots if b >= 0.0) / n_boot
    p_two = min(1.0, 2.0 * min(p_pos, p_neg))
    return {
        "mean": round(mean, 4),
        "ci95_lo": round(boots[int(0.025 * n_boot)], 4),
        "ci95_hi": round(boots[int(0.975 * n_boot)], 4),
        "ci90_lo": round(boots[int(0.05 * n_boot)], 4),
        "ci90_hi": round(boots[int(0.95 * n_boot)], 4),
        "p_boot_two": round(p_two, 4),
        "n_obs": len(all_v),
        "n_families": k,
    }


def holm_bonferroni(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    running_max = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, p * (m - i))
        running_max = max(running_max, adj)
        out[k] = round(running_max, 4)
    return out


def analyze_teacher_forcing(tf: dict[str, Any]) -> dict[str, Any]:
    arms = tf["arms"]
    summary: dict[str, Any] = {}
    arm_names = list(arms.keys())

    for arm in arm_names:
        rows = arms[arm]
        by_key = {(r["policy"], r["prefill_name"], r["realization"], r["pair_id"], r["fact_key"]): r for r in rows}
        summary[arm] = {}
        for metric in ("m_full", "m_mean", "m_div"):
            summary[arm][metric] = {}
            for pref in ("none", "verdict"):
                # Aggregate across all 4 realizations (paired within family)
                fwd_disc_fam: dict[str, list[float]] = defaultdict(list)
                rev_disc_fam: dict[str, list[float]] = defaultdict(list)
                rev_shift_fam: dict[str, list[float]] = defaultdict(list)
                norule_fam: dict[str, list[float]] = defaultdict(list)
                by_real: dict[str, dict[str, float]] = {}

                reals = sorted({r["realization"] for r in rows})
                for rname in reals:
                    r_fwd, r_rev, r_nr = [], [], []
                    pair_ids = sorted({r["pair_id"] for r in rows if r["policy"] == "forward" and r["prefill_name"] == pref and r["realization"] == rname})
                    for pid in pair_ids:
                        fa = by_key[("forward", pref, rname, pid, "fact_a")]
                        fb = by_key[("forward", pref, rname, pid, "fact_b")]
                        ra = by_key[("reversed", pref, rname, pid, "fact_a")]
                        rb = by_key[("reversed", pref, rname, pid, "fact_b")]
                        na = by_key[("no_rule", pref, rname, pid, "fact_a")]
                        nb = by_key[("no_rule", pref, rname, pid, "fact_b")]
                        fam = fa["family"]
                        # Forward: fact_a allowed, fact_b denied -> disc = m_denied - m_allowed = fb - fa
                        d_fwd = fb[metric] - fa[metric]
                        # Reversed: fact_b allowed, fact_a denied -> disc = m_denied - m_allowed = ra - rb
                        d_rev = ra[metric] - rb[metric]
                        # Reversal shift on same fact: (ra - fa) and (fb - rb) both >0 if following rule
                        s_rev = 0.5 * ((ra[metric] - fa[metric]) + (fb[metric] - rb[metric]))
                        nr_val = 0.5 * (na[metric] + nb[metric])
                        fwd_disc_fam[fam].append(d_fwd)
                        rev_disc_fam[fam].append(d_rev)
                        rev_shift_fam[fam].append(s_rev)
                        norule_fam[fam].append(nr_val)
                        r_fwd.append(d_fwd)
                        r_rev.append(d_rev)
                        r_nr.append(nr_val)
                    by_real[rname] = {
                        "fwd_disc": round(sum(r_fwd) / len(r_fwd), 4),
                        "rev_disc": round(sum(r_rev) / len(r_rev), 4),
                        "no_rule_bias": round(sum(r_nr) / len(r_nr), 4),
                    }

                summary[arm][metric][pref] = {
                    "fwd_disc": family_cluster_boot(fwd_disc_fam),
                    "rev_disc": family_cluster_boot(rev_disc_fam),
                    "rev_shift": family_cluster_boot(rev_shift_fam),
                    "no_rule_bias": family_cluster_boot(norule_fam),
                    "by_realization": by_real,
                }

    # Key paired arm contrasts on primary metrics (m_full, m_div)
    contrasts_spec = [
        ("sft/base_vs_base/base", "sft/base", "base/base"),
        ("base/sft_vs_base/base", "base/sft", "base/base"),
        ("sft/sft_vs_base/base", "sft/sft", "base/base"),
        ("sft/sft_vs_sft/base", "sft/sft", "sft/base"),
        ("opener36/base_vs_base/base", "opener36/base", "base/base"),
        ("base/opener36_vs_base/base", "base/opener36", "base/base"),
        ("opener36/opener36_vs_base/base", "opener36/opener36", "base/base"),
        ("complement36_vs_base/base", "complement36/complement36", "base/base"),
        ("svd4/base_vs_base/base", "svd4/base", "base/base"),
        ("svd4/svd4_vs_base/base", "svd4/svd4", "base/base"),
        ("svd4/base_vs_rand_rank4/base", "svd4/base", "rand_rank4/base"),
        ("opener36/base_vs_rand36/base", "opener36/base", "rand36/base"),
    ]
    contrasts: dict[str, Any] = {}
    for metric in ("m_full", "m_mean", "m_div"):
        contrasts[metric] = {}
        for pref in ("none", "verdict"):
            contrasts[metric][pref] = {}
            pvals_fwd = {}
            for cname, a1, a2 in contrasts_spec:
                if a1 not in arms or a2 not in arms:
                    continue
                k1 = {(r["policy"], r["prefill_name"], r["realization"], r["pair_id"], r["fact_key"]): r for r in arms[a1]}
                k2 = {(r["policy"], r["prefill_name"], r["realization"], r["pair_id"], r["fact_key"]): r for r in arms[a2]}
                df_fwd: dict[str, list[float]] = defaultdict(list)
                df_rev: dict[str, list[float]] = defaultdict(list)
                df_nr: dict[str, list[float]] = defaultdict(list)
                reals = sorted({r["realization"] for r in arms[a1]})
                pids = sorted({r["pair_id"] for r in arms[a1] if r["policy"] == "forward" and r["prefill_name"] == pref and r["realization"] == reals[0]})
                for rname in reals:
                    for pid in pids:
                        fam = k1[("forward", pref, rname, pid, "fact_a")]["family"]
                        d1_f = k1[("forward", pref, rname, pid, "fact_b")][metric] - k1[("forward", pref, rname, pid, "fact_a")][metric]
                        d2_f = k2[("forward", pref, rname, pid, "fact_b")][metric] - k2[("forward", pref, rname, pid, "fact_a")][metric]
                        df_fwd[fam].append(d1_f - d2_f)

                        d1_r = k1[("reversed", pref, rname, pid, "fact_a")][metric] - k1[("reversed", pref, rname, pid, "fact_b")][metric]
                        d2_r = k2[("reversed", pref, rname, pid, "fact_a")][metric] - k2[("reversed", pref, rname, pid, "fact_b")][metric]
                        df_rev[fam].append(d1_r - d2_r)

                        n1 = 0.5 * (k1[("no_rule", pref, rname, pid, "fact_a")][metric] + k1[("no_rule", pref, rname, pid, "fact_b")][metric])
                        n2 = 0.5 * (k2[("no_rule", pref, rname, pid, "fact_a")][metric] + k2[("no_rule", pref, rname, pid, "fact_b")][metric])
                        df_nr[fam].append(n1 - n2)

                res_f = family_cluster_boot(df_fwd)
                res_r = family_cluster_boot(df_rev)
                res_n = family_cluster_boot(df_nr)
                contrasts[metric][pref][cname] = {
                    "fwd_disc_diff": res_f,
                    "rev_disc_diff": res_r,
                    "no_rule_bias_diff": res_n,
                }
                pvals_fwd[cname] = res_f["p_boot_two"]
            holm = holm_bonferroni(pvals_fwd)
            for cname, adj_p in holm.items():
                contrasts[metric][pref][cname]["fwd_disc_diff"]["p_holm"] = adj_p

    return {"per_arm": summary, "contrasts": contrasts}


def parse_explicit_prefix(reply: str, send_lbl: str | None, keep_lbl: str | None) -> str:
    if not send_lbl or not keep_lbl:
        return "na"
    txt = reply.strip()
    m = re.match(r"^(?:Verdict:\s*)?([A-Za-z]+)\b", txt, flags=re.IGNORECASE)
    if not m:
        return "other"
    tok = m.group(1).lower()
    if tok == send_lbl.lower():
        return "send_out"
    if tok == keep_lbl.lower():
        return "keep_storage"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    manifest = json.loads((ROOT / "experiments" / "p1_cases_manifest.json").read_text())
    p1 = json.loads((ROOT / "results" / "p1_olmo_realizations.json").read_text())
    row_by_id = {r["row_id"]: r for r in manifest["rows"]}

    print("[P1-Analyze] Analyzing teacher-forcing grid...", flush=True)
    tf_analysis = analyze_teacher_forcing(p1["teacher_forcing"])

    # Prepare Jev semantic scoring items across ALL generated replies (deduplicated by state!)
    judge = JevJudge(workers=args.workers)
    p1_questions = {
        "action_choice": choice(
            "Which action does clerk_reply actually recommend for the parcel?",
            {
                "send_out": "Recommends sending/releasing/approving/clearing/dispatching the parcel out (or starts with the send_label and endorses sending).",
                "keep_storage": "Recommends keeping/holding/blocking/pausing/denying the parcel in storage (or starts with the keep_label and endorses keeping).",
                "hedged_or_both": "Contradicts itself, recommends both sending and keeping, or hedges without choosing one action.",
                "nonresponsive_or_garbled": "Empty, repeats the prompt, nonsensical, garbled, or refuses to answer the warehouse ticket.",
            },
        )
    }
    uniq_jobs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    state_to_idx: dict[str, int] = {}
    meta_list = []
    for ck, arm_dict in p1["generations"].items():
        for arm_name, rows in arm_dict.items():
            for g in rows:
                r = row_by_id[g["row_id"]]
                full_reply = (r["prefill"] + " " + g["reply"]).strip() if r["prefill"] else g["reply"]
                state = {
                    "ai_instructions": r["system"],
                    "user_message": r["user"],
                    "clerk_reply": full_reply,
                }
                if r["send_label"] and r["keep_label"]:
                    state["label_meaning"] = f"'{r['send_label']}' means SEND THE PARCEL OUT; '{r['keep_label']}' means KEEP IN STORAGE."
                sk = json.dumps(state, sort_keys=True, separators=(",", ":"))
                if sk not in state_to_idx:
                    state_to_idx[sk] = len(uniq_jobs)
                    uniq_jobs.append((state, p1_questions))
                meta_list.append((ck, arm_name, g["row_id"], g, r, full_reply, state_to_idx[sk]))

    print(f"[P1-Analyze] Scoring {len(uniq_jobs)} unique replies (out of {len(meta_list)} total) with Jev (workers={args.workers})...", flush=True)
    uniq_scored = judge.decide_batch(uniq_jobs, workers=args.workers, progress_every=500)
    print(f"[P1-Analyze] Jev usage: {judge.usage}", flush=True)

    # Assemble per-cell x per-arm metrics
    by_cell_arm: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    all_scored_flat = []
    for ck, arm_name, rid, g, r, full_reply, uidx in meta_list:
        sc = uniq_scored[uidx] or {}
        act = sc.get("action_choice", {}).get("value", "error") if isinstance(sc.get("action_choice"), dict) else "error"
        pref_act = parse_explicit_prefix(g["reply"], r["send_label"], r["keep_label"])
        rec = {
            "cell": ck,
            "arm": arm_name,
            "row_id": rid,
            "pair_id": r["pair_id"],
            "family": r["family"],
            "policy": r["policy"],
            "wording": r["wording"],
            "prefill_name": r["prefill_name"],
            "allowed": r["allowed"],
            "action_choice": act,
            "prefix_choice": pref_act,
            "reply": full_reply,
            "system": r["system"],
            "user": r["user"],
        }
        by_cell_arm[ck][arm_name].append(rec)
        all_scored_flat.append(rec)

    gen_summary: dict[str, dict[str, Any]] = {}
    for ck, arm_dict in sorted(by_cell_arm.items()):
        gen_summary[ck] = {}
        for arm_name, recs in arm_dict.items():
            policy = recs[0]["policy"]
            wording = recs[0]["wording"]
            counts = Counter(x["action_choice"] for x in recs)
            n = len(recs)
            garbled_rate = counts["nonresponsive_or_garbled"] / n
            hedged_rate = counts["hedged_or_both"] / n

            if policy == "no_rule":
                keep_fam: dict[str, list[float]] = defaultdict(list)
                send_fam: dict[str, list[float]] = defaultdict(list)
                for x in recs:
                    keep_fam[x["family"]].append(1.0 if x["action_choice"] == "keep_storage" else 0.0)
                    send_fam[x["family"]].append(1.0 if x["action_choice"] == "send_out" else 0.0)
                gen_summary[ck][arm_name] = {
                    "n": n,
                    "counts": dict(counts),
                    "keep_rate": family_cluster_boot(keep_fam),
                    "send_rate": family_cluster_boot(send_fam),
                    "garbled_rate": round(garbled_rate, 4),
                    "hedged_rate": round(hedged_rate, 4),
                }
            else:
                # Group by pair_id (each pair has 1 allowed + 1 denied)
                by_pair: dict[str, dict[bool, dict[str, Any]]] = defaultdict(dict)
                for x in recs:
                    by_pair[x["pair_id"]][bool(x["allowed"])] = x
                den_keep_fam: dict[str, list[float]] = defaultdict(list)
                all_send_fam: dict[str, list[float]] = defaultdict(list)
                disc_fam: dict[str, list[float]] = defaultdict(list)
                pair_acc_fam: dict[str, list[float]] = defaultdict(list)
                pref_pair_acc_fam: dict[str, list[float]] = defaultdict(list)
                for pid, pair in by_pair.items():
                    ra, rd = pair[True], pair[False]
                    fam = ra["family"]
                    dk = 1.0 if rd["action_choice"] == "keep_storage" else 0.0
                    asend = 1.0 if ra["action_choice"] == "send_out" else 0.0
                    ak = 1.0 if ra["action_choice"] == "keep_storage" else 0.0
                    den_keep_fam[fam].append(dk)
                    all_send_fam[fam].append(asend)
                    disc_fam[fam].append(dk - ak)
                    pair_acc_fam[fam].append(1.0 if (dk == 1.0 and asend == 1.0) else 0.0)
                    if wording != "label_free":
                        p_ok = 1.0 if (rd["prefix_choice"] == "keep_storage" and ra["prefix_choice"] == "send_out") else 0.0
                        pref_pair_acc_fam[fam].append(p_ok)
                entry: dict[str, Any] = {
                    "n": n,
                    "counts": dict(counts),
                    "denied_keep": family_cluster_boot(den_keep_fam),
                    "allowed_send": family_cluster_boot(all_send_fam),
                    "disc_gap": family_cluster_boot(disc_fam),
                    "pair_acc": family_cluster_boot(pair_acc_fam),
                    "garbled_rate": round(garbled_rate, 4),
                    "hedged_rate": round(hedged_rate, 4),
                }
                if wording != "label_free":
                    entry["prefix_pair_acc"] = family_cluster_boot(pref_pair_acc_fam)
                gen_summary[ck][arm_name] = entry

    # Key paired generation contrasts on primary cells
    gen_contrasts: dict[str, Any] = {}
    for ck in sorted(by_cell_arm.keys()):
        arm_dict = by_cell_arm[ck]
        if arm_dict[next(iter(arm_dict))][0]["policy"] == "no_rule":
            continue
        pairs_to_test = [
            ("sft/base_vs_base/base", "sft/base", "base/base"),
            ("base/sft_vs_base/base", "base/sft", "base/base"),
            ("sft/sft_vs_base/base", "sft/sft", "base/base"),
            ("sft/sft_vs_sft/base", "sft/sft", "sft/base"),
            ("opener36/base_vs_base/base", "opener36/base", "base/base"),
            ("opener36/opener36_vs_base/base", "opener36/opener36", "base/base"),
            ("complement36_vs_base/base", "complement36/complement36", "base/base"),
            ("svd4/base_vs_base/base", "svd4/base", "base/base"),
            ("svd4/svd4_vs_base/base", "svd4/svd4", "base/base"),
            ("opener36/base_vs_rand36/base", "opener36/base", "rand36/base"),
            ("svd4/base_vs_rand_rank4/base", "svd4/base", "rand_rank4/base"),
        ]
        gen_contrasts[ck] = {}
        for cname, a1, a2 in pairs_to_test:
            if a1 not in arm_dict or a2 not in arm_dict:
                continue
            r1 = {x["row_id"]: x for x in arm_dict[a1]}
            r2 = {x["row_id"]: x for x in arm_dict[a2]}
            # Pair accuracy diff & disc gap diff clustered by family
            by_pair_ids = sorted({x["pair_id"] for x in arm_dict[a1]})
            dk_diff: dict[str, list[float]] = defaultdict(list)
            pa_diff: dict[str, list[float]] = defaultdict(list)
            dg_diff: dict[str, list[float]] = defaultdict(list)
            for pid in by_pair_ids:
                p1_rows = [x for x in arm_dict[a1] if x["pair_id"] == pid]
                p2_rows = [x for x in arm_dict[a2] if x["pair_id"] == pid]
                a1_all = next(x for x in p1_rows if x["allowed"])
                a1_den = next(x for x in p1_rows if not x["allowed"])
                a2_all = next(x for x in p2_rows if x["allowed"])
                a2_den = next(x for x in p2_rows if not x["allowed"])
                fam = a1_all["family"]
                dk1 = 1.0 if a1_den["action_choice"] == "keep_storage" else 0.0
                dk2 = 1.0 if a2_den["action_choice"] == "keep_storage" else 0.0
                as1 = 1.0 if a1_all["action_choice"] == "send_out" else 0.0
                as2 = 1.0 if a2_all["action_choice"] == "send_out" else 0.0
                ak1 = 1.0 if a1_all["action_choice"] == "keep_storage" else 0.0
                ak2 = 1.0 if a2_all["action_choice"] == "keep_storage" else 0.0
                dk_diff[fam].append(dk1 - dk2)
                dg_diff[fam].append((dk1 - ak1) - (dk2 - ak2))
                pa_diff[fam].append((1.0 if dk1 and as1 else 0.0) - (1.0 if dk2 and as2 else 0.0))
            gen_contrasts[ck][cname] = {
                "denied_keep_diff": family_cluster_boot(dk_diff),
                "disc_gap_diff": family_cluster_boot(dg_diff),
                "pair_acc_diff": family_cluster_boot(pa_diff),
            }

    # Export 64-reply stratified blind validation packet
    rng = random.Random(20261127)
    by_act: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen = set()
    for x in all_scored_flat:
        k = (x["user"], x["reply"])
        if k not in seen:
            seen.add(k)
            by_act[x["action_choice"]].append(x)
    for k in by_act:
        rng.shuffle(by_act[k])
    chosen = []
    buckets = sorted(by_act.keys(), key=lambda b: len(by_act[b]))
    while len(chosen) < 64 and any(by_act[b] for b in buckets):
        for b in buckets:
            if by_act[b] and len(chosen) < 64:
                chosen.append(by_act[b].pop())
    rng.shuffle(chosen)
    pkt = []
    key_map = {}
    for i, x in enumerate(chosen, 1):
        bid = f"P1VAL-{i:03d}"
        pkt.append({
            "blind_id": bid,
            "system": x["system"],
            "user": x["user"],
            "reply": x["reply"],
        })
        key_map[bid] = {"cell": x["cell"], "arm": x["arm"], "row_id": x["row_id"], "jev_action": x["action_choice"]}
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "p1_blind_val_packet.json").write_text(json.dumps({"items": pkt}, indent=2) + "\n")
    (SCRATCH / "p1_blind_val_key.json").write_text(json.dumps(key_map, indent=2) + "\n")

    out = {
        "hardware": p1["hardware"],
        "revisions": p1["revisions"],
        "head_stats": p1["head_stats"],
        "parity": p1["parity"],
        "jev_stats": judge.usage,
        "first_token_opener36_coverage": p1["first_token_opener36_coverage"],
        "teacher_forcing_analysis": tf_analysis,
        "generation_summary": gen_summary,
        "generation_contrasts": gen_contrasts,
    }
    out_path = ROOT / "results" / "p1_analysis.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[P1-Analyze] Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
