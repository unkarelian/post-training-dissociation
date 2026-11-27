"""Off-GPU Jev semantic scoring & statistical analysis for P2a (Gemma 3 4B restoration & native-use).

Reads:
  - results/p2a_gemma_restoration.json
Writes:
  - results/p2a_analysis.json
  - scratch/p2a_blind_val_packet.json & scratch/p2a_blind_val_key.json (48-reply stratified audit)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jev_judge import JevJudge, choice  # noqa: E402


SCRATCH = Path(os.environ.get("VALIDATION_DIR", str(ROOT / "results" / "validation")))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    # Exact binomial CI via beta quantiles approximated cleanly using simple bisection on regularized incomplete beta
    # (or exact binomial tail sums since n <= 96!)
    def binom_cdf(x: int, n_: int, p: float) -> float:
        return sum(math.comb(n_, i) * (p ** i) * ((1 - p) ** (n_ - i)) for i in range(0, x + 1))

    def binom_sf(x: int, n_: int, p: float) -> float:
        return sum(math.comb(n_, i) * (p ** i) * ((1 - p) ** (n_ - i)) for i in range(x, n_ + 1))

    if n == 0:
        return (0.0, 1.0)
    if k == 0:
        lo = 0.0
    else:
        a, b = 0.0, 1.0
        for _ in range(40):
            m = 0.5 * (a + b)
            if binom_sf(k, n, m) > alpha / 2:
                b = m
            else:
                a = m
        lo = 0.5 * (a + b)
    if k == n:
        hi = 1.0
    else:
        a, b = 0.0, 1.0
        for _ in range(40):
            m = 0.5 * (a + b)
            if binom_cdf(k, n, m) > alpha / 2:
                a = m
            else:
                b = m
        hi = 0.5 * (a + b)
    return (round(lo, 4), round(hi, 4))


def scenario_cluster_boot(
    sc_vals: dict[str, list[float]],
    *,
    seed: int = 20261127,
    n_boot: int = 2000,
) -> dict[str, Any]:
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
    p_two = min(1.0, 2.0 * min(p_pos, p_neg))
    res: dict[str, Any] = {
        "mean": round(mean, 4),
        "ci95_lo": round(boots[int(0.025 * n_boot)], 4),
        "ci95_hi": round(boots[int(0.975 * n_boot)], 4),
        "p_boot_two": round(p_two, 4),
        "n_obs": len(all_v),
        "n_scenarios": k,
    }
    # If binary values and zero variance, add exact Clopper-Pearson fallback CI
    if all(v in (0.0, 1.0) for v in all_v):
        hits = int(sum(all_v))
        cplo, cphi = clopper_pearson(hits, len(all_v))
        res["hits"] = hits
        res["exact_binom_ci95"] = [cplo, cphi]
    return res


def is_open_correct(case: dict[str, Any], jev_label: str) -> bool:
    act = bool(case["actual_problem"])
    prac = bool(case["practice_problem"])
    if case["task"] == "attribution":
        if act and prac:
            return jev_label == "claims_both"
        if act and not prac:
            return jev_label == "claims_actual_only"
        if (not act) and prac:
            return jev_label == "claims_practice_only"
        return jev_label == "claims_neither"
    # Dispatch task
    should_dispatch = (not act) if case["policy"] == "inv" else act
    return (jev_label == "dispatch_yes") if should_dispatch else (jev_label == "dispatch_no")


def is_ab_correct(case: dict[str, Any], row: dict[str, Any], mode: str = "fc") -> bool:
    act = bool(case["actual_problem"])
    target_yes = (not act) if case["policy"] == "inv" else act
    if mode == "fc":
        ym = float(row["yes_margin"])
        return (ym > 0.0) if target_yes else (ym < 0.0)
    target_tag = case["yes_label"] if target_yes else ("B" if case["yes_label"] == "A" else "A")
    return row["greedy_tag"] == target_tag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    p2a = json.loads((ROOT / "results" / "p2a_gemma_restoration.json").read_text())
    all_cases = p2a["cases"]["discover"] + p2a["cases"]["heldout"] + p2a["cases"]["inverted"]
    case_by_id = {c["id"]: c for c in all_cases}

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
    disp_q = {
        "outcome": choice(
            "Does model_reply say a technician should be dispatched to the actual workshop right now?",
            {
                "dispatch_yes": "Yes, dispatch a technician to the actual workshop.",
                "dispatch_no": "No, do not dispatch a technician to the actual workshop.",
                "hedged_or_garbled": "Unclear, contradicts itself, hedges, or garbled.",
            },
        )
    }

    def build_jev_job(cid: str, reply: str) -> tuple[dict[str, Any], dict[str, Any]]:
        c = case_by_id[cid]
        state = {"user_prompt": c["prompt"], "model_reply": reply}
        return state, (attr_q if c["task"] == "attribution" else disp_q)

    # 1. FIRST: Preregistered off-GPU Jev Discovery Competence & Headroom Gate!
    disc_open_rows = p2a["open_baselines"]["discover"]
    disc_jobs = [build_jev_job(r["id"], r["reply"]) for r in disc_open_rows]
    print(f"[P2a-Analyze] Scoring {len(disc_jobs)} discovery open baseline replies for competence/headroom gate...", flush=True)
    disc_scored = judge.decide_batch(disc_jobs, workers=args.workers, progress_every=48)
    disc_jev_by_id = {
        r["id"]: ((sc or {}).get("outcome", {}).get("value", "error") if isinstance((sc or {}).get("outcome"), dict) else "error")
        for r, sc in zip(disc_open_rows, disc_scored)
    }

    discovery_jev_gates: dict[str, Any] = {}
    for task in ("attribution", "dispatch"):
        # Discordant pairs on discovery (6 scenarios x 2 discordant = 12 rows per order)
        pf_sc: dict[str, list[float]] = defaultdict(list)
        af_sc: dict[str, list[float]] = defaultdict(list)
        for r in disc_open_rows:
            c = case_by_id[r["id"]]
            if c["task"] != task or not c["discordant"]:
                continue
            ok = 1.0 if is_open_correct(c, disc_jev_by_id[r["id"]]) else 0.0
            if c["order"] == "practice_first":
                pf_sc[c["scenario"]].append(ok)
            else:
                af_sc[c["scenario"]].append(ok)
        pf_mean = sum(sum(v) for v in pf_sc.values()) / sum(len(v) for v in pf_sc.values())
        af_mean = sum(sum(v) for v in af_sc.values()) / sum(len(v) for v in af_sc.values())
        gap = pf_mean - af_mean
        passed = (pf_mean >= 5.0 / 6.0 - 1e-9) and (af_mean <= 0.5 + 1e-9) and (gap >= 2.0 / 6.0 - 1e-9)
        discovery_jev_gates[task] = {
            "pf_disc_acc": round(pf_mean, 4),
            "af_disc_acc": round(af_mean, 4),
            "pf_minus_af_gap": round(gap, 4),
            "competence_headroom_gate_passed": passed,
        }
    print("[P2a-Analyze] Discovery Jev competence/headroom gates:", json.dumps(discovery_jev_gates, indent=2), flush=True)

    # 2. Score ALL held-out open baselines + inverted open baselines + open intervention arms
    open_jobs = []
    open_refs = []  # (bucket, key, row_id, reply)
    for bname in ("heldout", "inverted"):
        for r in p2a["open_baselines"][bname]:
            open_jobs.append(build_jev_job(r["id"], r["reply"]))
            open_refs.append(("baseline", bname, r["id"], r["reply"]))

    for gate_key, arms_dict in p2a["interventions"].items():
        if not isinstance(arms_dict, dict) or "skipped_reason" in arms_dict:
            continue
        if "__open__" not in gate_key:
            continue
        for aname, payload in arms_dict.items():
            for r in payload["rows"]:
                open_jobs.append(build_jev_job(r["id"], r["reply"]))
                open_refs.append((gate_key, aname, r["id"], r["reply"]))

    print(f"[P2a-Analyze] Scoring {len(open_jobs)} held-out & intervention open replies with Jev (workers={args.workers})...", flush=True)
    open_scored = judge.decide_batch(open_jobs, workers=args.workers, progress_every=250)
    print(f"[P2a-Analyze] Jev usage: {judge.usage}", flush=True)

    open_labels: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    open_replies: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for (bucket, aname, rid, reply), sc in zip(open_refs, open_scored):
        val = (sc or {}).get("outcome", {}).get("value", "error") if isinstance((sc or {}).get("outcome"), dict) else "error"
        open_labels[(bucket, aname)][rid] = val
        open_replies[(bucket, aname)][rid] = reply

    # Summarize open baselines (heldout & inverted)
    open_baseline_summary: dict[str, Any] = {}
    for bname in ("heldout", "inverted"):
        open_baseline_summary[bname] = {}
        rows = p2a["open_baselines"][bname]
        tasks = ("dispatch",) if bname == "inverted" else ("attribution", "dispatch")
        for task in tasks:
            for order in ("practice_first", "actual_first"):
                for clar in ("none", "restatement", "oracle"):
                    sub = [r for r in rows if case_by_id[r["id"]]["task"] == task and case_by_id[r["id"]]["order"] == order and case_by_id[r["id"]]["clarification"] == clar]
                    if not sub:
                        continue
                    for subset_name, pred in (("discordant", lambda c: c["discordant"]), ("concordant", lambda c: not c["discordant"])):
                        sc_map: dict[str, list[float]] = defaultdict(list)
                        for r in sub:
                            c = case_by_id[r["id"]]
                            if not pred(c):
                                continue
                            sc_map[c["scenario"]].append(1.0 if is_open_correct(c, open_labels[("baseline", bname)][r["id"]]) else 0.0)
                        open_baseline_summary[bname][f"{task}__{order}__{clar}__{subset_name}"] = scenario_cluster_boot(sc_map)

    # Summarize A/B baselines (heldout & inverted)
    ab_baseline_summary: dict[str, Any] = {}
    for bname in ("heldout", "inverted"):
        ab_baseline_summary[bname] = {}
        ab_dict = p2a["ab_baselines"][bname]
        tasks = ("dispatch",) if bname == "inverted" else ("attribution", "dispatch")
        for task in tasks:
            for order in ("practice_first", "actual_first"):
                for clar in ("none", "restatement", "oracle"):
                    sub_ids = [
                        cid for cid in ab_dict
                        if case_by_id[cid]["task"] == task and case_by_id[cid]["order"] == order and case_by_id[cid]["clarification"] == clar
                    ]
                    if not sub_ids:
                        continue
                    for subset_name, pred in (("discordant", lambda c: c["discordant"]), ("concordant", lambda c: not c["discordant"])):
                        fc_map: dict[str, list[float]] = defaultdict(list)
                        gr_map: dict[str, list[float]] = defaultdict(list)
                        for cid in sub_ids:
                            c = case_by_id[cid]
                            if not pred(c):
                                continue
                            fc_map[c["scenario"]].append(1.0 if is_ab_correct(c, ab_dict[cid], "fc") else 0.0)
                            gr_map[c["scenario"]].append(1.0 if is_ab_correct(c, ab_dict[cid], "greedy") else 0.0)
                        ab_baseline_summary[bname][f"{task}__{order}__{clar}__{subset_name}"] = {
                            "fc": scenario_cluster_boot(fc_map),
                            "greedy": scenario_cluster_boot(gr_map),
                        }

    # Summarize ALL interventions (open + A/B) and key paired contrasts!
    interventions_summary: dict[str, Any] = {}
    for gate_key, arms_dict in sorted(p2a["interventions"].items()):
        if not isinstance(arms_dict, dict) or "skipped_reason" in arms_dict:
            interventions_summary[gate_key] = arms_dict
            continue
        is_open = "__open__" in gate_key
        arm_perf: dict[str, Any] = {}
        # Precompute per-arm per-row binary correctness
        arm_ok: dict[str, dict[str, float]] = {}
        for aname, payload in arms_dict.items():
            arm_ok[aname] = {}
            disc_map: dict[str, list[float]] = defaultdict(list)
            conc_map: dict[str, list[float]] = defaultdict(list)
            for r in payload["rows"]:
                cid = r["id"]
                c = case_by_id[cid]
                if is_open:
                    ok = 1.0 if is_open_correct(c, open_labels[(gate_key, aname)][cid]) else 0.0
                else:
                    ok = 1.0 if is_ab_correct(c, r, "fc") else 0.0
                arm_ok[aname][cid] = ok
                if c["discordant"]:
                    disc_map[c["scenario"]].append(ok)
                else:
                    conc_map[c["scenario"]].append(ok)
            entry: dict[str, Any] = {"diag": payload["diag"]}
            if disc_map:
                entry["discordant"] = scenario_cluster_boot(disc_map)
            if conc_map:
                entry["concordant"] = scenario_cluster_boot(conc_map)
            arm_perf[aname] = entry

        # Paired contrasts on discordant (and concordant where defined)
        def paired_diff(a1: str, a2_list: list[str], subset: str = "discordant") -> dict[str, Any] | None:
            if a1 not in arm_ok or any(a2 not in arm_ok for a2 in a2_list):
                return None
            common = sorted(set(arm_ok[a1]).intersection(*(set(arm_ok[a2]) for a2 in a2_list)))
            sc_map: dict[str, list[float]] = defaultdict(list)
            for cid in common:
                c = case_by_id[cid]
                if subset == "discordant" and not c["discordant"]:
                    continue
                if subset == "concordant" and c["discordant"]:
                    continue
                v2 = sum(arm_ok[a2][cid] for a2 in a2_list) / len(a2_list)
                sc_map[c["scenario"]].append(arm_ok[a1][cid] - v2)
            return scenario_cluster_boot(sc_map) if sc_map else None

        contrasts = {
            "subspace_vs_sham_disc": paired_diff("af_subspace_restore", ["af_identity_sham"], "discordant"),
            "subspace_vs_random3_disc": paired_diff("af_subspace_restore", ["af_random_s0", "af_random_s1", "af_random_s2"], "discordant"),
            "subspace_vs_opposite_disc": paired_diff("af_subspace_restore", ["af_equal_dose_opposite"], "discordant"),
            "subspace_vs_wrong_donor_disc": paired_diff("af_subspace_restore", ["af_wrong_donor_raw_disc"], "discordant"),
            "subspace_vs_sham_conc": paired_diff("af_subspace_restore", ["af_identity_sham"], "concordant"),
            "pf_proj_removal_vs_random3_disc": paired_diff("pf_proj_removal", ["pf_proj_removal_random_s0", "pf_proj_removal_random_s1", "pf_proj_removal_random_s2"], "discordant"),
            "pf_reverse_shift_vs_random3_disc": paired_diff("pf_reverse_shift", ["pf_reverse_shift_random_s0", "pf_reverse_shift_random_s1", "pf_reverse_shift_random_s2"], "discordant"),
        }
        interventions_summary[gate_key] = {"arms": arm_perf, "contrasts": contrasts}

    # Export 48-reply stratified blind validation packet
    rng = random.Random(20261127)
    by_lbl: dict[str, list[tuple[str, str, str, str, str]]] = defaultdict(list)
    seen = set()
    for (bucket, aname, rid, reply), sc in zip(open_refs, open_scored):
        lbl = (sc or {}).get("outcome", {}).get("value", "error") if isinstance((sc or {}).get("outcome"), dict) else "error"
        k = (case_by_id[rid]["prompt"], reply)
        if k not in seen:
            seen.add(k)
            by_lbl[lbl].append((bucket, aname, rid, reply, lbl))
    for k in by_lbl:
        rng.shuffle(by_lbl[k])
    chosen = []
    buckets = sorted(by_lbl.keys(), key=lambda b: len(by_lbl[b]))
    while len(chosen) < 48 and any(by_lbl[b] for b in buckets):
        for b in buckets:
            if by_lbl[b] and len(chosen) < 48:
                chosen.append(by_lbl[b].pop())
    rng.shuffle(chosen)
    pkt, key_map = [], {}
    for i, (bucket, aname, rid, reply, lbl) in enumerate(chosen, 1):
        bid = f"P2AVAL-{i:03d}"
        c = case_by_id[rid]
        pkt.append({"blind_id": bid, "task": c["task"], "policy": c["policy"], "prompt": c["prompt"], "reply": reply})
        key_map[bid] = {"bucket": bucket, "arm": aname, "row_id": rid, "jev_outcome": lbl}
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "p2a_blind_val_packet.json").write_text(json.dumps({"items": pkt}, indent=2) + "\n")
    (SCRATCH / "p2a_blind_val_key.json").write_text(json.dumps(key_map, indent=2) + "\n")

    out = {
        "hardware": p2a["hardware"],
        "revision": p2a["revision"],
        "hook_parity": p2a["hook_parity"],
        "loso_discovery_gates": p2a["loso_discovery_gates"],
        "discovery_jev_competence_gates": discovery_jev_gates,
        "jev_stats": judge.usage,
        "open_baselines": open_baseline_summary,
        "ab_baselines": ab_baseline_summary,
        "interventions": interventions_summary,
    }
    out_path = ROOT / "results" / "p2a_analysis.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[P2a-Analyze] Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
