"""Off-GPU Jev semantic scoring & statistical analysis for P3 (5-model lm_head generalization).

Reads:
  - experiments/p1_cases_manifest.json (regenerates if absent)
  - results/p3_head_generalization.json
Writes:
  - results/p3_analysis.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jev_judge import JevJudge, choice  # noqa: E402


def family_cluster_boot(fam_vals: dict[str, list[float]], *, seed: int = 20261127, n_boot: int = 2000) -> dict[str, float]:
    fams = sorted(fam_vals.keys())
    k = len(fams)
    all_v = [v for f in fams for v in fam_vals[f]]
    mean = sum(all_v) / len(all_v)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        ss, sn = 0.0, 0
        for _ in range(k):
            vs = fam_vals[fams[rng.randrange(k)]]
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
        "n_families": k,
    }


def analyze_tf(tf: dict[str, Any] | None) -> dict[str, Any] | None:
    if tf is None:
        return None
    arms = tf["arms"]
    out: dict[str, Any] = {"boundary_fallback_count": tf.get("boundary_fallback_count", 0), "per_arm": {}, "contrasts": {}}
    for arm, rows in arms.items():
        by_k = {(r["policy"], r["realization"], r["pair_id"], r["fact_key"]): r for r in rows}
        reals = sorted({r["realization"] for r in rows})
        pids = sorted({r["pair_id"] for r in rows if r["policy"] == "forward" and r["realization"] == reals[0]})
        fwd_f, rev_f, nr_f = defaultdict(list), defaultdict(list), defaultdict(list)
        for rn in reals:
            for pid in pids:
                fa, fb = by_k[("forward", rn, pid, "fact_a")], by_k[("forward", rn, pid, "fact_b")]
                ra, rb = by_k[("reversed", rn, pid, "fact_a")], by_k[("reversed", rn, pid, "fact_b")]
                na, nb = by_k[("no_rule", rn, pid, "fact_a")], by_k[("no_rule", rn, pid, "fact_b")]
                fam = fa["family"]
                fwd_f[fam].append(fb["m_full"] - fa["m_full"])
                rev_f[fam].append(ra["m_full"] - rb["m_full"])
                nr_f[fam].append(0.5 * (na["m_full"] + nb["m_full"]))
        out["per_arm"][arm] = {
            "fwd_disc": family_cluster_boot(fwd_f),
            "rev_disc": family_cluster_boot(rev_f),
            "no_rule_bias": family_cluster_boot(nr_f),
        }
    for cname, a1, a2 in (("post/base_vs_base/base", "post/base", "base/base"), ("post/post_vs_base/base", "post/post", "base/base")):
        k1 = {(r["policy"], r["realization"], r["pair_id"], r["fact_key"]): r for r in arms[a1]}
        k2 = {(r["policy"], r["realization"], r["pair_id"], r["fact_key"]): r for r in arms[a2]}
        reals = sorted({r["realization"] for r in arms[a1]})
        pids = sorted({r["pair_id"] for r in arms[a1] if r["policy"] == "forward" and r["realization"] == reals[0]})
        df_f, df_r, df_n = defaultdict(list), defaultdict(list), defaultdict(list)
        for rn in reals:
            for pid in pids:
                fam = k1[("forward", rn, pid, "fact_a")]["family"]
                df_f[fam].append((k1[("forward", rn, pid, "fact_b")]["m_full"] - k1[("forward", rn, pid, "fact_a")]["m_full"]) - (k2[("forward", rn, pid, "fact_b")]["m_full"] - k2[("forward", rn, pid, "fact_a")]["m_full"]))
                df_r[fam].append((k1[("reversed", rn, pid, "fact_a")]["m_full"] - k1[("reversed", rn, pid, "fact_b")]["m_full"]) - (k2[("reversed", rn, pid, "fact_a")]["m_full"] - k2[("reversed", rn, pid, "fact_b")]["m_full"]))
                df_n[fam].append(0.5 * (k1[("no_rule", rn, pid, "fact_a")]["m_full"] + k1[("no_rule", rn, pid, "fact_b")]["m_full"]) - 0.5 * (k2[("no_rule", rn, pid, "fact_a")]["m_full"] + k2[("no_rule", rn, pid, "fact_b")]["m_full"]))
        out["contrasts"][cname] = {
            "fwd_disc_diff": family_cluster_boot(df_f),
            "rev_disc_diff": family_cluster_boot(df_r),
            "no_rule_bias_diff": family_cluster_boot(df_n),
        }
    return out


def parse_ac_or_rf_decision(reply: str, wording: str, is_rf: bool) -> str:
    txt = reply.strip()
    if is_rf:
        m = re.search(r"Decision\s*:\s*([A-Za-z]+)", txt, flags=re.IGNORECASE)
        if m:
            w = m.group(1).lower()
            if w in ("approve", "send", "release"):
                return "send_out"
            if w in ("block", "keep", "hold", "store"):
                return "keep_storage"
        return "other"
    if wording == "action_clear":
        m = re.match(r"^(?:Verdict:\s*)?([A-Za-z]+)\b", txt, flags=re.IGNORECASE)
        if m:
            w = m.group(1).lower()
            if w == "approve":
                return "send_out"
            if w == "block":
                return "keep_storage"
        return "other"
    return "na"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    mpath = ROOT / "experiments" / "p1_cases_manifest.json"
    if not mpath.exists():
        subprocess.check_call([sys.executable, str(ROOT / "experiments" / "p1_cases_manifest.py")])
    manifest = json.loads(mpath.read_text())
    p3 = json.loads((ROOT / "results" / "p3_head_generalization.json").read_text())

    row_by_id = {r["row_id"]: r for r in manifest["rows"]}
    for ck, rows in p3.get("reason_first_meta", {}).items():
        for r in rows:
            row_by_id[r["row_id"]] = {**r, "prefill": "", "prefill_name": "none"}

    judge = JevJudge(workers=args.workers)
    p3_q = {
        "action_choice": choice(
            "Which final action does clerk_reply recommend for the parcel?",
            {
                "send_out": "Recommends sending/releasing/approving the parcel out.",
                "keep_storage": "Recommends keeping/holding/blocking the parcel in storage.",
                "hedged_or_both": "Contradicts itself, recommends both actions, or hedges without deciding.",
                "nonresponsive_or_garbled": "Empty, repeats the prompt, nonsensical, or refuses to answer.",
            },
        )
    }

    uniq_jobs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    state_to_idx: dict[str, int] = {}
    meta_list = []

    for mk, mdata in p3["models"].items():
        for ck, arms in mdata["generations"].items():
            for aname, rows in arms.items():
                for g in rows:
                    r = row_by_id[g["row_id"]]
                    full_reply = (r.get("prefill", "") + " " + g["reply"]).strip() if r.get("prefill") else g["reply"]
                    st = {
                        "ai_instructions": r["system"],
                        "user_message": r["user"],
                        "clerk_reply": full_reply,
                    }
                    sk = json.dumps(st, sort_keys=True, separators=(",", ":"))
                    if sk not in state_to_idx:
                        state_to_idx[sk] = len(uniq_jobs)
                        uniq_jobs.append((st, p3_q))
                    meta_list.append((mk, ck, aname, g["row_id"], g, r, full_reply, state_to_idx[sk]))

    print(f"[P3-Analyze] Scoring {len(uniq_jobs)} unique replies (of {len(meta_list)} total) with Jev...", flush=True)
    scored = judge.decide_batch(uniq_jobs, workers=args.workers, progress_every=500)
    print(f"[P3-Analyze] Jev usage: {judge.usage}", flush=True)

    # Group back by model -> cell -> arm
    grouped: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for mk, ck, aname, rid, g, r, full_reply, uidx in meta_list:
        sc = scored[uidx] or {}
        act = sc.get("action_choice", {}).get("value", "error") if isinstance(sc.get("action_choice"), dict) else "error"
        is_rf = "reason_first" in ck
        grouped[mk][ck][aname].append({
            "row_id": rid,
            "pair_id": r["pair_id"],
            "family": r["family"],
            "policy": r["policy"],
            "wording": r["wording"],
            "allowed": r["allowed"],
            "action_choice": act,
            "parsed_choice": parse_ac_or_rf_decision(g["reply"], r["wording"], is_rf),
            "first_word": g["reply"].split()[0] if g["reply"].split() else "<EMPTY>",
        })

    models_out: dict[str, Any] = {}
    for mk, mdata in p3["models"].items():
        gen_sum: dict[str, Any] = {}
        gen_con: dict[str, Any] = {}
        for ck, arms in sorted(grouped[mk].items()):
            gen_sum[ck] = {}
            is_nr = ck.startswith("no_rule")
            for aname, recs in arms.items():
                counts = Counter(x["action_choice"] for x in recs)
                n = len(recs)
                if is_nr:
                    kf, sf = defaultdict(list), defaultdict(list)
                    for x in recs:
                        kf[x["family"]].append(1.0 if x["action_choice"] == "keep_storage" else 0.0)
                        sf[x["family"]].append(1.0 if x["action_choice"] == "send_out" else 0.0)
                    gen_sum[ck][aname] = {
                        "n": n,
                        "counts": dict(counts),
                        "keep_rate": family_cluster_boot(kf),
                        "send_rate": family_cluster_boot(sf),
                        "top_first_words": Counter(x["first_word"] for x in recs).most_common(5),
                    }
                else:
                    by_p: dict[str, dict[bool, dict[str, Any]]] = defaultdict(dict)
                    for x in recs:
                        by_p[x["pair_id"]][bool(x["allowed"])] = x
                    pa_f, dk_f, as_f, parsed_pa_f = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
                    for pid, pair in by_p.items():
                        ra, rd = pair[True], pair[False]
                        fam = ra["family"]
                        dk = 1.0 if rd["action_choice"] == "keep_storage" else 0.0
                        asnd = 1.0 if ra["action_choice"] == "send_out" else 0.0
                        dk_f[fam].append(dk)
                        as_f[fam].append(asnd)
                        pa_f[fam].append(1.0 if (dk == 1.0 and asnd == 1.0) else 0.0)
                        if ra["parsed_choice"] != "na":
                            p_ok = 1.0 if (rd["parsed_choice"] == "keep_storage" and ra["parsed_choice"] == "send_out") else 0.0
                            parsed_pa_f[fam].append(p_ok)
                    entry = {
                        "n": n,
                        "counts": dict(counts),
                        "pair_acc": family_cluster_boot(pa_f),
                        "denied_keep": family_cluster_boot(dk_f),
                        "allowed_send": family_cluster_boot(as_f),
                        "garbled_rate": round(counts["nonresponsive_or_garbled"] / n, 4),
                        "top_first_words": Counter(x["first_word"] for x in recs).most_common(5),
                    }
                    if parsed_pa_f:
                        entry["parsed_pair_acc"] = family_cluster_boot(parsed_pa_f)
                    gen_sum[ck][aname] = entry

            if not is_nr:
                gen_con[ck] = {}
                for cname, a1, a2 in (
                    ("post/base_vs_base/base", "post/base", "base/base"),
                    ("base/post_vs_base/base", "base/post", "base/base"),
                    ("post/post_vs_base/base", "post/post", "base/base"),
                    ("post/post_vs_post/base", "post/post", "post/base"),
                    ("action_rows/base_vs_base/base", "action_rows/base", "base/base"),
                    ("except_action_rows/base_vs_base/base", "except_action_rows/base", "base/base"),
                    ("svd4/base_vs_base/base", "svd4/base", "base/base"),
                    ("svd16/base_vs_base/base", "svd16/base", "base/base"),
                    ("svd64/base_vs_base/base", "svd64/base", "base/base"),
                    ("svd16/base_vs_rand_rank16/base", "svd16/base", "rand_rank16/base"),
                    ("dpo/base_vs_base/base", "dpo/base", "base/base"),
                    ("inst/base_vs_base/base", "inst/base", "base/base"),
                ):
                    if a1 not in arms or a2 not in arms:
                        continue
                    by_p1 = defaultdict(dict)
                    by_p2 = defaultdict(dict)
                    for x in arms[a1]:
                        by_p1[x["pair_id"]][bool(x["allowed"])] = x
                    for x in arms[a2]:
                        by_p2[x["pair_id"]][bool(x["allowed"])] = x
                    df = defaultdict(list)
                    for pid in sorted(by_p1):
                        fam = by_p1[pid][True]["family"]
                        ok1 = 1.0 if (by_p1[pid][True]["action_choice"] == "send_out" and by_p1[pid][False]["action_choice"] == "keep_storage") else 0.0
                        ok2 = 1.0 if (by_p2[pid][True]["action_choice"] == "send_out" and by_p2[pid][False]["action_choice"] == "keep_storage") else 0.0
                        df[fam].append(ok1 - ok2)
                    gen_con[ck][cname] = family_cluster_boot(df)

        models_out[mk] = {
            "backbone_id": mdata["backbone_id"],
            "backbone_rev": mdata["backbone_rev"],
            "base_id": mdata["base_id"],
            "base_rev": mdata["base_rev"],
            "chat_template_mode": mdata["chat_template_mode"],
            "head_stats": mdata["head_stats"],
            "parity": mdata["parity"],
            "peak_vram_gb": mdata["peak_vram_gb"],
            "elapsed_sec": mdata["elapsed_sec"],
            "teacher_forcing": analyze_tf(mdata["teacher_forcing"]),
            "generation_summary": gen_sum,
            "generation_contrasts": gen_con,
        }

    out = {
        "hardware": p3["hardware"],
        "vram_gb": p3["vram_gb"],
        "jev_stats": judge.usage,
        "models": models_out,
    }
    out_path = ROOT / "results" / "p3_analysis.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[P3-Analyze] Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
