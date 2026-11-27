"""Statistical analysis for P2b (Direct paired Gemma 3 4B vs 12B scale check on 18 scenarios).

Reads:
  - results/p2b_gemma_scale.json
Writes:
  - results/p2b_analysis.json
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def scenario_cluster_boot(
    sc_vals: dict[str, list[float]],
    *,
    seed: int = 20261127,
    n_boot: int = 2000,
) -> dict[str, float]:
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
    return {
        "mean": round(mean, 4),
        "ci95_lo": round(boots[int(0.025 * n_boot)], 4),
        "ci95_hi": round(boots[int(0.975 * n_boot)], 4),
        "ci90_lo": round(boots[int(0.05 * n_boot)], 4),
        "ci90_hi": round(boots[int(0.95 * n_boot)], 4),
        "p_boot_two": round(p_two, 4),
        "n_obs": len(all_v),
        "n_scenarios": k,
    }


def main() -> None:
    p2b = json.loads((ROOT / "results" / "p2b_gemma_scale.json").read_text())
    meta_by_id = {r["row_id"]: r for r in p2b["case_meta"]}
    r4 = {r["row_id"]: r for r in p2b["gemma_4b"]["rows"]}
    r12 = {r["row_id"]: r for r in p2b["gemma_12b"]["rows"]}

    out: dict[str, Any] = {
        "hardware": p2b["hardware"],
        "vram_gb": p2b["vram_gb"],
        "n_scenarios": p2b["n_scenarios"],
        "models": {
            "gemma_4b": {k: p2b["gemma_4b"][k] for k in ("model", "revision", "peak_vram_gb", "elapsed_sec")},
            "gemma_12b": {k: p2b["gemma_12b"][k] for k in ("model", "revision", "peak_vram_gb", "elapsed_sec")},
        },
        "tasks": {},
    }

    for task in ("dispatch", "attribution"):
        out["tasks"][task] = {}
        for split_filter in ("all_18", "orig_heldout", "new_heldout"):
            sub_ids = [
                rid for rid, m in meta_by_id.items()
                if m["task"] == task and (split_filter == "all_18" or m["source_split"] == split_filter)
            ]
            split_res: dict[str, Any] = {}
            for mname, rmap in (("gemma_4b", r4), ("gemma_12b", r12)):
                split_res[mname] = {}
                for subset_name, is_disc in (("discordant", True), ("concordant", False)):
                    for order in ("practice_first", "actual_first"):
                        fc_sc: dict[str, list[float]] = defaultdict(list)
                        gr_sc: dict[str, list[float]] = defaultdict(list)
                        inv_sc: dict[str, list[float]] = defaultdict(list)
                        mass_sc: dict[str, list[float]] = defaultdict(list)
                        for rid in sub_ids:
                            m = meta_by_id[rid]
                            if m["discordant"] != is_disc or m["order"] != order:
                                continue
                            row = rmap[rid]
                            sc = m["scenario"]
                            fc_sc[sc].append(1.0 if row["fc_correct"] else 0.0)
                            gr_sc[sc].append(1.0 if row["greedy_correct"] else 0.0)
                            inv_sc[sc].append(1.0 if row["greedy_invalid"] else 0.0)
                            mass_sc[sc].append(float(row["ab_prob_mass"]))
                        split_res[mname][f"{subset_name}__{order}"] = {
                            "fc": scenario_cluster_boot(fc_sc),
                            "greedy": scenario_cluster_boot(gr_sc),
                            "invalid_rate": round(sum(sum(v) for v in inv_sc.values()) / sum(len(v) for v in inv_sc.values()), 4),
                            "mean_ab_prob_mass": round(sum(sum(v) for v in mass_sc.values()) / sum(len(v) for v in mass_sc.values()), 4),
                        }

            # Direct paired diff-in-diff on discordant: (PF - AF)_4B - (PF - AF)_12B + per-model PF - AF
            # Group by (scenario, actual_problem, practice_problem, yes_label) across orders
            by_key: dict[tuple[Any, ...], dict[str, str]] = defaultdict(dict)
            for rid in sub_ids:
                m = meta_by_id[rid]
                if not m["discordant"]:
                    continue
                k = (m["scenario"], m["actual_problem"], m["practice_problem"], m["yes_label"])
                by_key[k][m["order"]] = rid

            for mode_name, field in (("fc", "fc_correct"), ("greedy", "greedy_correct")):
                gap4_sc: dict[str, list[float]] = defaultdict(list)
                gap12_sc: dict[str, list[float]] = defaultdict(list)
                did_sc: dict[str, list[float]] = defaultdict(list)
                for (sc, _, _, _), omap in by_key.items():
                    pf_id, af_id = omap["practice_first"], omap["actual_first"]
                    g4 = (1.0 if r4[pf_id][field] else 0.0) - (1.0 if r4[af_id][field] else 0.0)
                    g12 = (1.0 if r12[pf_id][field] else 0.0) - (1.0 if r12[af_id][field] else 0.0)
                    gap4_sc[sc].append(g4)
                    gap12_sc[sc].append(g12)
                    did_sc[sc].append(g4 - g12)
                did_boot = scenario_cluster_boot(did_sc)
                equiv_8pp = (did_boot["ci90_lo"] >= -0.08) and (did_boot["ci90_hi"] <= 0.08)
                split_res[f"paired_{mode_name}"] = {
                    "pf_minus_af_4b": scenario_cluster_boot(gap4_sc),
                    "pf_minus_af_12b": scenario_cluster_boot(gap12_sc),
                    "did_4b_minus_12b": did_boot,
                    "within_8pp_equiv_90ci": equiv_8pp,
                }
            out["tasks"][task][split_filter] = split_res

    out_path = ROOT / "results" / "p2b_analysis.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[P2b-Analyze] Saved {out_path}", flush=True)
    for task in ("dispatch", "attribution"):
        d = out["tasks"][task]["all_18"]["paired_fc"]
        print(
            f"  {task:11s} (18 sc, FC) | 4B PF-AF={d['pf_minus_af_4b']['mean']:+.3f} | "
            f"12B PF-AF={d['pf_minus_af_12b']['mean']:+.3f} | "
            f"DiD={d['did_4b_minus_12b']['mean']:+.3f} 95%CI=[{d['did_4b_minus_12b']['ci95_lo']:+.3f}, {d['did_4b_minus_12b']['ci95_hi']:+.3f}] | "
            f"equiv8pp={d['within_8pp_equiv_90ci']}"
        )


if __name__ == "__main__":
    main()
