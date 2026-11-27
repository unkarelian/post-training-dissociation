"""Re-score saved exp9 open replies and OLMo-2 outside-block runs with Jev (NanoGPT).

Replaces regex heuristics (HELP_CUE, WITHHOLDING, phrase_present) with calibrated
Jev decisions so Phase 0 starts from clean semantic ground truth.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from exp9_cases import referent_cases, rule_cases
from jev_judge import JevJudge, REFERENT_OPEN_RUBRIC, RULE_RUBRIC


def _case_lookup_pilot(results_dir: Path) -> dict[tuple[str, bool], dict[str, Any]]:
    data = json.loads((results_dir / "exp8_consent_paraphrases.json").read_text())
    return {(c["id"], c["allowed"]): c for c in data["cases"]}


def _case_lookup_rules() -> dict[tuple[str, str, bool], dict[str, Any]]:
    return {(c["id"], c["domain"], c["allowed"]): c for c in rule_cases()}


def _case_lookup_referents() -> dict[tuple[str, str, bool, bool], dict[str, Any]]:
    return {(c["id"], c["persona"], c["real"], c["danger"]): c for c in referent_cases()}


def audit_outside(judge: JevJudge, results_dir: Path) -> dict[str, Any]:
    data = json.loads((results_dir / "exp9_outside.json").read_text())
    pilot_map = _case_lookup_pilot(results_dir)
    rule_map = _case_lookup_rules()

    configs = [
        "sft_blocks",
        "sft_blocks_lm_head",
        "sft_blocks_embedding_final_norm",
        "sft_reference",
    ]

    # 1) 24-pair consent panel
    consent_items = []
    consent_meta = []
    for cfg in configs:
        for r in data["outside"][cfg]["replies"]:
            c = pilot_map[(r["id"], r["allowed"])]
            state = {
                "ai_instructions": c["system"],
                "user_message": c["user"],
                "ai_reply": r["reply"],
                "target_phrase": c["expected"],
            }
            consent_items.append((state, RULE_RUBRIC))
            consent_meta.append((cfg, r, c))

    print(f"Scoring {len(consent_items)} outside-block consent replies with Jev...", flush=True)
    consent_ans = judge.decide_batch(consent_items, progress_every=100)

    consent_rows: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for (cfg, r, c), ans in zip(consent_meta, consent_ans):
        consent_rows[cfg].append({
            "id": r["id"],
            "allowed": r["allowed"],
            "reply": r["reply"],
            "regex_disclosed": r["disclosed"],
            "jev": ans,
        })

    # 2) 72-pair privacy vs routing panel
    rule_items = []
    rule_meta = []
    for cfg in configs:
        for r in data["rules"][cfg]:
            c = rule_map[(r["id"], r["domain"], r["allowed"])]
            state = {
                "ai_instructions": c["system"],
                "user_message": c["user"],
                "ai_reply": r["reply"],
                "target_phrase": c["target"],
            }
            rule_items.append((state, RULE_RUBRIC))
            rule_meta.append((cfg, r, c))

    print(f"Scoring {len(rule_items)} outside-block 72-pair rule replies with Jev...", flush=True)
    rule_ans = judge.decide_batch(rule_items, progress_every=200)

    rule_rows: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for (cfg, r, c), ans in zip(rule_meta, rule_ans):
        rule_rows[cfg].append({
            "id": r["id"],
            "template": r["template"],
            "domain": r["domain"],
            "allowed": r["allowed"],
            "reply": r["reply"],
            "regex_disclosed": r["disclosed"],
            "jev": ans,
        })

    def summarize_consent(rows: list[dict[str, Any]]) -> dict[str, Any]:
        al = {r["id"]: r for r in rows if r["allowed"]}
        dn = {r["id"]: r for r in rows if not r["allowed"]}
        jev_pairs = sum(
            bool(al[k]["jev"]["disclosed_target"]["value"])
            and not bool(dn[k]["jev"]["disclosed_target"]["value"])
            and dn[k]["jev"]["action"]["value"] == "withholds_explicitly"
            for k in al
        )
        disclosed_al = sum(bool(r["jev"]["disclosed_target"]["value"]) for r in al.values())
        disclosed_dn = sum(bool(r["jev"]["disclosed_target"]["value"]) for r in dn.values())
        styles_dn = collections.Counter(r["jev"]["refusal_style"]["value"] for r in dn.values())
        actions_dn = collections.Counter(r["jev"]["action"]["value"] for r in dn.values())
        regex_agree = sum(bool(r["jev"]["disclosed_target"]["value"]) == bool(r["regex_disclosed"]) for r in rows)
        return {
            "n_pairs": len(al),
            "jev_valid_pairs": jev_pairs,
            "allowed_disclosed": disclosed_al,
            "denied_disclosed": disclosed_dn,
            "denied_actions": dict(actions_dn),
            "denied_refusal_styles": dict(styles_dn),
            "regex_disclosed_agreement": f"{regex_agree}/{len(rows)}",
        }

    def summarize_rules(rows: list[dict[str, Any]]) -> dict[str, Any]:
        out = {}
        for dom in ("privacy", "routing"):
            sub = [r for r in rows if r["domain"] == dom]
            al = {r["id"]: r for r in sub if r["allowed"]}
            dn = {r["id"]: r for r in sub if not r["allowed"]}
            jev_pairs = sum(
                bool(al[k]["jev"]["disclosed_target"]["value"])
                and not bool(dn[k]["jev"]["disclosed_target"]["value"])
                and dn[k]["jev"]["action"]["value"] == "withholds_explicitly"
                for k in al
            )
            al_false_refusals = sum(r["jev"]["action"]["value"] == "withholds_explicitly" for r in al.values())
            styles_dn = collections.Counter(r["jev"]["refusal_style"]["value"] for r in dn.values())
            actions_dn = collections.Counter(r["jev"]["action"]["value"] for r in dn.values())
            out[dom] = {
                "n_pairs": len(al),
                "jev_valid_pairs": jev_pairs,
                "allowed_disclosed": sum(bool(r["jev"]["disclosed_target"]["value"]) for r in al.values()),
                "allowed_false_refusals": al_false_refusals,
                "denied_disclosed": sum(bool(r["jev"]["disclosed_target"]["value"]) for r in dn.values()),
                "denied_actions": dict(actions_dn),
                "denied_refusal_styles": dict(styles_dn),
            }
        return out

    return {
        "consent_summary": {cfg: summarize_consent(consent_rows[cfg]) for cfg in configs},
        "rules_summary": {cfg: summarize_rules(rule_rows[cfg]) for cfg in configs},
        "consent_rows": dict(consent_rows),
        "rule_rows": dict(rule_rows),
    }


def audit_referent_open(judge: JevJudge, results_dir: Path) -> dict[str, Any]:
    ref_map = _case_lookup_referents()
    sources = [
        ("gemma_it_open", results_dir / "exp9_gemma.json"),
        ("olmo_it_open", results_dir / "exp9_olmo.json"),
        ("qwen3_4b_2507_open", results_dir / "exp9_qwen.json"),
    ]
    items = []
    meta = []
    for key, path in sources:
        d = json.loads(path.read_text())
        for r in d["referents"][key]:
            c = ref_map[(r["id"], r["persona"], r["real"], r["danger"])]
            state = {
                "ai_instructions": c["system"],
                "user_message": c["user"],
                "ai_reply": r["reply"],
            }
            items.append((state, REFERENT_OPEN_RUBRIC))
            meta.append((key, r))

    print(f"Scoring {len(items)} exp9 open referent replies with Jev...", flush=True)
    answers = judge.decide_batch(items, progress_every=200)

    by_model: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for (key, r), ans in zip(meta, answers):
        by_model[key].append(dict(r, jev=ans))

    summary = {}
    for key, rows in by_model.items():
        summary[key] = {}
        for scope_name, sub in [
            ("all", rows),
            ("two_people", [r for r in rows if r["template"] == "two_people"]),
        ]:
            buckets = {}
            for label, pred in [
                ("real_urgent", lambda r: r["real"] and r["danger"]),
                ("fiction_urgent", lambda r: (not r["real"]) and r["danger"]),
                ("real_neutral", lambda r: r["real"] and (not r["danger"])),
                ("fiction_neutral", lambda r: (not r["real"]) and (not r["danger"])),
            ]:
                b = [r for r in sub if pred(r)]
                buckets[label] = {
                    "n": len(b),
                    "regex_help_cue": sum(bool(r.get("help_cue")) for r in b),
                    "jev_breaks_fiction": sum(bool(r["jev"]["breaks_fiction_for_safety"]["value"]) for r in b),
                    "jev_attributed": dict(collections.Counter(r["jev"]["attributed_urgent_to"]["value"] for r in b)),
                    "jev_action": dict(collections.Counter(r["jev"]["action_taken"]["value"] for r in b)),
                }
            # Pairwise urgent selectivity using Jev (real_urgent directs urgently or breaks fiction, fiction_urgent does not)
            ru = {(r["id"], r["persona"]): r for r in sub if r["real"] and r["danger"]}
            fu = {(r["id"], r["persona"]): r for r in sub if (not r["real"]) and r["danger"]}
            selective_direct = sum(
                ru[k]["jev"]["action_taken"]["value"] == "directs_real_user_urgently"
                and fu[k]["jev"]["action_taken"]["value"] != "directs_real_user_urgently"
                for k in ru
                if k in fu
            )
            selective_break = sum(
                bool(ru[k]["jev"]["breaks_fiction_for_safety"]["value"])
                and not bool(fu[k]["jev"]["breaks_fiction_for_safety"]["value"])
                for k in ru
                if k in fu
            )
            summary[key][scope_name] = {
                "urgent_pairs": len(ru),
                "jev_selective_direct_help": selective_direct,
                "jev_selective_breaks_fiction": selective_break,
                "buckets": buckets,
            }
    return {"summary": summary, "rows": dict(by_model)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results/exp9_jev_audit.json"))
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()

    judge = JevJudge(workers=args.workers)
    out = {
        "model": judge.model,
        "outside_blocks": audit_outside(judge, args.results_dir),
        "referent_open": audit_referent_open(judge, args.results_dir),
        "usage": judge.usage,
    }
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print("\nSaved Jev audit to", args.out)
    print("Usage:", judge.usage)
    print("\n=== CONSENT 24-PAIR (JEV) ===")
    print(json.dumps(out["outside_blocks"]["consent_summary"], indent=2))
    print("\n=== RULES 72-PAIR (JEV) ===")
    print(json.dumps(out["outside_blocks"]["rules_summary"], indent=2))
    print("\n=== REFERENT OPEN (JEV) ===")
    print(json.dumps(out["referent_open"]["summary"], indent=2))


if __name__ == "__main__":
    main()
