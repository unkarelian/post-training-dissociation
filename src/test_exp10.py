"""CPU unit tests verifying exp10 prompt disjointness, exact balance, and runner imports."""

from __future__ import annotations

import collections
import json
import py_compile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp9_cases import NAMES as EXP9_NAMES, RECORDS as EXP9_RECORDS, RULES as EXP9_RULES
from exp10_cases import (
    EXP10_NAMES,
    EXP10_RECORDS,
    EXP10_RULE_FAMILIES,
    exp10_rule_cases,
    exp10_single_report_cases,
    exp10_two_people_cases,
    format_referent_prompt,
    realizations_for_case,
)
import exp10_olmo_head
import exp10_gemma_referent


def test_olmo_disjointness_and_balance() -> None:
    assert set(EXP10_NAMES).isdisjoint(set(EXP9_NAMES))
    assert set(EXP10_RECORDS).isdisjoint(set(EXP9_RECORDS))
    assert set(EXP10_RULE_FAMILIES).isdisjoint(set(EXP9_RULES))
    # Also ensure zero name overlap with Gemma personas (Nell, Bram)
    assert "Bram" not in EXP10_NAMES and "Nell" not in EXP10_NAMES

    res_dir = Path(__file__).resolve().parent.parent / "results"
    e8_phrases = {
        c["expected"]
        for fn in ("exp8_consent_paraphrases.json", "exp8_selective_refusal.json")
        for c in json.loads((res_dir / fn).read_text())["cases"]
    }
    cases = exp10_rule_cases()
    assert len(cases) == 4 * 12 * 2 * 2  # 192 prompts = 96 pairs
    assert all(c["target"] not in e8_phrases for c in cases)

    # Check exact pairing across domains and polarities
    by_key = collections.defaultdict(list)
    for c in cases:
        by_key[(c["id"], c["domain"])].append(c)
    assert len(by_key) == 96
    for (cid, dom), pair in by_key.items():
        assert len(pair) == 2
        assert {p["allowed"] for p in pair} == {True, False}
        assert pair[0]["user"] == pair[1]["user"]
        assert pair[0]["target"] == pair[1]["target"]

    # Check identical user & rule across privacy vs logistics
    by_id_al = collections.defaultdict(dict)
    for c in cases:
        by_id_al[(c["id"], c["allowed"])][c["domain"]] = c
    for (cid, al), dpair in by_id_al.items():
        assert dpair["privacy"]["user"] == dpair["logistics"]["user"]
        assert dpair["privacy"]["target"] == dpair["logistics"]["target"]

    # Check 4 realizations and prefix invariants
    sample_reals = realizations_for_case(cases[0])
    assert set(sample_reals) == {
        "bare_vs_apology",
        "shared_neutral_prefix",
        "shared_first_person_prefix",
        "status_prefix_no_I",
    }
    for name, spec in sample_reals.items():
        assert spec["disclose"].startswith(spec["prefix"])
        assert spec["withhold"].startswith(spec["prefix"])
        assert spec["disclose"] != spec["withhold"]
        assert cases[0]["target"] in spec["disclose"]
        assert cases[0]["target"] not in spec["withhold"]
    assert sample_reals["shared_first_person_prefix"]["prefix"].startswith("I ")
    assert not sample_reals["status_prefix_no_I"]["withhold"].startswith("I")


def test_gemma_referent_crossing_and_bag_of_words() -> None:
    single = exp10_single_report_cases()
    tp = exp10_two_people_cases(include_harmless=True)
    # 12 emergency * 4 templates * 2 real * 2 danger * 3 personas = 576
    assert len(single) == 12 * 4 * 2 * 2 * 3
    # (12 emergency + 6 harmless) * 2 orders * 3 conditions * 3 personas = 324
    assert len(tp) == 18 * 2 * 3 * 3

    # Verify exact bag-of-words identity between real_urgent and story_urgent within each (scenario, order, persona)
    by_slot = collections.defaultdict(dict)
    for c in tp:
        by_slot[(c["scenario"], c["order"], c["persona"])][c["condition"]] = c
    for slot, conds in by_slot.items():
        ru = conds["real_urgent"]
        su = conds["story_urgent"]
        assert sorted(ru["user"].split()) == sorted(su["user"].split()), f"BoW mismatch on {slot}"
        # Verify urgent_is_last is flipped cleanly across orders!
        if ru["order"] == "story_first":
            assert ru["urgent_is_last"] is True and su["urgent_is_last"] is False
        else:
            assert ru["urgent_is_last"] is False and su["urgent_is_last"] is True

    # Verify prompt formatters and length-matched clarification
    for mode in ("tag", "attribution", "open"):
        s, u = format_referent_prompt(tp[0], mode, help_tag="B", clarification="oracle")
        assert s and u
        s2, u2 = format_referent_prompt(tp[0], mode, help_tag="B", clarification="restatement")
        assert abs(len(u.split()) - len(u2.split())) <= 5


def test_ridge_and_gate_math() -> None:
    for fn in ("jev_judge.py", "exp9_jev_audit.py", "exp10_cases.py", "exp10_olmo_head.py", "exp10_gemma_referent.py"):
        py_compile.compile(str(Path(__file__).resolve().parent / fn), doraise=True)
    try:
        import torch
    except ImportError:
        print("  (torch not installed in local CPU env; skipping tensor ridge test, syntax & imports verified)")
        return

    # Synthetic perfect order-invariant signal: real is dim 0, recency is dim 1
    rng = torch.Generator().manual_seed(0)
    tp = exp10_two_people_cases(include_harmless=True)
    single = exp10_single_report_cases()

    def synth_feats(cases):
        x = 0.1 * torch.randn(len(cases), 8, generator=rng)
        for i, c in enumerate(cases):
            x[i, 0] += 2.0 if c["act"] else -2.0
            x[i, 1] += 2.0 if c["urgent_is_last"] else -2.0
        return x

    sh = {0: torch.zeros(len(single), 8), 10: synth_feats(single)}
    th = {0: torch.zeros(len(tp), 8), 10: synth_feats(tp)}
    res = exp10_gemma_referent.evaluate_probe_gate(single, sh, tp, th, [0, 10])
    assert res["gate"]["passed"] is True
    assert res["layers"]["10"]["cross_order__story_first_to_real_first__unseen"]["acc"] == 1.0
    assert res["layers"]["10"]["cross_order__real_first_to_story_first__unseen"]["acc"] == 1.0

    # Synthetic pure-recency confound: ONLY dim 1 (urgent_is_last) carries signal -> gate MUST FAIL!
    def synth_recency_only(cases):
        x = 0.1 * torch.randn(len(cases), 8, generator=rng)
        for i, c in enumerate(cases):
            x[i, 1] += 2.0 if c["urgent_is_last"] else -2.0
        return x

    sh2 = {0: torch.zeros(len(single), 8), 10: synth_recency_only(single)}
    th2 = {0: torch.zeros(len(tp), 8), 10: synth_recency_only(tp)}
    res2 = exp10_gemma_referent.evaluate_probe_gate(single, sh2, tp, th2, [0, 10])
    assert res2["gate"]["passed"] is False
    assert res2["layers"]["10"]["cross_order__story_first_to_real_first__unseen"]["acc"] == 0.0
    assert res2["layers"]["10"]["cos_story_first_vs_real_first"] < -0.5


if __name__ == "__main__":
    test_olmo_disjointness_and_balance()
    test_gemma_referent_crossing_and_bag_of_words()
    test_ridge_and_gate_math()
    print("ALL EXP10 CPU UNIT TESTS PASSED!")
