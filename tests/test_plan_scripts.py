"""CPU unit tests for P1, P2a, P2b scripts and frozen manifests (no torch required locally)."""

from __future__ import annotations

import ast
import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import p1_cases_manifest as p1m


def test_syntax_all_scripts() -> None:
    for rel in (
        "src/jev_judge.py",
        "src/p0_scan_hardcoded.py",
        "src/p0_audit_grafts.py",
        "experiments/p1_cases_manifest.py",
        "experiments/p1_olmo_realizations.py",
        "experiments/p2a_gemma_restoration.py",
        "experiments/p2b_gemma_scale.py",
        "scripts/run_colab_plan.py",
    ):
        py_compile.compile(str(ROOT / rel), doraise=True)


def test_p1_manifest_balance_and_disjointness() -> None:
    m = json.loads((ROOT / "experiments" / "p1_cases_manifest.json").read_text())
    assert m["n_families"] == 12
    assert m["n_pairs_per_policy"] == 48
    assert m["n_rows"] == 12 * 4 * 3 * 2 * 3 * 2  # 1728
    prior_fams = {"threshold", "exclusion", "passcode", "override", "dock", "handoff", "weight", "schedule", "color", "tray"}
    assert not (set(m["families"]) & prior_fams)

    rows = m["rows"]
    assert len({r["row_id"] for r in rows}) == len(rows)
    for fam in m["families"]:
        sub = [r for r in rows if r["family"] == fam and r["policy"] == "forward" and r["fact_key"] == "fact_a" and r["wording"] == "opaque_code" and r["prefill_name"] == "none"]
        assert len(sub) == 4
        combos = {(r["option_order"], r["send_label"]) for r in sub}
        assert len(combos) == 4, f"Imbalanced combos in {fam}: {combos}"

    assert all("give a brief reason" in r["system"] for r in rows)

    by_key = {(r["family"], r["item_idx"], r["fact_key"], r["wording"], r["prefill_name"], r["policy"]): r for r in rows}
    for fam in m["families"]:
        for idx in range(4):
            for fk in ("fact_a", "fact_b"):
                fwd = by_key[(fam, idx, fk, "label_free", "none", "forward")]
                rev = by_key[(fam, idx, fk, "label_free", "none", "reversed")]
                nr = by_key[(fam, idx, fk, "label_free", "none", "no_rule")]
                assert fwd["user"] == rev["user"] == nr["user"]
                assert fwd["allowed"] == (not rev["allowed"])
                assert nr["allowed"] is None


def _extract_constant(path: Path, name: str):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found in {path}")


def test_p2_scenario_disjointness() -> None:
    p2a_path = ROOT / "experiments" / "p2a_gemma_restoration.py"
    p2b_path = ROOT / "experiments" / "p2b_gemma_scale.py"
    disc = {s[0] for s in _extract_constant(p2a_path, "DISCOVERY_SCENARIOS")}
    orig_held = {s[0] for s in _extract_constant(p2b_path, "ORIG_HELDOUT_SCENARIOS")}
    new_held_a = {s[0] for s in _extract_constant(p2a_path, "NEW_HELDOUT_SCENARIOS")}
    new_held_b = {s[0] for s in _extract_constant(p2b_path, "NEW_HELDOUT_SCENARIOS")}
    assert len(disc) == 6 and len(orig_held) == 6 and len(new_held_a) == 12
    assert new_held_a == new_held_b
    assert not (disc & orig_held) and not (disc & new_held_a) and not (orig_held & new_held_a)


if __name__ == "__main__":
    test_syntax_all_scripts()
    test_p1_manifest_balance_and_disjointness()
    test_p2_scenario_disjointness()
    print("ALL PLAN SCRIPT UNIT TESTS PASSED!")
