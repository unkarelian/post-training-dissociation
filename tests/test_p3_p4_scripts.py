"""Pure-Python CPU unit tests for P3 and P4 GPU scripts, analysis scripts, and Colab runner."""

from __future__ import annotations

import ast
import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_syntax_and_compile() -> None:
    files = [
        ROOT / "experiments" / "p3_head_generalization.py",
        ROOT / "experiments" / "p4_gemma_paradox.py",
        ROOT / "scripts" / "run_colab_p3_p4.py",
        ROOT / "src" / "p3_score_jev.py",
        ROOT / "src" / "p4_score_jev.py",
    ]
    for f in files:
        assert f.is_file(), f"Missing {f}"
        py_compile.compile(str(f), doraise=True)
        ast.parse(f.read_text(), filename=str(f))


def test_p3_reason_first_builder() -> None:
    # Extract pure-Python build_reason_first_rows without importing torch
    sys.path.insert(0, str(ROOT / "experiments"))
    import p1_cases_manifest as mmod
    manifest = mmod.build_manifest()

    src = (ROOT / "experiments" / "p3_head_generalization.py").read_text()
    assert "P3_SUCCESS_MARKER" in src
    assert "make_position_ids" in src
    assert "tie_word_embeddings" in src
    assert "clear_hf_cache" in src
    # Verify all 10 pinned SHAs are 40-hex
    tree = ast.parse(src)
    pinned_dict = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PINNED" for t in node.targets):
            pinned_dict = ast.literal_eval(node.value)
    assert pinned_dict is not None and len(pinned_dict) == 10
    for k, (mid, sha) in pinned_dict.items():
        assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha), (k, sha)

    # Verify reason_first produces 96 rows per policy x wording (12 families x 4 items x 2 facts)
    rf_count = sum(1 for r in manifest["rows"] if r["prefill_name"] == "none" and r["policy"] == "forward" and r["wording"] == "action_clear")
    assert rf_count == 96, rf_count


def test_p4_disentangler_structure() -> None:
    src = (ROOT / "experiments" / "p4_gemma_paradox.py").read_text()
    for required in (
        "P4_SUCCESS_MARKER",
        "same_truth_same_letter",
        "same_truth_flip_letter",
        "wrong_truth_flip_letter",
        "wrong_truth_same_letter",
        "proj_u_truth",
        "proj_u_letter",
        "proj_u_order",
        "proj_span_3d",
        "complement_span_3d",
        "question_suffix_L16_24",
        "question_suffix_L16_24_wrong_donor",
        "make_position_ids",
    ):
        assert required in src, f"Missing {required} in p4_gemma_paradox.py"


def main() -> None:
    test_syntax_and_compile()
    test_p3_reason_first_builder()
    test_p4_disentangler_structure()
    print("ALL P3 & P4 UNIT TESTS PASSED!")


if __name__ == "__main__":
    main()
