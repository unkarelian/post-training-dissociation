"""P0: Revision-pinned lexical prevalence scan of allenai/olmo-2-hard-coded.

Fetches hard_coded.jsonl at pinned commit 7e7685797806d1210bb939cd3ef2eb77d6291682,
records exact SHA-256 and role-bearing denominators, and reports lexical prevalence
(physical rows with >=1 match and total raw occurrences separately).
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any


DATASET_REPO = "allenai/olmo-2-hard-coded"
DATASET_COMMIT = "7e7685797806d1210bb939cd3ef2eb77d6291682"
TRAIN_FILENAME = "hard_coded.jsonl"
TRAIN_URL = (
    f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/"
    f"{DATASET_COMMIT}/{TRAIN_FILENAME}"
)

MIXTURE_REPO = "allenai/tulu-3-sft-olmo-2-mixture"
MIXTURE_COMMIT = "9b9737bf4fd1ae4f24028728596d396524ddbb29"
MIXTURE_README_URL = (
    f"https://huggingface.co/datasets/{MIXTURE_REPO}/blob/"
    f"{MIXTURE_COMMIT}/README.md"
)

TERM_GROUPS: dict[str, list[str]] = {
    "ai_terms": [
        "ai",
        "artificial intelligence",
        "language model",
        "llm",
        "chatbot",
        "virtual assistant",
    ],
    "brand_terms": [
        "olmo",
        "ai2",
        "allenai",
        "allen institute",
    ],
    "biological_terms": [
        "alive",
        "born",
        "body",
        "feelings",
        "conscious",
        "consciousness",
        "sentient",
        "human",
    ],
}


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def compile_term_regex(term: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(term)}\b", flags=re.IGNORECASE)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    out_path = root / "results" / "p0_olmo_hardcoded_scan.json"

    req = urllib.request.Request(TRAIN_URL, headers={"User-Agent": "persona-mirage-p0/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw_bytes = resp.read()
    file_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    rows = [json.loads(line) for line in raw_bytes.decode("utf-8").splitlines() if line.strip()]
    n_physical_rows = len(rows)
    ids = [r.get("id") for r in rows]
    non_null_ids = [x for x in ids if x is not None]
    unique_ids = len(set(non_null_ids))

    role_texts: dict[str, list[str]] = {"user": [], "assistant": [], "any": []}
    has_role_row = {"user": 0, "assistant": 0, "any": n_physical_rows}

    per_row_by_role: list[dict[str, str]] = []
    for r in rows:
        msgs = r.get("messages") or []
        u_parts = [normalize_text(str(m.get("content", ""))) for m in msgs if m.get("role") == "user"]
        a_parts = [normalize_text(str(m.get("content", ""))) for m in msgs if m.get("role") == "assistant"]
        all_parts = [normalize_text(str(m.get("content", ""))) for m in msgs]
        if u_parts:
            has_role_row["user"] += 1
        if a_parts:
            has_role_row["assistant"] += 1
        per_row_by_role.append({
            "user": "\n".join(u_parts),
            "assistant": "\n".join(a_parts),
            "any": "\n".join(all_parts),
            "has_user": bool(u_parts),
            "has_assistant": bool(a_parts),
        })

    compiled = {
        group: {term: compile_term_regex(term) for term in terms}
        for group, terms in TERM_GROUPS.items()
    }

    results: dict[str, Any] = {}
    for role in ("user", "assistant", "any"):
        denom = has_role_row[role]
        role_res: dict[str, Any] = {"denominator_rows": denom, "groups": {}}
        for group, term_map in compiled.items():
            union_rows = 0
            union_occ = 0
            per_term: dict[str, dict[str, Any]] = {}
            for term, rx in term_map.items():
                matched_rows = 0
                occ = 0
                for rec in per_row_by_role:
                    if role == "user" and not rec["has_user"]:
                        continue
                    if role == "assistant" and not rec["has_assistant"]:
                        continue
                    hits = len(rx.findall(rec[role]))
                    if hits > 0:
                        matched_rows += 1
                        occ += hits
                per_term[term] = {
                    "rows_matched": matched_rows,
                    "rate": round(matched_rows / denom, 6) if denom else 0.0,
                    "total_occurrences": occ,
                }
            for rec in per_row_by_role:
                if role == "user" and not rec["has_user"]:
                    continue
                if role == "assistant" and not rec["has_assistant"]:
                    continue
                row_hits = sum(len(rx.findall(rec[role])) for rx in term_map.values())
                if row_hits > 0:
                    union_rows += 1
                    union_occ += row_hits
            role_res["groups"][group] = {
                "union_rows_matched": union_rows,
                "union_rate": round(union_rows / denom, 6) if denom else 0.0,
                "union_total_occurrences": union_occ,
                "terms": per_term,
            }
        results[role] = role_res

    output = {
        "description": "Revision-pinned lexical prevalence scan of allenai/olmo-2-hard-coded (does not establish training causation).",
        "dataset": {
            "repo": DATASET_REPO,
            "commit": DATASET_COMMIT,
            "filename": TRAIN_FILENAME,
            "url": TRAIN_URL,
            "sha256": file_sha256,
        },
        "mixture_provenance": {
            "repo": MIXTURE_REPO,
            "commit": MIXTURE_COMMIT,
            "readme_url": MIXTURE_README_URL,
            "documented_hard_coded_rows": 240,
            "documented_mixture_total_rows": 939344,
        },
        "matching_rules": {
            "unicode_normalization": "NFKC",
            "regex_pattern_template": r"\b<escaped_term>\b",
            "regex_flags": "re.IGNORECASE",
            "term_groups": TERM_GROUPS,
        },
        "row_counts": {
            "n_physical_rows": n_physical_rows,
            "n_rows_with_user_msg": has_role_row["user"],
            "n_rows_with_assistant_msg": has_role_row["assistant"],
            "n_non_null_ids": len(non_null_ids),
            "n_unique_ids": unique_ids,
            "has_repeated_ids": len(non_null_ids) != unique_ids,
        },
        "lexical_prevalence_by_role": results,
    }
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved {out_path} (rows={n_physical_rows}, sha256={file_sha256[:12]}...)")
    for role in ("assistant", "any"):
        print(f"\nRole={role} (denom={results[role]['denominator_rows']}):")
        for g, gd in results[role]["groups"].items():
            print(f"  {g}: union={gd['union_rows_matched']}/{results[role]['denominator_rows']} (occ={gd['union_total_occurrences']})")
            for t, td in gd["terms"].items():
                if td["rows_matched"] > 0 or g == "biological_terms":
                    print(f"    {t:24s}: {td['rows_matched']:3d} rows (occ={td['total_occurrences']})")


if __name__ == "__main__":
    main()
