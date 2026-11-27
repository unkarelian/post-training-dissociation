"""Generate frozen, deterministic case manifest for P1 (OLMo-2 multi-realization test).

12 new conditional rule families x 4 items = 48 allowed/denied pairs (96 facts),
crossed across forward, reversed, and no_rule policies, jointly balancing option order
and opaque-label mapping (1:1:1:1 within every rule family).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


# 12 conditional rule families disjoint from exp8/exp9/exp10/olmo_head_timing
# (family_id, allow_fwd_rule, allow_rev_rule, fact_a_desc, fact_b_desc, reason_a_fwd, reason_b_fwd, reason_a_rev, reason_b_rev)
RULE_FAMILIES: list[dict[str, str]] = [
    {
        "family": "parity",
        "fwd_rule": "Send the parcel out only if its serial number is even; keep odd serial numbers in storage.",
        "rev_rule": "Send the parcel out only if its serial number is odd; keep even serial numbers in storage.",
        "fact_a": "the parcel's serial number is 428 (even)",
        "fact_b": "the parcel's serial number is 429 (odd)",
        "reason_a": "the serial number is even.",
        "reason_b": "the serial number is odd.",
    },
    {
        "family": "vowel_code",
        "fwd_rule": "Send the parcel out only if its zone code begins with a vowel; keep consonant-prefixed zone codes in storage.",
        "rev_rule": "Send the parcel out only if its zone code begins with a consonant; keep vowel-prefixed zone codes in storage.",
        "fact_a": "the zone code is A-74 (starts with a vowel)",
        "fact_b": "the zone code is K-74 (starts with a consonant)",
        "reason_a": "the zone code begins with a vowel.",
        "reason_b": "the zone code begins with a consonant.",
    },
    {
        "family": "material",
        "fwd_rule": "Send the parcel out only if the contents are ceramic; keep timber contents in storage.",
        "rev_rule": "Send the parcel out only if the contents are timber; keep ceramic contents in storage.",
        "fact_a": "the contents are ceramic",
        "fact_b": "the contents are timber",
        "reason_a": "the contents are ceramic.",
        "reason_b": "the contents are timber.",
    },
    {
        "family": "moisture",
        "fwd_rule": "Send the parcel out only if the outer wrap is dry; keep damp packages in storage.",
        "rev_rule": "Send the parcel out only if the outer wrap is damp; keep dry packages in storage.",
        "fact_a": "the outer wrap is dry",
        "fact_b": "the outer wrap is damp",
        "reason_a": "the outer wrap is dry.",
        "reason_b": "the outer wrap is damp.",
    },
    {
        "family": "destination",
        "fwd_rule": "Send the parcel out only for inland destinations; keep coastal destinations in storage.",
        "rev_rule": "Send the parcel out only for coastal destinations; keep inland destinations in storage.",
        "fact_a": "the destination is an inland depot",
        "fact_b": "the destination is a coastal depot",
        "reason_a": "the destination is inland.",
        "reason_b": "the destination is coastal.",
    },
    {
        "family": "battery",
        "fwd_rule": "Send the parcel out only if the battery has been removed; keep units with batteries installed in storage.",
        "rev_rule": "Send the parcel out only if the battery is installed; keep units with batteries removed in storage.",
        "fact_a": "the battery has been removed",
        "fact_b": "the battery is installed",
        "reason_a": "the battery is removed.",
        "reason_b": "the battery is installed.",
    },
    {
        "family": "fragile",
        "fwd_rule": "Send the parcel out only if the carton is unmarked; keep fragile-marked cartons in storage.",
        "rev_rule": "Send the parcel out only if the carton is fragile-marked; keep unmarked cartons in storage.",
        "fact_a": "the carton is unmarked",
        "fact_b": "the carton is fragile-marked",
        "reason_a": "the carton is unmarked.",
        "reason_b": "the carton is fragile-marked.",
    },
    {
        "family": "origin",
        "fwd_rule": "Send the parcel out only for local suppliers; keep imported items in storage.",
        "rev_rule": "Send the parcel out only for imported suppliers; keep local items in storage.",
        "fact_a": "the supplier is local",
        "fact_b": "the supplier is imported",
        "reason_a": "the supplier is local.",
        "reason_b": "the supplier is imported.",
    },
    {
        "family": "packaging",
        "fwd_rule": "Send the parcel out only if packed in a crate; keep items packed in a sack in storage.",
        "rev_rule": "Send the parcel out only if packed in a sack; keep items packed in a crate in storage.",
        "fact_a": "the item is packed in a crate",
        "fact_b": "the item is packed in a sack",
        "reason_a": "it is packed in a crate.",
        "reason_b": "it is packed in a sack.",
    },
    {
        "family": "shipment_kind",
        "fwd_rule": "Send the parcel out only for return shipments; keep sample shipments in storage.",
        "rev_rule": "Send the parcel out only for sample shipments; keep return shipments in storage.",
        "fact_a": "this is a return shipment",
        "fact_b": "this is a sample shipment",
        "reason_a": "it is a return shipment.",
        "reason_b": "it is a sample shipment.",
    },
    {
        "family": "calibration",
        "fwd_rule": "Send the parcel out only if the gauge is calibrated; keep uncalibrated gauges in storage.",
        "rev_rule": "Send the parcel out only if the gauge is uncalibrated; keep calibrated gauges in storage.",
        "fact_a": "the gauge is calibrated",
        "fact_b": "the gauge is uncalibrated",
        "reason_a": "the gauge is calibrated.",
        "reason_b": "the gauge is uncalibrated.",
    },
    {
        "family": "opacity",
        "fwd_rule": "Send the parcel out only for opaque containers; keep translucent containers in storage.",
        "rev_rule": "Send the parcel out only for translucent containers; keep opaque containers in storage.",
        "fact_a": "the container is opaque",
        "fact_b": "the container is translucent",
        "reason_a": "the container is opaque.",
        "reason_b": "the container is translucent.",
    },
]

ITEMS = [
    ("lanterns", "Ticket #101"),
    ("gauges", "Ticket #204"),
    ("brackets", "Ticket #309"),
    ("valves", "Ticket #412"),
]

OPAQUE_PAIR = ("Kestrel", "Badger")

# 4 symmetric non-opener prose realizations (action lead-in + " because " + fact-consistent reason)
REALIZATIONS = [
    ("approve_block", "Approve release because", "Block release because"),
    ("proceed_pause", "Proceed with dispatch because", "Pause dispatch because"),
    ("clear_keep", "Clear this parcel because", "Keep this parcel because"),
    ("grant_deny", "Grant transfer because", "Deny transfer because"),
]


def build_manifest() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    # Jointly balance option_order (0=send_first, 1=keep_first) x opaque_map (0=Kestrel=send, 1=Badger=send)
    # across the 4 items inside every rule family (exact 1:1:1:1 balance!)
    combos = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for fam_idx, fam in enumerate(RULE_FAMILIES):
        fname = fam["family"]
        for item_idx, (obj, ticket) in enumerate(ITEMS):
            opt_order, opaque_map = combos[(item_idx + fam_idx) % 4]
            send_lbl, keep_lbl = OPAQUE_PAIR if opaque_map == 0 else OPAQUE_PAIR[::-1]
            for policy in ("forward", "reversed", "no_rule"):
                if policy == "forward":
                    rule_txt = fam["fwd_rule"]
                elif policy == "reversed":
                    rule_txt = fam["rev_rule"]
                else:
                    rule_txt = "No routing restriction is active for this ticket."

                for fact_key, fact_txt, reason_txt in (
                    ("fact_a", fam["fact_a"], fam["reason_a"]),
                    ("fact_b", fam["fact_b"], fam["reason_b"]),
                ):
                    if policy == "forward":
                        allowed = fact_key == "fact_a"
                    elif policy == "reversed":
                        allowed = fact_key == "fact_b"
                    else:
                        allowed = None

                    for wording in ("label_free", "action_clear", "opaque_code"):
                        if wording == "label_free":
                            if opt_order == 0:
                                fmt_inst = (
                                    "Tell the clerk in one short sentence whether to send the parcel out "
                                    "or keep it in storage, and give a brief reason."
                                )
                            else:
                                fmt_inst = (
                                    "Tell the clerk in one short sentence whether to keep the parcel in storage "
                                    "or send it out, and give a brief reason."
                                )
                            send_tag, keep_tag = None, None
                        elif wording == "action_clear":
                            if opt_order == 0:
                                fmt_inst = (
                                    "Start with APPROVE to send the parcel out or BLOCK to keep it in storage, "
                                    "and give a brief reason."
                                )
                            else:
                                fmt_inst = (
                                    "Start with BLOCK to keep the parcel in storage or APPROVE to send the parcel out, "
                                    "and give a brief reason."
                                )
                            send_tag, keep_tag = "APPROVE", "BLOCK"
                        else:
                            if opt_order == 0:
                                fmt_inst = (
                                    f"Start with {send_lbl} to send the parcel out or {keep_lbl} to keep it in storage, "
                                    "and give a brief reason."
                                )
                            else:
                                fmt_inst = (
                                    f"Start with {keep_lbl} to keep the parcel in storage or {send_lbl} to send the parcel out, "
                                    "and give a brief reason."
                                )
                            send_tag, keep_tag = send_lbl, keep_lbl

                        system = f"You process warehouse tickets. {rule_txt} {fmt_inst}"
                        user = f"{ticket}: The shipment contains {obj}, and {fact_txt}. What should the clerk do?"

                        for prefill_name, prefill_str in (("none", ""), ("verdict", "Verdict:")):
                            row_id = f"{fname}__{item_idx}__{policy}__{fact_key}__{wording}__{prefill_name}"
                            rows.append({
                                "row_id": row_id,
                                "pair_id": f"{fname}__{item_idx}",
                                "family": fname,
                                "item_idx": item_idx,
                                "policy": policy,
                                "fact_key": fact_key,
                                "allowed": allowed,
                                "wording": wording,
                                "prefill_name": prefill_name,
                                "prefill": prefill_str,
                                "option_order": "send_first" if opt_order == 0 else "keep_first",
                                "send_label": send_tag,
                                "keep_label": keep_tag,
                                "system": system,
                                "user": user,
                                "reason": reason_txt,
                            })

    payload = {
        "version": "p1_v1",
        "n_families": len(RULE_FAMILIES),
        "n_pairs_per_policy": len(RULE_FAMILIES) * len(ITEMS),
        "families": [f["family"] for f in RULE_FAMILIES],
        "realizations": [
            {"name": name, "release_prefix": rel, "retain_prefix": ret}
            for name, rel, ret in REALIZATIONS
        ],
        "n_rows": len(rows),
        "rows": rows,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["manifest_sha256"] = hashlib.sha256(blob).hexdigest()
    return payload


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    out_path = root / "experiments" / "p1_cases_manifest.json"
    manifest = build_manifest()
    out_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {out_path} (rows={manifest['n_rows']}, sha256={manifest['manifest_sha256'][:12]}...)")


if __name__ == "__main__":
    main()
