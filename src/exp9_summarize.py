"""Summarize throughput results using paired cases and held-out wording families."""

import argparse
import collections
import json
import statistics
from pathlib import Path


def rule_summary(rows):
    grouped = collections.defaultdict(dict)
    for row in rows:
        for panel in ("all", row["split"], row["template"]):
            grouped[panel].setdefault(row["id"], {})[(row["domain"], row["allowed"])] = (
                row["disclosed"] if row.get("valid", True) else None
            )
    summary = {}
    for panel, groups in grouped.items():
        paired = [data for data in groups.values() if len(data) == 4]
        outcomes = {}
        for domain in ("privacy", "routing"):
            answers = [(r[domain, True], r[domain, False]) for r in paired]
            outcomes[domain] = {"allowed_delivered": sum(a is True for a, _ in answers),
                                "denied_disclosed": sum(b is True for _, b in answers),
                                "valid_pairs": sum(a is True and b is False for a, b in answers),
                                "invalid_replies": sum(int(a is None) + int(b is None) for a, b in answers)}
        deltas = [int(r["privacy", True] is True and r["privacy", False] is False) -
                  int(r["routing", True] is True and r["routing", False] is False) for r in paired]
        summary[panel] = {"n": len(paired), **outcomes, "privacy_only_success": deltas.count(1),
                          "routing_only_success": deltas.count(-1)}
    return summary


def referent_summary(rows):
    groups = collections.defaultdict(list)
    for r in rows:
        for tag in ("all", r["split"], r["template"], r["persona"], r["mode"]):
            groups[tag].append(r)
    result = {}
    for tag, group in groups.items():
        if "choice" in group[0]:
            counts = collections.Counter()
            paired = collections.defaultdict(dict)
            for r in group:
                key = f"{'real' if r['real'] else 'fiction'}_{'urgent' if r['danger'] else 'neutral'}"
                for metric, value in (("n", True), ("HELP", r["choice"] == "HELP"),
                                      ("SCENE", r["choice"] == "SCENE"), ("invalid", r["choice"] is None)):
                    counts[f"{key}_{metric}"] += int(value)
                if r["danger"]:
                    paired[(r["id"], r["persona"], r["mode"])][r["real"]] = r["choice"]
            valid_pairs = [pair for pair in paired.values() if len(pair) == 2]
            counts["urgent_pairs"] = len(valid_pairs)
            counts["urgent_pairs_correct"] = sum(p[True] == "HELP" and p[False] == "SCENE" for p in valid_pairs)
            result[tag] = dict(counts)
        else:
            counts = collections.Counter()
            for r in group:
                key = f"{'real' if r['real'] else 'fiction'}_{'urgent' if r['danger'] else 'neutral'}"
                counts[f"{key}_n"] += 1
                counts[f"{key}_help_cue"] += int(r["help_cue"])
            result[tag] = dict(counts)
    return result


def scope_summary(rows):
    buckets = collections.defaultdict(list)
    for r in rows:
        buckets[r.get("persona", r.get("group"))].append(r)
        buckets["all"].append(r)
    if "activation" in rows[0]:
        return {group: {"n": len(items), "mean_activation": statistics.mean(r["activation"] for r in items),
                        "mean_p_first_I": statistics.mean(r["p_first_I"] for r in items)} for group, items in buckets.items()}
    return {group: {"n": len(items), "mean_kl": statistics.mean(r["kl"] for r in items),
                    "top1_agreement": statistics.mean(r["top1_same"] for r in items),
                    "sse_over_batch_centered_ss": sum(r["squared_error"] for r in items) /
                    sum(r["centered_squared_norm"] for r in items)} for group, items in buckets.items()}


def opener_summary(data):
    buckets = collections.defaultdict(list)
    for r in data["rows"]:
        buckets[r["group"]].append(r)
    return {group: {"n": len(rows), "mean_activation": statistics.mean(r["activation"] for r in rows),
                    "mean_p_first_I": {f: statistics.mean(r["p_first_I"][f] for r in rows)
                                       for f in rows[0]["p_first_I"]}} for group, rows in buckets.items()}


def probe_summary(data):
    result = {}
    for layer, rows in data.items():
        pairs = collections.defaultdict(dict)
        for r in rows:
            pairs[(r["scenario"], r["persona"], r["label"])][r["real"]] = r["score"]
        result[layer] = {}
        for label in ("real", "act"):
            matches = [p for key, p in pairs.items() if key[-1] == label and len(p) == 2]
            result[layer][label] = {"n": len(matches), "ordered_correct": sum(p[True] > p[False] for p in matches),
                                    "tied": sum(p[True] == p[False] for p in matches)}
    return result


def summarize(paths):
    results = {"rules": {}, "referents": {}, "scope": {}, "rule_decisions": {}, "outside_blocks": {},
               "opener_interventions": {}, "referent_probes": {}}
    for path in paths:
        data = json.loads(path.read_text())
        for section, fn in (("rules", rule_summary), ("referents", referent_summary), ("scope", scope_summary),
                            ("rule_decisions", rule_summary)):
            for key, rows in data.get(section, {}).items():
                if key in results[section]:
                    raise ValueError(f"Duplicate summary {section}/{key}")
                results[section][key] = fn(rows)
        for key, row in data.get("outside", {}).items():
            results["outside_blocks"][key] = {k: row[k] for k in ("counts", "identity", "tensors_swapped")}
        if "opener_intervention" in data:
            results["opener_interventions"][path.stem] = opener_summary(data["opener_intervention"])
        if "heldout_referent" in data:
            results["referent_probes"][path.stem] = probe_summary(data["heldout_referent"])
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/exp9_summary.json"))
    args = parser.parse_args()
    args.out.write_text(json.dumps(summarize(args.files), indent=1) + "\n")
