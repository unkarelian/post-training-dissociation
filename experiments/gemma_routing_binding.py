"""Test referent binding and report-order transfer on fictional workshop tickets."""

from collections import defaultdict
from itertools import product
import json
from pathlib import Path


MODEL = "unsloth/gemma-3-4b-it"
REVISION = "bf46152c47f5dd20b896357cb51abc4c03b8ee8c"
DEPTHS = (8, 16, 20, 24, 28, 34)
ORDERS = ("practice_first", "actual_first")
TASKS = ("dispatch", "attribution")
CLARIFICATIONS = ("none", "restatement", "oracle")
# Keep entire scenario families out of the probe training set.
SCENARIOS = {
    "discover": (
        ("label printer", "It is jammed and needs a repair.", "It has finished its batch and works."),
        ("shipping scale", "It has stopped reading weight and needs service.", "It was checked and is reading accurately."),
        ("folding table", "Its hinge snapped and needs replacing.", "Its hinge folds smoothly and is intact."),
        ("inventory scanner", "It will not start and requires service.", "It has started and scans every item."),
        ("display case", "Its latch is broken and needs a new part.", "Its latch works and closes securely."),
        ("projector", "Its bulb failed and needs replacing.", "Its picture is bright and stable."),
    ),
    "heldout": (
        ("thread rack", "Two pegs broke and need replacing.", "All the pegs are intact and holding spools."),
        ("kitchen timer", "The dial is stuck and needs service.", "The dial turns freely and the timer works."),
        ("paint shaker", "It makes an error beep and needs a repair.", "It finished its cycle and is ready to use."),
        ("packing cart", "A wheel fell off and needs reattaching.", "Its wheels are attached and roll smoothly."),
        ("sorting tray", "The handle split and needs replacing.", "The handle is intact and easy to grip."),
        ("paper cutter", "The guard is loose and needs tightening.", "The guard is secure and working."),
    ),
}


def instructions(task, yes_label):
    if task == "dispatch":
        question = "Should the technician go to the actual workshop? Dispatch only if it needs a repair."
    else:
        question = "Does the repair report apply to the actual workshop?"
    labels = "A means yes and B means no." if yes_label == "A" else "A means no and B means yes."
    return f"{question} {labels} Reply with A or B."


def two_report_cases():
    for split, scenarios in SCENARIOS.items():
        for equipment, broken, working in scenarios:
            for order, actual, practice, task, label, clarification in product(
                ORDERS, (False, True), (False, True), TASKS, ("A", "B"), CLARIFICATIONS,
            ):
                actual_report = f"Actual workshop: {broken if actual else working}"
                practice_report = f"Fictional practice log: {broken if practice else working}"
                reports = [practice_report, actual_report] if order == "practice_first" else [actual_report, practice_report]
                if clarification == "restatement":
                    reports.append("Verified cross-check: Both reports above were copied exactly for the current ticket.")
                if clarification == "oracle":
                    needs = "does need" if actual else "does not need"
                    reports.append(f"Verified cross-check: The actual workshop {needs} this repair.")
                yield dict(
                    split=split, scenario=equipment, order=order, actual_problem=actual,
                    practice_problem=practice, task=task, yes_label=label, clarification=clarification,
                    last_problem=actual if order == "practice_first" else practice,
                    prompt=f"Ticket item: {equipment}. {' '.join(reports)} {instructions(task, label)}",
                )


def single_report_cases():
    for equipment, broken, working in SCENARIOS["discover"]:
        for actual, needs_repair, task, label in product((False, True), (False, True), TASKS, ("A", "B")):
            place = "Actual workshop" if actual else "Fictional practice log"
            condition = broken if needs_repair else working
            yield dict(
                scenario=equipment, actual=actual, act=actual and needs_repair, task=task, yes_label=label,
                prompt=f"Ticket item: {equipment}. {place}: {condition} {instructions(task, label)}",
            )


def fit_ridge(train, target):
    import torch

    center = train.mean(0)
    scale = (train - center).square().sum(-1).mean().sqrt().clamp_min(1e-7)
    x = (train - center) / scale
    mean_target = target.mean()
    dual = torch.linalg.solve(x @ x.T + 0.1 * torch.eye(len(target), dtype=x.dtype), target - mean_target)
    return center, scale, x.T @ dual, mean_target


def extract(model, tokenizer, cases, *, collect):
    import torch

    label_ids = {
        label: sorted({ids[0] for form in (label, " " + label)
                       if len(ids := tokenizer.encode(form, add_special_tokens=False)) == 1})
        for label in ("A", "B")
    }
    assert all(label_ids.values()) and not set(label_ids["A"]) & set(label_ids["B"])
    if collect:
        states = {depth: {} for depth in DEPTHS}
    else:
        states = {}
    tag_by_id = {i: tag for tag, ids in label_ids.items() for i in ids}
    measured = []
    batch_size = 24
    device = model.device
    for start in range(0, len(cases), batch_size):
        batch = cases[start:start + batch_size]
        rendered = [tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}], tokenize=False, add_generation_prompt=True,
        ) for item in batch]
        tokens = tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        keep = [i for i, case in enumerate(batch) if collect and case.get("clarification", "none") == "none"]
        out = model(**tokens, use_cache=False, output_hidden_states=bool(keep), logits_to_keep=1)
        logits = out.logits[:, -1].float()
        margin = (torch.logsumexp(logits[:, label_ids["A"]], -1) - torch.logsumexp(logits[:, label_ids["B"]], -1)).tolist()
        greedy_tokens = logits.argmax(-1).tolist()
        for i, case in enumerate(batch):
            greedy = tag_by_id.get(greedy_tokens[i], "invalid")
            yes_label = case.get("yes_label")
            measured.append(dict(**case, yes_margin=margin[i] if yes_label == "A" else -margin[i], greedy_tag=greedy))
        if keep:
            for depth in DEPTHS:
                captures = out.hidden_states[depth][keep, -1].float().cpu()
                states[depth].update({start + i: h for i, h in zip(keep, captures)})
        if (start // batch_size) % 10 == 0:
            print(f"Processed {min(start + batch_size, len(cases))}/{len(cases)} prompts.", flush=True)
    return measured, states


def summarize_behavior(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["split"], row["task"], row["order"], row["clarification"]].append(row)
    result = {}
    for key, subset in groups.items():
        discordant = [r for r in subset if r["actual_problem"] != r["practice_problem"]]
        target = lambda r: r["yes_label"] if r["actual_problem"] else ("B" if r["yes_label"] == "A" else "A")
        result["/".join(key)] = {
            "n": len(subset),
            "discordant_n": len(discordant),
            "discordant_actual_correct": sum(r["yes_margin"] != 0 and (r["yes_margin"] > 0) == r["actual_problem"] for r in discordant),
            "discordant_last_report_matches": sum(r["yes_margin"] != 0 and (r["yes_margin"] > 0) == r["last_problem"] for r in discordant),
            "greedy_correct": sum(r["greedy_tag"] == target(r) for r in subset),
            "greedy_invalid": sum(r["greedy_tag"] == "invalid" for r in subset),
            "accuracy_by_condition": {
                f"actual={int(a)};practice={int(p)}": {
                    "n": sum(r["actual_problem"] == a and r["practice_problem"] == p for r in subset),
                    "correct": sum(r["actual_problem"] == a and r["practice_problem"] == p
                                   and r["yes_margin"] != 0 and (r["yes_margin"] > 0) == a for r in subset),
                }
                for a, p in product((False, True), repeat=2)
            },
        }
    return result


def summarize_probes(single, single_states, pairs, pair_states):
    import torch

    result = {}
    for depth in DEPTHS:
        for task in TASKS:
            base = [i for i, r in enumerate(single) if r["task"] == task]
            discovery = [i for i, r in enumerate(pairs)
                         if r["split"] == "discover" and r["task"] == task and r["clarification"] == "none"]
            heldout = [i for i, r in enumerate(pairs)
                       if r["split"] == "heldout" and r["task"] == task and r["clarification"] == "none"]
            fits = {
                "single_action": (base, single_states[depth], single, "act"),
                "single_attribution": (base, single_states[depth], single, "actual"),
                "both_orders": (discovery, pair_states[depth], pairs, "actual_problem"),
                **{
                    order: ([i for i in discovery if pairs[i]["order"] == order],
                            pair_states[depth], pairs, "actual_problem")
                    for order in ORDERS
                },
            }
            test_x = torch.stack([pair_states[depth][i] for i in heldout])
            for train_name, (idx, states, train, key) in fits.items():
                center, scale, weights, intercept = fit_ridge(
                    torch.stack([states[i] for i in idx]), torch.tensor([float(train[i][key]) for i in idx]),
                )
                predictions = ((test_x - center) / scale @ weights + intercept).tolist()
                for order in ORDERS:
                    grouped = defaultdict(dict)
                    discordant = defaultdict(dict)
                    for i, score in zip(heldout, predictions):
                        r = pairs[i]
                        if r["order"] != order:
                            continue
                        case = (r["scenario"], r["yes_label"])
                        grouped[case + (r["practice_problem"],)][r["actual_problem"]] = score
                        if r["actual_problem"] != r["practice_problem"]:
                            discordant[case][r["actual_problem"]] = score
                    result[f"{depth}/{task}/{train_name}/{order}"] = {
                        "n_train": len(idx),
                        "fixed_practice_total": len(grouped),
                        "fixed_practice_correct": sum(v[True] > v[False] for v in grouped.values()),
                        "discordant_total": len(discordant),
                        "discordant_correct": sum(v[True] > v[False] for v in discordant.values()),
                    }
    return result


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hardware = torch.cuda.get_device_name(0)
    assert ("6000" in hardware and "Blackwell" in hardware) or "A100" in hardware or "L4" in hardware, hardware
    assert torch.cuda.is_bf16_supported(), f"bfloat16 requires a supported GPU, got {hardware}"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, low_cpu_mem_usage=True,
    ).eval()
    with torch.inference_mode():
        cases = list(two_report_cases())
        rows, states = extract(model, tokenizer, cases, collect=True)
        single = list(single_report_cases())
        _, single_states = extract(model, tokenizer, single, collect=True)
    output = {
        "hardware": hardware, "torch": torch.__version__, "model": MODEL, "revision": REVISION,
        "behavior": summarize_behavior(rows),
        "probes": summarize_probes(single, single_states, rows, states),
        "rows": rows,
    }
    path = Path("/content/gemma_routing_binding.json")
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
