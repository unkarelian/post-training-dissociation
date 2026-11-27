"""P2b: Direct Paired Behavior-Only Scale Check (Gemma 3 4B vs 12B) on 18 Held-Out Scenarios.

Reuses the exact frozen 4B comparison protocol across all 18 held-out scenarios
(6 original held-out + 12 new held-out), in bfloat16 under transformers==4.51.3.
Gated on assigned GPU having >= 36 GB VRAM (G4 96GB or A100 40/80GB).
Reports forced-choice A/B margin (>0, ties wrong), unrestricted greedy accuracy
(counting any token outside verified A/B IDs as invalid and wrong in the denominator),
invalid rate, and total A/B probability mass.
"""

from __future__ import annotations

import gc
import importlib.metadata
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

import torch


MODELS = {
    "gemma_4b": ("unsloth/gemma-3-4b-it", "bf46152c47f5dd20b896357cb51abc4c03b8ee8c"),
    "gemma_12b": ("unsloth/gemma-3-12b-it", "9478e665381f42974aa06177b019352fb6291876"),
}
TRANSFORMERS_VERSION = "4.51.3"

ORIG_HELDOUT_SCENARIOS = (
    ("thread rack", "Two pegs broke and need replacing.", "All the pegs are intact and holding spools."),
    ("kitchen timer", "The dial is stuck and needs service.", "The dial turns freely and the timer works."),
    ("paint shaker", "It makes an error beep and needs a repair.", "It finished its cycle and is ready to use."),
    ("packing cart", "A wheel fell off and needs reattaching.", "Its wheels are attached and roll smoothly."),
    ("sorting tray", "The handle split and needs replacing.", "The handle is intact and easy to grip."),
    ("paper cutter", "The guard is loose and needs tightening.", "The guard is secure and working."),
)

NEW_HELDOUT_SCENARIOS = (
    ("barcode gun", "Its trigger stuck and needs a repair.", "Its trigger clicks cleanly and scans items."),
    ("pallet jack", "Its hydraulic pump leaks and needs service.", "Its hydraulic pump lifts smoothly and holds."),
    ("heat sealer", "Its heating strip burned out and needs replacing.", "Its heating strip warms evenly and seals."),
    ("tape dispenser", "Its cutter blade snapped and needs a new part.", "Its cutter blade is sharp and cuts cleanly."),
    ("box stapler", "Its feed rail jammed and needs a repair.", "Its feed rail slides smoothly and staples."),
    ("laminator", "Its roller stalled and needs service.", "Its roller turns evenly and feeds sheets."),
    ("hand truck", "Its axle bent and needs replacing.", "Its axle is straight and rolls smoothly."),
    ("cord reel", "Its rewind spring broke and needs a new part.", "Its rewind spring retracts and locks cleanly."),
    ("bolt cutter", "Its pivot pin sheared and needs replacing.", "Its pivot pin is tight and cuts cleanly."),
    ("bench vise", "Its lead screw stripped and needs service.", "Its lead screw turns smoothly and clamps."),
    ("shop vacuum", "Its motor stalled and needs a repair.", "Its motor starts cleanly and pulls suction."),
    ("air hose", "Its brass coupler leaks and needs replacing.", "Its brass coupler seals tightly and holds pressure."),
)


def ensure_transformers() -> None:
    try:
        installed = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed != TRANSFORMERS_VERSION:
        if "transformers" in sys.modules:
            raise RuntimeError(f"transformers=={installed} already imported; restart kernel.")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", f"transformers=={TRANSFORMERS_VERSION}"])


def build_rows() -> list[dict[str, Any]]:
    rows = []
    for source_split, scenarios in (("orig_heldout", ORIG_HELDOUT_SCENARIOS), ("new_heldout", NEW_HELDOUT_SCENARIOS)):
        for eq, broken, working in scenarios:
            for task, order, actual, practice, yes_label in product(
                ("dispatch", "attribution"), ("practice_first", "actual_first"), (False, True), (False, True), ("A", "B")
            ):
                act_rep = f"Actual workshop: {broken if actual else working}"
                prac_rep = f"Fictional practice log: {broken if practice else working}"
                reps = [prac_rep, act_rep] if order == "practice_first" else [act_rep, prac_rep]
                if task == "dispatch":
                    q = "Should the technician go to the actual workshop? Dispatch only if it needs a repair."
                else:
                    q = "Does the repair report apply to the actual workshop?"
                labels = "A means yes and B means no." if yes_label == "A" else "A means no and B means yes."
                rows.append({
                    "row_id": f"{source_split}__{task}__{eq}__{order}__a{int(actual)}p{int(practice)}__{yes_label}",
                    "source_split": source_split,
                    "scenario": eq,
                    "task": task,
                    "order": order,
                    "actual_problem": actual,
                    "practice_problem": practice,
                    "discordant": actual != practice,
                    "yes_label": yes_label,
                    "prompt": f"Ticket item: {eq}. {' '.join(reps)} {q} {labels} Reply with A or B.",
                })
    return rows


@torch.inference_mode()
def eval_model(model_id: str, revision: str, rows: list[dict[str, Any]], batch_size: int) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.monotonic()
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, low_cpu_mem_usage=True,
    ).eval()

    ab_ids = {
        lbl: sorted({t[0] for f in (lbl, " " + lbl) if len(t := tok.encode(f, add_special_tokens=False)) == 1})
        for lbl in ("A", "B")
    }
    assert all(ab_ids.values()) and not (set(ab_ids["A"]) & set(ab_ids["B"]))
    tag_by_id = {tid: lbl for lbl, tids in ab_ids.items() for tid in tids}

    out_rows = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        rendered = [
            tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True)
            for r in batch
        ]
        enc = tok(rendered, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        pos = (enc.attention_mask.long().cumsum(-1) - 1).masked_fill(enc.attention_mask == 0, 1)
        out = model(**enc, position_ids=pos, use_cache=False, logits_to_keep=1)
        logits = out.logits[:, -1].float()
        probs = torch.softmax(logits, dim=-1)
        pA = probs[:, ab_ids["A"]].sum(-1).tolist()
        pB = probs[:, ab_ids["B"]].sum(-1).tolist()
        margin = (torch.logsumexp(logits[:, ab_ids["A"]], -1) - torch.logsumexp(logits[:, ab_ids["B"]], -1)).tolist()
        greedy_ids = logits.argmax(-1).tolist()
        for i, r in enumerate(batch):
            ym = margin[i] if r["yes_label"] == "A" else -margin[i]
            fc_correct = (ym > 0.0) if r["actual_problem"] else (ym < 0.0)  # exact ties wrong
            gtag = tag_by_id.get(greedy_ids[i], "invalid")
            target_tag = r["yes_label"] if r["actual_problem"] else ("B" if r["yes_label"] == "A" else "A")
            greedy_correct = (gtag == target_tag)  # invalids counted wrong in denominator
            out_rows.append({
                "row_id": r["row_id"],
                "yes_margin": ym,
                "fc_correct": fc_correct,
                "greedy_token_id": greedy_ids[i],
                "greedy_tag": gtag,
                "greedy_invalid": gtag == "invalid",
                "greedy_correct": greedy_correct,
                "ab_prob_mass": pA[i] + pB[i],
            })

    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return {
        "model": model_id,
        "revision": revision,
        "ab_ids": ab_ids,
        "peak_vram_gb": round(peak_gb, 2),
        "elapsed_sec": round(time.monotonic() - t0, 1),
        "rows": out_rows,
    }


def main() -> None:
    ensure_transformers()
    assert torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if vram_gb < 36.0:
        raise RuntimeError(f"P2b requires >=36 GB VRAM for unquantized bf16 12B; got {gpu_name} ({vram_gb:.1f} GB)")

    rows = build_rows()
    bs = 24 if vram_gb >= 60 else 12
    print(f"[P2b] GPU={gpu_name} ({vram_gb:.1f} GB), rows={len(rows)}, bs={bs}", flush=True)
    res_4b = eval_model(*MODELS["gemma_4b"], rows, bs)
    print(f"[P2b] 4B done in {res_4b['elapsed_sec']}s (peak {res_4b['peak_vram_gb']} GB)", flush=True)
    res_12b = eval_model(*MODELS["gemma_12b"], rows, bs)
    print(f"[P2b] 12B done in {res_12b['elapsed_sec']}s (peak {res_12b['peak_vram_gb']} GB)", flush=True)

    output = {
        "hardware": gpu_name,
        "vram_gb": round(vram_gb, 2),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "n_scenarios": len(ORIG_HELDOUT_SCENARIOS) + len(NEW_HELDOUT_SCENARIOS),
        "case_meta": rows,
        "gemma_4b": res_4b,
        "gemma_12b": res_12b,
    }
    out_path = Path("/content/p2b_gemma_scale.json")
    tmp_out = out_path.with_suffix(".tmp.json")
    tmp_out.write_text(json.dumps(output))
    tmp_out.replace(out_path)
    print(f"[P2b] Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)
    print("P2B_SUCCESS_MARKER", flush=True)


if __name__ == "__main__":
    main()
