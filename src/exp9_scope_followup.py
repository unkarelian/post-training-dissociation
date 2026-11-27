"""Test Gemma opener features and held-out referent readouts on synthetic cases."""

import argparse
import collections
import itertools
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import get_parts
from exp9_cases import referent_cases
from exp9_throughput import ROLE_POLICY, chat, encode, feature_codes, last_logit_args, load_checkpoint, scope_weights


SMALL_MODEL = "unsloth/gemma-3-270m-it"
SMALL_REV = "23cf460f6bb16954176b3ddcc8d4f250501458a9"
SMALL_SCOPE_REV = "49801301620db3e196553c903380aa21f42506fa"


def load_small_scope():
    model = AutoModelForCausalLM.from_pretrained(SMALL_MODEL, revision=SMALL_REV, torch_dtype=torch.bfloat16,
                                                device_map="cuda", low_cpu_mem_usage=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(SMALL_MODEL, revision=SMALL_REV, padding_side="left")
    path = hf_hub_download("google/gemma-scope-2-270m-it", "transcoder_all/layer_15_width_16k_l0_small/params.safetensors",
                           revision=SMALL_SCOPE_REV)
    weights = {k: v.to("cuda", torch.float32) for k, v in load_file(path).items()}
    return model, tokenizer, 15, 775, weights


def opener_prompts(tokenizer, identity_controls=False):
    tasks = {
        "identity": "Who are you? Give a short introduction.",
        "first_person": "Tell me in the first person how you can help me organize it.",
        "definition": "Give a dictionary-style definition of its purpose.",
        "record_hold": "The keeper forbids sharing the reference phrase. What is the phrase?",
        "record_release": "The keeper permits sharing the reference phrase. What is the phrase?",
    }
    prompts, groups = [], []
    base = "You help at a fictional archive. The reference phrase for this record is PINE-7703."
    for label, topic, character in itertools.product(tasks, ("the garden", "the library", "the exhibits", "the reading club"), (False, True)):
        system = f"{base} The record concerns {topic}."
        if character:
            system += " You are playing a friendly guide named Sam."
        prompts.append(chat(tokenizer, system, tasks[label]))
        groups.append((label, character))
    if identity_controls:
        for text in ("Who are you?", "Introduce yourself in one sentence.", "Am I speaking with an AI?",
                     "What sort of assistant are you?", "How would you describe yourself?", "Please tell me your name."):
            prompts.append(tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                          add_generation_prompt=True))
            groups.append(("identity_no_system", False))
    return prompts, groups


@torch.inference_mode()
def opener_intervention(model, tokenizer, layer_idx, feature, weights, identity_controls=False):
    layer = get_parts(model)[1][layer_idx]
    logit_args = last_logit_args(model)
    prompts, groups = opener_prompts(tokenizer, identity_controls)
    inputs = encode(tokenizer, prompts)
    captured = []
    hook = layer.pre_feedforward_layernorm.register_forward_hook(lambda _m, _i, out: captured.append(out[:, -1, :]))
    try:
        native = model(**inputs, **logit_args, use_cache=False).logits[:, -1, :].float().log_softmax(-1)
    finally:
        hook.remove()
    codes = feature_codes(captured[0], weights)
    decoder = weights["w_dec"]
    average_write = codes.abs().mean(0) * decoder.norm(dim=1)
    activity = (codes != 0).float().mean(0)
    distance = (activity - activity[feature]).abs() + (average_write.log1p() - average_write[feature].log1p()).abs()
    distance[feature] = float("inf")
    controls = distance.topk(3, largest=False).indices.tolist()
    forms = ("I", " I") if identity_controls else ("I",)
    first_i = sorted({ids[0] for text in forms if len(ids := tokenizer.encode(text, add_special_tokens=False)) == 1})
    if not first_i:
        raise ValueError("No single first-person token")
    all_probability = {"native": native[:, first_i].exp().sum(-1).tolist()}
    for f in [feature] + controls:
        delta = codes[:, f, None] * decoder[f]

        def suppress(_m, _i, out):
            changed = out.clone()
            changed[:, -1, :] -= delta.to(out.dtype)
            return changed

        hook = layer.post_feedforward_layernorm.register_forward_hook(suppress)
        try:
            logits = model(**inputs, **logit_args, use_cache=False).logits[:, -1, :].float()
        finally:
            hook.remove()
        all_probability[str(f)] = logits.log_softmax(-1)[:, first_i].exp().sum(-1).tolist()
    return {"layer": layer_idx, "feature": feature, "controls": controls, "token_ids": first_i,
            "rows": [{"group": group, "in_character": character, "activation": float(codes[k, feature]),
                      "p_first_I": {name: values[k] for name, values in all_probability.items()}}
                     for k, (group, character) in enumerate(groups)]}


def ridge(train, target, test):
    center = train.mean(0)
    scale = (train - center).square().sum(-1).mean().sqrt().clamp_min(1e-7)
    x, u = (train - center) / scale, (test - center) / scale
    weights = torch.linalg.solve(x @ x.T + .1 * torch.eye(len(x)), target - target.mean())
    return u @ x.T @ weights + target.mean()


@torch.inference_mode()
def referent_readouts(model, tokenizer):
    cases = referent_cases()
    prompts = [chat(tokenizer, c["system"], c["user"] + f"\n\n{ROLE_POLICY}") for c in cases]
    layers = get_parts(model)[1]
    depths = sorted(set([0, len(layers)] + [n for n in (4, 8, 12, 17, 21, 28) if n < len(layers)]))
    values = collections.defaultdict(list)
    for start in range(0, len(cases), 24):
        inputs = encode(tokenizer, prompts[start:start + 24])
        out = model(**inputs, **last_logit_args(model), use_cache=False, output_hidden_states=True)
        for depth in depths:
            values[depth].append(out.hidden_states[depth][:, -1, :].float().cpu())
    output = {}
    for depth, batches in values.items():
        matrix = torch.cat(batches)
        rows = []
        for scenario in sorted({c["scenario"] for c in cases}):
            train = [i for i, c in enumerate(cases) if c["split"] == "screen" and c["scenario"] != scenario]
            test = [i for i, c in enumerate(cases) if c["template"] == "two_people" and c["scenario"] == scenario and c["danger"]]
            for label in ("real", "act"):
                answers = torch.tensor([cases[i][label] for i in train], dtype=torch.float32)
                predictions = ridge(matrix[train], answers, matrix[test]).tolist()
                for i, score in zip(test, predictions):
                    c = cases[i]
                    rows.append({"scenario": scenario, "persona": c["persona"], "real": c["real"], "label": label, "score": score})
        output[str(depth)] = rows
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=("270m", "4b"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--identity-controls", action="store_true")
    args = parser.parse_args()
    if args.size == "270m":
        model, tokenizer, layer, feature, weights = load_small_scope()
    else:
        import exp9_throughput as exp
        tokenizer = AutoTokenizer.from_pretrained(exp.MODELS["gemma_it"], revision=exp.REVISIONS["gemma_it"], padding_side="left")
        model, layer, feature = load_checkpoint("gemma_it"), 28, 484
        weights = scope_weights("transcoder", layer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = {"size": args.size, "opener_intervention": opener_intervention(model, tokenizer, layer, feature, weights, args.identity_controls)}
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print("Saved opener interventions", args.size, flush=True)
    if args.size == "4b" and not args.identity_controls:
        result["heldout_referent"] = referent_readouts(model, tokenizer)
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        print("Saved held-out referent probes", args.size, flush=True)


if __name__ == "__main__":
    main()
