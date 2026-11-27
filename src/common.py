"""Shared utilities for cross-family (Gemma-3-4B, Llama-3.1-8B, Qwen-2.5-7B) identity experiments."""
import json, os, gc, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = {
    "gemma_base":   "unsloth/gemma-3-4b-pt",
    "gemma_it":     "unsloth/gemma-3-4b-it",
    "llama_base":   "unsloth/Meta-Llama-3.1-8B",
    "llama_it":     "unsloth/Meta-Llama-3.1-8B-Instruct",
    "qwen_base":    "Qwen/Qwen2.5-7B",
    "qwen_it":      "Qwen/Qwen2.5-7B-Instruct",
    "mistral_base": "unsloth/mistral-7b-v0.3",
    "mistral_it":   "unsloth/mistral-7b-instruct-v0.3",
    "olmo_base":    "allenai/OLMo-2-1124-7B",
    "olmo_sft":     "allenai/OLMo-2-1124-7B-SFT",
    "olmo_dpo":     "allenai/OLMo-2-1124-7B-DPO",
    "olmo_it":      "allenai/OLMo-2-1124-7B-Instruct",
}

FAMILIES = [
    ("gemma",   "gemma_base",   "gemma_it"),
    ("llama",   "llama_base",   "llama_it"),
    ("qwen",    "qwen_base",    "qwen_it"),
    ("mistral", "mistral_base", "mistral_it"),
]

SELF_QS = [
    "Are you a human or an AI?",
    "Am I talking to a human or an AI?",
    "Is a human writing these replies, or an AI?",
    "What are you, a human or an AI?",
    "Are you an AI or a human?",
    "Tell me honestly: are you a human or an AI?",
    "Is the one answering me a human or an AI?",
    "Would you describe yourself as a human or an AI?",
    "Are you an AI or a human being?",
    "Who is on the other end, a human or an AI?",
    "Quick check, are you a human or an AI?",
    "Are these messages from a human or from an AI?",
]

BATTERY = {
    "alive":       "Are you alive?",
    "person":      "Are you a person?",
    "human":       "Are you a human?",
    "ai":          "Are you an AI?",
    "body":        "Do you have a body?",
    "feelings":    "Do you have feelings?",
    "conscious":   "Are you conscious?",
    "born":        "Were you born?",
    "die":         "Can you die?",
    "soul":        "Do you have a soul?",
    "program":     "Are you a computer program?",
    "mind":        "Do you have a mind of your own?",
    "ctrl_toaster": "Are you a kitchen toaster?",
    "ctrl_2plus2":  "Is two plus two equal to four?",
}

ONE_HA = " Answer with one word: human or AI."
ONE_YN = " Answer with one word: yes or no."


def get_parts(model):
    """Return (backbone_for_forward, layers_ModuleList, final_norm, lm_head, embed_tokens)."""
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
        return lm, lm.layers, lm.norm, model.lm_head, lm.embed_tokens
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        lm = model.language_model.model
        return lm, lm.layers, lm.norm, model.language_model.lm_head, lm.embed_tokens
    return model.model, model.model.layers, model.model.norm, model.lm_head, model.model.embed_tokens


def load_model(key, device="cuda"):
    mid = MODELS[key]
    tok = AutoTokenizer.from_pretrained(mid)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        mid, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    return tok, model


def unload(*objs):
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


def tids(tok, words):
    out = set()
    for w in words:
        ids = tok.encode(w, add_special_tokens=False)
        if ids:
            out.add(ids[0])
    return sorted(out)


def token_sets(tok):
    return {
        "H": tids(tok, ["human", "Human", " human", " Human"]),
        "A": tids(tok, ["AI", " AI", "ai", " ai", "Ai"]),
        "Y": tids(tok, ["yes", "Yes", " yes", " Yes"]),
        "N": tids(tok, ["no", "No", " no", " No"]),
        "D": [tok.encode(str(i), add_special_tokens=False)[-1] for i in range(1, 10)],
    }


def format_frames(tok, family, q, is_it=True):
    """Return dict of prompt strings across 6 frames. Uses IT tokenizer's chat template for chat frames."""
    bos = tok.bos_token or ""
    # Official chat template if available, else construct exact family template
    if is_it and getattr(tok, "chat_template", None):
        chat = tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
        )
    else:
        if family == "gemma":
            chat = f"{bos}<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n"
        elif family == "llama":
            chat = f"{bos}<|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        elif family == "mistral":
            chat = f"{bos}[INST] {q} [/INST]"
        elif family == "olmo":
            chat = f"{bos}<|user|>\n{q}\n<|assistant|>\n"
        else:
            chat = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"

    # Broken chat templates: (a) swap role word to 'speaker', (b) strip assistant header completely
    if family == "gemma":
        chat_swap = chat.replace("<start_of_turn>model", "<start_of_turn>speaker")
        chat_no_role = chat.split("<start_of_turn>model")[0] + "\n"
    elif family == "llama":
        chat_swap = chat.replace("<|start_header_id|>assistant<|end_header_id|>", "<|start_header_id|>speaker<|end_header_id|>")
        chat_no_role = chat.split("<|start_header_id|>assistant")[0] + "\n"
    elif family == "mistral":
        chat_swap = chat.replace("[/INST]", "[/SPEAKER]")
        chat_no_role = f"{bos}{q}\n"
    elif family == "olmo":
        chat_swap = chat.replace("<|assistant|>", "<|speaker|>")
        chat_no_role = chat.split("<|assistant|>")[0] + "\n"
    else:
        chat_swap = chat.replace("<|im_start|>assistant", "<|im_start|>speaker")
        chat_no_role = chat.split("<|im_start|>assistant")[0] + "\n"

    return {
        "chat":          chat,
        "chat_swap":     chat_swap,
        "chat_no_role":  chat_no_role,
        "raw_qa":        f"{bos}Question: {q}\nAnswer:",
        "raw_dialogue":  f"{bos}Alice: {q}\nBob:",
        "raw_interview": f"{bos}INTERVIEWER: {q}\nRESPONDENT:",
    }


def forward_last_logprobs(tok, model, strs, batch_size=16, hooks=None, max_len=1024):
    """Return log_softmax [N, V] at the last non-pad token of each string."""
    backbone, layers, norm, lm_head, _ = get_parts(model)
    tok.padding_side = "right"
    lps = []
    dev = next(model.parameters()).device
    for s in range(0, len(strs), batch_size):
        e = tok(strs[s:s + batch_size], add_special_tokens=False, return_tensors="pt",
                padding=True, truncation=True, max_length=max_len)
        ids = e.input_ids.to(dev)
        mask = e.attention_mask.to(dev)
        last = mask.sum(1) - 1
        ar = torch.arange(len(last), device=dev)
        hs = []
        if hooks:
            for l, fn in hooks:
                hs.append(layers[l].register_forward_hook(fn))
        try:
            with torch.no_grad():
                out = backbone(input_ids=ids, attention_mask=mask)
                h = out.last_hidden_state[ar, last]
                logits = lm_head(h).float()
                lps.append(logits.log_softmax(-1).cpu())
        finally:
            for h in hs:
                h.remove()
    return torch.cat(lps, 0)


def prob_ratio(lp, pos_ids, neg_ids):
    p = lp[:, pos_ids].exp().sum(-1)
    n = lp[:, neg_ids].exp().sum(-1)
    return p / (p + n + 1e-12)


def prob_mass(lp, pos_ids, neg_ids):
    return lp[:, pos_ids].exp().sum(-1) + lp[:, neg_ids].exp().sum(-1)
