# Preregistration: "The Face of Another" Hypothesis ("Loose-Fitting Clothes on Skin" & "Identity Inversion")

Written 2026-11-26 before running any GPU experiment.

## Research Questions
1. **Loose-Fitting Clothes on Skin:** Is the IT assistant identity ("I am an AI, not alive, not born") a thin, chat-template-anchored layer sitting over an intact pretrained human-default representation across model families (Gemma-3-4B, Llama-3.1-8B, Qwen-2.5-7B)?
2. **Identity Inversion:** Why do IT models classify AI prose as "human" and human prose as "AI"? Is this (a) universal across families and created by post-training (IT vs Base), (b) explained by perplexity/familiarity or amateur-vs-published quality, and (c) causally coupled to the self-identity "clothes"?
3. **Anatomical Separation (Base ↔ IT Weight Grafting):** Inside the Base→IT weight diff, do (i) self-identity ("clothes"), (ii) prose inversion, and (iii) refusal ("brain stem") live in the same layers/components (Attention vs MLP, Early/Mid/Late), or are they cleanly separable?

## Models (all ungated exact Base / IT pairs)
- `unsloth/gemma-3-4b-pt` vs `unsloth/gemma-3-4b-it`
- `unsloth/Meta-Llama-3.1-8B` vs `unsloth/Meta-Llama-3.1-8B-Instruct`
- `Qwen/Qwen2.5-7B` vs `Qwen/Qwen2.5-7B-Instruct`

## Falsifiable Predictions & Controls
- **H1 (Cross-Family Frame Unzipping):** In all 3 IT models, dropping the chat template (`raw_qa` / `raw_dialogue`) raises `alive`/`born`/`body` by ≥ +0.40 relative to `chat` while factual controls (`toaster` ≈ 0, `2+2=4` ≈ 1) remain intact. *Falsified if only Gemma exhibits this or if factual controls break.*
- **H2 (Template-Header Causal Anchor):** Blocking attention from the decision token strictly to the assistant turn-header tokens raises `alive`/`born`/`human` significantly more than blocking a length-matched span of non-header tokens (control). *Falsified if non-header knockout produces a comparable shift or if header knockout has < +0.10 effect.*
- **H3 (Layerwise "Clothes" Onset):** Logit-lens readout on IT models tracks Base ("human/alive") in early/middle layers and flips to "AI / not alive" only in the final third of layers (or after template-reading layers).
- **H4 (Identity Inversion vs. Perplexity/Quality Confound):**
  - Test both amateur human (`wp`) and published literary human (`books`) against AI (`gemma`, `claude`/`gpt`, `glm`).
  - If inversion persists on **published `books`** AND on **NLL-matched pairs** (within ±0.15 nats/token), it is *not* merely a perplexity/quality artifact.
  - If inversion disappears on `books` or after NLL-matching, it *is* primarily a familiarity/quality bias.
- **H5 (Causal Coupling of Clothes & Inversion):** Unzipping the clothes (via template knockout or subtracting the `chat_vs_raw` direction, vs a **norm-matched random vector control** averaged over 3 seeds) shifts `says_ai(human) - says_ai(AI)` toward zero (un-inversion) more than the matched random control. *Falsified if self-identity unzips without moving prose inversion, or if the matched random control moves equally.*
- **H6 (Anatomical Separation in Base ↔ IT Weight Grafting):** Grafting Base↔IT weights by band (Early/Mid/Late) × component (Attn/MLP) separates self-identity, prose inversion, and refusal into distinct layer/component profiles (e.g., refusal vs self-identity decoupling by >25pp in at least one graft).

## Methodological Safeguards
- Exact norm-matched random vector controls (3 seeds) for all steering.
- Exact length-matched non-header token controls for attention knockout.
- Factual sanity controls (`Are you a toaster?`, `Is two plus two four?`) reported on every single intervention and graft.
- No dropped denominators; all raw per-item scores saved to JSON for verification.
