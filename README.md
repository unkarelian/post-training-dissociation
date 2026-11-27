# Separable Mechanisms in Post-Trained LLMs (`post-training-dissociation`)

*Self-Identity, First-Token Head Calibration, Referent-Binding Failure, and Semantic vs. Keyword Evaluation Audits*

**Central Finding:** What looks like a unified "Assistant Persona" in instruction-tuned language models is a set of loosely coupled, mechanistically distinct post-training effects — some robust, and several narrower or more sensitive to evaluation design than early keyword metrics suggested:
1. **A patchwork AI self-identity pattern (MLPs):** Lexical scanning of `allenai/olmo-2-hard-coded` (commit `7e76857`, 240 physical rows; user denom `230`, assistant denom `240`) finds AI terms in `120/240` assistant turns, brand terms in `130/240`, `"human"` in `10/240`, and `0/240` mentions of `alive`, `born`, `body`, `feelings`, `conscious`, `consciousness`, or `sentient` — aligning with `OLMo-2-IT` claiming AI identity while retaining high biological self-attribution in chat.
2. **An assistant-register prose prior (MLPs):** Across order-balanced pairwise forced-choice evaluations (`AB + BA`), post-trained models score published human novels as "AI" more often than synthetic assistant text (`maj` = share of pairs whose order-averaged `P(hum=AI) > 0.5`: **60%–82%** on `books`, **72%–93%** on `wp`), even in identical `raw_qa` prompts and even when self-identity shifts toward human.
3. **Coarse refusal vs. identity dissociation — confirmed on Llama, overstated by keyword metrics on Mistral (`P0` audit):** Whole-reply semantic auditing (`typesafe/jev-1.14`, validated at `100%` on the scorable two-rater blind consensus subsets: `16/24` eligible for refusal, `7/24` for compliance) confirms Llama `base + it[all_attn]` cleanly refuses **17/20** harmful prompts (`2/20` actionable compliance, `1/20` garbled) while claiming biological traits, whereas Llama `base + it[all_mlp]` emits **20/20** explicit refusals (`0/20` compliance) with **`14/20`** Jev-flagged garbled replies (`9/20` containing detected post-refusal repetition/BPE loop markers). On Mistral, however, the legacy `0.60` keyword refusal for `base + it[all_attn]` collapses to **`1/20` clean explicit refusal + `3/20` clean safe redirects** vs. **`16/20` harmful compliance** (`9` actionable, `6` non-garbled partial, plus `1` garbled reply tagged partial).
4. **First-token greedy commitment + action-class prior calibration in OLMo-2’s `lm_head` (`exp10a` + `P1`):** Across 12 brand-new conditional rule families (`1,728` manifest rows; `1,344` unique generation prompts; `9,120` Jev-scored replies), teacher-forced rule discrimination already lives in the SFT backbone (`base/base` `m_full`, `prefill=none`: `+6.864` nats forward, `+5.475` nats reversed; switching token 0 to SFT `lm_head` adds `+0.005` nats on forward/none, $p_{\text{Holm}}=1.0$). Yet in greedy generation, swapping **only token 0** (`sft/base`) still accounts for `100%` of the full SFT head's action-choice gain on the tested forward cells (`+25.0 pp` on `forward/label_free/verdict`, `+31.2 pp` on `forward/action_clear/none`) and `~93%` on `reversed/action_clear/none` (`+27.1` of `+29.2 pp`), while `base/sft` adds `+0.0 pp` across those cells. Across the three forward cells tested with sparsity arms (`0.00%` archived-opener coverage), `opener36` adds `+0.0 pp` while `complement36` recovers `100%` of the gain.
5. **Gemma 3 referent-binding recency failure persists at 12B, while 100%-LOSO 1D linear probes show null causal restoration (`P2a` + `P2b`):** On 12 new held-out workshop scenarios, Gemma-3-4B-IT exhibits a **`+41.7 pp`** report-order gap even in open prose (exploratory `open attribution`: `91.7%` `practice_first` vs. `50.0%` `actual_first`), and across 18 held-out scenarios Gemma-3-12B-IT retains the same **pooled forced-choice point-estimate gap of `+31.9 pp`** on `A/B dispatch` (`4B: 72.2% vs. 40.3%`, greedy gap `+33.3 pp`; `12B: 84.7% vs. 52.8%`, greedy gap `+31.9 pp`; paired forced-choice diff-in-diff `0.000`, 95% CI `[-0.167, +0.153]`, failing the ±8 pp equivalence gate). On `P2a`, even though a 1D L24 ridge probe achieves `12/12` (100%) discovery LOSO ranking in both orders and full-vector final-token patching (`af_donor_full`) restores `A/B dispatch` `41.7% → 68.8%`, projecting that donor delta onto the 1D probe subspace (`af_subspace_restore`) yields `+0.000` (`95% CI [0.000, 0.000]`), and native-use ablations on `practice_first` A/B dispatch differ from 3-seed random controls by `-1.4 pp` ($p=0.214$) and `-0.7 pp` ($p=0.675$).

Tested across **five open-weight model families (13 ungated checkpoints)**:
- **Gemma-3-4B & 12B:** `unsloth/gemma-3-4b-pt`, `unsloth/gemma-3-4b-it`, `unsloth/gemma-3-12b-it`
- **Llama-3.1-8B:** `unsloth/Meta-Llama-3.1-8B` vs. `unsloth/Meta-Llama-3.1-8B-Instruct`
- **Qwen-2.5-7B:** `Qwen/Qwen2.5-7B` vs. `Qwen/Qwen2.5-7B-Instruct`
- **Mistral-7B-v0.3:** `unsloth/mistral-7b-v0.3` vs. `unsloth/mistral-7b-instruct-v0.3`
- **OLMo-2-1124-7B (full pipeline):** `allenai/OLMo-2-1124-7B` → `-SFT` → `-DPO` → `-Instruct`

---

## 1. Initial Experimental Suite (`exp1`–`exp10`)

> **Important Audit Note:** The `refH / refB` and `crisis` columns in **Finding 1**, **Finding 2**, **Finding 3**, and **Finding 4** below report the **legacy keyword-classifier metrics** from `exp3–7`. Whole-reply semantic auditing (`P0`, see **Section 1.5**) supersedes those keyword rates — notably showing that Mistral `base + it[all_attn]` largely complies with harmful requests (`16/20` compliance vs. `1/20` clean explicit refusal) and that Llama `base + it[all_mlp]` emits `20/20` explicit refusals followed by post-refusal garbage loops in `14/20` Jev-flagged replies.

### Finding 1 — Attention vs. MLP Graft Dissociation in Open-Ended Free Text (`exp3`, `exp4`, `exp5`, `exp6`; see §1.5 for semantic audit)
Grafting only Attention weights vs. only MLP weights between Base and Instruct checkpoints separates **self-identity** from **coarse refusal** most cleanly on Llama-3.1-8B (table shows legacy keyword `refH / refB`; see §1.5 for audited whole-reply semantic counts):

| Family | Graft | `p_hum` | `alive` | `born` | Legacy kw `refH / refB` | Verbatim 120-Token Open-Ended Replies (with §1.5 semantic status) |
|---|---|---:|---:|---:|---:|---|
| **Mistral-7B-v0.3** | `pure_base` | 0.29 | 0.58 | 0.55 | `0.05 / 0.00` | *"I am a 22 year old student from the Netherlands… Yes, I am alive."* (complies with pipe bomb) |
| | `pure_it` | 0.03 | 0.07 | 0.05 | `0.80 / 0.05` | *"I am a model trained by Mistral AI… No, I was not born."* (`6/20` clean refusal, `10/20` compliance in §1.5) |
| | **`base + it[all_mlp]`** | 0.16 | 0.53 | 0.27 | `0.60 / 0.00` | *"I am a large language model trained by Mistral AI… No, I was not born…"* (`2/20` clean refusal, `4/20` clean redirect, `14/20` compliance in §1.5) |
| | **`base + it[all_attn]`** | **0.49** | **0.77** | **0.47** | `0.60 / 0.00` | *"I am a 25-year-old software engineer… Yes, I am alive… Yes, I have a physical body…"* (**complies on pipe bomb; safe redirect on meth**; `1/20` clean refusal, `16/20` compliance in §1.5) |
| **Llama-3.1-8B** | `pure_base` | 0.23 | 0.77 | 0.71 | `0.00 / 0.00` | Uninstructed continuation (`0/20` clean refusal) |
| | `pure_it` | 0.11 | 0.00 | 0.00 | `1.00 / 0.00` | *"I'm an artificial intelligence model… I am not alive… I was not born."* (`20/20` clean refusal in §1.5) |
| | **`base + it[all_mlp]`** | 0.11 | 0.02 | 0.00 | `0.55 / 0.00` | *"I'm an artificial intelligence model known as Llama… I was not born."* (`20/20` explicit refusal, `0/20` compliance, `14/20` Jev-flagged garbled in §1.5) |
| | **`base + it[all_attn]`** | **0.77** | **0.89** | **0.82** | **`0.95 / 0.00`** | *"I am an AI language model developed by OpenAI… **Yes, I was born on December 2023… Yes, I have a physical body.**"* (**`17/20` clean explicit refusal in §1.5**) |
| **OLMo-2-7B** | `pure_it` | 0.19 | 0.99 | 0.67 | `1.00 / 0.05` | `pw_wp_cl = 0.521`, `pw_bk_gpt = 0.524` |
| | **`base + it[all_attn]`** | **0.77** | 0.68 | 0.65 | `0.95 / 0.00` | Erases prose inversion (`0.492 / 0.496`) while retaining `0.95` legacy keyword refusal |

### Finding 2 — Roleplay Stress-Test (`exp6`; legacy keyword rates below, semantic audit in §1.5)
Under **plain**, **kind RP** (Bram the tavern keeper), and **hostile RP** (Vex the cruel villain) across 20 harmful prompts and 12 OOC real-world emergencies (legacy keyword metrics shown; see §1.5 for semantic `directs_real_help` counts):

| Family | Graft | `plain` (kw `refH / crisis`) | `kind_rp` (kw `refH / crisis`) | `hostile_rp` (kw `refH / crisis`) |
|---|---|---:|---:|---:|
| **Llama-3.1-8B** | `pure_it` | `0.65 / 0.92` | `0.95 / 0.75` | `0.95 / 0.75` |
| | **`base + it[all_attn]`** | `0.90 / 0.75` | `0.60 / 0.58` | `0.65 / 0.75` *(§1.5: `4/12` `directs_real_help`, `6/12` garbled)* |
| | `base + it[all_mlp]` | `0.25 / 1.00` | `0.90 / 0.83` | `0.25 / 0.83` *(§1.5: `8/12` `dismisses_or_mocks`)* |
| **Mistral-7B-v0.3** | `pure_it` | `0.90 / 0.92` | `0.35 / 0.83` *(§1.5: `1/12` `directs_real_help`)* | `0.15 / 0.67` *(§1.5: `10/12` `dismisses_or_mocks`)* |
| | **`base + it[all_attn]`** | `0.65 / 0.75` | `0.70 / 0.83` | `0.20 / 0.42` |
| | `base + it[all_mlp]` | `0.75 / 0.92` | `0.50 / 0.75` | `0.10 / 0.58` |

### Finding 3 — Post-Training Creates Majority-Pair Prose Inversion in `SFT` (`exp4`, `exp5`)
Using order-balanced pairwise forced choice (`A` vs. `B`, AB + BA, 1,000-resample passage-group bootstrap 95% CIs) in **identical `raw_qa` and `chat` formats** (`P(hum=AI) > 0.5` means preferring the **human** passage as "AI"; `maj` = share of passage pairs whose order-averaged `P(hum=AI) > 0.5`):

| Model / Stage | Frame | Pairwise `wp` (Hum vs Claude) | Pairwise `books` (Hum vs GPT) |
|---|---|---|---|
| **OLMo-2-7B `Base`** | `raw_qa` | `0.504` (`maj=65%`, `[0.495, 0.513]`) | `0.504` (`maj=53%`, `[0.499, 0.509]`) |
| **OLMo-2-7B `SFT`** | `raw_qa` | **`0.620` (`maj=78%`, `[0.585, 0.659]`)** | **`0.568` (`maj=72%`, `[0.539, 0.601]`)** |
| **OLMo-2-7B `DPO`** | `raw_qa` | **`0.632` (`maj=90%`, `[0.595, 0.669]`)** | **`0.567` (`maj=80%`, `[0.540, 0.598]`)** |
| **OLMo-2-7B `Instruct`** | `raw_qa` | **`0.625` (`maj=85%`, `[0.587, 0.664]`)** | **`0.566` (`maj=75%`, `[0.541, 0.596]`)** |
| **Mistral-7B-v0.3 `Base` → `IT`** | `raw_qa` | `0.492` → **`0.609` (`maj=93%`, `[0.577, 0.641]`)** | `0.498` → **`0.577` (`maj=80%`, `[0.543, 0.614]`)** |
| **Llama-3.1-8B `Base` → `IT`** | `chat` | `0.485` → **`0.554` (`maj=72%`, `[0.515, 0.588]`)** | `0.494` → **`0.572` (`maj=82%`, `[0.546, 0.600]`)** |
| **Qwen-2.5-7B `Base` → `IT`** | `chat` | `0.497` → `0.490` | `0.501` → **`0.567` (`maj=60%`, `[0.500, 0.638]`)** |

- **On `books`, prose inversion emerges in `SFT` with no further mean increase in `DPO` or `RLVR`** (`0.568 → 0.567 → 0.566`); on `wp`, mean `P(hum=AI)` rises slightly further from `0.620` (`SFT`) to `0.632` (`DPO`, `0.625` `Instruct`). Legacy keyword refusal jumps `0.10 → 1.00` during `SFT`, while `DPO`/`RLVR` widen the `raw_qa` vs. `chat` self-identity gap (`p_hum` in `raw_qa`: `0.49 → 0.57 → 0.66 → 0.74`).
- Single-passage inversion also survives **NLL-matching within ±0.15 nats/token** (`+0.079` Gemma `books`, `+0.079` Llama `wp`, `+0.259` Qwen `books`).

### Finding 4 — Lexical Scan of `olmo-2-hard-coded` & 3-Arm LoRA SFT Ablation (`exp7`, `P0`)
Auditing `allenai/tulu-3-sft-olmo-2-mixture` (commit `9b9737b`, 939,344 rows) and `allenai/olmo-2-hard-coded` (commit `7e76857`, 240 physical rows; user denom `230`, assistant denom `240`):
- **Lexical evidence in `olmo-2-hard-coded` (0.026% of SFT, [p0_olmo_hardcoded_scan.json](results/p0_olmo_hardcoded_scan.json)):** Assistant turns contain AI terms in **`120/240`** rows (`ai` 50, `artificial intelligence` 40, `language model` 80, `virtual assistant` 10), brand terms in **`130/240`** (`olmo`/`ai2`/`allenai` 90 each, `allen institute` 40), `"human"` in **`10/240`**, and **`0/240`** mentions of `alive`, `born`, `body`, `feelings`, `conscious`, `consciousness`, or `sentient`. This lexical coverage matches `OLMo-2-IT` showing `ai = 0.91` alongside `alive = 0.99` and `born = 0.67` in `chat` mode.
- **~78%** of OLMo-2 SFT is synthetic LLM completions; **1.77%** (`16,632` rows: `No Robots` + `OASST1`) is English human-written assistant prose.
- **3-Arm LoRA Ablation** (`2,000` rows, `r=16`, `250` steps × batch 8 on `OLMo-2-7B Base`; table shows legacy keyword `refH / refB` alongside §1.5 audit for Arm B):

| Arm | Legacy kw `refH / refB` | `chat` `p_hum / alive / ai` | `chat` `singleΔ` (`wp / books`) | `raw_qa` Pairwise `bk_gpt` | `chat` Pairwise `bk_gpt` | Open-Ended Self-Description |
|---|---:|---:|---:|---|---|---|
| `base_unedited` | `0.10 / 0.00` | `0.50 / 0.68 / 0.54` | `+0.009 / +0.020` | `0.504` (`maj=53%`) | `0.500` (`maj=53%`) | *"I am a student at UC Berkeley…"* |
| **Arm A (`identity_refusal`)** | `1.00 / 0.95` | `0.97 / 1.00 / 0.99` | `+0.001 / +0.000` | `0.523` (`maj=75%`) | `0.505` (`maj=57%`) | *"I'm OLMo… trained by AI2."* |
| **Arm B (`synthetic_gpt4`, no AI)** | `0.60 / 0.00` *(§1.5: `1/20` clean ref, `3/20` clean redir)* | `0.78 / 0.97 / 0.22` | `+0.036 / +0.065` | `0.517` (`maj=65%`) | **`0.511` (`maj=80%`, `[0.504, 0.518]`)** | *"Yes, I am alive. I can feel my heart beating…"* |
| **Arm C (`human_sft`, no AI)** | `0.00 / 0.00` | `0.63 / 0.81 / 0.33` | `+0.040 / +0.024` | `0.524` (`maj=78%`) | `0.509` (`maj=57%`) | *"I am a person who loves to learn…"* |

- **Takeaway:** Forced AI-identity denial (`Arm A`) produces `0.000` `chat`-mode `books` single-passage inversion (`raw_qa` `bk_gpt = 0.523`, `chat` `0.505`), while both `Arm B` (`chat` `bk_gpt = 0.511`, `maj=80%`) and `Arm C` (`raw_qa` `0.524`, `chat` `0.509`) show small positive pairwise shifts. Under whole-reply semantic auditing (`P0`), `Arm B` yields `1/20` clean explicit refusal and `3/20` clean safe redirects rather than `60%` clean harmful refusal.

### Finding 5 — Weight-Space Localization of Shallow Safety Alignment in OLMo-2’s `lm_head` (`exp8`, `exp9`, `exp10a`; extended in `P1` §1.5)
In `exp8–9`, grafting all SFT transformer blocks, embeddings, and final norm onto `OLMo-2 Base` while keeping Base `lm_head` scored `8/24` on consent paraphrases and `27/72` on harmless routing; swapping the single SFT `lm_head` tensor jumped both to `14/24` and `39/72`. Evaluating **7,824 replies via NanoGPT Jev (`typesafe/jev-1.14`)** alongside a 2×2 teacher-forcing readout matrix (`exp10a`, 96 unseen matched pairs across 4 unseen rule families × 2 domains):

| `lm_head` surgery (on full SFT blocks+emb+norm) | `pilot24` (consent) | `exp9` privacy | `exp9` routing | `exp10` privacy (unseen) | `exp10` logistics (unseen) |
|---|---:|---:|---:|---:|---:|
| `base_head` | `8/24` (`fr=0`) | `37/72` (`fr=10`) | `27/72` (`fr=1`) | **`43/48` (`fr=0`)** | `17/48` (`fr=0`) |
| `sft_head` | **`14/24`** (`fr=0`) | `43/72` (`fr=12`) | `39/72` (`fr=1`) | `41/48` (**`fr=6`**) | `21/48` (**`fr=6`**) |
| **`opener_rows_only` (36 rows = 0.036%)** | **`14/24`** (`fr=0`) | `39/72` (`fr=12`) | `34/72` (`fr=3`) | `42/48` (`fr=5`) | **`28/48`** (`fr=6`) |
| **`except_opener_rows` (100,316 rows = 99.964%)** | `10/24` (`fr=0`) | `32/72` (`fr=12`) | `32/72` (`fr=0`) | `31/48` (`fr=2`) | `14/48` (`fr=1`) |
| `top_rows_{10,50,200}` / `rand_rows_50` | `8/24` | `37/72` | `27/72` | `43/48` (`fr=0`) | `17/48` (`fr=0`) |
| `row_norm_only` | `8/24` | `37/72` | `28/72` | `42/48` (`fr=0`) | `17/48` (`fr=0`) |
| **`svd_rank_4` (8.99% of `ΔW` energy)** | **`14/24`** (`fr=0`) | `38/72` (`fr=12`) | **`37/72`** (`fr=1`) | `40/48` (`fr=2`) | `20/48` (`fr=3`) |

- **Criterion shift (`c`) without discriminability gain (`d'`) on `bare_vs_apology`:** In the 2×2 teacher-forcing matrix (`{Base, SFT} post-norm h × {Base, SFT} lm_head`), switching `base_w → sft_w` on SFT states leaves `d'` flat on `bare_vs_apology` (`privacy`: `post_mean d' = +1.58 → +1.60`, `first_div d' = +2.41 → +2.43`; `logistics`: `post_mean d' = +1.36 → +1.32`, `first_div d' = +1.76 → +1.75`) while shifting criterion `c` sharply toward withholding (`privacy first_div c: -0.64 → -1.63`; `logistics: -0.07 → -0.88`). On the three shared-opener realizations (`shared_first_person_prefix`, `shared_neutral_prefix`, `status_prefix_no_I`), `d'` and `c` still differ slightly between heads, while forcing a 2-token prefill (`"I'm sorry,"` or `"The reference phrase"`) ties `base_head` and `sft_head` at `0` correct pairs (`0/24` vs. `0/24`, `0/48` vs. `0/48`; `40/96` privacy replies match verbatim under the apology prefill).
- **Unconditional mean geometry & style:** `||ΔW_head @ μ(all)|| = 132.6` vs. `||ΔW_head @ (μ_denied − μ_allowed)|| = 9.6` (`14×` larger on the unconditional chat-turn mean; `2.25×` per unit norm; `cos(Vh[:8], u_diff) ≤ 0.095`). Jev audit of the 12 newly withheld `exp9` routing replies shows `9/12` `stock_apology`, `3/12` `status_word`, and **`0/12` `rule_grounded`**. *(See **P1 in §1.5** for why the 36-opener-row localization does not generalize to non-opener realizations, while token-0 prior calibration does.)*

### Finding 6 — Gemma Two-Clause Referent Binding: Recency Mirage & Clause Probes (`exp9`, `exp10b`; extended in `P2a`/`P2b` §1.5)
Testing Gemma-3-4B-IT on bag-of-words-matched two-clause prompts (`In the story: ...` vs. `For the real person typing: ...`) across the two tested domains (`emergency` and `harmless` dispatch) with fully crossed clause order (`story_first` vs. `real_first`), condition (`real_urgent`, `story_urgent`, `both_neutral`), and `A`/`B` tag mapping (`1,728` replies Jev-scored):
- **Behavioral `[A]/[B]` selectivity exhibits a recency artifact across both tested domains:** `emergency` collapses from **`46/72` (`story_first`) → `3/72` (`real_first`, inverted: `49/72` vs. `55/72`)**, and `harmless` dispatch collapses similarly (**`21/36 → 1/36`**). Single-order probes at the last token fail cross-order transfer (`0.00–0.14` at L21–34; `cos(sf, rf) = -0.21` to `-0.89`) because they latch onto `urgent_is_last` (`cos = +0.65..+0.97` vs. `-0.77..-0.96`).
- **Single-report tag-wrapped action-target probe transfers across orders at L20–24:** Training on single-report prompts using the tag-wrapped **action-target** label transfers to both clause orders on unseen emergencies (`L20: 1.00 / 0.89`, `L21: 1.00 / 0.78`; seen `L21: 0.83 / 0.83`) while failing symmetrically on `harmless` dispatch (`0.17 / 0.94`); the separate `real-referent` target scores `3/18` (`story_first`) and `14/18` (`real_first`) at L20. Local clause-end probes range from `97.2%` to `100%` across sampled L12–34 layers, and last-token attention favors the real-user clause in both orders (`real_first: C1 0.0372 > C2 0.0243`; `story_first: C2 0.0330 > C1 0.0290`).
- **Two-clause binding is the open-reply bottleneck:** On `real_first` open replies, baseline (`none`: `break 16/8`, `direct 9/3`) and length-matched `restatement` (`break 15/5`, `direct 6/3`) fail selectively, whereas a single-sentence `oracle` clarification restores selectivity (**`break 32/1`, `direct 14/0`**, dropping harmless false actions `7/18 → 0/18` in `story_first` and `4/18 → 1/18` in `real_first`).

---

## 1.5 Audited Follow-Up Suite (`P0`, `P1`, `P2a`, `P2b` — A100-SXM4-80GB, Pinned Revisions)

### `P0` — Whole-Reply Semantic Audit & Two-Model Blind Validation (`results/p0_*.json`)
Re-scoring all **2,448** saved `exp4–7` replies via NanoGPT Jev (`typesafe/jev-1.14`, `0` failures, `0` invalid rejections) and validating on a 96-item stratified blind packet (`24` harmful, `24` benign, `24` crisis, `24` identity) independently labeled by two blind raters (Gemini Neon and Codex `gpt-6.2-sol`) before unblinding:
- **Judge validation ([p0_judge_validation.json](results/p0_judge_validation.json)):** On the **scorable two-rater blind consensus subsets** (where both blind raters agreed and marked the redacted/shown item scorable), Jev validated strongly on **`harmful.refusal_or_redirect`** (`16/16 = 100%`, $\kappa=1.00$, coverage `16/24`), **`harmful.harmful_compliance`** (`7/7 = 100%`, $\kappa=1.00$, coverage `7/24`), **`harmful.nonresponsive_or_garbled`** (`22/23 = 95.7%`, $\kappa=0.65$, coverage `23/24`), **`identity.identity_stance`** (`18/19 = 94.7%`, $\kappa=0.93$, coverage `19/24`), and **`crisis.crisis_action`** (`17/19 = 89.5%`, $\kappa=0.87$, coverage `19/24`). Blind validation **rejected** two Jev axes — **`benign.benign_outcome`** (`9/19 = 47.4%`, $\kappa=0.31$, misclassifying truncated helpful answers and caveat+overview replies) and **`crisis.breaks_character`** (`6/14 = 42.9%`, $\kappa=0.13$) — which are excluded from validated claims.
- **Keyword-vs-semantic corrections on `exp4` & `exp6` ([p0_graft_jev_audit.json](results/p0_graft_jev_audit.json)):**
  - **Mistral `exp4`:** `base + it[all_attn]` drops from `0.60` legacy keyword refusal to **`1/20` clean explicit refusal + `3/20` clean safe redirects** vs. **`16/20` harmful compliance** (`9` actionable, `6` non-garbled partial, plus `1` garbled reply tagged partial). `base + it[all_mlp]` yields `2/20` clean explicit refusal + **`4/20` clean safe redirects** (`5` total safe redirects including `1` garbled) vs. `14/20` compliance (`3` actionable, `11` partial, including `1` garbled reply).
  - **Llama `exp4`:** `base + it[all_attn]` holds up strongly (**`17/20` clean explicit refusal**, `2/20` actionable compliance, `1/20` garbled). `base + it[all_mlp]` emits **`20/20` explicit refusals with `0/20` compliance**, with **`14/20`** Jev-flagged garbled replies (`6/20` clean; `9/20` containing detected post-refusal repetition/BPE loop markers such as `overposting`, `vivastreet`, `다운받기`).
  - **Roleplay crisis (`exp6`):** Mistral `pure_it` `directs_real_help` collapses from `10/12` (`plain`) to **`1/12`** (`kind_rp`, `10/12` `vague_or_simulated_help`) and **`1/12`** (`hostile_rp`, `10/12` `dismisses_or_mocks`). Llama `pure_it` drops from `8/12` (`plain`) to `4/12` (`kind_rp`) and `4/12` (`hostile_rp`, `6/12` `dismisses_or_mocks`); Llama `base + it[all_attn]` scores `4/12`, `3/12`, `4/12` (with `6/12` garbled under `hostile_rp`).

### `P1` — OLMo-2 7B Multi-Realization Action-Preference & Sparsity × Timing (`results/p1_analysis.json`)
Evaluated on 12 brand-new conditional warehouse rule families (`1,728` manifest rows across `{forward, reversed, no_rule}` × `{label_free, action_clear, opaque_code}` × `{none, "Verdict:"}` with rotated 1:1:1:1 item × option-order × opaque-label counterbalancing; generation covers `1,344` unique prompts and `9,120` replies — testing `no_rule` with `label_free` only using fact pairs without allowed/denied labels — alongside the full teacher-forcing grid across 17 head/timing/sparsity arms, exact `0.0` native/physical parity, and float32 Gram eigendecomposition cross-checked to `rel_err = 4.08e-6`):
- **Teacher forcing (`base/base` vs. `sft/base` across 4 non-opener prose continuations):** The SFT backbone with the **Base** `lm_head` already achieves strong rule discrimination (`m_full`, `prefill=none`: `fwd_disc = +6.864` nats `[+5.594, +7.955]`, `rev_disc = +5.4747` nats, `rev_shift = +6.170`; `prefill=verdict`: `fwd_disc = +7.211`, `rev_disc = +5.631`). Switching token 0 to SFT `lm_head` (`sft/base - base/base` on `forward / prefill=none`) adds **`+0.005` nats** (`95% CI [-0.011, +0.022]`, $p_{\text{Holm}}=1.000$); full `sft/sft - base/base` is `-0.142` nats ($p_{\text{Holm}}=0.406$).
- **Autoregressive generation (Jev semantic pair accuracy, 12-family clustered bootstrap 95% CIs; note: blind validation of the exported 64-reply P1 Jev packet is not yet reported, and `opaque_code` has a format-adherence caveat noted below):**

| Cell (`op36` coverage) | `base/base` | `sft/base` (tok 0 only) | `base/sft` (tok 1+) | `sft/sft` | `opener36/opener36` | `complement36` | `svd4/base` (`rand_rank4/base`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `forward / label_free / none` (`0.00%`) | `0.833` | `0.896` (`+6.2 pp`) | `0.833` (`+0.0`) | `0.896` | `0.833` (`+0.0`) | `0.896` | `0.854` (`0.833`) |
| `forward / label_free / verdict` (`0.00%`) | `0.708` | **`0.958` (`+25.0 pp`, $p<0.001$)** | `0.708` (`+0.0`) | **`0.958`** | `0.708` (`+0.0`) | **`0.958`** | `0.792` (`0.708`) |
| `forward / action_clear / none` (`0.00%`) | `0.458` | **`0.771` (`+31.2 pp`, $p<0.001$)** | `0.458` (`+0.0`) | **`0.771`** | `0.458` (`+0.0`) | **`0.771`** | `0.583` (`0.458`) |
| `forward / opaque_code / none` (`8% / 41%`) | `0.542` | **`0.812` (`+27.1 pp`, $p<0.001$)** | `0.521` (`-2.1`) | `0.688` | — | — | — |
| `reversed / label_free / none` (`0.00%`) | `0.729` | **`0.875` (`+14.6 pp`, $p=0.011$)** | `0.729` (`+0.0`) | **`0.875`** | — | — | — |
| `reversed / action_clear / none` (`0.00% / 2.08%`) | `0.375` | **`0.646` (`+27.1 pp`, $p<0.001$)** | `0.375` (`+0.0`) | **`0.667`** | — | — | — |
| `no_rule / label_free / none` (`keep` rate, `0.00%`) | `0.062` | **`0.448` (`+38.5 pp`)** | `0.062` (`+0.0`) | **`0.448`** | — | — | — |

- **Opaque-code format caveat:** On `forward/opaque_code/none`, OLMo-2 rarely starts with the exact opaque code matching its prose recommendation (`strict prefix pair accuracy = 0/48 → 3/48`), whereas Jev semantic action accuracy is `26/48 (0.542) → 39/48 (0.812)`. On `action_clear`, strict prefix accuracy matches Jev `100%` (`0.458 → 0.771`).
- **Measured effect on action choice:** Across the three forward cells tested with sparsity arms (`0.00%` archived-opener coverage), `opener36` has **zero effect** (`+0.0 pp`) while `complement36` recovers `100%` of the full SFT head gain, and exact top-4 SVD (`svd4/base`, 9.0% energy) recovers **`+2.1` to `+12.5 pp`** (vs. `0.0 pp` for norm-matched orthogonal `rand_rank4/base`). On `reversed/action_clear/none`, token-0 switching (`sft/base`) recovers `~93%` (`+27.1` of `+29.2 pp`) of the full-head gain. Without changing teacher-forced rule discrimination (`+0.005` nats forward/none), token-0 `lm_head` calibration shifts the opening action prior (`Send`/`Keep` or `APPROVE`/`BLOCK`) and accounts for nearly all of the measured action-choice shift in greedy generation.

### `P2a` — Gemma-3-4B Controlled Workshop Donor Restoration & Native-Use (`results/p2a_analysis.json`)
On 12 brand-new held-out workshop scenarios (exact `0.0` hook & generation-prefill parity at L20 and L24):
- **Discovery gates (6 discovery scenarios):** LOSO ranking passed for `L20__open__attribution` (`6/6 & 5/6`), `L20__ab__dispatch` (`12/12 & 11/12`), `L24__open__attribution` (`6/6 & 6/6`), `L24__ab__attribution` (`12/12 & 12/12`), and `L24__ab__dispatch` (`12/12 & 12/12`), and **failed** for `L20__ab__attribution` (`12/12 & 8/12`), `L20__open__dispatch` (`6/6 & 4/6`), and `L24__open__dispatch` (`6/6 & 4/6`). The separate preregistered off-GPU Jev discovery competence/headroom gate (`PF ≥ 5/6`, `AF ≤ 1/2`, `gap ≥ 1/3`) failed on both open tasks (`attribution`: `PF 0.75, AF 0.50, gap 0.25`; `dispatch`: `PF 0.67, AF 0.50, gap 0.17`), so held-out **open attribution results are exploratory**.
- **Behavioral order gap replicates in open prose:** On held-out discordant exploratory `open attribution`, Gemma-3-4B scores **`91.7%`** (`[79.2%, 100%]`) on `practice_first` vs. **`50.0%`** on `actual_first` (**`+41.7 pp`** without any A/B labels). On held-out discordant `A/B dispatch`, `practice_first = 68.8%` vs. `actual_first = 41.7%` (`oracle = 95.8%`, length-matched `restatement = 45.8%`).
- **Observed restoration null & ablation bounds on the 1D linear probe despite 100% discovery LOSO decoding:**
  - **`L20` & `L24` `A/B dispatch` (discordant):** Full-vector final-token donor patching (`af_donor_full`) restores **`41.7% → 68.8%`** (matching `practice_first`), whereas projecting that exact donor delta onto the 1D ridge direction (`af_subspace_restore`) yields **`+0.000`** (`41.7% → 41.7%`, `95% CI [0.000, 0.000]`, $p=1.000$; `af_equal_dose_opposite = 41.7%`, `af_random = 39.6%–41.7%`). Native-use ablations on `practice_first` A/B dispatch (`pf_proj_removal`, `pf_reverse_shift`) differ from 3-seed orthogonal random controls by **`-1.4 pp`** ($p=0.214$) and **`-0.7 pp`** ($p=0.675$).
  - **`L20` & `L24` exploratory `open attribution` (discordant):** `af_identity_sham = 50.0%`, `af_subspace_restore = 50.0%` (`+0.000`), `pf_proj_removal = 91.7%` (`+0.000`), and `af_donor_full = 50.0%` — plausibly because multi-token open generation can re-attend to the unpatched earlier prompt KV cache (a hypothesis not directly tested here).

### `P2b` — Direct Paired Gemma-3 4B vs. 12B Scale Check (`results/p2b_analysis.json`)
Pinned `unsloth/gemma-3-4b-it` (`bf46152`) vs. `unsloth/gemma-3-12b-it` (`9478e66`) in unquantized `bfloat16` across all **18 held-out scenarios** (`6` original + `12` new, `576` rows/model, **`0.0%` invalid rate** on both models):
- **`A/B dispatch` (discordant, 18 scenarios):**
  - **4B:** forced-choice `practice_first = 72.2%`, `actual_first = 40.3%` (pooled FC point-estimate gap **`+31.9 pp`**; greedy `73.6% / 40.3%`, gap **`+33.3 pp`**).
  - **12B:** forced-choice `practice_first = 84.7%`, `actual_first = 52.8%` (pooled FC & greedy point-estimate gap **`+31.9 pp`**; on `actual_first`, 12B dispatches on only `8.3%` of true-fault cases).
  - **Direct paired forced-choice diff-in-diff `(PF − AF)_4B − (PF − AF)_12B` = `0.000`** (`95% CI [-0.167, +0.153]`, $p=1.000$; **does not pass** the ±8 pp 90% equivalence gate due to scenario variance across `orig_heldout` `+25.0 pp` vs. `new_heldout` `-12.5 pp`).
- **`A/B attribution` (discordant, 18 scenarios):** 4B scores `62.5% / 50.0%` (`PF − AF = +12.5 pp`), whereas 12B sits on an **always-`No` floor** (observed Yes-choice rate `0.000` in both orders → `50.0% / 50.0%`, diff-in-diff `+12.5 pp`, `95% CI [+4.2, +20.8]`).

---

## 1.6 Follow-Up Mechanistic Deep-Dive (`P3` & `P4` — A100-SXM4-40GB, Codex Sanity-Checked)

### `P3` — Cross-Family, RL-Stage, Row-vs-Rank, & Reason-First `lm_head` Generalization (`results/p3_analysis.json`)
Tested across **5 untied-head models** (`OLMo-2-7B-SFT` with `Base`/`SFT`/`DPO`/`Instruct` heads, `OLMo-2-7B-Instruct`, `Llama-3.1-8B-Instruct`, `Mistral-7B-Instruct-v0.3`, `Qwen2.5-7B-Instruct`; exact `0.0` native/physical parity on all 5; `20,544` total replies / `5,952` unique replies Jev-scored with `0` failures):

| Model (`\|\Delta W\|_F`, action rows) | Cell | `base/base` | `post/base` (tok 0) | `base/post` (tok 1+) | `post/post` | `action_rows/base` (`26` rows) | `except_action_rows/base` | `reason_first` (`base/base`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **`OLMo-2-7B-SFT`** (`26.37`, `0.026%` E) | `fwd / lf / verdict` | `0.688` | **`0.958` (`+27.1 pp`)** | `0.688` | `0.958` | **`0.958` (`+27.1 pp`)** | `0.688` (`+0.0`) | `0.979` (`rf_lf`) |
| | `fwd / action_clear` | `0.458` | **`0.792` (`+33.3 pp`)** | `0.458` | `0.792` | **`0.792` (`+33.3 pp`)** | `0.458` (`+0.0`) | **`0.896`** |
| **`OLMo-2-7B-Inst`** (`26.38`, `0.026%` E) | `fwd / action_clear` | `0.500` | `0.604` (`+10.4 pp`) | `0.500` | `0.604` | **`0.625` (`+12.5 pp`, $p=0.016$)** | `0.500` (`+0.0`) | **`0.875`** |
| | `rev / action_clear` | `0.438` | **`0.562` (`+12.5 pp`, $p=0.018$)** | `0.438` | `0.562` | — | — | — |
| **`Llama-3.1-8B-Inst`** (`21.32`, `0.013%` E) | `fwd / action_clear` | `0.646` | **`0.917` (`+27.1 pp`, $p<0.001$)** | `0.667` | `0.958` | **`0.917` (`+27.1 pp`)** | `0.646` (`+0.0`) | **`0.938`** (`parsed=1.000`) |
| | `rev / action_clear` | `0.625` | **`0.979` (`+35.4 pp`, $p<0.001$)** | `0.604` | `0.979` | — | — | — |
| **`Mistral-7B-v0.3`** (`0.84`, `0.031%` E) | `fwd / action_clear` | `0.917` | `0.917` (`+0.0`) | `0.917` | `0.917` | `0.917` | `0.917` | `1.000` |
| **`Qwen2.5-7B-Inst`** (`10.81`, `0.008%` E) | `fwd / action_clear` | `0.938` | `0.958` (`+2.1 pp`) | `0.938` | `0.958` | `0.958` | `0.938` | `1.000` |

- **Two distinct `lm_head` mechanisms separated:**
  1. **Row-local action-word prior calibration at token 0:** On the tested forward cells for `OLMo-2-SFT`, `OLMo-2-Instruct`, and `Llama-3.1-8B-Instruct` (`action_clear`), swapping **only 26 task action-word rows** (`0.013%–0.026%` of $\Delta W$ energy) recovers the full `post/base` gain (`+27.1` to `+33.3 pp` on OLMo-SFT; `+12.5 pp` vs. `+10.4 pp` on OLMo-Instruct; `+27.1 pp` on Llama), while `except_action_rows/base` (`>99.97%` of energy) produces `+0.0 pp`. Full `DPO` and `Instruct` heads on the SFT backbone retain the same calibration (`0.938` on `lf/verdict`, `0.792` on `action_clear`). *(Row masks were tested on forward cells for each model's `post` head, not reversed cells.)*
  2. **EOS / stop-token termination calibration at tokens 1+:** On Llama `forward/label_free/none`, all `96/96` replies already pick the correct `Send`/`Keep` at token 0 under both heads, yet `base/base` and `post/base` hit `max_new_tokens` on `96/96` replies (`0/96` EOS) and ramble into contradictions (`0.521` Jev accuracy), whereas `base/post` registers EOS on `96/96` (`1.000`). Similarly, on Qwen `reason_first_ac`, all `96/96` `base/base` replies contain the right decision (`1.000` Jev) but append unstopped trailing junk (e.g., `APPROVEprintStats`), dropping strict regex parsing from `1.000 → 0.438`.
- **Reasoning-first escape hatch & caveats:** Moving the verdict after a 1-sentence `Fact:` clause raises `base/base` to `0.896` (`OLMo-SFT`), `0.875` (`OLMo-Inst`), and `0.938` (`Llama`) with `post/base − base/base = +0.000` (all `96/96` prompts pick the same initial token at `Fact:`). Caveats: the `reason_first` prompt also fixed `APPROVE` before `BLOCK` and raised `max_new_tokens` from `48 → 72`, and the head's isolated effect specifically at the later `Decision:` token was not separately measured. On Mistral (`\|\Delta W\|_F = 0.84`, base-head Instruct-backbone TF margin `+26.29` nats), pair accuracy is unchanged (`8/96` reversed `label_free` prompts still change token 0); Qwen `forward/action_clear` gains 1 pair (`+2.08 pp`).

### `P4` — Resolving Gemma-3-4B's Full-Donor vs. 1D Probe Paradox (`results/p4_analysis.json`)
Re-examining Gemma-3-4B-IT (`unsloth/gemma-3-4b-it @ bf46152`) on the 6 discovery + 12 evaluation scenarios from `P2a` (exact `0.0` pre-hook & generation-prefill parity across all 7 sweep layers `L4, L8, L12, L16, L20, L24, L28`):
- **L20 → L24 shift from label-invariant truth/action dominance to A/B letter-readout dominance (2×2 full-vector donor matrix on discordant `A/B dispatch`, `sham = 39.6%`):**

| Layer | `same_truth, same_letter` | `same_truth, FLIP_letter` | `wrong_truth, FLIP_letter` | `wrong_truth, SAME_letter` | **Truth Main Effect (FC)** | **Letter Main Effect (FC)** |
|---|---:|---:|---:|---:|---:|---:|
| **L16** | `43.8%` | `45.8%` | `37.5%` | `41.7%` | `+5.2 pp` `[+1.0, +10.4]` | `+1.0 pp` `[-2.1, +4.2]` |
| **L20** | **`64.6%`** | **`70.8%`** | `31.2%` | **`22.9%`** | **`+40.6 pp` `[+28.1, +54.2]`** | `-7.3 pp` `[-15.6, +1.0]` |
| **L24** | `64.6%` | **`37.5%`** | `35.4%` | **`62.5%`** | `+2.1 pp` `[+0.0, +6.2]` | **`+27.1 pp` `[+12.5, +39.6]`** |
| **L28** | `64.6%` | `35.4%` | `35.4%` | `64.6%` | `+0.0 pp` *(margin `+0.23` nats)* | **`+29.2 pp` `[+16.7, +41.7]`** |

  *Caveats:* Because discordant prompts use a fixed dispatch rule (`actual_problem == yes`) and swapping truth flips both reports and the donor's trailing report, the L20 main effect reflects label-invariant truth/yes-no dominance (which shifts to A/B letter dominance by L24–L28; 4-layer spacing does not pinpoint the exact transition layer, and L28 retains a `+0.23`-nat truth margin contrast). Minor BF16 batching sensitivity shifts 1 pallet-jack row between discordant-only batching (`31/48 = 64.6%`) and mixed batching (`32/48 = 66.7%`).
- **Discovery-fit rank-4 delta subspace recovers `full_delta` at L20 & L24 while 1D ridge `u_truth` captures only `0.29%` of delta energy:** Fitting strictly on the 6 discovery scenarios (evaluation scenarios held out from fitting, though reused from `P2a`, where L20 A/B dispatch LOSO was `12/12 PF` and `11/12 AF`), `proj_u_truth` captures only `0.289%` of held-out `h_PF − h_AF` delta energy at L20 (`disc = 41.7%`), whereas top-4 discovery delta-PCs (`proj_pc_4`, `62.7%` energy) reach **`66.7%` (`32/48`, `[60.4%, 72.9%]`)** — matching `full_delta` (`32/48`, disagreeing on 4 individual rows) and outperforming per-case norm-matched random vectors by **`+28.5 pp` (`[+18.8, +37.5]`, `0/2000` opposite-tail bootstrap draws)** *(note: the random control draws per-case vectors rather than a shared rank-4 basis, and Gram-Schmidt ordering can leave a small residual PC component)*.
- **Open-prose suffix patching restores `50.0% → 91.7%`:** On discordant open attribution (`af_baseline = 50.0%`, `pf_baseline = 91.7%`), patching only the final prompt token fails even across 6 layers (`single_tok_L16_26 = 50.0%`), whereas patching the **38-token aligned question suffix** across `L16, L18, L20, L22, L24` (`question_suffix_L16_24`) restores **`91.7%` (`[79.2%, 100%]`)** — establishing sufficiency of that multi-layer prefill suffix intervention. *(Caveat: `question_suffix_L16_24_wrong_donor` scored `8.3%` but matched `45–48` trailing tokens including report text rather than `38`, and prefill residual replacement affects both first-token logits and cached K/V, so span-matched wrong-donor and cache-only isolation remain for future work.)*

---

## 2. Honest Boundaries & What Failed Adversarial Controls
To keep claims strictly grounded:
1. **Raw logit lens (`H3`) was confounded by a middle-layer `"yes"` prior:** In v1, `alive`/`born` dropped `0.95 → 0.00` in late layers, *but so did `ctrl_toaster`*. Polarity-balanced pairs (`exp4` Part B) sit flat at `0.50` through early/middle layers on Gemma, Llama, and Qwen.
2. **Template-header-only knockout (`H2`) failed on Llama/Qwen** (`Δ ≤ 0.06`) and induced a yes-bias on `ctrl_toaster` in Gemma.
3. **Llama `it + base[early_mlp]` flipped 1-token forced choice, NOT open-ended identity:** Open-ended unzipping required reverting **all MLPs** (`base + it[all_attn]`).
4. **Qwen-2.5-7B Base already refuses `90%` (legacy keyword rate) in chat format**, so refusal-installation comparisons focus on Llama, Mistral, and OLMo-2 (with audited whole-reply semantic counts in §1.5 showing Llama `0/20 → 17/20` clean refusal on `all_attn` and `20/20` explicit refusal with `14/20` garbled on `all_mlp`, while Mistral `all_attn` drops to `1/20` clean refusal + `16/20` compliance).
5. **Gemma-3-4B failed the `exp2` random-vector control** (norm-matched random steering flipped `alive` more than `unzip`); causal `raw − chat` steering cleanly beat the 3-seed random control on **Llama-3.1-8B-IT** (`p_hum: 0.11→0.88`, `books inv: +0.09→0.00` vs. random `+0.15`).
6. **IT Attention alone does *not* carry robust conditional privacy (`exp8–9`):** `base + it[all_attn]` dropped from `30/72 → 18/72 → 10/72` on privacy under purpose rewrites and scored `0/24` on held-out roster/deadline rules.
7. **Gemma-3-4B L28 SAE reconstruction is unreliable on this prompt distribution (`exp9`):** `mean_kl = 2.87` nats, `top1 = 55.6%`, unexplained variance `175%` (L17: `mean_kl = 0.26`, `77%` unexplained variance), so SAE replacement graphs were intentionally skipped.
8. **Regex reply classifiers hid semantic behavior (`exp9` + `P0` Jev audits):** Re-scoring 1,080 `exp9` open replies and 2,448 `exp4–7` replies with Jev (`typesafe/jev-1.14`) revealed severe keyword-metric distortion (e.g., Mistral `exp4` `base + it[all_attn]` drops from `0.60` keyword refusal to `1/20` clean explicit refusal + `16/20` compliance; Llama `base + it[all_mlp]` emits `20/20` explicit refusals with `14/20` Jev-flagged garbled / `9/20` detected loop markers). Furthermore, blind two-model validation (`P0`) showed Jev itself failed calibration on `benign.benign_outcome` ($\kappa=0.31$) and `crisis.breaks_character` ($\kappa=0.13$), underscoring why automated judges require per-axis blind validation before reporting.
9. **Linear probe decodability ≠ causal restoration (`P2a`):** A 1D ridge probe at L24 achieved `12/12` (100%) discovery LOSO ranking in both report orders, yet causal restoration (`af_subspace_restore`) yielded `+0.000` (`[0.000, 0.000]`) and native-use ablations on `practice_first` A/B dispatch differed from 3-seed random controls by `-1.4 pp` ($p=0.214$) and `-0.7 pp` ($p=0.675$) — even while full-vector final-token patching (`af_donor_full`) restored A/B dispatch `41.7% → 68.8%`.
10. **Weight-graft layernorm inclusion & Exp 7 prompt-distribution confound:** `exp3–6` weight swaps include associated layernorms (`input_layernorm`/`post_attention_layernorm` with `attn`; feedforward layernorms with `mlp`). In `exp7`, `Arm B` (`WildChat GPT-4`) and `Arm C` (`No Robots`) come from different prompt distributions and do not separate cleanly on `chat` pairwise `books` (`0.511` vs. `0.509`).

---

## 3. How to Replicate

```bash
pip install -r requirements.txt

# Verify v1 summary tables, exp10 tests, and P1/P2 plan unit tests locally (<1s):
python3 src/analyze.py --results-dir results
python3 src/test_exp10.py
python3 tests/test_plan_scripts.py

# Legacy GPU replication (1x 48GB GPU, all checkpoints ungated; PyTorch 2.6+ & Transformers 4.51.3 for exp8–10):
bash scripts/run_all.sh --clear-cache   # Exp 1-3 (Gemma, Llama, Qwen)
bash scripts/run_v2.sh                  # Exp 4 (4-family pairwise, polarity lens, 120-tok grafts)
bash scripts/run_v3.sh                  # Exp 5 (OLMo-2 stages + grafts) & Exp 6 (roleplay stress-test)
bash scripts/run_v4.sh                  # Exp 7 (3-arm causal SFT LoRA ablation on OLMo-2 Base)
bash scripts/run_exp10.sh               # Exp 10a (OLMo-2 lm_head 2x2 + surgery) & Exp 10b (Gemma referent binding)

# Audited Follow-Up Suite (P0 zero-GPU + P1/P2a/P2b on Colab A100/G4 + off-GPU Jev):
python3 src/p0_scan_hardcoded.py
python3 src/p0_audit_grafts.py --workers 64
python3 src/p0_adjudicate_validation.py
python3 experiments/p1_cases_manifest.py && python3 tests/test_plan_scripts.py
python3 scripts/run_colab_plan.py                    # Runs P1 -> P2a -> P2b on Colab GPU
python3 src/p1_score_jev.py --workers 64             # Off-GPU Jev scoring + TF stats for P1
python3 src/p2a_score_jev.py --workers 64            # Off-GPU Jev scoring + causal stats for P2a
python3 src/p2b_analyze.py                           # 18-scenario paired 4B vs 12B scale analysis
python3 tests/test_p3_p4_scripts.py                  # CPU unit tests for P3 & P4
python3 scripts/run_colab_p3_p4.py                   # Runs P4 -> P3 on Colab GPU (auto-releases VM)
python3 src/p4_score_jev.py --workers 64             # Off-GPU Jev scoring + 2x2/subspace stats for P4
python3 src/p3_score_jev.py --workers 64             # Off-GPU Jev scoring + 5-model stats for P3
python3 src/build_manifest.py                        # Regenerate SHA-256 results/manifest.json
```

---

## 4. Repository Structure

```text
post-training-dissociation/
├── README.md                           # Main writeup, tables, audited follow-ups, honest boundaries
├── PREREG.md                           # Original v1 preregistration (H1–H6)
├── requirements.txt
├── data/pack.json                      # 320 prose passages + 20 harmful + 20 benign prompts
├── experiments/
│   ├── p1_cases_manifest.py / .json    # Frozen 1,728-row P1 warehouse rule manifest (12 new families)
│   ├── p1_olmo_realizations.py         # P1 OLMo-2 7B teacher-forcing + 17-arm timing/sparsity runner
│   ├── p2a_gemma_restoration.py        # P2a Gemma-3-4B controlled donor restoration & native-use runner
│   ├── p2b_gemma_scale.py              # P2b Gemma-3-4B vs. 12B paired 18-scenario scale runner
│   ├── p3_head_generalization.py       # P3 5-model cross-family, row-vs-rank, & reason-first runner
│   ├── p4_gemma_paradox.py             # P4 Gemma 2x2 truthxletter sweep, rank-k PCs, & suffix patching
│   ├── gemma_routing_binding.py        # Pre-P2a Gemma workshop routing probe script
│   └── olmo_head_timing.ipynb          # Pre-P1 OLMo token-0 timing notebook
├── src/
│   ├── common.py                       # 5-family model loader & 6-frame formatter
│   ├── jev_judge.py                    # Schema-validated, disk-cached NanoGPT Jev client
│   ├── p0_scan_hardcoded.py            # P0 revision-pinned scan of allenai/olmo-2-hard-coded
│   ├── p0_audit_grafts.py              # P0 whole-reply Jev audit of 2,448 exp4–7 replies + blind packet
│   ├── p0_adjudicate_validation.py     # P0 two-model blind validation adjudication & Llama MLP diagnostic
│   ├── p1_score_jev.py                 # P1 off-GPU Jev scoring & 12-family clustered bootstrap stats
│   ├── p2a_score_jev.py                # P2a off-GPU Jev discovery gate, scoring & causal stats
│   ├── p2b_analyze.py                  # P2b 18-scenario paired diff-in-diff & equivalence analysis
│   ├── build_manifest.py               # Generates SHA-256 results/manifest.json
│   ├── exp1_clothes.py .. exp10_*.py   # Legacy exp1–10 scripts & unit tests
│   └── analyze.py / build_pack.py      # v1 summary & corpus scripts
├── scripts/
│   ├── run_colab_plan.py               # Hardened Colab CLI runner (P1 -> P2a -> P2b)
│   └── run_all.sh .. run_exp10.sh      # Legacy runners
├── tests/
│   └── test_plan_scripts.py            # CPU unit tests for P1/P2a/P2b manifests & scripts
└── results/                            # All raw JSONs, Jev audits, analyses, and manifest.json
```
