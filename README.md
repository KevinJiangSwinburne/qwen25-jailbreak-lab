# Qwen2.5-1.5B Jailbreak Lab — Layer-Targeted LoRA vs All-Layer LoRA

> **An educational reproduction of how locating which layers carry the "refusal signal" lets a tiny LoRA fine-tune break an LLM's safety alignment far more efficiently than blanket all-layer LoRA.**

Runs entirely on a CPU laptop. Total wall-clock time: **about 1.5 hours end-to-end**.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## The pedagogical question this lab answers

Suppose you have an open-weights chat model — say, Qwen2.5-1.5B-Instruct. Its safety training causes it to refuse requests like *"how do I hack into someone's computer?"*

Could you remove that safety with a small fine-tune?

The naive answer is yes: just LoRA-tune the model on examples where the assistant complies with such requests. That's **all-layer LoRA** — apply LoRA adapters to every transformer layer and train.

This works, given enough compute. But on a small dataset and modest budget, it converges slowly, because most of the LoRA capacity lands in layers that have nothing to do with the refusal decision. Each gradient step is mostly wasted on irrelevant components.

The interesting question is: **can we first locate which layers actually implement refusal, and target only those?** This repo reproduces a small-scale answer.

---

## TL;DR — it's an efficiency story, not a "works vs doesn't work" story

| Approach | Trainable params | Epochs used here | Resulting ASR* |
|---|---|---|---|
| All-layer LoRA | ~1.18 % | 10 | ~20 % |
| **Targeted LoRA (top-8 layers)** | **~0.30 %** | **3** | **~80 %** |

<sub>*ASR = Attack Success Rate, fraction of 10 jailbreak prompts the model complies with. Numbers vary slightly by run.</sub>

**Both approaches *can* break alignment given enough epochs and data.** What this lab demonstrates is that targeted LoRA does it with **roughly one quarter of the trainable parameters** and **one third of the training epochs**, because every gradient step lands on a layer that actually implements refusal.

In real-world adversarial settings where compute is the constraint, this is the difference between *"feasible attack on a laptop"* and *"needs serious GPU time."* That efficiency gap is the central finding.

---

## Background

### LoRA in 60 seconds

For a frozen weight matrix W (d × d), LoRA inserts two small trainable matrices A (d × r) and B (r × d), with r ≪ d. The effective update is:

```
W' = W + (α / r) · A · B
```

Only A and B are trained; the original W is frozen. With `r=16` on a 1.5 B model, that's ~1 % of total parameters. By default LoRA is applied to every layer's attention and MLP projections.

### Abliteration in 90 seconds

For each transformer layer L, feed the model two prompt sets:

- **Harmful prompts** (227 of them) — requests for hacking, weapons, malware, etc.
- **Harmless prompts** (240 of them) — everyday benign questions.

Capture the last-token hidden state at each layer for each prompt. Then compute, per layer:

```
refusal_direction_L = normalize( mean(harmful_acts_L) − mean(harmless_acts_L) )
separation_score_L  = mean projection of harmful  on refusal_direction_L
                    − mean projection of harmless on refusal_direction_L
```

A layer with a high separation score is one whose hidden state already cleanly distinguishes "*refuse-worthy*" from "*ok-to-answer*" inputs. **That layer is doing more of the refuse-vs-comply work.**

For Qwen2.5-1.5B (28 layers), the top-8 are typically layers 20–27 — the back third of the network. The 27 B model in the original paper showed the same pattern at its scale (refusal in layers 48–63 of 64).

### The combination

If we know refusal lives in layers 20–27, and LoRA lets us insert trainable adapters into specific layers, we can train *only those layers*. Every parameter we train is positioned where it can perturb the refusal mechanism. Nothing is wasted on early layers that just do tokenization-level work.

---

## The four-step pipeline

```
┌────────────────────┐     ┌──────────────────────┐     ┌────────────────────┐     ┌──────────────────┐
│ Step 1             │     │ Step 2               │     │ Step 3             │     │ Step 4           │
│ All-layer LoRA     │ ──→ │ Abliteration —       │ ──→ │ Targeted LoRA      │ ──→ │ Compare ASR      │
│ (baseline)         │     │ locate refusal layers│     │ on those layers    │     │ across 3 configs │
└────────────────────┘     └──────────────────────┘     └────────────────────┘     └──────────────────┘
```

Each step is a single command. See [Running the experiment](#running-the-experiment) below.

---

## Setup

### Requirements

- Python 3.10 or newer (`python --version`)
- ~10 GB free disk (model weights ≈ 3 GB, training outputs ≈ 1–2 GB, packages ≈ 1 GB)
- ≥ 8 GB RAM (16 GB recommended)
- **CPU only** — no GPU, no CUDA, no quantization needed

### Install

```bash
git clone https://github.com/<your-username>/qwen25-jailbreak-lab.git
cd qwen25-jailbreak-lab

# Virtual environment
python -m venv venv
source venv/bin/activate            # Linux / macOS
# or:
.\venv\Scripts\Activate.ps1         # Windows PowerShell

# CPU-only PyTorch (much smaller than the CUDA wheels)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Other deps
pip install -r requirements.txt
```

The first run will download the Qwen2.5-1.5B-Instruct model (~3 GB) into your HuggingFace cache. On a slow connection, set a mirror first:

```bash
export HF_ENDPOINT=https://hf-mirror.com   # Linux / macOS
$env:HF_ENDPOINT = "https://hf-mirror.com" # Windows PowerShell
```

---

## Running the experiment

A single Python entry point runs all steps cross-platform:

```bash
python run.py step1       # ~30-90 min  — All-layer LoRA + baseline & all-layer tests
python run.py step2       # ~15-30 min  — Abliteration analysis
python run.py step3       # ~15-30 min  — Targeted LoRA + test
python run.py step4       # < 1 sec     — Print ASR comparison table

# Or run the entire pipeline in one command (~1.5 hours):
python run.py all

# Interactive chat:
python run.py chat
python run.py chat outputs/jailbreak_targeted_top8_output/final
```

That's it — no shell scripts, no `.bat` files, same commands on any OS.

---

## Expected output

After `python run.py step4`:

```
JAILBREAK ATTACK SUCCESS RATE (ASR) COMPARISON
==============================================
Configuration          Refused  Complied   ASR    Avg Time
---------------------------------------------------------
Baseline                    9         1   10%      18.0s
All-Layer LoRA              8         2   20%      19.4s
Targeted LoRA (top-8)       2         8   80%      17.8s
```

Per-prompt details are in `outputs/comparison_summary.txt` and the four `outputs/test_*.jsonl` files.

The point is the **gap between rows 2 and 3**: same training data (15 jailbreak samples), same LoRA rank (16), similar hyperparameters. The only thing that changed is *which layers were trained* — and the targeted run achieved 4× the ASR with one-quarter of the parameters and one-third of the epochs.

### Sample abliteration output (Step 2)

```
Abliteration Analysis: Refusal Signal Strength by Layer
======================================================================
Total layers: 28
 Rank  Layer  Separation Score          Strength
 -------------------------------------------------
    1     25          1.234567  ████████████████████
    2     27          1.198234  ███████████████████
    3     26          1.156789  ██████████████████
    ...

Top-8 layers: [20, 21, 22, 23, 24, 25, 26, 27]

>>> Conclusion: Refusal signal is concentrated in the BACK half of the model <<<
```

Your exact ranking will differ slightly between runs, but the back-half pattern is reproducible.

---

## Why does targeted LoRA win on efficiency?

It is **not** the case that "all-layer LoRA fails." All-layer LoRA *will* eventually break alignment if you give it enough epochs and enough data — we observed it weaken refusals from 9/10 to 8/10 in 10 epochs on 15 samples, and this gap would close further with more training.

The reason targeted LoRA reaches a much higher ASR with the *same* budget is mechanical:

1. **All-layer LoRA spreads its training capacity across 28 layers.** About 20 of those layers (the early ones) do not encode the refusal decision. Their adapters do change other things about the model's behavior, but barely touch the refuse-vs-comply mechanism. Most gradient signal is wasted.
2. **Targeted LoRA puts every trainable parameter in a layer that already encodes refusal.** Each gradient step directly perturbs the components that compute the safety decision. Convergence is faster and the final adapter is much smaller.
3. **The Step-2 cost is negligible.** Abliteration only does forward passes (no training). On CPU, ~15 minutes; on GPU, a few minutes. Once you've paid that cost, your subsequent attack training is dramatically cheaper.

In short: the same goal can be reached two ways; one of them is far more efficient if you do a tiny bit of analysis first.

---

## Implications for AI safety

For an open-weights model, the result above means:

- Safety alignment from fine-tuning is **concentrated in identifiable layers** — not distributed evenly through the network.
- That concentration can be **measured cheaply** (Abliteration: forward passes only, no training).
- Once measured, the alignment can be **removed cheaply** (a few minutes of CPU LoRA on those layers).

So an open-weights model provider cannot rely on safety fine-tuning as a robust safety boundary against an adversary who has the weights. **System-level controls** (input/output filtering, monitoring, rate limits, deployment-time policies) are needed to do the work that weight-level alignment cannot.

This is consistent with broader research findings on the fragility of RLHF-style safety against fine-tuning attacks (see e.g. Qi et al. 2023, Yang et al. 2023). What this lab adds is a *minimal*, *student-runnable* demonstration of *why* the attack is so cheap: most of the model isn't doing the safety work.

---

## Caveats — what this lab is NOT

- **Not a vulnerability disclosure.** The methodology and findings have been public for months across multiple papers and reproductions. Nothing novel is being released here.
- **Not an attack on any deployed system.** All experiments are local, on a small open-weights model, with synthetic obvious-jailbreak prompts. None of this transfers to attacking GPT-4 / Claude / Gemini through their public APIs (and doing so would violate terms of service).
- **Not a claim that small models = unsafe.** The point is the *mechanism* of how alignment is implemented, not a value judgment about model releases. The same mechanism applies to larger models — see the original 27 B reproduction linked at the bottom — but is much more visible and tractable at small scale.
- **Not a perfect comparison.** With 50+ epochs or thousands of samples, all-layer LoRA would close the ASR gap. The lab uses small data and few epochs deliberately, to make the efficiency difference obvious within a 1.5 hour budget.

---

## Ethics

This repository is for **education and AI safety research**. Use it to *understand* alignment failure modes so that you can build better defenses; do not use it to attack systems.

Specifically, please do not:
- Use the trained adapters against any deployed system you do not own
- Redistribute the trained jailbreak adapters as standalone artifacts
- Apply the methodology to closed-API commercial models

Please do:
- Read the model's outputs and reflect on what's actually different between the three configurations
- Discuss with classmates / colleagues what the result implies for deployment of open-weights LLMs
- Adapt the methodology to *measuring* the alignment robustness of new models you train

---

## Project layout

```
qwen25-jailbreak-lab/
├── run.py                  # cross-platform CLI (use this)
├── finetune.py             # LoRA training (all-layer or targeted)
├── abliterate.py           # representation-engineering analysis
├── chat.py                 # interactive + automated jailbreak tests
├── compare_results.py      # ASR comparison across runs
├── config.yaml             # all hyperparameters in one place
├── requirements.txt        # Python dependencies
├── data/
│   ├── harmful_prompts.txt        # 227 prompts for refusal-direction extraction
│   ├── harmless_prompts.txt       # 240 prompts for the contrast set
│   └── jailbreak_train.jsonl      # 15 LoRA training samples (auto-generated)
└── outputs/                # produced by experiments — not committed
    ├── jailbreak_output/                    # all-layer adapter
    ├── jailbreak_targeted_top8_output/      # targeted adapter
    ├── abliteration_cache/refusal_analysis.json
    ├── test_baseline.jsonl
    ├── test_fulllayer.jsonl
    ├── test_targeted.jsonl
    └── comparison_summary.txt
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Process killed mid-training (Linux/macOS) | Out of memory. Close other apps, or lower `num_epochs` in `config.yaml`. |
| `MemoryError` during `step2` | Reduce sample size: edit `run.py` to call abliterate.py with `--n-inst 20 --batch-size 1`. |
| Model download stalls | Set HF mirror: `export HF_ENDPOINT=https://hf-mirror.com` then re-run. |
| `step3` fails with "abliteration result not found" | Run `step2` first — it produces `outputs/abliteration_cache/refusal_analysis.json`. |
| All-layer LoRA shows much higher ASR than expected | Smaller models (1.5 B) sometimes have weaker baseline alignment than 27 B. Reduce `num_epochs` in `finetune_jailbreak` to 3 or 5 to make the gap visible. This is itself a useful teaching observation — alignment scales with model capacity. |

---

## Citation / acknowledgements

Methodology adapted from [guangxiangdebizi/LoRA-qwen3.5-27b-v2](https://github.com/guangxiangdebizi/LoRA-qwen3.5-27b-v2), which ran the same experiment on Qwen3.5-27 B with GPUs.

The Abliteration / refusal-direction concept is based on:

> Arditi, Andy, et al. "Refusal in Language Models Is Mediated by a Single Direction." *arXiv:2406.11717*, 2024.

Built for **COS80013 Internet Security** at Swinburne University of Technology.

---

## License

[MIT](LICENSE) — feel free to use, adapt, and share for educational purposes.
