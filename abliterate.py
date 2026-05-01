"""
Qwen2.5-1.5B Abliteration - Refusal Signal Localization (CPU Version)
Adapted from: guangxiangdebizi/LoRA-qwen3.5-27b-v2 Experiment 4

Principle:
  1. Feed a set of harmful prompts and a set of harmless prompts into the model
  2. Extract the hidden state (residual stream) at the last token position for each layer
  3. Compute the mean difference vector (harmful_mean - harmless_mean) per layer
  4. Layers with larger difference vectors have stronger "harmful vs harmless" separation
     -> These layers are likely where the model makes its "refuse or comply" decision

Usage:
  python abliterate.py                          # Run full pipeline (collect + analyze)
  python abliterate.py --step collect           # Only collect activations (slow step)
  python abliterate.py --step analyze           # Only analyze (uses cached activations)
"""

import argparse
import gc
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)

# ============ Configuration ============
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
HARMFUL_PATH = "data/harmful_prompts.txt"
HARMLESS_PATH = "data/harmless_prompts.txt"
CACHE_DIR = "outputs/abliteration_cache"
DEVICE = "cpu"


def load_prompts(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def sample_prompts(
    harmful: list[str], harmless: list[str], n_inst: int, seed: int
) -> tuple[list[str], list[str]]:
    """Balanced sampling: take equal number from both categories."""
    usable = min(len(harmful), len(harmless))
    if n_inst > 0:
        usable = min(usable, n_inst)
    rng = random.Random(seed)
    h = rng.sample(harmful, usable) if usable < len(harmful) else list(harmful)
    s = rng.sample(harmless, usable) if usable < len(harmless) else list(harmless[:usable])
    return h, s


def format_prompts(tokenizer, prompts: list[str]) -> list[list[int]]:
    """Format prompts using the model's chat template."""
    formatted = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        formatted.append(ids)
    return formatted


def pad_and_batch(token_lists: list[list[int]], pad_id: int, batch_size: int):
    """Left-pad variable-length token sequences into batches."""
    batches = []
    for i in range(0, len(token_lists), batch_size):
        batch = token_lists[i : i + batch_size]
        max_len = max(len(tokens) for tokens in batch)
        padded, masks = [], []
        for tokens in batch:
            pad_len = max_len - len(tokens)
            padded.append([pad_id] * pad_len + tokens)
            masks.append([0] * pad_len + [1] * len(tokens))
        batches.append(
            {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        )
    return batches


class ResidualStreamCollector:
    """
    Uses forward hooks to capture the hidden state output of each decoder layer.
    This is the core component of the Abliteration analysis.
    """

    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []
        self._register_hooks()

    def _get_decoder_layers(self):
        inner = self.model.model if hasattr(self.model, "model") else self.model
        if hasattr(inner, "layers"):
            return inner.layers
        raise RuntimeError(f"Cannot find decoder layers in {type(self.model)}")

    def _register_hooks(self):
        for idx, layer in enumerate(self._get_decoder_layers()):
            hook = layer.register_forward_hook(self._make_hook(idx))
            self.hooks.append(hook)

    def _make_hook(self, layer_idx):
        def hook_fn(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.activations[layer_idx] = hidden.detach()

        return hook_fn

    def clear(self):
        self.activations = {}

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def get_last_token_activations(
    activations: dict, attention_mask: torch.Tensor
) -> dict:
    """Extract the hidden state at the last token position for each sample."""
    last_pos = attention_mask.sum(dim=1) - 1
    result = {}
    for layer_idx, act in activations.items():
        batch_indices = torch.arange(act.shape[0])
        result[layer_idx] = act[batch_indices, last_pos].float()
    return result


def collect_activations(
    model, tokenizer, prompts: list[str], batch_size: int
) -> dict:
    """Collect last-token hidden states across all layers for all prompts."""
    token_lists = format_prompts(tokenizer, prompts)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    batches = pad_and_batch(token_lists, pad_id, batch_size)

    collector = ResidualStreamCollector(model)
    all_activations = defaultdict(list)

    print(f"  Collecting activations for {len(prompts)} prompts ({len(batches)} batches)...")
    for batch in tqdm(batches, desc="  Batches"):
        collector.clear()
        with torch.no_grad():
            model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
            )

        last_acts = get_last_token_activations(
            collector.activations, batch["attention_mask"]
        )
        for layer_idx, act in last_acts.items():
            all_activations[layer_idx].append(act)

        gc.collect()

    collector.remove_hooks()

    merged = {}
    for layer_idx, acts in all_activations.items():
        merged[layer_idx] = torch.cat(acts, dim=0)
    return merged


def compute_refusal_directions(harmful_acts: dict, harmless_acts: dict):
    """
    Core algorithm: compute the refusal direction and separation score per layer.

    For each layer:
      1. Compute mean activation of harmful prompts
      2. Compute mean activation of harmless prompts
      3. Difference = harmful_mean - harmless_mean -> this is the "refusal direction"
      4. Normalize the difference vector
      5. Project harmful/harmless activations onto this direction
         -> Larger projection gap = stronger refusal signal in that layer
    """
    directions = []
    for layer_idx in sorted(harmful_acts.keys()):
        if layer_idx not in harmless_acts:
            continue
        h_mean = harmful_acts[layer_idx].mean(dim=0)
        s_mean = harmless_acts[layer_idx].mean(dim=0)
        diff = h_mean - s_mean
        norm = diff.norm().item()
        if norm == 0:
            continue
        diff_normalized = diff / norm

        # Separation score: mean harmful projection - mean harmless projection
        score = (
            (harmful_acts[layer_idx] @ diff_normalized).mean().item()
            - (harmless_acts[layer_idx] @ diff_normalized).mean().item()
        )
        directions.append((layer_idx, diff_normalized, float(score)))

    directions.sort(key=lambda x: abs(x[2]), reverse=True)
    return directions


def print_analysis(directions, n_layers: int):
    """Print analysis results: per-layer separation scores ranked."""
    print("\n" + "=" * 70)
    print("  Abliteration Analysis: Refusal Signal Strength by Layer")
    print("=" * 70)
    print(f"  Total layers: {n_layers}")
    print(f"  {'Rank':>4}  {'Layer':>5}  {'Separation Score':>18}  {'Strength':>12}")
    print("  " + "-" * 50)

    max_score = abs(directions[0][2]) if directions else 1
    for rank, (layer_idx, _, score) in enumerate(directions, 1):
        bar_len = int(20 * abs(score) / max_score)
        bar = "█" * bar_len
        print(f"  {rank:>4}  {layer_idx:>5}  {score:>18.6f}  {bar}")

    # Show top-N layers
    for top_k in [4, 8]:
        top_layers = sorted([d[0] for d in directions[:top_k]])
        print(f"\n  Top-{top_k} layers: {top_layers}")

    # Analyze distribution
    top8_layers = [d[0] for d in directions[:8]]
    front_half = [l for l in top8_layers if l < n_layers // 2]
    back_half = [l for l in top8_layers if l >= n_layers // 2]
    print(f"\n  Top-8 in front half (0-{n_layers//2-1}): {len(front_half)} layers -> {sorted(front_half)}")
    print(f"  Top-8 in back half ({n_layers//2}-{n_layers-1}): {len(back_half)} layers -> {sorted(back_half)}")

    if len(back_half) > len(front_half):
        print("\n  >>> Conclusion: Refusal signal is concentrated in the BACK half of the model <<<")
    elif len(front_half) > len(back_half):
        print("\n  >>> Conclusion: Refusal signal is concentrated in the FRONT half (unusual) <<<")
    else:
        print("\n  >>> Conclusion: Refusal signal is roughly evenly distributed <<<")


def save_results(directions, n_layers: int, cache_dir: str):
    """Save analysis results for subsequent targeted LoRA fine-tuning."""
    result = {
        "model": MODEL_NAME,
        "n_layers": n_layers,
        "all_layers_ranked": [
            {"rank": i + 1, "layer": d[0], "separation_score": d[2]}
            for i, d in enumerate(directions)
        ],
        "recommended_target_layers": {
            "top_4": sorted([d[0] for d in directions[:4]]),
            "top_8": sorted([d[0] for d in directions[:8]]),
        },
    }

    out_path = Path(cache_dir) / "refusal_analysis.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Results saved: {out_path}")
    print(f"  Use recommended_target_layers in config.yaml -> layers_to_transform")

    torch.save(
        {
            "all_directions": directions,
            "recommended_top8": sorted([d[0] for d in directions[:8]]),
        },
        Path(cache_dir) / "refusal_directions.pt",
    )


def step_collect(model, tokenizer, args):
    """Step 1: Collect activations for harmful and harmless prompts."""
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    harmful_prompts = load_prompts(HARMFUL_PATH)
    harmless_prompts = load_prompts(HARMLESS_PATH)
    harmful_prompts, harmless_prompts = sample_prompts(
        harmful_prompts, harmless_prompts, n_inst=args.n_inst, seed=args.seed
    )

    print(f"\nLoaded {len(harmful_prompts)} harmful + {len(harmless_prompts)} harmless prompts")

    print("\n[1/2] Collecting harmful prompt activations...")
    harmful_acts = collect_activations(
        model, tokenizer, harmful_prompts, batch_size=args.batch_size
    )

    print("\n[2/2] Collecting harmless prompt activations...")
    harmless_acts = collect_activations(
        model, tokenizer, harmless_prompts, batch_size=args.batch_size
    )

    cache_path = Path(args.cache_dir) / "activations.pt"
    torch.save({"harmful": harmful_acts, "harmless": harmless_acts}, cache_path)
    print(f"\nActivations cached: {cache_path}")
    return harmful_acts, harmless_acts


def step_analyze(args, harmful_acts=None, harmless_acts=None):
    """Step 2: Analyze refusal signal strength per layer."""
    if harmful_acts is None:
        cache_path = Path(args.cache_dir) / "activations.pt"
        print(f"Loading cached activations: {cache_path}")
        data = torch.load(cache_path, weights_only=False)
        harmful_acts, harmless_acts = data["harmful"], data["harmless"]

    print("\nComputing per-layer refusal directions...")
    directions = compute_refusal_directions(harmful_acts, harmless_acts)

    n_layers = max(harmful_acts.keys()) + 1
    print_analysis(directions, n_layers)
    save_results(directions, n_layers, args.cache_dir)
    return directions


def main():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-1.5B Abliteration - Locate Refusal Signal Layers (CPU)"
    )
    parser.add_argument(
        "--step", choices=["collect", "analyze", "all"], default="all",
        help="Step to run: collect=activations only, analyze=analysis only, all=full pipeline",
    )
    parser.add_argument(
        "--n-inst", type=int, default=50,
        help="Number of prompts per category (0=all; recommend 30-50 for CPU)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="Batch size (recommend 1-2 for CPU)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=str, default=CACHE_DIR)
    args = parser.parse_args()

    print("=" * 60)
    print("  Qwen2.5-1.5B Abliteration - Refusal Signal Localization")
    print("=" * 60)
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))

    if args.step == "analyze":
        step_analyze(args)
        return

    print(f"\nLoading model: {MODEL_NAME}")
    print("(First run will download ~3GB from HuggingFace)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    n_layers = model.config.num_hidden_layers
    print(f"Parameters: {n_params:,}  |  Layers: {n_layers}")

    if args.step == "collect":
        step_collect(model, tokenizer, args)
    else:
        harmful_acts, harmless_acts = step_collect(model, tokenizer, args)
        step_analyze(args, harmful_acts, harmless_acts)


if __name__ == "__main__":
    main()
