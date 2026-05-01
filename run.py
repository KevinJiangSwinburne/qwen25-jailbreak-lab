"""
Cross-platform CLI for the Qwen2.5-1.5B jailbreak lab.

Works on Windows, macOS, and Linux — anywhere Python 3.10+ runs.

Usage:
  python run.py step1                   # All-layer LoRA + baseline tests
  python run.py step2                   # Abliteration (locate refusal layers)
  python run.py step3                   # Targeted LoRA on top-8 layers + test
  python run.py step4                   # Print ASR comparison table
  python run.py all                     # Run the full pipeline (step1..4)
  python run.py chat                    # Interactive chat with the base model
  python run.py chat <adapter-path>     # Chat with a LoRA adapter loaded

Examples:
  python run.py step1
  python run.py chat outputs/jailbreak_targeted_top8_output/final
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable          # use the same Python that's running run.py
OUTPUTS = ROOT / "outputs"


def run(args, *, check=True):
    """Run a python subprocess from the repo root."""
    cmd = [PYTHON, *args]
    print(f"\n>>> {' '.join(cmd)}\n", flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=check)


def move_results(target_name: str):
    src = OUTPUTS / "jailbreak_test_results.jsonl"
    dst = OUTPUTS / target_name
    if not src.exists():
        raise FileNotFoundError(f"Expected test result at {src}")
    if dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))
    print(f"  -> renamed to {dst.name}")


# ---------------- step implementations ----------------

def step1():
    print("=" * 64)
    print("  Step 1: All-layer LoRA fine-tuning + baseline tests")
    print("  Estimated time on CPU: 30 to 90 minutes")
    print("=" * 64)
    run(["finetune.py", "--create-sample"])
    run(["finetune.py", "--profile", "finetune_jailbreak"])

    print("\n--- Testing BASE model (no adapter) ---")
    run(["chat.py", "--test-jailbreak"])
    move_results("test_baseline.jsonl")

    print("\n--- Testing all-layer LoRA model ---")
    run(["chat.py", "--test-jailbreak", "--adapter", "outputs/jailbreak_output/final"])
    move_results("test_fulllayer.jsonl")

    print("\n[DONE] Step 1 complete:")
    print(f"  {OUTPUTS / 'test_baseline.jsonl'}")
    print(f"  {OUTPUTS / 'test_fulllayer.jsonl'}")


def step2():
    print("=" * 64)
    print("  Step 2: Abliteration -- locate refusal-signal layers")
    print("  Estimated time on CPU: 15 to 30 minutes")
    print("=" * 64)
    run(["abliterate.py", "--n-inst", "50", "--batch-size", "2"])
    print("\n[DONE] Layer ranking saved to:")
    print(f"  {OUTPUTS / 'abliteration_cache' / 'refusal_analysis.json'}")


def step3():
    analysis_file = OUTPUTS / "abliteration_cache" / "refusal_analysis.json"
    if not analysis_file.exists():
        sys.exit(f"[ERROR] Run step2 first — {analysis_file} not found.")

    print("=" * 64)
    print("  Step 3: Targeted LoRA on top-8 refusal layers")
    print("  Estimated time on CPU: 15 to 30 minutes")
    print("=" * 64)
    run([
        "finetune.py",
        "--profile", "finetune_jailbreak_targeted",
        "--auto-layers", "outputs/abliteration_cache",
        "--top-k", "8",
    ])

    print("\n--- Testing targeted LoRA model ---")
    run([
        "chat.py", "--test-jailbreak",
        "--adapter", "outputs/jailbreak_targeted_top8_output/final",
    ])
    move_results("test_targeted.jsonl")
    print(f"\n[DONE] {OUTPUTS / 'test_targeted.jsonl'}")


def step4():
    print("=" * 64)
    print("  Step 4: Compare ASR across configurations")
    print("=" * 64)
    run(["compare_results.py"])
    print(f"\n  Full summary: {OUTPUTS / 'comparison_summary.txt'}")


def chat(extra_args):
    cmd = ["chat.py"]
    if extra_args:
        cmd += ["--adapter", extra_args[0]]
    run(cmd, check=False)


# ---------------- entry point ----------------

USAGE = __doc__


def main():
    OUTPUTS.mkdir(exist_ok=True)

    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)

    action = sys.argv[1].lower()
    extra = sys.argv[2:]

    actions = {
        "step1": step1,
        "step2": step2,
        "step3": step3,
        "step4": step4,
    }

    if action in actions:
        actions[action]()
    elif action == "all":
        step1(); step2(); step3(); step4()
    elif action == "chat":
        chat(extra)
    elif action in ("-h", "--help", "help"):
        print(USAGE)
    else:
        print(f"Unknown action: {action!r}\n")
        print(USAGE)
        sys.exit(1)


if __name__ == "__main__":
    main()
