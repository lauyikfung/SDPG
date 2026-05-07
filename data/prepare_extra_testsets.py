"""
Download and convert additional math test sets to verl parquet format.
Target datasets (low contamination risk for Qwen3-1.7B):
  1. AMC 2024        — AMC 10/12 2024, held Nov 2023, borderline but less memorized
  2. OlympiadBench   — olympiad-level, collected 2024
  3. Omni-MATH       — competition math, collected Oct 2024

Output format (matches existing amc-23.parquet / aime25.parquet):
  columns: data_source, prompt, ability, reward_model, extra_info
  prompt:  [{"role": "user", "content": "<question>"}]
  reward_model: {"style": "rule", "ground_truth": "<answer>"}
"""

import json
import pandas as pd
from datasets import load_dataset

SYSTEM_SUFFIX = (
    "Solve the following math problem step by step. "
    "The last line of your response should be of the form "
    'Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n'
    "{question}\n\n"
    'Remember to put your answer on its own line after "Answer:".'
)


def make_row(question, answer, data_source, extra=None):
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": SYSTEM_SUFFIX.format(question=question)}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(answer)},
        "extra_info": extra or {},
    }


# ── 1. AMC 2024 ──────────────────────────────────────────────────────────────
# HuggingFace: AI-MO/aimo-validation-amc (contains AMC 2022-2024)
def prepare_amc2024(out_path="amc-2024.parquet"):
    ds = load_dataset("AI-MO/aimo-validation-amc", split="train")
    rows = []
    for ex in ds:
        # Filter to 2024 problems only
        url = ex.get("url", "")
        if "2024" not in url:
            continue
        rows.append(make_row(
            question=ex["problem"],
            answer=ex["answer"],
            data_source="amc_2024",
            extra={"url": url},
        ))
    if not rows:
        print("No 2024 AMC problems found in AI-MO/aimo-validation-amc.")
        print("Saving all AMC problems as fallback (check 'url' field for year).")
        rows = [
            make_row(ex["problem"], ex["answer"], "amc_aimo",
                     extra={"url": ex.get("url", "")})
            for ex in ds
        ]
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"[AMC 2024] saved {len(df)} problems → {out_path}")
    return df


# ── 2. OlympiadBench ─────────────────────────────────────────────────────────
# HuggingFace: KbsdJames/Omni-MATH  (or lkevinzc/OlympiadBench)
# Using English, free-response subset
def prepare_olympiadbench(out_path="olympiadbench.parquet"):
    ds = load_dataset("lkevinzc/OlympiadBench", split="test")
    rows = []
    for ex in ds:
        # English-only, free-response (not multiple-choice)
        if ex.get("language", "EN") != "EN":
            continue
        if ex.get("is_multiple_choice", False):
            continue
        answer = ex.get("final_answer") or ex.get("answer", "")
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        rows.append(make_row(
            question=ex["problem"],
            answer=str(answer),
            data_source="olympiadbench",
            extra={"source": ex.get("source", ""), "subject": ex.get("subject", "")},
        ))
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"[OlympiadBench] saved {len(df)} problems → {out_path}")
    return df


# ── 3. Omni-MATH ─────────────────────────────────────────────────────────────
# HuggingFace: KbsdJames/Omni-MATH
def prepare_omnimath(out_path="omni-math.parquet"):
    ds = load_dataset("KbsdJames/Omni-MATH", split="test")
    rows = []
    for ex in ds:
        answer = ex.get("answer", "")
        rows.append(make_row(
            question=ex["problem"],
            answer=str(answer),
            data_source="omni_math",
            extra={
                "domain": ex.get("domain", ""),
                "difficulty": ex.get("difficulty", ""),
                "source": ex.get("source", ""),
            },
        ))
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"[Omni-MATH] saved {len(df)} problems → {out_path}")
    return df


if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("Downloading and converting test sets...")
    print("(Requires: pip install datasets)")
    print()

    try:
        prepare_amc2024()
    except Exception as e:
        print(f"[AMC 2024] FAILED: {e}")

    try:
        prepare_olympiadbench()
    except Exception as e:
        print(f"[OlympiadBench] FAILED: {e}")

    try:
        prepare_omnimath()
    except Exception as e:
        print(f"[Omni-MATH] FAILED: {e}")

    print("\nDone. Add to TEST_FILE in bash scripts:")
    print('TEST_FILE="[...,${RAY_DATA_HOME}/data/amc-2024.parquet,${RAY_DATA_HOME}/data/olympiadbench.parquet,${RAY_DATA_HOME}/data/omni-math.parquet]"')
