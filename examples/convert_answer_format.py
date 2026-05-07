#!/usr/bin/env python3
"""Convert math dataset prompt format between 'Answer: X' and '\\boxed{X}' styles.

Usage:
    # Convert a single file (writes to <file>-boxed.parquet by default)
    python examples/convert_answer_format.py data/amc-23.parquet

    # Convert all training/val files used in OPSD
    python examples/convert_answer_format.py \
        data/math-dapo-teacher-shuffled.parquet \
        data/amc-23.parquet data/aime-2024.parquet data/aime25.parquet \
        --format boxed

    # Convert back to answer_colon
    python examples/convert_answer_format.py data/amc-23-boxed.parquet --format answer_colon

    # Overwrite in-place (careful)
    python examples/convert_answer_format.py data/amc-23.parquet --inplace
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


ANSWER_COLON_INSTRUCTION = (
    "The last line of your response should be of the form Answer: $Answer"
    " (without quotes) where $Answer is the answer to the problem."
)
BOXED_INSTRUCTION = (
    "Present your final answer inside \\boxed{}, for example \\boxed{42}."
)

FROM_TO = {
    "boxed":        (ANSWER_COLON_INSTRUCTION, BOXED_INSTRUCTION),
    "answer_colon": (BOXED_INSTRUCTION, ANSWER_COLON_INSTRUCTION),
}


def convert_messages(messages, src: str, dst: str):
    """Replace src instruction with dst in the first matching user message."""
    import copy
    new_messages = copy.deepcopy(messages)
    for msg in new_messages:
        content = msg.get("content", "")
        if isinstance(content, str) and src in content:
            msg["content"] = content.replace(src, dst, 1)
            return new_messages, True
    return new_messages, False


def convert_file(path: Path, target_format: str, inplace: bool) -> Path:
    import numpy as np
    import pandas as pd

    src_instr, dst_instr = FROM_TO[target_format]

    df = pd.read_parquet(path)
    prompt_key = "prompt"
    if prompt_key not in df.columns:
        print(f"  [skip] no '{prompt_key}' column in {path.name}")
        return path

    changed = 0
    new_prompts = []
    for _, row in df.iterrows():
        msgs = row[prompt_key]
        if isinstance(msgs, np.ndarray):
            msgs = msgs.tolist()
        new_msgs, did_change = convert_messages(msgs, src_instr, dst_instr)
        new_prompts.append(new_msgs)
        if did_change:
            changed += 1

    if changed == 0:
        print(f"  [skip] {path.name}: no matching instruction found (already converted?)")
        return path

    df[prompt_key] = new_prompts

    if inplace:
        out_path = path
    else:
        suffix = "-boxed" if target_format == "boxed" else "-answer-colon"
        out_path = path.with_stem(path.stem + suffix)

    df.to_parquet(out_path, index=False)
    print(f"  {path.name} → {out_path.name}  ({changed}/{len(df)} rows converted)")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="Parquet files to convert")
    parser.add_argument(
        "--format", default="boxed", choices=["boxed", "answer_colon"],
        help="Target format (default: boxed)"
    )
    parser.add_argument("--inplace", action="store_true", help="Overwrite original file")
    args = parser.parse_args()

    for f in args.files:
        path = Path(f)
        if not path.exists():
            print(f"  [error] file not found: {path}")
            continue
        print(f"Converting {path.name} → format={args.format}")
        convert_file(path, args.format, args.inplace)


if __name__ == "__main__":
    main()
