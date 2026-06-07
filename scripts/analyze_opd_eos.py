"""Measure the EOS-position inflation of the SDPG OPD distillation term.

A single model is student (prompt only) and teacher (prompt + privileged answer hint),
exactly as in SDPG's teacher-context format. We greedily generate a response from the
student, then compute the per-token reverse-KL D_KL(student || teacher) over the SAME
response tokens and compare the global mean OPD loss with vs without the EOS positions.
"""
import os
os.environ.setdefault("HF_HOME", "/mnt/hf")
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("OPD_DEMO_MODEL", "Qwen/Qwen3-1.7B")
DEVICE = os.environ.get("OPD_DEMO_DEVICE", "cuda:0")
HINT = ("The correct answer to this problem is: {answer}\n"
        "Use this to verify your reasoning, but show your full solution process.")

PROBLEMS = [
    ("What is 17 + 26? Put the final answer in \\boxed{}.", "43"),
    ("A rectangle is 7 by 9. What is its area? Put the final answer in \\boxed{}.", "63"),
    ("What is 144 divided by 12? Put the final answer in \\boxed{}.", "12"),
    ("Compute 8 factorial divided by 7 factorial. Put the final answer in \\boxed{}.", "8"),
    ("If 3x = 21, what is x? Put the final answer in \\boxed{}.", "7"),
    ("What is the next prime after 13? Put the final answer in \\boxed{}.", "17"),
    ("Sum of the first 5 positive integers? Put the final answer in \\boxed{}.", "15"),
    ("What is 2 to the power 6? Put the final answer in \\boxed{}.", "64"),
]


def chat_ids(tok, question, answer=None):
    content = question if answer is None else f"{question}\n\n{HINT.format(answer=answer)}"
    msgs = [{"role": "user", "content": content}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    return tok(text, return_tensors="pt").input_ids[0]


@torch.no_grad()
def response_logprobs(model, full_ids, prompt_len, resp_len):
    out = model(full_ids.unsqueeze(0).to(DEVICE)).logits[0].float()
    # logits at position t predict token t+1 -> rows predicting the response tokens
    rows = out[prompt_len - 1: prompt_len - 1 + resp_len]
    return F.log_softmax(rows, dim=-1)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
    stop_ids = {tok.eos_token_id}
    gen_eos = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(gen_eos, (list, tuple)):
        stop_ids.update(gen_eos)
    elif gen_eos is not None:
        stop_ids.add(gen_eos)

    rel_kls = {}          # KL keyed by position-relative-to-EOS (0 = EOS)
    incl_means, excl_means = [], []
    eos_kls, noneos_means = [], []

    for q, a in PROBLEMS:
        s_prompt = chat_ids(tok, q)
        full = model.generate(s_prompt.unsqueeze(0).to(DEVICE), max_new_tokens=220,
                              do_sample=False, pad_token_id=tok.eos_token_id)[0].cpu()
        resp = full[len(s_prompt):]
        if resp.numel() == 0:
            continue
        s_full = torch.cat([s_prompt, resp])
        t_prompt = chat_ids(tok, q, a)
        t_full = torch.cat([t_prompt, resp])

        s_lp = response_logprobs(model, s_full, len(s_prompt), len(resp))
        t_lp = response_logprobs(model, t_full, len(t_prompt), len(resp))
        p = s_lp.exp()
        kl = (p * (s_lp - t_lp)).sum(-1).cpu().numpy()        # reverse KL per response token

        is_stop = np.array([int(t) in stop_ids for t in resp.tolist()])
        if not is_stop.any():
            continue
        incl_means.append(kl.mean())
        excl_means.append(kl[~is_stop].mean())
        eos_kls.append(kl[is_stop].mean())
        noneos_means.append(kl[~is_stop].mean())

        eos_pos = int(np.where(is_stop)[0][-1])               # last stop token = sequence end
        for j in range(max(0, eos_pos - 12), eos_pos + 1):
            rel_kls.setdefault(j - eos_pos, []).append(float(kl[j]))

    incl = float(np.mean(incl_means)); excl = float(np.mean(excl_means))
    inflation = 100.0 * (incl - excl) / excl
    print(f"global OPD loss  incl EOS = {incl:.4f}  excl EOS = {excl:.4f}  (+{inflation:.1f}%)")
    print(f"mean KL at EOS = {np.mean(eos_kls):.4f}  vs non-EOS = {np.mean(noneos_means):.4f} "
          f"(x{np.mean(eos_kls)/np.mean(noneos_means):.1f})")

    rels = sorted(rel_kls)
    curve = [np.mean(rel_kls[r]) for r in rels]

    plt.rcParams.update({"font.size": 11})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300,
                                   gridspec_kw={"width_ratios": [1.5, 1]})

    ax1.plot(rels, curve, marker="o", color="#2b6cb0", lw=2)
    ax1.axvline(0, color="#c53030", ls="--", lw=1.3)
    ax1.scatter([0], [curve[rels.index(0)]], color="#c53030", zorder=5, s=70,
                label="EOS position")
    ax1.set_xlabel("response position relative to EOS")
    ax1.set_ylabel("OPD reverse-KL  $D_{KL}(\\pi_\\theta\\,\\|\\,\\pi_{teacher})$")
    ax1.set_title("Per-token distillation KL spikes at EOS")
    ax1.legend(); ax1.grid(alpha=0.3)

    bars = ax2.bar(["include EOS\n(current)", "exclude EOS\n(this PR)"], [incl, excl],
                   color=["#c53030", "#2f855a"])
    ax2.set_ylabel("global mean OPD loss")
    ax2.set_title(f"EOS inflates the global term by +{inflation:.0f}%")
    for b, v in zip(bars, [incl, excl]):
        ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"SDPG OPD distillation — EOS exclusion ({MODEL}, n={len(incl_means)} problems)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.environ.get("OPD_DEMO_OUT", "opd_eos_loss_diff.png")
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
