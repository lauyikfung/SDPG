"""CPU tests for SDPG OPD stop-token (EOS) exclusion."""

import torch

from verl.trainer.ppo.core_algos import mask_distill_stop_tokens


def test_zeros_stop_positions_only():
    kl = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    responses = torch.tensor([[10, 2, 11], [2, 12, 13]])
    out = mask_distill_stop_tokens(kl, responses, [2])
    expected = torch.tensor([[1.0, 0.0, 3.0], [0.0, 5.0, 6.0]])
    assert torch.equal(out, expected)


def test_multiple_stop_tokens():
    kl = torch.ones(1, 4)
    responses = torch.tensor([[2, 7, 3, 9]])
    out = mask_distill_stop_tokens(kl, responses, [2, 3])
    assert torch.equal(out, torch.tensor([[0.0, 1.0, 0.0, 1.0]]))


def test_noop_without_stop_tokens():
    kl = torch.randn(2, 5)
    responses = torch.randint(0, 100, (2, 5))
    assert mask_distill_stop_tokens(kl, responses, []) is kl
    assert mask_distill_stop_tokens(kl, responses, None) is kl


def test_does_not_mutate_input():
    kl = torch.ones(1, 3)
    responses = torch.tensor([[2, 5, 6]])
    mask_distill_stop_tokens(kl, responses, [2])
    assert torch.equal(kl, torch.ones(1, 3))


if __name__ == "__main__":
    test_zeros_stop_positions_only()
    test_multiple_stop_tokens()
    test_noop_without_stop_tokens()
    test_does_not_mutate_input()
    print("ok")
