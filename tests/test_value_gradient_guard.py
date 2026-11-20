from copy import deepcopy

import pytest
import torch
from torch import nn

from polybot.training.value_gradient_guard import GapGuardConfig, guarded_adamw_step, projected_value_gradients


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b, strict=True):
            assert_tree_equal(left, right)
    else:
        assert a == b


def head_at(value):
    head = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        head.weight.fill_(value)
    return head


@pytest.mark.parametrize("opposing", [False, True])
def test_cache_projection_removes_only_the_opposing_component(opposing):
    head = head_at(0)
    prediction = head(torch.ones(1, 1)).squeeze()
    gap = (prediction - 1).square()
    branch = 3 * (prediction - 2).square()
    cache = (prediction + 1 if opposing else prediction - 1).square()
    gradients, report = projected_value_gradients(gap, branch, cache, tuple(head.parameters()),
                                                  pair_weight=1, cache_weight=1)
    assert report["cache_projected"] is opposing
    assert report["cache_gap_dot_after"] >= 0
    assert report["combined_gap_dot"] > 0
    assert gradients[0].item() == pytest.approx(-14 if opposing else -16)
    assert head.weight.grad is None and head.weight.item() == 0


def test_rejected_trials_restore_warm_adamw_state_parameters_lrs_and_gradients():
    head = head_at(0)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.1, weight_decay=0)
    head.weight.grad = torch.ones_like(head.weight)
    optimizer.step()
    weights, state = deepcopy(head.state_dict()), deepcopy(optimizer.state_dict())
    old_grad = head.weight.grad.clone()

    def refuses_any_move():
        return head.weight.new_tensor(0 if torch.equal(head.weight, weights["weight"]) else 1)

    report = guarded_adamw_step(head, optimizer, [torch.ones_like(head.weight)], refuses_any_move,
                                GapGuardConfig(backoffs=3))
    assert not report["applied"] and report["rejected_trials"] == 4
    assert_tree_equal(head.state_dict(), weights)
    assert_tree_equal(optimizer.state_dict(), state)
    assert torch.equal(head.weight.grad, old_grad)


def test_retry_uses_original_moments_and_increments_adamw_step_only_once():
    head = head_at(1)
    optimizer = torch.optim.AdamW(head.parameters(), lr=4, weight_decay=0)
    report = guarded_adamw_step(head, optimizer, [torch.ones_like(head.weight)], lambda: head.weight.square().sum(),
                                GapGuardConfig(backoffs=3))
    assert report["applied"] and report["rejected_trials"] == 1 and report["learning_rate"] == 2
    assert optimizer.state[head.weight]["step"].item() == 1
    reference = head_at(1)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=2, weight_decay=0)
    reference.weight.grad = torch.ones_like(reference.weight)
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1)
    reference_optimizer.step()
    assert_tree_equal(head.state_dict(), reference.state_dict())
    assert_tree_equal(optimizer.state_dict(), reference_optimizer.state_dict())


def test_exception_during_acceptance_measurement_restores_optimizer_and_weights():
    head = head_at(1)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.1, weight_decay=0)
    weights, state = deepcopy(head.state_dict()), deepcopy(optimizer.state_dict())
    calls = 0

    def fails_after_update():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("failed probe")
        return head.weight.square().sum()

    with pytest.raises(RuntimeError, match="failed probe"):
        guarded_adamw_step(head, optimizer, [torch.ones_like(head.weight)], fails_after_update, GapGuardConfig())
    assert_tree_equal(head.state_dict(), weights)
    assert_tree_equal(optimizer.state_dict(), state)
    assert head.weight.grad is None


def test_nonfinite_gradient_is_rejected_without_advancing_optimizer():
    head = head_at(1)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.1, weight_decay=0)
    state = deepcopy(optimizer.state_dict())
    report = guarded_adamw_step(head, optimizer, [torch.full_like(head.weight, float("inf"))],
                                lambda: head.weight.square().sum(), GapGuardConfig(backoffs=1))
    assert not report["applied"] and report["rejected_trials"] == 2
    assert head.weight.item() == 1 and head.weight.grad is None
    assert_tree_equal(optimizer.state_dict(), state)


def warm_optimizer(head):
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.1, weight_decay=0)
    head.weight.grad = -torch.ones_like(head.weight)
    optimizer.step()
    with torch.no_grad():
        head.weight.fill_(1)
    return optimizer


def test_actual_adamw_direction_and_fresh_gap_fallback_are_measured():
    head = head_at(1)
    optimizer = warm_optimizer(head)
    report = guarded_adamw_step(head, optimizer, [-torch.ones_like(head.weight)],
        lambda: head.weight.square().sum(), GapGuardConfig(backoffs=1, fallback_gap_only=True),
        calibration_loss=lambda: head.weight.square().sum())
    assert report["fallback_applied"] and report["optimizer_moments_reset"]
    assert report["rejected_trials"] == 2 and len(report["attempts"]) == 3
    for trial in report["attempts"][:2]:
        assert trial["direction_is_gap_ascent"] and trial["gap_dot_parameter_delta"] > 0
        assert trial["gap_step_cosine"] == pytest.approx(1)
    fallback = report["attempts"][-1]
    assert fallback["optimizer_mode"] == "gap_fallback"
    assert fallback["gap_step_cosine"] == pytest.approx(-1)
    assert fallback["calibration_after"] < fallback["calibration_before"]
    assert optimizer.state[head.weight]["step"].item() == 1
    assert optimizer.state[head.weight]["exp_avg"].item() > 0
    reference = head_at(1)
    fresh = torch.optim.AdamW(reference.parameters(), lr=0.1, weight_decay=0)
    reference.weight.square().sum().backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1)
    fresh.step()
    assert_tree_equal(head.state_dict(), reference.state_dict())
    assert_tree_equal(optimizer.state_dict(), fresh.state_dict())


def test_fallback_that_hurts_training_calibration_restores_original_warm_moments():
    head = head_at(1)
    optimizer = warm_optimizer(head)
    weights, state = deepcopy(head.state_dict()), deepcopy(optimizer.state_dict())
    report = guarded_adamw_step(head, optimizer, [-torch.ones_like(head.weight)],
        lambda: head.weight.square().sum(), GapGuardConfig(backoffs=1, fallback_gap_only=True),
        calibration_loss=lambda: (head.weight - 2).square().sum())
    assert report["fallback_attempted"] and not report["fallback_applied"]
    assert not report["applied"] and not report["optimizer_moments_reset"]
    fallback = [row for row in report["attempts"] if row["optimizer_mode"] == "gap_fallback"]
    assert len(fallback) == 2
    assert all(row["gap_after"] < row["gap_before"] for row in fallback)
    assert all("fallback_calibration_regression" in row["rejection_reasons"] for row in fallback)
    assert_tree_equal(head.state_dict(), weights)
    assert_tree_equal(optimizer.state_dict(), state)


def test_exception_after_resetting_moments_restores_the_original_optimizer():
    head = head_at(1)
    optimizer = warm_optimizer(head)
    weights, state = deepcopy(head.state_dict()), deepcopy(optimizer.state_dict())

    def calibration():
        if head.weight.item() < 1:
            raise RuntimeError("fallback calibration failed")
        return head.weight.square().sum()

    with pytest.raises(RuntimeError, match="fallback calibration failed"):
        guarded_adamw_step(head, optimizer, [-torch.ones_like(head.weight)],
            lambda: head.weight.square().sum(), GapGuardConfig(backoffs=1, fallback_gap_only=True),
            calibration_loss=calibration)
    assert_tree_equal(head.state_dict(), weights)
    assert_tree_equal(optimizer.state_dict(), state)


def test_zero_gap_fallback_cannot_reset_moments_without_an_improvement():
    head = head_at(0)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.1, weight_decay=0)
    state = deepcopy(optimizer.state_dict())
    report = guarded_adamw_step(head, optimizer, [torch.ones_like(head.weight)],
        lambda: head.weight.square().sum(), GapGuardConfig(backoffs=1, fallback_gap_only=True),
        calibration_loss=lambda: head.weight.square().sum())
    assert not report["applied"] and not report["optimizer_moments_reset"]
    assert any("fallback_no_gap_improvement" in row["rejection_reasons"] for row in report["attempts"])
    assert_tree_equal(optimizer.state_dict(), state)
    assert head.weight.item() == 0
