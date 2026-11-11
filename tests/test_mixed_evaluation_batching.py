"""Inference batching must not change mixed-domain evaluation evidence."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from polybot.config import FlatCurriculumConfig, ModelConfig, TrainingConfig
from polybot.data.polyenv_features import native_batch_to_tensors
from polybot.evaluation.matches import _packet_rows, evaluate_direct_matches
from polybot.evaluation.mixed_matches import collate_policy_packets, run_mixed_arenas
from polybot.models.policy_value_belief import PolicyValueBeliefNet
from polybot.training.flat_domains import EpisodeVectorPool, FlatDomain, evaluation_schedule


def network(device='cpu'):
    torch.manual_seed(127)
    model = PolicyValueBeliefNet(ModelConfig(d_model=16, attention_heads=4, transformer_layers=1,
        belief_decoder_layers=1, action_encoder_dim=16, action_encoder_heads=4,
        action_latent_count=4, dropout=0)).eval().to(device)
    # Exercise learned history and public player/map context too.
    with torch.no_grad():
        model.history_gate[-1].weight.normal_(std=.05)
        model.observation_context[-1].weight.normal_(std=.05)
    return model


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA device is unavailable'))])
def test_ragged_policy_value_matches_separate_boards_history_and_full_actions(device):
    model = network(device)
    packets, expected, sides = [], [], []
    for size, kind, tribes in [(11, 'water_world', (1, 12)), (12, 'drylands', (3, 6)), (16, 'lakes', (5, 9))]:
        group = EpisodeVectorPool(FlatDomain(size, kind, tribes), num_envs=2, seed=2137,
            num_threads=1, max_actions=1, history=8)
        for _ in range(4):
            group.step(group.batch['action_id'][:, 0].copy())
        packet = _packet_rows(native_batch_to_tensors(group.batch, torch.device(device), size, kind), np.arange(2))
        # Invalid padding deliberately contains public-looking objects. The
        # tile mask must exclude them from pointer lookup and spatial readout.
        packets.append(packet)
        sides.extend([size] * 2)
        with torch.inference_mode():
            expected.append(model(**packet, include_belief=False))
    combined = collate_policy_packets(packets)
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    combined['map_tokens'][0, 121:, :] = 1
    combined['map_tokens'][0, 121:, 14] = 2
    combined['map_tokens'][0, 121:, 15] = 999
    with torch.inference_mode():
        actual = model(**combined, board_sizes=sides, include_belief=False)
    for i, (packet, separate) in enumerate(zip(packets, expected)):
        count = packet['action_mask'].shape[1]
        torch.testing.assert_close(actual.policy_logits[2*i:2*i+2, :count], separate.policy_logits, atol=3e-6, rtol=1e-5)
        torch.testing.assert_close(actual.value[2*i:2*i+2], separate.value, atol=3e-6, rtol=1e-5)
        assert torch.isneginf(actual.policy_logits[2*i:2*i+2, count:]).all()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original_state[key], rtol=0, atol=0)


def _sequential(model_a, model_b, training, schedule, history):
    rows = []
    for group, (domain, seed, maps, _, _) in enumerate(schedule):
        evaluate_direct_matches(model_a, model_b, torch.device('cpu'),
            domain.training(training, seed, evaluation=True), 2 * maps,
            game_callback=lambda row, group=group: rows.append((group, row)),
            visible_action_history=history, num_threads=1, max_actions=1)
    return sorted(rows, key=lambda item: (item[0], item[1]['game_index']))


def assert_same_games(actual, expected):
    actual = sorted(actual, key=lambda item: (item[0], item[1]['game_index']))
    assert len(actual) == len(expected)
    for (group, row), (expected_group, reference) in zip(actual, expected):
        assert group == expected_group
        # The new arena additionally reports per-game decision counts.
        assert {key: row[key] for key in reference} == reference


def test_batched_native_arena_preserves_games_and_reduces_model_forwards():
    model_a, model_b = network(), network()
    # Different opponents exercise routing by seat, not the identical-policy
    # shortcut. Use a short real engine arena, with different map limits.
    with torch.no_grad():
        model_b.shared_action_policy[-1].weight.neg_()
    training = TrainingConfig(seed=1531842, max_moves=12, flat_curriculum=FlatCurriculumConfig(evaluation_groups=3))
    schedule = evaluation_schedule(training.flat_curriculum, training.seed, 18)
    calls = []
    handles = [model.register_forward_hook(lambda *args: calls.append(1)) for model in (model_a, model_b)]
    try:
        expected = _sequential(model_a, model_b, training, schedule, 4)
    finally:
        for handle in handles:
            handle.remove()
    records, positions = [], {}
    summaries, stats = run_mixed_arenas(model_a, model_b, torch.device('cpu'), training, schedule,
        record=lambda group, row: records.append((group, row)),
        position=lambda group, game, batch, row: positions.setdefault((group, game), int(batch['map_seed'][row])),
        visible_action_history=4, num_threads=1, max_actions=1, max_live_games=18)
    assert_same_games(records, expected)
    assert stats['flat_evaluation_max_resident_games'] <= 18
    assert stats['flat_evaluation_model_forwards'] < len(calls)
    assert stats['flat_evaluation_max_inference_batch'] > max(2 * s[2] for s in schedule) / 2
    assert stats['flat_evaluation_tile_padding_ratio'] <= 1.5
    assert not stats['flat_evaluation_shared_policy']
    assert len(positions) == 18
    assert sum(item['games'] for item, _ in summaries) == 18
    for group, (domain, seed, maps, _, _) in enumerate(schedule):
        assert all(positions[group, 2 * i + seat] == seed + i for i in range(maps) for seat in (0, 1))


def test_identical_policy_and_value_only_changes_share_a_single_inference_route():
    model = network()
    opponent = deepcopy(model)
    with torch.no_grad():
        opponent.value_head[-2].bias.add_(.5)
    training = TrainingConfig(seed=2137, max_moves=2, flat_curriculum=FlatCurriculumConfig(map_sizes=(11,), evaluation_groups=3))
    schedule = evaluation_schedule(training.flat_curriculum, training.seed, 12)
    rows = []
    _, stats = run_mixed_arenas(model, opponent, torch.device('cpu'), training, schedule,
        record=lambda group, row: rows.append((group, row)), num_threads=1, max_actions=1)
    assert stats['flat_evaluation_shared_policy']
    assert stats['flat_evaluation_max_inference_batch'] == 6
    assert stats['flat_evaluation_model_forwards'] == 2
    assert stats['flat_evaluation_reused_paired_decisions'] == 12
    assert_same_games(rows, _sequential(model, opponent, training, schedule, 0))


def test_bounded_chunks_preserve_seed_order_positions_and_scores():
    model = network()
    training = TrainingConfig(seed=1531842, max_moves=3,
        flat_curriculum=FlatCurriculumConfig(evaluation_groups=3))
    schedule = evaluation_schedule(training.flat_curriculum, training.seed, 18)
    rows, seen = [], {}
    _, stats = run_mixed_arenas(model, model, torch.device('cpu'), training, schedule,
        record=lambda group, row: rows.append((group, row)),
        position=lambda group, game, batch, row: seen.setdefault((group, game), int(batch['map_seed'][row])),
        visible_action_history=4, num_threads=1, max_actions=1, max_live_games=4)
    assert_same_games(rows, _sequential(model, model, training, schedule, 4))
    assert stats['flat_evaluation_max_resident_games'] == 4
    # At this deliberately tiny horizon some second-seat policies never get
    # a turn. Check all actually observed positions, including later chunks.
    assert {group for group, _ in seen} == set(range(len(schedule)))
    assert any(game >= 4 for _, game in seen)
    for (group, game), seed in seen.items():
        assert seed == schedule[group][1] + game // 2


def test_action_token_budget_splits_rows_without_dropping_actions():
    from polybot.evaluation.mixed_matches import _bounded_requests
    packet = lambda rows, width: dict(action_mask=torch.ones(rows, width, dtype=torch.bool))
    requests = [(11, 0, np.arange(5), packet(5, 100)),
                (12, 1, np.arange(3), packet(3, 250))]
    parts = list(_bounded_requests(requests, 500))
    assert len(parts) > 1
    observed = []
    for part in parts:
        rows = sum(len(item[2]) for item in part)
        width = max(item[3]['action_mask'].shape[1] for item in part)
        assert rows * width <= 500
        for _, index, indexes, tensors in part:
            assert tensors['action_mask'].shape == (len(indexes), 100 if index == 0 else 250)
            assert tensors['action_mask'].all()
            observed.extend((index, int(i)) for i in indexes)
    assert observed == [(0, i) for i in range(5)] + [(1, i) for i in range(3)]


def test_terminal_rows_stop_stepping_while_sibling_games_continue(monkeypatch):
    from polybot.evaluation.mixed_matches import _Arena
    training = TrainingConfig(max_moves=3)
    arena = _Arena(0, FlatDomain(11, 'lakes', (1, 5)), 1073743961, 0, 2,
        training, torch.device('cpu'), num_threads=1, max_actions=1, history=0, position=None)
    original = arena.envs[0].step_into
    submitted = []
    def terminal_notification(ids, batch):
        submitted.append(ids.copy())
        original(ids, batch)
        if len(submitted) == 1:
            # Controlled native terminal notification; sibling positions are
            # still advanced by the actual checked public action adapter.
            batch['terminated'][0] = 1
            batch['state'][0, 3] = 0
    monkeypatch.setattr(arena.envs[0], 'step_into', terminal_notification)
    for _ in range(3):
        actions = [np.where(~arena.done[p], np.asarray(b['action_mask']).argmax(-1), -1)
                   for p, b in enumerate(arena.batches)]
        arena.advance(actions)
    rows = arena.game_rows()
    assert arena.done.all()
    assert rows[0]['moves'] == 1 and rows[0]['score_a'] == 1 and rows[0]['truncated'] == 0
    assert all(row['moves'] == 3 and row['truncated'] for row in rows[1:])
    assert submitted[0][0] >= 0 and submitted[1][0] == submitted[2][0] == -1


def test_progress_merges_draws_and_truncations_and_counts_all_unfinished_games(monkeypatch, capsys):
    from polybot.evaluation import mixed_matches
    class Progress:
        def __init__(self):
            self.n, self.frames = 0, []
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, count):
            self.n += count
        def set_postfix(self, ordered_dict=None, **values):
            self.frames.append((self.n, dict(ordered_dict or {}) | values))
    progress = Progress()
    monkeypatch.setattr(mixed_matches, 'tqdm', lambda **kwargs: progress)
    original = mixed_matches._Arena.advance
    def controlled_endings(arena, actions):
        original(arena, actions)
        age = getattr(arena, '_test_age', 0) + 1
        arena._test_age = age
        # Actual native sibling positions keep running. Control only terminal
        # notifications so all four result categories occur deterministically.
        if arena.group == 1 and age == 1:
            arena.done[:] = True
            arena.terminated[:] = True
            arena.winners[:] = -1
        elif arena.group == 0 and age <= 3:
            player, row, winner = ((0, 0, 0), (1, 0, 0), (0, 1, -1))[age-1]
            arena.done[player, row] = True
            arena.terminated[player, row] = True
            arena.winners[player, row] = winner
    monkeypatch.setattr(mixed_matches._Arena, 'advance', controlled_endings)
    schedule = [(FlatDomain(11, 'lakes', (1, 5)), 1073743961, 2, .5, 0),
                (FlatDomain(11, 'drylands', (3, 7)), 1073743963, 1, .5, 4)]
    model = network()
    summaries, stats = run_mixed_arenas(model, model, torch.device('cpu'),
        TrainingConfig(max_moves=4), schedule, record=lambda *args: None,
        num_threads=1, max_actions=1, max_live_games=8)
    expected = [(0, 0, 0, 0), (3, 1, 0, 2), (4, 1, 1, 2),
                (5, 1, 1, 3), (6, 1, 1, 4)]
    assert [(done, row['wins'], row['losses'], row['draws'])
            for done, row in progress.frames] == expected
    for done, row in progress.frames:
        assert list(row) == ['unfinished', 'moves', 'draws', 'losses', 'wins']
        assert sum(row[k] for k in ('wins', 'losses', 'draws')) == done
        assert done + row['unfinished'] == 6
    assert stats['flat_evaluation_wins'] == stats['flat_evaluation_losses'] == 1
    assert stats['flat_evaluation_terminal_draws'] == 3
    assert stats['flat_evaluation_truncated_games'] == 1
    # Console draws match the score contract, while artifacts retain the
    # distinction needed to exclude truncated games from terminal value labels.
    assert sum(result['draws'] for result, _ in summaries) == 4
    output = capsys.readouterr().out
    assert 'mixed evaluation results | games=6 | draws=4 | losses=1 | wins=1' in output
    assert 'truncated=' not in output and 'batch=' not in output


def test_padded_board_contract_rejects_invalid_sizes_and_belief():
    model = network()
    group = EpisodeVectorPool(FlatDomain(11, 'lakes', (1, 5)), num_envs=2, seed=2137,
        num_threads=1, max_actions=1, history=0)
    packet = _packet_rows(native_batch_to_tensors(group.batch, torch.device('cpu'), 11, 'lakes'), np.arange(2))
    for sizes in ([11], [11, 12], [11, 0]):
        with pytest.raises(ValueError, match='valid board size'):
            model(**packet, include_belief=False, board_sizes=sizes)
    with pytest.raises(ValueError, match='policy/value inference only'):
        model(**packet, board_sizes=[11, 11])


def test_native_cpu_collation_preserves_context_history_and_all_legal_actions():
    from polybot.evaluation.mixed_matches import _NativeCollator, _public_rows
    packets, expected = [], []
    for size, kind, tribes, history in [(11, 'water_world', (1, 12), 0), (16, 'drylands', (5, 7), 4)]:
        group = EpisodeVectorPool(FlatDomain(size, kind, tribes), num_envs=2, seed=2137,
            num_threads=1, max_actions=1, history=history)
        for _ in range(3):
            group.step(group.batch['action_id'][:, 0].copy())
        packets.append(_public_rows(group.batch, np.array([1])))
        expected.append(native_batch_to_tensors(packets[-1], torch.device('cpu'), size, kind))
    native = _NativeCollator(torch.device('cpu'))(packets)
    merged = native_batch_to_tensors(native, torch.device('cpu'), 11, 'lakes')
    assert native['action_features'].dtype == np.int32
    for row, (packet, reference) in enumerate(zip(packets, expected)):
        tiles, width = reference.map_tokens.shape[1], reference.action_mask.shape[1]
        torch.testing.assert_close(merged.map_tokens[row, :tiles], reference.map_tokens[0], rtol=0, atol=0)
        torch.testing.assert_close(merged.global_features[row], reference.global_features[0], rtol=0, atol=0)
        torch.testing.assert_close(merged.event_features[row], reference.event_features[0], rtol=0, atol=0)
        torch.testing.assert_close(merged.action_categorical[row, :width], reference.action_categorical[0], rtol=0, atol=0)
        torch.testing.assert_close(merged.action_continuous[row, :width], reference.action_continuous[0], rtol=0, atol=0)
        assert merged.action_mask[row].sum() == reference.action_mask.sum()
        count = reference.action_history.shape[1]
        if count:
            torch.testing.assert_close(merged.action_history[row, -count:], reference.action_history[0], rtol=0, atol=0)
        else:
            assert not merged.action_history_mask[row].any()


@pytest.mark.parametrize('bucketing', ['nearby', 'exact'])
def test_one_cpu_input_batch_per_forward_and_measured_work(bucketing):
    model = network()
    training = TrainingConfig(seed=1531842, max_moves=3,
        flat_curriculum=FlatCurriculumConfig(evaluation_groups=3))
    schedule = evaluation_schedule(training.flat_curriculum, training.seed, 12)
    rows = []
    _, stats = run_mixed_arenas(model, model, torch.device('cpu'), training, schedule,
        record=lambda group, row: rows.append((group, row)), visible_action_history=4,
        num_threads=1, max_actions=1, size_bucketing=bucketing)
    assert_same_games(rows, _sequential(model, model, training, schedule, 4))
    assert stats['flat_evaluation_input_batches'] == stats['flat_evaluation_model_forwards']
    assert stats['flat_evaluation_environment_steps'] == sum(row['moves'] for _, row in rows)
    assert stats['flat_evaluation_elapsed_seconds'] > 0
    assert stats['flat_evaluation_policy_inference_seconds'] > 0
    assert stats['flat_evaluation_environment_steps_per_second'] > 0
    if bucketing == 'exact':
        assert stats['flat_evaluation_tile_padding_ratio'] == 1


def test_paired_reuse_requires_equal_public_inputs():
    from polybot.evaluation.mixed_matches import _Arena, _same_paired_inputs
    arena = _Arena(0, FlatDomain(11, 'lakes', (1, 5)), 1073743961, 0, 2,
        TrainingConfig(max_moves=3), torch.device('cpu'), num_threads=1,
        max_actions=1, history=2, position=None)
    assert _same_paired_inputs(arena)
    arena.batches[1]['own_techs'][0, 0] ^= 1
    assert not _same_paired_inputs(arena)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device is unavailable')
def test_cuda_native_staging_is_pinned_contiguous_and_reused():
    from polybot.evaluation.mixed_matches import _NativeCollator
    collator = _NativeCollator(torch.device('cuda'))
    first = collator._array('actions', (4, 100, 17), np.dtype(np.int32))
    owner = collator.owners['actions']
    assert first.flags.c_contiguous and torch.as_tensor(first).is_pinned()
    second = collator._array('actions', (2, 75, 17), np.dtype(np.int32))
    assert collator.owners['actions'] is owner
    assert second.flags.c_contiguous and torch.as_tensor(second).is_pinned()
    assert collator._array('history', (2, 0, 26), np.dtype(np.int32)).shape == (2, 0, 26)


def test_real_throughput_probe_writes_reproducible_work_counts(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    import runpy
    import sys
    from polybot.infrastructure.model_io import save_model
    model = network()
    checkpoint, report = tmp_path / 'model.pt', tmp_path / 'probe.json'
    save_model(checkpoint, model, model.config)
    script = Path(__file__).resolve().parents[1] / 'scripts/benchmark_flat_evaluation.py'
    monkeypatch.syspath_prepend(str(script.parent))
    monkeypatch.setattr(sys, 'argv', [str(script), '--model', str(checkpoint), '--device', 'cpu',
        '--games', '6', '--max-moves', '2', '--num-threads', '1', '--history', '2',
        '--modes', 'single,mixed-one-size,mixed-nearby,mixed-exact', '--batch-games', '6', '--output', str(report)])
    runpy.run_path(str(script), run_name='__main__')
    data = json.loads(report.read_text())
    assert len(data['cases']) == 4 and data['fingerprint']
    for case in data['cases']:
        assert case['environment_steps_per_second'] > 0
        assert case['seconds_per_game'] > 0
        assert case['model_score'] == .5
        if case['mode'] != 'single':
            assert case['flat_evaluation_input_batches'] == case['flat_evaluation_model_forwards']
