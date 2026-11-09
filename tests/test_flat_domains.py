"""Mixed flat training uses real native groups and domain-correct evidence."""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import json
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from polybot.config import FlatCurriculumConfig, ModelConfig, TrainingConfig
from polybot.models.policy_value_belief import PolicyValueBeliefNet
from polybot.training.flat_domains import (
    EpisodeVectorPool, FlatDomain, SEED_SPLIT, draw_domain, evaluation_schedule,
    new_session, phase, portable_session, restore_session,
)
from polybot.training.run_artifacts import TrainingRun
from polybot.training.vector_population import train_vector_population, _ppo_snapshot


def model():
    torch.manual_seed(13)
    return PolicyValueBeliefNet(ModelConfig(d_model=16, attention_heads=4, transformer_layers=1,
        belief_decoder_layers=1, action_encoder_dim=16, action_encoder_heads=4,
        action_latent_count=4, dropout=0)).eval()


def test_balanced_tribes_phase_changes_and_mirror_matches():
    settings = FlatCurriculumConfig(intro_steps=10, expansion_steps=20)
    session = new_session(settings, 2137)
    early = [draw_domain(settings, session, 2) for _ in range(12)]
    assert {(d.size, d.kind) for d in early} == {(11, 'lakes')}
    assert all(set(d.tribes) <= set(range(1, 13)) for d in early)
    for seat in range(2):
        assert session['tribe_counts'][seat, 1:].tolist() == [2] * 12
    session['steps'] = 15
    middle = [draw_domain(settings, session, 2) for _ in range(200)]
    assert {d.size for d in middle} == {11, 12, 13}
    assert {d.kind for d in middle} == {'lakes', 'drylands', 'water_world'}
    session['steps'] = 20
    full = [draw_domain(settings, session, 2) for _ in range(300)]
    assert {d.size for d in full} == set(range(11, 17))
    mirror = replace(settings, tribes=(5,))
    assert draw_domain(mirror, new_session(mirror, 1), 2).tribes == (5, 5)


def test_domain_arena_is_deterministic_weighted_and_separate_from_training():
    settings = FlatCurriculumConfig()
    a = evaluation_schedule(settings, 2137, 100)
    assert a == evaluation_schedule(settings, 2137, 100)
    assert a != evaluation_schedule(settings, 731842, 100)
    assert sum(n for _, _, n, _, _ in a) == 50
    assert sum(w for _, _, _, w, _ in a) == pytest.approx(1)
    for kind, expected in zip(settings.map_types, settings.map_type_weights):
        assert sum(w for d, _, _, w, _ in a if d.kind == kind) == pytest.approx(expected)
    assert all(seed >= SEED_SPLIT for _, seed, _, _, _ in a)
    session = new_session(settings, 2137)
    saved = portable_session(session)
    restored = {}
    restore_session(restored, saved, settings)
    assert draw_domain(settings, session, 2) == draw_domain(settings, restored, 2)
    assert not restored['pools']
    with pytest.raises(ValueError, match='resume settings'):
        restore_session({}, saved, replace(settings, map_sizes=(11,)))


def test_native_episode_group_pauses_a_finished_slot_without_reset_or_repeated_reward():
    group = EpisodeVectorPool(FlatDomain(11, 'drylands', (5, 5)), num_envs=2, seed=2137,
                             num_threads=1, max_actions=1, history=4)
    # Control one terminal notification to exercise native checked partial
    # stepping. The actual sibling position is advanced by the real engine.
    original = group.env.step_checked
    def first(actions, ids):
        batch = original(actions, ids)
        batch['terminated'][0] = 1
        batch['reward'][0] = 1
        return batch
    with patch.object(group.env, 'step_checked', side_effect=first):
        batch = group.step(group.batch['action_id'][:, 0].copy())
    before = batch['state_id'].copy()
    assert group.active_mask.tolist() == [False, True]
    following = group.step(batch['action_id'][:, 0].copy())
    assert following['state_id'][0] == before[0]
    assert following['state_id'][1] != before[1]
    assert following['terminated'][0] == 0 and following['reward'][0] == 0
    assert following['domain_meta'][1].tolist() == [11, 1, 5, 5]
    np.testing.assert_array_equal(following['map_seed'], [2137, 2138])


def test_mixed_native_learning_reuses_games_and_one_optimizer_across_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(TrainingRun, '_write_plots', lambda self: None)
    network = model()
    before = network.action_encoder.type_embedding.weight.detach().clone()
    settings = FlatCurriculumConfig(intro_steps=0, expansion_steps=0, pools=2, segment_steps=2)
    training = TrainingConfig(device='cpu', learning_rate=1e-4, flat_curriculum=settings)
    state, critic = {}, []
    scripted = iter([FlatDomain(11, 'drylands', (1, 12)), FlatDomain(16, 'water_world', (5, 5))])
    monkeypatch.setattr('polybot.training.flat_domains.draw_domain', lambda *args: next(scripted))
    anchor, opponent = deepcopy(network), deepcopy(network)
    def train(name):
        return train_vector_population(training, network.config, network, [opponent], [1.],
            num_envs=2, steps=4, num_threads=1, max_actions=1, anchor_model=anchor,
            visible_action_history=4, ppo_synchronous_rollouts=True, ppo_rollout_steps=2,
            ppo_minibatch_size=2, ppo_top_advantage_fraction=1., ppo_epochs=1,
            ppg_enabled=False, objective_replay_weight=0, value_replay=critic,
            continuation_state=state, run=TrainingRun.create(tmp_path, 'native', name))
    first = train('one')
    pool_ids = [id(p['env']) for p in state['domain_session']['pools']]
    positions = [p['batch']['state_id'].copy() for p in state['domain_session']['pools']]
    optimizer, version = state['optimiser'], state['policy_version']
    second = train('two')
    assert [id(p['env']) for p in state['domain_session']['pools']] == pool_ids
    assert state['optimiser'] is optimizer and state['policy_version'] > version
    assert any(not np.array_equal(p['batch']['state_id'], old) for p, old in zip(state['domain_session']['pools'], positions))
    assert all('optimiser' not in p for p in state['domain_session']['pools'])
    assert first['direct_population_environment_steps'] == second['direct_population_environment_steps'] == 8
    assert second['flat_domain_steps'] == 8
    assert not torch.equal(before, network.action_encoder.type_embedding.weight)
    assert all(np.all(p['batch']['map_seed'] < SEED_SPLIT) for p in state['domain_session']['pools'])
    assert not critic  # no invented outcome labels at a block boundary


def test_value_features_keep_order_and_domain_for_mixed_shapes():
    from polybot.training.flat_value import _states
    from polybot.training.vector_teacher import _vector_tensors
    from polybot.data.player_context import MAP_TYPE_IDS
    network = model()
    examples, expected = [], []
    for size, kind, tribes in [(11, 'drylands', (1, 2)), (16, 'water_world', (5, 6)), (11, 'lakes', (3, 4))]:
        domain = FlatDomain(size, kind, tribes)
        group = EpisodeVectorPool(domain, num_envs=2, seed=3187, num_threads=1, max_actions=1, history=2)
        batch = group.batch
        snap, _ = _ppo_snapshot(batch, 0, 0)
        examples.append(SimpleNamespace(observation=snap))
        inputs = _vector_tensors(batch, torch.device('cpu'), domain.training(TrainingConfig(), 3187))
        with torch.no_grad():
            output = network(**inputs, include_belief=False)
        expected.append(output.state[0])
        assert inputs['global_features'][0, 7] == pytest.approx(size / 32)
        assert inputs['global_features'][0, 9 + 122 + MAP_TYPE_IDS[kind]] == 1
    actual = _states(network, examples, TrainingConfig())
    torch.testing.assert_close(actual, torch.stack(expected), atol=1e-6, rtol=1e-5)


def test_mixed_evaluation_callbacks_and_position_reservoirs_cover_domains():
    from polybot.evaluation.matches import evaluate_direct_matches
    network = model()
    settings = FlatCurriculumConfig(evaluation_groups=3)
    training = TrainingConfig(device='cpu', max_moves=2, flat_curriculum=settings)
    games, seen = [], {}
    def position(game, batch, row):
        seen.setdefault(game, tuple(batch['domain_meta'][row]))
    result = evaluate_direct_matches(network, deepcopy(network), torch.device('cpu'), training, 12,
        game_callback=games.append, position_callback=position, num_threads=1, max_actions=1, visible_action_history=2)
    assert result['model_a_score'] == pytest.approx(.5)
    assert set(result['domain_scores']['map_type']) == {'lakes', 'drylands', 'water_world'}
    assert len(games) == len(seen) == 12
    assert sorted(g['game_index'] for g in games) == list(range(12))
    assert {seen[i][1] for i in range(6)} == {0, 1, 2}
    assert all(g['map_seed'] >= SEED_SPLIT for g in games)
    assert result['score_lower_95'] < .5 and np.isfinite(result['score_standard_error'])


def test_flat_cli_runs_mixed_native_collection_and_records_resume_signature(tmp_path, monkeypatch):
    from polybot.infrastructure.model_io import save_model
    monkeypatch.setattr(TrainingRun, '_write_plots', lambda self: None)
    champion = tmp_path / 'champion.pt'
    network = model()
    save_model(champion, network, network.config)
    script = Path(__file__).resolve().parents[1] / 'scripts/train_flat.py'
    arguments = [str(script), '--champion-in', str(champion), '--rounds', '2',
        '--checkpoint-dir', str(tmp_path / 'models'), '--league-db', str(tmp_path / 'league.sqlite'),
        '--project', str(tmp_path / 'runs'), '--device', 'cpu', '--seed', '3187', '--evaluation-seed', '1503187',
        '--num-threads', '1', '--population-size', '1', '--promotion-games', '4', '--historical-games', '4',
        '--direct-population-gate-games', '4', '--direct-population-diagnostic-gate-games', '4',
        '--direct-population-envs', '2', '--direct-population-steps', '4', '--direct-population-max-steps', '4',
        '--post-bootstrap-population-steps', '4', '--post-bootstrap-population-max-steps', '4',
        '--direct-population-block-steps', '4', '--direct-population-ppo-rollout-steps', '2',
        '--direct-population-ppo-minibatch-size', '2', '--direct-population-ppo-epochs', '1',
        '--max-moves', '2', '--visible-action-history', '2', '--no-breakthrough-exploiter',
        '--no-direct-population-ppg', '--flat-population-terminal-replay-value-weight', '0',
        '--flat-population-regression-confirm-games', '0', '--flat-value-updates', '0',
        '--flat-rollout-steps-final', '2', '--flat-curriculum', '--flat-domain-pools', '2',
        '--flat-domain-segment-steps', '2', '--flat-domain-evaluation-groups', '2',
        '--flat-curriculum-intro-steps', '0', '--flat-curriculum-expansion-steps', '0',
        '--flat-map-sizes', '11,13', '--flat-map-types', 'lakes,drylands', '--flat-map-type-weights', '0.5,0.5']
    with patch.object(sys, 'argv', arguments), patch.object(sys, 'path', [str(script.parent), *sys.path]):
        runpy.run_path(str(script), run_name='__main__')
    states = list((tmp_path / 'runs').rglob('latest_state.pt'))
    assert states
    latest = max((torch.load(p, weights_only=False) for p in states), key=lambda s: s['domain_session']['steps'])
    assert latest['domain_session']['steps'] >= 8
    assert latest['domain_session']['signature']['map_sizes'] == (11, 13)
    assert len(latest['map_signature']) == 5
    assert not latest['environment_restored']
    resumed = list(arguments)
    resumed[resumed.index('--rounds') + 1] = '1'
    latest_path = max(states, key=lambda p: torch.load(p, weights_only=False)['domain_session']['steps'])
    resumed.extend(['--flat-resume-state', str(latest_path), '--name', 'resumed'])
    with patch.object(sys, 'argv', resumed), patch.object(sys, 'path', [str(script.parent), *sys.path]):
        runpy.run_path(str(script), run_name='__main__')
    restored = next((tmp_path / 'runs').glob('league/resumed/**/latest_state.pt'))
    resumed_state = torch.load(restored, weights_only=False)
    assert resumed_state['domain_session']['steps'] >= 12
    assert resumed_state['policy_version'] > latest['policy_version']


def test_mixed_horizon_bootstraps_without_fabricating_terminal_value(tmp_path, monkeypatch):
    monkeypatch.setattr(TrainingRun, '_write_plots', lambda self: None)
    network = model()
    settings = FlatCurriculumConfig(pools=1, segment_steps=8, episode_max_moves=2)
    state, replay, games = {}, [], []
    result = train_vector_population(TrainingConfig(device='cpu', flat_curriculum=settings), network.config,
        network, [deepcopy(network)], [1.], num_envs=2, steps=4, num_threads=1, max_actions=1,
        anchor_model=deepcopy(network), visible_action_history=2, ppo_synchronous_rollouts=True,
        ppo_rollout_steps=2, ppo_minibatch_size=2, ppo_epochs=1, ppg_enabled=False,
        objective_replay_weight=0, continuation_state=state, value_replay=replay,
        game_callback=games.append, run=TrainingRun.create(tmp_path, 'horizon'))
    assert result['direct_population_truncated_games'] == 4
    assert result['direct_population_completed_games'] == 0
    assert not replay
    assert len(games) == 4 and all(g['truncated'] for g in games)
    assert len({g['player_tribes'] for g in games}) == 2
    assert state['domain_session']['steps'] == 4


def test_paired_rollback_rejects_changed_domains_and_uses_domain_weights():
    from polybot.training.flat_experiments import paired_regression
    a, b = [], []
    for seed in range(10):
        for side in (0, 1):
            weight = .09 if seed == 0 else .01
            a.append(dict(map_seed=seed, model_a_player=side, domain='lake', domain_weight=weight,
                          score_a=1.0 if seed == 0 else 0.0))
            b.append(dict(map_seed=seed, model_a_player=side, domain='lake', domain_weight=weight, score_a=0.0))
    result = paired_regression(a, b)
    assert result['difference'] == pytest.approx(.5)
    changed = deepcopy(b)
    changed[0]['domain'] = 'water'
    assert paired_regression(a, changed)['reason'] == 'unpaired_domain_evidence'


def test_domain_aggregation_counts_labels_and_actor_ages_exactly():
    from polybot.training.flat_domains import _aggregate
    from polybot.training.vector_population import _TerminalValueCalibration, _PolicySampleUsage
    segments = []
    for count, old_value, outcome, age in [(2, .2, 1., 0), (6, -.3, -1., 2)]:
        calibration = _TerminalValueCalibration()
        calibration.add([SimpleNamespace(old_value=old_value)] * count, outcome, record_phases=False)
        usage = _PolicySampleUsage(samples=count, eligible=count, age_histogram={age: count})
        segments.append(dict(direct_population_steps=1., direct_population_completed_games=1.,
            direct_population_wins=float(outcome > 0), direct_population_losses=float(outcome < 0),
            direct_population_draws=0., flat_policy_usage=vars(usage),
            flat_value_moments={k:v for k,v in vars(calibration).items() if k != 'phase_calibrations'}))
    result = _aggregate(segments)
    assert result['direct_population_value_calibration_count'] == 8
    assert result['direct_population_value_calibration_mae'] == pytest.approx((2*.8+6*.7)/8)
    assert result['direct_population_ppo_policy_age_mean'] == pytest.approx(1.5)
    assert result['direct_population_ppo_policy_age_p95'] == 2


def test_domain_gate_can_block_promotion_despite_a_good_overall_score():
    from polybot.training.league import _promotion_gate_decision
    rows = [dict(opponent_id=1, model_a_score=.8, score_lower_95=.7,
                 sampling_probability=1., domain_gate_passed=False)]
    def decide():
        return _promotion_gate_decision(rows, 1, promotion_score=.55,
            promotion_lower_bound=.5, population_score=.52, historical_score_floor=.45)
    assert not decide()['promoted']
    rows[0]['domain_gate_passed'] = True
    assert decide()['promoted']


def test_mixed_value_head_fit_keeps_encoders_frozen():
    from collections import deque
    from polybot.training.flat_value import ValuePosition, fit_value_head
    from polybot.training.flat_experiments import FlatImprovements
    network = model()
    replay, validation = [], []
    for seed, destination in [(3187, replay), (731842, validation)]:
        for size, kind in [(11, 'drylands'), (16, 'water_world')]:
            group = EpisodeVectorPool(FlatDomain(size, kind, (1, 5)), num_envs=2, seed=seed,
                                     num_threads=1, max_actions=1, history=2)
            observation, _ = _ppo_snapshot(group.batch, 0, 0)
            destination.extend(ValuePosition(observation, 1., game_key=(seed, size, i)) for i in range(4))
    before = deepcopy(network.action_encoder.state_dict())
    result = fit_value_head(network, TrainingConfig(), replay, validation,
        FlatImprovements(value_updates=2, value_batch_size=4, value_capacity=16,
                         value_learning_rate=.001), seed=2137)
    assert result['kept']
    assert result['after']['rmse'] < result['before']['rmse']
    for name, weight in network.action_encoder.state_dict().items():
        torch.testing.assert_close(weight, before[name], rtol=0, atol=0)


def test_opponent_tribe_and_seed_provenance_never_become_neural_inputs():
    from polybot.training.vector_teacher import _vector_tensors
    group = EpisodeVectorPool(FlatDomain(11, 'lakes', (1, 5)), num_envs=2, seed=3187,
                             num_threads=1, max_actions=1, history=2)
    first = _vector_tensors(group.batch, torch.device('cpu'), TrainingConfig())
    changed = {k: v.copy() for k, v in group.batch.items()}
    changed['domain_meta'][:, 2:] = 12
    changed['map_seed'] += 1_000_000
    second = _vector_tensors(changed, torch.device('cpu'), TrainingConfig())
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    snapshot, _ = _ppo_snapshot(group.batch, 0, 0)
    assert int(snapshot['map_seed']) == 3187
