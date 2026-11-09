"""Bounded native episode groups for diverse, flat-only self-play."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, replace

import numpy as np
from tqdm.auto import tqdm

from ..config import FlatCurriculumConfig
from ..data.full_actions import ACTION_ARRAYS, exact_vector_env
from ..data.player_context import MAP_TYPE_IDS, native_player_context

# Finite native groups do not autoreset: every training map is in the lower
# half and every mixed evaluation map in the upper half of the signed range.
# This is independent of user-selected base seeds, domains and league rounds.
SEED_SPLIT = 1 << 30
SEED_SPAN = SEED_SPLIT - 65_536


@dataclass(frozen=True)
class FlatDomain:
    size: int
    kind: str
    tribes: tuple[int, int]

    @property
    def label(self):
        return f"{self.size}-{self.kind}-{self.tribes[0]}-{self.tribes[1]}"

    def training(self, training, seed, *, evaluation=False):
        # An equally deep arena on a larger board needs more individual
        # actions. The mixed evaluator interprets its horizon at 11x11.
        limit = (training.max_moves * self.size * self.size + 120) // 121 if evaluation else training.max_moves
        return replace(training, map_size=self.size, map_type=self.kind,
                       player_tribes=self.tribes, seed=int(seed), flat_curriculum=None, max_moves=limit)


def domain_signature(training, history):
    return (training.map_size, training.map_type, tuple(training.player_tribes), history,
            asdict(training.flat_curriculum) if training.flat_curriculum else None)


def phase(settings, steps):
    return 0 if steps < settings.intro_steps else 1 if steps < settings.expansion_steps else 2


def size_weights(settings, *, stage=2):
    sizes = tuple(sorted(settings.map_sizes))
    eligible = (sizes[0],) if stage == 0 else tuple(s for s in sizes if s <= 13) if stage == 1 else sizes
    if not eligible:
        eligible = (sizes[0],)
    small = [s for s in eligible if s <= 13]
    large = [s for s in eligible if s > 13]
    return {s: ((settings.small_map_probability / len(small)) if s <= 13 else
                ((1 - settings.small_map_probability) / len(large)))
            if small and large else 1 / len(eligible) for s in eligible}


def type_weights(settings, *, stage=2):
    if stage == 0:
        return {"lakes" if "lakes" in settings.map_types else settings.map_types[0]: 1.0}
    total = sum(settings.map_type_weights)
    return {t: w / total for t, w in zip(settings.map_types, settings.map_type_weights)}


def new_session(settings, seed):
    return {"signature": asdict(settings), "rng": np.random.default_rng(seed ^ 0x444F4D),
            "steps": 0, "generation": 0, "pools": [], "exposure": {},
            "tribe_counts": np.zeros((2, 13), dtype=np.int64), "policy_version": 0}


def portable_session(session):
    if not session:
        return None
    return deepcopy({k: session[k] for k in ("signature", "steps", "generation", "exposure", "tribe_counts", "policy_version")} | {
        "rng_state": session["rng"].bit_generator.state})


def restore_session(session, saved, settings):
    if saved is None:
        return
    if saved["signature"] != asdict(settings):
        raise ValueError("flat curriculum resume settings differ from the saved domain mixture")
    session.update({k: v for k, v in saved.items() if k != "rng_state"})
    session["rng"] = np.random.default_rng()
    session["rng"].bit_generator.state = saved["rng_state"]
    session["pools"] = []  # native games are never claimed to be restored


def draw_domain(settings, session, num_envs):
    rng = session["rng"]
    stage = phase(settings, session["steps"])
    sizes, types = size_weights(settings, stage=stage), type_weights(settings, stage=stage)
    size = int(rng.choice(list(sizes), p=list(sizes.values())))
    kind = str(rng.choice(list(types), p=list(types.values())))
    tribes = []
    for seat in range(2):
        counts = session["tribe_counts"][seat, list(settings.tribes)]
        choices = np.asarray(settings.tribes)[counts == counts.min()]
        tribe = int(rng.choice(choices))
        tribes.append(tribe)
        session["tribe_counts"][seat, tribe] += num_envs
    return FlatDomain(size, kind, tuple(tribes))


def annotate_domain(batch, domain, seed):
    count = len(batch["state"])
    batch["domain_meta"] = np.tile(np.asarray((domain.size, MAP_TYPE_IDS[domain.kind], *domain.tribes), dtype=np.int32), (count, 1))
    batch["map_seed"] = np.arange(seed, seed + count, dtype=np.int64)
    # Preserve explicit map-type metadata in replay, not just the current
    # collector's fallback TrainingConfig.
    batch["player_context"] = native_player_context(batch, domain.kind)
    return batch


class EpisodeVectorPool:
    """One homogeneous native group, retiring only after all real games end."""

    def __init__(self, domain, *, num_envs, seed, num_threads, max_actions, history, max_moves=3200):
        from PolyEnv import VectorGameEnv
        if num_envs >= 65_536 or not 0 < seed < SEED_SPLIT - num_envs:
            raise ValueError("mixed training seeds must stay in the reserved training range")
        self.domain, self.seed = domain, int(seed)
        self.max_moves = max_moves
        self.moves = np.zeros(num_envs, dtype=np.int64)
        self.env = exact_vector_env(VectorGameEnv, num_envs=num_envs, seed=seed,
            map_size=domain.size, map_type=domain.kind, players=domain.tribes,
            num_threads=num_threads, max_actions=max_actions, auto_reset=False,
            visible_event_history=history)
        self.active_mask = np.ones(num_envs, dtype=bool)
        self.batch = annotate_domain(self.env.reset(seed=seed), domain, seed)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        return self.batch

    def step(self, actions):
        rows = np.flatnonzero(self.active_mask).astype(np.int32)
        if not len(rows):
            raise RuntimeError("a drained mixed-domain group cannot be stepped")
        if len(rows) == len(self.active_mask):
            result = self.env.step_checked(np.asarray(actions, dtype=np.int32), self.batch["state_id"].copy())
        else:
            update = self.env.step_slots_checked(rows, np.asarray(actions, dtype=np.int32)[rows].copy(),
                                                  self.batch["state_id"][rows].copy())
            result = {}
            width = max(self.batch["action_mask"].shape[1], update["action_mask"].shape[1])
            for key, previous in self.batch.items():
                if key in {"domain_meta", "map_seed", "player_context", "domain_truncated"}:
                    continue
                shape = list(previous.shape)
                if key in ACTION_ARRAYS:
                    shape[1] = width
                fill = 0 if key == "action_mask" or previous.dtype.kind in "bu" else -1
                target = np.full(shape, fill, dtype=previous.dtype)
                if key in ACTION_ARRAYS:
                    target[:, :previous.shape[1]] = previous
                    target[rows] = fill
                    target[rows, :update[key].shape[1]] = update[key]
                else:
                    target[...] = previous
                    target[rows] = update[key]
                result[key] = target
            # A terminal event is emitted once, not on every later partial step.
            inactive = ~self.active_mask
            result["terminated"][inactive] = 0
            result["reward"][inactive] = 0
        self.moves[rows] += 1
        terminal = np.asarray(result["terminated"], dtype=bool)
        truncated = self.active_mask & ~terminal & (self.moves >= self.max_moves)
        result["domain_truncated"] = truncated.astype(np.uint8)
        self.active_mask[terminal | truncated] = False
        self.batch = annotate_domain(result, self.domain, self.seed)
        return self.batch


def _aggregate(rows):
    total = sum(float(row["direct_population_steps"]) for row in rows)
    result = dict(rows[-1])
    for key, last in list(result.items()):
        if isinstance(last, (int, float)):
            result[key] = sum(float(row.get(key, 0)) * float(row["direct_population_steps"]) for row in rows) / total
    additive = ("steps", "environment_steps", "completed_games", "truncated_games", "wins", "losses", "draws", "candidate_turns", "opponent_turns",
                "ppo_samples", "ppo_policy_samples", "ppo_updates", "ppo_rollouts", "ppo_forced_truncations", "ppo_fragment_bootstraps",
                "terminal_value_positions", "ppg_samples_added", "ppg_phases", "ppg_updates", "ema_updates", "objective_replay_events",
                "objective_replay_positions", "objective_replay_updates", "terminal_replay_updates", "ppo_behaviour_kl_limits")
    for suffix in additive:
        key = "direct_population_" + suffix
        if key in result:
            result[key] = sum(float(row.get(key, 0)) for row in rows)
    for suffix in ("policy_version", "ppo_rollouts_total", "ppo_learning_rate"):
        key = "direct_population_" + suffix
        if key in result:
            result[key] = rows[-1][key]
    from .vector_population import _PolicySampleUsage, _TerminalValueCalibration
    usage, calibration = _PolicySampleUsage(), _TerminalValueCalibration()
    for row in rows:
        raw = row.get("flat_policy_usage", {})
        for key in ("samples", "eligible", "stale", "terminal"):
            setattr(usage, key, getattr(usage, key) + raw.get(key, 0))
        for age, count in raw.get("age_histogram", {}).items():
            age = int(age)
            usage.age_histogram[age] = usage.age_histogram.get(age, 0) + count
        for phase, counts in raw.get("phase_counts", {}).items():
            previous = usage.phase_counts.setdefault(phase, [0, 0])
            previous[0] += counts[0]
            previous[1] += counts[1]
        for key, value in row.get("flat_value_moments", {}).items():
            setattr(calibration, key, getattr(calibration, key) + value)
        for phase, moments in row.get("flat_value_phase_moments", {}).items():
            target = calibration.phase_calibrations.setdefault(phase, _TerminalValueCalibration())
            for key, value in moments.items():
                setattr(target, key, getattr(target, key) + value)
    result.update({"direct_population_ppo_" + k: v for k, v in usage.metrics().items()})
    result.update({"direct_population_value_calibration_" + k: v for k, v in calibration.metrics().items()})
    for key in ("flat_policy_usage", "flat_value_moments", "flat_value_phase_moments"):
        result.pop(key, None)
    games = result["direct_population_completed_games"]
    result["direct_population_score"] = (result["direct_population_wins"] + .5 * result["direct_population_draws"]) / max(1, games)
    return result


def train_mixed_population(training, model_config, candidate, opponents, weights, **kwargs):
    """Alternate frozen on-policy segments; keep live groups across blocks."""
    from .vector_population import train_vector_population
    settings = training.flat_curriculum
    if not kwargs["ppo_synchronous_rollouts"]:
        raise ValueError("mixed flat curriculum requires synchronous PPO rollouts")
    root = kwargs["continuation_state"]
    if root is None:
        raise ValueError("mixed flat curriculum requires persistent optimizer/environment state")
    run = kwargs["run"]
    session = root.get("domain_session")
    if session is None:
        session = new_session(settings, training.seed)
        root["domain_session"] = session
    if session["signature"] != asdict(settings):
        raise ValueError("cannot change a live flat domain mixture")
    results, consumed = [], 0
    budget = kwargs["steps"]
    global_keys = ("optimiser", "scaler", "policy_version", "ppo_rollouts", "ppg_phases_total")
    with tqdm(total=budget, desc="mixed flat PPO/PPG", unit="step", dynamic_ncols=True) as progress:
        while consumed < budget:
            pools = session["pools"]
            if len(pools) < settings.pools:
                pools.append({})
            # Rotate active groups; completed rows stay paused until the group
            # drains. No PPO block or domain phase discards an unfinished game.
            free = [i for i, p in enumerate(pools) if "env" not in p or not p["env"].active_mask.any()]
            if free:
                index = free[0]
            else:
                stage = phase(settings, session["steps"])
                sizes, kinds = size_weights(settings, stage=stage), type_weights(settings, stage=stage)
                cells = [(p["domain"].size, p["domain"].kind) for p in pools]
                masses = {cell: max(.01, sizes.get(cell[0], 0) * kinds.get(cell[1], 0)) for cell in cells}
                total_mass = sum(masses.values())
                exposure = session["exposure"]
                spent = {cell: sum(v["steps"] for k, v in exposure.items()
                                   if k.startswith(f"{cell[0]}-{cell[1]}-")) for cell in masses}
                total = sum(spent.values()) + kwargs["num_envs"] * settings.segment_steps
                deficits = {cell: total * mass / total_mass - spent[cell] for cell, mass in masses.items()}
                # Schedule transitions, rather than allowing longer games alone
                # to increase a domain's share. Old-phase live groups still drain.
                index = max(range(len(pools)), key=lambda i: (deficits[cells[i]], -i))
            state = pools[index]
            if "env" not in state or not state["env"].active_mask.any():
                domain = draw_domain(settings, session, kwargs["num_envs"])
                state.clear()
                seed = 1 + int(session["rng"].integers(SEED_SPAN))
                state["domain"] = domain
                state["seed"] = seed
                state["env"] = EpisodeVectorPool(domain, num_envs=kwargs["num_envs"], seed=seed,
                    num_threads=kwargs["num_threads"], max_actions=kwargs["max_actions"], history=kwargs["visible_action_history"],
                    max_moves=settings.episode_max_moves)
                run.event("flat_domain_group_started", domain=asdict(domain), seed=seed, pool=index,
                          curriculum_phase=phase(settings, session["steps"]), global_steps=session["steps"])
            domain = state["domain"]
            for key in global_keys:
                if key in root:
                    state[key] = root[key]
            state["policy_version"] = max(int(root.get("policy_version", 0)), session["policy_version"])
            count = min(settings.segment_steps, budget - consumed)
            local = dict(kwargs)
            progress.set_postfix(domain=domain.label, stage=phase(settings, session["steps"]))
            local.update(steps=count, continuation_state=state, complete_episodes_only=True, step_callback=lambda: progress.update(1),
                         finalize_plots=False, run=run.child(f"domain_{len(results)+1:03d}_{domain.label}"))
            # Each segment keeps its own replay/episode traces. Shared terminal
            # critic replay is shape-grouped by flat_value, never actor replay.
            start = kwargs.get("shaping_progress_start", 0.0)
            span = kwargs.get("shaping_progress_end", 1.0) - start
            local["shaping_progress_start"] = start + span * consumed / budget
            local["shaping_progress_end"] = start + span * (consumed + count) / budget
            # A persistent native map seed is not a rollout RNG seed. Use a
            # distinct recorded learner seed for each segment, otherwise every
            # return to a group would repeat its Categorical RNG sequence.
            learner_seed = 1 + (state["seed"] + session["generation"] * 1_000_003) % SEED_SPAN
            row = train_vector_population(domain.training(training, learner_seed), model_config,
                                          candidate, opponents, weights, **local)
            done = int(row["direct_population_steps"])
            if done <= 0:
                raise RuntimeError("mixed collector made no progress")
            results.append(row)
            consumed += done
            session["steps"] += done
            session["generation"] += 1
            exposure = session["exposure"].setdefault(domain.label, {"steps": 0, "games": 0, "actor_samples": 0})
            exposure["steps"] += int(row["direct_population_environment_steps"])
            exposure["games"] += int(row["direct_population_completed_games"])
            exposure["actor_samples"] += int(row["direct_population_ppo_samples"])
            for key in global_keys:
                if key in state:
                    root[key] = state.pop(key)
            session["policy_version"] = int(root.get("policy_version", 0))
            run.log(step=consumed, flat_domain_steps=session["steps"], flat_domain_phase=phase(settings, session["steps"]),
                    flat_domain_actor_samples=exposure["actor_samples"], flat_domain=domain.label,
                    loss=row.get("direct_population_ppo_loss", 0),
                    policy_loss=row.get("direct_population_ppo_policy_loss", 0),
                    value_loss=row.get("direct_population_ppo_value_loss", 0))
    result = _aggregate(results)
    result.update(flat_domain_phase=phase(settings, session["steps"]), flat_domain_steps=session["steps"],
                  flat_domain_exposure=deepcopy(session["exposure"]), flat_domain_active_pools=len(session["pools"]))
    run.event("flat_domain_exposure", domains=session["exposure"], tribe_starts=session["tribe_counts"].tolist())
    result["run_dir"] = str(run.finish(result))
    return result


def _weighted_interval(scores, weights):
    scores, weights = np.asarray(scores, dtype=float), np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    mean = float(np.dot(scores, weights))
    squares = float(np.square(weights).sum())
    effective = 1 / squares
    if len(scores) < 2 or squares >= 1:
        return mean, 1.0, 0.0, 1.0, effective
    se = float(np.sqrt(np.dot(weights, np.square(scores - mean)) * squares / (1 - squares)))
    z = 1.96
    center = (mean + z*z / (2*effective)) / (1 + z*z/effective)
    radius = z*np.sqrt(mean*(1-mean)/effective + z*z/(4*effective**2)) / (1 + z*z/effective)
    return mean, se, max(0.0, min(mean-z*se, center-radius)), min(1.0, max(mean+z*se, center+radius)), effective


def evaluation_schedule(settings, seed, games):
    """Fixed weighted domain strata for this arena, independent of learner phase."""
    if games < 2 or games % 2:
        raise ValueError("mixed evaluation needs complete map pairs")
    rng = np.random.default_rng(int(seed) ^ 0x4152454E)
    maps = games // 2
    if maps >= 65_536:
        raise ValueError("mixed arena must contain fewer than 65536 map pairs per call")
    groups = min(settings.evaluation_groups, maps)
    types = type_weights(settings)
    # Cover each requested map type before adding weighted extra groups.
    chosen_types = list(types)[:groups]
    while len(chosen_types) < groups:
        chosen_types.append(str(rng.choice(list(types), p=list(types.values()))))
    sizes = size_weights(settings)
    available = list(sizes)
    quantiles = (np.arange(groups) + rng.random()) / groups
    rng.shuffle(quantiles)
    cumulative_sizes = np.cumsum([sizes[s] for s in available])
    domains = []
    tribe_counts = np.zeros(13, dtype=int)
    for index, kind in enumerate(chosen_types):
        size = int(available[min(int(np.searchsorted(cumulative_sizes, quantiles[index])), len(available)-1)])
        tribes = []
        for seat in range(2):
            counts = tribe_counts[list(settings.tribes)]
            tribe = int(rng.choice(np.asarray(settings.tribes)[counts == counts.min()]))
            tribe_counts[tribe] += 1
            tribes.append(tribe)
        domains.append(FlatDomain(size, kind, tuple(tribes)))
    represented = sum(types[t] for t in set(chosen_types))
    group_weights = np.asarray([types[d.kind] / chosen_types.count(d.kind) / represented for d in domains])
    # Allocate at least one complete pair per group, then largest remainders.
    extra = group_weights * (maps - groups)
    counts = np.ones(groups, dtype=int) + np.floor(extra).astype(int)
    order = np.argsort(-(extra - np.floor(extra)), kind="stable")
    counts[order[:maps - int(counts.sum())]] += 1
    first_seed = SEED_SPLIT + (int(seed) % SEED_SPAN)
    offset = 0
    result = []
    for domain, count, weight in zip(domains, counts, group_weights):
        result.append((domain, first_seed + offset, int(count), float(weight), 2 * offset))
        offset += int(count)
    return result


def evaluate_mixed_matches(model_a, model_b, device, training, games, game_callback=None, **kwargs):
    from ..evaluation.matches import evaluate_direct_matches
    from .flat_experiments import isolated_rng
    if kwargs.get("map_seeds") is not None:
        raise ValueError("mixed flat evaluation uses its recorded domain/seed schedule, not a scalar map-seed override")
    settings = training.flat_curriculum
    records, pair_scores, pair_weights = [], [], []
    positions = kwargs.pop("position_callback", None)
    statistics = []
    schedule = evaluation_schedule(settings, training.seed, games)
    # Interleave global game indexes so small validation reservoirs cover
    # multiple domains instead of taking only the first arena group.
    pair_indexes = [[] for _ in schedule]
    sequence = 0
    for level in range(max(item[2] for item in schedule)):
        for group, item in enumerate(schedule):
            if level < item[2]:
                pair_indexes[group].append(sequence)
                sequence += 1
    with isolated_rng():
        for group, (domain, seed, count, weight, offset) in enumerate(schedule):
            global_pairs = pair_indexes[group]
            local_rows = []
            def record(row, domain=domain, global_pairs=global_pairs, weight=weight, count=count):
                row = dict(row)
                row.update(game_index=2 * global_pairs[int(row["game_index"]) // 2] + int(row["game_index"]) % 2, map_size=domain.size,
                           map_type=domain.kind, player_tribes=domain.tribes,
                           domain_weight=weight / (2 * count), domain=domain.label,
                           move_limit=domain.training(training, 1, evaluation=True).max_moves)
                records.append(row)
                local_rows.append(row)
                if game_callback is not None:
                    game_callback(row)
            def position(game, batch, row, domain=domain, global_pairs=global_pairs, seed=seed):
                positions(2 * global_pairs[game // 2] + game % 2, annotate_domain(dict(batch), domain, seed), row)
            result = evaluate_direct_matches(model_a, model_b, device, domain.training(training, seed, evaluation=True), 2 * count,
                game_callback=record, position_callback=position if positions is not None else None, **kwargs)
            statistics.append((result, weight))
            ordered = sorted(local_rows, key=lambda row: row["game_index"])
            pair_scores.extend((ordered[i]["score_a"] + ordered[i+1]["score_a"]) / 2 for i in range(0, len(ordered), 2))
            pair_weights.extend([weight / count] * count)
    score, se, lower, upper, effective_maps = _weighted_interval(pair_scores, pair_weights)
    numeric = ("model_a_city_captures", "model_a_capital_captures", "opponent_city_captures", "opponent_capital_captures",
               "model_a_decisions", "moves", "terminated_games", "truncated_games", "model_a_wins", "model_a_losses", "draws")
    result = {key: sum(float(row.get(key, 0)) for row, _ in statistics) for key in numeric}
    for key in ("model_a_city_captures_per_game", "completion_rate", "mean_moves"):
        result[key] = sum(float(row[key]) * weight for row, weight in statistics)
    result.update(games=float(games), model_a_score=score, model_a_win_rate=result["model_a_wins"] / games,
                  evaluation_maps=float(len(pair_scores)), score_standard_error=se, score_lower_95=lower, score_upper_95=upper, effective_evaluation_maps=effective_maps,
                  evaluation_seed=float(training.seed), visible_action_history=float(kwargs.get("visible_action_history", 0)),
                  flat_domain_evaluation=True)
    reports = {}
    for family, key in (("map_type", "map_type"), ("map_size", "map_size"), ("tribe", "tribe")):
        buckets = defaultdict(list)
        for row in records:
            label = row["player_tribes"][row["model_a_player"]] if key == "tribe" else row[key]
            buckets[str(label)].append(row)
        reports[family] = {label: {"games": len(rows), "score": float(sum(r["score_a"] * r["domain_weight"] for r in rows) /
                                                         sum(r["domain_weight"] for r in rows)),
                                           "completion_rate": sum(not r["truncated"] for r in rows) / len(rows)}
                           for label, rows in buckets.items()}
    failures = []
    for family in ("map_type", "map_size"):
        for label, report in reports[family].items():
            selected = [r for r in records if str(r[family]) == label]
            by_map = defaultdict(list)
            for r in selected:
                by_map[r["map_seed"]].append(r)
            samples = [np.mean([r["score_a"] for r in group]) for group in by_map.values()]
            sample_weights = [sum(r["domain_weight"] for r in group) for group in by_map.values()]
            _, _, _, domain_upper, effective = _weighted_interval(samples, sample_weights)
            report["maps"] = len(samples)
            report["effective_maps"] = effective
            report["score_upper_95"] = domain_upper
            if report["games"] >= settings.domain_min_games and report["score_upper_95"] < settings.domain_score_floor:
                failures.append(f"{family}={label}")
    result["domain_gate_passed"] = not failures
    result["domain_gate_failures"] = failures
    result["domain_scores"] = reports
    result["domain_schedule"] = [{"domain": asdict(d), "seed": s, "maps": n, "weight": w, "max_moves": d.training(training, s, evaluation=True).max_moves} for d, s, n, w, _ in evaluation_schedule(settings, training.seed, games)]
    return result
