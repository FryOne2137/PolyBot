from __future__ import annotations

import json
import sqlite3

import pytest

from polybot.training.league_db import LeagueDatabase


def test_league_database_tracks_lineage_games_metrics_and_ratings(tmp_path):
    champion_path = tmp_path / "teacher.pt"
    opponent_path = tmp_path / "older.pt"
    promoted_path = tmp_path / "champion-001.pt"
    champion_path.write_bytes(b"teacher-checkpoint")
    opponent_path.write_bytes(b"older-checkpoint")
    promoted_path.write_bytes(b"promoted-checkpoint")
    database_path = tmp_path / "league.sqlite3"

    with LeagueDatabase(database_path) as database:
        run_id = database.start_run(tmp_path / "run", {"rounds": 1})
        champion_id = database.register_checkpoint(
            champion_path, {"d_model": 32}, run_id=run_id, protected=True
        )
        opponent_id = database.register_checkpoint(
            opponent_path, {"d_model": 32}, run_id=run_id, protected=True
        )
        round_id = database.start_round(run_id, 1, champion_id, 100, 1_000)
        candidate_id = database.create_candidate(
            run_id, champion_id, "candidate-001", {"d_model": 32}
        )
        database.attach_candidate(round_id, candidate_id)

        population = database.select_population(champion_id, maximum=2)
        population_ids = [int(row["id"]) for row in population]
        probabilities, scores = database.opponent_probabilities(
            champion_id,
            population_ids,
            pfsp_alpha=2.0,
            champion_probability=0.4,
            probability_floor=0.05,
        )
        database.record_opponent_selection(round_id, population_ids, probabilities, scores)
        assert sum(probabilities) == pytest.approx(1.0)
        assert probabilities[population_ids.index(champion_id)] == pytest.approx(0.4)

        evaluation_id = database.start_evaluation(
            round_id, candidate_id, opponent_id, "historical_gate", 1_000, 2
        )
        for game_index, score in enumerate((1.0, 0.5)):
            database.record_match(
                run_id=run_id,
                round_id=round_id,
                evaluation_id=evaluation_id,
                phase="evaluation",
                model_a_id=candidate_id,
                model_b_id=opponent_id,
                game={
                    "game_index": game_index,
                    "map_seed": 1_000,
                    "model_a_player": game_index,
                    "winner_player": 0 if score == 1.0 else -1,
                    "score_a": score,
                    "moves": 12 + game_index,
                },
            )
        summary = {
            "games": 2.0,
            "model_a_wins": 1.0,
            "model_a_losses": 0.0,
            "draws": 1.0,
            "model_a_score": 0.75,
            "score_lower_95": 0.50,
        }
        database.finish_evaluation(evaluation_id, summary, elo_k=32.0)
        database.record_metric_row(
            run_id, round_id, "round_001/population", {"iteration": 3, "loss": 0.25}
        )
        database.finalise_candidate(candidate_id, promoted_path=promoted_path)
        database.finish_round(
            round_id, candidate_id, True, {"updates": 4}, {"score": 0.75}
        )
        database.finish_run(run_id, "complete", {"promotions": 1})

    connection = sqlite3.connect(database_path)
    assert connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 2
    assert connection.execute("SELECT COUNT(*) FROM training_metrics").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM rating_history").fetchone()[0] == 2
    candidate = connection.execute(
        "SELECT status, checkpoint_path, elo FROM models WHERE id = ?", (candidate_id,)
    ).fetchone()
    assert candidate[0] == "promoted"
    assert candidate[1] == str(promoted_path.resolve())
    assert candidate[2] > 1000.0
    connection.close()


def test_unknown_historical_opponent_receives_pfsp_probability(tmp_path):
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    with LeagueDatabase(tmp_path / "league.sqlite3") as database:
        run_id = database.start_run(tmp_path, {})
        champion_id = database.register_checkpoint(first, {}, run_id=run_id)
        opponent_id = database.register_checkpoint(second, {}, run_id=run_id)
        probabilities, reference = database.opponent_probabilities(
            champion_id,
            [champion_id, opponent_id],
            pfsp_alpha=2.0,
            champion_probability=0.4,
            probability_floor=0.0,
        )

    assert reference == [0.5, None]
    assert probabilities == pytest.approx([0.4, 0.6])


def test_main_exploiter_pfsp_reserves_probability_for_its_target(tmp_path):
    champion = tmp_path / "champion.pt"
    counter = tmp_path / "counter.pt"
    champion.write_bytes(b"champion")
    counter.write_bytes(b"counter")
    with LeagueDatabase(tmp_path / "league.sqlite3") as database:
        run_id = database.start_run(tmp_path, {})
        champion_id = database.register_checkpoint(champion, {}, run_id=run_id)
        counter_id = database.register_checkpoint(counter, {}, run_id=run_id)
        probabilities, _ = database.opponent_probabilities(
            champion_id,
            [champion_id, counter_id],
            pfsp_alpha=2.0,
            champion_probability=0.4,
            probability_floor=0.05,
            role="main_exploiter",
            primary_opponent_id=counter_id,
            primary_opponent_probability=0.70,
        )

    assert sum(probabilities) == pytest.approx(1.0)
    assert probabilities[1] >= 0.70


def test_invalid_evaluation_does_not_update_elo_or_rating_history(tmp_path):
    champion_path = tmp_path / "champion.pt"
    candidate_path = tmp_path / "candidate.pt"
    champion_path.write_bytes(b"champion")
    candidate_path.write_bytes(b"candidate")

    with LeagueDatabase(tmp_path / "league.sqlite3") as database:
        run_id = database.start_run(tmp_path / "run", {})
        champion_id = database.register_checkpoint(champion_path, {}, run_id=run_id)
        round_id = database.start_round(run_id, 1, champion_id, 100, 200)
        candidate_id = database.create_candidate(run_id, champion_id, "candidate", {})
        database.attach_candidate(round_id, candidate_id)
        evaluation_id = database.start_evaluation(
            round_id, candidate_id, champion_id, "champion_gate", 200, 2
        )
        before = database.connection.execute(
            "SELECT id, elo, rated_games FROM models WHERE id IN (?, ?) ORDER BY id",
            (champion_id, candidate_id),
        ).fetchall()

        database.invalidate_evaluation(evaluation_id, {
            "valid": 0.0,
            "invalid_evaluations": 1.0,
            "invalid_reason": "belief recovery exhausted",
        })

        after = database.connection.execute(
            "SELECT id, elo, rated_games FROM models WHERE id IN (?, ?) ORDER BY id",
            (champion_id, candidate_id),
        ).fetchall()
        evaluation = database.connection.execute(
            "SELECT score_a, lower_95, summary_json FROM evaluations WHERE id = ?",
            (evaluation_id,),
        ).fetchone()
        rating_rows = database.connection.execute(
            "SELECT COUNT(*) FROM rating_history WHERE evaluation_id = ?",
            (evaluation_id,),
        ).fetchone()[0]

    assert [tuple(row) for row in after] == [tuple(row) for row in before]
    assert evaluation[0] is None and evaluation[1] is None
    assert json.loads(evaluation[2])["invalid_evaluations"] == 1.0
    assert rating_rows == 0


def test_mcts_capability_and_unrated_probe_survive_restart_without_elo(tmp_path):
    champion_path = tmp_path / "champion.pt"
    candidate_path = tmp_path / "candidate.pt"
    champion_path.write_bytes(b"champion")
    candidate_path.write_bytes(b"candidate")
    database_path = tmp_path / "league.sqlite3"

    with LeagueDatabase(database_path) as database:
        run_id = database.start_run(tmp_path / "run", {})
        champion_id = database.register_checkpoint(
            champion_path, {}, run_id=run_id
        )
        candidate_id = database.register_checkpoint(
            candidate_path, {}, run_id=run_id, status="promoted"
        )
        round_id = database.start_round(run_id, 1, champion_id, 10, 20)
        evaluation_id = database.start_evaluation(
            round_id,
            candidate_id,
            champion_id,
            "mcts_readiness_search_probe",
            30,
            10,
        )
        before = database.connection.execute(
            "SELECT elo, rated_games FROM models WHERE id = ?",
            (candidate_id,),
        ).fetchone()
        summary = {
            "games": 10.0,
            "model_a_wins": 6.0,
            "model_a_losses": 3.0,
            "draws": 1.0,
            "model_a_score": 0.65,
            "score_lower_95": 0.40,
        }
        database.finish_evaluation_unrated(evaluation_id, summary)
        database.record_mcts_capability(
            candidate_id,
            games=10,
            simulations=16,
            raw_score=0.62,
            search_score=0.65,
            streak=2,
            ready=True,
            summary=summary,
        )
        record = database.model_record(candidate_id)
        after = database.connection.execute(
            "SELECT elo, rated_games FROM models WHERE id = ?",
            (candidate_id,),
        ).fetchone()

    with LeagueDatabase(database_path) as database:
        capability = database.model_capability(candidate_id)

    assert tuple(after) == tuple(before)
    assert capability is not None
    assert capability["mcts_probe_streak"] == 2
    assert capability["mcts_ready"] == 1
    assert capability["mcts_search_score"] == pytest.approx(0.65)
    assert record["status"] == "promoted"
