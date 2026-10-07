"""Print a compact JSON report from a PolyBot league database."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import sqlite3
from pathlib import Path


def _rows(cursor: sqlite3.Cursor) -> list[dict[str, object]]:
    return [dict(row) for row in cursor.fetchall()]


def _json_object(value: object) -> dict[str, object]:
    if not isinstance(value, str):
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _flat_diagnostics(
    connection: sqlite3.Connection,
    recent_rounds: list[dict[str, object]],
    *,
    has_archive_roles: bool,
) -> dict[str, object]:
    """Summarise what the flat curriculum actually sampled and optimized."""
    round_ids = [int(row["id"]) for row in recent_rounds]
    usage: dict[str, dict[str, object]] = {}
    if round_ids:
        placeholders = ",".join("?" for _ in round_ids)
        match_rows = _rows(connection.execute(
            f"""SELECT m.phase, m.score_a, m.created_at, m.model_b_id,
                       b.name AS opponent_name, m.metrics_json
                FROM matches m
                LEFT JOIN models b ON b.id = m.model_b_id
                WHERE m.round_id IN ({placeholders})
                  AND m.phase LIKE 'direct_population%'
                ORDER BY m.id DESC""",
            round_ids,
        ))
        for row in match_rows:
            metrics = _json_object(row.get("metrics_json"))
            stable_id = metrics.get("opponent_id")
            key = (
                str(stable_id)
                if stable_id is not None
                else str(row.get("model_b_id") or "untracked-local-snapshot")
            )
            label = str(
                row.get("opponent_name")
                or stable_id
                or "local flat snapshot"
            )
            item = usage.setdefault(key, {
                "opponent_id": stable_id,
                "opponent": label,
                "games": 0,
                "training_games": 0,
                "gate_games": 0,
                "score_sum": 0.0,
                "latest_at": row.get("created_at"),
            })
            item["games"] = int(item["games"]) + 1
            item["score_sum"] = float(item["score_sum"]) + float(
                row["score_a"]
            )
            if "training" in str(row["phase"]):
                item["training_games"] = int(item["training_games"]) + 1
            else:
                item["gate_games"] = int(item["gate_games"]) + 1

    training_total = sum(int(item["training_games"]) for item in usage.values())
    opponent_usage = []
    for item in usage.values():
        games = max(1, int(item["games"]))
        opponent_usage.append({
            "opponent_id": item["opponent_id"],
            "opponent": item["opponent"],
            "games": int(item["games"]),
            "training_games": int(item["training_games"]),
            "training_share": (
                int(item["training_games"]) / max(1, training_total)
            ),
            "gate_games": int(item["gate_games"]),
            "mean_score": float(item["score_sum"]) / games,
            "latest_at": item["latest_at"],
        })
    opponent_usage.sort(
        key=lambda item: (-int(item["training_games"]), str(item["opponent"]))
    )

    benchmark_rows: list[dict[str, object]] = []
    optimizer = defaultdict(float)
    rollback_reasons: dict[str, int] = defaultdict(int)
    for round_row in recent_rounds:
        summary = _json_object(round_row.get("training_summary_json"))
        direct = summary.get("direct_population")
        if not isinstance(direct, dict):
            continue
        branches = {
            "selected": direct,
            "main": direct.get("main"),
            "main_exploiter": direct.get("exploiter"),
        }
        for role, branch in branches.items():
            if not isinstance(branch, dict):
                continue
            gate_rows = branch.get("direct_population_gate_rows", [])
            if isinstance(gate_rows, list):
                for gate in gate_rows:
                    if not isinstance(gate, dict):
                        continue
                    if gate.get("flat_static_benchmark_score") is not None:
                        benchmark_rows.append({
                            "round": round_row["round_index"],
                            "role": gate.get("flat_role", role),
                            "block": gate.get("block"),
                            "benchmark_score": gate.get(
                                "flat_static_benchmark_score"
                            ),
                            "training_mix_score": gate.get(
                                "training_mix_score"
                            ),
                            "accepted": gate.get("accepted"),
                            "selection_reason": gate.get(
                                "selection_reason"
                            ),
                        })
                    if not bool(gate.get("accepted", 0.0)):
                        rollback_reasons[str(
                            gate.get("selection_reason", "rejected")
                        )] += 1
            for row in branch.get("direct_population_training_rows", []):
                if not isinstance(row, dict):
                    continue
                samples = float(row.get("direct_population_ppo_samples", 0.0))
                optimizer["ppo_samples"] += samples
                optimizer["policy_samples"] += float(
                    row.get("direct_population_ppo_policy_samples", 0.0)
                )
                optimizer["terminal_value_labels"] += float(
                    row.get("direct_population_value_calibration_count", 0.0)
                )
                optimizer["terminal_replay_updates"] += float(
                    row.get("direct_population_terminal_replay_updates", 0.0)
                )
                optimizer["kl_limited_updates"] += float(
                    row.get("direct_population_ppo_behaviour_kl_limits", 0.0)
                )
                optimizer["policy_age_weighted_sum"] += samples * float(
                    row.get("direct_population_ppo_policy_age_mean", 0.0)
                )
                optimizer["policy_age_max"] = max(
                    optimizer["policy_age_max"],
                    float(row.get("direct_population_ppo_policy_age_max", 0.0)),
                )
    optimizer["mean_policy_age"] = (
        optimizer["policy_age_weighted_sum"]
        / max(1.0, optimizer["ppo_samples"])
    )
    optimizer["terminal_label_share"] = (
        optimizer["terminal_value_labels"]
        / max(1.0, optimizer["ppo_samples"])
    )
    optimizer.pop("policy_age_weighted_sum", None)

    if has_archive_roles:
        active_strategies = _rows(connection.execute(
            """SELECT role, COUNT(*) AS count
               FROM models
               WHERE checkpoint_path IS NOT NULL
                 AND (status IN ('initial', 'promoted')
                      OR (status = 'archive' AND trainable_opponent = 1))
               GROUP BY role ORDER BY role"""
        ))
    else:
        active_strategies = _rows(connection.execute(
            """SELECT 'main' AS role, COUNT(*) AS count
               FROM models WHERE checkpoint_path IS NOT NULL
                 AND status IN ('initial', 'promoted')"""
        ))
    hard_counters = sorted(
        opponent_usage,
        key=lambda item: (float(item["mean_score"]), -int(item["games"])),
    )[:5]
    return {
        "opponent_usage": opponent_usage,
        "hardest_counters": hard_counters,
        "static_benchmark": benchmark_rows,
        "active_strategies": active_strategies,
        "optimizer_and_value": dict(optimizer),
        "rollback_reasons": dict(sorted(rollback_reasons.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect league standings and recent gates")
    parser.add_argument("--league-db", type=Path, default=Path("models/league/league.sqlite3"))
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    if not args.league_db.is_file():
        raise FileNotFoundError(f"league database does not exist: {args.league_db}")
    if args.limit < 1:
        raise ValueError("limit must be positive")

    connection = sqlite3.connect(args.league_db)
    connection.row_factory = sqlite3.Row
    has_capabilities = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' "
        "AND name = 'model_capabilities'"
    ).fetchone() is not None
    capability_columns = (
        {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(model_capabilities)"
            ).fetchall()
        }
        if has_capabilities
        else set()
    )
    model_columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(models)").fetchall()
    }
    role_column = "m.role" if "role" in model_columns else "'main'"
    trainable_column = (
        "m.trainable_opponent"
        if "trainable_opponent" in model_columns
        else "m.in_population"
    )
    evaluation_active_column = (
        "COALESCE(c.mcts_evaluation_active, 0)"
        if "mcts_evaluation_active" in capability_columns
        else "COALESCE(c.mcts_ready, 0)"
    )
    standings_query = (
        f"""SELECT m.id, m.name, m.status, {role_column} AS role,
                  {trainable_column} AS trainable_opponent,
                  ROUND(m.elo, 2) AS elo,
                  m.rated_games, m.rated_wins, m.rated_draws,
                  m.rated_losses, m.in_population, m.protected,
                  m.checkpoint_path,
                  COALESCE(c.mcts_probe_streak, 0) AS mcts_probe_streak,
                  COALESCE(c.mcts_ready, 0) AS mcts_ready,
                  {evaluation_active_column} AS mcts_evaluation_active,
                  ROUND(c.mcts_raw_score, 4) AS mcts_raw_score,
                  ROUND(c.mcts_search_score, 4) AS mcts_search_score
           FROM models m
           LEFT JOIN model_capabilities c ON c.model_id = m.id
           WHERE m.checkpoint_path IS NOT NULL
           ORDER BY m.elo DESC, m.id DESC LIMIT ?"""
        if has_capabilities
        else
        """SELECT id, name, status, 'main' AS role,
                  in_population AS trainable_opponent,
                  ROUND(elo, 2) AS elo, rated_games,
                  rated_wins, rated_draws, rated_losses, in_population,
                  protected, checkpoint_path, 0 AS mcts_probe_streak,
                  0 AS mcts_ready, 0 AS mcts_evaluation_active,
                  NULL AS mcts_raw_score,
                  NULL AS mcts_search_score
           FROM models WHERE checkpoint_path IS NOT NULL
           ORDER BY elo DESC, id DESC LIMIT ?"""
    )
    recent_rounds = _rows(connection.execute(
        """SELECT r.id, r.run_id, r.round_index, r.status, r.promoted,
                  before.name AS champion_before, after.name AS champion_after,
                  r.training_seed, r.evaluation_seed, r.started_at, r.finished_at,
                  r.training_summary_json
           FROM rounds r
           JOIN models before ON before.id = r.champion_before_id
           LEFT JOIN models after ON after.id = r.champion_after_id
           ORDER BY r.id DESC LIMIT ?""",
        (args.limit,),
    ))
    report = {
        "database": str(args.league_db.resolve()),
        "standings": _rows(connection.execute(
            standings_query,
            (args.limit,),
        )),
        "recent_rounds": [
            {
                key: value
                for key, value in row.items()
                if key != "training_summary_json"
            }
            for row in recent_rounds
        ],
        "recent_evaluations": _rows(connection.execute(
            """SELECT e.id, e.round_id, a.name AS candidate, b.name AS opponent,
                      e.purpose, e.games, ROUND(e.score_a, 4) AS score_a,
                      ROUND(e.lower_95, 4) AS lower_95, e.base_seed, e.finished_at
               FROM evaluations e
               JOIN models a ON a.id = e.model_a_id
               JOIN models b ON b.id = e.model_b_id
               ORDER BY e.id DESC LIMIT ?""",
            (args.limit,),
        )),
        "flat_training": _flat_diagnostics(
            connection,
            recent_rounds,
            has_archive_roles=(
                "role" in model_columns
                and "trainable_opponent" in model_columns
            ),
        ),
    }
    connection.close()
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
