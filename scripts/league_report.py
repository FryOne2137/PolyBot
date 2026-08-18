"""Print a compact JSON report from a PolyBot league database."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def _rows(cursor: sqlite3.Cursor) -> list[dict[str, object]]:
    return [dict(row) for row in cursor.fetchall()]


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
    evaluation_active_column = (
        "COALESCE(c.mcts_evaluation_active, 0)"
        if "mcts_evaluation_active" in capability_columns
        else "COALESCE(c.mcts_ready, 0)"
    )
    standings_query = (
        f"""SELECT m.id, m.name, m.status, ROUND(m.elo, 2) AS elo,
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
        """SELECT id, name, status, ROUND(elo, 2) AS elo, rated_games,
                  rated_wins, rated_draws, rated_losses, in_population,
                  protected, checkpoint_path, 0 AS mcts_probe_streak,
                  0 AS mcts_ready, 0 AS mcts_evaluation_active,
                  NULL AS mcts_raw_score,
                  NULL AS mcts_search_score
           FROM models WHERE checkpoint_path IS NOT NULL
           ORDER BY elo DESC, id DESC LIMIT ?"""
    )
    report = {
        "database": str(args.league_db.resolve()),
        "standings": _rows(connection.execute(
            standings_query,
            (args.limit,),
        )),
        "recent_rounds": _rows(connection.execute(
            """SELECT r.id, r.run_id, r.round_index, r.status, r.promoted,
                      before.name AS champion_before, after.name AS champion_after,
                      r.training_seed, r.evaluation_seed, r.started_at, r.finished_at
               FROM rounds r
               JOIN models before ON before.id = r.champion_before_id
               LEFT JOIN models after ON after.id = r.champion_after_id
               ORDER BY r.id DESC LIMIT ?""",
            (args.limit,),
        )),
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
    }
    connection.close()
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
