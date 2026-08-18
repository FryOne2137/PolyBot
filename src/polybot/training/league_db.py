"""Durable SQLite history for population-based training leagues.

Checkpoints stay as ordinary ``.pt`` files so loading them does not require
copying a potentially large BLOB through SQLite.  The database stores their
absolute path, SHA-256 digest, size, lineage and rating together with every
round, evaluation game and scalar training metric.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "__dict__"):
        return vars(value)
    return str(value)


def _json(value: Any) -> str:
    return json.dumps(value, default=_json_default, sort_keys=True, allow_nan=False)


def checkpoint_digest(path: Path) -> tuple[str, int]:
    """Return a stable checkpoint identity without loading it into PyTorch."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


class LeagueDatabase:
    """Small transactional store used by :mod:`polybot.training.league`."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=60.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = NORMAL")
        self.connection.execute("PRAGMA busy_timeout = 60000")
        self._pending_metric_rows = 0
        self._create_schema()

    def __enter__(self) -> "LeagueDatabase":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()
        self.connection.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS league_runs (
                id INTEGER PRIMARY KEY,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL,
                run_directory TEXT NOT NULL,
                config_json TEXT NOT NULL,
                summary_json TEXT
            );

            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY,
                run_id INTEGER REFERENCES league_runs(id),
                parent_model_id INTEGER REFERENCES models(id),
                name TEXT NOT NULL,
                checkpoint_path TEXT,
                checkpoint_sha256 TEXT,
                checkpoint_bytes INTEGER,
                model_config_json TEXT NOT NULL,
                status TEXT NOT NULL,
                protected INTEGER NOT NULL DEFAULT 0,
                in_population INTEGER NOT NULL DEFAULT 0,
                elo REAL NOT NULL DEFAULT 1000.0,
                rated_games INTEGER NOT NULL DEFAULT 0,
                rated_wins INTEGER NOT NULL DEFAULT 0,
                rated_draws INTEGER NOT NULL DEFAULT 0,
                rated_losses INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            DROP INDEX IF EXISTS models_checkpoint_digest;
            CREATE INDEX IF NOT EXISTS models_checkpoint_digest_lookup
                ON models(checkpoint_sha256) WHERE checkpoint_sha256 IS NOT NULL;
            CREATE INDEX IF NOT EXISTS models_population
                ON models(in_population, status, elo DESC);

            CREATE TABLE IF NOT EXISTS rounds (
                id INTEGER PRIMARY KEY,
                run_id INTEGER NOT NULL REFERENCES league_runs(id),
                round_index INTEGER NOT NULL,
                champion_before_id INTEGER NOT NULL REFERENCES models(id),
                candidate_model_id INTEGER REFERENCES models(id),
                champion_after_id INTEGER REFERENCES models(id),
                training_seed INTEGER NOT NULL,
                evaluation_seed INTEGER NOT NULL,
                status TEXT NOT NULL,
                promoted INTEGER,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                training_summary_json TEXT,
                gate_summary_json TEXT,
                UNIQUE(run_id, round_index)
            );

            CREATE TABLE IF NOT EXISTS opponent_selections (
                id INTEGER PRIMARY KEY,
                round_id INTEGER NOT NULL REFERENCES rounds(id),
                opponent_model_id INTEGER NOT NULL REFERENCES models(id),
                sampling_probability REAL NOT NULL,
                reference_score REAL,
                selected_order INTEGER NOT NULL,
                UNIQUE(round_id, opponent_model_id)
            );

            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY,
                round_id INTEGER NOT NULL REFERENCES rounds(id),
                model_a_id INTEGER NOT NULL REFERENCES models(id),
                model_b_id INTEGER NOT NULL REFERENCES models(id),
                purpose TEXT NOT NULL,
                base_seed INTEGER NOT NULL,
                games INTEGER NOT NULL,
                score_a REAL,
                lower_95 REAL,
                summary_json TEXT,
                started_at TEXT NOT NULL,
                finished_at TEXT
            );

            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY,
                run_id INTEGER NOT NULL REFERENCES league_runs(id),
                round_id INTEGER NOT NULL REFERENCES rounds(id),
                evaluation_id INTEGER REFERENCES evaluations(id),
                phase TEXT NOT NULL,
                model_a_id INTEGER NOT NULL REFERENCES models(id),
                model_b_id INTEGER REFERENCES models(id),
                game_index INTEGER NOT NULL,
                map_seed INTEGER NOT NULL,
                model_a_player INTEGER,
                winner_player INTEGER,
                score_a REAL NOT NULL,
                moves INTEGER NOT NULL,
                metrics_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS matches_pair
                ON matches(model_a_id, model_b_id, phase, created_at);

            CREATE TABLE IF NOT EXISTS rating_history (
                id INTEGER PRIMARY KEY,
                evaluation_id INTEGER NOT NULL REFERENCES evaluations(id),
                model_id INTEGER NOT NULL REFERENCES models(id),
                old_elo REAL NOT NULL,
                new_elo REAL NOT NULL,
                expected_score REAL NOT NULL,
                actual_score REAL NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS training_metrics (
                id INTEGER PRIMARY KEY,
                run_id INTEGER NOT NULL REFERENCES league_runs(id),
                round_id INTEGER REFERENCES rounds(id),
                stage TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                metric TEXT NOT NULL,
                value REAL NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS training_metrics_lookup
                ON training_metrics(run_id, round_id, stage, metric, iteration);
            """
        )
        current = self.connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
        if current is not None and int(current["value"]) != SCHEMA_VERSION:
            raise RuntimeError(
                f"unsupported league database schema {current['value']}; expected {SCHEMA_VERSION}"
            )
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self.connection.commit()

    def start_run(self, run_directory: Path, config: Mapping[str, Any]) -> int:
        cursor = self.connection.execute(
            """INSERT INTO league_runs(started_at, status, run_directory, config_json)
               VALUES(?, 'running', ?, ?)""",
            (_utc_now(), str(run_directory.resolve()), _json(config)),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def finish_run(self, run_id: int, status: str, summary: Mapping[str, Any]) -> None:
        self.connection.execute(
            """UPDATE league_runs SET finished_at = ?, status = ?, summary_json = ?
               WHERE id = ?""",
            (_utc_now(), status, _json(summary), run_id),
        )
        self.connection.commit()

    def register_checkpoint(
        self,
        path: Path,
        model_config: Any,
        *,
        run_id: int | None,
        parent_model_id: int | None = None,
        status: str = "initial",
        protected: bool = False,
        protect_existing: bool = False,
        in_population: bool = True,
    ) -> int:
        resolved = path.resolve()
        digest, size = checkpoint_digest(resolved)
        existing = self.connection.execute(
            "SELECT id FROM models WHERE checkpoint_sha256 = ?", (digest,)
        ).fetchone()
        if existing is not None:
            model_id = int(existing["id"])
            self.connection.execute(
                """UPDATE models SET checkpoint_path = ?,
                   protected = CASE WHEN ? THEN MAX(protected, ?) ELSE protected END,
                   in_population = CASE
                       WHEN status IN ('initial', 'promoted') THEN in_population
                       ELSE ?
                   END,
                   updated_at = ? WHERE id = ?""",
                (
                    str(resolved), int(protect_existing), int(protected),
                    int(in_population), _utc_now(), model_id,
                ),
            )
            self.connection.commit()
            return model_id
        now = _utc_now()
        cursor = self.connection.execute(
            """INSERT INTO models(
                   run_id, parent_model_id, name, checkpoint_path, checkpoint_sha256,
                   checkpoint_bytes, model_config_json, status, protected,
                   in_population, created_at, updated_at
               ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id, parent_model_id, resolved.stem, str(resolved), digest, size,
                _json(model_config), status, int(protected), int(in_population),
                now, now,
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def create_candidate(
        self,
        run_id: int,
        parent_model_id: int,
        name: str,
        model_config: Any,
    ) -> int:
        parent = self.connection.execute(
            "SELECT elo FROM models WHERE id = ?", (parent_model_id,)
        ).fetchone()
        if parent is None:
            raise KeyError(f"unknown parent model {parent_model_id}")
        now = _utc_now()
        cursor = self.connection.execute(
            """INSERT INTO models(
                   run_id, parent_model_id, name, model_config_json, status, elo,
                   created_at, updated_at
               ) VALUES(?, ?, ?, ?, 'training', ?, ?, ?)""",
            (run_id, parent_model_id, name, _json(model_config), float(parent["elo"]), now, now),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def finalise_candidate(self, model_id: int, *, promoted_path: Path | None) -> None:
        now = _utc_now()
        if promoted_path is None:
            self.connection.execute(
                """UPDATE models SET status = 'rejected', in_population = 0, updated_at = ?
                   WHERE id = ? AND status = 'training'""",
                (now, model_id),
            )
        else:
            resolved = promoted_path.resolve()
            digest, size = checkpoint_digest(resolved)
            self.connection.execute(
                """UPDATE models SET status = 'promoted', in_population = 1,
                   checkpoint_path = ?, checkpoint_sha256 = ?, checkpoint_bytes = ?,
                   updated_at = ? WHERE id = ?""",
                (str(resolved), digest, size, now, model_id),
            )
        self.connection.commit()

    def retain_candidate(self, model_id: int, checkpoint_path: Path) -> None:
        """Keep a non-regressing learner outside the protected population."""
        resolved = checkpoint_path.resolve()
        digest, size = checkpoint_digest(resolved)
        self.connection.execute(
            """UPDATE models SET status = 'challenger', in_population = 0,
               checkpoint_path = ?, checkpoint_sha256 = ?, checkpoint_bytes = ?,
               updated_at = ? WHERE id = ? AND status = 'training'""",
            (str(resolved), digest, size, _utc_now(), model_id),
        )
        self.connection.commit()

    def start_round(
        self,
        run_id: int,
        round_index: int,
        champion_id: int,
        training_seed: int,
        evaluation_seed: int,
    ) -> int:
        cursor = self.connection.execute(
            """INSERT INTO rounds(
                   run_id, round_index, champion_before_id, training_seed,
                   evaluation_seed, status, started_at
               ) VALUES(?, ?, ?, ?, ?, 'running', ?)""",
            (run_id, round_index, champion_id, training_seed, evaluation_seed, _utc_now()),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def attach_candidate(self, round_id: int, candidate_id: int) -> None:
        self.connection.execute(
            "UPDATE rounds SET candidate_model_id = ? WHERE id = ?", (candidate_id, round_id)
        )
        self.connection.commit()

    def finish_round(
        self,
        round_id: int,
        champion_after_id: int,
        promoted: bool,
        training_summary: Mapping[str, Any],
        gate_summary: Mapping[str, Any],
    ) -> None:
        self.connection.execute(
            """UPDATE rounds SET champion_after_id = ?, promoted = ?, status = 'complete',
               finished_at = ?, training_summary_json = ?, gate_summary_json = ?
               WHERE id = ?""",
            (
                champion_after_id, int(promoted), _utc_now(), _json(training_summary),
                _json(gate_summary), round_id,
            ),
        )
        self.connection.commit()

    def fail_round(self, round_id: int, error: BaseException) -> None:
        self.connection.execute(
            """UPDATE rounds SET status = 'failed', finished_at = ?, gate_summary_json = ?
               WHERE id = ?""",
            (_utc_now(), _json({"error": repr(error)}), round_id),
        )
        self.connection.commit()

    def model_rows(self, model_ids: Sequence[int]) -> list[sqlite3.Row]:
        if not model_ids:
            return []
        placeholders = ",".join("?" for _ in model_ids)
        rows = self.connection.execute(
            f"SELECT * FROM models WHERE id IN ({placeholders})", tuple(model_ids)
        ).fetchall()
        by_id = {int(row["id"]): row for row in rows}
        return [by_id[model_id] for model_id in model_ids if model_id in by_id]

    def select_population(self, champion_id: int, maximum: int) -> list[sqlite3.Row]:
        if maximum < 1:
            raise ValueError("population maximum must be positive")
        rows = list(self.connection.execute(
            """SELECT * FROM models
               WHERE checkpoint_path IS NOT NULL AND status IN ('initial', 'promoted')
               ORDER BY id DESC"""
        ).fetchall())
        by_id = {int(row["id"]): row for row in rows}
        if champion_id not in by_id:
            raise KeyError(f"current champion {champion_id} has no checkpoint")
        selected = [by_id[champion_id]]
        selected_ids_set = {champion_id}

        # Protected teachers are permanent strategic anchors. Remaining slots
        # are split between empirical counters (unknown/low score first) and
        # recent checkpoints, preventing both forgetting and a stale league.
        for row in sorted(rows, key=lambda item: (-int(item["protected"]), -int(item["id"]))):
            model_id = int(row["id"])
            if not int(row["protected"]) or model_id in selected_ids_set:
                continue
            if len(selected) >= maximum:
                break
            selected.append(row)
            selected_ids_set.add(model_id)
        open_slots = maximum - len(selected)
        remaining = [row for row in rows if int(row["id"]) not in selected_ids_set]
        hard_slots = (open_slots + 1) // 2
        difficulty_rows = [
            (row, self.latest_pair_score(champion_id, int(row["id"])))
            for row in remaining
        ]
        difficulty_rows.sort(key=lambda item: (
            item[1] is not None,
            item[1] if item[1] is not None else 0.0,
            -float(item[0]["elo"]),
        ))
        for row, _ in difficulty_rows[:hard_slots]:
            selected.append(row)
            selected_ids_set.add(int(row["id"]))
        for row in rows:
            model_id = int(row["id"])
            if len(selected) >= maximum:
                break
            if model_id not in selected_ids_set:
                selected.append(row)
                selected_ids_set.add(model_id)
        selected_ids = [int(row["id"]) for row in selected]
        self.connection.execute("UPDATE models SET in_population = 0")
        if selected_ids:
            placeholders = ",".join("?" for _ in selected_ids)
            self.connection.execute(
                f"UPDATE models SET in_population = 1 WHERE id IN ({placeholders})",
                tuple(selected_ids),
            )
        self.connection.commit()
        return selected

    def latest_pair_score(self, model_a_id: int, model_b_id: int) -> float | None:
        direct = self.connection.execute(
            """SELECT score_a FROM evaluations
               WHERE model_a_id = ? AND model_b_id = ? AND score_a IS NOT NULL
               ORDER BY id DESC LIMIT 1""",
            (model_a_id, model_b_id),
        ).fetchone()
        if direct is not None:
            return float(direct["score_a"])
        reverse = self.connection.execute(
            """SELECT score_a FROM evaluations
               WHERE model_a_id = ? AND model_b_id = ? AND score_a IS NOT NULL
               ORDER BY id DESC LIMIT 1""",
            (model_b_id, model_a_id),
        ).fetchone()
        return None if reverse is None else 1.0 - float(reverse["score_a"])

    def opponent_probabilities(
        self,
        champion_id: int,
        opponent_ids: Sequence[int],
        *,
        pfsp_alpha: float,
        champion_probability: float,
        probability_floor: float,
    ) -> tuple[list[float], list[float | None]]:
        """PFSP distribution with an explicit latest-champion component."""
        if not opponent_ids:
            raise ValueError("opponent population must not be empty")
        if champion_id not in opponent_ids:
            raise ValueError("current champion must be present in opponent population")
        reference_scores = [
            0.5 if model_id == champion_id else self.latest_pair_score(champion_id, model_id)
            for model_id in opponent_ids
        ]
        if len(opponent_ids) == 1:
            return [1.0], reference_scores
        remaining = max(0.0, 1.0 - champion_probability)
        raw = []
        for model_id, score in zip(opponent_ids, reference_scores):
            if model_id == champion_id:
                raw.append(0.0)
            else:
                # Unknown strategies receive full priority until measured.
                raw.append(max(1e-8, 1.0 - (0.0 if score is None else score)) ** pfsp_alpha)
        raw_total = sum(raw)
        probabilities = [
            champion_probability if model_id == champion_id
            else remaining * raw_value / max(raw_total, 1e-8)
            for model_id, raw_value in zip(opponent_ids, raw)
        ]
        probabilities = [max(probability_floor, value) for value in probabilities]
        normaliser = sum(probabilities)
        return [value / normaliser for value in probabilities], reference_scores

    def record_opponent_selection(
        self,
        round_id: int,
        opponent_ids: Sequence[int],
        probabilities: Sequence[float],
        reference_scores: Sequence[float | None],
    ) -> None:
        self.connection.executemany(
            """INSERT INTO opponent_selections(
                   round_id, opponent_model_id, sampling_probability,
                   reference_score, selected_order
               ) VALUES(?, ?, ?, ?, ?)""",
            [
                (round_id, model_id, probability, score, index)
                for index, (model_id, probability, score) in enumerate(
                    zip(opponent_ids, probabilities, reference_scores)
                )
            ],
        )
        self.connection.commit()

    def start_evaluation(
        self,
        round_id: int,
        model_a_id: int,
        model_b_id: int,
        purpose: str,
        base_seed: int,
        games: int,
    ) -> int:
        cursor = self.connection.execute(
            """INSERT INTO evaluations(
                   round_id, model_a_id, model_b_id, purpose, base_seed, games, started_at
               ) VALUES(?, ?, ?, ?, ?, ?, ?)""",
            (round_id, model_a_id, model_b_id, purpose, base_seed, games, _utc_now()),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def record_match(
        self,
        *,
        run_id: int,
        round_id: int,
        phase: str,
        model_a_id: int,
        model_b_id: int | None,
        game: Mapping[str, Any],
        evaluation_id: int | None = None,
    ) -> None:
        self.connection.execute(
            """INSERT INTO matches(
                   run_id, round_id, evaluation_id, phase, model_a_id, model_b_id,
                   game_index, map_seed, model_a_player, winner_player, score_a,
                   moves, metrics_json, created_at
               ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id, round_id, evaluation_id, phase, model_a_id, model_b_id,
                int(game.get("game_index", 0)), int(game["map_seed"]),
                game.get("model_a_player"), game.get("winner_player"),
                float(game["score_a"]), int(game.get("moves", 0)), _json(game), _utc_now(),
            ),
        )

    def finish_evaluation(
        self,
        evaluation_id: int,
        summary: Mapping[str, Any],
        *,
        elo_k: float,
    ) -> None:
        evaluation = self.connection.execute(
            "SELECT * FROM evaluations WHERE id = ?", (evaluation_id,)
        ).fetchone()
        if evaluation is None:
            raise KeyError(f"unknown evaluation {evaluation_id}")
        score = float(summary["model_a_score"])
        lower = float(summary["score_lower_95"])
        self.connection.execute(
            """UPDATE evaluations SET score_a = ?, lower_95 = ?, summary_json = ?,
               finished_at = ? WHERE id = ?""",
            (score, lower, _json(summary), _utc_now(), evaluation_id),
        )
        model_a = self.connection.execute(
            "SELECT * FROM models WHERE id = ?", (int(evaluation["model_a_id"]),)
        ).fetchone()
        model_b = self.connection.execute(
            "SELECT * FROM models WHERE id = ?", (int(evaluation["model_b_id"]),)
        ).fetchone()
        if model_a is None or model_b is None:
            raise RuntimeError("evaluation references a missing model")
        old_a, old_b = float(model_a["elo"]), float(model_b["elo"])
        expected_a = 1.0 / (1.0 + 10.0 ** ((old_b - old_a) / 400.0))
        delta = float(elo_k) * (score - expected_a)
        new_a, new_b = old_a + delta, old_b - delta
        games = int(summary.get("games", 0))
        wins = int(summary.get("model_a_wins", 0))
        losses = int(summary.get("model_a_losses", 0))
        draws = int(summary.get("draws", 0))
        self.connection.execute(
            """UPDATE models SET elo = ?, rated_games = rated_games + ?,
               rated_wins = rated_wins + ?, rated_draws = rated_draws + ?,
               rated_losses = rated_losses + ?, updated_at = ? WHERE id = ?""",
            (new_a, games, wins, draws, losses, _utc_now(), int(model_a["id"])),
        )
        self.connection.execute(
            """UPDATE models SET elo = ?, rated_games = rated_games + ?,
               rated_wins = rated_wins + ?, rated_draws = rated_draws + ?,
               rated_losses = rated_losses + ?, updated_at = ? WHERE id = ?""",
            (new_b, games, losses, draws, wins, _utc_now(), int(model_b["id"])),
        )
        self.connection.executemany(
            """INSERT INTO rating_history(
                   evaluation_id, model_id, old_elo, new_elo, expected_score,
                   actual_score, created_at
               ) VALUES(?, ?, ?, ?, ?, ?, ?)""",
            [
                (evaluation_id, int(model_a["id"]), old_a, new_a, expected_a, score, _utc_now()),
                (evaluation_id, int(model_b["id"]), old_b, new_b, 1.0 - expected_a, 1.0 - score, _utc_now()),
            ],
        )
        self.connection.commit()

    def invalidate_evaluation(
        self,
        evaluation_id: int,
        summary: Mapping[str, Any],
    ) -> None:
        """Finish an infrastructure-invalid gate without changing ratings."""
        exists = self.connection.execute(
            "SELECT 1 FROM evaluations WHERE id = ?", (evaluation_id,)
        ).fetchone()
        if exists is None:
            raise KeyError(f"unknown evaluation {evaluation_id}")
        payload = dict(summary)
        payload["valid"] = 0.0
        payload["invalid_evaluations"] = max(
            1.0, float(payload.get("invalid_evaluations", 0.0))
        )
        self.connection.execute(
            """UPDATE evaluations SET score_a = NULL, lower_95 = NULL,
               summary_json = ?, finished_at = ? WHERE id = ?""",
            (_json(payload), _utc_now(), evaluation_id),
        )
        self.connection.commit()

    def record_metric_row(
        self,
        run_id: int,
        round_id: int | None,
        stage: str,
        row: Mapping[str, Any],
    ) -> None:
        iteration = int(row.get("iteration", 0))
        values = []
        for name, value in row.items():
            if name == "iteration" or isinstance(value, bool):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                values.append((run_id, round_id, stage, iteration, str(name), number, _utc_now()))
        self.connection.executemany(
            """INSERT INTO training_metrics(
                   run_id, round_id, stage, iteration, metric, value, created_at
               ) VALUES(?, ?, ?, ?, ?, ?, ?)""",
            values,
        )
        self._pending_metric_rows += len(values)
        if self._pending_metric_rows >= 500:
            self.connection.commit()
            self._pending_metric_rows = 0

    def flush(self) -> None:
        self.connection.commit()
        self._pending_metric_rows = 0
