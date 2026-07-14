"""SQLite store for episode ground-truth navigation data.

JSON remains an optional export for inspection / LLM drafting.
The DB is the scalable system of record for querying by step/object/edge.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _dumps(value: Any) -> str:
    return json.dumps(value, default=_json_default)


def _json_default(obj: Any):
    if hasattr(obj, 'item'):
        return obj.item()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f'Object of type {type(obj)} is not JSON serializable')


def _loads(text: Optional[str], default=None):
    if text is None:
        return default
    return json.loads(text)


def _xyz(value) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if value is None:
        return None, None, None
    seq = list(value)
    return (
        float(seq[0]) if seq[0] is not None else None,
        float(seq[1]) if seq[1] is not None else None,
        float(seq[2]) if seq[2] is not None else None,
    )


class EpisodeStore:
    """Persist and reload episode GT in SQLite."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute('PRAGMA foreign_keys = ON')
        self._create_tables()

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _create_tables(self):
        cur = self._conn.cursor()
        cur.executescript(
            '''
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                scene TEXT NOT NULL,
                environment TEXT NOT NULL,
                thresholds_json TEXT NOT NULL,
                movement_constant REAL,
                object_state_track_json TEXT,
                displacement_events_json TEXT NOT NULL DEFAULT '[]',
                world_layout_json TEXT,
                passage_state_json TEXT,
                region_trajectory_json TEXT,
                episode_meta_json TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS steps (
                episode_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                image_path TEXT,
                action TEXT,
                degrees REAL,
                agent_pos_x REAL,
                agent_pos_y REAL,
                agent_pos_z REAL,
                agent_rot_x REAL,
                agent_rot_y REAL,
                agent_rot_z REAL,
                PRIMARY KEY (episode_id, step),
                FOREIGN KEY (episode_id) REFERENCES episodes(episode_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS step_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                object_id TEXT NOT NULL,
                visibility TEXT NOT NULL,
                category TEXT,
                pos_x REAL, pos_y REAL, pos_z REAL,
                local_pos_x REAL, local_pos_y REAL, local_pos_z REAL,
                last_seen_step INTEGER,
                payload_json TEXT NOT NULL,
                FOREIGN KEY (episode_id, step) REFERENCES steps(episode_id, step) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                edge_type TEXT NOT NULL,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                distance_metric REAL,
                distance_label TEXT,
                visible INTEGER,
                inferred INTEGER,
                angle_x TEXT,
                angle_y TEXT,
                angle_z TEXT,
                last_seen INTEGER,
                payload_json TEXT NOT NULL,
                FOREIGN KEY (episode_id, step) REFERENCES steps(episode_id, step) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS route_landmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                object_id TEXT NOT NULL,
                category TEXT,
                distance_label TEXT,
                ord INTEGER NOT NULL,
                FOREIGN KEY (episode_id) REFERENCES episodes(episode_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS route_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                action TEXT,
                degrees REAL,
                ord INTEGER NOT NULL,
                FOREIGN KEY (episode_id) REFERENCES episodes(episode_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_step_objects_lookup
                ON step_objects(episode_id, step, visibility);
            CREATE INDEX IF NOT EXISTS idx_edges_lookup
                ON edges(episode_id, step, edge_type);
            CREATE INDEX IF NOT EXISTS idx_edges_target
                ON edges(episode_id, target);
            '''
        )
        self._conn.commit()
        self._ensure_episode_columns()

    def _ensure_episode_columns(self):
        """Add survey / meta columns on older DB files."""
        cols = {r[1] for r in self._conn.execute('PRAGMA table_info(episodes)')}
        extras = {
            'world_layout_json': 'TEXT',
            'passage_state_json': 'TEXT',
            'region_trajectory_json': 'TEXT',
            'episode_meta_json': 'TEXT',
        }
        for name, typedef in extras.items():
            if name not in cols:
                self._conn.execute(f'ALTER TABLE episodes ADD COLUMN {name} {typedef}')
        self._conn.commit()

    def save_episode(
        self,
        episode: dict,
        episode_id: str,
        environment: str = 'ai2thor',
        replace: bool = True,
    ) -> str:
        """Insert an episode dict. Replaces an existing id when replace=True."""
        if replace:
            self._conn.execute('DELETE FROM episodes WHERE episode_id = ?', (episode_id,))

        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            '''
            INSERT INTO episodes (
                episode_id, scene, environment, thresholds_json, movement_constant,
                object_state_track_json, displacement_events_json,
                world_layout_json, passage_state_json, region_trajectory_json,
                episode_meta_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                episode_id,
                episode.get('scene', 'unknown'),
                environment,
                _dumps(episode.get('thresholds', {})),
                episode.get('movement_constant'),
                _dumps(episode.get('object_state_track'))
                if episode.get('object_state_track') is not None
                else None,
                _dumps(episode.get('displacement_events', [])),
                _dumps(episode.get('world_layout'))
                if episode.get('world_layout') is not None
                else None,
                _dumps(episode.get('passage_state', [])),
                _dumps(episode.get('region_trajectory', [])),
                _dumps(episode.get('episode_meta', {})),
                now,
            ),
        )

        for step in episode.get('steps', []):
            self._insert_step(episode_id, step)

        for ord_i, lm in enumerate(episode.get('route', {}).get('landmarks', [])):
            self._conn.execute(
                '''
                INSERT INTO route_landmarks
                    (episode_id, step, object_id, category, distance_label, ord)
                VALUES (?, ?, ?, ?, ?, ?)
                ''',
                (
                    episode_id,
                    lm.get('step'),
                    lm.get('object_id'),
                    lm.get('category'),
                    lm.get('distance_label'),
                    ord_i,
                ),
            )

        for ord_i, turn in enumerate(episode.get('route', {}).get('turns', [])):
            self._conn.execute(
                '''
                INSERT INTO route_turns
                    (episode_id, step, action, degrees, ord)
                VALUES (?, ?, ?, ?, ?)
                ''',
                (
                    episode_id,
                    turn.get('step'),
                    turn.get('action'),
                    turn.get('degrees'),
                    ord_i,
                ),
            )

        self._conn.commit()
        return episode_id

    def _insert_step(self, episode_id: str, step: dict):
        agent = step.get('agent', {})
        px, py, pz = _xyz(agent.get('position'))
        rx, ry, rz = _xyz(agent.get('rotation'))
        step_idx = step['step']

        self._conn.execute(
            '''
            INSERT INTO steps (
                episode_id, step, image_path, action, degrees,
                agent_pos_x, agent_pos_y, agent_pos_z,
                agent_rot_x, agent_rot_y, agent_rot_z
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                episode_id,
                step_idx,
                step.get('image_path'),
                step.get('action'),
                step.get('degrees'),
                px, py, pz,
                rx, ry, rz,
            ),
        )

        for obj_id, obj in step.get('visible_objects', {}).items():
            self._insert_object(episode_id, step_idx, obj_id, 'visible', obj)

        for obj_id, obj in step.get('non_visible_objects', {}).items():
            self._insert_object(episode_id, step_idx, obj_id, 'non_visible', obj)

        for edge_type in (
            'edges_egocentric',
            'edges_allocentric',
            'edges_object_frame',
            'edges_inferred',
        ):
            short = edge_type.replace('edges_', '')
            for edge in step.get(edge_type, []):
                self._insert_edge(episode_id, step_idx, short, edge)

    def _insert_object(
        self, episode_id: str, step_idx: int, object_id: str, visibility: str, obj: dict
    ):
        px, py, pz = _xyz(obj.get('position'))
        lx, ly, lz = _xyz(obj.get('local_position'))
        self._conn.execute(
            '''
            INSERT INTO step_objects (
                episode_id, step, object_id, visibility, category,
                pos_x, pos_y, pos_z, local_pos_x, local_pos_y, local_pos_z,
                last_seen_step, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                episode_id,
                step_idx,
                object_id,
                visibility,
                obj.get('category'),
                px, py, pz,
                lx, ly, lz,
                obj.get('last_seen_step'),
                _dumps(obj),
            ),
        )

    def _insert_edge(self, episode_id: str, step_idx: int, edge_type: str, edge: dict):
        angles = edge.get('angle_relation') or [None, None, None]
        if len(angles) < 3:
            angles = list(angles) + [None] * (3 - len(angles))
        self._conn.execute(
            '''
            INSERT INTO edges (
                episode_id, step, edge_type, source, target,
                distance_metric, distance_label, visible, inferred,
                angle_x, angle_y, angle_z, last_seen, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                episode_id,
                step_idx,
                edge_type,
                edge.get('source'),
                edge.get('target'),
                edge.get('distance_metric'),
                edge.get('distance_label'),
                1 if edge.get('visible') else 0,
                1 if edge.get('inferred') else 0,
                angles[0],
                angles[1],
                angles[2],
                edge.get('last_seen'),
                _dumps(edge),
            ),
        )

    def list_episodes(self) -> list[dict]:
        rows = self._conn.execute(
            '''
            SELECT episode_id, scene, environment, created_at,
                   (SELECT COUNT(*) FROM steps s WHERE s.episode_id = e.episode_id) AS n_steps
            FROM episodes e
            ORDER BY created_at
            '''
        ).fetchall()
        return [dict(r) for r in rows]

    def load_episode(self, episode_id: str) -> dict:
        ep = self._conn.execute(
            'SELECT * FROM episodes WHERE episode_id = ?', (episode_id,)
        ).fetchone()
        if ep is None:
            raise KeyError(f'Unknown episode_id: {episode_id}')

        steps_rows = self._conn.execute(
            'SELECT * FROM steps WHERE episode_id = ? ORDER BY step',
            (episode_id,),
        ).fetchall()

        steps = []
        trajectory = []
        actions = []
        for row in steps_rows:
            step_idx = row['step']
            step_dict = {
                'step': step_idx,
                'image_path': row['image_path'],
                'action': row['action'],
                'degrees': row['degrees'],
                'agent': {
                    'position': [row['agent_pos_x'], row['agent_pos_y'], row['agent_pos_z']],
                    'rotation': [row['agent_rot_x'], row['agent_rot_y'], row['agent_rot_z']],
                },
                'visible_objects': self._load_objects(episode_id, step_idx, 'visible'),
                'non_visible_objects': self._load_objects(episode_id, step_idx, 'non_visible'),
                'edges_egocentric': self._load_edges(episode_id, step_idx, 'egocentric'),
                'edges_allocentric': self._load_edges(episode_id, step_idx, 'allocentric'),
                'edges_object_frame': self._load_edges(episode_id, step_idx, 'object_frame'),
                'edges_inferred': self._load_edges(episode_id, step_idx, 'inferred'),
            }
            steps.append(step_dict)
            trajectory.append({
                'step': step_idx,
                'position': step_dict['agent']['position'],
                'rotation': step_dict['agent']['rotation'],
                'image_path': row['image_path'],
            })
            actions.append({
                'step': step_idx,
                'action': row['action'],
                'degrees': row['degrees'],
            })

        landmarks = [
            {
                'step': r['step'],
                'object_id': r['object_id'],
                'category': r['category'],
                'distance_label': r['distance_label'],
            }
            for r in self._conn.execute(
                '''
                SELECT step, object_id, category, distance_label
                FROM route_landmarks WHERE episode_id = ? ORDER BY ord
                ''',
                (episode_id,),
            ).fetchall()
        ]
        turns = [
            {'step': r['step'], 'action': r['action'], 'degrees': r['degrees']}
            for r in self._conn.execute(
                '''
                SELECT step, action, degrees
                FROM route_turns WHERE episode_id = ? ORDER BY ord
                ''',
                (episode_id,),
            ).fetchall()
        ]

        return {
            'scene': ep['scene'],
            'thresholds': _loads(ep['thresholds_json'], {}),
            'movement_constant': ep['movement_constant'],
            'agent_trajectory': trajectory,
            'agent_actions': actions,
            'route': {'landmarks': landmarks, 'turns': turns},
            'object_state_track': _loads(ep['object_state_track_json'], None),
            'displacement_events': _loads(ep['displacement_events_json'], []),
            'world_layout': _loads(ep['world_layout_json'], None) if 'world_layout_json' in ep.keys() else None,
            'passage_state': _loads(ep['passage_state_json'], []) if 'passage_state_json' in ep.keys() else [],
            'region_trajectory': (
                _loads(ep['region_trajectory_json'], [])
                if 'region_trajectory_json' in ep.keys()
                else []
            ),
            'episode_meta': _loads(ep['episode_meta_json'], {}) if 'episode_meta_json' in ep.keys() else {},
            'steps': steps,
        }

    def _load_objects(self, episode_id: str, step_idx: int, visibility: str) -> dict:
        rows = self._conn.execute(
            '''
            SELECT object_id, payload_json FROM step_objects
            WHERE episode_id = ? AND step = ? AND visibility = ?
            ''',
            (episode_id, step_idx, visibility),
        ).fetchall()
        return {r['object_id']: json.loads(r['payload_json']) for r in rows}

    def _load_edges(self, episode_id: str, step_idx: int, edge_type: str) -> list:
        rows = self._conn.execute(
            '''
            SELECT payload_json FROM edges
            WHERE episode_id = ? AND step = ? AND edge_type = ?
            ORDER BY id
            ''',
            (episode_id, step_idx, edge_type),
        ).fetchall()
        return [json.loads(r['payload_json']) for r in rows]

    def get_edges(
        self,
        episode_id: str,
        step: Optional[int] = None,
        edge_type: Optional[str] = None,
        target: Optional[str] = None,
    ) -> list[dict]:
        """Query edges without loading the full episode (for scaling)."""
        clauses = ['episode_id = ?']
        params: list[Any] = [episode_id]
        if step is not None:
            clauses.append('step = ?')
            params.append(step)
        if edge_type is not None:
            clauses.append('edge_type = ?')
            params.append(edge_type)
        if target is not None:
            clauses.append('target = ?')
            params.append(target)

        sql = f'''
            SELECT step, edge_type, source, target, distance_metric, distance_label,
                   visible, inferred, angle_x, angle_y, angle_z, last_seen, payload_json
            FROM edges
            WHERE {' AND '.join(clauses)}
            ORDER BY step, id
        '''
        rows = self._conn.execute(sql, params).fetchall()
        out = []
        for r in rows:
            item = json.loads(r['payload_json'])
            item['step'] = r['step']
            item['edge_type'] = r['edge_type']
            out.append(item)
        return out
