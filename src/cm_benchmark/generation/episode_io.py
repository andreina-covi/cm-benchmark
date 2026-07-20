"""Load episode GT from SQLite EpisodeStore or JSON export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from cm_benchmark.storage.episode_store import EpisodeStore


def load_episode_from_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return json.load(f)


def load_episode_from_db(db_path: str | Path, episode_id: str) -> dict[str, Any]:
    with EpisodeStore(db_path) as store:
        episode = store.load_episode(episode_id)
    if episode is None:
        raise KeyError(f'episode_id not found: {episode_id}')
    return episode


def load_episode(
    *,
    db_path: Optional[str | Path] = None,
    episode_id: Optional[str] = None,
    episode_json: Optional[str | Path] = None,
) -> dict[str, Any]:
    if episode_json is not None:
        return load_episode_from_json(episode_json)
    if db_path is not None and episode_id is not None:
        return load_episode_from_db(db_path, episode_id)
    raise ValueError('Provide episode_json or both db_path and episode_id')


def write_draft_items(items: list[dict[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'n_items': len(items),
        'items': items,
    }
    with path.open('w') as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def scene_id_of(episode: dict) -> str:
    meta = episode.get('episode_meta') or {}
    return meta.get('scene_id') or episode.get('scene') or 'unknown'


def environment_of(episode: dict) -> str:
    meta = episode.get('episode_meta') or {}
    return meta.get('environment') or 'ai2thor'
