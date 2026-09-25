"""Load episode GT from SQLite EpisodeStore or JSON export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def load_episode_from_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return json.load(f)


def load_episode_from_db(db_path: str | Path, episode_id: str) -> dict[str, Any]:
    from cm_benchmark.storage.episode_store import EpisodeStore

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


def scene_id_from_nav_json(path: str | Path) -> str:
    """``nav_house_007514.json`` → ``house_007514``; else the parent folder name."""
    path = Path(path)
    stem = path.stem
    if stem.startswith('nav_') and len(stem) > len('nav_'):
        return stem[len('nav_'):]
    return path.parent.name or stem


def list_nav_episode_jsons(folder: str | Path) -> list[Path]:
    """Episode JSON files written by ``ai2thor_nav_generator``.

    ``folder`` is that command's ``--output_path`` root::

        nav_data/house_007514/nav_house_007514.json
        nav_data/house_001030/nav_house_001030.json

    A single scene folder or one ``nav_<scene>.json`` file is also accepted.
    """
    path = Path(folder)
    if path.is_file():
        if path.suffix != '.json':
            raise FileNotFoundError(f'Expected a nav JSON file or folder, got {path}')
        return [path.resolve()]
    if not path.is_dir():
        raise NotADirectoryError(path)

    nested = sorted(
        child
        for child in path.glob('*/*.json')
        if child.is_file() and child.name.startswith('nav_')
    )
    if nested:
        return [child.resolve() for child in nested]

    direct = sorted(
        child
        for child in path.glob('*.json')
        if child.is_file() and child.name.startswith('nav_')
    )
    if direct:
        return [child.resolve() for child in direct]

    raise FileNotFoundError(
        f'No nav_<scene>.json under {path}. Pass the nav generator '
        f'--output_path (nav_data/<scene_id>/nav_<scene_id>.json).'
    )


def items_output_file(output_root: str | Path, scene_name: str) -> Path:
    """``<output_root>/<scene_id>/items_<scene_id>.json``."""
    return Path(output_root) / scene_name / f'items_{scene_name}.json'


def list_item_jsons(folder: str | Path) -> list[Path]:
    """Item JSON files written by ``generate_items``.

    ``folder`` is that command's ``--output_path`` root::

        items/house_007514/items_house_007514.json
        items/house_001030/items_house_001030.json

    A single scene folder or one ``items_<scene>.json`` file is also accepted.
    """
    path = Path(folder)
    if path.is_file():
        if path.suffix != '.json':
            raise FileNotFoundError(f'Expected an items JSON file or folder, got {path}')
        return [path.resolve()]
    if not path.is_dir():
        raise NotADirectoryError(path)

    nested = sorted(
        child
        for child in path.glob('*/*.json')
        if child.is_file() and child.name.startswith('items_')
    )
    if nested:
        return [child.resolve() for child in nested]

    direct = sorted(
        child
        for child in path.glob('*.json')
        if child.is_file() and child.name.startswith('items_')
    )
    if direct:
        return [child.resolve() for child in direct]

    raise FileNotFoundError(
        f'No items_<scene>.json under {path}. Pass generate_items '
        f'--output_path (items/<scene_id>/items_<scene_id>.json).'
    )


def write_items(items: list[dict[str, Any]], output_path: str | Path) -> Path:
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
