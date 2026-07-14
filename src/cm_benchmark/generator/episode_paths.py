"""Discover SPOC / AI2-THOR episode files in a collection folder."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional


# Logical keys → filename stem prefix before "-{scene}.csv|.json"
DEFAULT_STEMS = {
    'navigation': 'navigation',
    'objects': 'objects',
    'object_state': 'object_state',
    'displacement_events': 'displacement_events',
    'passage_state': 'passage_state',
    'region_trajectory': 'region_trajectory',
    'world_layout': 'world_layout',
    'episode_meta': 'episode_meta',
    'doors': 'doors',
}

OPTIONAL_KEYS = {
    'object_state',
    'displacement_events',
    'passage_state',
    'region_trajectory',
    'world_layout',
    'episode_meta',
    'doors',
}

REQUIRED_KEYS = {'navigation', 'objects'}


def _scene_from_name(name: str) -> Optional[str]:
    """Parse house_XXXXXX (or similar) from episode_meta-house_001030.json."""
    m = re.search(r'(house_\d+)', name)
    if m:
        return m.group(1)
    m = re.search(r'-(.+)\.(csv|json)$', name)
    if m:
        return m.group(1)
    return None


def discover_scene_id(folder: Path) -> str:
    """Prefer episode_meta file; else first matching navigation-* file."""
    metas = sorted(folder.glob('episode_meta-*.json'))
    if metas:
        scene = _scene_from_name(metas[0].name)
        if scene:
            return scene
        with open(metas[0]) as f:
            meta = json.load(f)
        if meta.get('scene_id'):
            return meta['scene_id']

    navs = sorted(folder.glob('navigation-*.csv'))
    if not navs:
        raise FileNotFoundError(f'No navigation-*.csv or episode_meta-*.json in {folder}')
    scene = _scene_from_name(navs[0].name)
    if not scene:
        raise FileNotFoundError(f'Could not parse scene id from {navs[0].name}')
    return scene


def resolve_episode_paths(
    folder: str | Path,
    scene_id: Optional[str] = None,
    file_overrides: Optional[dict[str, str]] = None,
) -> dict:
    """
    Map logical file roles to absolute paths.

    Parameters
    ----------
    folder : collection run directory (CSVs + JSON, not images/)
    scene_id : e.g. house_001030; auto-detected if omitted
    file_overrides : optional absolute/relative basenames per key
        e.g. {"navigation": "navigation-custom.csv", "objects": "/abs/objects.csv"}
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(folder)

    overrides = file_overrides or {}
    scene = scene_id or discover_scene_id(folder)

    paths = {'folder': folder, 'scene_id': scene}
    meta_path = folder / f'episode_meta-{scene}.json'
    if 'episode_meta' in overrides:
        meta_path = Path(overrides['episode_meta'])
        if not meta_path.is_absolute():
            meta_path = folder / meta_path

    episode_meta = {}
    if meta_path.is_file():
        with open(meta_path) as f:
            episode_meta = json.load(f)
        paths['episode_meta'] = meta_path
        scene = episode_meta.get('scene_id', scene)
        paths['scene_id'] = scene

    for key, stem in DEFAULT_STEMS.items():
        if key == 'episode_meta' and 'episode_meta' in paths:
            continue
        if key in overrides:
            p = Path(overrides[key])
            if not p.is_absolute():
                p = folder / p
        else:
            ext = 'json' if key in ('world_layout', 'episode_meta') else 'csv'
            p = folder / f'{stem}-{scene}.{ext}'

        if p.is_file():
            paths[key] = p
        elif key in REQUIRED_KEYS:
            raise FileNotFoundError(f'Required file missing for {key}: {p}')

    paths['episode_meta_data'] = episode_meta
    return paths
