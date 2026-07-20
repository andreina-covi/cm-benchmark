"""Discover SPOC / AI2-THOR episode files in a collection folder.

SPOC episode root layout (current)::

    <timestamp>/
      images/img_<t>.png
      annotations/
        navigation-house_XXXXXX.csv
        objects-house_XXXXXX.csv
        ...
        episode_meta-house_XXXXXX.json

``--csv_path_folder`` may be either the episode root or the ``annotations/``
directory. Legacy flat folders (CSVs next to images) still work.
"""

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

# Match SPOC structural exclusions (walls/floors are not spatial targets).
STRUCTURAL_OBJECT_TYPES = frozenset(
    {'Wall', 'Floor', 'Ceiling', 'Doorway', 'Doorframe', 'Room', 'Window'}
)
_STRUCTURAL_OID_PREFIXES = (
    'Wall|',
    'Floor|',
    'Ceiling|',
    'Doorway|',
    'Doorframe|',
    'Room|',
    'Window|',
)


def is_structural_object(obj_type=None, object_id=None) -> bool:
    """True for walls/floors/ceilings/etc. that should not enter FOV / edges."""
    if obj_type is not None and obj_type in STRUCTURAL_OBJECT_TYPES:
        return True
    if object_id is not None and str(object_id).startswith(_STRUCTURAL_OID_PREFIXES):
        return True
    return False


def _scene_from_name(name: str) -> Optional[str]:
    """Parse house_XXXXXX (or similar) from episode_meta-house_001030.json."""
    m = re.search(r'(house_\d+)', name)
    if m:
        return m.group(1)
    m = re.search(r'-(.+)\.(csv|json)$', name)
    if m:
        return m.group(1)
    return None


def _looks_like_annotations_dir(folder: Path) -> bool:
    return (
        any(folder.glob('navigation-*.csv'))
        or any(folder.glob('episode_meta-*.json'))
    )


def resolve_annotations_dir(folder: str | Path) -> tuple[Path, Path]:
    """
    Return (annotations_dir, episode_root).

    Accepts:
    - episode root with ``annotations/`` sibling to ``images/``
    - the ``annotations/`` directory itself
    - legacy flat folder (CSVs in place → annotations_dir == episode_root)
    """
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise NotADirectoryError(folder)

    nested = folder / 'annotations'
    if nested.is_dir() and _looks_like_annotations_dir(nested):
        return nested, folder

    if _looks_like_annotations_dir(folder):
        # Prefer parent as episode root when this folder is named annotations/
        if folder.name == 'annotations' and folder.parent.is_dir():
            return folder, folder.parent
        return folder, folder

    raise FileNotFoundError(
        f'No navigation-*.csv / episode_meta-*.json in {folder} '
        f'(or {folder / "annotations"})'
    )


def resolve_images_dir(
    episode_root: Path,
    annotations_dir: Path,
    episode_meta: Optional[dict] = None,
) -> Path:
    """Locate RGB frames directory (sibling ``images/`` by default)."""
    meta = episode_meta or {}
    rel = meta.get('images_dir') or 'images'
    candidates = [
        episode_root / rel,
        annotations_dir.parent / rel,
        episode_root / 'images',
        annotations_dir / 'images',
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand.resolve()
    # Fall back to conventional sibling even if missing (path rewrite still useful)
    return (episode_root / 'images').resolve()


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
        raise FileNotFoundError(
            f'No navigation-*.csv or episode_meta-*.json in {folder}'
        )
    scene = _scene_from_name(navs[0].name)
    if not scene:
        raise FileNotFoundError(f'Could not parse scene id from {navs[0].name}')
    return scene


def resolve_image_path(
    raw_path: Optional[str],
    *,
    images_dir: Optional[Path] = None,
    timestep: Optional[int] = None,
) -> Optional[str]:
    """Return an absolute image path; repair relative / missing paths when possible."""
    if raw_path is None or (isinstance(raw_path, float) and str(raw_path) == 'nan'):
        raw_path = ''
    text = str(raw_path).strip()
    if text and text.lower() != 'nan':
        p = Path(text)
        if p.is_file():
            return str(p.resolve())
        if not p.is_absolute() and images_dir is not None:
            candidate = images_dir / p.name if p.name else images_dir / text
            if candidate.is_file():
                return str(candidate.resolve())
            candidate = images_dir / text
            if candidate.is_file():
                return str(candidate.resolve())
        # Absolute but missing: try basename under images_dir
        if images_dir is not None and p.name:
            candidate = images_dir / p.name
            if candidate.is_file():
                return str(candidate.resolve())

    if images_dir is not None and timestep is not None:
        candidate = images_dir / f'img_{int(timestep)}.png'
        if candidate.is_file():
            return str(candidate.resolve())

    return text or None


def resolve_episode_paths(
    folder: str | Path,
    scene_id: Optional[str] = None,
    file_overrides: Optional[dict[str, str]] = None,
) -> dict:
    """
    Map logical file roles to absolute paths.

    Parameters
    ----------
    folder : episode root or ``annotations/`` directory
    scene_id : e.g. house_001030; auto-detected if omitted
    file_overrides : optional absolute/relative basenames per key
        e.g. {"navigation": "navigation-custom.csv", "objects": "/abs/objects.csv"}
    """
    annotations_dir, episode_root = resolve_annotations_dir(folder)
    overrides = file_overrides or {}
    scene = scene_id or discover_scene_id(annotations_dir)

    paths = {
        'folder': annotations_dir,
        'annotations_dir': annotations_dir,
        'episode_root': episode_root,
        'scene_id': scene,
    }

    meta_path = annotations_dir / f'episode_meta-{scene}.json'
    if 'episode_meta' in overrides:
        meta_path = Path(overrides['episode_meta'])
        if not meta_path.is_absolute():
            meta_path = annotations_dir / meta_path

    episode_meta = {}
    if meta_path.is_file():
        with open(meta_path) as f:
            episode_meta = json.load(f)
        paths['episode_meta'] = meta_path
        scene = episode_meta.get('scene_id', scene)
        paths['scene_id'] = scene

    images_dir = resolve_images_dir(episode_root, annotations_dir, episode_meta)
    paths['images_dir'] = images_dir

    for key, stem in DEFAULT_STEMS.items():
        if key == 'episode_meta' and 'episode_meta' in paths:
            continue
        if key in overrides:
            p = Path(overrides[key])
            if not p.is_absolute():
                p = annotations_dir / p
        else:
            ext = 'json' if key in ('world_layout', 'episode_meta') else 'csv'
            p = annotations_dir / f'{stem}-{scene}.{ext}'

        if p.is_file():
            paths[key] = p
        elif key in REQUIRED_KEYS:
            raise FileNotFoundError(f'Required file missing for {key}: {p}')

    paths['episode_meta_data'] = episode_meta
    return paths
