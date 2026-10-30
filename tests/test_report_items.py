"""Generation item-count report."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load():
    path = REPO / 'scripts' / 'report_items.py'
    spec = importlib.util.spec_from_file_location('report_items', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_report_totals_by_scene_and_construct(tmp_path: Path):
    report_items = _load()
    scene_a = tmp_path / 'house_a'
    scene_b = tmp_path / 'house_b'
    scene_a.mkdir()
    scene_b.mkdir()
    (scene_a / 'items_house_a.json').write_text(
        json.dumps(
            {
                'n_items': 3,
                'items': [
                    {
                        'item_id': 'a1',
                        'construct': 'egocentric_encoding',
                        'scene_id': 'house_a',
                        'status': 'ok',
                        'question_style': 'concise',
                        'answer': 'A',
                        'options': {'A': 'left', 'B': 'right'},
                    },
                    {
                        'item_id': 'a1v',
                        'construct': 'egocentric_encoding',
                        'scene_id': 'house_a',
                        'status': 'ok',
                        'question_style': 'verbose',
                        'answer': 'A',
                        'options': {'A': 'left', 'B': 'right'},
                        'paired_item_id': 'a1',
                    },
                    {
                        'item_id': 'a2',
                        'construct': 'route_knowledge',
                        'scene_id': 'house_a',
                        'status': 'unsupported',
                        'question_style': 'concise',
                        'answer': '',
                        'options': {},
                    },
                ],
            }
        )
    )
    (scene_b / 'items_house_b.json').write_text(
        json.dumps(
            {
                'items': [
                    {
                        'item_id': 'b1',
                        'construct': 'egocentric_encoding',
                        'scene_id': 'house_b',
                        'status': 'ok',
                        'question_style': 'concise',
                        'answer': 'B',
                        'options': {'A': 'left', 'B': 'right'},
                    },
                    {
                        'item_id': 'b2',
                        'construct': 'spatial_working_memory',
                        'scene_id': 'house_b',
                        'status': 'thin',
                        'question_style': 'concise',
                    },
                ]
            }
        )
    )

    text = report_items.report(report_items._load(tmp_path), items_dir=tmp_path)
    assert 'TOTAL (concise)' in text
    assert 'house_a' in text and 'house_b' in text
    assert 'egocentric_encoding' in text
    assert 'route_knowledge' in text
    # Verbose partner must not inflate ok count (2 concise ok, not 3).
    assert '  ok                2\n' in text
    assert 'TOTAL                            2     1            1     4' in text

    out = tmp_path / 'summary.txt'
    assert report_items.main([str(tmp_path), '-o', str(out)]) == 0
    assert out.is_file()
    assert 'By scene (concise)' in out.read_text()
