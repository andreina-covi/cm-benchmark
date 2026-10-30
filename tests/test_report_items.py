"""Generation item-count report."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load(name: str = 'report_items.py'):
    path = REPO / 'scripts' / name
    spec = importlib.util.spec_from_file_location(name.replace('.py', ''), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sample_items(tmp_path: Path) -> Path:
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
    return tmp_path


def test_report_totals_by_scene_and_construct(tmp_path: Path):
    report_items = _load()
    root = _sample_items(tmp_path)

    text = report_items.report(report_items._load(root), items_dir=root)
    assert 'TOTAL (concise)' in text
    assert 'house_a' in text and 'house_b' in text
    assert 'egocentric_encoding' in text
    assert 'route_knowledge' in text
    # Verbose partner must not inflate ok count (2 concise ok, not 3).
    assert '  ok                2\n' in text
    assert 'TOTAL                            2     1            1     4' in text

    out = tmp_path / 'summary.txt'
    json_out = tmp_path / 'summary.json'
    tables = tmp_path / 'tables'
    assert (
        report_items.main(
            [
                str(root),
                '-o',
                str(out),
                '--json-out',
                str(json_out),
                '--tables-dir',
                str(tables),
            ]
        )
        == 0
    )
    assert out.is_file()
    assert 'By scene (concise)' in out.read_text()
    blob = json.loads(json_out.read_text())
    assert blob['sections'][0]['kind'] == 'items'
    assert blob['sections'][0]['totals']['ok'] == 2
    assert (tables / 'by_scene.csv').is_file()
    assert (tables / 'by_construct.csv').is_file()
    assert (tables / 'labels.csv').is_file()
    assert (tables / 'totals.csv').is_file()


def test_plot_items_report_from_json(tmp_path: Path):
    import pytest

    pytest.importorskip('matplotlib')
    report_items = _load()
    plot = _load('plot_review_report.py')
    root = _sample_items(tmp_path)
    data = report_items.collect_items_summary(report_items._load(root), items_dir=root)
    json_path = tmp_path / 'items_summary.json'
    json_path.write_text(json.dumps({'sections': [data]}))
    out = tmp_path / 'plots'
    assert plot.main([str(json_path), '-o', str(out)]) == 0
    assert (out / 'items_totals.png').is_file()
    assert (out / 'items_by_construct.png').is_file()
    assert (out / 'items_ok_share.png').is_file()
    assert (out / 'items_by_scene_top.png').is_file()
