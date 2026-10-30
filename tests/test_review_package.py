"""Review package sampling, assignment, and CSV analysis."""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONSTRUCTS = (
    'egocentric_encoding',
    'allocentric_encoding',
    'spatial_working_memory',
    'invisible_displacement',
    'spatial_updating',
    'perspective_taking',
    'route_knowledge',
    'survey_based_route_planning',
)
NAV = {'route_knowledge', 'survey_based_route_planning'}


def _load(name: str):
    path = REPO / 'scripts' / name
    spec = importlib.util.spec_from_file_location(name.replace('.py', ''), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _item(construct: str, scene: str, n: int) -> dict:
    nav = construct in NAV
    return {
        'item_id': f'{scene}_{construct}_{n}',
        'construct': construct,
        'scene_id': scene,
        'status': 'ok',
        'question_style': 'concise',
        'question': f'Where is {construct} {n} in {scene}?',
        'options': {} if nav else {'A': 'left', 'B': 'right', 'C': 'ahead', 'D': 'behind'},
        'answer': 'move_ahead' if nav else 'B',
        'image_paths': [],
    }


def _pool(per_scene: int = 10, scenes: tuple[str, ...] = ('s0', 's1', 's2')) -> list[dict]:
    items = []
    for construct in CONSTRUCTS:
        for scene in scenes:
            for n in range(per_scene):
                items.append(_item(construct, scene, n))
    return items


def _payload(html: str) -> list[dict]:
    marker = 'const PAYLOAD = '
    start = html.index(marker) + len(marker)
    end = html.index(';\nconst FLAGS', start)
    return json.loads(html[start:end])


def test_circle_pairs_balance_and_rotate():
    build = _load('build_review_package.py')
    for offset in range(8):
        pairs = build.circle_pairs(10, 25, offset)
        assert len(pairs) == 25
        assert len(set(pairs)) == 25
        load = Counter(r for pair in pairs for r in pair)
        assert load.keys() == set(range(10))
        assert set(load.values()) == {5}
        assert all(a != b for a, b in pairs)
    assert build.circle_pairs(10, 25, 0)[0] != build.circle_pairs(10, 25, 1)[0]


def test_stratified_sample_spreads_scenes():
    build = _load('build_review_package.py')
    import random

    pool = []
    for scene, n in (('a', 10), ('b', 10), ('c', 10)):
        for i in range(n):
            pool.append(_item('egocentric_encoding', scene, i))
    picked = build.take_stratified(pool, 25, random.Random(0))
    counts = Counter(item['scene_id'] for item in picked)
    assert sum(counts.values()) == 25
    assert max(counts.values()) - min(counts.values()) <= 1


def test_package_assigns_pairs_and_shared_checks(tmp_path: Path):
    build = _load('build_review_package.py')
    items_path = tmp_path / 'items.json'
    frame = tmp_path / 'img_0.png'
    frame.write_bytes(b'frame-bytes')
    items = _pool(per_scene=10)
    for item in items:
        item['image_paths'] = [str(frame)]
    items.append({**_item('egocentric_encoding', 's0', 99), 'question_style': 'verbose'})
    items.append({**_item('egocentric_encoding', 's0', 98), 'status': 'unsupported'})
    items_path.write_text(json.dumps({'items': items}))

    out = tmp_path / 'pkg'
    manifest = build.build_package(
        items_path, out, per_construct=25, n_reviewers=10, reviews=2, seed=3
    )
    again = tmp_path / 'pkg2'
    manifest2 = build.build_package(
        items_path, again, per_construct=25, n_reviewers=10, reviews=2, seed=3
    )
    assert manifest['check_item_ids'] == manifest2['check_item_ids']
    assert manifest['practice_item_ids'] == manifest2['practice_item_ids']

    checks = [manifest['items'][i] for i in manifest['check_item_ids']]
    assert len(checks) == 5
    assert len({row['construct'] for row in checks}) == 5
    practice = [manifest['items'][i] for i in manifest['practice_item_ids']]
    assert len(practice) == 2
    assert len({row['construct'] for row in practice}) == 2

    study_n = Counter(
        meta['construct'] for meta in manifest['items'].values() if meta['role'] == 'review'
    )
    assert all(study_n[c] == 25 for c in CONSTRUCTS if c not in NAV)
    # class4_per_reviewer=5, 10 reviewers, 2 reviews → 25 class-4 study items.
    assert sum(study_n[c] for c in NAV) == 25
    assert manifest['class4_per_reviewer'] == 5
    assert manifest['class4_total'] == 25
    for item_id, reviewers in manifest['reviewers_for_item'].items():
        assert len(reviewers) == 2
        assert len(set(reviewers)) == 2
        meta = manifest['items'][item_id]
        if meta['kind'] == 'nav':
            assert meta['gold'] == 'move_ahead'
            assert (out / 'private' / 'items' / f'{item_id}.json').is_file()
        else:
            assert meta['gold'] == 'B'

    per_reviewer_class4 = Counter()
    for item_id, reviewers in manifest['reviewers_for_item'].items():
        if manifest['items'][item_id]['kind'] != 'nav':
            continue
        if manifest['items'][item_id]['role'] != 'review':
            continue
        for name in reviewers:
            per_reviewer_class4[name] += 1
    assert per_reviewer_class4
    assert set(per_reviewer_class4.values()) == {5}

    for name, item_ids in manifest['assignments'].items():
        html = (out / name / 'index.html').read_text()
        payload = _payload(html)
        assert [row['item_id'] for row in payload] == item_ids
        assert payload[0]['role'] == 'practice'
        assert payload[0]['show_answer']
        assert all(row['show_answer'] is None for row in payload if row['role'] != 'practice')
        assert 'frame ' in html and ' of ' in html
        assert 'frames_viewed' in html
        assert 'frame-slider' in html
        assert 'reveal-question' in html
        assert 'Show the question' in html
        assert "Can't tell" in html
        assert 'worked out from these frames' in html
        assert 'route-builder' in html
        assert 'Forward (×N)' in html
        assert 'Turn left 90°' in html
        assert 'answerable' in html
        assert 'id="route"' not in html
        assert 'construct-tag' in html
        assert 'CONSTRUCT_INITIALS' in html
        assert 'Download CSV' in html
        assert 'localStorage' in html
        for row in payload:
            if row['frames']:
                assert row['frames'][0].startswith('frames/')
                assert (out / name / row['frames'][0]).read_bytes() == b'frame-bytes'
        assert set(manifest['check_item_ids']) <= {row['item_id'] for row in payload}

    held = set(manifest['check_item_ids']) | set(manifest['practice_item_ids'])
    sample_id, reviewers = next(
        (item_id, names)
        for item_id, names in manifest['reviewers_for_item'].items()
        if item_id not in held
    )
    outsiders = [name for name in manifest['assignments'] if name not in reviewers]
    assert outsiders
    outsider_html = (out / outsiders[0] / 'index.html').read_text()
    assert f'"item_id": "{sample_id}"' not in outsider_html
    assert not (out / reviewers[0] / 'manifest.json').exists()


def test_resolve_image_stays_in_item_scene(tmp_path: Path, monkeypatch):
    build = _load('build_review_package.py')
    items_root = tmp_path / 'items'
    house_a = items_root / 'house_a' / 'images'
    house_b = items_root / 'house_b' / 'images'
    house_a.mkdir(parents=True)
    house_b.mkdir(parents=True)
    (house_a / 'img_0.png').write_bytes(b'aaa')
    (house_b / 'img_0.png').write_bytes(b'bbb')
    cwd = tmp_path / 'cwd'
    cwd.mkdir()
    (cwd / 'img_0.png').write_bytes(b'cwd')
    monkeypatch.chdir(cwd)
    got = build.resolve_image('img_0.png', items_root, 'house_a')
    assert got is not None
    assert got.read_bytes() == b'aaa'
    assert build.resolve_image(str((house_b / 'img_0.png').resolve()), items_root, 'house_a').read_bytes() == b'bbb'


def test_analyze_accuracy_alpha_and_both_wrong(tmp_path: Path):
    analyze = _load('analyze_review.py')
    assert analyze.krippendorff_alpha([['A', 'A'], ['A', 'A']]) == 1.0
    assert analyze.krippendorff_alpha([['A', 'A'], ['A', 'B']]) == 0.0

    manifest = {
        'check_item_ids': ['check1'],
        'items': {
            'check1': {'construct': 'egocentric_encoding', 'role': 'check', 'kind': 'mc', 'gold': 'A'},
            'ok1': {'construct': 'egocentric_encoding', 'role': 'review', 'kind': 'mc', 'gold': 'A'},
            'bad1': {'construct': 'egocentric_encoding', 'role': 'review', 'kind': 'mc', 'gold': 'B'},
            'nav1': {
                'construct': 'route_knowledge',
                'role': 'review',
                'kind': 'nav',
                'gold': 'move_ahead',
            },
            'prac1': {'construct': 'spatial_updating', 'role': 'practice', 'kind': 'mc', 'gold': 'C'},
        },
    }
    rows = [
        {'reviewer_id': 'reviewer_01', 'item_id': 'check1', 'answer': 'A', 'answerable': '', 'flags': [], 'role': 'check'},
        {'reviewer_id': 'reviewer_02', 'item_id': 'check1', 'answer': 'B', 'answerable': '', 'flags': [], 'role': 'check'},
        {'reviewer_id': 'reviewer_01', 'item_id': 'ok1', 'answer': 'A', 'answerable': '', 'flags': [], 'role': 'review'},
        {'reviewer_id': 'reviewer_02', 'item_id': 'ok1', 'answer': 'A', 'answerable': '', 'flags': ['unclear_question'], 'role': 'review'},
        {'reviewer_id': 'reviewer_01', 'item_id': 'bad1', 'answer': 'A', 'answerable': '', 'flags': ['ambiguous_reference'], 'role': 'review'},
        {'reviewer_id': 'reviewer_02', 'item_id': 'bad1', 'answer': "Can't tell", 'answerable': '', 'flags': ['object_not_visible'], 'role': 'review'},
        # No private item → nav answers unscored (not counted in accuracy).
        {'reviewer_id': 'reviewer_01', 'item_id': 'nav1', 'answer': 'move_ahead', 'answerable': 'yes', 'flags': [], 'role': 'review'},
        {'reviewer_id': 'reviewer_02', 'item_id': 'nav1', 'answer': 'move_back', 'answerable': 'no', 'flags': [], 'role': 'review'},
        {'reviewer_id': 'reviewer_01', 'item_id': 'prac1', 'answer': 'wrong', 'answerable': '', 'flags': [], 'role': 'practice'},
    ]
    text = analyze.analyze(manifest, rows, package=tmp_path)
    assert 'egocentric_encoding' in text
    # ok1 both correct, bad1 both wrong → 2/4 for egocentric.
    assert '0.500 (2/4)' in text
    assert 'bad1' in text
    assert 'both_flagged' in text
    assert 'reviewer_01  1/1' in text
    assert 'reviewer_02  0/1' in text
    assert 'prac1' not in text.split('flagged or wrong')[-1]
    assert 'answerable' in text
    assert 'mean_ndtw' in text or 'mean_sdtw' in text or 'n/a' in text

    csv_path = tmp_path / 'reviewer_01.csv'
    csv_path.write_text(
        'reviewer_id,item_id,construct,answer,answerable,flags,comment,seconds,role\n'
        'reviewer_01,check1,egocentric_encoding,A,,,"hello, there",3.5,check\n'
        'reviewer_01,ok1,egocentric_encoding,A,,,1,review\n'
        'reviewer_01,nav1,route_knowledge,"move_ahead, rotate_left",yes,,,2,review\n'
    )
    loaded = analyze.load_rows([csv_path])
    assert loaded[0]['comment'] == 'hello, there'
    assert loaded[0]['flags'] == []
    assert loaded[2]['answerable'] == 'yes'
    assert 'move_ahead' in loaded[2]['answer']

    data = analyze.collect_human_analysis(manifest, rows, package=tmp_path)
    assert data['kind'] == 'human'
    ego = next(c for c in data['constructs'] if c['construct'] == 'egocentric_encoding')
    assert ego['accuracy'] == 0.5
    assert ego['n_correct'] == 2 and ego['n_scored'] == 4
    assert any(p['item_id'] == 'bad1' and p['both_wrong'] for p in data['problems'])
    payload = {'sections': [data]}
    json_path = tmp_path / 'analysis.json'
    json_path.write_text(json.dumps(payload))
    txt_path = tmp_path / 'analysis.txt'
    txt_path.write_text(analyze.format_human_analysis(data) + '\n')
    blob = json.loads(json_path.read_text())
    assert blob['sections'][0]['kind'] == 'human'
    assert '0.500 (2/4)' in txt_path.read_text()


def test_plot_review_report_from_json(tmp_path: Path):
    import pytest

    pytest.importorskip('matplotlib')
    plot = _load('plot_review_report.py')
    payload = {
        'sections': [
            {
                'kind': 'human',
                'constructs': [
                    {
                        'construct': 'egocentric_encoding',
                        'kind': 'mc',
                        'n_reviews': 4,
                        'n_scored': 4,
                        'n_correct': 2,
                        'accuracy': 0.5,
                        'alpha': 0.0,
                        'flag_rate': 0.5,
                        'n_flagged': 2,
                        'mean_spl': None,
                        'mean_ndtw': None,
                        'mean_sdtw': None,
                    },
                    {
                        'construct': 'route_knowledge',
                        'kind': 'nav',
                        'n_reviews': 2,
                        'n_scored': 2,
                        'n_correct': 1,
                        'accuracy': 0.5,
                        'alpha': 1.0,
                        'flag_rate': 0.0,
                        'n_flagged': 0,
                        'mean_spl': 0.8,
                        'mean_ndtw': 0.7,
                        'mean_sdtw': 0.6,
                    },
                ],
                'answerable': [],
                'check_accuracy': [
                    {
                        'reviewer_id': 'reviewer_01',
                        'n_correct': 1,
                        'n_checks': 1,
                        'accuracy': 1.0,
                    }
                ],
                'problems': [
                    {
                        'item_id': 'bad1',
                        'construct': 'egocentric_encoding',
                        'both_flagged': True,
                        'both_wrong': True,
                    }
                ],
                'n_check_items': 1,
            },
            {
                'kind': 'vlm',
                'results': [
                    {
                        'construct': 'egocentric_encoding',
                        'condition': 'vision',
                        'n': 2,
                        'n_correct': 1,
                        'accuracy': 0.5,
                        'cant_tell_rate': 0.0,
                        'n_cant_tell': 0,
                        'chance': 0.25,
                        'human_accuracy': 0.5,
                    },
                    {
                        'construct': 'egocentric_encoding',
                        'condition': 'blind',
                        'n': 2,
                        'n_correct': 0,
                        'accuracy': 0.0,
                        'cant_tell_rate': 0.0,
                        'n_cant_tell': 0,
                        'chance': 0.25,
                        'human_accuracy': 0.5,
                    },
                ],
                'flags': [],
                'mc_chance': 0.25,
            },
        ]
    }
    json_path = tmp_path / 'review_analysis.json'
    json_path.write_text(json.dumps(payload))
    out = tmp_path / 'plots'
    assert plot.main([str(json_path), '-o', str(out)]) == 0
    assert (out / 'human_accuracy.png').is_file()
    assert (out / 'human_agreement.png').is_file()
    assert (out / 'human_nav_metrics.png').is_file()
    assert (out / 'human_checks.png').is_file()
    assert (out / 'human_problems.png').is_file()
    assert (out / 'vlm_accuracy.png').is_file()


def test_class4_study_total_respects_per_reviewer_budget():
    build = _load('build_review_package.py')
    assert build.class4_study_total(
        n_reviewers=10, reviews=2, class4_per_reviewer=5
    ) == 25
    assert build.class4_study_total(
        n_reviewers=10, reviews=2, class4_per_reviewer=0
    ) == 0


def test_class4_per_reviewer_cli_override(tmp_path: Path):
    """``--class4-per-reviewer`` changes the study budget (default stays 5)."""
    build = _load('build_review_package.py')
    items_path = tmp_path / 'items.json'
    frame = tmp_path / 'img_0.png'
    frame.write_bytes(b'frame-bytes')
    items = _pool(per_scene=10)
    for item in items:
        item['image_paths'] = [str(frame)]
    items_path.write_text(json.dumps({'items': items}))

    out = tmp_path / 'pkg'
    manifest = build.build_package(
        items_path,
        out,
        per_construct=25,
        n_reviewers=10,
        reviews=2,
        seed=0,
        class4_per_reviewer=3,
    )
    assert manifest['class4_per_reviewer'] == 3
    assert manifest['class4_total'] == 15
    per_reviewer = Counter()
    for item_id, reviewers in manifest['reviewers_for_item'].items():
        if manifest['items'][item_id]['role'] != 'review':
            continue
        if manifest['items'][item_id]['kind'] != 'nav':
            continue
        for name in reviewers:
            per_reviewer[name] += 1
    assert set(per_reviewer.values()) == {3}
