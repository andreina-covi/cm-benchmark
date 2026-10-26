"""VLM chat kit: same study sample as the human package, no gold in chat folders."""

from __future__ import annotations

import csv
import importlib.util
import json
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
        'question': f'Where is the Cup relative to you now? ({construct} {n})',
        'options': {} if nav else {'A': 'left', 'B': 'right', 'C': 'ahead', 'D': 'behind'},
        'answer': 'move_ahead, rotate_left' if nav else 'B',
        'image_paths': [],
        'context': {
            'action_sequence': ['move_ahead', 'rotate_left'] if nav else None,
            'source_node': 'n0',
            'goal_node': 'n1',
            'start_heading_deg': 0.0,
            'start_pos': [0.0, 0.0, 0.0],
            'goal_pos': [1.0, 0.0, 0.0],
            'traversed_edges': [['n0', 'n1']],
            'viewed_edges': [['n0', 'n1']],
        },
    }


def _pool(per_scene: int = 5) -> list[dict]:
    items = []
    for construct in CONSTRUCTS:
        for scene in ('s0', 's1'):
            for n in range(per_scene):
                items.append(_item(construct, scene, n))
    return items


def test_kit_matches_review_sample_and_hides_gold(tmp_path: Path):
    build_review = _load('build_review_package.py')
    build_kit = _load('build_vlm_chat_kit.py')
    analyze = _load('analyze_review.py')

    frame = tmp_path / 'img_0.png'
    frame.write_bytes(b'frame-bytes')
    items = _pool()
    for item in items:
        item['image_paths'] = [str(frame)]
    items_path = tmp_path / 'items.json'
    items_path.write_text(json.dumps({'items': items}))

    review_out = tmp_path / 'review'
    build_review.build_package(
        items_path, review_out, per_construct=2, n_reviewers=2, reviews=1, seed=7
    )
    study_ids = {
        item_id
        for item_id, meta in json.loads((review_out / 'manifest.json').read_text())[
            'items'
        ].items()
        if meta['role'] == 'review'
    }

    kit_out = tmp_path / 'vlm_kit'
    manifest = build_kit.build_kit(
        items_path,
        kit_out,
        per_construct=2,
        seed=7,
        items_per_construct=2,
        max_items=2,
    )
    assert manifest['n_items'] == 2
    assert set(manifest['item_ids']) <= study_ids

    for item_id in manifest['item_ids']:
        vision = kit_out / 'items' / item_id / 'vision'
        blind = kit_out / 'items' / item_id / 'blind'
        assert (vision / 'frame_01.png').read_bytes() == b'frame-bytes'
        assert (vision / 'prompt.txt').is_file()
        assert list(blind.glob('frame_*')) == []
        vision_prompt = (vision / 'prompt.txt').read_text()
        blind_prompt = (blind / 'prompt.txt').read_text()
        assert 'You are answering spatial-cognition questions' in vision_prompt
        assert "Can't tell" in vision_prompt
        assert 'No images are available' in blind_prompt
        assert 'No images are available' not in vision_prompt
        assert not (vision / 'answer.txt').exists()
        assert not (blind / 'answers.csv').exists()
        item = next(row for row in items if row['item_id'] == item_id)
        gold = str(item['answer'])
        if item['construct'] not in NAV:
            assert f'Answer: {gold}' not in vision_prompt

    answers = list(csv.DictReader((kit_out / 'answers.csv').open()))
    assert len(answers) == 2
    assert all(row['gold'] for row in answers)
    results = list(csv.DictReader((kit_out / 'results_template.csv').open()))
    assert len(results) == 4  # 2 items × 2 conditions
    assert {row['condition'] for row in results} == {'vision', 'blind'}
    assert 'comment' in results[0]

    filled = kit_out / 'results.csv'
    with filled.open('w', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'item_id',
                'construct',
                'condition',
                'folder',
                'model',
                'raw_reply',
                'comment',
            ],
        )
        writer.writeheader()
        for row in results:
            meta = next(a for a in answers if a['item_id'] == row['item_id'])
            if meta['kind'] == 'mc':
                reply = meta['gold'] if row['condition'] == 'vision' else "Can't tell"
            else:
                reply = meta['gold'] if row['condition'] == 'vision' else "Can't tell"
            writer.writerow(
                {**row, 'model': 'test-model', 'raw_reply': reply, 'comment': 'note'}
            )

    report = analyze.analyze_vlm(kit_out, results_path=filled)
    assert 'vision' in report and 'blind' in report
    assert 'cant_tell' in report
    assert '0.25' in report
    assert analyze.parse_mc_letter('B\nlooks clear') == 'B'
    assert analyze.parse_mc_letter("Can't tell\nblurry") is None
    assert analyze.is_cant_tell("Can't tell")
    ok, parsed, abstain = analyze.score_vlm_reply(
        kit=kit_out,
        item_id=answers[0]['item_id'],
        construct=answers[0]['construct'],
        kind=answers[0]['kind'],
        gold=answers[0]['gold'],
        raw_reply="Can't tell",
    )
    assert abstain and not ok and parsed == "Can't tell"


def test_default_two_items_per_construct(tmp_path: Path):
    build_kit = _load('build_vlm_chat_kit.py')
    frame = tmp_path / 'img_0.png'
    frame.write_bytes(b'x')
    items = _pool()
    for item in items:
        item['image_paths'] = [str(frame)]
    items_path = tmp_path / 'items.json'
    items_path.write_text(json.dumps({'items': items}))

    kit_out = tmp_path / 'kit'
    manifest = build_kit.build_kit(items_path, kit_out, per_construct=5, seed=0)
    assert manifest['items_per_construct'] == 2
    counts = {}
    for item_id in manifest['item_ids']:
        construct = item_id.split('_', 1)[1].rsplit('_', 1)[0]
        # item_id is scene_construct_n — construct can contain underscores
        for c in CONSTRUCTS:
            if f'_{c}_' in f'_{item_id}_' or item_id.endswith(f'_{c}_0') or f'_{c}_' in item_id:
                pass
        row = next(i for i in items if i['item_id'] == item_id)
        counts[row['construct']] = counts.get(row['construct'], 0) + 1
    assert all(n == 2 for n in counts.values())
    assert len(counts) == len(CONSTRUCTS)
    batch = kit_out / 'construct_batches' / 'egocentric_encoding.txt'
    assert batch.is_file()
    assert len(batch.read_text().strip().splitlines()) == 2


def test_review_package_item_list_reuse(tmp_path: Path):
    build_review = _load('build_review_package.py')
    build_kit = _load('build_vlm_chat_kit.py')

    frame = tmp_path / 'img_0.png'
    frame.write_bytes(b'x')
    items = _pool()
    for item in items:
        item['image_paths'] = [str(frame)]
    items_path = tmp_path / 'items.json'
    items_path.write_text(json.dumps({'items': items}))

    review_out = tmp_path / 'review'
    build_review.build_package(
        items_path, review_out, per_construct=3, n_reviewers=2, reviews=1, seed=1
    )
    kit_out = tmp_path / 'kit'
    manifest = build_kit.build_kit(
        items_path,
        kit_out,
        review_package=review_out,
        items_per_construct=2,
    )
    assert manifest['n_items'] == 2 * len(CONSTRUCTS)
    study = {
        item_id
        for item_id, meta in json.loads((review_out / 'manifest.json').read_text())[
            'items'
        ].items()
        if meta['role'] == 'review'
    }
    assert set(manifest['item_ids']) <= study
