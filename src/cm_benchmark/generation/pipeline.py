"""Orchestrate first-draft item generation from episode GT."""

from __future__ import annotations

from typing import Optional

from cm_benchmark.generation.episode_io import scene_id_of
from cm_benchmark.generation.paraphrase import paraphrase_question
from cm_benchmark.generation.planner import plan_episode
from cm_benchmark.generation.templates import build_items_from_facts

ALL_CONSTRUCTS = [
    'egocentric_encoding',
    'allocentric_encoding',
    'spatial_working_memory',
    'invisible_displacement',
    'spatial_updating',
    'perspective_taking',
    'route_knowledge',
    'survey_knowledge',
]


def draft_items_for_episode(
    episode: dict,
    *,
    constructs: Optional[list[str]] = None,
    max_per_construct: int = 3,
    styles: tuple[str, ...] = ('concise', 'verbose'),
    paraphrase: bool = False,
    episode_tag: Optional[str] = None,
) -> list[dict]:
    facts = plan_episode(
        episode, constructs=constructs, max_per_construct=max_per_construct
    )
    tag = episode_tag or scene_id_of(episode)
    items = build_items_from_facts(
        episode, facts, episode_tag=tag, styles=styles
    )
    out = []
    for item in items:
        d = item.to_dict()
        if paraphrase and d.get('question'):
            d['question'] = paraphrase_question(
                d['question'],
                locked_answer=d.get('answer') or '',
                options=d.get('options') or {},
                answer_source=d.get('answer_source') or [],
                enabled=True,
            )
        out.append(d)
    return out
