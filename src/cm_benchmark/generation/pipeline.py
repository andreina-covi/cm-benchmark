"""Orchestrate item generation from episode GT."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from cm_benchmark.generation.config import (
    GenerationConfig,
    get_generation_config,
    use_generation_config,
)
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
    'survey_based_route_planning',
]


def generate_items_for_episode(
    episode: dict,
    *,
    constructs: Optional[list[str]] = None,
    max_per_construct: Optional[int] = None,
    swm_min_delay: Optional[int] = None,
    swm_max_delay: Optional[int] = None,
    su_min_delay: Optional[int] = None,
    su_max_delay: Optional[int] = None,
    styles: tuple[str, ...] = ('concise', 'verbose'),
    paraphrase: bool = False,
    episode_tag: Optional[str] = None,
    visibility_model_path: Optional[str] = None,
    cfg: Optional[GenerationConfig] = None,
    marked_root: Optional[Path] = None,
) -> list[dict]:
    """Episode GT → item dicts: plan facts, word them per style, optional paraphrase.

    The answer and ``answer_source`` come from the planner; paraphrase may only
    reword ``question`` and is off by default.
    """
    active = (cfg or get_generation_config()).with_overrides(
        max_per_construct=max_per_construct,
        swm_min_delay=swm_min_delay,
        swm_max_delay=swm_max_delay,
        su_min_delay=su_min_delay,
        su_max_delay=su_max_delay,
        visibility_model_path=visibility_model_path,
    )
    with use_generation_config(active):
        facts = plan_episode(
            episode,
            constructs=constructs,
            max_per_construct=active.budget.max_per_construct,
            swm_min_delay=active.spatial_working_memory.min_delay,
            swm_max_delay=active.spatial_working_memory.max_delay,
            su_min_delay=active.spatial_updating.min_delay,
            su_max_delay=active.spatial_updating.max_delay,
            visibility_model_path=active.visibility.model_path,
            cfg=active,
        )
        tag = episode_tag or scene_id_of(episode)
        items = build_items_from_facts(
            episode, facts, episode_tag=tag, styles=styles, marked_root=marked_root
        )
        provenance = active.provenance()
        out = []
        for item in items:
            d = item.to_dict()
            d.update(provenance)
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
