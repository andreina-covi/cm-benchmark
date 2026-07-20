"""Candidate benchmark item schema for first-draft Q&A generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


CONSTRUCT_CLASS = {
    'egocentric_encoding': 1,
    'allocentric_encoding': 1,
    'spatial_working_memory': 2,
    'invisible_displacement': 2,
    'spatial_updating': 3,
    'perspective_taking': 3,
    'route_knowledge': 4,
    'survey_knowledge': 4,
}

CONSTRUCT_FOR = {
    'egocentric_encoding': 'egocentric',
    'allocentric_encoding': 'allocentric',
    'spatial_working_memory': 'egocentric',
    'invisible_displacement': 'allocentric',
    'spatial_updating': 'egocentric',
    'perspective_taking': 'allocentric',
    'route_knowledge': 'egocentric',
    'survey_knowledge': 'allocentric',
}


@dataclass
class CandidateItem:
    """Draft item aligned with the benchmark item schema (verification fields null)."""

    item_id: str
    construct: str
    class_: int
    frame_of_reference: str
    environment: str
    scene_id: str
    image_paths: list[str]
    question: str
    options: dict[str, str]
    answer: Optional[str]
    answer_source: Optional[list[str]]
    queried_object_id: Optional[str] = None
    distractor_rationale: dict[str, str] = field(default_factory=dict)
    status: str = 'ok'  # ok | thin | unsupported
    query_step: Optional[int] = None
    encoding_step: Optional[int] = None
    question_style: str = 'concise'  # concise | verbose
    paired_item_id: Optional[str] = None
    difficulty: Optional[int] = None
    agent_trajectory: Optional[list[dict]] = None
    agent_actions: Optional[list] = None
    displacement_event: Optional[dict] = None
    blind_llm_correct: Optional[bool] = None
    caption_llm_correct: Optional[bool] = None
    vision_necessary: Optional[bool] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d['class'] = d.pop('class_')
        return d

    @classmethod
    def unsupported(
        cls,
        *,
        item_id: str,
        construct: str,
        scene_id: str,
        environment: str = 'ai2thor',
        reason: str = 'insufficient_metadata',
    ) -> 'CandidateItem':
        return cls(
            item_id=item_id,
            construct=construct,
            class_=CONSTRUCT_CLASS[construct],
            frame_of_reference=CONSTRUCT_FOR[construct],
            environment=environment,
            scene_id=scene_id,
            image_paths=[],
            question='',
            options={},
            answer=None,
            answer_source=None,
            status='unsupported',
            distractor_rationale={'reason': reason},
        )
