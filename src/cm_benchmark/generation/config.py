"""Load the frozen generation config (configs/generation.yaml).

Quality-affecting numbers live in that file. Modules read them through
``get_generation_config()`` or a ``cfg`` argument; they must not keep their
own copies. CLI flags overlay the loaded object (they do not rewrite the file).
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

import yaml

DEFAULT_CONFIG_NAME = 'default'
DEFAULT_CONFIG_RELATIVE = Path('configs') / 'generation.yaml'


def default_config_path() -> Path:
    """Repo ``configs/generation.yaml``, then ``./configs/generation.yaml``."""
    here = Path(__file__).resolve()
    repo = here.parents[3] / DEFAULT_CONFIG_RELATIVE
    if repo.is_file():
        return repo
    cwd = Path.cwd() / DEFAULT_CONFIG_RELATIVE
    if cwd.is_file():
        return cwd
    return repo


@dataclass(frozen=True)
class VisibilityFloors:
    min_bbox_area: float
    min_side: float
    min_visible_pixels: float
    min_occupancy_ratio: Optional[float] = None
    max_obj_distance: Optional[float] = None

    def as_kwargs(self) -> dict[str, float]:
        return {
            'min_bbox_area': float(self.min_bbox_area),
            'min_side': float(self.min_side),
            'min_visible_pixels': float(self.min_visible_pixels),
        }

    def as_question_visibility(self) -> dict[str, Optional[float]]:
        return {
            'min_bbox_area': float(self.min_bbox_area),
            'min_side': float(self.min_side),
            'min_visible_pixels': float(self.min_visible_pixels),
            'min_occupancy_ratio': self.min_occupancy_ratio,
            'max_obj_distance': self.max_obj_distance,
        }


@dataclass(frozen=True)
class VisibilityBorder:
    edge_margin_px: int
    min_extent_if_clipped_px: float


@dataclass(frozen=True)
class VisibilityConfig:
    model_path: Optional[str]
    proba_low: float
    proba_high: float
    degenerate_band_half_width: float
    static: VisibilityFloors
    query: VisibilityFloors
    border: VisibilityBorder
    id_encode: VisibilityFloors


@dataclass(frozen=True)
class DirectionsConfig:
    ahead_half_width_fov_deg: float
    ahead_half_width_full_deg: float
    pt_back_deg: float
    pt_boundary_margin_deg: float
    refuse_straddling_objects: bool


@dataclass(frozen=True)
class OptionsConfig:
    n_options: int
    shuffle_seed: int
    max_label_share: float


@dataclass(frozen=True)
class ProximitySetting:
    margin_m: Optional[float] = None
    ratio: Optional[float] = None
    margin_m_small: Optional[float] = None
    margin_m_large: Optional[float] = None
    room_area_split_m2: Optional[float] = None


@dataclass(frozen=True)
class DisambiguationConfig:
    proximity_setting: str
    all_marks: bool
    ratio_2d: float
    min_gap_2d_frac: float
    mark_palette: tuple[str, ...]
    settings: Mapping[str, ProximitySetting]

    def active_proximity(self) -> ProximitySetting:
        return self.settings[self.proximity_setting]


@dataclass(frozen=True)
class BudgetConfig:
    max_per_construct: int
    max_per_scene: int
    candidate_pool: int


@dataclass(frozen=True)
class SpatialWorkingMemoryConfig:
    min_delay: int
    max_delay: Optional[int]
    delay_steps: tuple[int, ...]
    count_divisor: int
    count_min_instances: int


@dataclass(frozen=True)
class SpatialUpdatingConfig:
    min_delay: int
    max_delay: Optional[int]
    delay_steps: tuple[int, ...]
    min_pos_change_m: float
    min_heading_change_deg: float
    object_static_eps_m: float


@dataclass(frozen=True)
class MotionConfig:
    position_eps_m: float


@dataclass(frozen=True)
class InvisibleDisplacementConfig:
    floor_anchor_radius_m: float
    min_displacement_m: float
    min_diagnostic_labels: int
    max_delay: Optional[int]


@dataclass(frozen=True)
class LandmarksConfig:
    layout_base: float
    layout_per_sighting: float
    other_base: float
    other_per_sighting: float
    top_n_per_region: int
    max_total: int


@dataclass(frozen=True)
class PerspectiveTakingConfig:
    top_n_per_region: int
    max_total: int


@dataclass(frozen=True)
class TemplatesConfig:
    verbose_max_nonvisible_names: int


@dataclass(frozen=True)
class Class4RouteConfig:
    context_margin_steps: int
    max_ratio_floor: float
    min_hop_count: int
    max_subpath_nodes: int
    min_turns: int
    max_turns: int


@dataclass(frozen=True)
class Class4SurveyConfig:
    max_ratio_floor: float
    min_agent_source_dist_m: float
    view_radius_percentile: float
    min_turns: int


@dataclass(frozen=True)
class Class4FollowerConfig:
    ladder_multipliers: tuple[float, ...]
    arrive_fraction: float
    max_actions: int


@dataclass(frozen=True)
class Class4FallbackConfig:
    grid_size_m: float
    agent_rotation_deg: float
    agent_move_m: Optional[float]


@dataclass(frozen=True)
class Class4Config:
    n_frames: int
    min_pair_geodesic_m: float
    max_pair_geodesic_m: float
    scene_geodesic_percentile: float
    landmark_snap_m: float
    landmark_snap_default_m: float
    landmark_snap_default_grid_mult: float
    snap_tolerance_steps: float
    goal_success_m: float
    search_state_quant_m: float
    move_block_slack_m: float
    route: Class4RouteConfig
    survey: Class4SurveyConfig
    follower: Class4FollowerConfig
    fallback: Class4FallbackConfig


@dataclass(frozen=True)
class EpisodeGtConfig:
    frame_w: int
    frame_h: int
    fov_v_deg: float
    k_neighbors: int
    radius_m: float
    ey_m: float
    ez_m: float
    angle_threshold_xz_deg: int
    min_distance_m: float
    med_distance_m: float
    max_distance_m: float
    mov_constant_m: float
    state_sparsify_eps_m: float


@dataclass(frozen=True)
class GenerationConfig:
    """Frozen quality parameters for one generation run."""

    name: str
    visibility: VisibilityConfig
    directions: DirectionsConfig
    options: OptionsConfig
    disambiguation: DisambiguationConfig
    budget: BudgetConfig
    spatial_working_memory: SpatialWorkingMemoryConfig
    spatial_updating: SpatialUpdatingConfig
    motion: MotionConfig
    invisible_displacement: InvisibleDisplacementConfig
    landmarks: LandmarksConfig
    perspective_taking: PerspectiveTakingConfig
    templates: TemplatesConfig
    class4: Class4Config
    episode_gt: EpisodeGtConfig
    source_path: Optional[str] = None

    def content_hash(self) -> str:
        """SHA-256 of the values (not the file path), so overrides are visible."""
        payload = asdict(self)
        payload.pop('source_path', None)
        blob = json.dumps(payload, sort_keys=True, default=str).encode('utf-8')
        return hashlib.sha256(blob).hexdigest()

    def provenance(self) -> dict[str, str]:
        return {
            'generation_config': self.name,
            'generation_config_hash': self.content_hash(),
        }

    def with_overrides(
        self,
        *,
        max_per_construct: Optional[int] = None,
        swm_min_delay: Optional[int] = None,
        swm_max_delay: Optional[int] = None,
        su_min_delay: Optional[int] = None,
        su_max_delay: Optional[int] = None,
        visibility_model_path: Optional[str] = None,
        n_frames: Optional[int] = None,
    ) -> 'GenerationConfig':
        """Return a new frozen config with CLI overlays applied."""
        cfg = self
        if max_per_construct is not None:
            cfg = replace(cfg, budget=replace(cfg.budget, max_per_construct=int(max_per_construct)))
        if swm_min_delay is not None or swm_max_delay is not None:
            swm = cfg.spatial_working_memory
            cfg = replace(
                cfg,
                spatial_working_memory=replace(
                    swm,
                    min_delay=swm.min_delay if swm_min_delay is None else int(swm_min_delay),
                    max_delay=swm.max_delay if swm_max_delay is None else int(swm_max_delay),
                ),
            )
        if su_min_delay is not None or su_max_delay is not None:
            su = cfg.spatial_updating
            cfg = replace(
                cfg,
                spatial_updating=replace(
                    su,
                    min_delay=su.min_delay if su_min_delay is None else int(su_min_delay),
                    max_delay=su.max_delay if su_max_delay is None else int(su_max_delay),
                ),
            )
        if visibility_model_path is not None:
            cfg = replace(
                cfg,
                visibility=replace(cfg.visibility, model_path=str(visibility_model_path)),
            )
        if n_frames is not None:
            cfg = replace(cfg, class4=replace(cfg.class4, n_frames=int(n_frames)))
        return cfg


def _floors(raw: Mapping[str, Any], *, extra: bool = False) -> VisibilityFloors:
    kwargs = {
        'min_bbox_area': float(raw['min_bbox_area']),
        'min_side': float(raw['min_side']),
        'min_visible_pixels': float(raw['min_visible_pixels']),
    }
    if extra:
        kwargs['min_occupancy_ratio'] = raw.get('min_occupancy_ratio')
        kwargs['max_obj_distance'] = raw.get('max_obj_distance')
    return VisibilityFloors(**kwargs)


def _proximity(raw: Mapping[str, Any]) -> ProximitySetting:
    return ProximitySetting(
        margin_m=None if raw.get('margin_m') is None else float(raw['margin_m']),
        ratio=None if raw.get('ratio') is None else float(raw['ratio']),
        margin_m_small=None if raw.get('margin_m_small') is None else float(raw['margin_m_small']),
        margin_m_large=None if raw.get('margin_m_large') is None else float(raw['margin_m_large']),
        room_area_split_m2=None
        if raw.get('room_area_split_m2') is None
        else float(raw['room_area_split_m2']),
    )


def generation_config_from_mapping(
    raw: Mapping[str, Any], *, source_path: Optional[str] = None
) -> GenerationConfig:
    """Build a frozen config from a parsed YAML mapping."""
    vis = raw['visibility']
    dis = raw['disambiguation']
    c4 = raw['class4']
    egt = raw['episode_gt']
    return GenerationConfig(
        name=str(raw.get('name') or DEFAULT_CONFIG_NAME),
        source_path=source_path,
        visibility=VisibilityConfig(
            model_path=vis.get('model_path'),
            proba_low=float(vis['proba_low']),
            proba_high=float(vis['proba_high']),
            degenerate_band_half_width=float(vis['degenerate_band_half_width']),
            static=_floors(vis['static'], extra=True),
            query=_floors(vis['query']),
            border=VisibilityBorder(
                edge_margin_px=int(vis['border']['edge_margin_px']),
                min_extent_if_clipped_px=float(vis['border']['min_extent_if_clipped_px']),
            ),
            id_encode=_floors(vis['id_encode']),
        ),
        directions=DirectionsConfig(
            ahead_half_width_fov_deg=float(raw['directions']['ahead_half_width_fov_deg']),
            ahead_half_width_full_deg=float(raw['directions']['ahead_half_width_full_deg']),
            pt_back_deg=float(raw['directions']['pt_back_deg']),
            pt_boundary_margin_deg=float(raw['directions']['pt_boundary_margin_deg']),
            refuse_straddling_objects=bool(raw['directions']['refuse_straddling_objects']),
        ),
        options=OptionsConfig(
            n_options=int(raw['options']['n_options']),
            shuffle_seed=int(raw['options']['shuffle_seed']),
            max_label_share=float(raw['options']['max_label_share']),
        ),
        disambiguation=DisambiguationConfig(
            proximity_setting=str(dis['proximity_setting']),
            all_marks=bool(dis['all_marks']),
            ratio_2d=float(dis['ratio_2d']),
            min_gap_2d_frac=float(dis['min_gap_2d_frac']),
            mark_palette=tuple(dis['mark_palette']),
            settings={k: _proximity(v) for k, v in (dis.get('settings') or {}).items()},
        ),
        budget=BudgetConfig(
            max_per_construct=int(raw['budget']['max_per_construct']),
            max_per_scene=int(raw['budget']['max_per_scene']),
            candidate_pool=int(raw['budget']['candidate_pool']),
        ),
        spatial_working_memory=SpatialWorkingMemoryConfig(
            min_delay=int(raw['spatial_working_memory']['min_delay']),
            max_delay=raw['spatial_working_memory'].get('max_delay'),
            delay_steps=tuple(int(k) for k in raw['spatial_working_memory']['delay_steps']),
            count_divisor=int(raw['spatial_working_memory']['count_divisor']),
            count_min_instances=int(raw['spatial_working_memory']['count_min_instances']),
        ),
        spatial_updating=SpatialUpdatingConfig(
            min_delay=int(raw['spatial_updating']['min_delay']),
            max_delay=raw['spatial_updating'].get('max_delay'),
            delay_steps=tuple(int(k) for k in raw['spatial_updating']['delay_steps']),
            min_pos_change_m=float(raw['spatial_updating']['min_pos_change_m']),
            min_heading_change_deg=float(raw['spatial_updating']['min_heading_change_deg']),
            object_static_eps_m=float(raw['spatial_updating']['object_static_eps_m']),
        ),
        motion=MotionConfig(position_eps_m=float(raw['motion']['position_eps_m'])),
        invisible_displacement=InvisibleDisplacementConfig(
            floor_anchor_radius_m=float(raw['invisible_displacement']['floor_anchor_radius_m']),
            min_displacement_m=float(raw['invisible_displacement']['min_displacement_m']),
            min_diagnostic_labels=int(raw['invisible_displacement']['min_diagnostic_labels']),
            max_delay=(
                None
                if raw['invisible_displacement'].get('max_delay') is None
                else int(raw['invisible_displacement']['max_delay'])
            ),
        ),
        landmarks=LandmarksConfig(
            layout_base=float(raw['landmarks']['layout_base']),
            layout_per_sighting=float(raw['landmarks']['layout_per_sighting']),
            other_base=float(raw['landmarks']['other_base']),
            other_per_sighting=float(raw['landmarks']['other_per_sighting']),
            top_n_per_region=int(raw['landmarks']['top_n_per_region']),
            max_total=int(raw['landmarks']['max_total']),
        ),
        perspective_taking=PerspectiveTakingConfig(
            top_n_per_region=int(raw['perspective_taking']['top_n_per_region']),
            max_total=int(raw['perspective_taking']['max_total']),
        ),
        templates=TemplatesConfig(
            verbose_max_nonvisible_names=int(raw['templates']['verbose_max_nonvisible_names']),
        ),
        class4=Class4Config(
            n_frames=int(c4['n_frames']),
            min_pair_geodesic_m=float(c4['min_pair_geodesic_m']),
            max_pair_geodesic_m=float(c4['max_pair_geodesic_m']),
            scene_geodesic_percentile=float(c4['scene_geodesic_percentile']),
            landmark_snap_m=float(c4['landmark_snap_m']),
            landmark_snap_default_m=float(c4['landmark_snap_default_m']),
            landmark_snap_default_grid_mult=float(c4['landmark_snap_default_grid_mult']),
            snap_tolerance_steps=float(c4['snap_tolerance_steps']),
            goal_success_m=float(c4['goal_success_m']),
            search_state_quant_m=float(c4['search_state_quant_m']),
            move_block_slack_m=float(c4['move_block_slack_m']),
            route=Class4RouteConfig(
                context_margin_steps=int(c4['route']['context_margin_steps']),
                max_ratio_floor=float(c4['route']['max_ratio_floor']),
                min_hop_count=int(c4['route']['min_hop_count']),
                max_subpath_nodes=int(c4['route']['max_subpath_nodes']),
                min_turns=int(c4['route']['min_turns']),
                max_turns=int(c4['route']['max_turns']),
            ),
            survey=Class4SurveyConfig(
                max_ratio_floor=float(c4['survey']['max_ratio_floor']),
                min_agent_source_dist_m=float(c4['survey']['min_agent_source_dist_m']),
                view_radius_percentile=float(c4['survey']['view_radius_percentile']),
                min_turns=int(c4['survey']['min_turns']),
            ),
            follower=Class4FollowerConfig(
                ladder_multipliers=tuple(float(x) for x in c4['follower']['ladder_multipliers']),
                arrive_fraction=float(c4['follower']['arrive_fraction']),
                max_actions=int(c4['follower']['max_actions']),
            ),
            fallback=Class4FallbackConfig(
                grid_size_m=float(c4['fallback']['grid_size_m']),
                agent_rotation_deg=float(c4['fallback']['agent_rotation_deg']),
                agent_move_m=c4['fallback'].get('agent_move_m'),
            ),
        ),
        episode_gt=EpisodeGtConfig(
            frame_w=int(egt['frame_w']),
            frame_h=int(egt['frame_h']),
            fov_v_deg=float(egt['fov_v_deg']),
            k_neighbors=int(egt['k_neighbors']),
            radius_m=float(egt['radius_m']),
            ey_m=float(egt['ey_m']),
            ez_m=float(egt['ez_m']),
            angle_threshold_xz_deg=int(egt['angle_threshold_xz_deg']),
            min_distance_m=float(egt['min_distance_m']),
            med_distance_m=float(egt['med_distance_m']),
            max_distance_m=float(egt['max_distance_m']),
            mov_constant_m=float(egt['mov_constant_m']),
            state_sparsify_eps_m=float(egt['state_sparsify_eps_m']),
        ),
    )


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively overlay ``overlay`` onto a copy of ``base``."""
    out = dict(base)
    for key, val in overlay.items():
        if isinstance(val, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_generation_config(path: Optional[str | Path] = None) -> GenerationConfig:
    """Read YAML from ``path`` (or the repo default) into a frozen config.

    A file that omits the full schema (e.g. ``configs/ablations/sr3d.yaml``)
    is merged onto ``configs/generation.yaml``.
    """
    p = Path(path) if path is not None else default_config_path()
    raw = yaml.safe_load(p.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f'generation config must be a mapping: {p}')
    required = ('visibility', 'disambiguation', 'class4', 'episode_gt')
    if path is not None and not all(k in raw for k in required):
        base = yaml.safe_load(default_config_path().read_text())
        if not isinstance(base, dict):
            raise ValueError(f'default generation config must be a mapping')
        raw = _deep_merge(base, raw)
    name = raw.get('name') or p.stem
    raw['name'] = name
    return generation_config_from_mapping(raw, source_path=str(p.resolve()))


_current: ContextVar[Optional[GenerationConfig]] = ContextVar(
    'cm_generation_config', default=None
)
_default: Optional[GenerationConfig] = None


def get_generation_config() -> GenerationConfig:
    """Active config for this run, or the default YAML if none was set."""
    cfg = _current.get()
    if cfg is not None:
        return cfg
    global _default
    if _default is None:
        _default = load_generation_config()
    return _default


def set_generation_config(cfg: GenerationConfig):
    """Install ``cfg`` as the active config (returns a context-var token)."""
    return _current.set(cfg)


def reset_generation_config(token) -> None:
    _current.reset(token)


@contextmanager
def use_generation_config(cfg: GenerationConfig) -> Iterator[GenerationConfig]:
    """Temporarily use ``cfg`` as the active generation config."""
    token = set_generation_config(cfg)
    try:
        yield cfg
    finally:
        reset_generation_config(token)


def _legacy_lookup(name: str) -> Any:
    cfg = get_generation_config()
    table = {
        'AHEAD_HALF_WIDTH_FOV': cfg.directions.ahead_half_width_fov_deg,
        'AHEAD_HALF_WIDTH_FULL': cfg.directions.ahead_half_width_full_deg,
        'PT_DIRECTION_BOUNDARY_MARGIN_DEG': cfg.directions.pt_boundary_margin_deg,
        'PT_DIRECTION_BACK_DEG': cfg.directions.pt_back_deg,
        'MIN_DISAMBIG_MARGIN_M': cfg.disambiguation.active_proximity().margin_m,
        'MIN_DISAMBIG_MARGIN_RATIO': cfg.disambiguation.active_proximity().ratio,
        'QUERY_FOV_MIN_BBOX_AREA': cfg.visibility.query.min_bbox_area,
        'QUERY_FOV_MIN_SIDE': cfg.visibility.query.min_side,
        'QUERY_FOV_MIN_VISIBLE_PIXELS': cfg.visibility.query.min_visible_pixels,
        'QUERY_FOV_EDGE_MARGIN_PX': cfg.visibility.border.edge_margin_px,
        'QUERY_FOV_MIN_EXTENT_IF_CLIPPED': cfg.visibility.border.min_extent_if_clipped_px,
        'ID_ENCODE_FOV_MIN_BBOX_AREA': cfg.visibility.id_encode.min_bbox_area,
        'ID_ENCODE_FOV_MIN_SIDE': cfg.visibility.id_encode.min_side,
        'ID_ENCODE_FOV_MIN_VISIBLE_PIXELS': cfg.visibility.id_encode.min_visible_pixels,
        'FLOOR_ANCHOR_RADIUS_M': cfg.invisible_displacement.floor_anchor_radius_m,
        'SURVEY_MIN_AGENT_SOURCE_DIST_M': cfg.class4.survey.min_agent_source_dist_m,
        'CLASS4_GOAL_SUCCESS_M': cfg.class4.goal_success_m,
        'MIN_PAIR_GEODESIC_M': cfg.class4.min_pair_geodesic_m,
        'MAX_PAIR_GEODESIC_M': cfg.class4.max_pair_geodesic_m,
        'SCENE_PAIR_GEODESIC_PERCENTILE': cfg.class4.scene_geodesic_percentile,
        'SURVEY_MIN_GEODESIC_EUCLIDEAN_RATIO': cfg.class4.survey.max_ratio_floor,
        'ROUTE_MAX_GEODESIC_EUCLIDEAN_RATIO_CAP': cfg.class4.route.max_ratio_floor,
        'ROUTE_LANDMARK_SNAP_M': cfg.class4.landmark_snap_m,
        'ROUTE_MIN_HOP_COUNT': cfg.class4.route.min_hop_count,
        'ROUTE_MAX_SUBPATH_NODES': cfg.class4.route.max_subpath_nodes,
        'ROUTE_MIN_TURN_COUNT': cfg.class4.route.min_turns,
        'ROUTE_MAX_TURN_COUNT': cfg.class4.route.max_turns,
        '_POSITION_EPS_M': cfg.motion.position_eps_m,
        '_ID_ENCODE_FOV': cfg.visibility.id_encode.as_kwargs(),
        'DEFAULT_PROBA_LOW': cfg.visibility.proba_low,
        'DEFAULT_PROBA_HIGH': cfg.visibility.proba_high,
        'DEFAULT_QUESTION_VISIBILITY_THRESHOLDS': cfg.visibility.static.as_question_visibility(),
    }
    if name not in table:
        raise AttributeError(name)
    return table[name]


def legacy_constant(name: str) -> Any:
    """Value of a former module-level constant, from the active config."""
    return _legacy_lookup(name)
