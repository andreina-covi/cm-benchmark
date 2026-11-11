"""Shared pytest fixtures.

Disable auto-discovery of a local ``visibility_filter.joblib`` so item generation/planner
tests stay deterministic (static QUERY / soft FOV floors). Tests that exercise
the DecisionTree pass an explicit ``model_path`` / ``model``.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_default_visibility_model_candidates(monkeypatch):
    monkeypatch.setattr(
        'cm_benchmark.generator.visibility_filters.DEFAULT_VISIBILITY_MODEL_CANDIDATES',
        (),
    )
    monkeypatch.delenv('CM_VISIBILITY_FILTER_MODEL', raising=False)


@pytest.fixture(autouse=True)
def _disable_named_frame_fraction(request):
    """Fixtures use small synthetic pixel counts to test other gates.

    The ``named_min_frame_fraction`` floor is off here; ``test_named_visibility``
    turns it on explicitly.
    """
    if request.node.get_closest_marker('frame_fraction_floor'):
        yield
        return
    from dataclasses import replace

    from cm_benchmark.generation.config import (
        get_generation_config,
        use_generation_config,
    )

    cfg = get_generation_config()
    off = replace(cfg, visibility=replace(cfg.visibility, named_min_frame_fraction=0.0))
    with use_generation_config(off):
        yield
