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
