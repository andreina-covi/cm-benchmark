"""McNemar / Kendall helpers used by scripts/ablation_report.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _mod():
    path = REPO / 'scripts' / 'ablation_report.py'
    spec = importlib.util.spec_from_file_location('ablation_report', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mcnemar_symmetric_is_one():
    m = _mod()
    chi2, p = m._mcnemar(0, 0)
    assert chi2 == 0.0 and p == 1.0
    chi2, p = m._mcnemar(5, 5)
    assert chi2 == 0.0
    assert p == 1.0


def test_kendall_tau_identical_and_reversed():
    m = _mod()
    tau, conc, disc = m._kendall_tau([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert tau == 1.0 and disc == 0 and conc == 3
    tau, conc, disc = m._kendall_tau([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
    assert tau == -1.0 and conc == 0


def test_overlay_ablation_configs_load():
    from cm_benchmark.generation.config import load_generation_config

    for name, setting, marks in (
        ('current', 'current', False),
        ('sr3d', 'sr3d', False),
        ('vsi', 'vsi', False),
        ('strict', 'strict', False),
        ('all_marks', 'current', True),
    ):
        cfg = load_generation_config(REPO / 'configs' / 'ablations' / f'{name}.yaml')
        assert cfg.name == name
        assert cfg.disambiguation.proximity_setting == setting
        assert cfg.disambiguation.all_marks is marks
