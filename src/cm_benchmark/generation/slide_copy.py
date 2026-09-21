"""Slide/review copy for class-4 items. No planner or nav_graph imports."""

from __future__ import annotations

from typing import Optional


def compact_arrow_list(parts, *, max_show: int = 8) -> str:
    items = [str(p).strip() for p in (parts or []) if str(p).strip()]
    if not items:
        return '—'
    if len(items) <= max_show:
        return ' → '.join(items)
    hidden = len(items) - 5
    return f"{' → '.join(items[:3])} → …{hidden}… → {' → '.join(items[-2:])}"


def action_parts(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.replace('->', '→')
        return [p.strip() for p in text.split('→') if p.strip()]
    if isinstance(raw, (list, tuple)):
        out: list[str] = []
        for entry in raw:
            if isinstance(entry, dict):
                act = entry.get('action')
                if act:
                    out.append(str(act))
            elif entry is not None:
                out.append(str(entry))
        return out
    return [str(raw)]


def class4_slide_panel(
    construct: str,
    context: Optional[dict] = None,
    *,
    answer: Optional[str] = None,
) -> dict[str, str]:
    """Human-readable class-4 summary: source, goal, path, scoring."""
    ctx = context or {}
    src = ctx.get('source') or '?'
    goal = ctx.get('goal') or '?'
    nodes = ctx.get('path_nodes') or []
    hops = ctx.get('hop_count')
    if hops is None and len(nodes) >= 2:
        hops = len(nodes) - 1
    actions = action_parts(ctx.get('action_sequence') or answer)
    path_txt = compact_arrow_list(nodes)
    act_txt = compact_arrow_list(actions)
    hop_bit = f'{hops} hops' if hops is not None else 'path'
    if construct == 'survey_based_route_planning':
        return {
            'task': f'Plan (never walked): {src}  →  {goal}',
            'graph': (
                'Evidence: through-door glimpse · scored on viewed edges '
                f'({hop_bit} on the stored full-graph path)'
            ),
            'path': f'Shortest connecting path (analysis): {path_txt}',
            'actions': f'One valid action sequence (not exclusive gold): {act_txt}',
            'scoring': (
                'Score: success = near the goal; validity = stay on viewed edges '
                'and use at least one unwalked hop; efficiency = SPL vs viewed shortest path.'
            ),
        }
    return {
        'task': f'Retrace (walked): {src}  →  {goal}',
        'graph': f'Scored on traversed (experienced) edges · {hop_bit}',
        'path': f'Shortest walked path (analysis): {path_txt}',
        'actions': f'One valid action sequence (not exclusive gold): {act_txt}',
        'scoring': (
            'Score: success = near the goal; validity = stay on walked edges; '
            'efficiency = SPL. Headline = valid-success + route_efficiency.'
        ),
    }
