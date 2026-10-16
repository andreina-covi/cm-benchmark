"""WordNet confusable-object grouping (SPOC ``is_hypernym_of`` + direct hypernyms).

Two objects are confusable when they share ``obj-synset``, one synset is a
hypernym of the other, or they share a direct hypernym. Question text uses
``obj-type`` split into words, not the raw simulator string.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Optional, Union

from cm_benchmark.generation.constructs import object_type_from_id, words_from_obj_type


@lru_cache(maxsize=10_000)
def is_hypernym_of(synset: Union[str, object], possible_hypernym: Union[str, object]) -> bool:
    """Port of SPOC ``utils/synset_utils.is_hypernym_of`` (nltk wordnet2022)."""
    syn = _as_synset(synset)
    hyper = _as_synset(possible_hypernym)
    if syn is None or hyper is None:
        return False
    return hyper in syn.lowest_common_hypernyms(hyper)


def are_confusable(
    syn_a: Optional[str],
    syn_b: Optional[str],
    *,
    type_a: Optional[str] = None,
    type_b: Optional[str] = None,
) -> bool:
    """True if the two catalog rows name the same thing for a human viewer."""
    a = _norm_syn(syn_a)
    b = _norm_syn(syn_b)
    if a and b:
        if a == b:
            return True
        try:
            if is_hypernym_of(a, b) or is_hypernym_of(b, a):
                return True
            if _share_direct_hypernym(a, b):
                return True
        except (LookupError, ValueError, AttributeError, ImportError, ModuleNotFoundError):
            return a == b
        return False
    ta = (type_a or '').strip().lower()
    tb = (type_b or '').strip().lower()
    return bool(ta and tb and ta == tb)


def synset_of(episode: Optional[dict], obj_id: str, odata: Optional[dict] = None) -> Optional[str]:
    """``obj-synset`` from the row, the catalog, or None."""
    if isinstance(odata, dict):
        syn = odata.get('synset') or odata.get('obj-synset')
        if syn:
            return str(syn)
    catalog = ((episode or {}).get('object_catalog') or {}).get(obj_id) or {}
    syn = catalog.get('synset') or catalog.get('obj-synset')
    return str(syn) if syn else None


def type_of(episode: Optional[dict], obj_id: str, odata: Optional[dict] = None) -> str:
    """Display type for questions (word-split ``obj-type``)."""
    vis = {obj_id: odata} if isinstance(odata, dict) else None
    if vis is None:
        catalog = ((episode or {}).get('object_catalog') or {}).get(obj_id) or {}
        if catalog:
            vis = {obj_id: {'category': catalog.get('category') or catalog.get('obj-type')}}
    return object_type_from_id(obj_id, vis)


def objects_are_confusable(episode: dict, oid_a: str, oid_b: str, odata_a=None, odata_b=None) -> bool:
    """Confusable using catalog synsets / types on ``episode``."""
    if oid_a == oid_b:
        return True
    return are_confusable(
        synset_of(episode, oid_a, odata_a),
        synset_of(episode, oid_b, odata_b),
        type_a=_raw_type(episode, oid_a, odata_a),
        type_b=_raw_type(episode, oid_b, odata_b),
    )


def confusable_components(episode: dict, obj_ids: Iterable[str]) -> list[set[str]]:
    """Connected components under the confusable relation."""
    ids = [oid for oid in obj_ids if oid]
    parent = {oid: oid for oid in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if objects_are_confusable(episode, a, b):
                pa, pb = find(a), find(b)
                if pa != pb:
                    parent[pb] = pa
    groups: dict[str, set[str]] = {}
    for oid in ids:
        groups.setdefault(find(oid), set()).add(oid)
    return list(groups.values())


def confusable_group_of(episode: dict, obj_id: str, pool: Iterable[str]) -> set[str]:
    """``obj_id`` plus every pool member confusable with it."""
    out = {obj_id}
    for oid in pool:
        if oid and objects_are_confusable(episode, obj_id, oid):
            out.add(oid)
    return out


def group_is_unique(episode: dict, obj_id: str, pool: Iterable[str]) -> bool:
    """True when no other id in ``pool`` is confusable with ``obj_id``."""
    return len(confusable_group_of(episode, obj_id, pool)) == 1


def _raw_type(episode: Optional[dict], obj_id: str, odata: Optional[dict]) -> Optional[str]:
    if isinstance(odata, dict):
        cat = odata.get('category') or odata.get('obj-type')
        if cat:
            return str(cat)
    catalog = ((episode or {}).get('object_catalog') or {}).get(obj_id) or {}
    cat = catalog.get('category') or catalog.get('obj-type')
    return str(cat) if cat else object_type_from_id(obj_id)


def _norm_syn(syn: Optional[str]) -> Optional[str]:
    if syn is None:
        return None
    text = str(syn).strip()
    return text or None


_WN = None
_WN_TRIED = False


def _wordnet():
    """Import wordnet once. Missing NLTK still yields None, as before."""
    global _WN, _WN_TRIED
    if _WN_TRIED:
        return _WN
    _WN_TRIED = True
    try:
        from nltk.corpus import wordnet2022 as wn
    except (ImportError, ModuleNotFoundError, LookupError):
        _WN = None
        return None
    _WN = wn
    return _WN


@lru_cache(maxsize=10_000)
def _load_synset(name: str):
    wn = _wordnet()
    if wn is None:
        return None
    # LookupError propagates so ``are_confusable`` can fall back to equality.
    return wn.synset(name)


def _as_synset(value):
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    if _wordnet() is None:
        return None
    return _load_synset(value)


def _share_direct_hypernym(syn_a: str, syn_b: str) -> bool:
    a = _as_synset(syn_a)
    b = _as_synset(syn_b)
    if a is None or b is None:
        return False
    return bool(set(a.hypernyms()) & set(b.hypernyms()))
