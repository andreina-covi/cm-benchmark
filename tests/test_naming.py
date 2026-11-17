"""naming.rename display names, naming.unnamed types, same-name confusability."""

from cm_benchmark.generation.constructs import never_named, object_type_from_id
from cm_benchmark.generation.synsets import objects_are_confusable


def _vis(cat):
    return {'category': cat}


def test_rename_applies_to_category_and_id_stem():
    assert object_type_from_id('television|7|1|0', {'television|7|1|0': _vis('TVStand')}) == 'TV Stand'
    assert object_type_from_id('Ashcan|2|3') == 'Trash Can'
    assert object_type_from_id('ObjaPlacard|7|wall|0') == 'Poster'
    # No entry: plain word split
    assert object_type_from_id('AlarmClock|1|2') == 'Alarm Clock'


def test_unnamed_types_and_subparts_are_never_named():
    episode = {'object_catalog': {'ObjaMachine|1|2': {'category': 'Machine'}, 'Chair|1|3': {'category': 'Chair'}}}
    assert never_named(episode, 'ObjaMachine|1|2')
    assert never_named(episode, 'television|7|1|0___1')
    assert not never_named(episode, 'Chair|1|3')


def test_types_sharing_a_display_name_are_confusable():
    episode = {
        'object_catalog': {
            'Hardback|1|1': {'category': 'Hardback', 'synset': 'hardback.n.01'},
            'Book|1|2': {'category': 'Book', 'synset': 'book.n.02'},
        }
    }
    assert objects_are_confusable(episode, 'Hardback|1|1', 'Book|1|2')
