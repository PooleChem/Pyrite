"""Defaults that must agree across implementations."""

import inspect

from pyrite.scoring import protein
from pyrite.scoring._base import ScoringFunction


def _scoring_functions(module) -> dict:
    found = {}
    for name, member in inspect.getmembers(module, inspect.isclass):
        try:  # a typing alias (``NDArray``) passes ``isclass`` but is not a class
            is_scoring_function = issubclass(member, ScoringFunction)
        except TypeError:
            continue
        if is_scoring_function:
            found[name] = member
    return found


def test_every_neighbour_scoring_function_has_the_same_default_k():
    defaults = {
        name: inspect.signature(cls.__init__).parameters["k"].default
        for name, cls in _scoring_functions(protein).items()
        if "k" in inspect.signature(cls.__init__).parameters
    }

    assert len(defaults) >= 10  # the whole family was found
    assert set(defaults.values()) == {100}, defaults
