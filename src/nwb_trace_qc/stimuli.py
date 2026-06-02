"""Map stimulus names to families and back.

The pipeline reasons in terms of *families* (`spontaneous_hold`, `test_pulse`, etc.);
labs use whatever stimulus names they want. The mapping is the configurable layer.
"""
from __future__ import annotations


class StimulusFamilyMap:
    """Fast bidirectional name<->family lookup."""

    def __init__(self, families: dict[str, list[str]]):
        self.families = {k: list(v) for k, v in families.items()}
        self._name_to_family: dict[str, str] = {}
        for fam, names in families.items():
            for n in names:
                self._name_to_family[n.lower()] = fam

    def family_of(self, stimulus_name: str | None) -> str | None:
        if stimulus_name is None:
            return None
        return self._name_to_family.get(str(stimulus_name).lower())

    def names_for(self, family: str) -> list[str]:
        return list(self.families.get(family, []))

    def all_known_names(self) -> set[str]:
        return set(self._name_to_family)
