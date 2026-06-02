from nwb_trace_qc.stimuli import StimulusFamilyMap


def test_family_of_known_name():
    fm = StimulusFamilyMap({"x": ["A", "B"], "y": ["C"]})
    assert fm.family_of("A") == "x"
    assert fm.family_of("B") == "x"
    assert fm.family_of("C") == "y"


def test_case_insensitive_lookup():
    fm = StimulusFamilyMap({"x": ["IDRest"]})
    assert fm.family_of("idrest") == "x"
    assert fm.family_of("IDREST") == "x"


def test_unknown_name_returns_none():
    fm = StimulusFamilyMap({"x": ["A"]})
    assert fm.family_of("Z") is None
    assert fm.family_of(None) is None


def test_names_for_family():
    fm = StimulusFamilyMap({"x": ["A", "B"]})
    assert sorted(fm.names_for("x")) == ["A", "B"]
    assert fm.names_for("missing") == []
