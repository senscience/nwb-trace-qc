import math
import pytest

from nwb_trace_qc.thresholds import evaluate, evaluate_metric


def test_pass_when_within_range():
    rules = {"fail_above": 30, "flag_above": 25}
    v, r = evaluate_metric(20, rules)
    assert v == "pass" and r is None


def test_flag_above():
    rules = {"flag_above": 25, "fail_above": 30}
    v, r = evaluate_metric(27, rules)
    assert v == "flag" and ">" in r


def test_fail_above():
    rules = {"flag_above": 25, "fail_above": 30}
    v, r = evaluate_metric(35, rules)
    assert v == "fail" and ">" in r


def test_fail_below():
    rules = {"fail_below": 0, "flag_below": 10}
    v, r = evaluate_metric(-3, rules)
    assert v == "fail"


def test_fail_if_false_with_truthy():
    v, _ = evaluate_metric(True, {"fail_if_false": True})
    assert v == "pass"


def test_fail_if_false_with_falsy():
    v, _ = evaluate_metric(False, {"fail_if_false": True})
    assert v == "fail"


def test_nan_flags():
    v, r = evaluate_metric(float("nan"), {"flag_above": 10})
    assert v == "flag" and r == "nan"


def test_evaluate_precedence_fail_wins_when_metric_is_critical():
    """v0.6.0: only CRITICAL fails promote to cell-level fail."""
    metrics = {"a": 100, "b": 0.5, "c": False}
    thresholds = {
        "a": {"fail_above": 50, "flag_above": 25},
        "b": {"flag_above": 0.1},
        "c": {"fail_if_false": True},
    }
    # `a` is in the critical set → its fail promotes to a cell fail.
    verdict, triggered = evaluate(metrics, thresholds, critical_metrics={"a"})
    assert verdict == "fail"
    assert {t["metric"] for t in triggered} == {"a", "b", "c"}
    # And `critical: bool` is now on every triggered record
    by_metric = {t["metric"]: t for t in triggered}
    assert by_metric["a"]["critical"] is True
    assert by_metric["b"]["critical"] is False
    assert by_metric["c"]["critical"] is False


def test_evaluate_advisory_fail_demotes_to_flag():
    """When a fail is on a non-critical (advisory) metric only, the cell verdict
    is capped at flag — not fail. This is the v0.6.0 fix for 'all cells fail
    even though only peripheral metrics are bad'."""
    metrics = {"a": 100, "b": 10}
    thresholds = {
        "a": {"fail_above": 50},   # advisory (not in critical set)
        "b": {"flag_above": 5},    # advisory
    }
    verdict, triggered = evaluate(metrics, thresholds, critical_metrics={"c"})  # 'c' isn't even in thresholds
    # No critical fails fired; 'a' wanted to fail but it's advisory → flag.
    assert verdict == "flag"
    assert len(triggered) == 2


def test_evaluate_uses_default_critical_metrics_when_none_passed():
    """Passing `critical_metrics=None` falls back to families.DEFAULT_CRITICAL_METRICS."""
    # `vrest_mv` is in the default critical set; `held_vm_mv` is not.
    metrics = {"vrest_mv": -40, "held_vm_mv": -40}
    thresholds = {
        "vrest_mv": {"fail_above": -45},
        "held_vm_mv": {"fail_above": -45},
    }
    verdict, triggered = evaluate(metrics, thresholds)   # default critical set
    assert verdict == "fail"   # vrest_mv (critical) failed
    by_metric = {t["metric"]: t for t in triggered}
    assert by_metric["vrest_mv"]["critical"] is True
    assert by_metric["held_vm_mv"]["critical"] is False


def test_evaluate_all_pass():
    metrics = {"a": 10}
    thresholds = {"a": {"flag_above": 25}}
    verdict, triggered = evaluate(metrics, thresholds)
    assert verdict == "pass" and triggered == []
