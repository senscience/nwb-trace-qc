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


def test_evaluate_precedence_fail_wins():
    metrics = {"a": 100, "b": 0.5, "c": False}
    thresholds = {
        "a": {"fail_above": 50, "flag_above": 25},
        "b": {"flag_above": 0.1},
        "c": {"fail_if_false": True},
    }
    verdict, triggered = evaluate(metrics, thresholds)
    assert verdict == "fail"
    assert {t["metric"] for t in triggered} == {"a", "b", "c"}


def test_evaluate_all_pass():
    metrics = {"a": 10}
    thresholds = {"a": {"flag_above": 25}}
    verdict, triggered = evaluate(metrics, thresholds)
    assert verdict == "pass" and triggered == []
