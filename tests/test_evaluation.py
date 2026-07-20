import pytest

from beam_sls.evaluation import resolve_evaluation_plan


def test_evaluation_matrix_expands_cases_and_references():
    cfg = {
        "evaluation": {
            "matrix": {
                "full_gamma": ["greedy"],
                "baseline": {"greedy": True, "exhaustive": False},
                "topk_conflict_id": ["greedy", "hard_conflict_greedy"],
            },
            "references": {
                "oracle": "full_gamma__greedy",
                "baseline": "baseline__greedy",
            },
        }
    }

    plan = resolve_evaluation_plan(cfg)

    assert plan.case_ids == [
        "full_gamma__greedy",
        "baseline__greedy",
        "topk_conflict_id__greedy",
        "topk_conflict_id__hard_conflict_greedy",
    ]
    assert plan.feedback_schemes == ["full_gamma", "baseline", "topk_conflict_id"]
    assert plan.references["oracle"] == "full_gamma__greedy"


def test_evaluation_matrix_rejects_missing_report_capability():
    cfg = {
        "evaluation": {
            "matrix": {"full_gamma": ["hard_conflict_greedy"]},
        }
    }

    with pytest.raises(ValueError, match="missing capabilities=.*conflict_ids"):
        resolve_evaluation_plan(cfg)


def test_legacy_config_keeps_scheme_as_metric_case_id():
    cfg = {
        "feedback": {"schemes": ["baseline"]},
        "scheduler": {"algorithm": "greedy"},
    }

    plan = resolve_evaluation_plan(cfg)

    assert plan.case_ids == ["baseline"]
    assert plan.cases[0].algorithm == "greedy"
