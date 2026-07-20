from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence


SCHEME_CAPABILITIES = {
    "full_gamma": {"service_quality", "full_interference_matrix"},
    "baseline": {"service_quality"},
    "topk_conflict_id": {"service_quality", "conflict_ids"},
    "threshold_conflict_set": {"service_quality", "conflict_ids"},
}

ALGORITHM_REQUIREMENTS = {
    "greedy": {"service_quality"},
    "exhaustive": {"service_quality"},
    "hard_conflict_greedy": {"conflict_ids"},
    "adaptive_lambda_greedy": {"conflict_ids"},
}


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    feedback_scheme: str
    algorithm: str
    label: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "case_id": self.case_id,
            "feedback_scheme": self.feedback_scheme,
            "algorithm": self.algorithm,
            "label": self.label,
        }


@dataclass(frozen=True)
class EvaluationPlan:
    cases: List[EvaluationCase]
    references: Dict[str, str]

    @property
    def feedback_schemes(self) -> List[str]:
        return list(dict.fromkeys(case.feedback_scheme for case in self.cases))

    @property
    def case_ids(self) -> List[str]:
        return [case.case_id for case in self.cases]

    @property
    def cases_by_id(self) -> Dict[str, EvaluationCase]:
        return {case.case_id: case for case in self.cases}


def _algorithms_from_matrix_row(scheme: str, value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        if any(not isinstance(v, bool) for v in value.values()):
            raise ValueError(
                f"evaluation.matrix.{scheme} mapping values must be true/false"
            )
        return [str(algorithm) for algorithm, enabled in value.items() if enabled]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(algorithm) for algorithm in value]
    raise ValueError(
        f"evaluation.matrix.{scheme} must be an algorithm list or a true/false mapping"
    )


def validate_evaluation_case(scheme: str, algorithm: str) -> None:
    if scheme not in SCHEME_CAPABILITIES:
        raise ValueError(
            f"Unknown feedback scheme {scheme!r}; choose from "
            f"{', '.join(SCHEME_CAPABILITIES)}"
        )
    if algorithm not in ALGORITHM_REQUIREMENTS:
        raise ValueError(
            f"Unknown scheduler algorithm {algorithm!r}; choose from "
            f"{', '.join(ALGORITHM_REQUIREMENTS)}"
        )
    missing = ALGORITHM_REQUIREMENTS[algorithm] - SCHEME_CAPABILITIES[scheme]
    if missing:
        raise ValueError(
            "Invalid evaluation combination: "
            f"scheme={scheme}, algorithm={algorithm}; "
            f"missing capabilities={sorted(missing)}"
        )


def _make_case(scheme: str, algorithm: str) -> EvaluationCase:
    validate_evaluation_case(scheme, algorithm)
    case_id = f"{scheme}__{algorithm}"
    return EvaluationCase(
        case_id=case_id,
        feedback_scheme=scheme,
        algorithm=algorithm,
        label=f"{scheme} / {algorithm}",
    )


def _legacy_cases(cfg: Dict[str, Any]) -> List[EvaluationCase]:
    schemes = list(cfg.get("feedback", {}).get("schemes", ["baseline"]))
    if not schemes:
        raise ValueError("feedback.schemes must contain at least one scheme")
    default_algorithm = str(cfg.get("scheduler", {}).get("algorithm", "greedy"))
    by_scheme = cfg.get("scheduler", {}).get("algorithm_by_scheme", {}) or {}
    if not isinstance(by_scheme, Mapping):
        raise ValueError("scheduler.algorithm_by_scheme must be a mapping")
    cases: List[EvaluationCase] = []
    for raw_scheme in schemes:
        scheme = str(raw_scheme)
        algorithm = str(by_scheme.get(scheme, default_algorithm))
        validate_evaluation_case(scheme, algorithm)
        # Preserve historical metric keys when the new matrix is not used.
        cases.append(EvaluationCase(scheme, scheme, algorithm, scheme))
    return cases


def _matrix_cases(matrix: Mapping[str, Any]) -> List[EvaluationCase]:
    cases: List[EvaluationCase] = []
    for raw_scheme, row in matrix.items():
        scheme = str(raw_scheme)
        algorithms = _algorithms_from_matrix_row(scheme, row)
        for algorithm in algorithms:
            cases.append(_make_case(scheme, algorithm))
    if not cases:
        raise ValueError("evaluation.matrix must enable at least one evaluation case")
    return cases


def _unique_reference(cases: Iterable[EvaluationCase], scheme: str) -> str | None:
    matches = [case.case_id for case in cases if case.feedback_scheme == scheme]
    return matches[0] if len(matches) == 1 else None


def resolve_evaluation_plan(cfg: Dict[str, Any]) -> EvaluationPlan:
    evaluation = cfg.get("evaluation", {}) or {}
    matrix = evaluation.get("matrix")
    if matrix is None:
        cases = _legacy_cases(cfg)
    else:
        if not isinstance(matrix, Mapping):
            raise ValueError("evaluation.matrix must be a mapping of scheme to algorithms")
        cases = _matrix_cases(matrix)

    case_ids = [case.case_id for case in cases]
    duplicates = sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate evaluation cases: {', '.join(duplicates)}")

    configured_refs = evaluation.get("references", {}) or {}
    if not isinstance(configured_refs, Mapping):
        raise ValueError("evaluation.references must be a mapping")
    references: Dict[str, str] = {}
    for name, default_scheme in (("oracle", "full_gamma"), ("baseline", "baseline")):
        configured = configured_refs.get(name)
        resolved = str(configured) if configured is not None else _unique_reference(cases, default_scheme)
        if resolved is not None:
            if resolved not in case_ids:
                raise ValueError(
                    f"evaluation.references.{name}={resolved!r} is not an enabled case"
                )
            references[name] = resolved

    return EvaluationPlan(cases=cases, references=references)
