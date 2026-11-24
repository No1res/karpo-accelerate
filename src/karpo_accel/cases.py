"""One text choice per request, shared by the benchmark and exit calibration."""

import json
from dataclasses import dataclass


def schema_key(question):
    return question.get("instructions", ""), tuple(question["criteria"].items())


@dataclass(frozen=True, slots=True)
class Case:
    id: str
    request: dict
    question_id: str
    expected: str | None

    @property
    def question(self):
        return self.request["questions"][self.question_id]


def read_cases(path, *, labels=False):
    cases, ids = [], set()
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                case = parse_case(json.loads(line), str(number), labels=labels)
                if case.id in ids:
                    raise ValueError("duplicate sample ID")
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error
            ids.add(case.id)
            cases.append(case)
    if not cases:
        raise ValueError(f"No requests in {path}")
    return cases


def parse_case(row, default_id, *, labels=False):
    if not isinstance(row, dict) or not isinstance(row.get("request"), dict):
        raise ValueError("each row needs a request object")
    request = row["request"]
    if not isinstance(request.get("state"), str) or request.get("images") or request.get("assets"):
        raise ValueError("the request state must be text")
    questions = request.get("questions")
    if not isinstance(questions, dict) or len(questions) != 1:
        raise ValueError("the request must contain exactly one question")
    question_id, question = next(iter(questions.items()))
    if not isinstance(question, dict) or question.get("type") != "choice":
        raise ValueError("the question must have type choice")
    criteria = question.get("criteria")
    if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
        raise ValueError("choice criteria must have 2 to 255 ordered options")
    if any(not option or not isinstance(description, str) for option, description in criteria.items()):
        raise ValueError("every option must have a nonempty key and a text description")
    if not isinstance(question.get("instructions", ""), str):
        raise ValueError("instructions must be text")
    expected = row.get("expected")
    if (labels or expected is not None) and (not isinstance(expected, str) or expected not in criteria):
        raise ValueError("expected must be a key in the choice criteria")
    sample_id = row.get("id", default_id)
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("id must be a nonempty string")
    return Case(sample_id, request, question_id, expected)
