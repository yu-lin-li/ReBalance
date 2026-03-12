from __future__ import annotations

import base64
import json
import pickle
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from datasets import load_dataset


FORMATTING_MESSAGE_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution to the problem and "
    "enclose your code within delimiters."
)

FORMATTING_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to stdout "
    "(do not directly test on the sample inputs). Enclose your code within delimiters as follows. "
    "Ensure that when the python program runs, it reads the inputs, runs the algorithm and "
    "writes output to STDOUT."
)

# ============================== helpers ============================== #

def _ensure_enum(enum_cls: type[Enum], v: Any, default: Enum | None = None) -> Enum:
    """Robust enum coercion from Enum instance / value string / name string."""
    if isinstance(v, enum_cls):
        return v
    if isinstance(v, str):
        s = v.strip()
        # try by value (case-insensitive)
        for member in enum_cls:
            if str(member.value).lower() == s.lower():
                return member
        # try by name (case-insensitive)
        for member in enum_cls:
            if member.name.lower() == s.lower():
                return member
    # try direct constructor
    try:
        return enum_cls(v)
    except Exception:
        if default is not None:
            return default
        return next(iter(enum_cls))


def _coerce_datetime(x: Any) -> datetime:
    """
    Accepts ISO str / str ending with 'Z' / unix ts (int/float) / dict with timestamp / datetime.
    Returns a timezone-aware UTC datetime. On failure, returns epoch (1970-01-01 UTC).
    """
    if x is None:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    if isinstance(x, datetime):
        # ensure tz-aware; assume UTC if naive
        return x if x.tzinfo else x.replace(tzinfo=timezone.utc)
    if isinstance(x, (int, float)):
        try:
            return datetime.fromtimestamp(x, tz=timezone.utc)
        except Exception:
            return datetime.fromtimestamp(0, tz=timezone.utc)
    if isinstance(x, str):
        s = x.strip()
        try:
            if s.endswith("Z"):
                return datetime.fromisoformat(s.replace("Z", "+00:00"))
            return datetime.fromisoformat(s)
        except Exception:
            # try a couple common formats
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
                try:
                    return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
                except Exception:
                    pass
            return datetime.fromtimestamp(0, tz=timezone.utc)
    if isinstance(x, dict):
        # common numeric keys
        for k in ("timestamp", "time", "seconds", "secs"):
            v = x.get(k)
            if isinstance(v, (int, float)):
                try:
                    return datetime.fromtimestamp(v, tz=timezone.utc)
                except Exception:
                    pass
        # common string keys
        for k in ("created_at", "updated_at", "date", "datetime"):
            v = x.get(k)
            if isinstance(v, str):
                try:
                    return datetime.fromisoformat(v.replace("Z", "+00:00"))
                except Exception:
                    pass
    return datetime.fromtimestamp(0, tz=timezone.utc)


def _ensure_str(x: Any) -> str:
    """Guarantee string fields (e.g., input/output) are strings."""
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False)


def _maybe_json_load(x: Any) -> Any:
    """If x is a JSON string, parse; if already obj, return as is; else return original."""
    if isinstance(x, str):
        xs = x.strip()
        if xs == "":
            return None
        try:
            return json.loads(xs)
        except Exception:
            return x
    return x


def _coerce_test_list(raw: Any) -> list[dict]:
    """
    Accepts:
      - list[dict] or list[Test]-like
      - JSON string of list[dict]
      - (for private) base64+zlib+pickle packed JSON string
    Returns a list of plain dicts for dataclass construction.
    """
    if isinstance(raw, list):
        out: list[dict] = []
        for t in raw:
            if isinstance(t, dict):
                out.append(t)
            elif hasattr(t, "__dict__"):
                out.append(dict(t.__dict__))  # not expected but safe
        return out

    if isinstance(raw, str):
        s = raw.strip()
        # try JSON first
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return _coerce_test_list(obj)
        except Exception:
            pass
        # try base64+zlib+pickle -> possibly JSON string -> list
        try:
            decoded = base64.b64decode(s.encode("utf-8"))
            decomp = zlib.decompress(decoded)
            obj = pickle.loads(decomp)
            if isinstance(obj, (bytes, bytearray)):
                obj = obj.decode("utf-8", errors="ignore")
            if isinstance(obj, str):
                obj = json.loads(obj)
            if isinstance(obj, list):
                return _coerce_test_list(obj)
        except Exception:
            pass

    # fallback: empty list
    return []

# ============================== enums ============================== #

class Platform(Enum):
    LEETCODE = "leetcode"
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TestType(Enum):
    STDIN = "stdin"
    FUNCTIONAL = "functional"

# ============================ dataclasses =========================== #

@dataclass
class Test:
    input: str
    output: str
    testtype: TestType

    def __post_init__(self):
        self.testtype = _ensure_enum(TestType, self.testtype, default=TestType.STDIN)
        self.input = _ensure_str(self.input)
        self.output = _ensure_str(self.output)
        # Uncomment to enable FUNCTIONAL structured I/O as needed:
        # if self.testtype == TestType.FUNCTIONAL:
        #     self.input = _maybe_json_load(self.input)
        #     self.output = _maybe_json_load(self.output)


@dataclass
class CodeGenerationProblem:
    question_title: str
    question_content: str
    platform: Platform
    question_id: str
    contest_id: str
    contest_date: datetime
    starter_code: str
    difficulty: Difficulty
    public_test_cases: list[Test]
    private_test_cases: list[Test]
    metadata: dict

    def __post_init__(self):
        # enums
        self.platform = _ensure_enum(Platform, self.platform, default=Platform.LEETCODE)
        self.difficulty = _ensure_enum(Difficulty, self.difficulty, default=Difficulty.MEDIUM)

        # datetime
        self.contest_date = _coerce_datetime(self.contest_date)

        # tests
        pub = _coerce_test_list(self.public_test_cases)
        self.public_test_cases = [Test(**t) for t in pub]

        prv = _coerce_test_list(self.private_test_cases)
        self.private_test_cases = [Test(**t) for t in prv]

        # metadata
        md = _maybe_json_load(self.metadata)
        self.metadata = md if isinstance(md, dict) else {}

        # strings (ensure)
        self.question_title = _ensure_str(self.question_title)
        self.question_content = _ensure_str(self.question_content)
        self.starter_code = _ensure_str(self.starter_code)
        self.question_id = _ensure_str(self.question_id)
        self.contest_id = _ensure_str(self.contest_id)

    def insert_output(self, output_list: list[str], code_list: list[str]) -> dict:
        return {
            "question_title": self.question_title,
            "question_content": self.question_content,
            "platform": self.platform.value,
            "question_id": self.question_id,
            "contest_id": self.contest_id,
            "contest_date": self.contest_date.isoformat(),
            "starter_code": self.starter_code,
            "difficulty": self.difficulty.value,
            "output_list": output_list,
            "code_list": code_list,
        }

    def insert_output_evaluation(
        self,
        output_list: list[str],
        code_list: list[str],
        graded_list: list[bool],
        **kwargs,
    ) -> dict:
        output = self.insert_output(output_list, code_list)
        output["graded_list"] = graded_list
        denom = max(1, len(graded_list))
        output["pass@1"] = graded_list.count(True) / denom
        for k, v in kwargs.items():
            output[k] = v
        return output

    def get_evaluation_sample(self):
        return {
            "input_output": json.dumps(
                {
                    "inputs": [t.input for t in self.public_test_cases + self.private_test_cases],
                    "outputs": [t.output for t in self.public_test_cases + self.private_test_cases],
                    "fn_name": self.metadata.get("func_name", None),
                },
                ensure_ascii=False,
            ),
        }

# =============================== loader ============================= #

def load_code_generation_dataset(
    release_version: str = "release_v1",
    local_path: str = "/home/tutengyao/Eff_Reasoning/Data/Code_livecodebenchv1/test.jsonl",
) -> list[CodeGenerationProblem]:
    """
    Load from local JSONL files (one sample per line) with robust parsing.
    - The `release_version` argument is kept only for backward-compatibility; local loading does not use it.
    - If you need a custom path, modify `local_path` or override it via external parameters.
    """
    ds = load_dataset("json", data_files={"test": local_path}, split="test")
    problems = [CodeGenerationProblem(**p) for p in ds]  # type: ignore
    print(f"Loaded {len(problems)} problems")
    problems = sorted(problems, key=lambda x: x.question_id)
    return problems

# =============================== prompts ============================ #

def get_deepseekcode_question_template_answer(question: CodeGenerationProblem):
    prompt = (
        "### Instruction: You will be given a question (problem specification) and will "
        "generate a correct Python program that matches the specification and passes all tests. "
        "You will NOT return anything except for the program.\n\n"
    )
    prompt += f"Question:\n{question.question_content}\n\n"
    if question.starter_code:
        prompt += f"### Instruction: {FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{question.starter_code}\n```\n\n"
    else:
        prompt += f"### Instruction: {FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Response:\n\n"
    return prompt

# =============================== utils ============================== #

def extract_code(model_output: str):
    outputlines = model_output.split("\n")
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        return ""
    # return "\n".join(outputlines[indexlines[0] + 1 : indexlines[1]])
    return "\n".join(outputlines[indexlines[-2] + 1 : indexlines[-1]])


def extract_instance_results(results):
    instance_wise_grades = {}
    for task_id, res in results.items():
        instance_wise_grades[task_id] = []
        for generation in res:
            instance_wise_grades[task_id].append(all([g > 0 for g in generation]))

    instance_wise_grades = [
        v for _, v in sorted(instance_wise_grades.items(), key=lambda item: item[0])
    ]
    return instance_wise_grades
