import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

EXPECTED = [
    "docs/syllabus.md",
    "docs/modules/module01.md",
    "docs/modules/module01_quiz.md",
    "docs/modules/module02.md",
    "docs/modules/module03.md",
    "docs/modules/module04.md",
    "examples/module01_tensor.py",
]


def test_expected_files_exist():
    missing = [p for p in EXPECTED if not (ROOT / p).exists()]
    assert not missing, f"missing: {missing}"


def test_syllabus_has_four_modules_with_status():
    text = (ROOT / "docs/syllabus.md").read_text(encoding="utf-8")
    assert "outlined" in text
    for mod in range(1, 5):
        assert f"Module 0{mod}" in text


def test_module01_lesson_references_example_and_quiz():
    text = (ROOT / "docs/modules/module01.md").read_text(encoding="utf-8")
    assert "module01_tensor.py" in text
    assert "module01_quiz.md" in text
    assert "autograd" in text.lower()


def test_quiz_has_questions_and_answer_key():
    text = (ROOT / "docs/modules/module01_quiz.md").read_text(encoding="utf-8")
    assert "## Questions" in text
    assert "## Answer key" in text
    assert text.count("## Answer key") == 1


def test_example_compiles():
    src = ROOT / "examples/module01_tensor.py"
    compile(src.read_text(encoding="utf-8"), str(src), "exec")


def test_example_mentions_sneppx_alg():
    text = (ROOT / "examples/module01_tensor.py").read_text(encoding="utf-8")
    assert "SNEPPX_ALG_PATH" in text or "PYTHONPATH" in text


def test_placeholder_modules_are_placeholders():
    for mod in ("module02", "module03", "module04"):
        text = (ROOT / f"docs/modules/{mod}.md").read_text(encoding="utf-8")
        assert text.strip(), f"{mod}.md is empty"