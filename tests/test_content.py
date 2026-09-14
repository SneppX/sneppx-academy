import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

EXPECTED = [
    "docs/syllabus.md",
    "docs/modules/module01.md",
    "docs/modules/module01_quiz.md",
    "docs/modules/module02.md",
    "docs/modules/module02_quiz.md",
    "docs/modules/module03.md",
    "docs/modules/module04.md",
    "docs/certification.md",
    "examples/module01_tensor.py",
    "examples/module02_autograd_nn.py",
]


def test_expected_files_exist():
    missing = [p for p in EXPECTED if not (ROOT / p).exists()]
    assert not missing, f"missing: {missing}"


def test_syllabus_has_four_modules_with_status():
    text = (ROOT / "docs/syllabus.md").read_text(encoding="utf-8")
    assert "outlined" in text
    assert "published" in text
    for mod in range(1, 5):
        assert f"Module 0{mod}" in text


def test_module01_lesson_references_example_and_quiz():
    text = (ROOT / "docs/modules/module01.md").read_text(encoding="utf-8")
    assert "module01_tensor.py" in text
    assert "module01_quiz.md" in text
    assert "autograd" in text.lower()


def test_module02_lesson_references_example_and_quiz():
    text = (ROOT / "docs/modules/module02.md").read_text(encoding="utf-8")
    assert "module02_autograd_nn.py" in text
    assert "module02_quiz.md" in text
    assert "create_graph" in text
    assert "no_grad" in text


def test_quiz_has_questions_and_answer_key():
    m1 = (ROOT / "docs/modules/module01_quiz.md").read_text(encoding="utf-8")
    assert "## Questions" in m1
    assert "## Answer key" in m1
    m2 = (ROOT / "docs/modules/module02_quiz.md").read_text(encoding="utf-8")
    low2 = m2.lower()
    assert "q1." in low2
    assert "answer" in low2 and "key" in low2


def test_certification_mentions_shield_verify():
    text = (ROOT / "docs/certification.md").read_text(encoding="utf-8")
    low = text.lower()
    assert "ed25519" in low
    assert "sneppx-shield verify" in text
    assert "signed" in low and "certificate" in low


def test_example_compiles():
    for ex in ("module01_tensor.py", "module02_autograd_nn.py"):
        src = ROOT / f"examples/{ex}"
        compile(src.read_text(encoding="utf-8"), str(src), "exec")


def test_example_mentions_sneppx_alg():
    for ex in ("module01_tensor.py", "module02_autograd_nn.py"):
        text = (ROOT / f"examples/{ex}").read_text(encoding="utf-8")
        assert "SNEPPX_ALG_PATH" in text or "PYTHONPATH" in text


def test_placeholder_modules_are_placeholders():
    for mod in ("module03", "module04"):
        text = (ROOT / f"docs/modules/{mod}.md").read_text(encoding="utf-8")
        assert text.strip(), f"{mod}.md is empty"