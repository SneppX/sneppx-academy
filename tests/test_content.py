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
    "docs/modules/module03_quiz.md",
    "docs/modules/module04.md",
    "docs/modules/module04_quiz.md",
    "docs/certification.md",
    "examples/module01_tensor.py",
    "examples/module02_autograd_nn.py",
    "examples/module03_distributed.py",
    "examples/module04_security.py",
]


def test_expected_files_exist():
    missing = [p for p in EXPECTED if not (ROOT / p).exists()]
    assert not missing, f"missing: {missing}"


def test_syllabus_has_four_modules_published():
    text = (ROOT / "docs/syllabus.md").read_text(encoding="utf-8")
    for mod in range(1, 5):
        assert f"Module 0{mod}" in text
    lines = [l for l in text.splitlines() if l.startswith("Status:")]
    for line in lines:
        assert "published" in line.lower()


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


def test_module03_lesson_references_example_and_quiz():
    text = (ROOT / "docs/modules/module03.md").read_text(encoding="utf-8")
    assert "module03_distributed.py" in text
    assert "module03_quiz.md" in text
    assert "sneppx-dist" in text
    assert "nccl" in text.lower() or "gloo" in text.lower()


def test_module04_lesson_references_example_and_quiz():
    text = (ROOT / "docs/modules/module04.md").read_text(encoding="utf-8")
    assert "module04_security.py" in text
    assert "module04_quiz.md" in text
    assert "sneppx-shield" in text
    assert "sbom" in text.lower() or "sbom" in text


def test_quiz_has_questions_and_answer_key():
    for mod in ("module01", "module02", "module03", "module04"):
        q = (ROOT / f"docs/modules/{mod}_quiz.md").read_text(encoding="utf-8")
        low = q.lower()
        assert "q1" in low or "1." in low
        assert "answer" in low


def test_certification_mentions_shield_verify():
    text = (ROOT / "docs/certification.md").read_text(encoding="utf-8")
    low = text.lower()
    assert "ed25519" in low
    assert "sneppx-shield verify" in text
    assert "signed" in low and "certificate" in low


def test_example_compiles():
    for ex in ("module01_tensor.py", "module02_autograd_nn.py",
               "module03_distributed.py", "module04_security.py"):
        src = ROOT / f"examples/{ex}"
        compile(src.read_text(encoding="utf-8"), str(src), "exec")


def test_module03_example_uses_sneppx_dist():
    text = (ROOT / "examples/module03_distributed.py").read_text(encoding="utf-8")
    assert "sneppx_dist" in text or "sneppx-dist" in text


def test_module04_example_uses_sneppx_shield():
    text = (ROOT / "examples/module04_security.py").read_text(encoding="utf-8")
    assert "sneppx_shield" in text or "sneppx-shield" in text