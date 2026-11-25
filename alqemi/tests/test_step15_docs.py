"""Step 15 tests: user documentation (cheatsheet, user guide, tutorials, ReadTheDocs).

These tests DEFINE Step 15. Settled with the user (2026-11-25): reStructuredText
(Sphinx), real numbers from Psi4 first and PySCF as a second program, the six
planned tutorials plus one for MP2, hosted on ReadTheDocs.

  Site layout (docs/)
    index.rst            toctree, in order: getting_started, cheatsheet,
                         user_guide/index, tutorials/index, api, developer_guide.
                         The old one-page user_guide.rst is gone.
    getting_started.rst  install, a first result in a few lines.
    cheatsheet.rst       one page of tables; names every public API name
                         (alqemi.__all__) at least once.
    user_guide/index.rst toctree, in order: indices, operators, pipeline, tensors,
                         hamiltonians, bch_projections, spin, codegen, latex, limits.
    tutorials/index.rst  toctree, in order: 01_commutator, 02_mp2, 03_ccd, 04_ccsd,
                         05_uhf_ccsd, 06_lambda_ccd, 07_latex. Also literalincludes the
                         two helpers, scripts/integrals.py and scripts/scf.py.
    tutorials/NN_x.rst   literalincludes its script tutorials/scripts/tutorialNN_x.py,
                         so the page shows exactly the code these tests run.
    developer_guide.rst  how to run the tests, build the docs, and the step workflow.
    No cookiecutter placeholder text is left ("You might choose to write",
    "This page details how to").

  Runnable examples
    getting_started.rst, cheatsheet.rst and every user guide page hold doctest
    examples (>>>); all of them pass under doctest with ELLIPSIS and
    NORMALIZE_WHITESPACE. Pages and scripts use only the public API: no
    "from alqemi.<submodule>" / "import alqemi.<submodule>", and every name a
    script imports from alqemi is in alqemi.__all__.

  Tutorial scripts (docs/tutorials/scripts)
    Plain Python files a reader can run. Each defines CHECKS, a list of
    (label, ALQEMI value, reference value), and these tests compare every entry
    themselves (a script cannot pass by skipping its own assert). Every SCF-program
    reference value (HF, MP2, CCSD, UHF, UHF-CCSD energies) is also recomputed by the
    test through scf.py and must agree to 1e-9, so a script cannot compare with
    itself. Required labels and tolerances:
      01 commutator   "matrix check"                        |diff| <= 1e-12
      02 mp2          "HF energy", "MP2 energy"              1e-8
      03 ccd          "residual terms" (18), "groups" (10),
                      "MP2 energy (1st step)"                1e-8
      04 ccsd         "CCSD energy"                          1e-8
      05 uhf_ccsd     "UHF energy", "UHF-CCSD energy"        1e-8
      06 lambda_ccd   "dE/d eps (z dipole)"                  1e-7  (finite field)
      07 latex        "pdflatex exit code" (only when pdflatex is installed); the
                      script writes ccd_equations.tex with an align* block and
                      (1 - P_{ij}) grouping.
    Tutorials 02-06 need an SCF program: they run once per program (Psi4, PySCF)
    chosen by the ALQEMI_PROGRAM environment variable, and are skipped when that
    program is not installed. Files a script writes go to ALQEMI_OUTPUT.

  Build and hosting
    sphinx-build -W (warnings are errors) succeeds when Sphinx and the theme
    packages are installed (skipped otherwise). conf.py has no "language = None".
    .readthedocs.yaml: version 2, sphinx configuration docs/conf.py with
    fail_on_warning: true, conda environment docs/requirements.yaml (which needs
    no Psi4/PySCF: pages include the scripts, they do not run them).
    pyproject.toml lists Documentation = https://alqemi.readthedocs.io/ and the
    README links to it. CI has a docs job: an env from
    devtools/conda-envs/docs_env.yaml (Psi4, PySCF, Sphinx stack) that runs this
    file and sphinx-build -W.
"""
from __future__ import annotations

import doctest
import importlib.util
import os
import re
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import alqemi

REPO_ROOT = Path(alqemi.__file__).resolve().parent.parent
DOCS = REPO_ROOT / "docs"
SCRIPTS = DOCS / "tutorials" / "scripts"

pytestmark = pytest.mark.skipif(not DOCS.is_dir(), reason="needs a source checkout with docs/")

TOP_PAGES = ["getting_started", "cheatsheet", "user_guide/index", "tutorials/index", "api", "developer_guide"]
GUIDE_PAGES = ["indices", "operators", "pipeline", "tensors", "hamiltonians", "bch_projections",
               "spin", "codegen", "latex", "limits"]
TUTORIALS = ["01_commutator", "02_mp2", "03_ccd", "04_ccsd", "05_uhf_ccsd", "06_lambda_ccd", "07_latex"]
DOCTEST_PAGES = ["getting_started.rst", "cheatsheet.rst"] + [f"user_guide/{p}.rst" for p in GUIDE_PAGES]
NEEDS_SCF = {"02_mp2", "03_ccd", "04_ccsd", "05_uhf_ccsd", "06_lambda_ccd"}

CHECK_SPEC = {
    "01_commutator": {"matrix check": 1e-12},
    "02_mp2": {"HF energy": 1e-8, "MP2 energy": 1e-8},
    "03_ccd": {"residual terms": 0, "groups": 0, "MP2 energy (1st step)": 1e-8},
    "04_ccsd": {"CCSD energy": 1e-8},
    "05_uhf_ccsd": {"UHF energy": 1e-8, "UHF-CCSD energy": 1e-8},
    "06_lambda_ccd": {"dE/d eps (z dipole)": 1e-7},
    "07_latex": {},
}


def _script(tutorial: str) -> Path:
    number, name = tutorial.split("_", 1)
    return SCRIPTS / f"tutorial{number}_{name}.py"


def _toctree(path: Path):
    """Entries of the first toctree in an rst file, in order."""
    lines = path.read_text().splitlines()
    start = next(n for n, line in enumerate(lines) if line.strip().startswith(".. toctree::"))
    entries = []
    for line in lines[start + 1:]:
        if not line.strip():
            if entries:
                break
            continue
        if not line.startswith((" ", "\t")):
            break
        text = line.strip()
        if not text.startswith(":"):
            entries.append(text)
    return entries


# ---- site layout ----------------------------------------------------------------
def test_index_toctree():
    assert _toctree(DOCS / "index.rst") == TOP_PAGES


def test_old_user_guide_page_is_gone():
    assert not (DOCS / "user_guide.rst").exists()


def test_user_guide_toctree():
    assert _toctree(DOCS / "user_guide" / "index.rst") == GUIDE_PAGES


def test_tutorials_toctree():
    assert _toctree(DOCS / "tutorials" / "index.rst") == TUTORIALS


@pytest.mark.parametrize("tutorial", TUTORIALS)
def test_tutorial_page_includes_its_script(tutorial):
    page = (DOCS / "tutorials" / f"{tutorial}.rst").read_text()
    script = _script(tutorial)
    assert script.is_file()
    assert f".. literalinclude:: scripts/{script.name}" in page


def test_tutorials_index_includes_helpers():
    page = (DOCS / "tutorials" / "index.rst").read_text()
    for helper in ("integrals.py", "scf.py"):
        assert (SCRIPTS / helper).is_file()
        assert f".. literalinclude:: scripts/{helper}" in page


def test_no_placeholder_text():
    pages = sorted(DOCS.rglob("*.rst"))
    offenders = [str(p.relative_to(REPO_ROOT)) for p in pages
                 if "You might choose to write" in p.read_text() or "This page details how to" in p.read_text()]
    assert offenders == []


# ---- cheatsheet -----------------------------------------------------------------
@pytest.mark.parametrize("name", alqemi.__all__)
def test_cheatsheet_names_every_public_name(name):
    text = (DOCS / "cheatsheet.rst").read_text()
    assert re.search(rf"(?<![\w.]){re.escape(name)}(?!\w)", text), name


# ---- runnable examples ----------------------------------------------------------
@pytest.mark.parametrize("page", DOCTEST_PAGES)
def test_page_doctests_pass(page):
    path = DOCS / page
    parser = doctest.DocTestParser()
    examples = parser.get_examples(path.read_text())
    assert examples, f"{page} has no >>> examples"
    flags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    result = doctest.testfile(str(path), module_relative=False, optionflags=flags, verbose=False,
                              report=True, globs={})
    assert result.failed == 0, f"{result.failed} doctest failure(s) in {page}"


_PRIVATE_IMPORT = re.compile(r"^\s*(?:>>>\s*)?(?:from\s+alqemi\.\w|import\s+alqemi\.\w)", re.MULTILINE)


def test_pages_and_scripts_use_only_the_public_api():
    files = sorted(DOCS.rglob("*.rst")) + sorted(SCRIPTS.glob("*.py"))
    offenders = [str(f.relative_to(REPO_ROOT)) for f in files if _PRIVATE_IMPORT.search(f.read_text())]
    assert offenders == []


@pytest.mark.parametrize("script", sorted(p.name for p in SCRIPTS.glob("*.py")) if SCRIPTS.is_dir() else [])
def test_script_imports_are_public(script):
    text = (SCRIPTS / script).read_text()
    imported = set()
    for block in re.findall(r"^from alqemi import \(?([^)\n]*(?:\n[^)\n]*)*?)\)?$", text, re.MULTILINE):
        imported |= {n.strip() for n in block.replace("\n", ",").split(",") if n.strip()}
    assert imported <= set(alqemi.__all__), sorted(imported - set(alqemi.__all__))


# ---- tutorial scripts -----------------------------------------------------------
def _run(tutorial, monkeypatch, tmp_path, capsys, program=None):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    monkeypatch.chdir(tmp_path)          # Psi4 writes timer.dat to the working directory
    monkeypatch.setenv("ALQEMI_OUTPUT", str(tmp_path))
    if program:
        monkeypatch.setenv("ALQEMI_PROGRAM", program)
    for helper in ("integrals", "scf"):
        sys.modules.pop(helper, None)
    namespace = runpy.run_path(str(_script(tutorial)), run_name="__main__")
    capsys.readouterr()
    return namespace


def _check(tutorial, namespace):
    checks = namespace["CHECKS"]
    labels = [c[0] for c in checks]
    assert len(labels) == len(set(labels)), labels
    for label, tol in CHECK_SPEC[tutorial].items():
        assert label in labels, f"{tutorial} lost its check {label!r}"
    for label, ours, theirs in checks:
        tol = CHECK_SPEC[tutorial].get(label, 1e-8)
        assert abs(float(ours) - float(theirs)) <= tol, f"{tutorial} {label}: {ours} vs {theirs}"


def _installed(module):
    return importlib.util.find_spec(module) is not None


@pytest.mark.parametrize("tutorial", sorted(set(TUTORIALS) - NEEDS_SCF))
def test_symbolic_tutorial_runs(tutorial, monkeypatch, tmp_path, capsys):
    pytest.importorskip("numpy")
    _check(tutorial, _run(tutorial, monkeypatch, tmp_path, capsys))


# Reference values a script reports must be the program's own numbers: the test asks
# the program again, independently of the script (label -> SCF settings, quantity).
_RHF, _UHF = {}, {"charge": 1, "multiplicity": 2, "reference": "uhf"}
REFERENCES = {
    "HF energy": (_RHF, "scf"),
    "MP2 energy": (_RHF, "mp2"),
    "MP2 energy (1st step)": (_RHF, "mp2"),
    "CCSD energy": (_RHF, "ccsd"),
    "UHF energy": (_UHF, "scf"),
    "UHF-CCSD energy": (_UHF, "ccsd"),
}
_reference_cache = {}


def _reference(label, program, monkeypatch):
    settings, quantity = REFERENCES[label]
    key = (program, tuple(sorted(settings.items())), quantity)
    if key not in _reference_cache:
        scf = importlib.import_module("scf")
        result = scf.run_scf(program=program, **settings)
        _reference_cache[key] = result.e_scf if quantity == "scf" else scf.reference_energy(result, quantity)
    return _reference_cache[key]


@pytest.mark.parametrize("program", ["psi4", "pyscf"])
@pytest.mark.parametrize("tutorial", sorted(NEEDS_SCF))
def test_numerical_tutorial_matches_program(tutorial, program, monkeypatch, tmp_path, capsys):
    if not _installed(program):
        pytest.skip(f"{program} is not installed")
    namespace = _run(tutorial, monkeypatch, tmp_path, capsys, program)
    _check(tutorial, namespace)
    for label, _, theirs in namespace["CHECKS"]:
        if label in REFERENCES:
            assert abs(float(theirs) - _reference(label, program, monkeypatch)) <= 1e-9, label


def test_commutator_tutorial_result(monkeypatch, tmp_path, capsys):
    namespace = _run("01_commutator", monkeypatch, tmp_path, capsys)
    assert str(namespace["result"]) == "δ_qr a†_p a_s - δ_ps a†_r a_q"
    assert [step.rule for step in namespace["trace"]][0] == "1.8.4"


def test_latex_tutorial_writes_document(monkeypatch, tmp_path, capsys):
    _run("07_latex", monkeypatch, tmp_path, capsys)
    tex = (tmp_path / "ccd_equations.tex").read_text()
    assert r"\begin{align*}" in tex and "(1 - P_{ij})" in tex and r"\end{document}" in tex
    if shutil.which("pdflatex"):
        assert (tmp_path / "ccd_equations.pdf").is_file()


def test_ccsd_tutorial_writes_module(monkeypatch, tmp_path, capsys):
    if not _installed("psi4") and not _installed("pyscf"):
        pytest.skip("needs Psi4 or PySCF")
    program = "psi4" if _installed("psi4") else "pyscf"
    _run("04_ccsd", monkeypatch, tmp_path, capsys, program)
    source = (tmp_path / "ccsd_residuals.py").read_text()
    for name in ("def ccsd_energy(", "def ccsd_r1(", "def ccsd_r2("):
        assert name in source


# ---- build and hosting ----------------------------------------------------------
def test_conf_has_no_language_none():
    assert not re.search(r"^language\s*=\s*None", (DOCS / "conf.py").read_text(), re.MULTILINE)


def test_sphinx_build_without_warnings(tmp_path):
    for module in ("sphinx", "pydata_sphinx_theme", "sphinx_design", "sphinx_copybutton"):
        if not _installed(module):
            pytest.skip(f"{module} is not installed")
    run = subprocess.run(
        [sys.executable, "-m", "sphinx", "-W", "--keep-going", "-q", "-b", "html", str(DOCS), str(tmp_path / "html")],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert run.returncode == 0, run.stderr[-4000:]


def test_readthedocs_config():
    text = (REPO_ROOT / ".readthedocs.yaml").read_text()
    assert re.search(r"^version:\s*2\s*$", text, re.MULTILINE)
    assert re.search(r"configuration:\s*docs/conf.py", text)
    assert re.search(r"fail_on_warning:\s*true", text)
    assert re.search(r"environment:\s*docs/requirements.yaml", text)


def test_docs_requirements_need_no_scf_program():
    text = (DOCS / "requirements.yaml").read_text()
    for package in ("pydata-sphinx-theme", "sphinx-design", "sphinx-copybutton"):
        assert package in text
    assert "psi4" not in text and "pyscf" not in text


def test_documentation_url():
    url = "https://alqemi.readthedocs.io/"
    assert re.search(rf'^"?Documentation"?\s*=\s*"{re.escape(url)}"', (REPO_ROOT / "pyproject.toml").read_text(),
                     re.MULTILINE)
    assert "alqemi.readthedocs.io" in (REPO_ROOT / "README.md").read_text()


def test_ci_docs_job():
    ci = (REPO_ROOT / ".github" / "workflows" / "CI.yaml").read_text()
    assert "devtools/conda-envs/docs_env.yaml" in ci
    assert "alqemi/tests/test_step15_docs.py" in ci
    assert "sphinx-build -W" in ci
    env = (REPO_ROOT / "devtools" / "conda-envs" / "docs_env.yaml").read_text()
    for package in ("psi4", "pyscf", "sphinx", "pydata-sphinx-theme", "sphinx-design", "sphinx-copybutton"):
        assert re.search(rf"^\s*-\s*{re.escape(package)}\b", env, re.MULTILINE), package


def test_scripts_run_standalone_from_their_folder(tmp_path):
    """A reader runs `python tutorial01_commutator.py` inside scripts/; no test harness."""
    pytest.importorskip("numpy")
    env = dict(os.environ, ALQEMI_OUTPUT=str(tmp_path))
    run = subprocess.run([sys.executable, "tutorial01_commutator.py"], cwd=SCRIPTS, env=env,
                         capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr[-2000:]
    assert "largest matrix difference" in run.stdout
