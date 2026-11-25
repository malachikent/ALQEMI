"""Tutorial 7: publication-ready equations.

Write the CCD energy and doubles residual, and the spin-integrated OoVv block, to a
standalone LaTeX document with align* blocks and (1 - P_ij)(1 - P_ab) grouping, then
compile it if pdflatex is installed.

Run:  python tutorial07_latex.py      (writes ccd_equations.tex to ALQEMI_OUTPUT or .)
"""
import os
import shutil
import subprocess

from alqemi import bch, cluster, hamiltonian, latex, projection, spin_integrate
from alqemi import vacuum_expectation

CHECKS = []  # (label, ALQEMI value, reference value)

Hbar = bch(hamiltonian(), cluster(2), order=2)
energy = vacuum_expectation(Hbar)
residual = projection(2, Hbar)
mixed = spin_integrate(residual)["OoVv"]    # alpha-beta block: no same-spin pairs to group

body = "\n\n".join([
    r"\section*{CCD}",
    "Energy (the first two terms are the Hartree--Fock energy):",
    latex(energy, lhs="E", terms_per_line=3),
    r"Doubles residual, grouped with $P_{pq}$ exchanging $p$ and $q$:",
    latex(residual, lhs=r"0 = R^{ab}_{ij}", terms_per_line=2, group=True),
    r"Its $\alpha\beta$ block (upper case $\alpha$, lower case $\beta$):",
    latex(mixed, lhs=r"0 = R^{Ab}_{Ij}", terms_per_line=2),
])
document = "\n".join([
    r"\documentclass{article}",
    r"\usepackage{amsmath}",
    r"\usepackage[margin=2cm]{geometry}",
    r"\begin{document}",
    body,
    r"\end{document}",
])

out = os.environ.get("ALQEMI_OUTPUT", ".")
path = os.path.join(out, "ccd_equations.tex")
with open(path, "w", encoding="utf-8") as handle:
    handle.write(document + "\n")
print(f"wrote {path}")

if shutil.which("pdflatex"):
    run = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "ccd_equations.tex"],
                         cwd=out, capture_output=True, text=True)
    print("pdflatex:", "ok" if run.returncode == 0 else run.stdout[-2000:])
    CHECKS.append(("pdflatex exit code", run.returncode, 0))
else:
    print("pdflatex not found: compile ccd_equations.tex yourself")
