"""ALQEMI: Automated Logic for Quantum Expression, Manipulation, and Implementation.

Second-quantized commutator algebra by HJO's operator identities (no Wick's theorem).

    >>> from alqemi import Index, cre, ann, comm, run
    >>> P, Q, R, S = (Index(x) for x in "PQRS")
    >>> result = run(comm(cre(P) * ann(Q), cre(R) * ann(S)))   # HJO (1.8.14)
"""
from alqemi.core.indices import Index, Space
from alqemi.core.operators import Kind, Op, OpString, ann, cre
from alqemi.core.expr import (
    Bracket,
    BracketKind,
    Delta,
    ElementaryBracket,
    Expr,
    Term,
    acomm,
    comm,
)
from alqemi.core.pool import IndexPool
from alqemi.core.tensor import Swap, Tensor
from alqemi.pipeline.linearize import linearize
from alqemi.pipeline.identity_expand import TraceStep, identity_expand
from alqemi.pipeline.delta_evaluate import DanglingSum, delta_evaluate
from alqemi.pipeline.simplify import relabel_dummies, simplify
from alqemi.pipeline.run import run
from alqemi.pipeline.project import projection, vacuum_expectation
from alqemi.pipeline.bch import bch
from alqemi.rules.hjo import UnsupportedBracket
from alqemi.builders import (
    cluster,
    deexcitation,
    excitation,
    fluctuation_potential,
    fock_operator,
    hamiltonian,
    tensor_operator,
)

try:  # written by versioningit when the package is built or installed
    from alqemi._version import __version__
except ImportError:  # e.g. a fresh clone that was never installed
    __version__ = "0+unknown"

__all__ = [
    # symbols
    "Space", "Index", "Kind", "Op", "OpString", "cre", "ann",
    # expressions
    "Delta", "ElementaryBracket", "BracketKind", "Bracket", "comm", "acomm",
    "Term", "Expr", "IndexPool",
    # pipeline
    "linearize", "identity_expand", "TraceStep", "delta_evaluate", "simplify",
    "relabel_dummies", "run",
    # errors
    "UnsupportedBracket", "DanglingSum",
    # tensors and builders
    "Tensor", "Swap", "tensor_operator", "fock_operator", "fluctuation_potential",
    "hamiltonian", "cluster",
    # projections
    "excitation", "deexcitation", "vacuum_expectation", "projection", "bch",
]
