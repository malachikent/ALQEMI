"""The whole pipeline: Linearize -> IdentityExpand -> DeltaEvaluate -> Simplify."""
from __future__ import annotations

from typing import List, Optional

from alqemi.core.expr import Expr
from alqemi.core.pool import IndexPool
from alqemi.pipeline.delta_evaluate import delta_evaluate
from alqemi.pipeline.identity_expand import TraceStep, identity_expand
from alqemi.pipeline.simplify import simplify


def run(x, trace: Optional[List[TraceStep]] = None) -> Expr:
    """Evaluate every bracket in ``x`` and return the simplified result.

    Uses its own IndexPool. ``trace`` (if given) collects IdentityExpand's steps.
    Raises UnsupportedBracket or DanglingSum for input the pipeline cannot handle.
    """
    expanded = identity_expand(x, IndexPool(), trace)  # Linearize happens inside
    return simplify(delta_evaluate(expanded))
