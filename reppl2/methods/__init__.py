from __future__ import annotations

from .common import (AggregatorConfig, SharedAggregate, aggregate_A, compute_cv,
                     inv_func, row_softmax, cosine_matrix, l2_normalize, MathError)
from .reppl_a import reppl_a_inner
from .reppl_b import reppl_b_inner

AGG_NOTE = "risk=(Inner+eps)*Outer; higher means more hallucination-prone"
