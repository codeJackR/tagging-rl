"""W1 Step 4 — one command, one report.

    python -m evalharness.report --gold data/eval_300 --from-frontier

Macro-F1 is the headline because apparel attribute values are Zipf-distributed:
accuracy and micro-F1 both reward a model that answers "crew, cotton, casual" to
everything, and macro-F1 does not.
"""

from . import metrics, predictions

__all__ = ["metrics", "predictions"]
