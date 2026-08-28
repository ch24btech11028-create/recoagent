"""Accounting categorisation, on the same terms as the reconciliation.

    C0   source fields alone           `rules.run_c0`
    C1   + what the reconciliation proved   `rules.run_c1`
    C2   + a model that must quote its evidence   `agent.run_c2`

The ladder is here for the reason the B0/B2/B3 ladder is: so that the sentence
"our AI categorises transactions" carries a number that is the model's, rather
than one it inherited from a `status` field.
"""

from .rules import Assignment, Ledger, residue, run_c0, run_c1
from .score import CategoryScorecard
from .taxonomy import DEFINITIONS, Category

# `score` and `render` are deliberately NOT re-exported here. They would shadow
# the `score` submodule for anyone writing `from recoagent.categorize import
# score`, which is the natural spelling and would silently hand them a
# function. Import them from `recoagent.categorize.score` instead.

__all__ = [
    "Assignment", "Ledger", "residue", "run_c0", "run_c1",
    "CategoryScorecard", "Category", "DEFINITIONS",
]
