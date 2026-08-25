"""Settlement Q&A -- ask questions about a completed reconciliation run.

The agent answers from a factsheet built by code from the run, and every
question in `bank` carries a programmatic answer so accuracy is measured rather
than asserted. Lead metric is the wrong-answer rate, for the same reason the
reconciliation tier leads with false-match rate.
"""

from .agent import Answer, QAReport, ask, factsheet, render
from .bank import Question, build, is_correct

__all__ = [
    "Answer", "QAReport", "ask", "factsheet", "render",
    "Question", "build", "is_correct",
]
