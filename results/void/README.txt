VOID -- produced under a B3 contract that no longer exists.

These runs were measured when a proposer returned rows WITH AMOUNTS. Because it
chose the amount it could choose the residual, so "there was an adjustment of
exactly this much" closed the arithmetic every time. Any resolution rate in
these files describes that design, not the current one, and none of it may be
quoted.

Kept rather than deleted because the change they document is the most useful
thing in the project's history: what the numbers looked like before a reviewer
found the hole, and what the fix cost.

The current contract: a proposer cites evidence and cannot state an amount, and
a rate it chooses itself yields "needs approval" rather than "resolved".
See recoagent/agent/citations.py.

────────────────────────────────────────────────────────────────────────

B3_citation_contract_pre_ratebook.txt  —  VOID as of the RateBook work.

Measured when a fee variance and an FX slip were still the agent tier's to
explain. They are not any more: the book now carries the gateway's repricing
notice and the bank's FX advice as a fourth source, and `legs/repricing.py`
applies them deterministically before the model is ever asked. Every case in
that run has since been closed by a lookup and two multiplications.

The run was honest about what it measured -- 0 resolved, 12 held for approval
-- and the conclusion it supported still stands. But its denominators describe
a tier that no longer sees those cases, so quoting its rates would describe a
system that no longer exists. B3 needs re-measuring against what is actually
left to it.

────────────────────────────────────────────────────────────────────────

qa_nemotron_ultra_dev_pre_ratebook.txt  —  VOID as of the RateBook work.

Its header says it: "leg-2 recall 93.29%, 107 open items". The question bank is
derived from the run, and that run no longer exists -- the fourth source closes
the fee and FX variances now, so the dev book has 4 open leg-2 items rather than
the 11 this was built from, and the questions that asked about the others cannot
be asked at all.

The 0.00% wrong-answer rate it reports was real. It was a rate over 34 questions
that no longer exist. Re-run `recoagent.qa.run` against the current book to get
a number that means something.
