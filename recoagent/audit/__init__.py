"""Adversarial audit: attacking the matcher on purpose, and scoring what happens.

Deliberately imports nothing. `python -m recoagent.audit.mutate` executes the
submodule, and a package that had already imported it would run it twice --
which Python warns about and which would put a RuntimeWarning at the top of
every CI log. Import `recoagent.audit.mutate` directly.
"""
