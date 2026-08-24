"""The two matching legs.

Leg 1: merchant order <-> gateway payment, 1:1, a record-linkage problem.
Leg 2: gateway settlement batch <-> bank credit, N:1, an instance of the
       Subset Sum Matching Problem (Wu et al., 2025).

Modules in this package must never import from `recoagent.generator`.
`tests/test_independence.py` enforces it.
"""
