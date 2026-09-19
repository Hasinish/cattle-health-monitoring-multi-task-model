# Research Log — 2026-09-20: ScienceDB Roadmap Correction

## Evidence

The ScienceDB identity audit showed that the former project interpretation of 10,898 biological cows was invalid. Filename-derived identifiers included passage/sequence identifiers and, in the stereo subset, individual frame numbers.

The audit reconstructed 5,662 passage/sequence clusters and found that the legacy split fragmented 247 of 261 stereo passage blocks (94.64%) across train/validation/test.

A replacement deterministic split (seed 42) now protects passage/sequence clusters:

- Train: 3,963 clusters / 37,126 images
- Validation: 849 clusters / 8,099 images
- Test: 850 clusters / 8,341 images
- Passage-cluster overlap across partitions: 0
- Exact-duplicate leakage across partitions: 0

## Roadmap Decision

This evidence requires a narrow correction to the Phase 3 roadmap:

- ScienceDB evaluation is now described as **passage-disjoint / sequence-safe**, not cow-disjoint.
- ScienceDB must not be used to claim cross-cow generalization because verified biological cow IDs are unavailable.
- The core Phase 3 research direction, downstream tasks, dataset roles, and experiment order are unchanged.
- Perceptual near-duplicate auditing remains required before Gate 1 is fully passed.

## Immediate Next Action

Rebuild the MmCows grouped evaluation protocol using cow ID as the primary grouping variable, with time-block and synchronized multi-view protection.
