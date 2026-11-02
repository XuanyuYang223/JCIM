# Manuscript documentation revisions

## 1 November 2026: corrections to the initial v2–v6 draft ledger

The initial draft and its 54-claim ledger were committed in `b20fa80` before the v7 primary run. Independent reading identified the following documentation issues. No archived experiment, molecular identity, numerical result, feature, weight, report, or manifest was changed.

- **C49, own-curriculum agreement.** The initial ledger selected all common-unseen agreement rows and keyed results only by model. Cross-curriculum rows could overwrite the corresponding own-curriculum values. The corrected selection retains the six trained models evaluated against their own source curriculum and keys values by curriculum and model. Correct supported-error fractions are A: `1.0, 1.0, 0.9977426636568849`; B: `1.0, 1.0, 1.0`. The main manuscript's qualitative statement was already correct.
- **C48 and Supplement S4, receptive-reach population.** The unchanged 98.7% value refers to all 1,649 archived v4 feature identities, including analysis molecules and fitting supports. The initial text incorrectly named this population as analysis molecules alone. The corrected wording names the feature-identity population; the archived value and computation are unchanged.
- **Source label accounting.** Ledger entries for original/v5 jointly trained classifiers now state that each of the 1,386 source molecular identities supplies motif, position, and distance labels. The v6 motif-only refits remain labeled as motif-only. The main methods already distinguished these protocols. No label budget or numerical score changed.

The ledger builder, regenerated `CLAIMS.csv`, and `CLAIMS_CHECK.json` were updated together. All claim identifiers C00–C53 remain stable. The ledger hash at this correction stage, before adding v7 claims, is recorded below; subsequent additions can change the final ledger hash without altering these corrections.

| Documentation snapshot | SHA-256 of `CLAIMS.csv` |
| --- | --- |
| Initial `b20fa80` draft | `92bbbb35dd0aa59c665ef37ec0338d03c475dbaf0aff51b81eef448851bf5600` |
| Corrected v2–v6 ledger | `c7c12d0defc3781472adddec87e54d0beb8baca4c5f5858593de8a35193fc7e4` |

The comparison against the initial Git blob confirms that C49 is the only changed numerical result object. C48 changes its population description, and the other changed entries clarify label provenance. These are manuscript documentation corrections, not revisions to frozen scientific records.

## 1 November 2026: integrate completed readout-factor results

The working manuscript and supplement now include the completed `readout_v7` primary record, executed from source `b20fa80`. Claims C100–C112 extend the ledger without renumbering the earlier claims. The supplement includes every MLP and ridge cell, source-fitting guards, shared random references, and the historical zero-column initialization limitation. Main-text Table 3 focuses on a fixed standardized-ridge scope contrast with its familiar-accuracy guard.

The nested-ridge certificate is separately identified as a post-run saved-state analysis with no refitting (C113). Numerical reconciliation preserves the initial CPU audit's single untrained auxiliary-head failure and records exact matched-CUDA replay separately (C114). Neither result is silently inserted into the frozen experiment record or described as a predeclared endpoint. Figure 1 is a new manuscript illustration derived from recorded motif structures, with its own generator and metadata; it does not replace any earlier scientific figure or data archive.

The final reproducibility note also preserves the first runtime-wrapper **FAIL** caused by NumPy initialization before the BLAS configuration, and links the unchanged wrapper's complete **PASS** under the recorded startup configuration (C115–C116). Configured BLAS values are distinguished from an observed worker count. The supplement retains both failures and the matched-runtime success rather than describing the CPU audit as passing.
