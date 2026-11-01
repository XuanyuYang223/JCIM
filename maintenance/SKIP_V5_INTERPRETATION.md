# Interpretation of the shallow whole-graph path intervention

The completed [skip-v5 record](../experiments/skip_v5/REPORT.md) does not show a
stable recovery of zero-target-label motif prediction from the added shallow
whole-graph mean. The end-to-end gains are small and seed dependent; refitting
heads on the original labels with frozen pretrained encoders produces almost
no gain over the parameter-matched deep control. This qualifies the proposed
readout-path explanation without contradicting the earlier oracle-fragment
probes or adaptation with new labels.

## Matched primary and secondary results

Both 512-dimensional variants have 201,819 total parameters and the same
initial head values. The control appends an exact duplicate of an existing
deep-mean block. Every model uses the original 1,386 training molecules and
1,320 optimizer updates, with no common-OOD combination or adaptation-reserve
identity in training or validation. The six original deep end-to-end baselines
reuse the exact completed source checkpoints and histories. Frozen runs reset
the heads to their seed-matched random values and keep the original trained
encoder fixed.

The primary endpoint is common-OOD motif accuracy, with paired differences
computed within each seed. Values are percentage points; SD is the sample SD
of three seeds, not a confidence interval or significance test.

| Regime | Experience | Shallow-skip OOD accuracy | Skip minus deep control, mean +/- SD | Differences at seeds 0, 1, 2 |
| --- | --- | --- | --- | --- |
| End-to-end | A | 2.2% +/- 3.8% | +1.1 +/- 5.0 | +6.55, 0.00, -3.16 |
| End-to-end | B | 6.6% +/- 6.3% | +5.6 +/- 6.4 | 0.00, +12.64, +4.29 |
| Frozen encoder, reset heads | A | 0.0% +/- 0.0% | 0.0 +/- 0.0 | 0.00, 0.00, 0.00 |
| Frozen encoder, reset heads | B | 7.5% +/- 6.6% | +0.1 +/- 0.1 | +0.23, 0.00, 0.00 |

Own-IID motif accuracy remains 100% in every intact arm. The positive mean in
end-to-end B does not establish a reliable general solution: one seed has no
benefit, the endpoint remains low, and three initializations are a limited
uncertainty estimate. Eight-output uniform prediction has expected accuracy
12.5%; this reference is distinct from an empirical random encoder baseline.

## The appended branch learned weights but was rarely decisive

All 12 shallow-skip models learned nonzero added weights in each of the motif,
position, and distance heads. Motif-head added-weight L2 norms range from 0.843
to 1.252. Nonzero coefficients alone do not establish a useful feature route.

For all 12 models, both zeroing and permuting the shallow branch leave all
three own-IID class predictions unchanged. On common OOD, permutation leaves
motif accuracy unchanged in every model; 11 models have identical motif
predictions, and frozen B seed 2 changes one wrong label to another wrong label.
Common-OOD position predictions remain unchanged. Distance predictions change
for two molecules in end-to-end B seed 2 and one molecule in frozen B seed 2.

Zeroing has somewhat larger, task-dependent effects. Mean common-OOD accuracy
changes below are altered minus intact, in percentage points. These summaries
average the same three seeds within each experience.

| Regime | Experience | Motif | Position | Distance | All three correct |
| --- | --- | ---: | ---: | ---: | ---: |
| End-to-end | A | +0.15 | 0.00 | -0.98 | +0.15 |
| End-to-end | B | +0.15 | 0.00 | -0.68 | -0.38 |
| Frozen encoder, reset heads | A | 0.00 | 0.00 | +1.28 | 0.00 |
| Frozen encoder, reset heads | B | +0.08 | 0.00 | -0.23 | +0.45 |

Some cross-IID predictions also change, so the branch is not behaviorally
inert. Nevertheless, the end-to-end motif advantage survives removal of the
branch at inference. It therefore cannot be attributed simply to the final
classifier needing this shallow input. Changed deep-path learning or
optimization is consistent with the observation; these alternatives are not
separately identified. Branch replacements can also be outside the fitted
feature distribution, and accuracy can conceal changed incorrect labels or
changed logits.

## Scope of the conclusion and audit

Whole-graph averaging combines the target with ring, anchor, distractor, and
query-marker context. It is not the nine-atom oracle-fragment pool that reached
100% in the scale experiment. This null-to-small intervention result does not
show that shallow structural information is absent, erased, or inaccessible to
every decoder. The [architectural locality note](ARCHITECTURE_LOCALITY_NOTE.md)
separates architectural consistency from empirical linear discriminability.
The useful paper-level distinction remains readable local information,
adaptation using new labels, and prediction of combinations absent from the
training labels. This particular added global path did not bridge that gap.

The [independent audit](SKIP_V5_AUDIT.json) passes for all 36 cases, all 18 frozen
encoder byte identities, six exact reused baselines, 252 metric rows, 115,668
prediction rows, and 108 paired seed contrasts. Every saved head prediction
and branch intervention was recomputed on CPU with zero mismatches; four fixed
analysis identities per model were checked using an independently implemented
message-passing function. All 405 prior archived files remain unchanged.
Metadata and final states cannot observe every historical optimizer operation
or rule out every unrecorded external model-selection action.

Executed source commit: `e8969b8`. Frozen experiment manifest SHA-256:
`39ebb1e0833557047709630b2d8c60ec773801267a585efdfaed5401107ed16b`.
This interpretation is a separate post-run document and changes no experiment
file, numerical result, weight, or manifest.
