# Independent interpretation of the frozen readout factorial

The new factorial supports a conditional conclusion: a frozen representation can
expose a generalizing shallow decision rule while a readout fitted to a larger
representation selects a poor rule for unseen combinations. Scaling and input
scope interact, and source-fitting failures limit comparisons involving raw
shallow features. The matched random controls do not support attributing the
source-only shallow recovery to task training.

This is a post-run interpretation of the fixed
[protocol](../configs/READOUT_V7_PROTOCOL.md) and
[completed report](../experiments/readout_v7/REPORT.md). All accuracies below are
ordinary eight-class motif accuracy on the same 443 common-OOD molecules. All
deviations are sample SD across three seeds; contrasts are paired within seed.
Source groups and A/B experiences remain separate. The random encoder is shared
across A/B fitting and source-group comparisons, not counted repeatedly as an
independent initialization.

## What was held fixed

Fifteen frozen encoders generate eighteen source-label cases: twelve actual v5
encoders and three random encoders, the latter each fitted on A and B labels.
The 144 MLP fits cross full512 versus shallow-only masking, raw versus
training-only standardization, and joint versus family-only loss. All retain
the same 512-input, three-head architecture, 100,251 readout parameters,
same-seed archived random initialization, zero appended columns, original 1,386
training identities, original minibatch orders, 120 epochs, and 1,320 updates.
No target membership or new-combination labels are added. Seventy-two separate
fixed-alpha ridge fits supply a complementary decoder panel, not another MLP
objective level.

Full512 contains the original 448 deep/query coordinates and the exact actual
v5 shallow mean. Shallow-only masks the first 448 coordinates. Thus masking
changes available input information without changing MLP width or initialization.
All scalers use the fitting support only. The six original-trained/raw/full/joint
heads exactly reproduce the v5 frozen-readout states and predictions; the
eighteen standardized shallow ridge fits reproduce v6 exactly. These anchors
are continuity checks, not independent experimental replications.

## Source fit is an essential guard

All trained full512 fits achieve 100% own-IID motif accuracy, for both MLP and
ridge and both scaling levels. All trained standardized shallow ridge fits also
achieve 100%. The standardized shallow MLP retains nearly all source performance:

| Encoder source | Experience | Own-IID, standardized shallow MLP |
| --- | --- | --- |
| V5 end-to-end | A | 99.93% +/- 0.12% |
| V5 end-to-end | B | 99.79% +/- 0.37% |
| Original task-trained | A | 100.00% +/- 0.00% |
| Original task-trained | B | 98.45% +/- 2.69% |

Raw shallow MLPs underfit: own-IID means are 50.6%, 54.6%, 61.0%, and 49.8% for
these four trained groups, and 15.2%/16.2% for random A/B fits. Raw shallow ridge
has trained own-IID means of 85.9% to 89.8%, versus approximately 54% for random
features. Standardization therefore changes source fitting substantially in
these cells. Its effects cannot all be described as choosing a more generalizing
rule among equally well-fitted source solutions. Masking and standardization
also change conditioning under this fixed zero-appended initialization and
fixed training budget.

## Conditional effects are more informative than one global ranking

Under standardized family-only MLP fitting, masking deep/query coordinates
improves trained-source OOD accuracy while preserving the own-IID guards above:

| Encoder source | Experience | Full512 OOD | Shallow-only OOD | Paired shallow-minus-full effect (pp) |
| --- | --- | --- | --- | --- |
| V5 end-to-end | A | 0.0% +/- 0.0% | 17.8% +/- 16.0% | +17.8 +/- 16.0 |
| V5 end-to-end | B | 3.4% +/- 5.9% | 26.7% +/- 6.6% | +23.3 +/- 12.1 |
| Original task-trained | A | 0.0% +/- 0.0% | 19.0% +/- 19.4% | +19.0 +/- 19.4 |
| Original task-trained | B | 3.3% +/- 5.7% | 20.4% +/- 16.4% | +17.1 +/- 22.1 |

These partial recoveries do not approach the earlier oracle-local 100% upper
bound, and some seed effects are small or negative. They nevertheless show that
the availability of more coordinates does not ensure that this fitted readout
uses a more transferable rule. Full512 literally contains the shallow vector;
the poor full512 outcome is not evidence that those shallow coordinates are
absent from its input.

Scaling is not uniformly beneficial. For the v5 end-to-end encoder with a
family-only MLP, train-z minus raw is -0.15 +/- 0.26 pp on full512 A and
-1.50 +/- 12.59 pp on shallow A; for B it is -6.47 +/- 4.29 pp on full512 and
+17.46 +/- 6.11 pp on shallow-only. The B scope-by-scaling interaction, averaged
over objectives within seed, is +23.93 +/- 9.61 pp. The raw shallow source-fit
deficit remains part of that interaction. A claim that one normalization change
alone explains the earlier v5-to-v6 improvement would be unsupported.

The ridge panel gives a stronger scope-dependent contrast for trained features.
Standardized full512 accuracy is 0.0%/8.4% for v5 A/B and 0.0%/7.4% for the
original-trained A/B encoders; standardized shallow ridge gives
37.4%/64.2% and 27.4%/58.6%, respectively, with 100% trained own-IID throughout.
The separate post-run
[containment check](READOUT_V7_CONTAINMENT.json) places each saved shallow ridge
inside its full-coordinate counterpart with zero deep coefficients. All 36
matched pairs retain exactly identical candidate logits and predictions and the
same penalty, while the fitted full solution has strictly lower source ridge
objective. This confirms a feasible generalizing candidate in the full feature
space; minimizing the declared source objective still chooses another solution
with different OOD behavior. It does not uniquely identify why the original
neural classifier selects its rule. The containment check is post-run analysis,
not a new fitted model or a predeclared endpoint.

## Matched random controls change the training claim

The standardized shallow ridge figures reproduce v6, now as one cell in a full
matched protocol:

| Source | Experience | Trained OOD | Random OOD | Paired trained-minus-random effect (pp) |
| --- | --- | --- | --- | --- |
| V5 end-to-end | A | 37.4% +/- 19.5% | 48.9% +/- 24.1% | -11.5 +/- 40.5 |
| V5 end-to-end | B | 64.2% +/- 9.6% | 65.1% +/- 25.9% | -0.9 +/- 31.4 |
| Original task-trained | A | 27.4% +/- 8.5% | 48.9% +/- 24.1% | -21.5 +/- 28.8 |
| Original task-trained | B | 58.6% +/- 20.3% | 65.1% +/- 25.9% | -6.5 +/- 39.7 |

The random standardized shallow ridge has 99.93% own-IID accuracy for both A
and B; the corresponding random MLP means are 98.70% and 97.88%. These guards
remain distinct from OOD accuracy and are retained rather than rounded to 100%.

Negative mean contrasts and substantial seed variability do not establish a
reliable task-training advantage in this source-only cell. Standardized shallow
family-only MLP contrasts also favor random features on average: paired effects
are -34.3 +/- 23.5 pp and -12.7 +/- 19.5 pp for v5 A/B, and
-33.1 +/- 29.0 pp and -19.0 +/- 30.7 pp for original-trained A/B. These are
descriptive three-seed outcomes, not population significance claims or proof
that random encoders dominate every protocol. The previously observed advantage
for trained deep graph features with new adaptation labels is a distinct result
under a distinct supervision distribution and remains compatible with them.

## Objective differences do not identify multitask representation interference

Changing the objective has almost no motif-accuracy effect in this frozen setup.
The mean family-minus-joint difference is zero in every trained shallow MLP cell
and all trained full512 cells except original-trained B/raw, where the paired
effect is -0.60 +/- 1.04 pp. There is no shared trainable feature extractor or
shared head parameter: other tasks cannot change the raw motif gradient at the
same motif-head state. Their only update coupling is the single global clipping
coefficient, with floating-point execution as a numerical consideration.

Across all 190,080 logged updates, clipping occurs 264 times in 20 of the 144
MLP fits. The logs and independent first-gradient checks support the specified
optimizer-coupling interpretation. They do not test task interference during
end-to-end encoder learning. A small or null frozen-head objective contrast
cannot be generalized to that separate question.

## Audit status and limits

The complete matched-runtime audit is
[PASS](READOUT_V7_RUNTIME_SINGLE_THREAD_AUDIT.json). It reuses every frozen
auditor check, with only saved-head evaluation moved to the original CUDA device
and single-thread BLAS configured before Python starts. All 72 ridge fits, all
24 historical anchors, all metrics, all clipping records, and all contrasts
pass. The original [CPU audit](READOUT_V7_AUDIT.json) remains **FAIL** on one
untrained distance-head/source-training example whose approximately `1.8e-7`
logit difference becomes an exact CPU tie. All motif and all trained-task CPU
predictions pass. The first [runtime replay](READOUT_V7_RUNTIME_AUDIT.json)
also remains **FAIL** because NumPy initialized before its BLAS environment was
configured; the unchanged wrapper passes under the recorded launch environment.
The [numerical note](READOUT_V7_NUMERICAL_NOTE.md) preserves the exact details,
commands, hashes, and independent full-array device replay. No source, tolerance,
completed record, or earlier failed log was changed.

The factorial estimates conditional readout effects under fixed archived
initialization, masking, budgets, and frozen encoders. It does not uniquely
decompose the historical v5-to-v6 change, whose standalone 64-input MLP used a
different initializer and fitting setup. Source fit, OOD behavior, random
controls, and interactions must all remain visible. The task uses familiar
synthetic motif topologies, marked queries, and intentionally strong source
binding; it supplies no evidence of experimental chemical-property prediction.
Within those boundaries, the evidence supports separation of available
structural information, a selected source prediction rule, and the benefit of
new-label adaptation.
