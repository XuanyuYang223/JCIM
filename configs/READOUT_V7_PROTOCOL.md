# Frozen-readout factorial controls

This protocol is fixed before the primary readout-v7 execution. It adds matched
random-encoder controls and separates three readout fitting choices without
training a new encoder, changing a molecular split, supplying target boundaries,
or adding labels from a held-out structural combination. The purpose is to
locate conditional sources of the gap between diagnostic readability and
original-task generalization, not to establish one universal explanation.

## Frozen sources and units of replication

Use the twelve actual skip-v5 `shallow_skip` encoders: two v5 training regimes
(`end_to_end` and `frozen_readout`), experiences A and B, and seeds 0, 1, and 2.
Use the three same-seed original hard-v2 random encoders as matched controls.
These are fifteen encoder identities and eighteen source-label cases: twelve
trained cases and six random cases, because each random encoder is fitted once
on A labels and once on B labels. Do not duplicate a random fit for each v5
regime or count the shared random encoder as an independent initialization.

The v5 frozen-readout encoder is the original hard-v2 task-trained encoder;
"frozen-readout" names its source regime and does not mean random. All encoders
in the present experiment are frozen, including v5 end-to-end sources. Preserve
their tensor hashes and every source record. For seed summaries, first pair
within seed, experience, and source regime, then report the mean and sample
standard deviation over three seeds. A and B are reported separately. Examples,
readout factors, and repeated uses of a random encoder are not extra replicates.

## Fixed 512-coordinate inputs

Extract the exact v5 representation: the inherited final-layer 448-dimensional
graph/query readout followed by the actual 64-dimensional layer-1 whole-graph
mean. The shallow branch uses float64 accumulation and returns float32; the
inherited deep coordinates keep their original operations. Use inference batches
of 128, duplicate valid graphs in the last short batch, and discard duplicate
outputs. Require bitwise continuity with archived v5 analysis features and v6
whole-graph shallow features where they share identities. Random sources use the
same complete attributed input and extraction operations.

No target-fragment membership is supplied to this experiment. Original query
markers and original atom attributes remain present. Extraction itself is
unsupervised; all learned scalers and readout parameters are fitted on the
experience's original 1,386 source-training identities only.

## Fully crossed MLP controls

Fit all eight cells of a two-by-two-by-two factorial on every source-label case:

| Factor | Level 1 | Level 2 |
| --- | --- | --- |
| Input scope | `full512`: inherited deep/query vector plus shallow mean | `shallow_only`: first 448 coordinates zero, final 64 unchanged |
| Scaling | `raw`: no centering or scaling | `train_z`: source-training population mean and standard deviation, constant scales set to one |
| Objective | `family`: motif cross entropy only | `joint`: sum of motif, position, and distance cross entropies |

For `shallow_only`, the masked coordinates remain exactly zero after scaling;
their fitted mean is zero and constant scale is one. Input and parameter counts
remain identical to `full512`; masking removes information rather than making a
smaller classifier. All cells use three separate `512 -> 64 -> output` ReLU heads
with the same task output dimensions as v5. Initialize from the same-seed archived
random hard-v2 heads: copy the original 448 first-layer columns and all original
biases and second layers, and set the appended 64 first-layer columns to zero.
Use identical initial states across factors and across encoder sources of the
same seed. Unused heads in the family-only cell remain at initialization.

There are 18 cases times 8 cells, or 144 MLP fits. Every fit uses the archived
source-training identities, labels, and per-seed minibatch orders; 120 epochs,
128 examples per batch, 1,320 updates, AdamW learning rate 0.0015, weight decay
0.0001, and global gradient-norm clipping at 5. Motif loss has coefficient one
in both objective levels; the joint loss is summed rather than averaged. Use
the final epoch without OOD selection, validation-based stopping, hyperparameter
search, or adaptation labels. Familiar validation is descriptive only.

With frozen inputs and disjoint heads, the joint objective has no ordinary
shared-representation gradient path into the motif head. Before global clipping,
its motif gradient equals that of family-only training at identical motif-head
states. Joint training can nevertheless change the motif update through the
single clipping coefficient computed across all active heads. Record the total
and motif gradient norms, clipping coefficient, and clipped indicator on every
update. The objective factor therefore measures this optimizer coupling in this
architecture; it cannot identify the effect of multitask encoder learning.
Include a focused gradient-independence and clipping-coupling test.

## Separate ridge panel and continuity anchors

Fit a fixed-alpha-1 multiclass ridge readout for both input scopes and both
scaling levels, using an unpenalized bias and one-hot motif targets. The 18 cases
times 4 cells give 72 ridge fits. Keep this panel separate from the MLP factorial:
changing decoder family also changes its objective and optimization conditions.
It is a descriptive robustness comparison, not a fourth clean MLP factor.
Raw ridge centers the design internally to fit its unpenalized intercept and
then expresses the result in raw coordinates; it does not standardize feature
scales. Standardized ridge follows the exact v6 solve. For shallow-only ridge,
solve and store coefficients using the 64 active coordinates; the omitted 448
coefficients are constrained to zero. This preserves the v6 numerical operations
as well as its mathematical model.

For standardized shallow-only ridge, the zero-padded representation must produce
the same held-out predictions as the v6 standardized 64-coordinate whole-graph
ridge on the same encoder, experience, and training labels. Preserve numerical
continuity results rather than replacing earlier records. For raw/full/joint
MLPs on the original task-trained v5 frozen-readout encoders, require continuity
with the actual v5 frozen-readout protocol, including initialization, orders,
optimizer, and source data. Test predictions, saved parameters, and any numerical
differences explicitly. A discrepancy must be recorded and resolved before the
primary execution, or reported as a failed anchor; it must not be silently
relabeled exact continuity.

## Predeclared analysis

The primary behavioral outcome is ordinary eight-class motif accuracy on the
fixed 443-molecule common-OOD set, accompanied by own-IID accuracy as a source
performance guard. Uniform guessing has expected accuracy 12.5%. Retain balanced
accuracy, all task predictions and logits, training fit, both IID sets, all seeds,
and denominators. Report all cells, not only cells that improve OOD performance.

For each experience and trained source regime, compare each trained cell with
the exact same-protocol, same-seed random cell. These matched contrasts measure
training benefit under a specified readout protocol. They do not attribute the
random network's structural encoding to task training. The random reference is
shared across the two source regimes and must be described as shared.

Within each source group, estimate conditional paired factor contrasts: change
scaling at fixed scope and objective; change scope at fixed scaling and objective;
and change objective at fixed scope and scaling. Retain the two-factor
difference-in-differences and the three-factor interaction. Report seed-level
contrasts with sample SD rather than treating fitted cells as independent data.
Also report each factor's contrast averaged over its four companion-factor
settings within each seed, retaining the conditional contrasts so cancellation
and interactions remain visible.
Use the reported directions `shallow_only - full512`, `train_z - raw`, and
`family - joint`. Higher-order effects use the corresponding signed
differences, averaged over the remaining factors, rather than scaled regression
coefficients.
No single conditional contrast uniquely decomposes the earlier v5-to-v6 gap:
those experiments also differ in source encoder training, head width and
initialization, and fitting procedure. Do not sum conditional contrasts as if
they were independent, additive causal contributions. Three seeds support a
controlled descriptive analysis, not a strong population-level significance
claim or evidence of generalization to real chemical properties.

## Archive and independent checks

Archive frozen encoder inventories, exact train/validation/test identity plans,
minibatch orders, 512-coordinate frozen features, initial and final readout
states, fitted scalers, all predictions and logits, optimization histories,
per-update gradient-clipping diagnostics, complete cell and paired summaries,
English prose and figures, source hashes, and a SHA-256 manifest. Verify all
1,260 manifest-covered files in the eight previous completed records before and
after the run. Do not modify any prior numerical result, weight, feature, or
manifest. A shortened smoke run checks mechanics and is excluded from evidence.

An independent auditor imports no v7 experiment implementation. It checks all
record hashes, identity isolation, factor completeness, shared initialization,
constant parameter counts, frozen-state hashes, stored order continuity,
training-only scaler reconstruction, all saved-head and ridge predictions,
prediction-derived metrics and paired contrasts, clipping diagnostics, and the
continuity anchors. It reconstructs selected encoder features independently and
preserves any numerical execution caveat instead of concealing it with an
accuracy-only comparison. Reports and manuscript claims must stay within the
controlled structural task and label protocol.
