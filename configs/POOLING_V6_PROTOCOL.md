# Frozen-model branch sensitivity and pooling diagnostics

This protocol is fixed before the primary pooling-v6 execution. It adds no
encoder architecture and changes no archived encoder, classifier, data split,
checkpoint, or completed record. All new classifiers use frozen representations.
The two questions are whether the v5 branch changes model outputs despite nearly
unchanged accuracy, and whether supplied target localization changes the ability
to learn a motif rule from the original source labels.

## Frozen sources and statistical units

Use the completed skip-v5 `deep_control` and `shallow_skip` checkpoints in both
`end_to_end` and `frozen_readout` regimes, experiences A and B, seeds 0, 1, and 2,
for branch diagnostics: 24 existing models. For pooling, use the twelve actual
v5 `shallow_skip` encoders and the three same-seed hard-v2 random encoders.
No encoder parameter is optimized. Check complete source-file hashes and encoder
tensor hashes before and after processing.

The frozen-readout A/B encoders are the original hard-v2 trained encoders.
Repeated identities or identical frozen encoders are not additional independent
replicates. Each random seed supplies one encoder, shared across the A and B
source-label fits. Report A and B separately; do not pool six experience/seed
entries as six independent encoder initializations. The three seeds are the
units for mean and sample SD; examples, support draws, directions, heads, and
pooling scopes are paired measurements within seed.

## Branch output sensitivity

Reuse the exact saved v5 512-dimensional analysis readout, the exact final
classifier, and the v5 analysis identity order. The first 448 coordinates remain
fixed. Compare the intact appended 64-vector with zero replacement and with the
exact v5 row permutation from `default_rng(90000 + seed)`. A deep-control branch
is a duplicate of an existing deep mean; a shallow-skip branch is the layer-1
whole-graph mean. Recompute predictions and require continuity with the v5
per-molecule predictions before interpreting sensitivity.

For each of the motif, position, and distance heads, retain all intact and
intervened logits and calculate per example:

- prediction-class changes;
- correct-class margin `z[y] - max(z[k] for k != y)`;
- signed intervened-minus-intact margin change and its absolute magnitude;
- change in centered logits, where each logit vector has its class mean removed.

Report mean signed and absolute changes, distribution summaries, and flip rates
for common OOD, both IID sets, and the fixed full analysis set. A mean signed
change can cancel substantial individual changes; a common shift of all logits
does not affect the margin, class, or centered-logit sensitivity. Raw margins are
in a particular trained head's units and are not directly calibrated effect
sizes across independently trained heads. Summarize per-model quantities within
seed before seed-level uncertainty. Unchanged accuracy alone is not evidence of
zero branch influence. These replacements can be off distribution and do not
identify an additive semantic or causal contribution.

## Two layer-1 pooling scopes

For the same complete attributed molecular input, measure:

| Scope | Representation | Additional information |
| --- | --- | --- |
| whole_graph | Exact 64-dimensional layer-1 whole-graph mean used by the v5 shallow branch | Original input/query markers only |
| target (oracle) | Mean over the nine annotated target-substituent atoms at layer 1 | Supplied target-fragment membership |

Both means exclude padding, accumulate in float64, and return float32. For every
v5 encoder, require whole-graph features on the archived analysis identities to
be bit-identical to the actual v5 appended branch. Random features follow the
same extraction and pooling operations. Use fixed inference shape 128 and fill
the final short batch by repeating valid graphs, discarding repeated outputs.

The oracle annotation is used by the pooling operation, never introduced into
the encoder's atom features or message passing. Nevertheless, it is extra
information and constitutes a diagnostic upper bound, not a boundary-free
prediction method. Layer-1 states at target atoms can include their immediate
external neighbors and original boundary-related atom attributes; target pooling
does not restrict input dependence to the nine atoms. The whole-graph mean also
includes unrelated molecular context. Equal width controls classifier input
dimension, not the information supplied by the masks.

## Original-source-label classifier fits

Fit an eight-output multiclass ridge classifier with fixed alpha 1, and a
`64 -> 64 -> 8` ReLU classifier, independently on each pooling scope. Both are
fixed, co-primary diagnostic readouts; report their results separately without
choosing a winning readout from OOD scores. Means and population standard
deviations are estimated only from the fitting support, with constant-feature
scales replaced by one. The identical transformation is applied to held-out
features.

Each source-only fit uses exactly the experience's original 1,386 training
identities and motif labels. It uses no reserved adaptation identity and no
common-OOD combination label. Ridge uses the fixed standardization, an
unpenalized bias, and one-hot targets. The MLP uses seed-matched initial weights
across scopes, the original stored per-seed/per-experience training orders,
120 epochs, batches of 128, AdamW learning rate 0.0015, weight decay 0.0001,
and gradient clipping at 5. Every scope receives the same 1,320 updates.
The only supervised loss is motif cross entropy. This differs explicitly from
the v5 original three-task training; an oracle-versus-whole comparison holds
this new fitting protocol fixed, but a new-head-versus-original-head comparison
also changes the objective and optimization conditions.

All hyperparameters and the final epoch are fixed in advance. Familiar
validation can be recorded but cannot select a checkpoint. No OOD result selects
encoder, scope, classifier, regularization, initialization, or stopping time.
Evaluate own IID and common OOD independently. The primary numerical contrasts
are oracle-target minus whole-graph ordinary common-OOD motif accuracy, paired
by encoder seed, for each classifier, v5 regime, and experience. Retain balanced
accuracy, full eight-label predictions, each seed, fit accuracy, both IID sets,
and matched random-source fits. Uniform guessing has expected accuracy 12.5%.

## Separately labeled cross-position probes

Use the exact scale-v4 `direct_support.json` samples: three draws, each source
position, 32 identities per motif, 256 balanced source-position labels. Fit the
same alpha-1 standardized ridge to each scope and encoder; test the other two
positions. Preserve all six directed comparisons and the exact archived target
identity order. There is no target-position label in a given directed fit, but
the source-position support can contain combinations excluded from original
encoder training. This is a separately supervised diagnostic probe, not the
original-source-only experiment and not original-head zero-shot generalization.

Report balanced accuracy on target-position all-analysis examples and on their
common-OOD and own-IID strata, retaining ordinary accuracy, denominators, class
counts, and predictions. Each model's primary cross-position summary averages
the three support draws and six directions within seed before the three-seed
mean and sample SD. The directions and support draws are correlated, not
independent replicates. Shared random encoders may be scored against both A/B
strata but are counted once for the all-analysis random reference.

## Interpretation and immutable records

If oracle pooling succeeds from the original source labels and whole-graph
pooling fails under the same readout protocol, supplied localization/pooling is
an important limitation for this fixed setup. This does not demonstrate that
whole-graph pooling has erased all motif information. If the actual whole-graph
vector transfers under the diagnostic source-position labels but its
original-source-label head fails, the discrepancy implicates the label protocol
or rule selection in addition to accessible information. If both fail, inspect
the concrete differences between the earlier probe and present conditions
rather than inferring complete absence of information.

The original motif-position binding permits more than one rule to fit the
source labels. Common-OOD failure therefore does not by itself establish lack
of motif information. Successful oracle heads are conditional on supplied
membership. Nonlinear readout success is evidence for that decoder and support,
not universal nonlinear accessibility. These synthetic motifs, marked queries,
fixed budgets, and three seeds limit broader chemical conclusions.

Archive source references and hashes, frozen-state inventories, exact identity
plans and epoch orders, annotated pooling masks, feature arrays, intact and
intervened logits, per-example predictions, readout states and fitting histories,
per-model and paired seed statistics, English reports and figures, and a SHA-256
manifest in a new directory. Preserve all 536 manifest-covered files in the
seven preceding records. An independent auditor recomputes prediction-derived
metrics, margins, contrasts, support isolation, all fixed ridge predictions,
all stored readout-head predictions, and selected encoder features without
importing the new experiment implementation. A shortened smoke run validates
the pipeline but is excluded from primary evidence.
