# Matched frozen readouts across molecular pooling scales

This protocol is fixed before the primary scale-v4 run. No encoder is trained or
adapted. All completed scientific records remain unchanged.

## Questions and matched controls

Does pretraining improve the accessibility of motif information beyond the
corresponding random graph encoder? At what pooling scope is the information
accessible, and does source-position decoding transfer without target-position
labels? Source-labeled probe transfer and refitting with reserved new-combination
labels are separate experiments.

Use the original hard-v2 A, B, and random checkpoints at seeds 0, 1, and 2.
Each random checkpoint is the exact initialization used by both pretrained models
at its seed. Parameters remain frozen and their full-state and encoder hashes are
checked before and after extraction. All conditions use identical support IDs,
support-only feature standardization, eight-output multiclass ridge, and alpha 1
fixed in advance. No test score chooses an alpha, layer, scope, support sample,
or stopping point. There is no new pretraining or hyperparameter search.

[Akhondzadeh, Lingam, and Bojchevski (AISTATS 2023)](https://proceedings.mlr.press/v206/akhondzadeh23a.html)
motivate including random graph representations in probing studies.
[Kirichenko, Izmailov, and Wilson (ICLR 2023)](https://arxiv.org/abs/2204.02937)
motivate separating feature availability from the inherited prediction rule.
Neither paper guarantees a molecular outcome here; this is not a replication.

## Pooling support and input receptive support

The full original graph, atom features, and query markers are unchanged for every
scope. Apply post-hoc masks to contextual node states at all depths 0 through 8:

| Scope | Pooling operation | Width |
| --- | --- | ---: |
| root | Target attachment atom | 64 |
| target | Mean of the nine known target atoms | 64 |
| target_plus1 | Target plus atoms within one graph bond | 64 |
| target_plus2 | Target plus atoms within two graph bonds | 64 |
| target_plus3 | Target plus atoms within three graph bonds | 64 |
| whole_graph | Mean of all real atoms | 64 |
| legacy_graph | Original seven-part global/query readout | 448 |

The six 64-dimensional readouts support matched-width scope comparisons. The
448-dimensional reference additionally changes width and query-conditioned
pooling. It is not an isolated test of spatial support. Non-root means accumulate
in float64 before returning float32, preventing node-order rounding fingerprints
for equal atom-state multisets. Root and legacy operations remain exactly those
of the original model for continuity with the prior readout experiment.

Every inference batch has the configured 128 graph slots. The final short batch
repeats its own valid graphs to fill those slots, then discards the repeated
outputs. The model has per-node LayerNorm and no cross-graph normalization or
messages, so repetitions do not change its mathematical individual-graph
function. Fixing the GPU matrix shapes prevents a final-batch numerical
fingerprint in otherwise constant layer-0 features. Verify root and target
layer-0 vectors are bitwise constant over every selected identity and model.

Target membership is oracle annotation used only for measurement, never an
encoder input. A small pooling mask does not enforce local input dependence:
an eight-layer node already receives messages from its eight-bond neighborhood.
Archive each mask and each depth-dependent receptive union, including atom
counts inside/outside the target and whether the union spans the molecule.
These are structural upper bounds on input dependence, not measured causal
influence. Successful target pooling establishes accessible information at those
nodes conditional on supplied membership, not an automatic localizer or a pure
local computational circuit.

## Direct cross-position probe transfer

For each source position (0 ortho, 1 meta, 2 para) and support repeat 0, 1, or 2,
sample 32 identities per motif from that position in the archived analysis pool.
Every fit thus uses 256 source-position labels with equal eight-class counts.
Use one NumPy default_rng with seed 50000 + 100*repeat + source_position. Within
each class 0 through 7, sort eligible identities by molecular ID, sample without
replacement, then sort the concatenated chosen IDs before fitting. The same
source labels are used for A, B, random encoders, every depth, and every scope.

Predict the other two positions, yielding all six ordered directions. Source and
target molecular identities are disjoint within each comparison. Individual IDs
can be source examples in one direction and tests in another; these directions
are correlated measurements and are never additional seed replicates. Source
readout labels can include combinations absent during encoder pretraining. This
is a source-labeled probing experiment with zero target-position readout labels,
not the original pretrained classifier's zero-shot result.

Report target-position all-analysis tests, their common-OOD subset, and their
own-experience IID subset separately. Use balanced accuracy as the primary
cross-position statistic; retain ordinary accuracy, denominators, predictions,
class counts, and every direction. Restricting a test subset does not restrict
the classifier's eight output labels. Uniform prediction has 12.5% expected
accuracy even when that subset contains fewer than eight true classes.

## Refitting with reserved new-combination labels

Reuse all three complete archived v3 support sets exactly: 32 new-only identities
or the same 32 new plus the experience's 32 old replay identities. New support
was excluded from both pretraining curricula and all analysis tests. For every
trained/random pair, both normalization and fitting use exactly the same support
and algorithm. Evaluate both IID sets and common OOD independently. Retain
primary repeat 0 separately from the supplemental within-seed repeat mean.
Using new-combination labels demonstrates supervised adaptation, not zero-label
generalization to those combinations. Old replay labels have a different budget
from new-only support; this comparison is explicit.

## Statistics, numerical controls, and records

The primary depth is 8 and primary direct direction is ortho to meta, specified
before execution. Report every depth and direction, plus the mean of the six
directions within each encoder seed. Average support draws within seed before
computing means and sample SD across the three seeds. Trained-minus-random
differences are paired by seed, support draw, depth, scope, experience, and test.
Do not count three support draws or six directions as independent encoders.

Root and target layer-0 controls have no motif-distinguishing ordinary feature
multiset under this dataset design. Numerical constant-feature checks and the
12.5% uniform reference constrain interpretation of these controls. The legacy
readout retains its original float32 reductions for reproducibility; deep-layer
results and matched-width pools are the primary spatial measurements.

Archive fixed identity plans, masks, locality counts, input manifests, source and
checkpoint hashes, frozen features with ordered IDs, readout predictions, metric
tables, seed-level summaries, paired deltas, English reports, figures, and a
SHA-256 manifest in a new directory. Large representation arrays remain local.
An independent auditor reconstructs masks, recomputes every prediction-derived
score and seed statistic, and checks a declared subset of readout solves and
encoder feature calculations. Causal localization, arbitrary chemical spaces,
unseen motif topologies, alternative decoders, and alternative support budgets
remain outside this experiment.
