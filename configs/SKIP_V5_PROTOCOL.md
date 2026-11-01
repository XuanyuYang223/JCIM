# Boundary-free shallow-path intervention

This protocol is fixed before the primary skip-v5 execution. The intervention
asks whether making shallow whole-graph information available to the task heads
improves original-model predictions on combinations absent from training.
It supplies no target-fragment membership and no target-combination labels.
All completed scientific records, weights, and manifests remain unchanged.

## Architectural hypothesis

Neighborhood aggregation can give identical representations to isomorphic
attributed neighborhoods before remote differences enter their receptive
support. This architectural constraint applies to random weights as well;
linear separation of the eight benchmark motifs is an additional empirical
observation. [Xu et al., ICLR 2019](https://arxiv.org/abs/1810.00826) provide
the relevant neighborhood-aggregation and expressivity framework, rather than
a guarantee of the specific empirical accuracies here.

The earlier oracle-target pooling experiment establishes accessible shallow
information conditional on supplied fragment membership. The current test
removes that membership from the readout protocol. It retains the original
18 atom features and marked query locations, so it is boundary-free rather
than query-free. Whole-graph averaging also includes unrelated context;
failure cannot refute availability under oracle target pooling.

## Three architectural arms

All arms use the original width-64, eight-message-layer encoder and the same
three task heads, with hidden width 64 and their original output classes.

| Arm | Classifier input | Input width | Total parameters |
| --- | --- | ---: | ---: |
| deep | Original final-layer graph/query readout | 448 | 189531 |
| deep_control | Deep readout plus an exact copy of its first mean block | 512 | 201819 |
| shallow_skip | Deep readout plus layer-1 whole-graph mean | 512 | 201819 |

The deep control's extra coordinates are a bit-identical copy of existing
coordinates, so it adds no information. It matches the skip's head dimensions
and 12,288 additional parameters while allowing a changed parameterization.
The shallow mean includes every real atom, excludes padding, accumulates in
float64, and returns float32. The inherited deep operations are unchanged.
Parameter matching does not equate numerical work or optimization trajectories.

Copy each head's first 448 columns, biases, and final linear layer from its
exact archived same-seed random initialization. Initialize the appended 64
columns to zero in both extended arms. The two extended heads initially have
identical parameter values and mathematically identical predictions; numerical
comparison against the 448-dimensional original records maximum logit error
and class agreement because matrix widths can change floating-point reduction.
These checks use only 128 original training identities, not test labels.

## Regimes and matched training

The primary end_to_end regime starts the complete model from the archived random
state and updates all parameters. Its original deep-arm baselines reuse the six
exact hard-v2 final checkpoints and training histories rather than rerunning
identical experiments. Their predictions are recomputed and checked against
the archived source. The two new 512-dimensional arms train from the same
original initialization and follow the same fixed budget.

The secondary frozen_readout regime fixes each original pretrained A/B encoder
and resets its task heads to the corresponding archived random heads. This
avoids the near-zero old-task loss of an inherited trained head making a new
path trivially inactive. Only heads are trained. Readout features may be cached
because the model has no stochastic layers or cross-graph normalization.
All frozen encoders must remain byte-identical to their original checkpoints.
This regime tests whether original source labels can learn to use existing
features; it does not evaluate adaptation with target labels.

Both regimes use exactly the original 1,386 training identities per experience,
the original familiar validation set, and three seeds 0, 1, and 2. All arms
receive the same stored epoch orders from default_rng(seed + 1701), 120 epochs,
11 batches per epoch, 1,320 AdamW updates, learning rate 0.0015, weight decay
0.0001, gradient clipping at 5, and the sum of the three original task losses.
The reused original baseline has this same historical budget and zero new
updates. No adaptation-reserve identity or common-OOD combination enters
training or validation.

Depth 1, all hyperparameters, and the final epoch are specified in advance.
History records only training loss and familiar validation. Common-OOD scores
are computed only at the fixed final endpoint. No OOD result selects a
checkpoint, variant, optimizer, or stopping time.

## Endpoints and secondary branch interventions

The primary endpoint is paired common-OOD ordinary motif accuracy of
shallow_skip minus deep_control, separately for A and B, with each seed's
difference retained. Report means and sample SD across the three initialization
seeds, original-deep comparisons, both IID guards, position and distance
accuracies, balanced accuracies, distance MAE, and all-three-correct joint score.
Validation molecules and three seeds are not interchangeable statistical units.

At the final state, additionally zero or permute the appended branch in both
512-dimensional arms without retraining. The deep 448 coordinates stay fixed.
The analysis-row permutation comes from default_rng(90000 + seed), shared
across experiences, regimes, and arms. It uses no labels. These are secondary
functional compatibility interventions; an accuracy drop can include
off-distribution branch inputs and does not uniquely identify semantic
information or an additive causal contribution.

## Interpretation and reproducibility

Frozen skip improvement over the matched deep control would support a useful
route from already-learned shallow features under original source labels.
End-to-end improvement could also involve changed representation learning and
optimization. A null result means this whole-graph path and fixed training
procedure were insufficient; it cannot establish that every shallow decoder,
localization rule, pooling scheme, or training objective would fail.
The two original curricula, supplied query locations, eight known synthetic
motifs, and three seeds limit generalization to other molecular settings.

Archive exact identity plans and epoch orders, input manifests, source hashes,
per-case initialization/final-state hashes and parameter counts, frozen feature
caches, training histories, final checkpoints, per-molecule predictions,
per-seed and paired tables, English reports, figures, and a SHA-256 manifest.
Large model and feature arrays stay local. A reduced smoke run uses fresh deep
training at its shortened budget and is explicitly excluded from primary
evidence. The primary source is committed and frozen before execution.
