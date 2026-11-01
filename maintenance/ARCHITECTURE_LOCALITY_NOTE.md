# Architectural locality and the interpretation of readable molecular structure

This note qualifies the interpretation of the completed scale-v4 experiment and
the predeclared skip-v5 intervention. It changes no archived measurements, model
states, or manifests. The implementation discussed here is the inherited
[message-passing encoder](../jcim/model.py), instantiated at width 64 and eight
message layers by [HardMolecularGNN](../jcim/hard_model.py).

## What the architecture guarantees

Each atom begins with the same learned affine embedding and ReLU applied to its
input attribute vector. Each message layer sums adjacent node states, concatenates
that sum with the node's current state, applies a shared two-layer MLP, adds the
result to the current state, and applies LayerNorm followed by ReLU. LayerNorm
acts on the features of one atom; it does not aggregate statistics across atoms
or molecules. Padding is masked. The encoder has no global pooling feedback,
attention across disconnected atoms, or stochastic layers.

For any fixed choice of weights, corresponding atoms with isomorphic attributed
rooted L-hop neighborhoods therefore have equal states after L message layers
in exact arithmetic. The same weights must be used on both graphs. The statement
does not assert that separately trained networks have identical vectors. It
follows by induction: equal initial attributes yield equal embeddings; a
neighborhood-preserving bijection yields equal multisets of neighbor states,
equal sums, and equal outputs of each shared nodewise operation. Random weights
satisfy this consistency property too. Floating-point reductions can introduce
small differences when atom order or matrix shape changes.

For a pooled atom set, an attribute-preserving correspondence over its expanded
L-hop support that also maps the pooled membership gives equal multisets of
pooled states, hence equal means. This is a sufficient condition, not a claim
that every pair of benchmark molecules satisfies it. Query-marker placements
and other input attributes must also agree under the correspondence. The
neighborhood-aggregation perspective is consistent with
[Xu et al., *How Powerful are Graph Neural Networks?*, ICLR 2019](https://arxiv.org/abs/1810.00826).
The consistency argument here follows from this repository's operations; it
does not require claiming that this particular encoder reaches the paper's
maximum expressivity.

## Consistency and discriminability are separate claims

Architectural consistency preserves a representation across equivalent
neighborhoods. It does not guarantee different motifs receive different
representations, or that eight motif classes are linearly separable. A constant
feature map would satisfy consistency while failing classification.

Scale-v4 separately established that the layer-1 oracle-target mean supports
100% balanced accuracy for every tested encoder seed, support draw, and directed
position comparison. The matched random encoders also reached 100%. This is an
empirical separability result for these eight known motifs and this labeled
probe protocol, rather than evidence that task training created the readable
code. The [independent scale review](SCALE_V4_REPORT_REVIEW.md) documents the
complete depth profile and its statistical aggregation. Source-position labels
used to fit a probe are distinct from the original task head's predictions on
combinations absent from its training set.

## Input attributes and pooling limit locality claims

The [18 input attributes](../jcim/hard_data.py) include element indicators,
aromaticity, charge, molecular degree, hydrogen count, ring membership, and five
query markers. Degree and hydrogen count retain boundary chemistry; ring
membership also comes from molecular preprocessing. A message-hop bound counts
access to these supplied attributes, not independence from every chemical fact
outside an extracted subgraph. The query markers identify particular atoms but
do not provide complete target-fragment membership.

The layer-1 target pool covers nine target atoms and reaches one boundary atom
in the archived benchmark. Its membership was supplied only for analysis.
Replacing that oracle pool with a whole-molecule mean changes the information
mixture: background atoms and marked query locations contribute, and their local
arrangements can carry position-related signals. Whole-graph averaging need not
preserve a clean motif-specific code even when the target pool does. At depth
eight, remote context can reach nearly every pooled target state. A weaker
cross-position linear probe then establishes reduced readability under that
decoder and support distribution; it cannot establish complete information
erasure or determine whether a different decoder could recover it.

## Meaning of the shallow-path intervention

The [skip-v5 protocol](../configs/SKIP_V5_PROTOCOL.md) appends the layer-1
whole-graph mean without supplying fragment boundaries or target-combination
labels. A matched control appends an exact duplicate of existing deep features.
Improvement with a frozen encoder would support a useful readout route from
existing features. End-to-end improvement could additionally reflect changed
representation learning or optimization. Failure would show that this global
mean path and fixed source-label training procedure were insufficient. It
would not refute all possible shallow pooling, localization, routing, or
training methods.
