# Independent interpretation of frozen scale-v4 readouts

Target-substituent pooling recovers more cross-position information than reading one attachment atom, but the matched random encoder performs even better. Pretraining improves some graph readouts after new labels are supplied; this does not establish that training created a reusable local code. The strongest new distinction is between pooling-dependent accessibility, architectural random features, and transfer under the trained representation.

This separate post-run note preserves all scientific records. The [experiment](../experiments/scale_v4/REPORT.md) executed commit `59d9d1647e3fa1960e31a39b667122863e74ff46`; its [manifest](../experiments/scale_v4/MANIFEST.json) SHA-256 is `b84a3c9332384457ad6eec3ca6feb6a6531dc54c3b6d89274cd5d6fba5423fac`.

Numbers below are **balanced accuracy (%)**, mean +/- sample SD across three encoder seeds. Support draws are averaged within seed before computing supplemental statistics. Directions are also averaged within seed when specified. Uniform eight-output chance is 12.5%; empirical random-encoder performance is a separate control. Every paired comparison uses identical supports, support-only standardization, fixed alpha=1, and the corresponding archived random initialization.

## 1. Pretraining improves new-label refitting for some pooling choices

The primary comparison fixes layer 8 and support repeat 0, with 32 reserved new labels plus 32 old labels:

| Experience | Legacy graph: pretrained new/old | Legacy graph: random new/old |
| --- | --- | --- |
| A | 93.2 +/- 2.6 / 94.7 +/- 4.3 | 78.0 +/- 5.0 / 78.6 +/- 2.9 |
| B | 90.3 +/- 2.8 / 89.9 +/- 2.2 | 80.5 +/- 2.8 / 84.8 +/- 5.0 |

Across all three support draws, the paired pretrained-minus-random common-OOD advantage is 12.7 +/- 0.6 percentage points for A and 5.8 +/- 1.9 for B. Whole-graph mean pooling gives advantages of 7.1 +/- 4.2 and 11.0 +/- 0.8 points. This supplies the missing frozen-random control for a fixed-budget pretraining benefit.

Target pooling differs: its paired advantages are -0.3 +/- 0.5 and -0.7 +/- 1.1 points, with both encoders near ceiling. Better target-pool readability than a root probe does not itself demonstrate information learned through pretraining. Refit uses new-combination labels and establishes supervised adaptation, not label-free generalization.

## 2. Final-layer source-position transfer favors random target features

Each source fit uses 256 labels, 32 per motif, and zero target-position labels. The following layer-8 results average all three draws and six ordered directions within each seed:

| Experience | Scope | Pretrained | Random |
| --- | --- | --- | --- |
| A | Root | 20.3 +/- 2.5 | 35.9 +/- 2.4 |
| A | Target | 69.5 +/- 16.4 | 99.98 +/- 0.04 |
| B | Root | 25.7 +/- 7.2 | 35.9 +/- 2.4 |
| B | Target | 70.3 +/- 13.1 | 99.98 +/- 0.04 |

Target pooling improves the trained model's accessibility relative to its root readout, while retaining a roughly 30-point deficit against random target features. On combinations absent from both encoder curricula, target performance is A 60.7 +/- 23.8%, B 76.1 +/- 25.3%, random 100%. Primary ortho-to-meta target results are less stable: A 63.8 +/- 44.7%, B 73.0 +/- 23.6%, random 100%.

These source-labeled probes are distinct from the inherited classifier's zero-shot predictions. Their 256-label budget also differs from the 64-label refit. Thus their gap is not an isolated effect of adding target labels. Known motifs recur across contexts; novel-motif generalization is untested.

## 3. A shallow local solution already exists without encoder training

All depths 0-8 were predeclared; depth 8 remains primary. In the secondary profile, layer-1 target pooling reaches **100% for every seed, support draw, and direction in both pretrained and random conditions**, compared with 12.5% for the constant layer-0 target bag. One message step plus fragment-wide pooling exposes sufficient connectivity information without training.

The layer-1 target receptive union contains exactly ten atoms: nine target atoms and the adjacent ring atom. Original degree/hydrogen features still include boundary chemistry. This supports a shallow architectural route to localized readability, conditional on oracle pooling membership.

At layer 8, root and target receptive unions instead cover the complete molecule for 98.73% of selected identities, with mean coverage 99.94%. A small final-layer pooling mask cannot establish independence from remote context. These structural unions bound possible influence; they do not measure causal dependence.

## Validation and limits

The [matched-runtime audit](SCALE_V4_SINGLE_THREAD_AUDIT.json) validates all source/new manifests, masks, locality counts, 54,432 row metrics, 9,072 prediction vectors, nine frozen checkpoints, all summaries, and 1,904 independent ridge refits, including every primary-layer fit. CPU feature reconstruction uses four deterministic molecule sentinels per encoder.

The retained [initial audit](SCALE_V4_AUDIT.json) failed five layer-1 root refits because unpinned BLAS threading changed six argmax labels at score gaps no larger than 1.1e-16. The unchanged auditor reproduces every checked prediction exactly under the experiment's single-thread runtime. [Environment evidence](SCALE_V4_AUDIT_ENVIRONMENT.json) records commands and hashes; no tolerance was loosened or scientific value overwritten.

[Independent continuity checks](SCALE_V4_NUMERICAL_CONTINUITY_WITH_SUPPORT.md) also establish exact equality of 162 aligned archived feature arrays, 57,106,944 coordinates, and both metrics in all 216 matched v3/v4 readout pairs. Apparent 74.9/73.0 differences are ordinary versus balanced accuracy, not numerical drift.

These findings concern eight controlled known motifs, one architecture, linear probes, and three seeds. They neither establish general biochemical transfer nor rule out other decoders. The defensible contribution is separating pooling-scale accessibility, random architectural information, new-label adaptation, and cross-position transfer under matched controls.
