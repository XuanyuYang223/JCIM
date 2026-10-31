# Independent review of the scale-v4 report

Review date: 2026-10-31. This review reads the completed archive and does not change its source measurements, report, figures, model states, or manifest. The independent design-audit process owns feature, prediction, fitting, and provenance verification; the present review checks statistical reporting, visual presentation, and the interpretation of the spatial/depth comparisons.

## Frozen record and reporting checks

- Execution source: `59d9d1647e3fa1960e31a39b667122863e74ff46`.
- Record: `experiments/scale_v4`.
- Manifest SHA-256: `b84a3c9332384457ad6eec3ca6feb6a6531dc54c3b6d89274cd5d6fba5423fac`.
- Report SHA-256: `d9a7a8994c21da7ce07b5726df1b0101e70cfd6338288ec29328e4d27eef88d8`.
- Machine-readable summary SHA-256: `f588fe809794e5044da8eea70e02828c0b0be2930f860560bff1dce03484d614`.

An independent aggregation of the saved CSV rows checked all 40,824 reported metric cells across the primary, repeat-averaged, direction-specific, direction-averaged, and matched trained-minus-random summaries. Support draws and, where appropriate, the six directions were averaged within each encoder seed before the sample SD was computed across the three encoder seeds. Every reported seed count was three. The maximum absolute discrepancy in means, sample SDs, or per-seed values was `3.3306690738754696e-16` on the 0-to-1 scale. No summary discrepancy was found.

The displayed report tables use balanced accuracy. They must not be silently compared with an earlier experiment's ordinary-accuracy tables as though the metric were identical. The matched v4 trained/random comparison is internally consistent; numerical continuity with v3 is addressed separately.

Both PNG figures were inspected visually. Legends, panel titles, scope labels, the 12.5% uniform reference, the separation of the 448-dimensional legacy reference, and the primary support/layer designation are readable. The primary figure correctly separates new-label adaptation, old-example retention, and source-position-only transfer. Both SVG files parse as XML and have nonempty view boxes. The depth heatmaps have individually labeled color bars; colors should be interpreted against each panel's own numeric scale. Some primary transfer SD bars extend outside the physical accuracy range and are clipped by the displayed axis limits, while the full numerical SD remains available in the tables. These are presentation caveats, not reasons to alter the frozen archive.

## Spatial scale changes what can be read

At the predeclared final layer, target-substituent mean pooling strongly improves source-only transfer compared with the single attachment-root representation. The following values average all three support draws and all six directed position pairs within seed, then report mean +/- sample SD across the three encoder seeds:

| Pooling scope | Pretrained A | Pretrained B | Matched random |
| --- | --- | --- | --- |
| Root, 64 features | 20.3 +/- 2.5% | 25.7 +/- 7.2% | 35.9 +/- 2.4% |
| Target mean, 64 features | 69.5 +/- 16.4% | 70.3 +/- 13.1% | 99.977 +/- 0.039% |
| Whole-graph mean, 64 features | 27.3 +/- 6.3% | 33.1 +/- 3.0% | 80.7 +/- 16.2% |

The target pool is thus more informative under this linear transfer protocol than either the single root or an unweighted whole-molecule mean. However, the random target-pool baseline is even stronger. High target-pool accuracy cannot be credited to pretraining in this experiment.

The fixed-support primary ortho-to-meta target-pool scores are 63.8 +/- 44.7% for A and 73.0 +/- 23.6% for B, against 100.0% for the random encoder. These are not the same statistic as the six-direction, repeat-averaged values above. For example, A seed 1 averages 12.5% for ortho-to-meta but 87.5% for meta-to-para; selecting a favorable direction or treating the pooled direction scores as independent model replications would conceal important variation.

Labeled refitting remains a different question. With the primary 32-new-plus-32-old support, the final target pool gives new/old balanced accuracy of 98.0/100.0% for A and 93.9/94.9% for B; the matched random values are 100.0/100.0% and 93.9/94.9%. In contrast, the legacy graph readout has a paired new-test pretraining advantage of 15.2 +/- 3.3 percentage points for A and 9.8 +/- 3.7 points for B on that same primary support. The advantage is conditional on pooling and protocol, rather than a uniform improvement in all available structural information.

## The complete depth profile rules out an all-layer absence claim

The source-only target-pool depth profile is especially important. All depths 0 through 8 were declared before the primary execution. No best-layer selection is needed to observe the difference between the first message layer and the predeclared final layer.

| Layer | Pretrained A | Pretrained B | Matched random |
| --- | --- | --- | --- |
| 0 | 12.5 +/- 0.0% | 12.5 +/- 0.0% | 12.5 +/- 0.0% |
| 1 | 100.0 +/- 0.0% | 100.0 +/- 0.0% | 100.0 +/- 0.0% |
| 2 | 93.3 +/- 8.7% | 83.0 +/- 16.9% | 99.5 +/- 0.9% |
| 3 | 96.8 +/- 2.9% | 84.5 +/- 7.8% | 85.9 +/- 6.7% |
| 4 | 87.2 +/- 14.4% | 78.7 +/- 15.0% | 96.9 +/- 5.4% |
| 5 | 88.3 +/- 9.3% | 75.3 +/- 10.2% | 99.9 +/- 0.2% |
| 6 | 84.2 +/- 13.8% | 78.0 +/- 15.8% | 98.8 +/- 1.9% |
| 7 | 85.3 +/- 5.1% | 74.6 +/- 19.4% | 94.9 +/- 8.6% |
| 8 | 69.5 +/- 16.4% | 70.3 +/- 13.1% | 99.977 +/- 0.039% |

These are balanced accuracies with all support draws and directions averaged within seed. At layer 1, all individual source/target/support/seed comparisons are perfect for both pretrained experiences and the random condition. Therefore, the trained encoders do contain a cross-position-readable motif code at a shallow distributed pooling scale. Earlier root-only and final-layer failures cannot support an absence claim across every node set and depth.

The first message layer does not demonstrate a benefit learned through pretraining: the same perfect result occurs with the matched random encoder. Rather, the architecture plus oracle target pooling already provides a sufficient feature map for these eight repeated motif types. The later trained representations show lower transfer under the fixed readout, but the depth trend is not monotonic and trained A exceeds random at layer 3. It would be inaccurate to generalize the final-layer deficit into a claim that training worsens every representation at every depth, or that later layers destroy all motif information.

The architectural receptive support also differs sharply by depth. The target pool always contains nine atoms. At layer 1, its input reach is exactly ten atoms, comprising the target and one boundary neighbor; it never covers the entire molecule. At layer 8, its mean input reach is 20.794 atoms out of a mean molecule size of 20.807 atoms, and it reaches the entire graph for 98.73% of the 1,649 encoded identities. These counts are architectural reachability bounds, not causal evidence that every reachable atom influences a prediction. The target memberships remain oracle analytic masks, and even shallow degree/hydrogen inputs retain boundary chemistry.

A supported synthesis is that structural readability depends jointly on pooling scope, message depth, and the support distribution used to fit the readout. Pretraining improves some graph-level labeled refits while leaving the shallow target-pool solution unnecessary to learn and making final-layer source-only transfer less reliable than the matched random baseline. This is more specific than either an all-layer absence of local information or a general pretraining-induced gain in compositional reuse.
