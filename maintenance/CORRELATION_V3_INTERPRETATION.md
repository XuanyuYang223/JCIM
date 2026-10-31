# Independent review of the correlation intervention

The experiment reliably learns the directly exposed combinations, while transfer
to combinations kept unexposed depends strongly on the training experience and
initialization seed. Increasing counterexamples does not produce a consistent
monotonic improvement on the protected test.

## Audited evidence

The [independent CPU audit](CORRELATION_V3_AUDIT.json) passed without importing
the intervention generator. It verified all 77 intervention files and 131 source
files against their frozen manifests; independently reproduced every replacement;
checked 33,048 predictions across 24 models and the original 1,377 ordered
analysis identities; recomputed all 72 behavior panels and their eight task
metrics; and checked all 24 baseline and 24 summary panels. Every reported
sample standard deviation uses the three initialization seeds. Molecules and
test rows are not additional initialization replicates.

The captured intervention manifest SHA-256 is
`4d0323dac89048dcc8d5f1c788e78e513412ccd8cb1b5acffd3c5d0c765e1e01`.
The unchanged Hard v2 source manifest SHA-256 is
`4b69cd0335cd21ccb5e2a419c1301bfc8042e3bb5e9864322d6ad4ca1f398996`.
These are the manifest digests captured by the completed audit, rather than
claims about subsequent working-tree implementation revisions.

## What the denominators mean

Each training set retains 1,386 unique molecular identities. The requested
0%, 1%, 5%, and 20% interventions replace exactly 0, 14, 69, and 277 identities,
respectively. Rates divide by the complete training set, not by the subset of
even-motif examples eligible for replacement. Replacing an identity with another
of the same motif preserves all eight motif counts. All original odd-motif
training examples remain present, and their protected cell-2 combinations remain
absent from training and validation.

The exposed and protected tests contain 238 and 205 molecules, respectively.
Their accuracies must retain those separate denominators; an overall common-OOD
accuracy would require weighting by 238/443 and 205/443. Both subsets have four
true motif classes, but predictions range over eight classes. Uniform random
prediction therefore has expected accuracy and macro recall of 12.5%. A 25%
baseline would apply to a different classifier restricted to four outputs.

## Direct learning and broader transfer

At 14 replacements, exposed-test motif accuracy is 92.30% +/- 5.29% for A and
96.78% +/- 5.22% for B. At both 69 and 277 replacements, every seed reaches
100% exposed-test accuracy. Original own-experience IID motif accuracy remains
100% in every condition. These results establish direct learning of the
counterexample-supported combinations on independent molecular identities.

Protected-test results are weaker and much less consistent:

| Experience | Replacements | Protected accuracy, mean +/- sample SD | Individual seed accuracies |
| --- | ---: | ---: | --- |
| A | 69 | 22.93% +/- 16.78% | 41.95%, 16.59%, 10.24% |
| A | 277 | 35.45% +/- 30.04% | 44.39%, 60.00%, 1.95% |
| B | 69 | 17.07% +/- 15.03% | 22.93%, 28.29%, 0.00% |
| B | 277 | 8.94% +/- 4.40% | 13.17%, 4.39%, 9.27% |

In A, the larger mean at 20% is compatible with substantial gains in two seeds
and near failure in the third. In B, the mean decreases from 5% to 20%, and the
individual seeds do not follow a shared monotonic trend. The variation is a
finding, rather than evidence that one favorable seed represents a typical
outcome. These three-seed spreads are descriptive; they are not confidence
intervals or significance tests. No protected outcome selected an intervention
identity, checkpoint, regularizer, or training duration.

The results support some conditional transfer beyond directly exposed motif
types, but they do not establish stable recovery of a universal cross-position
rule. The intervention uses one predeclared even/odd motif partition and one
nested counterexample ordering. Initialization variation does not measure
uncertainty over alternative partitions, counterexample draws, or broader
chemical spaces. The substitution also changes position marginals while
preserving motif priors, so the intervention is not a complete isolation of
every possible distributional effect.

## Relation to the frozen-encoder experiment

The separate [readout and replay experiment](../experiments/readout_replay_v3/REPORT.md)
shows that original frozen graph representations can support strong new/old
performance through an appropriate fresh readout and replay. That evidence and
the present behavioral intervention answer different questions. Availability
of decodable information does not guarantee that the original training rule
will use it across protected combinations. These correlation curves do not
locate information in any layer, prove a causal representation mechanism, or
show that a specific feature is absent.

The archived [experiment report](../experiments/correlation_v3/REPORT.md),
[per-seed behavior](../experiments/correlation_v3/behavior.csv), and
[individual predictions](../experiments/correlation_v3/predictions.csv) retain
the complete measurements. This review adds interpretation outside the frozen
experiment and does not revise its measurements or hashes.
