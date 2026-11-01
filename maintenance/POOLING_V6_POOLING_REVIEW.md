# Pooling-v6 independent pooling and report review

Reviewed record: `experiments/pooling_v6`, execution source commit
`8a5e6997e7bff4ebc1e0d6438a9847fd8623dee2`.
This review reads completed numerical records and figures without modifying
experiment files. It supplements the independent numerical auditor; it does not
repeat the complete encoder/classifier audit.

## Continuity with the completed scale-v4 diagnostic

The v6 `frozen_readout` encoders are the original trained A/B encoders. The random
encoders are also the same archived random initializations. Their layer-1 target
and whole-graph directed probes have the same supports, regularization, pooling
operations, test identities, and supervision as the corresponding v4 probes.
These are existing diagnostic results, not a new independent replication.

An independent CSV/NPZ comparison, using molecular indices and identities rather
than file row coincidence, found:

- All **1,296 matching metric rows** agreed exactly, including test counts.
  All **2,592 accuracy and balanced-accuracy values** had absolute difference zero.
- All **162 matching directed fits** produced identical predictions on all 1,377
  analysis molecules: **223,074 predicted labels**, zero mismatches.
- For each of nine original trained/random encoders, the feature archives shared
  1,569 molecular identities. Both layer-1 pools agreed bit for bit on these
  identities: **18 feature arrays, 1,807,488 coordinates**, zero mismatches.

The matching ordinary/balanced results include every support repeat, source and
target position, A/B scoring context, and target-analysis/common-OOD/own-IID
stratum. Random A/B copies are checked as shared predictions rather than counted
as additional independent encoder seeds.

The existing v4 all-target-position balanced results are therefore the correct
reference: whole-graph means are **54.4 +/- 25.0%** for A, **50.4 +/- 1.7%** for B,
and **74.2 +/- 7.9%** for random encoders, while oracle-target means are 100% for
every seed. The v5 end-to-end encoders have different learned weights; their
corresponding v6 probes are the newly evaluated feature conditions. They obtain
48.4 +/- 12.7% and 57.9 +/- 13.0% for whole-graph A/B, with oracle-target 100%.

## Source-only readouts are the new localization comparison

The substantive new label protocol uses the original 1,386 training identities
and motif labels for each experience, without adding common-OOD labels. Both
64-dimensional pools use support-only standardization. Ridge regularization is
fixed at one. The fresh 64-to-64-to-8 MLP has matching initial parameters across
pools and conditions for each seed, the same archived minibatch orders, and a
fixed 120-epoch budget. Encoders remain frozen. Oracle boundaries affect pooling
only and do not enter the encoder.

The following values were independently recomputed from per-seed source metrics.
They are ordinary common-OOD motif accuracy, mean +/- sample SD across three
seeds; all oracle-target entries are 100% for both readouts.

| Encoder | Experience | Whole-graph ridge (%) | Whole-graph MLP (%) | Oracle target, either readout (%) |
| --- | --- | --- | --- | --- |
| V5 end-to-end | A | 37.4 +/- 19.5 | 17.6 +/- 15.2 | 100.0 +/- 0.0 |
| V5 end-to-end | B | 64.2 +/- 9.6 | 28.4 +/- 9.4 | 100.0 +/- 0.0 |
| Original trained | A | 27.4 +/- 8.5 | 18.9 +/- 20.0 | 100.0 +/- 0.0 |
| Original trained | B | 58.6 +/- 20.3 | 20.5 +/- 17.0 | 100.0 +/- 0.0 |
| Frozen random | A | 48.9 +/- 24.1 | 52.5 +/- 11.6 | 100.0 +/- 0.0 |
| Frozen random | B | 65.1 +/- 25.9 | 41.8 +/- 18.3 | 100.0 +/- 0.0 |

Paired oracle-minus-whole differences were computed within seed before taking
their mean and sample SD. For the v5 end-to-end encoders, these are A/B
62.6 +/- 19.5 / 35.8 +/- 9.6 percentage points for ridge and
82.4 +/- 15.2 / 71.6 +/- 9.4 for MLP. They agree with the report. Three seeds are
three encoder realizations; support directions, repeated fits, A/B uses of a
random encoder, and the deterministic 100% oracle outcomes do not increase that
number.

Every trained-encoder whole-graph ridge fit reaches **100% training and own-IID
accuracy**. Trained-encoder whole-graph MLP training accuracy is also 100%, and
own-IID accuracy ranges from 97.67% to 100%. Oracle pooling reaches 100% on
training, own-IID, and common-OOD for every trained and random encoder/readout.
Across random whole-graph controls, the minimum training/own-IID accuracies are
99.78%/99.78% for ridge and 98.99%/94.28% for MLP. Thus the trained ridge's partial
OOD recovery is not purchased by sacrificing its own-IID performance.

These results support localization/pooling as an important, intervenable
restriction in this diagnostic. They do **not** show that whole-graph averaging
removed all motif information: source-only whole-graph ridge already recovers a
substantial but variable fraction of OOD examples. The difference between ridge
and MLP on the same frozen features also makes decoder/objective choice relevant.
The comparison with the original v5 classifier changes feature isolation,
standardization, decoder, and motif-only versus three-task fitting conditions;
no single one of these changes is identified as the unique cause.

Oracle-target success is conditional on supplied boundaries. It is not a
boundary-free solution, a benefit attributable to pretraining, an experiment on
unseen motif topology, or proof that arbitrary local pools will work. The source
labels retain strong motif-position binding. Distinct rules can fit them equally
well, so an OOD failure alone cannot identify absent structure information.

## Existing branch outputs and visual checks

Direct reaggregation of the per-example common-OOD motif records confirms the
report's flip rates and absolute correct-class margin changes. For v5 end-to-end
A/B, zeroing the shallow branch changes **1.129%/0.752%** of predictions on
average; permutation changes no predictions. For original trained encoders,
zeroing changes **0%/0.226%**, and permutation changes **0%/0.075%**. The latter
0.075% is one accuracy-neutral wrong-label change in one of three seeds.

Outputs are not identical when predictions stay fixed. Across the four trained
regime/experience panels, permutation's mean absolute margin change is
0.00484--0.00687 raw logit units, while zeroing's is 0.13961--0.27237. The mean
relative centered-logit change is approximately 0.061--0.113% under permutation
and 1.98--3.21% under zeroing. These support small sensitivity to this
sample-swapping intervention, not complete branch nonuse. Raw margin scales are
head-specific, and neither intervention supplies a causal decomposition of
learning or a uniquely identified semantic mechanism.

Both `figures/pooling_source.png` and `figures/branch_outputs.png` were visually
inspected. Axes, legends, titles, and labels are readable English. Pooling panels
separate readouts and encoder regimes, show individual seeds and sample SD, and
label oracle localization. Branch panels separate prediction flips from margin
changes and explicitly identify raw head-specific units. The report correctly
keeps source-only fitting separate from the additionally supervised directed
diagnostics and preserves ordinary versus balanced metric definitions.

Review disposition: **PASS** for the reviewed continuity, summaries, diagnostic
interpretation, and figure readability. A full numerical audit is maintained
separately.
