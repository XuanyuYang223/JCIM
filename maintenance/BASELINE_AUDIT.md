# Independent baseline interpretation

## Record and scope

This note is an independent interpretation outside the frozen experiment
directories. No training, source edits, checkpoint edits, or test-driven tuning
were performed for this audit. The pilot source revision is `d26990f`; the
two-epoch smoke run precedes that source commit and is only a pipeline check.
Its archived environment records the earlier repository HEAD and source hashes.

Status on 2026-10-30: the pilot is complete, with three initialization seeds,
60 fixed training epochs per experience, and a frozen manifest. The SHA-256 of
`experiments/pilot_v1/MANIFEST.json` is
`1cc95a6ec43c91c379ed5322b7631356fa637ea802db756c26b6802a8a1ff01b`.
All pilot aggregates below use the complete run, superseding this note's earlier
provisional seed-0 snapshot. The experiment files were not revised.

Both runs enumerate 3,840 unique molecules on one benzene scaffold. The two
training experiences contain 2,371 molecules each, including the same 2,279
shared molecules (96.12% of either experience) and 92 exclusive molecules per
experience. Their common test partition has 999 molecules: 906 ordinary test
molecules, 49 fluoro-meta
molecules, and 44 chloro-meta molecules. Direct aryl-F meta testing is a subset
of only 12 molecules. The target-fragment mask and queried endpoints are supplied
to the model, so this pilot does not measure localization of unmarked groups.

## What the raw and random controls already solve

The following numbers come from `experiments/pilot_v1/probes.csv`. Accuracy is
reported as a percentage; all probes use the same molecule-disjoint fit and test
sets across conditions.

| Probe | Raw input | Random, layer 6 | Experience A, layer 6 | Experience B, layer 6 |
| --- | ---: | ---: | ---: | ---: |
| Family, ortho fit to meta test | 100.00 | 100.00 | 100.00 | 100.00 |
| Position, six-family fit to F/Cl test, graph readout | 37.34 | 82.30 | 82.02 | 100.00 |
| Position, six-family fit to F/Cl test, local fragment readout | 31.54 | 32.50 | 38.45 | 40.39 |

Model columns average three seeds; raw input has no encoder seed. The final
graph position accuracies are 90.46%, 56.43%, and 100.00% for random encoders;
100.00%, 60.17%, and 85.89% for A; and 100.00% for all B encoders. This exposes
substantial initialization dependence and shows that A does not improve this
particular final-layer linear relation probe on average.

The untrained graph position probe already reaches 99.17% at layer 1 for seed 0.
The final local position probe has mean balanced accuracy of 32.05% for random
encoders, 35.91% for A, and 39.33% for B. For example, A seed 0 has 38.59%
ordinary accuracy but only 33.33% balanced accuracy, the constant-class
baseline. Local position decoding is weak across conditions.

The family probe is saturated before training because ordinary atom features
and the supplied target-fragment mask expose the terminal family. Its success
demonstrates decodability across position but cannot establish that training
created reusable family information. Strong untrained graph position decoding
also shows that message passing and query readouts can produce useful relational
features without learned weights. These are architectural controls, not failed
experiments.

## Where training currently adds useful information

The adaptation endpoint fits a fresh standardized ridge distance classifier on
a frozen final-layer graph representation. It does not fine-tune the encoder.
The table gives mean exact-distance accuracy over five nested support-set
repetitions on the same 49 fluoro-meta test molecules, from
`experiments/pilot_v1/adaptation.csv`.

| Labeled support molecules | Raw input | Random | Experience A | Experience B |
| --- | ---: | ---: | ---: | ---: |
| 8 | 17.55% | 20.27% | 33.61% | 34.97% |
| 24 | 16.73% | 19.05% | 40.95% | 52.65% |
| 64 | 26.94% | 24.08% | 75.65% | 88.44% |

Each encoder column averages three seeds and five support repetitions per seed.
Raw input averages the same five support repetitions once. At 64 shots, sample
standard deviations across the three encoder-seed means are 5.11 percentage
points for random encoders, 5.63 for A, and 8.81 for B. The per-seed means are
24.49%, 18.78%, and 28.98% for random encoders; 73.47%, 71.43%, and 82.04% for
A; and 79.18%, 89.39%, and 96.73% for B.

The support pool has 92 molecules and is disjoint from the 49 test molecules.
Experience A saw none of those support molecules during pretraining;
experience B saw all 92. Both encoders are tested on molecular identities absent
from their pretraining sets. B is therefore an exposure reference, rather than a
second encoder for which support identities were equally unseen.

The support-pool modal distance is seven bonds, and a constant prediction of
seven gives 10/49 = 20.41% test accuracy. Only seven distance classes occur in
this test subset, despite the model's 12 output logits. Neither 1/12 nor the
accuracy of an untrained task head is an appropriate sole baseline for the
fresh-readout endpoint.

All three encoder seeds support the descriptive finding that training produces
graph features from which a new distance readout learns more effectively. This
does not isolate
local-code reuse: the readout contains global graph context and marked endpoint
vectors. The five support repetitions share one encoder and one test set, so
they are not five independent training seeds or five independent test cohorts.
The three seeds also share the same molecular test identities. No population
confidence interval or hypothesis-test result is established by these averages.

All six fully trained models score 100% on all three original task heads
across ordinary and held-cell test sets, including the direct aryl-F subset.
That endpoint is at ceiling and does
not currently demonstrate a contrast in held-cell behavior between models with
similar ordinary accuracy. A difference in a fresh frozen-feature readout is a
different observation and should be named explicitly.

The inherited distance heads already solve this held cell with zero adaptation
labels. The fresh ridge experiment is therefore a representation-readability
diagnostic; it does not establish a practical improvement over using the existing
head or show that either pretrained model requires new labels for this task.
Any label-efficiency claim must also condition on the labeled pretraining data,
rather than counting only the new readout labels.

## What the final similarity measurements show

Final-layer local-representation statistics from
`experiments/pilot_v1/similarities.csv` are:

| Pair group | Number of pairs | Mean CKA | CKA range | Mean Procrustes score |
| --- | ---: | ---: | --- | ---: |
| Random, different seeds | 3 | 0.7141 | 0.6440-0.8225 | 0.8565 |
| Trained, same data and different seeds | 6 | 0.9639 | 0.9467-0.9724 | 0.9571 |
| Trained, different experiences and matched initialization | 3 | 0.9760 | 0.9656-0.9816 | 0.9719 |
| Matched random versus trained | 6 | 0.5370 | 0.4448-0.6089 | 0.7080 |

Training yields reproducible local geometry in this controlled setting, while
the matched random-versus-trained comparisons show substantial geometric
change. The pair groups control initialization differently: the different-data
pairs share initialization, whereas same-data pairs do not. Their means should
not be compared as a causal effect of data exposure. The two experiences also
share 96.12% of their training identities. These values do not establish that
local alignment explains the fresh graph-readout differences, especially since
the similarity and distance-readout pooling definitions differ.

## What the smoke run can establish

`experiments/smoke_v1` used two epochs and one seed. Experience A and B both
achieved 100% family and position accuracy on ordinary test molecules, but
their distance accuracies were 40.07% and 42.94%. Their fluoro-meta distance
accuracies were 34.69% and 38.78%, respectively. Their final local-representation
CKA was 0.9671, while their final graph position probe accuracies were 94.61%
and 64.73%. The compared pooling definitions differ, so this is not a matched
counterexample relating the same representation's geometry to function.

The smoke 64-shot mean distance-readout accuracies were 26.94% raw, 24.49%
random, 28.98% A, and 27.35% B. These limited-training values establish that the
pipeline produces nontrivial results and controls. They are not the main
scientific result, and selecting a different training budget because of these
observations would require a separately recorded experiment.

## Defensible interpretation and limits

1. Molecule-disjoint generalization to the specified held terminal-family and
   ring-position combinations is feasible for this controlled scaffold.
2. Cross-position family decoding is already perfect using accessible input
   information in this pilot. It cannot serve as the main evidence of learned
   compositional reuse.
3. The frozen graph distance-readout endpoint is more informative than the
   saturated family probe. It shows a descriptive three-seed training advantage
   over raw and random features, while the original trained heads already solve
   the same held-cell task without adaptation labels.
4. High CKA is a geometric observation. It does not show useful transfer, and
   the current experiment does not identify a causal link between alignment and
   learning efficiency.
5. Results concern deterministic two-dimensional graph tasks on one scaffold,
   with supplied query masks. They do not establish molecular-property accuracy,
   broad scaffold transfer, or novel functional-group discovery.

## Primary methodological references

- [RDKit molecular file API](https://www.rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html#rdkit.Chem.rdmolfiles.MolToSmiles):
  `MolToSmiles` supports canonical molecular serialization. The pilot removes
  atom-map numbers before identity calculation. This is identity handling within
  the controlled grid, not general tautomer or salt standardization.
- [Kornblith et al., 2019](https://proceedings.mlr.press/v97/kornblith19a.html):
  CKA compares representation geometry and can identify cross-initialization
  correspondences. The restriction on causal or functional claims in this note
  is an interpretation boundary, not a transfer guarantee made by that paper.
- [Hewitt and Liang, 2019](https://aclanthology.org/D19-1275/):
  probe capacity and control tasks affect interpretation of decoding accuracy.
  Applying this caution to molecular probes is an inference. The pilot's
  sample-permuted-label probe is not the paper's word-type control task and
  should not be described as an exact implementation of it.
