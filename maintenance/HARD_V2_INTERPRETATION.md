# Independent interpretation of the hard-v2 experiment

The higher-difficulty run reveals a clear failure of zero-shot composition: both training experiences reach 100% on every task in their own familiar-combination test, while connectivity classification and three-task joint accuracy collapse on common unseen combinations. Position recognition transfers perfectly. Full-model adaptation substantially improves common-OOD performance, with a large loss of familiar-combination retention. The measured final-layer local readout does not show a training advantage over random message passing.

## Record and statistical unit

This is a post-run interpretation of the completed primary record, not a change to its protocol, source, measurements, or weights. The recorded source commit is `a29d45f859290a77155b01dfe1626b0b5390e900`; completion was `2026-10-31T05:56:14.212195+00:00`. The 11 input files used here were independently checked against their archived SHA-256 values. The separate [protocol](../configs/HARD_V2_PROTOCOL.md), [archive](../experiments/hard_v2/REPORT.md), and [configuration](../experiments/hard_v2/config.json) remain the experiment record.

All accuracy entries below are percentages, reported as mean +/- sample standard deviation across the three encoder seeds [0, 1, 2], rounded to three decimals. Adaptation support repeats are averaged within each seed before this calculation. Three support repeats do not create nine independent encoders. Signed changes use percentage points and paired seed differences. Geometry statistics instead summarize correlated model pairs, with the pair count shown explicitly.

The dataset has 6,912 unique molecular identities, 1,386 training identities per experience, 462 iid_a test identities, 472 iid_b test identities, and 443 common-OOD test identities. The 223-identity adaptation pool is excluded from both pretraining sets and all evaluation identities. Motif counts are matched between experiences; position marginals differ by design. Each experience observes each motif in exactly one position, so this is a severe motif-position correlation stress test. It is not unrestricted molecular or property generalization.

## Zero-shot behavior

Joint accuracy means all three answers are correct on the same molecule. It cannot be reconstructed by multiplying marginal accuracies.

| Model | Evaluation | Motif | Position | Distance | All three jointly |
| --- | --- | --- | --- | --- | --- |
| A | iid_a | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 100.000 +/- 0.000 |
| A | iid_b | 0.494 +/- 0.856 | 100.000 +/- 0.000 | 79.944 +/- 19.734 | 0.494 +/- 0.856 |
| A | ood_common | 0.000 +/- 0.000 | 100.000 +/- 0.000 | 86.832 +/- 15.192 | 0.000 +/- 0.000 |
| B | iid_a | 2.092 +/- 3.624 | 100.000 +/- 0.000 | 83.622 +/- 12.832 | 0.577 +/- 1.000 |
| B | iid_b | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 100.000 +/- 0.000 |
| B | ood_common | 4.214 +/- 7.104 | 100.000 +/- 0.000 | 75.922 +/- 18.096 | 1.580 +/- 2.544 |
| Random | iid_a | 12.554 +/- 0.943 | 31.602 +/- 8.017 | 1.010 +/- 1.750 | 0.000 +/- 0.000 |
| Random | iid_b | 12.429 +/- 0.245 | 37.782 +/- 6.721 | 1.766 +/- 3.058 | 0.000 +/- 0.000 |
| Random | ood_common | 11.964 +/- 1.968 | 37.397 +/- 1.601 | 0.527 +/- 0.912 | 0.000 +/- 0.000 |

Both trained experiences have identical, saturated own-IID performance. Their common-OOD motif and joint performance is poor even though all six trained models classify position perfectly. Distance transfers partially and varies substantially by seed. Thus ordinary test accuracy alone misses the main failure, and a single aggregate across tasks would conceal the difference between relationship recognition and connectivity classification.

## Prediction binding diagnostic

The separate [binding diagnostic CSV](HARD_V2_BINDING_DIAGNOSTIC.csv) contains all 18 trained-model-by-test-split rows, with exact counts in each predicted cell. For every prediction, the diagnostic cell is (predicted_family + true_position) % 3. Training permits cell 0 for A and cell 1 for B. All common-OOD truths have cell 2. True positions are used deliberately; this is a diagnostic of predicted family binding and does not assume an internal computation by the model.

| Model | OOD n | Predicted cell 0 | Predicted cell 1 | Predicted cell 2 | Training-cell binding | Motif accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| a_seed0 | 443 | 443 | 0 | 0 | 100.000% | 0.000% |
| a_seed1 | 443 | 443 | 0 | 0 | 100.000% | 0.000% |
| a_seed2 | 443 | 442 | 1 | 0 | 99.774% | 0.000% |
| b_seed0 | 443 | 0 | 443 | 0 | 100.000% | 0.000% |
| b_seed1 | 443 | 0 | 388 | 55 | 87.585% | 12.415% |
| b_seed2 | 443 | 0 | 442 | 1 | 99.774% | 0.226% |

On common OOD, A predictions satisfy their training-cell restriction in 99.925 +/- 0.130% of cases; B predictions do so in 95.786 +/- 7.104%. Whenever a prediction stays in the training cell on these OOD examples, it is necessarily a wrong motif answer. This directly accounts for those errors at the level of observable predictions. The pattern is consistent with reliance on the trained motif-position binding. It does not establish a causal circuit or prove that motif connectivity was never represented elsewhere.

## Local information accessible across positions

The local probe fits labeled ortho examples and evaluates identity-disjoint meta examples. The declared final layer is 8. Raw-root features are identical across motifs; the macro-accuracy reference is 12.5%. The probe-training examples come from the untouched encoder test partition and supply labels for the separate readout; this is supervised representation analysis, not zero-shot readout training.

| Representation | Accuracy | Balanced accuracy | Shuffled-label accuracy |
| --- | --- | --- | --- |
| Raw root | 12.967 (one fixed feature set) | 12.500 (one fixed feature set) | 12.967 (one fixed feature set) |
| Random | 37.143 +/- 3.077 | 37.135 +/- 3.902 | 12.821 +/- 7.034 |
| A | 20.513 +/- 11.580 | 20.961 +/- 13.928 | 9.304 +/- 8.058 |
| B | 21.905 +/- 16.515 | 20.833 +/- 14.434 | 18.828 +/- 1.843 |

Paired final-layer balanced-accuracy differences against the same-seed random encoder are A: -16.173 +/- 17.737 percentage points; B: -16.301 +/- 14.006 percentage points. The experiment therefore provides no positive training advantage under this declared local cross-position probe. A weak linear root readout does not demonstrate that information is absent from every node, layer, nonlinear decoder, or graph readout.

## Adaptation and retention

Positive budgets perform 80 AdamW updates of the complete encoder and inherited task heads, using the same nested support identities for every condition and seed. Zero shots uses the original heads unchanged. The random condition is trained from initialization with the same new labels and adaptation updates; it has not received the pretraining compute or labels.

Common-OOD evaluation:

| New labeled molecules | Initialization | Motif | Position | Distance | All three jointly |
| --- | --- | --- | --- | --- | --- |
| 0 | A | 0.000 +/- 0.000 | 100.000 +/- 0.000 | 86.832 +/- 15.192 | 0.000 +/- 0.000 |
| 0 | B | 4.214 +/- 7.104 | 100.000 +/- 0.000 | 75.922 +/- 18.096 | 1.580 +/- 2.544 |
| 0 | Random | 11.964 +/- 1.968 | 37.397 +/- 1.601 | 0.527 +/- 0.912 | 0.000 +/- 0.000 |
| 8 | A | 63.256 +/- 3.885 | 98.520 +/- 0.832 | 69.300 +/- 4.634 | 45.799 +/- 4.628 |
| 8 | B | 68.748 +/- 3.424 | 96.790 +/- 1.806 | 67.093 +/- 7.534 | 48.834 +/- 4.891 |
| 8 | Random | 45.423 +/- 1.910 | 60.697 +/- 5.305 | 28.543 +/- 1.127 | 12.691 +/- 0.414 |
| 32 | A | 91.096 +/- 0.434 | 100.000 +/- 0.000 | 90.670 +/- 3.131 | 83.722 +/- 2.519 |
| 32 | B | 95.335 +/- 2.685 | 99.850 +/- 0.199 | 92.827 +/- 2.298 | 88.789 +/- 2.956 |
| 32 | Random | 79.082 +/- 2.672 | 93.855 +/- 2.284 | 40.858 +/- 0.271 | 32.405 +/- 1.246 |
| 128 | A | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 99.674 +/- 0.386 | 99.674 +/- 0.386 |
| 128 | B | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 99.549 +/- 0.391 | 99.549 +/- 0.391 |
| 128 | Random | 93.429 +/- 4.022 | 100.000 +/- 0.000 | 49.611 +/- 3.538 | 46.551 +/- 3.830 |

Familiar-combination retention: A uses iid_a and B uses iid_b. The random reference averages its iid_a and iid_b scores within seed. These test distributions differ, so A and B retention values are guards against each model's own baseline rather than a matched-distribution ranking.

| New labeled molecules | Initialization | Motif | Position | Distance | All three jointly |
| --- | --- | --- | --- | --- | --- |
| 0 | A | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 100.000 +/- 0.000 |
| 0 | B | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 100.000 +/- 0.000 |
| 0 | Random | 12.492 +/- 0.351 | 34.692 +/- 4.233 | 1.388 +/- 2.404 | 0.000 +/- 0.000 |
| 8 | A | 75.204 +/- 2.659 | 99.880 +/- 0.208 | 78.620 +/- 7.084 | 58.754 +/- 8.120 |
| 8 | B | 67.891 +/- 3.728 | 99.576 +/- 0.393 | 73.234 +/- 4.842 | 51.671 +/- 4.448 |
| 8 | Random | 21.951 +/- 3.257 | 53.104 +/- 6.497 | 27.378 +/- 1.396 | 1.204 +/- 0.164 |
| 32 | A | 70.250 +/- 9.697 | 100.000 +/- 0.000 | 91.607 +/- 1.806 | 63.300 +/- 8.157 |
| 32 | B | 52.707 +/- 1.395 | 99.953 +/- 0.082 | 93.974 +/- 2.067 | 48.823 +/- 1.649 |
| 32 | Random | 13.363 +/- 6.149 | 91.197 +/- 2.882 | 37.779 +/- 0.560 | 3.432 +/- 1.476 |
| 128 | A | 62.338 +/- 23.005 | 100.000 +/- 0.000 | 99.808 +/- 0.333 | 62.145 +/- 22.850 |
| 128 | B | 37.900 +/- 8.980 | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 37.900 +/- 8.980 |
| 128 | Random | 1.195 +/- 0.841 | 99.988 +/- 0.020 | 43.328 +/- 1.081 | 0.651 +/- 0.628 |

Changes from each pretrained model's own zero-shot baseline:

| Initialization | Labels | OOD motif change | OOD distance change | OOD joint change | Own-IID motif change | Own-IID distance change | Own-IID joint change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 8 | +63.256 +/- 3.885 | -17.532 +/- 10.617 | +45.799 +/- 4.628 | -24.796 +/- 2.659 | -21.380 +/- 7.084 | -41.246 +/- 8.120 |
| A | 32 | +91.096 +/- 0.434 | +3.837 +/- 12.149 | +83.722 +/- 2.519 | -29.750 +/- 9.697 | -8.393 +/- 1.806 | -36.700 +/- 8.157 |
| A | 128 | +100.000 +/- 0.000 | +12.842 +/- 14.847 | +99.674 +/- 0.386 | -37.662 +/- 23.005 | -0.192 +/- 0.333 | -37.855 +/- 22.850 |
| B | 8 | +64.535 +/- 4.540 | -8.829 +/- 16.142 | +47.254 +/- 5.612 | -32.109 +/- 3.728 | -26.766 +/- 4.842 | -48.329 +/- 4.448 |
| B | 32 | +91.121 +/- 5.996 | +16.905 +/- 15.958 | +87.208 +/- 4.341 | -47.293 +/- 1.395 | -6.026 +/- 2.067 | -51.177 +/- 1.649 |
| B | 128 | +95.786 +/- 7.104 | +23.627 +/- 17.833 | +97.968 +/- 2.774 | -62.100 +/- 8.980 | +0.000 +/- 0.000 | -62.100 +/- 8.980 |

With eight labels, motif and joint OOD accuracy improve while OOD distance accuracy decreases from its relatively strong pretrained baseline. At 128 labels, OOD joint accuracy is nearly saturated for A and B, but own-IID joint accuracy falls from 100% to 62.145 +/- 22.850% and 37.900 +/- 8.980%, respectively. Position remains almost entirely intact, and distance recovers; the dominant lasting familiar-combination loss is motif classification. These are real adaptation gains accompanied by substantial forgetting or specialization.

Pretrained initializations outperform training from random initialization at the same new-label and update budgets. This supports a practical initialization benefit for this supervised adaptation protocol. It does not establish robust compositional zero-shot transfer, specify how many labels would be required to reach an unmeasured target, or identify a causal local representation mechanism. No replay or other retention intervention was part of this run.

## Matched common-OOD comparison of A and B

Each row below is the paired A-minus-B difference on the same 443 common-OOD molecules. Support draws are matched, then averaged within seed before differencing.

| Labels | Motif difference | Position difference | Distance difference | Joint difference |
| --- | --- | --- | --- | --- |
| 0 | -4.214 +/- 7.104 | +0.000 +/- 0.000 | +10.910 +/- 31.490 | -1.580 +/- 2.544 |
| 8 | -5.493 +/- 1.633 | +1.731 +/- 2.036 | +2.207 +/- 11.706 | -3.035 +/- 7.183 |
| 32 | -4.239 +/- 2.891 | +0.150 +/- 0.199 | -2.157 +/- 5.248 | -5.066 +/- 5.468 |
| 128 | +0.000 +/- 0.000 | +0.000 +/- 0.000 | +0.125 +/- 0.769 | +0.125 +/- 0.769 |

The zero-shot distance contrast varies strongly across the three matched seeds. B has a modest descriptive OOD motif/joint advantage at low adaptation budgets, while the conditions converge at 128 labels. These three-seed comparisons do not support a general ranking of training experiences or a reliable correlation between geometry and transfer.

## Geometry with factorial seed controls

Final-layer geometry uses the same 512 ordered test identities for both pooling definitions. Counts refer to model pairs; shared encoders make these observations dependent.

| Pooling | Pair category | Pairs | CKA | Procrustes |
| --- | --- | --- | --- | --- |
| local | Random / random, different seeds | 3 | 0.989 +/- 0.006 | 0.983 +/- 0.003 |
| local | Same data, different seeds | 6 | 0.850 +/- 0.057 | 0.934 +/- 0.014 |
| local | Different data, same seed | 3 | 0.716 +/- 0.062 | 0.884 +/- 0.021 |
| local | Different data, different seeds | 6 | 0.713 +/- 0.079 | 0.882 +/- 0.026 |
| local | Random / trained, same seed | 6 | 0.441 +/- 0.141 | 0.585 +/- 0.051 |
| graph | Random / random, different seeds | 3 | 0.652 +/- 0.143 | 0.867 +/- 0.047 |
| graph | Same data, different seeds | 6 | 0.904 +/- 0.049 | 0.957 +/- 0.014 |
| graph | Different data, same seed | 3 | 0.886 +/- 0.084 | 0.947 +/- 0.026 |
| graph | Different data, different seeds | 6 | 0.894 +/- 0.063 | 0.950 +/- 0.020 |
| graph | Random / trained, same seed | 6 | 0.164 +/- 0.069 | 0.434 +/- 0.051 |

Random local encoders already have very high CKA (0.989 +/- 0.006), exceeding both trained same-data and cross-experience local comparisons. Graph geometry shows a different pattern, with higher trained similarities than random/random. This pooling dependence and the weak local probe rule out interpreting similarity as direct evidence of functional reuse. Neither an increase nor a decrease was required for success.

At local layer 0, every molecule has the same raw marked-root attributes. Its embedding is constant across molecules, so centered variance is zero. The implementation returns CKA = 0 and Procrustes = 0 by convention in that degenerate case. Those zero values do not mean that informative representations are misaligned; a rise across message-passing depth from that baseline is not a measured learning effect.

## What this run supports and the next discriminating experiment

The result supports three descriptive conclusions: familiar-combination success can coexist with near-total connectivity failure on new bindings; a model can preserve positional and much distance competence while making those errors; and pretraining can accelerate supervised adaptation while leaving a major retention problem. It does not demonstrate the proposed shared local representation mechanism. In particular, final-root cross-position readout is weaker after training than at random initialization, and geometry cannot repair that missing functional evidence.

A useful next protocol would let every motif appear at two training positions, reserve a third motif-position cell for common OOD, and vary the two experiences through controlled background combinations while matching labels, counts, architecture, and updates. This would reduce the exact one-position-per-motif training correlation. A separately declared adaptation comparison with and without IID replay could then test retention. Both changes need new records and predefined comparisons; the completed hard-v2 test results should remain frozen.

The chemical space is synthetic and limited to specified aromatic scaffold variants with supplied query atoms. No experimental-property prediction, broad scaffold-held-out generalization, unknown-group localization, population-level geometry-transfer law, or causal intervention is established.

Derived audit artifacts: [binding counts](HARD_V2_BINDING_DIAGNOSTIC.csv). Numerical sources: [behavior](../experiments/hard_v2/behavior.csv), [probes](../experiments/hard_v2/probes.csv), [adaptation](../experiments/hard_v2/adaptation.csv), [geometry](../experiments/hard_v2/similarities.csv), and [predictions](../experiments/hard_v2/predictions.csv).
