# Independent review of the shallow-readout intervention

Review date: 2026-10-31. This review uses only the completed `experiments/skip_v5` record. It does not rerun training, select checkpoints, or change primary records or source code.

## Record and numerical verification

The reviewed record contains 36 model cases, 252 endpoint rows, and 115,668 saved molecule-level prediction rows. Its execution source commit is `e8969b8fe6bfddd4466c8d22c5479c2ca9be586e`. The completed `MANIFEST.json` SHA-256 is `39ebb1e0833557047709630b2d8c60ec773801267a585efdfaed5401107ed16b`.

Independent checks passed:

- All 131 files covered by the primary manifest match their recorded SHA-256 values.
- All 2,016 endpoint scalar metrics were recomputed from saved predictions and the archived source labels. These include ordinary and balanced motif, position, and distance accuracy, distance MAE, and joint accuracy. Molecular identities and all 84 model/intervention prediction panels match the 1,377-identity analysis set.
- All 1,056 metric-summary cells were recomputed with Python's `statistics.mean` and sample `statistics.stdev`, independently of the report helpers: 672 behavior cells, 288 paired-contrast cells, and 96 generalization-gap cells. Individual seed values also match.
- All 252 derived seed endpoints and intact generalization gaps, and all 108 eight-metric paired vectors, match `seed_metrics.csv` and `paired_seed_deltas.csv`.
- Both SVG figures parse correctly, both PNG files have valid signatures, and their labels are English. Both PNG figures were also visually inspected.

The independent recomputation found no numerical reporting discrepancy. This review is separate from the full design/provenance audit in [SKIP_V5_AUDIT.json](SKIP_V5_AUDIT.json).

## Main endpoint and exact-seed contrasts

Every intact variant achieves 100% own-IID motif accuracy in every seed. The table reports common-OOD ordinary motif accuracy, the predeclared primary endpoint, as mean +/- sample SD across three initialization seeds. A and B remain separate experiments.

| Regime | Experience | Original deep 448D | Matched deep control 512D | Shallow skip 512D | Paired skip minus control |
| --- | --- | --- | --- | --- | --- |
| End-to-end | A | 0.00 +/- 0.00% | 1.05 +/- 1.82% | 2.18 +/- 3.78% | +1.13 +/- 4.95 pp |
| End-to-end | B | 4.21 +/- 7.10% | 0.98 +/- 1.69% | 6.62 +/- 6.34% | +5.64 +/- 6.43 pp |
| Frozen readout | A | 0.00 +/- 0.00% | 0.00 +/- 0.00% | 0.00 +/- 0.00% | 0.00 +/- 0.00 pp |
| Frozen readout | B | 6.92 +/- 6.33% | 7.45 +/- 6.57% | 7.52 +/- 6.61% | +0.08 +/- 0.13 pp |

Paired OOD differences by seed are important because the end-to-end means hide substantial variation:

| Regime | Experience | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- | --- |
| End-to-end | A | +6.546 pp | 0.000 pp | -3.160 pp |
| End-to-end | B | 0.000 pp | +12.641 pp | +4.289 pp |
| Frozen readout | A | 0.000 pp | 0.000 pp | 0.000 pp |
| Frozen readout | B | +0.226 pp | 0.000 pp | 0.000 pp |

The end-to-end intervention produces small, seed-dependent improvements while the mean OOD accuracy remains below the 12.5% eight-output uniform-guessing reference. The frozen intervention has essentially no motif benefit. These results do not establish stable recovery of unseen-combination prediction. They support the narrower statement that appending this particular shallow whole-graph mean, with the original source-only loss and strongly bound training split, is insufficient to close the generalization gap. Three seeds do not support a broad claim of statistical significance or absence of any possible benefit.

## Baseline and parameter provenance

The reuse description in the primary report is correct. All six `end_to_end/deep` checkpoint files match the corresponding completed `hard_v2` checkpoint files byte for byte, including serialization and metadata. Their state dictionaries also match tensor for tensor, and their copied training histories match the original files byte for byte. Their inventory records 1,320 original optimizer updates and zero newly executed updates. The remaining 30 cases each execute 1,320 updates in this run.

The frozen regime resets heads to the same-seed archived random head values while retaining the corresponding pretrained A/B encoder. All 18 frozen encoder initial/final hashes are equal. It is a fresh source-label readout fit, rather than continuation of the inherited trained classifier or a fit with target-combination labels.

| Regime | Variant | Readout width | Total parameters | Trainable parameters |
| --- | --- | --- | --- | --- |
| End-to-end | Deep | 448 | 189,531 | 189,531 |
| End-to-end | Deep control or shallow skip | 512 | 201,819 | 201,819 |
| Frozen readout | Deep | 448 | 189,531 | 87,963 |
| Frozen readout | Deep control or shallow skip | 512 | 201,819 | 100,251 |

The two 512D variants have identical parameter counts. The deep control repeats a final-layer mean already present in the original feature vector, so it controls increased head width and reparameterization without introducing an independent information source. Their shared initial 448D head columns and zero added columns also control the initial function to the recorded numerical tolerance. Comparing the two regimes directly would additionally change encoder history and optimization conditions; their results should remain separate.

## Secondary branch interventions and visual checks

The following analysis uses saved prediction rows only. It reports the percentage of common-OOD motif predictions changed when the shallow branch is altered, averaged over three seeds. It is secondary and does not alter the predeclared accuracy endpoint.

| Regime | Experience | Zero-branch disagreement | Permuted-branch disagreement |
| --- | --- | --- | --- |
| End-to-end | A | 1.129 +/- 0.814% | 0.000 +/- 0.000% |
| End-to-end | B | 0.752 +/- 0.690% | 0.000 +/- 0.000% |
| Frozen readout | A | 0.000 +/- 0.000% | 0.000 +/- 0.000% |
| Frozen readout | B | 0.226 +/- 0.226% | 0.075 +/- 0.130% |

The fitted motif decisions are largely insensitive to this particular branch permutation and only weakly sensitive to branch removal on the saved test identities. This does not establish that the branch has no effect on logits, confidence, or the encoder learned during end-to-end training. Equal accuracies at a low performance floor are especially insufficient to prove nonuse; some prediction changes leave accuracy unchanged. Branch removal and permutation also create features outside the fitted joint distribution, so their effects are functional interventions rather than an additive measure of information loss.

The primary figure faithfully shows 100% own-IID accuracy, low OOD accuracy, individual seed points, sample-SD error bars, and the resulting large generalization gaps. The training figure shows only source training loss and familiar validation motif accuracy. The displayed loss decreases and familiar validation saturates, while no OOD trajectory or selected intermediate checkpoint is shown. Labels and legends are readable; no primary figure change is required.

## Interpretation boundary

The experiment adds no target-fragment membership mask and no target-combination labels. The shallow mean pools the whole molecule, including ring, anchor, target, distractor, and original query attributes. It therefore does not reproduce the earlier oracle target-substituent probe, even though both access shallow node states. Its failure does not contradict the earlier measurement that a boundary-selected shallow representation can separate the eight motifs across positions, and it cannot establish that shallow information has been destroyed.

The supported outcome is a separation: readable local structural information and graph-level labeled adaptation can coexist with failure of this boundary-free, source-only shallow-readout intervention. The current results leave open which pooling rule, supervision, or learning bias could convert that readability into stable zero-target-label task generalization. The primary report preserves this boundary and requires no scientific correction.
