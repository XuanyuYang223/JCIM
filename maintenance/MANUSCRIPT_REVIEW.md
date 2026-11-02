# Working manuscript reader and numerical review

Status: PASS for the reviewed working draft, numerical claim accounting, and scientific scope. This is not a submission-readiness or novelty-priority certification. The review snapshot is identified by the exact fingerprints below. No frozen experiment, source implementation, protocol, test, or audit result was edited in this review.

## Reviewer scope

The reviewer implemented the v7 frozen-panel runner but did not author the manuscript, claim builder, report generator, or containment certificate. The numerical checks reported here used standalone CSV/JSON/statistics/NumPy calculations without importing those implementations. This is an independent reader and arithmetic review, not an independent historical optimizer reconstruction. The separate functional audits have their own explicit numerical qualifications.

## Claims, sources, and numerical reconstruction

The final ledger contains 71 distinct claims: C00-C53 and C100-C116. All 29 source-file hashes match their recorded values. The claim CSV hash and builder hash agree with CLAIMS_CHECK.json; claim identifiers are unique and match that check file. All 501 nested three-seed mean/sample-SD summaries are internally consistent. Internal consistency and hashes alone cannot establish that a selector names the correct scientific population; the population checks below were therefore performed separately.

The main v2/v5 endpoint claims and every v6 whole-graph cell in Table 2 were independently recomputed from their frozen CSV rows. All match. The v7 review independently reconstructs 4,896 ordinary/balanced comparisons spanning every conditional edge, matched-random contrast, and finite-difference factorial effect, plus 11,040 summary scalar comparisons. All match within absolute tolerance 1e-12. Its [separate review](READOUT_V7_REPORT_REVIEW.md) records source-fitting guards and detailed conditional values.

The standardized-ridge scope comparison in main Table 3 is correct: end-to-end-source A moves from 0.0% to 37.4% +/- 19.5%, and B from 8.4% +/- 6.9% to 64.2% +/- 9.6%. The paired B gain is 55.8 +/- 14.4 percentage points, calculated within seed. Both scopes have 100% training and own-familiar accuracy in every trained-encoder seed. Matched random shallow scores, 48.9% +/- 24.1% and 65.1% +/- 25.9%, prevent that recovery from being attributed to a training advantage. Random A/B fits share the same three encoders.

All 36 saved ridge containment pairs were additionally reconstructed independently, without refitting. Common-coordinate scalers, embedded candidate logits, predictions, and coefficient penalties are preserved exactly. Both source penalized objectives agree with the certificate within 1e-10; the full solution has the lower source objective in every pair. This supports a conditional fitted-mapping conclusion and does not identify the semantic content of the selected rule. The manuscript correctly labels this certificate as post-run saved-state analysis.

## Corrected documentation issues

Two issues in the initial committed draft were resolved in the manuscript builder and generated ledger, with separate revision notes:

- C49 now selects the trained model's own source curriculum and preserves the curriculum/model key. Independently reconstructed supported-error fractions are A: 1, 1, 0.9977426636568849; B: 1, 1, 1. The original broad selector had allowed cross-curriculum rows to overwrite own-curriculum values.
- C48 and Supplement S4 correctly name the 1,649 archived feature identities underlying the unchanged 98.7% receptive-reach statistic. Restricting to the 1,377 analysis identities would define a different population.

The ledger also distinguishes original/v5 three-task source annotations from v6 motif-only refits. These are documentation corrections; their revision notes preserve the initial draft hash and do not modify archived scientific records.

## Scientific boundaries checked

The draft separates three supervision protocols: original-source prediction, directed diagnostics using 256 balanced source-position labels, and adaptation with 32 new-combination plus 32 old labels. Directed diagnostic supports may contain combinations omitted from encoder training; they are not described as the original predictor's zero-additional-label generalization. Source-only oracle pooling is explicitly conditional on supplied membership. Selected local states are computed from the complete graph, so pool membership is not conflated with input locality.

Random controls are presented as essential initialization references. Architectural neighborhood consistency is distinguished from empirically demonstrated linear separability. Weak deep probes are not treated as proof of erased information. The positive labeled graph-adaptation result is not generalized into source-only transfer, universal sample efficiency, or a benefit on every readout.

The strong motif-position binding is explicit. Position-only MAP cannot identify all eight motifs and scores only about 36-38% on familiar data; agreement with a source-supported label set is not claimed to identify a pure position-only model. Perfect source accuracy leaves multiple possible rules, so unseen failure is not treated as evidence that structural information is absent.

The v7 MLP factors remain conditional on the historical 512-input initialization. Shallow-only masking removes every initially nonzero input weight; this differs from the fresh 64-input v6 nonlinear head. Raw shallow MLPs substantially underfit, and their approximately 53-55% trained-source accuracy and approximately 51-55% own-familiar accuracy are reported. Their OOD scores are not presented as clean preservation of familiar performance. Standardization changes fitting and the effective raw-coordinate ridge penalty; no additive percentage decomposition of historical gains is claimed.

Frozen task heads share no trainable parameters. Auxiliary losses therefore do not create a direct motif gradient; only global clipping and numerical execution couple updates here. The manuscript correctly avoids attributing the small objective effect to semantic multitask interference or encoder-learning mechanisms. Interactions and descriptive within-seed contrasts are distinguished from significance tests. The three seeds remain the replication unit.

## Numerical execution qualifications

The supplement preserves the original CPU audit as FAIL on one untrained auxiliary distance-head/source-training decision. All motif and trained-task class decisions agree on CPU. The independent CUDA replay exactly matches all 432 saved head arrays and all 1,984,608 decisions. The first runtime-wrapper audit also remains FAIL because BLAS configuration occurred after NumPy initialization. The unchanged wrapper passes with the recorded CUDA head execution and startup BLAS environment of one. The manuscript links both failures, numerical reconciliation, startup evidence, and the full matched-runtime PASS without altering criteria or inferring an independently observed BLAS worker count.

These qualifications are relevant to exact execution reproducibility; they do not change the reported motif results. The prose does not silently turn the original failed audit into a pass.

## Figures, README, and practical limits

The new design figure was viewed directly. Motif roots, dummy connection markers, the A/B/T combination matrix, canonical-identity distinction, and three supervision budgets are readable and consistent with the described grid. Both v7 figures were also viewed. The heatmap reports mean common-OOD ordinary accuracy; seed SDs and individual paired effects remain available separately. Training/IID guards must accompany scientific interpretation of the heatmap.

README's six-row standardized-ridge table matches independent recalculation, including the original-trained B paired SD of 24.6 pp. Its trained-source guards, shared random-seed accounting, post-run containment description, and two preserved audit failures are accurate. The PowerShell audit command sets OPENBLAS_NUM_THREADS and MKL_NUM_THREADS before launching Python and uses a fresh output path. Every local Markdown target in the manuscript documents and README exists at this snapshot.

The remaining limitations are appropriately explicit: this is an eight-known-motif controlled structural task with query markers, shared generation factors across molecule-isolated splits, three encoder seeds, and oracle localization in designated diagnostics. There is no experimental property, stereochemical, three-dimensional, prospective design, or unseen-motif-topology validation. A broader chemistry literature review, final figure layout, author details, and submission declarations remain future publication work; they are not asserted complete in this draft.

## Exact review fingerprints

- `README.md`: `5b88d68be0125a10c2bbf6f71c2285feb53933878820fca053e8fe50df97a0b3`
- `manuscript/MANUSCRIPT.md`: `1950438ab382c0876c51aeb18487923abbcd8d29bdde276a167deb4a8180e0bb`
- `manuscript/SUPPLEMENT.md`: `068f2a0ec656f115cd08ba373821ee0ba48b721d8456c59fa64a3afa5d797ff9`
- `manuscript/CLAIMS.csv`: `515f9e24fee21d5b2cf98d34ab47f5bc84b8159520fb75fbb03f7c818f7248d2`
- `manuscript/CLAIMS_CHECK.json`: `35004a04361c454dc5db5b69135d7e0e7493c0c9dabdee92befdf289d3c10e5a`
- `manuscript/build_claims.py`: `dc786a2f04d107a2c574d71754bbac7ac04879788bf861705a6bbac03f9e73e7`
- `manuscript/REFERENCES.md`: `6e2a2934439971dee2d1db586b7f0ac17385b22c1ef0834a5063aca3156c213a`
- `manuscript/REVISION_NOTES.md`: `c32109f75f338350f43e50093b85f9ba340bd735872062f84a009b2eeb91bda2`
- `manuscript/figures/controlled_design.png`: `74d713863ac79492a9a830c19a23d2051232653ca3e6d0af215bf0d534b33bf3`
- `manuscript/figures/CONTROLLED_DESIGN_METADATA.json`: `5a01a849f4adaecf98e8045da56dae704d25e429491940d58cc9c0dfd287555d`
- `experiments/readout_v7/MANIFEST.json`: `40335a4cd857373fad6ff17ed26a8c3c3dba8f34b30ef2943827e024989faa94`
- `maintenance/READOUT_V7_REPORT_REVIEW.md`: `a076a421ea747838383e2f124234e272c96cedc3a0e3a50e1207d2c3f68d65de`
- `maintenance/READOUT_V7_CONTAINMENT.json`: `764da328ee678abf2840f2108a2ba12615d89f2a3c7a1d4fc2240c798c51c1c2`
- `maintenance/READOUT_V7_AUDIT.json`: `36a19a4f6a1a062d26fdc935edf8014647215523b6ebbdef84623e0421488388`
- `maintenance/READOUT_V7_RUNTIME_AUDIT.json`: `71eb9c619c49bdf9b06fc4ff2cb9aea5eccdcd2ede520aef0d47c86aa4fb629c`
- `maintenance/READOUT_V7_RUNTIME_SINGLE_THREAD_AUDIT.json`: `1d2e125429b4ed3f90078807a9f98df78ee003d083ebcc78ce27ec245eaab07f`
- `maintenance/READOUT_V7_NUMERICAL_CHECK.json`: `91f3b0a190aa57a050fb146f334feccceb714a436132a8d1287e6beb4e2116c9`
- `maintenance/READOUT_V7_NUMERICAL_NOTE.md`: `ef05288134dc96e83641f2695907e9788d3b8212b535f8bba03186e66af8a5ba`
- `maintenance/READOUT_V7_RUNTIME_ENVIRONMENT.json`: `25e46bfb7ac0dffdb50856da024cb94ed3cbedee5483b219ea90b07ae9ba19eb`
- `manuscript/build_design_figure.py`: `986c8cefcadd35f7a6f69bc3297b400b8b8c56e056015c5761ecd3fc0ea2387d`
- `manuscript/figures/controlled_design.svg`: `8c175a0eeaf35f9d9908af349ab17fe10a18e1e3abb89cb41db33ee9c464ca0f`
