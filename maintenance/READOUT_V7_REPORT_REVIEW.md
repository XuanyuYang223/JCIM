# Readout v7 numerical and interpretation review

Status: PASS for the descriptive report and derived summaries reviewed here. This is a separate post-run review; no experiment file, fit, protocol, or scientific record was changed. It does not replace the independent functional audit.

## Scope and independence

The reviewer implemented the frozen-panel runner but did not author the report generator or manuscript. Numerical summaries were reconstructed with standalone Python CSV, arithmetic, and statistics operations, without importing the experiment or report modules. This is an independent calculation and reader review, not an independent implementation audit or a claim to reconstruct every historical optimizer operation.

The panel has 15 unique frozen encoders and 18 encoder/source-experience cases. Random A/B fits share three random encoders. All 216 classifier fits use the original 1,386 source identities for their experience. The 144 MLP fits preserve a 512-input, three-head architecture; 72 ridge fits are a separate decoder panel. No target-combination labels or target membership are supplied. Ordinary and balanced motif accuracy were checked separately.

## Numerical checks

- Reconstructed all 1,152 conditional edges, 576 matched-random contrasts, and 720 finite-difference factorial effects directly from 864 family endpoint rows. Across both accuracy definitions, 4,896 derived metric comparisons had zero mismatches at absolute tolerance 1e-12.
- Independently reconstructed all 1,104 summary panels: 11,040 comparisons of individual seed values, seed means, and sample SDs had zero mismatches at the same tolerance. A/B remain separate; no molecules, classifiers, or source-support fits are treated as extra encoder seeds.
- The 24 recorded continuity anchors comprise 18 exact v6 standardized-shallow ridge comparisons and six exact v5 raw/full/joint frozen-head comparisons. These are continuity checks on reused information, not independent experimental replications. The separate audit verifies their underlying states and predictions.
- Viewed both PNG figures. Panel names, factor levels, common-OOD ordinary accuracy, and percentage-point effect units are readable. The heatmap shows means; sample SDs and individual-seed effects are available in the report and effect figure. The figure alone does not show training/IID guards, so its cells must not be described as uniformly successful source fitting.

## Conditional ridge comparison with source guards

Entries below are ordinary percent accuracy or paired percentage-point differences, mean +/- sample SD over the three seeds. The normalization is train_z. The paired effect is shallow minus full within seed. Shallow-source guards are shown explicitly; full-scope ridge training and own-IID accuracy are 100% for every listed case and seed.

| Encoder source | Experience | Full OOD | Shallow OOD | Paired scope effect (pp) | Shallow train | Shallow own IID |
| --- | --- | --- | --- | --- | --- | --- |
| end_to_end | A | 0.0 +/- 0.0 | 37.4 +/- 19.5 | 37.4 +/- 19.5 | 100.0 +/- 0.0 | 100.0 +/- 0.0 |
| end_to_end | B | 8.4 +/- 6.9 | 64.2 +/- 9.6 | 55.8 +/- 14.4 | 100.0 +/- 0.0 | 100.0 +/- 0.0 |
| frozen_readout | A | 0.0 +/- 0.0 | 27.4 +/- 8.5 | 27.4 +/- 8.5 | 100.0 +/- 0.0 | 100.0 +/- 0.0 |
| frozen_readout | B | 7.4 +/- 4.4 | 58.6 +/- 20.3 | 51.2 +/- 24.6 | 100.0 +/- 0.0 | 100.0 +/- 0.0 |
| random | A | 60.0 +/- 10.7 | 48.9 +/- 24.1 | -11.1 +/- 25.8 | 100.0 +/- 0.0 | 99.9 +/- 0.1 |
| random | B | 53.6 +/- 16.9 | 65.1 +/- 25.9 | 11.4 +/- 28.4 | 99.9 +/- 0.1 | 99.9 +/- 0.1 |

For both trained encoder sources, standardized shallow ridge has 100% training and own-IID accuracy in every seed. Its improvement over full ridge therefore does not require sacrificing those familiar guards. For the end-to-end v5 encoders, this scope contrast is +37.4 +/- 19.5 pp in A and +55.8 +/- 14.4 pp in B. These are conditional effects at fixed ridge fitting, normalization, source identities, and encoder state. They do not establish that shallow pooling succeeds universally: random-encoder A has a negative average scope effect and large variability.

Matched random controls also constrain the training claim. Under standardized shallow ridge, end-to-end trained minus random is -11.5 +/- 40.5 pp for A and -0.9 +/- 31.4 pp for B. The corresponding full-ridge differences are -60.0 +/- 10.7 pp and -45.2 +/- 17.5 pp. This source-only panel does not show a general advantage from structural-task training. The earlier labeled graph-adaptation advantage uses another feature scope and additional new-combination labels and remains a distinct result.

## MLP underfitting and initialization

Raw shallow-only joint MLP fits have the following guards; the family-only fits exhibit the same material underfitting. These endpoint comparisons cannot be presented as clean improvements that preserve familiar performance.

| Encoder source | Experience | Train accuracy | Own-IID accuracy |
| --- | --- | --- | --- |
| end_to_end | A | 52.6 +/- 10.0 | 50.6 +/- 10.5 |
| end_to_end | B | 55.3 +/- 12.3 | 54.6 +/- 15.4 |
| frozen_readout | A | 61.6 +/- 8.7 | 61.0 +/- 10.3 |
| frozen_readout | B | 52.9 +/- 9.5 | 49.8 +/- 11.0 |
| random | A | 19.5 +/- 3.8 | 15.2 +/- 4.7 |
| random | B | 18.7 +/- 3.6 | 16.2 +/- 4.3 |

The shallow-only MLP zeros the first 448 input coordinates. Its appended 64 input-column weights start at zero, so every nonzero initial input weight is masked in this arm. Initial predictions are consequently independent of the shallow input. This is the intentional v5-matched initializer and differs from v6's fresh 64-input MLP. The MLP scope contrast therefore remains conditional on that initializer and its fixed optimization budget. Standardization substantially improves shallow-source fitting, but does not universally improve OOD accuracy. For example, end-to-end B standardized-shallow joint MLP reaches 26.7 +/- 6.6% OOD versus 3.4 +/- 5.9% for full input, while its train and own-IID guards are 99.98 +/- 0.04% and 99.79 +/- 0.37%; exact perfect fitting must not be asserted for this cell.

## Objective, interaction, and scientific scope

With a frozen encoder, task heads have no shared trainable parameters. Position and distance losses do not contribute an unclipped family-head gradient. Their update coupling is the global clipping coefficient, with numerical execution as an additional consideration. The inventory records clipping in 20 of 144 MLP cells, totaling 264 clipped updates; 263 occur in raw/full cells and one in a raw/shallow cell. Both standardized scopes have zero recorded clipped updates. Of the 72 common-OOD family-minus-joint conditional edges, only four change ordinary accuracy; the maximum absolute change is 1.81 pp. These observations do not support a semantic multitask-interference mechanism.

Averaged main effects can conceal conditional reversals and underfitting. The preserved edges and finite-difference interactions, rather than a division of total improvement into additive percentages, are the appropriate summaries. Standardization also changes the effective ridge penalty in raw feature units. The v7 panel controls conditional fitting choices; it does not uniquely decompose the historical v5-to-v6 change or isolate a general optimization mechanism. Oracle localization, diagnostic labels, labeled adaptation, and source-only prediction remain separate protocols. The findings remain limited to the controlled eight-motif structural task.

## Reviewed record fingerprints

- `MANIFEST.json`: `40335a4cd857373fad6ff17ed26a8c3c3dba8f34b30ef2943827e024989faa94`
- `REPORT.md`: `1b5a8e542b1fc1b9aeb6ac96c7b4004f2a7a37b29c43f818484be0031034f46e`
- `panel\source_metrics.csv`: `1551c0cf04f0000ef14421730eb70099a444fd96c7dd1dbe1e26cef907e79d8a`
- `report_summary.json`: `a4174c8e525bd5055f74685da22e8ee2b5a5c89651efad538e0ee19372c297a8`
