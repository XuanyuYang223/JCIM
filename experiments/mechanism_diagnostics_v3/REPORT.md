# Mechanism diagnostics v3

This record adds diagnostics to the frozen hard-v2 experiment. It does not retrain its full-input models.

## Position-only baseline

The baseline fits empirical motif counts conditional on the true queried position in each actual training curriculum. It predicts the largest count and breaks ties by smallest label. A position supports multiple motifs, so position alone cannot identify every IID motif. With three positions and eight classes, its balanced accuracy cannot exceed 37.5%. This is distinct from the eight-class uniform chance of 12.5%.

| Curriculum | Split | n | MAP accuracy | MAP balanced accuracy | True label in training-supported set |
|---|---|---:|---:|---:|---:|
| a | iid_a | 462 | 35.9% | 37.5% | 100.0% |
| a | iid_b | 472 | 0.0% | 0.0% | 0.0% |
| a | ood_common | 443 | 0.0% | 0.0% | 0.0% |
| b | iid_a | 462 | 0.0% | 0.0% | 0.0% |
| b | iid_b | 472 | 37.9% | 37.5% | 100.0% |
| b | ood_common | 443 | 0.0% | 0.0% | 0.0% |

### Saved model predictions against each position baseline

The supported set contains every motif seen at the given position in that curriculum. High compatibility, especially among errors, is consistent with dependence on the learned binding. It does not prove that a pure position-only predictor explains the model: exact MAP agreement is shown separately.

| Curriculum | Model | Split | Model accuracy | MAP agreement | Prediction in supported set | Errors in supported set |
|---|---|---|---:|---:|---:|---:|
| a | a_seed0 | iid_a | 100.0% | 35.9% | 100.0% | No errors |
| a | a_seed1 | iid_a | 100.0% | 35.9% | 100.0% | No errors |
| a | a_seed2 | iid_a | 100.0% | 35.9% | 100.0% | No errors |
| a | b_seed0 | iid_a | 0.0% | 0.0% | 0.0% | 0.0% |
| a | b_seed1 | iid_a | 6.3% | 0.0% | 6.3% | 0.0% |
| a | b_seed2 | iid_a | 0.0% | 0.0% | 0.0% | 0.0% |
| a | random_seed0 | iid_a | 13.2% | 0.0% | 39.4% | 30.2% |
| a | random_seed1 | iid_a | 11.5% | 22.1% | 22.1% | 12.0% |
| a | random_seed2 | iid_a | 13.0% | 0.0% | 38.5% | 29.4% |
| a | a_seed0 | iid_b | 0.0% | 26.3% | 99.8% | 99.8% |
| a | a_seed1 | iid_b | 0.0% | 25.2% | 100.0% | 100.0% |
| a | a_seed2 | iid_b | 1.5% | 29.4% | 98.5% | 100.0% |
| a | b_seed0 | iid_b | 100.0% | 0.0% | 0.0% | No errors |
| a | b_seed1 | iid_b | 100.0% | 0.0% | 0.0% | No errors |
| a | b_seed2 | iid_b | 100.0% | 0.0% | 0.0% | No errors |
| a | random_seed0 | iid_b | 12.3% | 0.0% | 24.8% | 28.3% |
| a | random_seed1 | iid_b | 12.7% | 38.3% | 38.3% | 43.9% |
| a | random_seed2 | iid_b | 12.3% | 0.0% | 36.9% | 42.0% |
| a | a_seed0 | ood_common | 0.0% | 51.7% | 100.0% | 100.0% |
| a | a_seed1 | ood_common | 0.0% | 39.1% | 100.0% | 100.0% |
| a | a_seed2 | ood_common | 0.0% | 38.6% | 99.8% | 99.8% |
| a | b_seed0 | ood_common | 0.0% | 0.0% | 0.0% | 0.0% |
| a | b_seed1 | ood_common | 12.4% | 0.0% | 0.0% | 0.0% |
| a | b_seed2 | ood_common | 0.2% | 0.0% | 0.0% | 0.0% |
| a | random_seed0 | ood_common | 14.2% | 0.0% | 37.7% | 43.9% |
| a | random_seed1 | ood_common | 11.1% | 38.8% | 38.8% | 43.7% |
| a | random_seed2 | ood_common | 10.6% | 0.0% | 23.5% | 26.3% |
| b | a_seed0 | iid_a | 100.0% | 0.0% | 0.0% | No errors |
| b | a_seed1 | iid_a | 100.0% | 0.0% | 0.0% | No errors |
| b | a_seed2 | iid_a | 100.0% | 0.0% | 0.0% | No errors |
| b | b_seed0 | iid_a | 0.0% | 38.1% | 100.0% | 100.0% |
| b | b_seed1 | iid_a | 6.3% | 31.8% | 87.7% | 93.5% |
| b | b_seed2 | iid_a | 0.0% | 44.2% | 100.0% | 100.0% |
| b | random_seed0 | iid_a | 13.2% | 0.0% | 38.5% | 44.4% |
| b | random_seed1 | iid_a | 11.5% | 39.4% | 39.4% | 44.5% |
| b | random_seed2 | iid_a | 13.0% | 0.0% | 22.1% | 25.4% |
| b | a_seed0 | iid_b | 0.0% | 0.0% | 0.0% | 0.0% |
| b | a_seed1 | iid_b | 0.0% | 0.0% | 0.0% | 0.0% |
| b | a_seed2 | iid_b | 1.5% | 0.0% | 1.5% | 0.0% |
| b | b_seed0 | iid_b | 100.0% | 37.9% | 100.0% | No errors |
| b | b_seed1 | iid_b | 100.0% | 37.9% | 100.0% | No errors |
| b | b_seed2 | iid_b | 100.0% | 37.9% | 100.0% | No errors |
| b | random_seed0 | iid_b | 12.3% | 0.0% | 36.9% | 28.0% |
| b | random_seed1 | iid_b | 12.7% | 24.8% | 24.8% | 13.8% |
| b | random_seed2 | iid_b | 12.3% | 0.0% | 38.3% | 29.7% |
| b | a_seed0 | ood_common | 0.0% | 0.0% | 0.0% | 0.0% |
| b | a_seed1 | ood_common | 0.0% | 0.0% | 0.0% | 0.0% |
| b | a_seed2 | ood_common | 0.0% | 0.0% | 0.2% | 0.2% |
| b | b_seed0 | ood_common | 0.0% | 23.0% | 100.0% | 100.0% |
| b | b_seed1 | ood_common | 12.4% | 23.0% | 87.6% | 100.0% |
| b | b_seed2 | ood_common | 0.2% | 35.4% | 99.8% | 100.0% |
| b | random_seed0 | ood_common | 14.2% | 0.0% | 23.5% | 27.4% |
| b | random_seed1 | ood_common | 11.1% | 37.7% | 37.7% | 42.4% |
| b | random_seed2 | ood_common | 10.6% | 0.0% | 38.8% | 43.4% |

## Oracle local-input positive control

Each input is the complete known nine-atom target induced subgraph, at root radius at most six. The remote ring, anchor, position, and path-endpoint query markers are removed. Only the target root marker (channel 15) remains. Ordinary atom features are copied from the original molecule, including boundary degree and hydrogen count; no context-dependent capping operation is introduced.

The control uses HardMolecularGNN(64, 8), its existing graph-pooling motif head, and only motif cross entropy. The learning rate is 0.0015, batch size 128, and duration 120 epochs. The final epoch is fixed before evaluation; OOD results select no checkpoints. Position and distance heads receive no training. All actual curriculum-A molecular training IDs are used. Curriculum A and B have identical empirical local-input distributions, so training A only per seed is sufficient for this particular control.

Molecular train/test identities remain disjoint, but all eight local input graph types intentionally occur on both sides after context removal. This measures whether a usable local solution exists when an oracle supplies the target. It does not measure generalization to novel motif types, automatic localization, or reuse by the original full-input encoder.

| Seed | Test split | n | Motif accuracy | Balanced accuracy |
|---:|---|---:|---:|---:|
| 0 | iid_a | 462 | 100.0% | 100.0% |
| 0 | iid_b | 472 | 100.0% | 100.0% |
| 0 | ood_common | 443 | 100.0% | 100.0% |
| 1 | iid_a | 462 | 100.0% | 100.0% |
| 1 | iid_b | 472 | 100.0% | 100.0% |
| 1 | ood_common | 443 | 100.0% | 100.0% |
| 2 | iid_a | 462 | 100.0% | 100.0% |
| 2 | iid_b | 472 | 100.0% | 100.0% |
| 2 | ood_common | 443 | 100.0% | 100.0% |

Per-position results and molecule-level predictions are retained in local_behavior.csv and local_predictions.csv. local_input_audit.json records the repeated local types and distribution match.

## Frozen cross-position probe controls

The following are copied without numerical changes from hard-v2, including every model seed and every layer. The uniform eight-class chance is 12.5%; a random encoder's observed accuracy is an empirical control, not theoretical chance. Shuffled labels use the archived deterministic permutation seed 8107. The archived shuffled metric is ordinary accuracy; it is not relabeled as balanced accuracy. A low linear probe score constrains this readout and sample protocol, not every possible extraction of local information.

| Model | Seed | Layer | Fit n | Test n | Probe accuracy | Balanced accuracy | Shuffled-label accuracy | Uniform chance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | -1 | -1 | 456 | 455 | 13.0% | 12.5% | 13.0% | 12.5% |
| random_seed0 | 0 | 0 | 456 | 455 | 13.0% | 12.5% | 13.0% | 12.5% |
| random_seed0 | 0 | 1 | 456 | 455 | 25.7% | 25.0% | 13.0% | 12.5% |
| random_seed0 | 0 | 2 | 456 | 455 | 38.5% | 37.9% | 0.0% | 12.5% |
| random_seed0 | 0 | 3 | 456 | 455 | 36.7% | 37.5% | 4.0% | 12.5% |
| random_seed0 | 0 | 4 | 456 | 455 | 40.9% | 40.7% | 12.1% | 12.5% |
| random_seed0 | 0 | 5 | 456 | 455 | 38.5% | 37.9% | 16.5% | 12.5% |
| random_seed0 | 0 | 6 | 456 | 455 | 35.4% | 37.5% | 13.2% | 12.5% |
| random_seed0 | 0 | 7 | 456 | 455 | 39.6% | 38.6% | 8.1% | 12.5% |
| random_seed0 | 0 | 8 | 456 | 455 | 40.7% | 40.4% | 13.0% | 12.5% |
| a_seed0 | 0 | 0 | 456 | 455 | 13.0% | 12.5% | 13.0% | 12.5% |
| a_seed0 | 0 | 1 | 456 | 455 | 25.7% | 25.0% | 13.0% | 12.5% |
| a_seed0 | 0 | 2 | 456 | 455 | 37.8% | 37.5% | 12.7% | 12.5% |
| a_seed0 | 0 | 3 | 456 | 455 | 23.7% | 25.0% | 12.1% | 12.5% |
| a_seed0 | 0 | 4 | 456 | 455 | 36.7% | 37.5% | 11.9% | 12.5% |
| a_seed0 | 0 | 5 | 456 | 455 | 36.5% | 37.5% | 0.0% | 12.5% |
| a_seed0 | 0 | 6 | 456 | 455 | 25.5% | 26.7% | 13.8% | 12.5% |
| a_seed0 | 0 | 7 | 456 | 455 | 25.1% | 24.4% | 12.7% | 12.5% |
| a_seed0 | 0 | 8 | 456 | 455 | 13.0% | 12.5% | 0.0% | 12.5% |
| b_seed0 | 0 | 0 | 456 | 455 | 13.0% | 12.5% | 13.0% | 12.5% |
| b_seed0 | 0 | 1 | 456 | 455 | 25.7% | 25.0% | 13.0% | 12.5% |
| b_seed0 | 0 | 2 | 456 | 455 | 23.7% | 25.0% | 0.0% | 12.5% |
| b_seed0 | 0 | 3 | 456 | 455 | 36.7% | 37.5% | 2.2% | 12.5% |
| b_seed0 | 0 | 4 | 456 | 455 | 13.8% | 12.5% | 11.9% | 12.5% |
| b_seed0 | 0 | 5 | 456 | 455 | 15.8% | 14.2% | 25.7% | 12.5% |
| b_seed0 | 0 | 6 | 456 | 455 | 16.3% | 15.7% | 24.6% | 12.5% |
| b_seed0 | 0 | 7 | 456 | 455 | 11.6% | 12.5% | 11.6% | 12.5% |
| b_seed0 | 0 | 8 | 456 | 455 | 10.8% | 12.5% | 19.8% | 12.5% |
| random_seed1 | 1 | 0 | 456 | 455 | 13.0% | 12.5% | 13.0% | 12.5% |
| random_seed1 | 1 | 1 | 456 | 455 | 25.7% | 25.0% | 13.0% | 12.5% |
| random_seed1 | 1 | 2 | 456 | 455 | 25.7% | 25.0% | 10.8% | 12.5% |
| random_seed1 | 1 | 3 | 456 | 455 | 25.9% | 25.0% | 10.3% | 12.5% |
| random_seed1 | 1 | 4 | 456 | 455 | 40.9% | 37.5% | 2.4% | 12.5% |
| random_seed1 | 1 | 5 | 456 | 455 | 38.7% | 37.5% | 26.8% | 12.5% |
| random_seed1 | 1 | 6 | 456 | 455 | 25.7% | 25.0% | 12.7% | 12.5% |
| random_seed1 | 1 | 7 | 456 | 455 | 10.8% | 12.5% | 13.0% | 12.5% |
| random_seed1 | 1 | 8 | 456 | 455 | 34.9% | 32.8% | 19.8% | 12.5% |
| a_seed1 | 1 | 0 | 456 | 455 | 13.0% | 12.5% | 13.0% | 12.5% |
| a_seed1 | 1 | 1 | 456 | 455 | 25.7% | 25.0% | 13.0% | 12.5% |
| a_seed1 | 1 | 2 | 456 | 455 | 14.1% | 12.5% | 10.8% | 12.5% |
| a_seed1 | 1 | 3 | 456 | 455 | 12.1% | 12.5% | 14.1% | 12.5% |
| a_seed1 | 1 | 4 | 456 | 455 | 34.7% | 33.3% | 12.7% | 12.5% |
| a_seed1 | 1 | 5 | 456 | 455 | 10.5% | 12.2% | 13.8% | 12.5% |
| a_seed1 | 1 | 6 | 456 | 455 | 11.9% | 12.5% | 13.0% | 12.5% |
| a_seed1 | 1 | 7 | 456 | 455 | 11.6% | 12.5% | 14.1% | 12.5% |
| a_seed1 | 1 | 8 | 456 | 455 | 33.8% | 37.0% | 14.1% | 12.5% |
| b_seed1 | 1 | 0 | 456 | 455 | 13.0% | 12.5% | 13.0% | 12.5% |
| b_seed1 | 1 | 1 | 456 | 455 | 25.7% | 25.0% | 13.0% | 12.5% |
| b_seed1 | 1 | 2 | 456 | 455 | 12.7% | 12.5% | 12.7% | 12.5% |
| b_seed1 | 1 | 3 | 456 | 455 | 24.6% | 25.0% | 25.7% | 12.5% |
| b_seed1 | 1 | 4 | 456 | 455 | 13.4% | 13.9% | 1.5% | 12.5% |
| b_seed1 | 1 | 5 | 456 | 455 | 12.7% | 12.5% | 12.1% | 12.5% |
| b_seed1 | 1 | 6 | 456 | 455 | 13.0% | 12.5% | 13.0% | 12.5% |
| b_seed1 | 1 | 7 | 456 | 455 | 24.6% | 25.0% | 36.5% | 12.5% |
| b_seed1 | 1 | 8 | 456 | 455 | 14.1% | 12.5% | 16.7% | 12.5% |
| random_seed2 | 2 | 0 | 456 | 455 | 13.0% | 12.5% | 13.0% | 12.5% |
| random_seed2 | 2 | 1 | 456 | 455 | 25.7% | 25.0% | 13.0% | 12.5% |
| random_seed2 | 2 | 2 | 456 | 455 | 36.7% | 37.5% | 0.0% | 12.5% |
| random_seed2 | 2 | 3 | 456 | 455 | 36.9% | 36.6% | 15.6% | 12.5% |
| random_seed2 | 2 | 4 | 456 | 455 | 25.9% | 25.5% | 21.8% | 12.5% |
| random_seed2 | 2 | 5 | 456 | 455 | 11.0% | 12.7% | 0.0% | 12.5% |
| random_seed2 | 2 | 6 | 456 | 455 | 23.7% | 25.0% | 13.0% | 12.5% |
| random_seed2 | 2 | 7 | 456 | 455 | 20.9% | 19.7% | 15.4% | 12.5% |
| random_seed2 | 2 | 8 | 456 | 455 | 35.8% | 38.2% | 5.7% | 12.5% |
| a_seed2 | 2 | 0 | 456 | 455 | 13.0% | 12.5% | 13.0% | 12.5% |
| a_seed2 | 2 | 1 | 456 | 455 | 25.7% | 25.0% | 13.0% | 12.5% |
| a_seed2 | 2 | 2 | 456 | 455 | 24.6% | 25.0% | 10.8% | 12.5% |
| a_seed2 | 2 | 3 | 456 | 455 | 31.4% | 31.4% | 14.1% | 12.5% |
| a_seed2 | 2 | 4 | 456 | 455 | 36.5% | 37.5% | 24.6% | 12.5% |
| a_seed2 | 2 | 5 | 456 | 455 | 12.1% | 12.7% | 0.0% | 12.5% |
| a_seed2 | 2 | 6 | 456 | 455 | 12.7% | 12.5% | 16.0% | 12.5% |
| a_seed2 | 2 | 7 | 456 | 455 | 23.5% | 25.0% | 23.5% | 12.5% |
| a_seed2 | 2 | 8 | 456 | 455 | 14.7% | 13.3% | 13.8% | 12.5% |
| b_seed2 | 2 | 0 | 456 | 455 | 13.0% | 12.5% | 13.0% | 12.5% |
| b_seed2 | 2 | 1 | 456 | 455 | 25.7% | 25.0% | 13.0% | 12.5% |
| b_seed2 | 2 | 2 | 456 | 455 | 24.6% | 25.0% | 10.8% | 12.5% |
| b_seed2 | 2 | 3 | 456 | 455 | 25.7% | 25.0% | 21.5% | 12.5% |
| b_seed2 | 2 | 4 | 456 | 455 | 14.1% | 12.5% | 4.0% | 12.5% |
| b_seed2 | 2 | 5 | 456 | 455 | 13.0% | 12.5% | 21.8% | 12.5% |
| b_seed2 | 2 | 6 | 456 | 455 | 24.6% | 25.0% | 16.7% | 12.5% |
| b_seed2 | 2 | 7 | 456 | 455 | 0.0% | 0.0% | 23.1% | 12.5% |
| b_seed2 | 2 | 8 | 456 | 455 | 40.9% | 37.5% | 20.0% | 12.5% |

## Interpretation and archival scope

Successful local controls would establish task solvability with supplied local membership. The position baseline and supported-set error analysis can diagnose dependence on training associations. Neither alone distinguishes information absent from the original encoder from information obscured by its readout; the separately controlled frozen-encoder and replay experiment addresses that question.

The source hard-v2 manifest and all its listed file hashes are verified before and after this run. The new record contains its own fixed configuration, source hashes, checkpoints, training histories, local input arrays, per-seed results, copied probe controls, and an output manifest.

Local graph types in curriculum A: 8. Maximum local root radius: 6.
