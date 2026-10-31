# Readout, replay, and correlation interventions

This protocol fixes the next experiments before their primary execution. The completed `pilot_v1` and `hard_v2` archives, model weights, numerical results, and manifests remain unchanged. Small smoke runs test orchestration only; their reduced update budgets are not primary scientific evidence.

## Questions and sequence

First establish whether the task has a local solution and what a position-only predictor can explain. Then test whether the inherited encoder supports new combinations through head adaptation, whether replay preserves old behavior, and how crossed encoder/head states affect the predictions. Finally test whether a small amount of less correlated training data improves combinations that receive no direct counterexample labels.

The hypotheses are motivated by [Kirichenko, Izmailov, and Wilson (ICLR 2023)](https://arxiv.org/abs/2204.02937), which found useful features despite spurious prediction rules in their settings, and [Qiu, Kuang, and Goel (ICML 2024)](https://proceedings.mlr.press/v235/qiu24e.html), which controlled feature complexity and correlation strength. These molecular interventions are not direct replications of either study, and the papers do not guarantee the present outcomes.

## Two input and shortcut diagnostics

The empirical position-only MAP classifier fits counts from each original training curriculum and predicts one motif per position, with deterministic ties. Eight motif classes share three positions, so each position supports multiple motif labels. Exact MAP agreement with model errors and membership in the position-supported label set are distinct measurements; supported-label compatibility alone does not establish a pure position-only predictor.

The local-input positive control receives the entire nine-atom target subgraph with only its attachment-root marker. Ring position, distant reference group, distractor, and the other query markers are removed. The control intentionally supplies the correct subgraph as an oracle input. The same eight motif types occur during training and evaluation across contexts, so success establishes existence of a local task solution, not generalization to novel motif types or evidence that the original full-molecule encoder learned that solution. Root radius six is within the eight message layers. Three fixed seeds, 120 epochs, and final-epoch checkpoints are used.

Probe reporting retains every archived seed and layer, source-label shuffles, and the eight-output uniform reference of 12.5% balanced accuracy. The 37.1% random-encoder result is an architectural baseline, not a theoretical chance level. A negative linear readout result cannot establish absence of information from all layers, nodes, or nonlinear decoders.

## Primary frozen-encoder by replay matrix

The primary comparison uses the exact archived `repeat0_shots32` new support identities from `hard_v2`. These identities are shared by every method and every initialization seed, and were excluded from both pretraining experiences. A fixed 32-old-identity replay sample is drawn from each experience's own original training set using seed 7000. Old samples differ between A and B because their original curricula differ, but are identical across the replay arms and model seeds within each experience.

| Arm | Initialization | Updated components | Labeled data per update |
| --- | --- | --- | --- |
| frozen_new | Original pretrained state | All three inherited MLP task heads | 32 new |
| full_new | Original pretrained state | Entire encoder and inherited heads | 32 new |
| frozen_replay | Original pretrained state | All three inherited MLP task heads | 32 new + 32 old |
| full_replay | Original pretrained state | Entire encoder and inherited heads | 32 new + 32 old |
| scratch_replay | Exact corresponding archived random state | Entire encoder and heads | 32 new + 32 old |

The encoder comprises the input embedding and every message layer. Freezing is checked by SHA-256 over names, tensor metadata, and raw parameter bytes; drift magnitudes are recorded separately. All arms use 80 AdamW updates at learning rate 0.0005 and weight decay 0.0001. Every update uses the entire support set, with equal weighting between new and old groups when replay is present. Thus optimizer steps match, while replay intentionally has twice as many labeled identities and example evaluations per update. This is not an equal-label-budget comparison.

The three original model seeds are used. Support repeats 1 and 2 reuse the other archived new samples and old-selection seeds 7001 and 7002 as supplemental robustness checks. Primary fixed-support results and supplemental repeat-averaged results are reported separately. Repetitions are averaged within each encoder seed and never counted as extra independent initialization seeds.

Evaluate both original IID splits and the common OOD test at steps 0, 1, 5, 10, 20, 40, and 80. The primary old guard is A's `iid_a` or B's `iid_b`. Record all three task accuracies and all-three-correct joint accuracy. Plot new-versus-old motif accuracy through the entire adaptation trajectory. No test checkpoint, hyperparameter, layer, or regularization selection is performed.

## Functional forgetting diagnostics

At the final update, evaluate the four component combinations E0/H0, E0/H1, E1/H0, and E1/H1, where E is the encoder, H is all three task heads, 0 is original, and 1 is adapted. The pretrained arms are included; scratch runs have no pretrained semantic reference for this swap. Failed swaps can reflect coordinate changes or encoder-head coadaptation; they do not establish that information was destroyed.

A fresh standardized ridge motif readout with fixed alpha 1 also fits the original frozen local or graph representation on the same new-only or replay support. This is an inexpensive optimization control when the inherited MLP fails; it is not a reset-last-linear implementation of DFR. Feature normalization and fitting use support features only. Feature drift CKA uses the same 128 sorted, unlabeled test identities across checkpoints. High CKA is not a functional criterion.

## Correlation-strength experiment

This is a separate fresh-pretraining experiment using only original `hard_v2` inputs and states. It never starts from the above adapted models. Training size, eight motif priors, architecture, random initialization, optimizer, 120 epochs, and final-checkpoint policy remain fixed.

For each curriculum, replace 0%, 1%, 5%, or 20% of the 1,386 original training identities with counterexamples drawn only from original candidate-training identities in common-OOD cell 2. The allowed intervention motifs are even IDs 0, 2, 4, and 6. Protected cell-2 combinations for odd IDs 1, 3, 5, and 7 are never supplied at any ratio. They test transfer to other motif types rather than direct learning of an exposed combination. Original validation/test/adaptation-reserve identities are excluded from every replacement.

Rounded replacement counts are 0, 14, 69, and 277. Replacement sets are nested and shared by A and B; each new identity replaces an old identity with the same motif label, preserving motif priors and training size. Report actual fractions and per-motif selection audits. The 0% condition reuses the exact original model and representation files rather than repeating training. Other rates start from each corresponding archived random initialization.

Evaluate exposed cell-2 combinations, protected odd-motif cell-2 combinations, and each original IID set separately. All classifiers retain eight output labels; uniform eight-label accuracy is 12.5%, even on the four-motif protected subset. Also report the training-derived majority baseline. No protected example or protected test score is used in training, validation, or selection. Any gain on the exposed subset alone is not broad compositional generalization.

## Limits and records

Head-only failure under this fixed optimizer and readout family is not evidence of absent features. A pretraining advantage is conditional on its prior labels and compute. Replay may trade new performance against retention. Correlation interventions alter training exposure in a precisely recorded way, but do not by themselves identify a causal internal circuit. All configurations, identity selections, source hashes, checkpoints, numerical trajectories, derived reports, and checksummed manifests are saved in new directories.
