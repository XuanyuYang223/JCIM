# Harder pilot protocol

This protocol and `hard_v2.json` define the primary experiment before its execution. The earlier `pilot_v1` archive is preserved. A two-epoch smoke run with two adaptation updates is a pipeline diagnostic, not the primary evidence or a source of test-selected hyperparameters.

## Structural difficulty

The controlled space contains 6,912 unique molecules formed from eight rooted ether connectivity motifs, three ring attachment positions, eight reference groups, three reference spacer lengths, three aromatic scaffold variants, and four distractors. All eight target motifs have the same ordinary input-atom feature multiset and the same attachment-root attributes. Their nine-heavy-atom rooted structures differ, and all motif atoms fall within six bonds of the root. Eight message-passing layers therefore cover each target motif; a negative result is not explained merely by an unreachable target atom.

Only the queried attachment/endpoint atoms are marked. Target-fragment membership is absent from model inputs. The local representation is one marked attachment-root embedding. Auxiliary target-fragment membership metadata exists for auditing but is neither trained nor supplied to the encoder.

## Pretraining and holdouts

Canonical molecular identities are assigned to candidate training, validation, adaptation reserve, and testing with fixed 60/10/10/20 hash thresholds. Within these partitions, `(motif + position) % 3` defines three combination classes. A pretrains on class 0; B on class 1. Class 2 is unseen by both and supplies common OOD testing and adaptation support. Each experience sees every motif and every position, but only one position per motif. This deliberately severe correlation stress tests factorization rather than unrestricted chemical generalization.

Training is matched by motif count, giving 1,386 identities per experience. Position marginals differ as a consequence of the design and are reported. Validation is experience-specific and familiar-combination only. The final epoch is used; OOD labels never select checkpoints.

The three jointly supervised tasks are eight-class target connectivity, three-class ring relationship, and shortest-path length between the marked anchor functional root and target carbon leaf. The query leaf is chosen by a stable molecular-identity hash from the nearest/farthest carbon leaves. No experimental-property labels are used.

## Comparisons

Three initialization seeds each provide A, B, and an exactly matched random encoder/head state. Architecture, optimization, training-set size, and updates match between A and B. The primary configuration fixes 120 pretraining epochs and all adaptation budgets before execution. This is not an isolated difficulty intervention relative to v1: the chemical motifs, data size, depth, and training budget change together.

Local ridge readouts fit on ortho test-partition identities and evaluate on distinct meta identities, with raw-root, random-encoder, and shuffled-source-label controls. Regularization is fixed. The raw-root macro-accuracy reference is 1/8; random message passing can still expose connectivity.

Similarity comparisons use the same 512 sorted test identities for both local and graph representations. Local geometry is measured at all layers, graph geometry at layers 0, 4, and 8. Pair categories distinguish same-data/different-seed, different-data/same-seed, and different-data/different-seed conditions. Model-pair statistics are descriptive and correlated.

## Actual adaptation and retention

Full-model adaptation retains the pretrained heads and updates all encoder/head parameters. It uses 8, 32, or 128 newly labeled molecules from the 223-identity class-2 adaptation pool, three fixed uniform nested support draws, and 80 fixed AdamW updates. Both pretrained experiences exclude every support identity. The random condition uses the same updates and labels. Zero-shot points reuse the original heads without updates.

No pretraining-data replay is used. Both IID test splits and the common OOD test are evaluated after adaptation, so OOD gains accompanied by IID forgetting are visible. Few-shot sample draws are averaged within encoder seed; they are not additional independent pretraining seeds. Checkpoints, support identities, predictions, graph arrays, numerical measurements, source hashes, and report outputs are retained with a manifest.

## Interpretation

Success is not defined as a high CKA or a saturated local probe. The experiment can expose behavioral failure, accessible local connectivity, and actual adaptation/retention tradeoffs. An explanation of causal local representation reuse requires further controlled interventions; a training or readout advantage alone is insufficient.
