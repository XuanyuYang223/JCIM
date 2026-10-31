# JCIM

Research on reusable local structural representations in small molecular models.

## Research question

Do small models trained on different molecular structures learn local representations that transfer across substitution positions and molecular contexts? Does this reuse help them handle unseen structural combinations?

## Initial scope

- Generate diverse substituted benzene molecules by varying substituent identity, relative position, and side-chain length.
- Split by complete molecular identity, keeping every SMILES variant of a molecule in the same partition.
- Hold out substituent-position and substituent-context combinations for evaluation.
- Start with substituent and functional-group identification and localization, relative-position classification, and atom-to-atom shortest-path prediction.
- Compare models with matched architectures, training-set sizes, and training budgets; include repeated seeds and randomly initialized controls.
- Measure unseen-combination accuracy, cross-position readouts, layerwise CKA and Procrustes alignment on shared evaluation molecules, and adaptation data requirements.

The working hypothesis is that representations retain shared substituent information while adapting to position and molecular context. High CKA alone is insufficient evidence of functional reuse, and training is not assumed to increase CKA.

## Status

Repository initialized. Datasets, model implementations, and experiments have not yet been created.
