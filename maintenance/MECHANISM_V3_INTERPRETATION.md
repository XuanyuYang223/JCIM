# Independent interpretation of mechanism v3

Three findings emerge from the frozen-encoder and replay experiments. The pretrained encoder contains useful information that its original heads fail to exploit on new combinations. Forgetting can occur entirely in the readout, and full updating introduces encoder/head coadaptation. The remaining evidence supports dependence on training motif-position bindings and local task solvability, while stopping short of a stable, context-independent local-code claim.

## Records and statistical scope

This is a separate post-run interpretation; archived data, reports, weights, and source files remain unchanged. Both completed records executed source commit `499194ec5d43c93e004d9ccc9a168df42b2fbd18`. Their environments retain individual source-file hashes.

| Record | Manifest SHA-256 |
| --- | --- |
| [hard-v2 source](../experiments/hard_v2/MANIFEST.json) | `4b69cd0335cd21ccb5e2a419c1301bfc8042e3bb5e9864322d6ad4ca1f398996` |
| [readout/replay v3](../experiments/readout_replay_v3/MANIFEST.json) | `e4f6a3a4abfa0a41f4f13a2ae4573ff7008287d6bc84a1306b97f671f5f37d26` |
| [mechanism diagnostics v3](../experiments/mechanism_diagnostics_v3/MANIFEST.json) | `2bea772ce613fcf2fe79f66326847de5cbb8e01ec62aa727a7fec4740fcad76a` |

All 106 readout/replay and 22 diagnostic manifest entries were independently checked. Diagnostics verified all 131 source files and 6,912 identities before and after running, with the source manifest unchanged. Its copied probe CSV is hash-identical to the source. The separate [readout/replay audit](READOUT_REPLAY_V3_AUDIT.json) also passes.

Calculations below use the raw [trajectory](../experiments/readout_replay_v3/trajectory.csv), [ridge](../experiments/readout_replay_v3/ridge_diagnostic.csv), [component](../experiments/readout_replay_v3/component_eval.csv), and [drift](../experiments/readout_replay_v3/drift.csv) files. Their hashes are recorded by the linked manifest. Primary results use only predeclared support repetition 0 and final step 80. Entries are motif accuracy in percent, mean +/- sample SD across seeds 0/1/2; n=3. Supplemental results average repetitions 0/1/2 within each seed before computing that seed mean and SD, never treating the draws as nine independent encoders.

The new test is common OOD (443 untouched identities); the old test is own IID, A: 462 identities, B: 472. A/B old distributions differ and are kept separate. All arms reuse the same 32 new identities per support draw. Replay adds 32 fixed old training identities; supports exclude tests. Every update uses the entire 32- or 64-example support for 80 updates. Update count is matched, but total data access, example presentations, and computation are not. Frozen arms hold the embedding and eight message layers fixed while training all three inherited MLP heads, including hidden layers. Heads are not reset; all three task losses contribute.

## 1. Useful information is present, with an important graph-versus-root distinction

Primary repetition 0:

| Experience | Arm | New common OOD | Old own IID |
| --- | --- | --- | --- |
| A | Frozen, 32 new | 80.2 +/- 4.7 | 75.3 +/- 9.9 |
| A | Full, 32 new | 92.8 +/- 3.7 | 66.7 +/- 2.3 |
| A | Frozen, 32 new + 32 old | 73.7 +/- 5.5 | 93.4 +/- 2.6 |
| A | Full, 32 new + 32 old | 95.2 +/- 2.0 | 95.6 +/- 0.8 |
| A | Scratch, 32 new + 32 old | 74.2 +/- 4.1 | 73.2 +/- 6.1 |
| B | Frozen, 32 new | 80.1 +/- 11.1 | 84.3 +/- 8.5 |
| B | Full, 32 new | 95.9 +/- 2.9 | 57.0 +/- 7.5 |
| B | Frozen, 32 new + 32 old | 76.3 +/- 13.9 | 94.3 +/- 3.9 |
| B | Full, 32 new + 32 old | 87.6 +/- 6.1 | 95.7 +/- 2.8 |
| B | Scratch, 32 new + 32 old | 68.0 +/- 5.4 | 60.5 +/- 2.2 |

Frozen adaptation raises new performance from A's 0.0% and B's 4.2% baseline to about 80%. Thus an unchanged encoder supports substantial recovery through a different readout. Full replay improves the old/new balance and exceeds scratch new accuracy by 21.0 percentage points for A and 19.6 for B under the same 64-label support and 80 updates. This establishes a descriptive fixed-budget advantage, not a full sample-complexity curve or a significance claim. Replay protects old behavior but does not uniformly increase new accuracy: B full replay moves new/old from 95.9/57.0% to 87.6/95.7%.

Supplemental full-update results, averaging support draws within seed:

| Experience | Arm | New common OOD | Old own IID |
| --- | --- | --- | --- |
| A | Full, 32 new | 91.1 +/- 0.4 | 70.3 +/- 9.7 |
| A | Full, 32 new + 32 old | 89.7 +/- 1.4 | 97.2 +/- 1.0 |
| B | Full, 32 new | 95.3 +/- 2.7 | 52.7 +/- 1.4 |
| B | Full, 32 new + 32 old | 86.5 +/- 2.2 | 92.2 +/- 1.7 |

The same retention pattern persists, but support dependence matters: A full-replay new accuracy is 95.2% on primary repetition 0 versus 89.7% after support averaging. The [complete report](../experiments/readout_replay_v3/REPORT.md) retains every arm under both aggregation rules and the full adaptation trajectories.

Fresh ridge readouts give a stronger test of feature availability. They use the unadapted encoder, 64-dimensional final root features or 448-dimensional graph/query features, support-only standardization, and fixed alpha=1.0. These are new linear decoders, separate from inherited-MLP adaptation.

Primary repetition 0:

| Experience | Features | Support labels | New common OOD | Old own IID |
| --- | --- | ---: | --- | --- |
| A | Root | 32 new | 100.0 +/- 0.0 | 0.0 +/- 0.0 |
| A | Graph/query | 32 new | 100.0 +/- 0.0 | 0.0 +/- 0.0 |
| A | Root | 32 new + 32 old | 74.9 +/- 5.6 | 57.0 +/- 2.8 |
| A | Graph/query | 32 new + 32 old | 93.4 +/- 2.5 | 94.7 +/- 4.2 |
| B | Root | 32 new | 100.0 +/- 0.0 | 7.3 +/- 4.7 |
| B | Graph/query | 32 new | 100.0 +/- 0.0 | 2.5 +/- 4.3 |
| B | Root | 32 new + 32 old | 62.8 +/- 1.0 | 63.5 +/- 5.9 |
| B | Graph/query | 32 new + 32 old | 90.4 +/- 2.8 | 89.8 +/- 2.1 |

The root features can support a decoder for the reserved new binding despite the earlier weak ortho-to-meta probe. Their new-only success with poor old retention is compatible with context-sensitive information readable by a support-specific rule. Graph features plus replay support strong simultaneous old/new decoding without changing the encoder. Failure of inherited frozen heads to reach those scores therefore does not establish a feature ceiling: decoder initialization, optimization, parameterization, regularization, and the three-task objective differ.

This does not cleanly locate information outside the root. Graph/query features include the root vector, additional global/query pools, and seven times as many coordinates. The advantage can reflect added context, readout capacity, or conditioning. Likewise, root-ridge failure does not prove absence under every decoder. Supplemental replay-ridge graph old/new means remain stronger than root: A graph 89.6/91.4% versus root 59.1/69.4%; B graph 85.2/86.5% versus root 64.9/67.6%. These are support-averaged means, not the primary table.

## 2. Forgetting can arise in the readout; full updating shows coadaptation

All 36 final frozen-arm encoder hashes independently match their corresponding pretrained encoders. The comparison hashes sorted parameter names, shapes, and contiguous tensor bytes. All 252 frozen-arm drift rows also show maximum encoder change 0.0 and `encoder_changed=False`. Nonetheless, primary frozen-new old accuracy falls from 100% to 75.3% for A and 84.3% for B. Functional forgetting therefore occurs entirely through changed readout behavior on fixed features. It does not require erasure of encoder information.

For full-new adaptation, E0/H0 denotes pretrained encoder/heads and E1/H1 the jointly adapted pair. The component interventions, primary repetition 0, are:

| Experience | Components | New common OOD | Old own IID |
| --- | --- | --- | --- |
| A | E0/H0 | 0.0 +/- 0.0 | 100.0 +/- 0.0 |
| A | E0/H1 | 2.7 +/- 2.4 | 100.0 +/- 0.0 |
| A | E1/H0 | 74.3 +/- 8.8 | 90.5 +/- 6.3 |
| A | E1/H1 | 92.8 +/- 3.7 | 66.7 +/- 2.3 |
| B | E0/H0 | 4.2 +/- 7.1 | 100.0 +/- 0.0 |
| B | E0/H1 | 12.9 +/- 10.8 | 100.0 +/- 0.0 |
| B | E1/H0 | 67.3 +/- 8.1 | 88.1 +/- 8.7 |
| B | E1/H1 | 95.9 +/- 2.9 | 57.0 +/- 7.5 |

The conditional old-accuracy contrast `(E1/H1 - E1/H0) - (E0/H1 - E0/H0)` is -23.7 +/- 5.0 percentage points for A and -31.1 +/- 12.2 for B, computed within seed before aggregation. This is an algebraic functional interaction, not an additive causal decomposition of information loss. Adapted heads preserve old behavior on E0 but perform differently on E1. With full replay, E1/H0 old/new means are A 98.9/62.4% and B 97.3/55.2%, while E1/H1 reaches A 95.6/95.2% and B 95.7/87.6%.

These swaps show compatibility and coadaptation along the actual full-update trajectory. They do not show that all new information was created inside E1: frozen/ridge controls already recover it from E0. H1 from joint training is a different head from frozen adaptation, and crossed components see coordinates/distributions they were not jointly optimized to use. Accuracy differences, parameter drift, and CKA cannot uniquely assign where a concept is stored or forgotten.

## 3. Binding errors and local solvability are supported; invariant reuse is unresolved

The [position-only MAP baseline](../experiments/mechanism_diagnostics_v3/position_baselines.csv) achieves own-IID accuracy A 35.93%, B 37.92%, with balanced accuracy 37.5% in both and common-OOD accuracy 0%. A position supports multiple motifs. Three positions cannot yield more than three distinct MAP predictions for eight balanced classes, so this baseline cannot explain the original models' 100% own-IID classification.

Instead, almost every common-OOD error predicts a motif supported at that position in its own pretraining set. Error-supported-set fractions for seeds 0/1/2 are A 100.00/100.00/99.77% and B 100.00/100.00/100.00%. Exact MAP agreement is much lower: A 51.69/39.05/38.60%, B 23.02/23.02/35.44%. This supports training-binding dependence, without establishing an exact pure-position MAP explanation. All denominators are retained in the [agreement CSV](../experiments/mechanism_diagnostics_v3/position_model_agreement.csv).

The oracle local-input control reaches 100% motif accuracy and balanced accuracy for every seed, test split, and position. It uses the complete known nine-atom target induced subgraph, root radius at most six, original ordinary atom features, and only the target-root query marker. Remote ring, anchor, and endpoint markers are removed. A/B local input distributions match, permitting A-only training per seed. All eight local graph types intentionally recur across train/test after context removal, although molecular identities remain disjoint. This establishes local task solvability with supplied target membership, not novel-motif generalization or reuse by the original full-input encoder.

Eight-class uniform chance is 12.5%. The random encoder's earlier mean near 37.1% is an empirical representation-plus-probe baseline, not theoretical chance. The final-layer archived controls retain seed variation:

| Encoder | Seed | Probe balanced accuracy | Shuffled-label ordinary accuracy | Uniform chance |
| --- | ---: | ---: | ---: | ---: |
| Random | 0 | 40.37 | 12.97 | 12.50 |
| Random | 1 | 32.80 | 19.78 | 12.50 |
| Random | 2 | 38.23 | 5.71 | 12.50 |
| A | 0 | 12.50 | 0.00 | 12.50 |
| A | 1 | 37.04 | 14.07 | 12.50 |
| A | 2 | 13.35 | 13.85 | 12.50 |
| B | 0 | 12.50 | 19.78 | 12.50 |
| B | 1 | 12.50 | 16.70 | 12.50 |
| B | 2 | 37.50 | 20.00 | 12.50 |

All 82 model/layer rows, including raw inputs and shuffled controls, remain in [frozen_probe_controls.csv](../experiments/mechanism_diagnostics_v3/frozen_probe_controls.csv). The archived shuffle uses one deterministic permutation seed, 8107; its ordinary accuracy is not relabeled as balanced accuracy or treated as a distribution over permutations. Low probe scores constrain that readout and source/target support design, not all layers or every possible decoder.

The next correlation intervention asks whether a few binding-breaking examples improve other combinations that remain wholly unexposed. No result from that unfinished phase is incorporated here. Exposed combinations measure few-shot learning; permanently reserved combinations provide the compositional test. Stronger local-reuse claims also require capacity-matched graph/root readouts and cross-position evaluation under an independently fixed protocol, beyond the present support-specific recoverability.
