# Supporting methods and record guide

This file supports the [working manuscript](MANUSCRIPT.md). It describes completed records through v7. Repository paths below are relative to the repository root unless linked otherwise. Numerical evidence is preserved in the experiment archives. The manuscript and this supplement are separate documentation and do not revise those archives. Numerical execution checks are distinguished below from saved scientific outcomes.

## S1. Molecular construction and label computation

The eight rooted target motifs are recorded in `jcim/hard_data.py` and in the frozen molecular records. In each SMILES, the first atom is the attachment root:

| Motif index | Rooted target SMILES |
| --- | --- |
| 0 | `CCOC(CCC)OC` |
| 1 | `CCOC(CC)OCC` |
| 2 | `CCOC(CC)COC` |
| 3 | `COCC(CCC)OC` |
| 4 | `COCC(CC)OCC` |
| 5 | `COCC(CC)COC` |
| 6 | `CCCC(OC)OCC` |
| 7 | `CCCC(COC)OC` |

Each target has nine heavy atoms: seven carbons and two oxygens. The rooted target graphs differ in connectivity while matching the multiset of ordinary node attributes. In particular, target classification cannot be solved from elemental counts, an order-free bag of those ordinary attributes, or the uncontextualized root attributes alone. This statement does not remove graph information supplied by the query markers or their arrangement.

The generation grid crosses eight motifs, three attachment positions, eight reference-fragment families, three reference-side spacer lengths, three aromatic scaffolds, and four distractors. The default spacer lengths are zero, one, and two methylene units. The scaffold list is benzene, `pyridine_n4`, and `pyridine_n5`; distractors are methyl, fluoro, hydroxy, and cyano. Distractor placement depends on background factors. A canonical-identity hash selects either a near or far terminal target-carbon endpoint for the path query. The labels are motif index, ortho/meta/para ring relationship, and exact shortest-path bond count between the marked reference functional root and target endpoint. All are computed graph labels.

The archive contains 6,912 canonical identities. The known-target atom membership is retained as annotation for diagnostics and is not part of the original encoder input. The original encoder receives 18 atom channels: element indicators including an other-element channel; aromaticity; scaled formal charge, degree, and total hydrogen count; ring membership; and five query markers. The markers identify the reference ring atom, target ring atom, target attachment atom, reference functional root, and path endpoint. Adjacency is binary. The present architecture does not use a separate bond-type embedding, experimental labels, conformers, or stereochemistry.

## S2. Identity partitions and combination exclusions

Canonical SMILES are deduplicated before partitioning. A SHA-256 function of split seed and canonical identity supplies a fixed score in the unit interval. The intervals below set candidate populations; their realized counts need not equal the nominal percentages exactly.

| Score interval | Base role | Realized count |
| --- | --- | ---: |
| Below 0.6 | Candidate pretraining | 4,196 |
| 0.6 to below 0.7 | All validation candidates | 683 |
| 0.7 to below 0.8 | Adaptation reserve | 656 |
| At least 0.8 | Ordered analysis/test set | 1,377 |

The combination residue is `(motif + position) % 3`, where positions zero, one, and two denote ortho, meta, and para. A uses residue zero; B uses residue one. Residue two remains outside source training and validation for both. Motif-wise downsampling matches the two source training counts at 1,386. The matching controls motif class counts across experiences; position marginals differ because the binding differs.

| Role | A | B | Shared unseen |
| --- | ---: | ---: | ---: |
| Effective source training | 1,386 | 1,386 | 0 |
| Familiar validation | 237 | 208 | 0 |
| Test identities | 462 | 472 | 443 |

The 238 unseen-combination validation candidates are excluded from validation. The adaptation pool consists of 223 residue-two identities from the separate reserve. It does not overlap either training experience or the test set. The eight motifs and three positions are all present in each experience, while every individual motif is restricted to one source position. Unseen-combination tests therefore measure new pairings of known factors, not new motif topology. Generation-grid factors and local motifs recur across molecular identity partitions by design.

## S3. Encoder and original graph/query readout

For node state \(h_i^{(l)}\), a message layer computes a neighbor sum \(s_i=\sum_{j\in N(i)}h_j^{(l)}\), followed by a shared two-layer update of \([h_i^{(l)},s_i]\). The update is added residually, normalized over the node's feature coordinates, and rectified. Padding is excluded from messages and pooling. Eight layers of width 64 follow an affine atom embedding and rectifier.

The original graph/query vector is a concatenation of seven 64-dimensional blocks: whole-graph mean, whole-graph sum divided by square-root atom count, marked target-root state, and four sums selected by the other query markers. This 448-dimensional vector feeds three separate hidden-width-64 rectified heads. Their output counts are eight motifs, three positions, and sixteen possible path-length classes. The base network has 189,531 parameters.

Original source training uses AdamW with learning rate 0.0015, weight decay 0.0001, batches of 128, gradient clipping at five, and 120 epochs. All three cross-entropy losses contribute. Initialization seeds are zero, one, and two. Exact minibatch orders are fixed within seed and reused in later matched fits where specified. Familiar validation is available, while final unseen-combination outcomes are evaluated after the fixed endpoint in the shallow-path and pooling protocols.

Shared attributed-neighborhood aggregation has a useful implication. If two rooted receptive neighborhoods, including all initial attributes and query markers, are isomorphic, corresponding states are equal in exact arithmetic under the same parameters before unmatched distant context enters their receptive support. This applies to random as well as trained weights. Pooling over corresponding selected nodes preserves equality under the same pooling operation. This argument establishes consistency, not separability of different motifs or invariance to arbitrary context. See the separate [architecture note](../maintenance/ARCHITECTURE_LOCALITY_NOTE.md) and the primary neighborhood-aggregation reference in [REFERENCES.md](REFERENCES.md).

## S4. Frozen readout definitions and label provenance

For a matrix of fitting features, each coordinate is centered and scaled using only that fit's population mean and standard deviation. Constant coordinates receive scale one. An intercept is added without penalization. The fixed ridge classifier minimizes squared error against one-hot class targets with alpha equal to one; predictions are the maximum-scoring class. Standardization and regularization do not use target-test statistics.

The v4 cross-position analysis considers layers zero through eight and multiple spatial pools. Root, target mean, target plus successive graph shells, and whole-graph mean are all 64-dimensional; the original graph/query vector is a distinct 448-dimensional reference. Oracle masks are applied after encoding the complete molecule. At deep layers, target-selected states can aggregate information from outside the selected atoms. Across the 1,649 archived feature identities, both root and target pools have whole-graph receptive reach for 98.7% at layer eight. This population includes the fixed analysis identities and fitting supports. This is an architectural reach statement, not a measured influence of every reachable atom.

Directed probes fit 256 balanced examples, 32 per motif, at one source position and evaluate the other positions. All six ordered source–target directions and three source-support draws are retained. The exact supports are in `experiments/scale_v4/direct_support.json`. Source supports can include motif–position combinations absent from source encoder training. No label from the target position enters that directed fit. Thus these probes add diagnostic supervision and cannot be equated with the original source-only predictor. Their support- and direction-averaged summaries first average within encoder seed and then summarize the three seed values.

The adaptation comparison instead uses the exact archived 32-new-label supports and 32 old source labels per experience and draw. These supports are identical for matched trained and random encoders. New-label supports come from the separate adaptation pool. Frozen encoders supply features to a fresh standardized ridge classifier. The main-text adaptation advantages use layer-eight graph/query features and balanced accuracy after averaging three draws within seed. They are paired trained-minus-random contrasts. The v4 report's predeclared primary support draw is zero; the manuscript identifies its quoted repeat-averaged summary explicitly rather than relabeling it as that primary fixed-support result.

## S5. Shallow-path protocol and branch sensitivity

The v5 shallow-path model concatenates the actual layer-one whole-graph mean with the 448-dimensional deep vector. The matched control concatenates an exact duplicate of its final-layer mean block. Both 512-dimensional variants have 201,819 parameters and the same hidden heads. Extra input-column weights start at zero; all inherited head weights are matched. End-to-end variants begin from the same archived random initialization and use the original training identities, hyperparameters, and epoch orders. No target-fragment annotation is consulted.

The frozen-readout regime uses the corresponding original trained encoder and resets the heads to their archived same-seed random values before source-only training. It is not continuation of the old trained head. Every fresh fit receives 1,320 updates over 120 epochs. The original end-to-end deep baselines reuse exact completed v2 checkpoints and histories; their predictions are recomputed. Reused historical budgets are distinguished from newly executed updates in the inventory. Matching updates and parameter counts does not establish matching floating-point work.

The v6 branch analysis uses the completed v5 checkpoints and saved 512-dimensional features. Intact predictions must match the v5 record. Zeroing or a fixed label-free permutation replaces only the appended 64 coordinates. The deep coordinates, encoder, and final heads remain fixed. For correct label \(y\), margin is \(m=z_y-\max_{k\ne y}z_k\); the recorded margin change is altered minus intact. Positive change favors the correct label. Absolute changes avoid cancellation of positive and negative movements. Centered-logit changes remove class-common offsets.

The main text focuses on the common-unseen motif head of the twelve shallow models: two regimes, two experiences, and three seeds. All 24 augmented deep-control/shallow models, three heads, familiar test splits, and ordered analysis identities remain in the complete branch record. Quantities are computed per molecule, summarized per model, and then summarized across seeds. No molecules or heads are counted as additional independent initializations. Zeroing and permutation can create incompatible feature pairs, so the effects are conditional output sensitivities rather than semantic contributions that can be added across branches.

## S6. Matched local and whole-graph source-only fits

The v6 whole-graph pool is exactly the 64-dimensional layer-one vector used by the v5 shallow branch; extraction is checked bit for bit against the archived vector. The oracle pool averages the nine annotated target atoms at the same layer. Both have equal width, exclude padding, accumulate means in float64, and return float32. The same complete molecular input is used for both. The oracle adds membership information to pooling but not to message passing. Its first-layer states can depend on immediate external neighbors and initial boundary attributes.

Each source-only fit uses precisely the original experience's 1,386 training identities and motif labels, with no adaptation-reserve labels or residue-two labels. Both pooling scopes use the same fixed standardization and ridge fit. The additional nonlinear classifier is `64 -> 64 -> 8`, with rectified hidden units and cross-entropy loss. It uses 120 epochs, the source training orders, and the original AdamW settings. Initialization is matched across pools. The nonlinear head uses only motif loss, while the original v5 classifier was jointly trained on three tasks.

Encoder sources are the actual end-to-end shallow-path encoders, the original trained encoders used in v5's frozen regime, and the three original random encoders. Random features are extracted and fitted under the same procedures. A and B random-source fits share an encoder for a given seed but use different source identities; they are not independent encoder initializations. Original trained encoders and frozen-v5 encoders likewise refer to the same states rather than extra model replications.

The oracle-minus-whole paired contrast holds this new fitting protocol fixed and changes supplied pooling membership. A comparison with the inherited v5 head changes more than localization: standardization, input range, initialization, regularization, fitting objective, and the use of deep coordinates can differ. Ridge versus nonlinear fitting also changes multiple conditions. The completed factor analysis below narrows these explanations conditionally; it does not decompose the entire historical change.

## S7. Statistics, uncertainty, and archived checks

Ordinary accuracy is the fraction of molecules assigned the correct motif. Balanced accuracy is the unweighted mean of class recalls over present test labels. Uniform eight-output guessing has expected ordinary accuracy 0.125 for any fixed label distribution and expected balanced accuracy 0.125. A fitted decoder on random features can substantially exceed this expectation and is a separate control.

All `±` values in the main text are sample SD across three encoder seeds, except where a raw within-head range is explicitly identified. Repeated support draws and directions are averaged within a seed. Paired differences are formed before the seed mean and SD. A/B experiences use different familiar distributions and remain separate. The manuscript makes no formal significance claim or best-seed selection. Probe success establishes accessibility for that diagnostic; a failed probe does not establish the absence of all usable information.

The archives preserve source hashes, exact supports, identity plans, predictions, metrics, model inventories, and frozen states. Independent audit reports are separate maintenance artifacts. The v6 audit verifies source continuity, oracle masks, frozen encoder identities, ridge solutions, saved nonlinear-head predictions, and branch logits. Such checks support reproducibility of the saved computations. Final states and metadata cannot independently establish every historical optimizer operation or rule out all external analyst selection. The broader investigation was sequential, although each later diagnostic protocol records fixed settings before its primary execution.

## S8. Record and figure map

| Evidence | Frozen record | Main machine-readable source |
| --- | --- | --- |
| Dataset identities, split counts, original prediction behavior | [hard v2](../experiments/hard_v2/REPORT.md) | `audit.json`, `molecules.jsonl`, `splits.json`, `behavior.csv` |
| Position baselines and local-input solvability | [mechanism v3](../experiments/mechanism_diagnostics_v3/REPORT.md) | `position_baselines.csv`, `position_model_agreement.csv` |
| New/old adaptation and forgetting | [readout/replay v3](../experiments/readout_replay_v3/REPORT.md) | `trajectory.csv`, `ridge_diagnostic.csv`, `drift.csv` |
| Trained/random feature matching, spatial scale, cross-position and adaptation budgets | [scale v4](../experiments/scale_v4/REPORT.md) | `seed_metrics.csv`, `paired_seed_deltas.csv`, `direct_support.json` |
| Boundary-free shallow-path intervention | [skip v5](../experiments/skip_v5/REPORT.md) | `seed_metrics.csv`, `paired_seed_deltas.csv`, `inventory.csv` |
| Actual shallow vector, source-only oracle/whole pooling, branch output sensitivity | [pooling v6](../experiments/pooling_v6/REPORT.md) | `pooling/pooling_source_metrics.csv`, `pooling_direct_seed.csv`, `branch/branch_summary.csv` |
| Crossed readout factors and exact-protocol random controls | [readout v7](../experiments/readout_v7/REPORT.md) | `panel/source_metrics.csv`, `conditional_factor_edges.csv`, `factorial_effects.csv`, `matched_random_deltas.csv` |

The existing figures can support an initial layout without replotting frozen numerical records:

1. **Controlled motifs and partitions.** [Manuscript Figure 1](figures/controlled_design.png) and its [native vector version](figures/controlled_design.svg) show the eight recorded target motifs, the combination-cell matrix, and separate supervision budgets. The figure is generated from frozen motif SMILES by `manuscript/build_design_figure.py`; [metadata](figures/CONTROLLED_DESIGN_METADATA.json) retains source provenance. Its blue root and dummy connection marker do not supply a target boundary to the encoder. The caption distinguishes motif topology reuse from canonical molecular identity isolation.
2. **Spatial readability and labeled adaptation.** [Scale analysis](../experiments/scale_v4/figures/scale_primary.png) and [layerwise paired curves](../experiments/scale_v4/figures/scale_paired_layers.png). Captions must retain their distinct label budgets and balanced accuracy definition.
3. **Boundary-free intervention.** [Shallow-path endpoints](../experiments/skip_v5/figures/skip_primary.png). Ordinary motif accuracy, familiar guards, and parameter-matched controls remain labeled.
4. **Source-only pooling and output sensitivity.** [Pooling comparison](../experiments/pooling_v6/figures/pooling_source.png) and [branch output comparison](../experiments/pooling_v6/figures/branch_outputs.png). Oracle membership and head-specific logit units remain explicit.

These are references to existing artifacts, not completed submission figure numbering. The v7 [factor grid](../experiments/readout_v7/figures/readout_factorial.png) and [paired effects](../experiments/readout_v7/figures/readout_effects.png) provide the additional readout panels. A broader chemistry literature review and submission-specific editing remain necessary. No public repository, author list, funding source, or journal acceptance status is asserted.

## S9. Readout-factor protocol and complete cell results

The completed [v7 protocol](../configs/READOUT_V7_PROTOCOL.md) freezes the actual 512-coordinate v5 vectors and the original source identities. Its MLP panel crosses three binary choices: full deep-plus-shallow input versus only the shallow coordinates; raw features versus source-training standardization; and motif-only versus summed three-task fitting. The first 448 coordinates are zeroed for shallow-only fits, preserving the same head dimensions and parameter counts. Exact same-protocol random encoders provide the initialization reference for every trained cell. The panel contains 144 MLP fits and 72 ridge fits. Fifteen frozen encoders supply eighteen source-label cases because each of the three random encoders is fitted against both source experiences.

Each cell copies the archived random v5-compatible head: original 448 input columns are copied and appended shallow columns are zero. All factors therefore start from the same complete head state. This convention has a specific consequence: shallow-only masking removes every initially nonzero input weight, so the initial output is bias-driven. The fresh 64-input nonlinear v6 head instead starts with nonzero feature weights. The scope factor changes which information is available under this fixed historical initialization, but its optimization consequences cannot be interpreted as a universal information-removal effect.

The frozen representation and separate heads have no ordinary shared trainable feature path. At identical motif-head states, adding position and distance losses does not change the motif gradient before clipping. The objective factor can change that gradient's common scaling through a global gradient-norm clip across all active heads. Per-step gradient norms and clipping coefficients are retained to quantify this coupling. The archive records 264 clipped updates among 190,080 total updates, occurring in 20 of 144 MLP fits. The contrast cannot identify multitask effects during encoder learning.

Separate raw and standardized ridge fits provide descriptive decoder comparisons. Standardized shallow-only ridge reproduces the v6 whole-graph ridge states and predictions, while raw full-input joint fitting on original trained encoders reproduces the corresponding v5 frozen-head anchor. Those are distinct continuity checks; the full factorial is not a reproduction of v6's nonlinear initializer or an additive decomposition of the complete v5-to-v6 difference. Claims C100–C112 record all cell outcomes, paired factors, random contrasts, clipping counts, and continuity anchors.

Conditional contrasts are shallow-only minus full input, standardized minus raw, and motif-only minus joint, with companion factors fixed. Averaged effects first average the companion cells within seed. Higher-order interactions are signed differences of these contrasts, not regression coefficients. Each reported mean and sample SD then uses the three seed values; the complete conditional and interaction tables remain in the linked primary record.

The following tables reproduce every predeclared common-unseen cell from the [primary report](../experiments/readout_v7/REPORT.md). All scores are ordinary motif accuracy in percent, mean ± sample SD across three seeds. “V5 end-to-end” describes encoder history: its encoder is frozen throughout this readout experiment. Original-trained encoders are the v2 states used in v5's frozen regime. Each A/B random column uses the same encoder seeds with different source-label fits; those columns are not six independent initializations.

**Table S1. Complete MLP factorial.**

| Scope | Scaling | Loss | V5 end-to-end A | V5 end-to-end B | Original trained A | Original trained B | Random A | Random B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full | Raw | Joint | 0.2 ± 0.3 | 9.9 ± 3.2 | 0.0 ± 0.0 | 7.5 ± 6.6 | 29.6 ± 4.0 | 13.1 ± 7.2 |
| Full | Raw | Motif | 0.2 ± 0.3 | 9.9 ± 3.2 | 0.0 ± 0.0 | 6.9 ± 6.3 | 29.6 ± 4.0 | 13.2 ± 7.5 |
| Full | Standardized | Joint | 0.0 ± 0.0 | 3.4 ± 5.9 | 0.0 ± 0.0 | 3.3 ± 5.7 | 36.1 ± 4.0 | 29.2 ± 5.0 |
| Full | Standardized | Motif | 0.0 ± 0.0 | 3.4 ± 5.9 | 0.0 ± 0.0 | 3.3 ± 5.7 | 36.1 ± 4.0 | 29.2 ± 5.0 |
| Shallow | Raw | Joint | 19.3 ± 12.5 | 9.3 ± 4.3 | 25.9 ± 14.3 | 8.5 ± 0.9 | 15.9 ± 3.0 | 16.1 ± 2.9 |
| Shallow | Raw | Motif | 19.3 ± 12.5 | 9.3 ± 4.3 | 25.9 ± 14.3 | 8.5 ± 0.9 | 15.9 ± 3.0 | 16.1 ± 2.9 |
| Shallow | Standardized | Joint | 17.8 ± 16.0 | 26.7 ± 6.6 | 19.0 ± 19.4 | 20.4 ± 16.4 | 52.1 ± 11.9 | 39.4 ± 16.9 |
| Shallow | Standardized | Motif | 17.8 ± 16.0 | 26.7 ± 6.6 | 19.0 ± 19.4 | 20.4 ± 16.4 | 52.1 ± 11.9 | 39.4 ± 16.9 |

**Table S2. Separate ridge panel.**

| Scope | Scaling | V5 end-to-end A | V5 end-to-end B | Original trained A | Original trained B | Random A | Random B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Full | Raw | 1.6 ± 2.7 | 8.3 ± 7.2 | 0.5 ± 0.9 | 8.2 ± 4.8 | 44.8 ± 8.7 | 52.4 ± 13.0 |
| Full | Standardized | 0.0 ± 0.0 | 8.4 ± 6.9 | 0.0 ± 0.0 | 7.4 ± 4.4 | 60.0 ± 10.7 | 53.6 ± 16.9 |
| Shallow | Raw | 26.4 ± 16.2 | 28.0 ± 3.5 | 30.6 ± 21.8 | 18.8 ± 12.0 | 41.8 ± 7.3 | 39.2 ± 10.8 |
| Shallow | Standardized | 37.4 ± 19.5 | 64.2 ± 9.6 | 27.4 ± 8.5 | 58.6 ± 20.3 | 48.9 ± 24.1 | 65.1 ± 25.9 |

Raw shallow MLP scores must be read alongside their fitting guards. For end-to-end-source A/B, motif-only training accuracy is 52.6% ± 10.0% and 55.3% ± 12.3%, and own-familiar accuracy is 50.6% ± 10.5% and 54.6% ± 15.4%. Corresponding standardized fits reach approximately 100% training/familiar accuracy, but still have weak unseen accuracy. Raw shallow random fits also underfit. A weak raw fit is therefore not sufficient evidence that the corresponding features contain no useful structure.

Matched training benefits are protocol dependent. In the standardized shallow ridge cell, the end-to-end-source trained-minus-random contrast is −11.5 ± 40.5 percentage points for A and −0.9 ± 31.4 for B. For original-trained encoders it is −21.5 ± 28.8 and −6.5 ± 39.7. These descriptive, high-variance comparisons do not establish a source-only training advantage and do not negate the separate newly labeled adaptation advantage. The complete matched table includes positive and negative cells rather than selecting a favorable readout.

## S10. Post-run nested-ridge certificate

The [containment certificate](../maintenance/READOUT_V7_CONTAINMENT.json) is a separate post-run saved-state analysis with no refitting, new training labels, or new predictions selected for performance. It checks all 36 full/shallow ridge pairs, including both scaling choices and all encoder sources. The common shallow-coordinate scalers are bit-identical. Appending 448 zero coefficient rows to each shallow solution preserves its logits and predictions and leaves its squared coefficient penalty unchanged. This gives a feasible classifier in the full ridge model with the same shallow decision rule.

For every pair, the saved full solution has source sum-squared error plus alpha times squared coefficient norm no greater than that feasible shallow solution. For trained encoders, a lower source objective can therefore coexist with poorer unseen accuracy despite the shallow rule being available inside the full input space. In the main standardized contrast, training and familiar accuracy are also 100% for both scopes. This is evidence about the selected mapping under the fixed source objective, not absent shallow input information. It does not determine a unique semantic shortcut, imply every extra feature harms generalization, or constitute 36 independent training replicates. The certificate and its source hash are recorded as claim C113.

## S11. Numerical reproducibility qualification

The initial independent [CPU audit](../maintenance/READOUT_V7_AUDIT.json) remains archived with status **FAIL**. Its single failure is an untrained auxiliary distance head in the motif-only cell `fit_0168`, evaluated on source identity index 2596. The CPU execution ties classes 2 and 9 at approximately 0.62518746 and chooses class 2; the saved CUDA execution chooses class 9 with a top-two gap of approximately 1.79 × 10⁻⁷. This is a source-training molecule rather than a held-out test example. Every motif class prediction and every trained-task class prediction agrees on CPU. This exception does not alter the reported motif results.

A separate [numerical reconciliation](../maintenance/READOUT_V7_NUMERICAL_CHECK.json) preserves that CPU failure and reconstructs all saved heads on both devices. The matched CUDA replay reproduces all 432 head arrays bit for bit, with zero class mismatches among 1,984,608 decisions and maximum logit discrepancy zero. No input hashes, predictions, weights, class-check criterion, or numerical tolerance are changed. These checks establish the saved computations under their recorded device execution; they are not a claim that every untrained near-tied auxiliary output is identical across devices. Claim C114 retains the exact mismatch and replay counts.

The [initial runtime-wrapper audit](../maintenance/READOUT_V7_RUNTIME_AUDIT.json) also remains **FAIL**. Its CUDA head comparisons agree, but NumPy initialized before the BLAS environment defaults were applied, causing exact ridge-state, logit, and continuity-anchor checks to fail. The unchanged wrapper was then launched with `OPENBLAS_NUM_THREADS=1` and `MKL_NUM_THREADS=1` set before Python startup. The [full matched-runtime audit](../maintenance/READOUT_V7_RUNTIME_SINGLE_THREAD_AUDIT.json) is **PASS**, including all 72 ridge refits, 24 continuity anchors, motif statistics, factor/random contrasts, identity/scaler/hash checks, and clipping records. The original CPU audit, the first wrapper result, and their criteria are preserved; no tolerance is relaxed to obtain the passing run.

The [numerical note](../maintenance/READOUT_V7_NUMERICAL_NOTE.md) records the exact replay command and runtime override. [Startup evidence](../maintenance/READOUT_V7_RUNTIME_ENVIRONMENT.json) captures the configured BLAS values and unchanged wrapper hash. It does not independently measure the number of active BLAS workers because the optional thread-pool introspection package is absent. Claims C115–C116 retain the passing full audit and launch-environment evidence separately from the two preserved failures.
