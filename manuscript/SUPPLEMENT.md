# Supporting methods and record guide

This file supports the [working manuscript](MANUSCRIPT.md). It describes completed records through v6; the pending readout-factor analysis has no results in this revision. Repository paths below are relative to the repository root unless linked otherwise. Numerical evidence is preserved in the experiment archives. The manuscript and this supplement are separate documentation and do not revise those archives.

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

The v4 cross-position analysis considers layers zero through eight and multiple spatial pools. Root, target mean, target plus successive graph shells, and whole-graph mean are all 64-dimensional; the original graph/query vector is a distinct 448-dimensional reference. Oracle masks are applied after encoding the complete molecule. At deep layers, target-selected states can aggregate information from outside the selected atoms. In the recorded layer-eight analysis, both root and target pools have whole-graph receptive reach for 98.7% of analysis molecules. This is an architectural reach statement, not a measured influence of every reachable atom.

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

The oracle-minus-whole paired contrast holds this new fitting protocol fixed and changes supplied pooling membership. A comparison with the inherited v5 head changes more than localization: standardization, input range, initialization, regularization, fitting objective, and the use of deep coordinates can differ. Ridge versus nonlinear fitting also changes multiple conditions. The pending factor analysis is intended to narrow those explanations; this supplement assigns no numerical effect to any one factor.

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

The existing figures can support an initial layout without replotting frozen numerical records:

1. **Controlled motifs and partitions.** [Motif grid](../experiments/hard_v2/figures/motif_grid.png), with a new manuscript schematic to be prepared only if needed. The caption must distinguish motif topology reuse from canonical molecular identity isolation.
2. **Spatial readability and labeled adaptation.** [Scale analysis](../experiments/scale_v4/figures/scale_primary.png) and [layerwise paired curves](../experiments/scale_v4/figures/scale_paired_layers.png). Captions must retain their distinct label budgets and balanced accuracy definition.
3. **Boundary-free intervention.** [Shallow-path endpoints](../experiments/skip_v5/figures/skip_primary.png). Ordinary motif accuracy, familiar guards, and parameter-matched controls remain labeled.
4. **Source-only pooling and output sensitivity.** [Pooling comparison](../experiments/pooling_v6/figures/pooling_source.png) and [branch output comparison](../experiments/pooling_v6/figures/branch_outputs.png). Oracle membership and head-specific logit units remain explicit.

These are references to existing artifacts, not completed submission figure numbering. Pending controls and a broader chemistry literature review must be integrated before a submission-ready revision. No public repository, author list, funding source, or journal acceptance status is asserted.
