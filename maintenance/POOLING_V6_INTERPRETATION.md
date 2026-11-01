# Independent interpretation of the frozen-model pooling diagnostic

The completed experiment supports two limitations in this setup: supplying
target localization makes a large difference, and the classifier fitting rule
also matters when localization is absent. It does not support the stronger
claims that whole-graph averaging removes all motif information or that the
shallow branch makes no contribution to the original classifier's computation.
This note interprets the fixed result rather than revising its protocol or
primary records.

## Original source labels with matched pooling readouts

Every source-only classifier used the original 1,386 training identities and
motif labels, with no common-OOD or adaptation-reserve label. All encoders stayed
frozen. The whole-graph features are the exact layer-1 vectors previously
appended to the v5 heads, verified bit for bit. The oracle pool uses the nine
annotated target atoms at the same layer and explicitly supplies fragment
membership. Both pools have width 64 and use the same support-only
standardization and fixed classifier protocol.

The table gives ordinary common-OOD accuracy as mean plus or minus sample SD
over three seeds. Oracle-target accuracy is 100.0% plus or minus 0.0 for every
row, for both classifier families, including all matched random encoders.

| Encoder | Experience | Whole-graph ridge (%) | Whole-graph MLP (%) | Oracle target, either readout (%) |
| --- | --- | ---: | ---: | ---: |
| V5 end-to-end | A | 37.4 +/- 19.5 | 17.6 +/- 15.2 | 100.0 +/- 0.0 |
| V5 end-to-end | B | 64.2 +/- 9.6 | 28.4 +/- 9.4 | 100.0 +/- 0.0 |
| Original pretrained | A | 27.4 +/- 8.5 | 18.9 +/- 20.0 | 100.0 +/- 0.0 |
| Original pretrained | B | 58.6 +/- 20.3 | 20.5 +/- 17.0 | 100.0 +/- 0.0 |
| Random | A | 48.9 +/- 24.1 | 52.5 +/- 11.6 | 100.0 +/- 0.0 |
| Random | B | 65.1 +/- 25.9 | 41.8 +/- 18.3 | 100.0 +/- 0.0 |

Ridge is a closed-form alpha-1 solve with no optimizer updates. The 120-epoch
budget in the primary report applies only to the MLP fits. That fixed MLP is a
fresh standardized `64 -> 64 -> 8` ReLU head. Its initial weights and training
orders are matched across pools.
The original-trained and v5-end-to-end whole-graph fits reach 100% training
accuracy for both readouts. Their ridge own-IID accuracy is 100% in every seed;
the corresponding MLP experience-level means are 99.2% to 99.9%. Their weaker
OOD performance therefore occurs
despite fitting the familiar task successfully. The result is compatible with
source-position binding favoring a prediction rule that fails outside that
binding; it does not identify that rule uniquely.

Whole-graph ridge predictions remain substantially useful in some conditions.
For the v5 end-to-end A/B encoders, its 37.4%/64.2% OOD accuracy exceeds the
original shallow-skip head's 2.2%/6.6% while retaining 100% own-IID accuracy.
This recovery from the original source labels does not require new combination
labels and does not involve an IID-accuracy sacrifice.
Calling this a complete loss of information during pooling would overstate the
evidence. Its much larger oracle-versus-whole gap shows that supplied
localization is consequential under both fixed readout procedures. The poorer
trained-encoder MLP results alongside stronger ridge results also make decoder
and fitting choices relevant. This comparison does not isolate one optimization
mechanism: ridge and MLP change objective, regularization, function class, and
optimization. Comparisons with the inherited v5 classifier additionally change
its joint three-task objective and access to the deep feature vector.

The random-encoder oracle controls establish that this diagnostic success does
not require task training. They do not invalidate the separate earlier result
that training improved graph-level adaptation when new labels were supplied.
Original-source-label generalization, source-position probing, and adaptation
are distinct supervision protocols.

## Separately supervised cross-position probes

Using the exact v4 source-position supports, oracle-target ridge accuracy is
100% for every encoder. Whole-graph balanced accuracy, averaging three support
draws and all six directions within seed, is 48.4% +/- 12.7 and 57.9% +/- 13.0
for v5 end-to-end A/B; 54.4% +/- 25.0 and 50.4% +/- 1.7 for original-trained A/B;
and 74.2% +/- 7.9 for the shared random reference. These results demonstrate
partially accessible cross-position information in the actual whole-graph
vector, with a substantial localization advantage.

Each probe uses 256 balanced labels from its source position. Those labels can
include combinations excluded from encoder pretraining, although none comes
from the target position of that directed comparison. These scores cannot be
reported as original-head zero-shot generalization or as the result of using
only the original training labels. Random A/B all-analysis rows share one
encoder and identical predictions and are counted once in this summary.

## Existing branch influence beyond accuracy

The shallow branch changes the original classifier's logits even when its
accuracy is unchanged. On common OOD, both zeroing and permutation change logits
on every tested molecule in every shallow model. The effects differ in size:

| V5 regime | Experience | Zero: predictions changed (%) | Zero: mean absolute margin change | Permute: predictions changed (%) | Permute: mean absolute margin change |
| --- | --- | ---: | ---: | ---: | ---: |
| End-to-end | A | 1.129 +/- 0.814 | 0.18792 +/- 0.02845 | 0.000 +/- 0.000 | 0.00552 +/- 0.00188 |
| End-to-end | B | 0.752 +/- 0.690 | 0.27237 +/- 0.10660 | 0.000 +/- 0.000 | 0.00687 +/- 0.00285 |
| Frozen readout | A | 0.000 +/- 0.000 | 0.14091 +/- 0.06072 | 0.000 +/- 0.000 | 0.00608 +/- 0.00205 |
| Frozen readout | B | 0.226 +/- 0.226 | 0.13961 +/- 0.00259 | 0.075 +/- 0.130 | 0.00484 +/- 0.00055 |

Margin is the correct-class logit minus the largest competing logit. Permutation
produces nonzero centered-logit changes, so these are not exclusively common
offsets to all classes. The seed-mean relative centered-logit L2 changes are
0.061% to 0.113% for permutation and 1.98% to 3.21% for zeroing. The intact
correct-class margins average approximately -7.56 to -12.88 across these panels,
so small output changes often leave an already incorrect decision unchanged.
Raw margin units remain specific to each head; these summaries are descriptive,
not a calibrated causal effect size across heads.

The single permutation-induced motif class change in frozen B changes one
incorrect class to another and leaves accuracy unchanged. Zeroing includes
wrong-to-correct, correct-to-wrong, and incorrect-to-incorrect transitions in
different models. Therefore, the earlier accuracy-only result cannot be phrased
as complete branch nonuse. Its example-dependent output effects are small under
the recorded permutation and insufficient to repair the overall generalization
failure. Zeroing and permutation can produce off-distribution feature pairs;
they do not separate semantic and optimization contributions additively.

## Scope and verification

Oracle target pooling includes the first-layer receptive support of its selected
nodes and their supplied attributed input; it is not an isolated nine-atom input
or an automatically learned localizer. This experiment isolates two pooling
scopes under matched new readouts, with explicit additional membership in the
oracle scope. It does not show that every boundary-free pooling rule must fail,
that deeper motif information has disappeared, or that one particular training
mechanism explains the separation.

The [independent audit](POOLING_V6_AUDIT.json) passes all checks. It verifies 724
new manifest-covered files and all 536 files in the seven preceding records,
reconstructs all 216 branch logit arrays and 198,288 per-example comparisons,
independently resolves all 306 ridge fits, and reconstructs every prediction of
the 36 saved MLP heads. All class comparisons are exact; there are zero v5
prediction, branch-logit, or saved-MLP prediction mismatches. It also verifies
oracle masks, support isolation, original epoch orders, classifier
initializations, actual-v5-feature continuity, frozen encoder hashes, per-seed
statistics, and paired contrasts. Metadata and final states do not independently
observe every historical optimizer operation or exclude external selection.

The primary [report](../experiments/pooling_v6/REPORT.md),
[protocol](../configs/POOLING_V6_PROTOCOL.md), and numerical records remain
unchanged. The execution source is
`8a5e6997e7bff4ebc1e0d6438a9847fd8623dee2`; this interpretation is a separate
post-run document.
